#!/usr/bin/env python3
"""
Add foley effects to a WHOLE video with the Hunyuanfoley.json workflow.

    python foley_batch.py --video "C:\\clips\\film.mp4"

The sampler only renders 30 s at a time, so the video is cut into windows
(foley_config.txt -> WINDOW), each window is rendered with the same graph, the
window audios are joined, and that one track is muxed onto the untouched video
stream. Nothing is re-encoded except the audio.

Graph, exactly as in Hunyuanfoley.json:
    VHS_LoadVideo ─┬──────────────────────────────► VHS_VideoCombine (images)
                   ├──► HunyuanFoleySampler ──────► VHS_VideoCombine (audio)
                   └──► VHS_VideoInfo (fps, duration) ─► sampler
    HunyuanModelLoader ─► HunyuanFoleySampler   (HunyuanFoleyTorchCompile skipped:
                        the portable launcher sets TORCHINDUCTOR_DISABLE=1)
    HunyuanDependenciesLoader ─► HunyuanFoleySampler

The sound: answer the prompt question with a sentence and it is used for every
window; press Enter and foley_director.py lets a local Ollama vision model look
at two frames per window and write one prompt per window - a constant ambience
plus an accent on the big beats. foley_prompt.txt is only the fallback.

By default the original audio track is kept and the effects are mixed under it
(MIX=1, FOLEY_DB=-6 in foley_config.txt). Standard library plus ffmpeg/ffprobe
on PATH.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "foley_config.txt"
PROMPT_FILE = HERE / "foley_prompt.txt"

DEFAULT_SERVER = "http://127.0.0.1:8188"
MAX_WINDOW = 30.0          # HunyuanFoleySampler duration max
DEFAULT_NEGATIVE = ("music, soundtrack, piano, instruments, singing, voices, "
                    "speech, humming")
DEFAULT_OLLAMA = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3.8:27b-Q3"
DEFAULT_STYLE = ("loud, close, physical foley for an educational lesson: a constant "
                 "real room ambience under every second, and a clear accent on each "
                 "big visual beat, so the viewer stays pulled into the explanation")
VIDEO_EXT = (".mp4", ".webm", ".mkv", ".mov")


# --------------------------------------------------------------------------- #
# config / tools
# --------------------------------------------------------------------------- #
def read_config() -> dict[str, str]:
    """foley_config.txt: KEY=VALUE lines, # comments, blank lines ignored."""
    cfg: dict[str, str] = {}
    if not CONFIG.is_file():
        return cfg
    for raw in CONFIG.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cfg[key.strip().upper()] = value.strip().strip('"').strip("'")
    return cfg


def comfy_root(explicit: str | None, cfg: dict[str, str]) -> Path:
    """The ComfyUI folder whose `input` the running server reads."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if cfg.get("COMFY"):
        candidates.append(Path(cfg["COMFY"]).expanduser())
    candidates += [HERE, *HERE.parents]
    for cand in candidates:
        for path in (cand, cand / "ComfyUI"):
            if (path / "main.py").is_file() and (path / "input").is_dir():
                return path.resolve()
    sys.exit(f"could not find ComfyUI (checked {HERE} and foley_config.txt COMFY=)")


def need(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        sys.exit(f"{tool} is not on PATH - install ffmpeg or add it to PATH")
    return path


def probe(path: Path) -> tuple[float, float]:
    """Return (duration_seconds, fps) of the video stream."""
    raw = subprocess.check_output(
        [need("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "format=duration:stream=avg_frame_rate,r_frame_rate",
         "-of", "json", str(path)],
        text=True,
    )
    data = json.loads(raw)
    duration = float(data.get("format", {}).get("duration") or 0.0)
    stream = (data.get("streams") or [{}])[0]
    for key in ("avg_frame_rate", "r_frame_rate"):
        rate = stream.get(key) or "0/0"
        try:
            num, den = rate.split("/")
            fps = float(num) / float(den) if float(den) else 0.0
        except ValueError:
            fps = 0.0
        if fps > 0:
            return duration, round(fps, 3)
    sys.exit(f"could not read the frame rate of {path.name}")


def cut(video: Path, start: float, dur: float, dest: Path) -> None:
    """Write [start, start+dur] of `video` to `dest`, frame exact, no audio."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [need("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video), "-t", f"{dur:.3f}", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
            "-pix_fmt", "yuv420p", str(dest)]
    subprocess.run(cmd, check=True)


def post_json(path: str, payload: dict, server: str) -> dict:
    req = urllib.request.Request(
        server + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read().decode())


def get_json(path: str, server: str):
    with urllib.request.urlopen(server + path, timeout=180) as resp:
        return json.loads(resp.read().decode())


# --------------------------------------------------------------------------- #
# graph (the Hunyuanfoley.json workflow, API form)
# --------------------------------------------------------------------------- #
def build_graph(clip: str, prompt: str, negative: str, fps: float, dur: float,
                steps: int, cfg: float, seed: int) -> dict:
    return {
        "10": {  # HunyuanModelLoader
            "class_type": "HunyuanModelLoader",
            "inputs": {
                "model_name": "hunyuanvideo_foley.pth",
                "precision": "bf16",
                "quantization": "fp8_e4m3fn",
            },
        },
        "11": {  # HunyuanDependenciesLoader
            "class_type": "HunyuanDependenciesLoader",
            "inputs": {
                "vae_name": "vae_128d_48k.pth",
                "synchformer_name": "synchformer_state_dict.pth",
            },
        },
        "2": {  # VHS_LoadVideo
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": clip,
                "force_rate": 0,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
                "format": "None",
            },
        },
        "3": {  # VHS_VideoInfo - only used to keep the graph readable
            "class_type": "VHS_VideoInfo",
            "inputs": {"video_info": ["2", 3]},
        },
        "14": {  # HunyuanFoleySampler
            "class_type": "HunyuanFoleySampler",
            "inputs": {
                "hunyuan_model": ["10", 0],
                "hunyuan_deps": ["11", 0],
                "image": ["2", 0],
                "fps": float(fps),
                "duration": float(dur),
                "prompt": prompt,
                "negative_prompt": negative,
                "cfg_scale": float(cfg),
                "steps": int(steps),
                "sampler": "euler",
                "batch_size": 1,
                "seed": int(seed),
                "force_offload": True,
            },
        },
        "5": {"class_type": "PreviewAudio", "inputs": {"audio": ["14", 0]}},
        "6": {  # VHS_VideoCombine - saves the window audio (video stream unused)
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["2", 0],
                "audio": ["14", 0],
                "frame_rate": float(fps),
                "loop_count": 0,
                "filename_prefix": "BatchSFX",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": False,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        },
    }


def saved_files(entry: dict, root: Path) -> list[Path]:
    found: list[Path] = []
    for node_out in (entry.get("outputs") or {}).values():
        for value in node_out.values():
            if not isinstance(value, list):
                continue
            for item in value:
                if not isinstance(item, dict) or "filename" not in item:
                    continue
                name = item["filename"]
                if not name.lower().endswith(VIDEO_EXT):
                    continue
                sub = item.get("subfolder") or ""
                base = root / ("output" if item.get("type") == "output" else "temp")
                found.append(base / sub / name)
    return found


def wait_for(server: str, prompt_id: str, label: str) -> dict:
    started = time.time()
    while True:
        time.sleep(2)
        try:
            history = get_json(f"/history/{prompt_id}", server)
        except urllib.error.URLError as exc:
            sys.exit(f"lost ComfyUI while waiting ({exc.reason}) - segments so far are kept")
        if prompt_id in history:
            entry = history[prompt_id]
            break
        elapsed = time.time() - started
        try:
            status = get_json("/queue", server)
            running = any(i[1] == prompt_id for i in status.get("queue_running", []))
            pending = any(i[1] == prompt_id for i in status.get("queue_pending", []))
            state = "running" if running else ("queued" if pending else "finishing")
        except Exception:
            state = "working"
        print(f"      {label} {elapsed:6.0f}s  {state}        ", end="\r", flush=True)
        if elapsed > 7200:
            sys.exit("gave up on this window after two hours")
    print(" " * 60, end="\r")
    status = entry.get("status") or {}
    if status.get("status_str") == "error":
        for kind, payload in status.get("messages", []):
            if kind == "execution_error":
                sys.exit(f"ComfyUI error: {payload.get('exception_message', payload)}")
    return entry


# --------------------------------------------------------------------------- #
# windows
# --------------------------------------------------------------------------- #
def plan_windows(total: float, size: float) -> list[tuple[float, float]]:
    """[(start, length)] covering the whole video, last one at least 1 s."""
    plan: list[tuple[float, float]] = []
    t = 0.0
    while t < total - 0.05:
        length = min(size, total - t)
        if length < 1.0 and plan:
            plan[-1] = (plan[-1][0], plan[-1][1] + length)
        else:
            plan.append((t, length))
        t += length
    return plan


def extract_audio(src: Path, dest: Path) -> None:
    subprocess.run(
        [need("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(src), "-vn", "-ac", "2", "-ar", "48000",
         "-c:a", "pcm_s16le", str(dest)],
        check=True,
    )


def join_audio(parts: list[Path], dest: Path, work: Path) -> None:
    listing = work / "segments.txt"
    listing.write_text(
        "".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8"
    )
    subprocess.run(
        [need("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
         "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c", "copy", str(dest)],
        check=True,
    )


def has_audio(video: Path) -> bool:
    """True when the source already carries an audio track (narration, music...)."""
    raw = subprocess.check_output(
        [need("ffprobe"), "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=index", "-of", "json", str(video)],
        text=True,
    )
    return bool(json.loads(raw).get("streams"))


def normalise(track: Path, lufs: float, dest: Path) -> None:
    """Bring the effects track to a fixed loudness so it is actually audible."""
    subprocess.run(
        [need("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(track), "-af", f"loudnorm=I={lufs:g}:TP=-1.5:LRA=11",
         "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(dest)],
        check=True,
    )


def mux(video: Path, audio: Path, dest: Path, *, keep_original: bool,
        foley_db: float) -> None:
    """Keep the video stream untouched; mix the effects under the original audio."""
    cmd = [need("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(video), "-i", str(audio)]
    if keep_original:
        cmd += ["-filter_complex",
                f"[1:a]volume={foley_db:g}dB,apad[fx];"
                f"[0:a][fx]amix=inputs=2:duration=first:normalize=0[a]",
                "-map", "0:v:0", "-map", "[a]"]
    else:
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-af", "apad"]
    cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart", str(dest)]
    subprocess.run(cmd, check=True)


def plan_meta(video: Path, window: float, model: str, style: str, hint: str,
              count: int) -> dict:
    """Everything a cached plan.json has to match to still be valid."""
    stat = video.stat()
    return {"video": video.name, "size": stat.st_size, "mtime": int(stat.st_mtime),
            "window": round(window, 3), "model": model, "style": style,
            "hint": hint, "count": count}


def load_plan(work: Path, meta: dict) -> list[str] | None:
    path = work / "plan.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("meta") != meta:
        return None
    prompts = [str(p) for p in (data.get("prompts") or [])]
    return prompts or None


def save_plan(work: Path, meta: dict, plan: list[tuple[float, float]],
              prompts: list[str]) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / "plan.json").write_text(json.dumps(
        {"meta": meta, "prompts": prompts,
         "windows": [{"start": round(s, 3), "length": round(l, 3)} for s, l in plan]},
        indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
def main() -> int:
    cfg = read_config()
    window = float(cfg.get("WINDOW") or 15)
    steps = int(cfg.get("STEPS") or 50)
    cfg_scale = float(cfg.get("CFG") or 4.5)
    negative = cfg.get("NEGATIVE") or DEFAULT_NEGATIVE
    server = (cfg.get("SERVER") or DEFAULT_SERVER).rstrip("/")
    ollama = cfg.get("OLLAMA") or DEFAULT_OLLAMA
    model = cfg.get("MODEL") or DEFAULT_MODEL
    style = cfg.get("STYLE") or DEFAULT_STYLE
    loudness = float(cfg.get("LOUDNESS") or 0)
    mix_original = (cfg.get("MIX") or "1").strip().lower() not in ("0", "no", "off", "false")
    foley_db = float(cfg.get("FOLEY_DB") or -6)

    ap = argparse.ArgumentParser(description="whole video -> HunyuanVideo-Foley")
    ap.add_argument("--video", required=True)
    ap.add_argument("--prompt", default="",
                    help="one sentence for the whole video; empty = Ollama writes "
                         "one prompt per window from the video")
    ap.add_argument("--hint", default="", help="extra words for the prompt writer")
    ap.add_argument("--replan", action="store_true",
                    help="ignore a cached plan.json and look at the video again")
    ap.add_argument("--model", help="Ollama vision model that writes the prompts")
    ap.add_argument("--ollama", help="Ollama base url")
    ap.add_argument("--style", help="sound direction handed to the prompt writer")
    ap.add_argument("--mix", dest="mix", action="store_true", default=None,
                    help="mix the effects under the original audio (default: MIX=)")
    ap.add_argument("--no-mix", dest="mix", action="store_false",
                    help="drop the original audio, effects only")
    ap.add_argument("--foley-db", type=float, help="effect level under the original audio")
    ap.add_argument("--loudness", type=float, help="normalise the effects to this LUFS, 0 = off")
    ap.add_argument("--window", type=float, help=f"seconds per window, max {MAX_WINDOW:g}")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--cfg", type=float)
    ap.add_argument("--negative")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", help="output file (default: <video>_foley.mp4)")
    ap.add_argument("--restart", action="store_true",
                    help="ignore finished windows and render everything again")
    ap.add_argument("--dry-run", action="store_true", help="only print the plan")
    ap.add_argument("--server")
    ap.add_argument("--comfy")
    args = ap.parse_args()

    window = min(max(1.0, args.window or window), MAX_WINDOW)
    steps = args.steps or steps
    cfg_scale = args.cfg or cfg_scale
    negative = args.negative or negative
    server = (args.server or server).rstrip("/")
    model = args.model or model
    ollama = args.ollama or ollama
    style = args.style or style
    mix_original = mix_original if args.mix is None else args.mix
    foley_db = foley_db if args.foley_db is None else args.foley_db
    loudness = loudness if args.loudness is None else args.loudness
    root = comfy_root(args.comfy, cfg)

    video = Path(args.video).expanduser().resolve()
    if not video.is_file():
        sys.exit(f"no such video: {video}")

    total, fps = probe(video)
    plan = plan_windows(total, window)
    work = video.with_name(f"{video.stem}_foley_work")
    dest = (Path(args.out).expanduser() if args.out
            else video.with_name(f"{video.stem}_foley.mp4"))

    # ---- where the prompts come from ------------------------------------- #
    prompts: list[str] = []
    fixed = (args.prompt or "").strip()
    if fixed:
        prompts = [fixed] * len(plan)
        source = "typed by you, same sentence for every window"
    else:
        meta = plan_meta(video, window, model, style, args.hint, len(plan))
        cached = None if args.replan else load_plan(work, meta)
        if cached and len(cached) == len(plan):
            prompts = cached
            source = f"plan.json, written by {model}"
            print(f"  reusing the prompts in {work.name}\\plan.json")
        else:
            try:
                from foley_director import Unavailable, script_for
                prompts = script_for(video, plan, total=total, work=work,
                                     ollama=ollama, model=model, style=style,
                                     hint=args.hint, log=print)
                if len(prompts) == len(plan):
                    save_plan(work, meta, plan, prompts)
                source = f"{model}, one prompt per window"
            except ImportError:
                print("  foley_director.py is missing - cannot write the prompts")
            except Unavailable as exc:
                print(f"  no prompts from the video: {exc}")
            except RuntimeError as exc:
                print(f"  the prompt writer stopped: {exc}")
            except KeyboardInterrupt:
                sys.exit("stopped while writing the prompts")

    if not prompts or len(prompts) != len(plan):
        file_prompt = ""
        if PROMPT_FILE.is_file():
            file_prompt = " ".join(
                line for line in PROMPT_FILE.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ).strip()
        if not file_prompt:
            sys.exit("no sound prompt: the video could not be read and "
                     f"{PROMPT_FILE.name} is empty - type a sentence at the question")
        prompts = [file_prompt] * len(plan)
        source = f"{PROMPT_FILE.name} (fallback, same sentence for every window)"

    print("------------------------------------------------------------")
    print(f"  video       : {video.name}   {total:.1f}s, {fps:g} fps")
    print(f"  sound       : {source}")
    print(f"  keeping out : {negative}")
    print(f"  windows     : {len(plan)} x {window:g}s   steps {steps}  cfg {cfg_scale}")
    print(f"  comfyui     : {server}   ({root})")
    print("  mix         : " + (f"original audio + effects at {foley_db:g} dB"
                             if mix_original and has_audio(video)
                             else "effects only (original track dropped)"))
    print(f"  output      : {dest}")
    print("------------------------------------------------------------")
    print()

    if args.dry_run:
        for i, (start, length) in enumerate(plan, 1):
            print(f"  {i:3d}/{len(plan)}  {start:7.2f}s -> {start + length:7.2f}s  ({length:5.2f}s)")
            print(textwrap.fill(prompts[i - 1], width=94, initial_indent=" " * 9,
                                subsequent_indent=" " * 9))
        print("\ndry run - nothing was queued.")
        return 0

    try:
        if not get_json("/object_info/HunyuanFoleySampler", server).get("HunyuanFoleySampler"):
            sys.exit("the HunyuanVideo-Foley nodes are not loaded - restart ComfyUI")
    except urllib.error.URLError as exc:
        sys.exit(f"cannot reach ComfyUI at {server} ({exc.reason}) - start it first")

    work.mkdir(parents=True, exist_ok=True)
    clip = root / "input" / "batchsfx_clip.mp4"
    parts: list[Path] = []
    done_before = 0
    took = 0.0
    started_all = time.time()

    for i, (start, length) in enumerate(plan, 1):
        part = work / f"seg_{i:03d}.wav"
        parts.append(part)
        label = f"[{i:3d}/{len(plan)}] {start:6.1f}s"
        if part.is_file() and part.stat().st_size > 1000 and not args.restart:
            done_before += 1
            print(f"{label}  already rendered ({part.stat().st_size / 1e6:.1f} MB)")
            continue

        print(f"{label}  rendering {length:.1f}s ...", flush=True)
        t0 = time.time()
        cut(video, start, length, clip)
        clip_dur, clip_fps = probe(clip)
        seed = args.seed if args.seed is not None else random.randrange(2 ** 31)
        graph = build_graph("batchsfx_clip.mp4", prompts[i - 1], negative,
                            clip_fps, min(clip_dur, MAX_WINDOW), steps, cfg_scale, seed)
        try:
            queued = post_json("/prompt", {"prompt": graph, "client_id": "batch_sfx"}, server)
        except urllib.error.HTTPError as exc:
            sys.exit(f"ComfyUI refused the graph ({exc.code}):\n{exc.read().decode(errors='replace')}")
        entry = wait_for(server, queued["prompt_id"], label)
        files = [p for p in saved_files(entry, root) if p.is_file()]
        if not files:
            sys.exit("ComfyUI finished but saved no file - check the ComfyUI console")
        raw = work / f"raw_{i:03d}.mp4"
        shutil.copy2(files[0], raw)
        extract_audio(raw, part)
        raw.unlink(missing_ok=True)
        took += time.time() - t0
        print(f"{label}  done in {time.time() - t0:.0f}s  ({part.stat().st_size / 1e6:.1f} MB)")

    if done_before:
        print(f"\n{done_before} window(s) were already rendered and were reused.")

    print("\njoining the windows ...", flush=True)
    track = work / "foley.wav"
    join_audio(parts, track, work)
    track_dur, _ = probe_audio(track)
    print(f"  foley track : {track_dur:.2f}s   (video {total:.2f}s)")
    if loudness:
        level = work / "foley_level.wav"
        normalise(track, loudness, level)
        track = level
        print(f"  level       : normalised to {loudness:g} LUFS")

    keep = bool(mix_original and has_audio(video))
    print("muxing onto the video ...", flush=True)
    dest = free_name(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    mux(video, track, dest, keep_original=keep, foley_db=foley_db)

    out_dur, _ = probe(dest)
    print()
    print("------------------------------------------------------------")
    print(f"  result : {dest}")
    print(f"  length : {out_dur:.2f}s")
    print("  audio  : " + (f"original track + effects ({foley_db:g} dB)"
                          if keep else "effects only (original track dropped)"))
    print(f"  sound  : {source}")
    print(f"  time   : {time.time() - started_all:.0f}s total, {took:.0f}s rendering")
    print(f"  pieces : {work}")
    print("------------------------------------------------------------")
    return 0


def probe_audio(path: Path) -> tuple[float, float]:
    raw = subprocess.check_output(
        [need("ffprobe"), "-v", "error", "-select_streams", "a:0",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        text=True,
    )
    return float(json.loads(raw).get("format", {}).get("duration") or 0.0), 0.0


def free_name(path: Path) -> Path:
    if not path.exists():
        return path
    n = 2
    while path.with_name(f"{path.stem}_{n}{path.suffix}").exists():
        n += 1
    return path.with_name(f"{path.stem}_{n}{path.suffix}")


if __name__ == "__main__":
    raise SystemExit(main())
