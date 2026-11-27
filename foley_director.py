#!/usr/bin/env python3
"""
Write the sound script for foley_batch.py with a local Ollama vision model.

foley_batch.py renders the video window by window (HunyuanVideo-Foley cannot do
more than 30 s at once). One sentence for a whole six-minute lesson is why the
result sounds like a single faint room tone: the model never learns about the
beats. This module looks at two still frames per window, asks a local Ollama
model what is on screen, and writes one prompt per window, so every window gets
its own ambience and its own accent on the big moment.

    python foley_director.py --video "C:\\clips\\lesson.mp4"
    python foley_director.py --video lesson.mp4 --window 10 --hint "soft chalk on a board"

foley_batch.py calls script_for() by itself when the prompt question is answered
with Enter. The finished plan is cached as
<video>_foley_work/plan.json and reused on the next run.
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from foley_batch import (DEFAULT_MODEL, DEFAULT_OLLAMA, DEFAULT_STYLE, need,
                         plan_meta, plan_windows, probe, read_config, save_plan)

FRAMES_PER_WINDOW = 2          # frames sent to the model for each window
FRAMES_AT = (0.25, 0.72)       # where in the window those frames are taken
WINDOWS_PER_CALL = 4           # windows per Ollama request (8 frames)
FRAME_WIDTH = 512
MAX_CALL_TRIES = 2
CALL_TIMEOUT = 1800


class Unavailable(RuntimeError):
    """The prompt writer cannot run (Ollama down, model missing, ...)."""


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #
def _post(url: str, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def check_model(ollama: str, model: str) -> None:
    """Fail with a plain sentence when the model cannot do this job."""
    try:
        info = _post(ollama.rstrip("/") + "/api/show", {"model": model}, 60)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise Unavailable(
                f"Ollama does not have '{model}' - run: ollama pull {model}"
            ) from None
        raise Unavailable(f"Ollama refused /api/show ({exc.code})") from None
    except urllib.error.URLError as exc:
        raise Unavailable(f"cannot reach Ollama at {ollama} ({exc.reason})") from None
    caps = info.get("capabilities") or []
    if "vision" not in caps:
        raise Unavailable(
            f"'{model}' cannot see images (capabilities: {', '.join(caps) or 'none'})"
        )


def unload(ollama: str, model: str) -> None:
    """Free the VRAM again - ComfyUI needs it for the render right afterwards."""
    try:
        _post(ollama.rstrip("/") + "/api/generate", {"model": model, "keep_alive": 0}, 30)
    except Exception:
        pass


def _chat(ollama: str, model: str, content: str, images: list[str]) -> str:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [{"role": "user", "content": content, "images": images}],
        "options": {"temperature": 0.6, "num_predict": 1200, "num_ctx": 16384},
    }
    url = ollama.rstrip("/") + "/api/chat"
    try:
        data = _post(url, payload, CALL_TIMEOUT)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        if exc.code == 400 and "think" in detail.lower():
            payload.pop("think", None)
            data = _post(url, payload, CALL_TIMEOUT)
        else:
            raise RuntimeError(f"Ollama refused the request ({exc.code}): {detail}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"lost Ollama ({exc.reason})") from None
    return (data.get("message") or {}).get("content") or ""


# --------------------------------------------------------------------------- #
# frames
# --------------------------------------------------------------------------- #
def _grab(video: Path, at: float, dest: Path, ffmpeg: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
         "-ss", f"{at:.3f}", "-i", str(video), "-frames:v", "1",
         "-vf", f"scale={FRAME_WIDTH}:-2", "-q:v", "5", str(dest)],
        check=True,
    )


# --------------------------------------------------------------------------- #
# what the model is asked
# --------------------------------------------------------------------------- #
def _brief(name: str, total: float, window: float, first: int, count: int,
           style: str, hint: str) -> str:
    lines = [
        "You are writing sound prompts for a video-to-foley model (HunyuanVideo-Foley).",
        "That model hears ONLY your sentence. It renders one short clip at a time and",
        "it cannot hear this video; the still frames below are all you know about it.",
        "",
        f"Video: {name}, {total:.0f} s long, cut into clips of {window:g} s.",
        f"The frames are clips {first} to {first + count - 1}, in order,",
        f"{FRAMES_PER_WINDOW} frames per clip (early frame, then a later one).",
        "",
        f"Sound direction: {style}",
    ]
    if hint.strip():
        lines += ["", f"The editor asked for: {hint.strip()}"]
    lines += [
        "",
        "For EACH clip write ONE English sentence, 10-25 words, that:",
        "1. names the real place and its constant quiet ambience (room tone, air, hum,",
        "   distant outside), so the clip is never silent;",
        "2. says what the physical action in the picture sounds like (cloth, skin,",
        "   taps, steps, paper, wood, water);",
        "3. if that clip has a strong visual beat - a cut, a slam, a fast gesture, a",
        "   big movement - names ONE clear, loud accent right on that beat (thump,",
        "   whoosh, smack); if nothing big happens, keeps the sound soft and steady.",
        "",
        "Hard rules: physical sound in the room only. Never mention music, soundtrack,",
        "instruments, singing, speech, narration, a voice, or a narrator. No camera or",
        "editing words. Present tense, one sentence, English, no numbering.",
        "",
        f"Reply with ONLY a JSON array of exactly {count} strings, in clip order.",
        'Example: ["sentence for clip A", "sentence for clip B"]',
    ]
    return "\n".join(lines)


def _clean(text: str) -> str:
    line = " ".join(str(text).split()).strip().strip('"').strip("'").strip()
    line = re.sub(r"^(?:clip\s*\d+\s*[:.-]\s*|-\s*)", "", line, flags=re.I)
    if len(line) > 400:
        cut = line[:400].rsplit(" ", 1)[0]
        line = cut if len(cut) > 120 else line[:400]
    return line


def _parse_list(text: str, count: int) -> list[str]:
    """Pull the JSON array out of whatever the model wrapped it in."""
    text = text.strip()
    for candidate in (text, re.search(r"\[[\s\S]*\]", text).group(0)
                      if re.search(r"\[[\s\S]*\]", text) else ""):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        out: list[str] = []
        for item in data:
            if isinstance(item, str):
                out.append(_clean(item))
            elif isinstance(item, dict):
                for key in ("prompt", "sound", "text", "description", "sentence"):
                    if isinstance(item.get(key), str):
                        out.append(_clean(item[key]))
                        break
        if out:
            return out
    # last resort: numbered / bulleted lines
    lines: list[str] = []
    for raw in text.splitlines():
        line = _clean(re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", raw))
        if len(line) > 15:
            lines.append(line)
    return lines[:count]


def _fallback_prompt(hint: str) -> str:
    place = hint.strip() or "a quiet interior room"
    return (
        f"Steady ambience of {place} with continuous room tone and air, soft cloth "
        f"and skin movement, and one clear accent where the action peaks"
    )


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #
def script_for(video: Path, plan: list[tuple[float, float]], *, total: float,
               work: Path, ollama: str = DEFAULT_OLLAMA, model: str = DEFAULT_MODEL,
               style: str = DEFAULT_STYLE, hint: str = "", log=print) -> list[str]:
    """One foley prompt per window, written from the picture."""
    try:
        check_model(ollama, model)
    except Unavailable as exc:
        raise Unavailable(str(exc)) from None

    ffmpeg = need("ffmpeg")
    frames = work / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    window = plan[0][1] if plan else 0.0
    fallback = _fallback_prompt(hint)
    log(f"  reading the video with {model} ({len(plan)} clips, "
        f"{FRAMES_PER_WINDOW} frames each, {WINDOWS_PER_CALL} clips per request)")
    started = time.time()
    prompts: list[str] = []

    for first in range(0, len(plan), WINDOWS_PER_CALL):
        chunk = plan[first:first + WINDOWS_PER_CALL]
        images: list[str] = []
        for i, (start, length) in enumerate(chunk, first + 1):
            for n, frac in enumerate(FRAMES_AT):
                frame = frames / f"w{i:03d}_{n}.jpg"
                if not frame.is_file():
                    _grab(video, start + length * frac, frame, ffmpeg)
                images.append(base64.b64encode(frame.read_bytes()).decode())
        brief = _brief(video.name, total, window, first + 1, len(chunk), style, hint)
        got = _ask(ollama, model, brief, images, len(chunk), fallback, log)
        for i, (start, length) in enumerate(chunk, first + 1):
            log(f"  [{i:3d}/{len(plan)}] {start:6.1f}-{start + length:6.1f}s  {got[i - first - 1]}")
        prompts += got

    log(f"  {len(prompts)} prompts written in {time.time() - started:.0f}s")
    unload(ollama, model)
    return prompts


def _ask(ollama: str, model: str, brief: str, images: list[str], count: int,
         fallback: str, log=print) -> list[str]:
    prompts: list[str] = []
    for attempt in range(1, MAX_CALL_TRIES + 1):
        text = brief if attempt == 1 else (
            brief + f"\n\nAgain, and this time exactly {count} strings in one JSON array."
        )
        try:
            raw = _chat(ollama, model, text, images)
        except RuntimeError as exc:
            log(f"      {exc}")
            break
        prompts = [p for p in _parse_list(raw, count) if p]
        if len(prompts) >= count:
            return prompts[:count]
        log(f"      got {len(prompts)} of {count} prompts - "
            f"{'trying again' if attempt < MAX_CALL_TRIES else 'filling the rest'}")
    if len(prompts) < count:
        prompts += [fallback] * (count - len(prompts))
    return prompts[:count]


# --------------------------------------------------------------------------- #
def main() -> int:
    import argparse

    cfg = read_config()
    ap = argparse.ArgumentParser(description="write one foley prompt per window")
    ap.add_argument("--video", required=True)
    ap.add_argument("--window", type=float, help="seconds per window (default from foley_config.txt)")
    ap.add_argument("--hint", default="", help="extra words for the prompt writer")
    ap.add_argument("--style", default=cfg.get("STYLE") or DEFAULT_STYLE)
    ap.add_argument("--model", default=cfg.get("MODEL") or DEFAULT_MODEL)
    ap.add_argument("--ollama", default=cfg.get("OLLAMA") or DEFAULT_OLLAMA)
    ap.add_argument("--save", action="store_true", help="write <video>_foley_work/plan.json")
    args = ap.parse_args()

    video = Path(args.video).expanduser().resolve()
    if not video.is_file():
        sys.exit(f"no such video: {video}")
    total, _fps = probe(video)
    window = min(max(1.0, args.window or float(cfg.get("WINDOW") or 15)), 30.0)
    plan = plan_windows(total, window)
    work = video.with_name(f"{video.stem}_foley_work")

    try:
        prompts = script_for(video, plan, total=total, work=work,
                             ollama=args.ollama, model=args.model,
                             style=args.style, hint=args.hint, log=print)
    except Unavailable as exc:
        sys.exit(f"{exc}\n(answer the prompt question yourself and nothing is needed from Ollama)")

    if args.save:
        save_plan(work, plan_meta(video, window, args.model, args.style, args.hint,
                                  len(plan)), plan, prompts)
        print(f"\nsaved: {work / 'plan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
