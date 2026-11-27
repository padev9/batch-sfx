# Batch SFX 🎬

Any video in, **HunyuanVideo-Foley** effects over the whole thing out. Two
questions: the video, and the sound.

The model, graph and sampler values come from
`../../Hunyuanfoley.json` (Tencent HunyuanVideo-Foley, custom node
`Comfyui-HunyuanFoley`):

```
VHS_LoadVideo ─┬─────────────────────────────► VHS_VideoCombine (images)
               ├──► HunyuanFoleySampler ─────► VHS_VideoCombine (audio)
               └──► VHS_VideoInfo (fps, duration) ─► sampler
HunyuanModelLoader (hunyuanvideo_foley.pth, bf16, fp8_e4m3fn) ─► sampler
HunyuanDependenciesLoader (vae_128d_48k.pth, synchformer_state_dict.pth) ─► sampler
```

## Run it

1. Start ComfyUI (`run_nvidia_gpu.bat`) and Ollama (for the automatic prompts).
2. Double-click **`1 add effects.bat`** (or drop a video file on it).
3. Paste/drag the video path, press Enter.
4. **Sound:** type one sentence and it is used for every window; or press
   **Enter** and Ollama watches two frames per window and writes one prompt per
   window by itself.

Result: `<name>_foley.mp4` next to the original. The original file is never
modified, the video stream is stream-copied, and the effects are **mixed under
the existing audio** (`MIX=1`, `FOLEY_DB=-6`) - narration stays, the room around
it becomes real. `MIX=0` drops the original track and gives effects only (the old
behaviour).

When the run ends (success or error) the bat plays **`done.mp3`** (or `done.wav`)
from its own folder; with neither file present it falls back to a two-tone beep.
If ffplay is missing it decodes the mp3 to a wav with ffmpeg first, because
PowerShell's `SoundPlayer` cannot play mp3. Both are addressed by bare file name
on purpose - a full path through this emoji folder can reach the player with `?`
in place of the emoji, which silently finds nothing.

## Prompts: one per window, written from the picture

A single sentence for a whole lesson is why the first version sounded like one
faint room tone: the model never learned where the beats were. Now each window
gets its own sentence - `foley_director.py` takes two stills per window (25 % and
72 % in), sends up to four windows at a time to the Ollama vision model from
`foley_config.txt` (`MODEL=qwen3.8:27b-Q3`), and asks for one sentence per window
with three things in it:

1. the real place and its **constant ambience** (room tone, air, hum),
2. the **physical action** the picture makes (cloth, skin, taps, paper),
3. **one loud accent** on a big visual beat if the window has one, otherwise the
   sound stays soft and steady.

`STYLE=` in `foley_config.txt` is the direction handed to that model, so tuning
the whole feel of the effects is one line ("bigger impacts", "quieter room",
"more presence"). `--hint "..."` adds one run's note.

The plan is cached as `<name>_foley_work/plan.json` and reused on the next run.
It is rewritten by itself when the video, `WINDOW`, `MODEL`, `STYLE` or `--hint`
changes; `--replan` forces a rewrite. `--dry-run` prints every window's sentence
without rendering anything, which is the fast way to check the sound direction.

Fallback order for the prompt: typed sentence → cached `plan.json` → Ollama →
`foley_prompt.txt` (one sentence for every window, used when Ollama is down).
The writer frees the GPU again (`keep_alive: 0`) before the first render.

## Why windows

`HunyuanFoleySampler.duration` maxes out at **30 s**, so the runner cuts the video
into `WINDOW`-second slices (default 15), renders each, joins the slices' audio
into one track, and muxes it onto the untouched video. Video length is preserved
(the track is padded, then cut to the video length). A smaller `WINDOW` reacts to
the picture more closely but means more prompts and more renders.

The effects track is normalised to `LOUDNESS` (`-16` LUFS) before the mix so it
cannot come out faint, and the `NEGATIVE` sentence (`music, soundtrack, piano,
instruments, singing, voices, speech, humming`, applied at `cfg 4.5`) is what
keeps music and voices out.

## Resuming

Each finished window is kept as `<name>_foley_work/seg_NNN.wav`. If a run stops
(VRAM, crash, Ctrl-C), start the same video again: finished windows are reused and
only the rest is rendered. `--restart` throws the pieces away and redoes all of
them. Delete `<name>_foley_work/` to clean up after a good run (that also drops
the cached prompts).

## Command line

```bat
python foley_batch.py --video "C:\clips\film.mp4" --dry-run      REM plan + prompts, nothing queued
python foley_batch.py --video film.mp4 --prompt "rain on a tin roof"
python foley_batch.py --video film.mp4 --hint "keep the impacts huge"
python foley_batch.py --video film.mp4 --no-mix                  REM effects only
python foley_batch.py --video film.mp4 --replan --window 10
python foley_director.py --video film.mp4 --window 15 --save     REM just the prompts
```

| Flag | Default | Meaning |
|---|---|---|
| `--prompt` | empty | one sentence for the whole video; empty = Ollama writes one per window |
| `--hint` | empty | extra note for the prompt writer, e.g. "huge impacts on the cuts" |
| `--replan` | off | ignore the cached `plan.json` and look at the video again |
| `--model` / `--ollama` | from config | which vision model writes the prompts |
| `--style` | from config | sound direction handed to the prompt writer |
| `--mix` / `--no-mix` | `MIX=1` | keep the original audio and mix the effects under it |
| `--foley-db` | `FOLEY_DB=-6` | level of the effects in that mix |
| `--loudness` | `LOUDNESS=-16` | LUFS the effects are normalised to, 0 = off |
| `--window` | 15 | seconds per render, max 30. Lower it if CUDA runs out of memory |
| `--steps` | 50 | 30 is a decent speed/quality trade |
| `--cfg` | 4.5 | higher follows the sentence harder, can sound harsher |
| `--negative` | from config | overrides the config line for this run |
| `--seed` | random | same seed + same window = same sound |
| `--restart` | off | re-render windows that already finished |
| `--dry-run` | off | print the window plan and its prompts, queue nothing |

## Landmines

- **ComfyUI must already be running**; the script says so plainly instead of
  queueing into nothing.
- **Ollama must be running for the automatic prompts**, and the model must report
  `vision` in `ollama show <model>` - a text-only model is refused with a clear
  message, and the run falls back to `foley_prompt.txt` instead of guessing.
- **VRAM**: the 27B prompt model is unloaded before the render starts. If a run
  still dies with a CUDA OOM on the first window, `ollama stop qwen3.8:27b-Q3`
  and rerun - the prompts are cached in `plan.json`, so nothing is re-read.
- **Prompt writing is not instant**: ~25-30 s per four windows on this box, so a
  6-minute lesson adds roughly 3-4 minutes before the first render.
- **More than one ComfyUI install exists on this box.** `COMFY=` in
  `foley_config.txt` pins the one that serves port 8188, because the trimmed clip
  is written into *its* `input\batchsfx_clip.mp4`. If that path is wrong the
  render fails with "cannot see batchsfx_clip.mp4" style errors.
- **The nodes must be loaded.** If ComfyUI was started before the pack was
  installed, restart it; the script checks `/object_info/HunyuanFoleySampler`.
- **`HunyuanFoleyTorchCompile` is intentionally skipped** — the portable launcher
  sets `TORCHINDUCTOR_DISABLE=1`, so compiling only costs time.
- **Model names are bare filenames** (`hunyuanvideo_foley.pth`, ...). The UI
  workflow shows a `HunyuanVideo-Foley\` prefix that is wrong for this layout.
- **Timing**: first window pays the ~10 GB checkpoint load. After that roughly
  100-150 s per 15 s window on this box (measured: 6 s clip incl. load = 116 s).
- Windows are rendered independently, so the sound can jump slightly at a slice
  boundary. Bigger `WINDOW` = fewer seams; 30 is the ceiling.
- **The vision model can be wrong about the scene.** It sees two stills, not the
  motion. If a window's sentence is nonsense, run `--dry-run`, then either rerun
  with `--replan` (the sampler is random, the model may read it differently) or
  just type the sentence yourself for that video.
