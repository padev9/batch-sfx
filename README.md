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

## What it does

The Foley sampler can only render **30 seconds at a time**, so the video is
cut into windows (default 15 s), each window is rendered with the same
ComfyUI graph, the window audios are joined into one track, and that track
is muxed onto the **untouched** video stream (stream copy - the picture is
never re-encoded). The result lands next to the original as
`<name>_foley.mp4`.

By default the original audio is **kept** and the effects are mixed under
it (`MIX=1`, `FOLEY_DB=-6` in `foley_config.txt`). `MIX=0` drops the
original track and gives you effects only.

## The two questions

| Question | Answer |
|---|---|
| **Video** | path to an `.mp4` / `.webm` / `.mkv` / `.mov` |
| **Sound** | type one sentence → it is used for *every* window; or press **Enter** → a local Ollama vision model looks at two frames per window and writes *one prompt per window*: constant ambience plus an accent on the big visual beat |

`foley_prompt.txt` is only the last-resort fallback (one sentence for the
whole video, used when Ollama cannot be reached).

## Files

| File | Role |
|---|---|
| `1 add effects.bat` | double-click entry point: asks the two questions, runs `foley_batch.py`, plays `done.mp3` when finished (or a beep) |
| `foley_batch.py` | the runner: window plan, ComfyUI queue, per-window render, join, mix, mux. Standard library + `ffmpeg`/`ffprobe` on PATH |
| `foley_director.py` | the prompt writer: two stills per window → local Ollama vision model → one sentence per window. Called automatically by `foley_batch.py` when you press Enter at the sound question |
| `foley_config.txt` | all settings: `COMFY`, `WINDOW`, `STEPS`, `CFG`, `NEGATIVE`, `OLLAMA`, `MODEL`, `STYLE`, `MIX`, `FOLEY_DB`, `LOUDNESS` |
| `foley_prompt.txt` | fallback single prompt for the whole video |
| `my_prompts.txt` | scratch notes for prompts you have tried |
| `done.mp3` | end-of-run chime |

## Setup

1. **ComfyUI** running and serving `http://127.0.0.1:8188`, with the
   `Comfyui-HunyuanFoley` custom node installed, and these model files in
   place (bare filenames, no folder prefix):
   - `hunyuanvideo_foley.pth`
   - `vae_128d_48k.pth`
   - `synchformer_state_dict.pth`
2. **ffmpeg / ffprobe** on `PATH`.
3. **Ollama** running with a *vision* model (the config default is
   `qwen3.8:27b-Q3`, ~14 GB). Only needed for the automatic prompts.
4. `COMFY=` in `foley_config.txt` points at the ComfyUI install that serves
   port 8188 (the trimmed clip is written into *its* `input` folder).

## Run

```bat
:: the easy way
start "1 add effects.bat"

:: or straight from the command line
python foley_batch.py --video "C:\clips\lesson.mp4"
python foley_batch.py --video lesson.mp4 --prompt "rain on a tin roof"
python foley_batch.py --video lesson.mp4 --hint "soft chalk on a board"
python foley_batch.py --video lesson.mp4 --dry-run     REM print the plan, render nothing
python foley_director.py --video lesson.mp4 --window 10 --save   REM just write prompts
```

### Useful flags (all optional)

| Flag | Meaning |
|---|---|
| `--window N` | seconds per window, max 30 (default from config, 15) |
| `--steps N` / `--cfg N` | sampler quality/speed (config: 50 / 4.5) |
| `--model NAME` / `--ollama URL` | override the prompt-writer model |
| `--style "..."` | sound direction handed to the prompt writer |
| `--hint "..."` | one extra note for the prompt writer this run |
| `--mix` / `--no-mix` | keep / drop the original audio |
| `--foley-db -6` | effect level under the original audio |
| `--loudness -16` | normalise effects to this LUFS before mixing, 0 = off |
| `--negative "..."` | what must not be in the result |
| `--seed N` | same seed + same window = same sound |
| `--replan` | ignore the cached prompt plan, look at the video again |
| `--restart` | re-render windows that already finished |
| `--out FILE` | output path (default `<video>_foley.mp4`) |
| `--dry-run` | print the window plan and prompts, queue nothing |

## Resuming

Each finished window is kept as `<name>_foley_work/seg_NNN.wav`. If a run
stops (VRAM, crash, Ctrl-C), run the same video again: finished windows are
reused, only the rest is rendered. `--restart` throws the pieces away.

## Timing / VRAM

- First window pays the ~10 GB checkpoint load.
- After that roughly **100–150 s per 15 s window** on a 24 GB card.
- A 6-minute lesson ≈ 4 windows ≈ 10 minutes of rendering, plus prompt
  writing if Ollama is doing it (~25–30 s per 4 windows).
- The Ollama model is unloaded again before the first render so the GPU is
  free for ComfyUI.

## Landmines

- **ComfyUI must already be running**; the script checks and says so plainly
  instead of queueing into nothing.
- **Ollama must be running** for the automatic prompts, and the model must
  report `vision` in `ollama show <model>`. A text-only model is refused with
  a clear message and the run falls back to `foley_prompt.txt`.
- **VRAM**: if a run dies with a CUDA OOM on the first window,
  `ollama stop qwen3.8:27b-Q3` and rerun - the prompts are cached in
  `<name>_foley_work/plan.json`, so nothing is re-read.
- **More than one ComfyUI install?** `COMFY=` in `foley_config.txt` pins the
  right one; a wrong path shows up as "cannot see batchsfx_clip.mp4" style
  errors.
- **`HunyuanFoleyTorchCompile` is intentionally skipped** - the portable
  launcher sets `TORCHINDUCTOR_DISABLE=1`, so compiling only costs time.
- **Model names are bare filenames** (`hunyuanvideo_foley.pth`, ...). The UI
  workflow's `HunyuanVideo-Foley\` prefix is wrong for this layout.
- Windows render independently, so the sound can jump slightly at a slice
  boundary. Bigger `WINDOW` = fewer seams; 30 is the ceiling.
- **The vision model can be wrong about the scene.** It sees two stills, not
  the motion. If a window's sentence is nonsense, rerun with `--replan` or
  just type the sentence yourself for that video.
