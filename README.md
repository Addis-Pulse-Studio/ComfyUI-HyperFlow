# ComfyUI-HyperFlow

[HyperFlow](https://github.com/Video-Rebirth/hyperflow) is Video Rebirth's 8-step adapter for MiniMax-H3: t2va,
fl2va and ref2va in 8 model calls instead of 49. Upstream only ships a diffusers loader. This pack runs it on
**ComfyUI's native MiniMax H3** with ComfyUI's own samplers.

You can't just drop the file into `models/loras/` and use `LoraLoaderModelOnly`, for three reasons:

| Problem | What this pack does |
|---|---|
| The file uses diffusers/PEFT names (`transformer.transformer_blocks.i.attn.to_q…`), so ComfyUI's H3 LoRA key map matches none of them and **loads nothing**. | It converts the names. `to_q/to_k/to_v` become row-slice patches of the fused `qkv_proj`, `ff.net.0.proj` becomes `mlp.fc1` with its SwiGLU halves swapped, and `time_embedder.linear_1/2` becomes `proj_in/proj_out`. All 314 native targets have to map, or loading stops with an error. |
| HyperFlow **isn't only a LoRA**. It conditions each step on the interval it integrates, `(t, r)` with `r = 1 − σ_next`, through a second, LoRA'd time embedder: `emb_t(t) + gate·(emb_r(r) − emb_t(t))`. ComfyUI's sampler never passes `r`. | It builds the endpoint embedder (the checkpoint's time embedder plus the endpoint LoRA) and patches `time_embedder.forward`. A diffusion-model wrapper then works out every row's endpoint per step: video and text get `1 − σ_next`, audio gets its own shifted endpoint, and pinned keyframe/reference rows get `r = t`. |
| Curve-form H3 checkpoints (`*_pruned_*`) have **no time embedder**; a baked `adaln_t_table` replaces it. | The loader refuses them and says why. |

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Addis-Pulse-Studio/ComfyUI-HyperFlow
```

It has no dependencies beyond ComfyUI's own.

**Weights.** HyperFlow's weights are a Model Derivative of MiniMax-H3 under the
[MiniMax H3 Community License](https://huggingface.co/videorebirth/hyperflow/blob/main/LICENSE). That licence
excludes the EU, UK, South Korea and the US unless MiniMax authorises you, and it carries an Acceptable Use Policy.
Read it before you download:

```bash
hf download videorebirth/hyperflow minimax_h3_hyperflow_8step_v1.0.safetensors --local-dir ComfyUI/models/loras/minimaxh3
```

The pre-converted file from [drbaph/Hyperflow-Comfyui](https://huggingface.co/drbaph/Hyperflow-Comfyui),
`custom_node_hyperflow_8step_v1.0_comfyui.safetensors`, works too, from `models/loras/` or `models/hyperflow/`. It
holds the same weights under ComfyUI's names, with q/k/v packed block-diagonally. The loader unpacks it into the
original layout; checked bit for bit against the Video Rebirth file, it produces identical patches. Its `mlp.fc1`
rows are still in diffusers order, and a `hyperflow_fc1_layout: gate_value` header marks a file that isn't. The
`_pruned` variant has no endpoint embedder, so it can't run the two-time recipe and the loader refuses it.

**Checkpoint.** Use a non-pruned H3 checkpoint, such as `minimax_h3_fl2va_int8_convrot` or `minimax_h3_fl2va_bf16`
(for ref2va, the `ref2va` equivalent). To check the files before you load them:

```bash
python tools/inspect_hyperflow.py models/loras/minimaxh3/minimax_h3_hyperflow_8step_v1.0.safetensors \
    models/diffusion_models/minimax/minimax_h3_fl2va_int8_convrot.safetensors
```

**First load is slow.** The first step of a fresh load includes about 5 minutes of model initialization on an int8
checkpoint. That covers dequantizing, patching and requantizing 314 targets. Steps after that take about 9–24 s each
on an RTX 5090 at 1344×768 and 124 frames.

## Use

```
UNETLoader (non-pruned H3) → HyperFlow LoRA Loader ─┬→ BasicGuider (cfg 1.0) ─┐
                                                    └→ HyperFlow Sigmas ──────┤
KSamplerSelect (euler) ───────────────────────────────────────────────────────┼→ SamplerCustomAdvanced
MiniMax H3 Image to Video / Reference to Video → positive, latent ────────────┘
```

- **HyperFlow LoRA Loader (MiniMax H3)**: takes `model`, `lora_name` and `strength`. Leave `strength` at 1.0, the
  only value HyperFlow was trained at. It scales both LoRAs together. `apply_mode` picks how the LoRA goes on:
  - `patch` merges it into the weights. On a bf16 checkpoint that is exact and costs nothing per step. On a
    quantized one ComfyUI dequantizes the weight, adds the LoRA, then hands the result to the op's `set_weight`,
    which re-quantizes it (`requantize_from_float(w, scale="recalculate", stochastic_rounding=…)`). So all 314
    weights are quantized a second time, with the delta in them — and the first load spends about 5 minutes doing it.
  - `bypass` leaves the weights **bit-identical to the checkpoint** and adds `up(down(x))` in each forward, in the
    activation's own dtype, which is how HyperFlow was distilled. q/k/v are stacked by output rows with no
    block-diagonal padding, and the SwiGLU that ComfyUI folds into `mlp.fc2`'s kernel is applied to the LoRA branch
    too, so the delta is exactly the one `patch` would merge. It loads at once, keeps the LoRA (about 3.6 GB) on
    the GPU and adds a little time per step.
  - `auto` (the default) merges when no target is quantized and bypasses when any is, and logs which it picked.

  The modes are numerically identical on a bf16 checkpoint, which is what the CPU tests check at all 8 steps. On a
  quantized one only `patch` re-quantizes, so `auto` prefers `bypass` there: it is the mode that cannot add a
  second quantization error, and it skips the 5-minute merge. The cost is the resident 3.6 GB.

  How much that error is worth is **not measured here yet** — see [docs/VERIFICATION.md](docs/VERIFICATION.md).
  To measure it on your own checkpoint, `HF_APPLY_MODE=patch` / `=bypass` in `tools/gpu_verify.py` renders the
  same seed both ways.
- **HyperFlow LoRA Loader (Advanced)**: the same, plus `gate` and `sigmas` overrides for ablations. Both default to
  the weights file's own header, so the defaults reproduce the released model.
- **HyperFlow Sigmas (8-step)**: outputs the sigma grid stored in the weights file, shifted by the model's video
  shift (12). There's no steps input.
- **HyperFlow Audio Length (audio → duration)**: measures an audio track and returns the scene length as `seconds`
  (for a duration input) and `length` (frames, snapped up to H3's 17k+5 grid), so the render lasts as long as the
  dialogue with no duration widget to keep in sync. `min_length` / `max_length` are in frames, defaulting to H3's
  trained range, 124 (5.167 s) to 362 (15.083 s): shorter audio is padded up to the floor, longer audio is cut at the
  ceiling and the cut is reported. Because only whole grid steps exist, a 13 s take renders 328 frames (13.667 s) and
  a 13.92 s take 345 (14.375 s); pad the track to the frame count so the lips and the mux stay on the same clock.

**Fixed settings:** the 8-step grid, the `euler` sampler, cfg 1.0 and shifts 12/3. The pack raises an error if the
sampler gets another schedule, or uses a multi-stage sampler (heun, dpm_2, …) that evaluates between grid points.
Either would condition every step on the wrong interval without any warning. A contiguous piece of the grid still
works, e.g. after `SplitSigmas`.

**Don't** stack HyperFlow with other step-distillation LoRAs (FastH3, H3 Turbo, TaoMate). ComfyUI attention and
offload patches are fine.

## Supported workflows

All four were run live on an RTX 5090 with the non-pruned `int8_convrot` checkpoints. See
[docs/VERIFICATION.md](docs/VERIFICATION.md) for frame checks, timings and audio-alignment numbers (those runs
predate `auto` and used `patch`).

**Audio reference.** Three different things, so pick by what you have:

| You have | Use | What H3 does with it |
|---|---|---|
| a voice you want the character to *sound like* | `ref2va`, `ref_audio_0` | speaks the prompt in that voice |
| a recording the video must *follow* | `audio2va`, `Add Guide` at frame 0 | regenerates a soundtrack that tracks its timing and loudness (envelope correlation 0.97, 0 ms lag), and moves the lips with it |
| the exact take that must *be* in the MP4 | `HyperFlow Audio Lock`, `lock_source` | never regenerates it: the real waveform is in the latent at every step and is muxed sample for sample |

| Workflow | Checkpoint | Conditioning node | Example |
|---|---|---|---|
| **t2va** — text → video + audio | `fl2va` | `MiniMax H3 Image to Video`, no frames connected | `example_workflows/hyperflow_t2va.json` |
| **fl2va** — first/last frame → video + audio | `fl2va` | `MiniMax H3 Image to Video` + `first_frame` / `last_frame` | `example_workflows/hyperflow_fl2va.json` |
| **audio2va** — external audio → synchronized video + audio | `fl2va` | `Add Guide for MiniMax H3` with `audio` at frame 0, length from `HyperFlow Audio Length` | `example_workflows/hyperflow_audio2va.json` |
| **ref2va** — reference image + reference audio → video + audio (avatar) | `ref2va` | `MiniMax H3 Reference to Video` (`ref_image_0` + `ref_audio_0`) | `example_workflows/hyperflow_ref2va.json` |
| Singularity fine-tune: T2V / I2V / Ref2V / V2V in one graph (mode groups toggled with Ctrl+B) | `ref2va` ([Singularity v1.3 int8](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity), not `_Pruned_`) | `MiniMax H3 Reference to Video` (+ `Add Guide` for I2V, `Get Video Components` for V2V) | `example_workflows/hyperflow_singularity_multimodal.json` |

One HyperFlow file serves all of them. Reference and keyframe rows, including anchored audio, are pinned with
`r = t`; only the generated rows step along the grid. In the audio-guided and Ref2VA runs, the generated soundtrack
follows the reference voice's loudness envelope with correlation 0.97–0.98, 0 ms lag and no drift.

## Multi-character dialogue

`example_workflows/hyperflow_ref2va_2speaker_dialogue.json` gives two characters their own dialogue. Each one is
bound to their own picture, voice and line, and the recordings drive the mouths and play in the MP4 at exactly their
own seconds.

```
LoadAudio ×2 → HyperFlow Dialogue Line ×2 → HyperFlow Dialogue Track ─┬ prompt, length → MiniMax H3 Reference to Video
                                                                       ├ drive/final audio → HyperFlow Audio Lock → sampler
                                                                       │                         └ mux_audio → CreateVideo
                                                                       └ plan → Speaker Segment → LatentSync → Speaker Paste
```

- **HyperFlow Dialogue Line** is one line of dialogue. It records the speaker, their `<Picture N>` and `<Audio N>`
  ordinals, and when the line starts and how long it lasts. A start of `-1` means right after the previous line.
  `drive_audio` moves the mouth. `final_audio` is optional: a clean take muxed at the same seconds.
- **HyperFlow Dialogue Track** places every line sample-exactly on one track. The track is exactly as long as the
  video: the H3 frame count divided by 24, so 15 s becomes 362 frames and a 15.083 s track. The node outputs the drive
  track, the final track, that frame count and the plan, and it can append the timing schedule to the prompt.
  Overlapping lines raise an error unless `overlap` is `mix`.
- **HyperFlow Audio Lock** takes the latent from `MiniMax H3 Reference to Video` or `Image to Video`. It
  VAE-encodes the drive track into the target audio stream. In `lock_source` mode it masks that stream at 0, so H3
  never regenerates the dialogue and the real waveform drives the mouths at every step. H3 labels those rows with its
  conditioning timestep, and HyperFlow pins them (`r = t`). `mux_audio` is the exact final track, cut to the video's
  length.
- **Speaker Segment / Speaker Paste** are for when the lock alone isn't enough. They correct one character at a time:
  - The segment node cuts out only the frames where that character speaks, and optionally only their face region. It
    also cuts their audio for exactly those frames, relabelled for LatentSync's 25 fps.
  - The paste node writes the corrected frames back at those frames.
  - Handles only extend into silence. They stop where another speaker's line begins, so one character's correction
    never repaints the other's mouth.

`remix_source` re-noises the locked audio to `remix_strength`. HyperFlow wasn't trained on partially masked rows, so
`lock_source` is the validated mode.

The drive/final split, timed windows and `lock_source` follow the pattern of
[T8mars/comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8) (GPL-3.0). No code from
it is included. This pack implements the pattern from ComfyUI's H3 latent layout and noise-mask semantics.

## How it maps onto ComfyUI

ComfyUI's `MiniMaxH3Model._forward` embeds each forward's distinct timesteps with one call,
`time_embedder(t_vals)`. Rows then index that table by `timestep × 3 + modality`, the same structure as diffusers'
`adaln_indices`. So a per-forward `t → r` map is enough, with one exception. At σ = 1 the video and audio timesteps
are both 0, so ComfyUI gives them one row, but their endpoints differ. The pack evaluates that first step at
σ·(1 − 1e-4) so the two get separate rows. That moves the sinusoidal input by less than 1e-3 rad.

ComfyUI carries the audio stream on the video schedule (`audio_scale = 12/3`). Under euler this reproduces
per-stream audio Euler exactly, so no custom sampler is needed.

## Tools

- `tools/inspect_hyperflow.py`: checks a HyperFlow file and an H3 checkpoint from their headers alone.
- `tools/gpu_verify.py`: queues the four live runs on a running ComfyUI (`COMFYUI_URL`). `HF_APPLY_MODE=patch`
  / `=bypass` renders the same seed both ways, for the comparison on your own checkpoint.
- `tools/check_av_alignment.py`: measures envelope correlation, lag and drift between a generated soundtrack and a
  reference track.
- `tools/make_example_workflows.py`: regenerates `example_workflows/`, including the two-speaker dialogue graph
  and the Singularity multimodal graph.

## Tests

```bash
COMFYUI_PATH=/path/to/ComfyUI /path/to/comfyui/python run_tests.py -v
```

The tests run on CPU against a 2-layer MiniMax H3 built from ComfyUI's real classes. The patched forward must match
an independent reference at all 8 steps, for fl2va and t2va, in both apply modes. That reference merges the LoRA in
diffusers layout, fuses it back by hand, and feeds explicit `(t, r)` pairs. A real `comfy.sample` run must chain
`r_i = t_{i+1}` for both streams. The guards must fire.

CPU in fp32 only ever takes ComfyUI's eager Linear path, so the bypass adapter is also checked against the calls a
quantized kernel makes — the SwiGLU folded into `mlp.fc2`, a folded residual, an argument it does not understand,
and the `auto` / merge-fallback routing — and against both shapes of `MLP.forward`, before and after ComfyUI PR
\#16816. `tests/test_workflows.py` lints the shipped graphs and needs neither torch nor ComfyUI.

Without `COMFYUI_PATH`, the pure, dialogue and workflow tests run.

## Licence

The code is Apache-2.0 (see `LICENSE` and `NOTICE`). Parts are ported from HyperFlow (Apache-2.0). No ComfyUI code
(GPL-3.0) is included. The weights are separate; see above. This pack isn't affiliated with Video Rebirth or MiniMax.
