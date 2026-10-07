# Live GPU verification

**Date:** 2026-09-18.
**Hardware and software:** RTX 5090 (32 GB, cudaMallocAsync), Windows, ComfyUI 0.36.0, torch 2.12 + cu130.
**Weights:** `minimax_h3_hyperflow_8step_v1.0.safetensors` (HyperFlow 1.0: 316 modules, rank 256, gate 0.25, base
MiniMax-H3 @ `83db0c0`), loaded onto the non-pruned `minimax_h3_{fl2va,ref2va}_int8_convrot` checkpoints.

**Apply mode:** `patch` (the only mode at the time; the log line below says so). The loader now defaults to
`auto`, which picks `bypass` on these quantized checkpoints because merging re-quantizes the adapter — see the
README. **The numbers in this document are `patch` numbers and have not been re-measured in `bypass`.** To do
that: `HF_APPLY_MODE=bypass python tools/gpu_verify.py` and `HF_APPLY_MODE=patch python tools/gpu_verify.py`
render the same seed both ways, into `hyperflow_verify/<run>_<mode>`.

**How it was run:** a second ComfyUI instance loading only this pack
(`--disable-all-custom-nodes --whitelist-custom-nodes ComfyUI-HyperFlow`), with the API prompts from
`tools/gpu_verify.py`. Every run used 1344×768, 124 frames, seed 42, the `HyperFlow Sigmas` grid, `euler` and
`BasicGuider` (cfg 1.0).

## Results

| Workflow | Checkpoint | Conditioning | Result | Wall time |
|---|---|---|---|---|
| T2VA | fl2va int8_convrot | text only | ✅ | 476 s (includes first load) |
| FL2VA | fl2va int8_convrot | first + last frame | ✅ | 155 s |
| Audio2VA | fl2va int8_convrot | first frame + external voice anchored at frame 0 (`MiniMaxH3AddGuide`) | ✅ | 135 s |
| Ref2VA | ref2va int8_convrot | reference image + reference audio (`MiniMaxH3ReferenceToVideo`) | ✅ | 391 s (includes checkpoint swap) |

The run names match `tools/gpu_verify.py` and the example workflows; the 2026-09-18 runs used the earlier names
(`t2v`, `fl2v`, `av_guided`).

What held in every run:
- The loader logged `HyperFlow 1.0: 314 weight patches + two-time embedder (gate 0.25, rank 256, strength 1)`.
- The server log had no traceback, CUDA or allocator error, OOM, or `lora key not loaded` line.
- Each output had 124 video frames (5.167 s) and a 32 kHz stereo soundtrack of 5.167 s. The two streams are the
  same length.
- **Sampling speed:** the first step of a fresh load included about 5 minutes of model initialization. That covers
  loading the int8 checkpoint and dequantizing, patching and requantizing the 314 targets. After that, steps took
  9–24 s each. `bypass` does not pay that 5 minutes — it never rewrites a weight — but holds about 3.6 GB of
  adapter on the GPU and adds a little to each step; not yet measured here.

What the frames showed:
- **T2V:** the subject and scene match the prompt and stay coherent through the clip.
- **FL2V:** frame 0 reproduces the first-frame image. Frame 123 reproduces the last-frame image (center-cropped, as
  the node does for the second keyframe).
- **Audio-guided and Ref2VA:** the mouth opens during speech and is closed at the start and end. Ref2VA keeps the
  reference identity in a new, prompted scene.

### Audio alignment against the external track

This comes from `tools/check_av_alignment.py`, run on the audio inside the delivered MP4s. It compares the 10 ms
loudness envelope against the 5.3 s reference voice clip used as the external audio. The test media are private and
not part of this repository.

| Output | Envelope corr @ 0 ms | Best lag | Lag, first half / second half |
|---|---|---|---|
| Audio2VA | **+0.974** | 0 ms | 0 / 0 ms |
| Ref2VA | **+0.980** | 0 ms | 0 / 0 ms |
| T2VA (control: unrelated audio) | +0.136 | — | — |

The generated soundtrack follows the reference's timing with zero lag and no drift between the two halves. The raw
waveform correlation is low (about −0.1) by design: the audio VAE regenerates the sound rather than copying samples,
so phase differs even where timing and loudness match.

## Why there are no schedule collisions

At σ = 1, ComfyUI gives the video and audio streams one shared time-embedding row. HyperFlow needs different
endpoints for them. The pack evaluates that first step at σ·(1 − 1e-4) so the streams get separate rows. Reference
audio rows and keyframe rows stay pinned (r = t).

The CPU test `SamplingTest.test_eight_steps_chain_their_endpoints` checks three things through a real `comfy.sample`
run:
- both streams keep separate rows at every step, including step 0;
- each step's endpoint equals the next step's timestep, per stream;
- the last step ends at r = 1.

## Still to measure

`apply_mode` was added after these runs and `auto` now defaults to `bypass` on a quantized checkpoint, because
`bypass` leaves the checkpoint's weights bit-identical and so cannot add a second quantization error (README,
*HyperFlow LoRA Loader*).

How large that error is has **not** been established, and it is probably smaller than it first looks.
`ModelPatcher.patch_weight_to_device` dequantizes, merges, and calls `set_weight` →
`QuantizedTensor.requantize_from_float(w, scale="recalculate", stochastic_rounding=string_to_seed(key))`, which
re-runs `quantize(..., convrot=True, per_channel=True)`: a freshly calibrated per-output-channel quantization with
the same Hadamard rotation, not a naive round-trip. Simulating a per-tensor int8 round-trip on Gaussian weights at
rank 256 / alpha 256 moves the merged weight about 1.24%, against 1.20% for leaving it alone — and per-channel
with `convrot` should be tighter still. The mechanism is real; the magnitude is unproven.

What is **not** measured yet:

- the same seed in `patch` and in `bypass` on `*_int8_convrot`, side by side — the comparison that would settle
  whether the difference matters in practice;
- per-step time and peak VRAM in `bypass`;
- that `auto` resolves to `patch` on a `*_bf16` checkpoint and that the two modes agree there.

`HF_APPLY_MODE=bypass python tools/gpu_verify.py` then `HF_APPLY_MODE=patch python tools/gpu_verify.py` produces
the pair; `tools/check_av_alignment.py` gives the audio numbers for the audio-driven runs.
