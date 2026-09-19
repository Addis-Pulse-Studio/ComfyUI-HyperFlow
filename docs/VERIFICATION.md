# Live GPU verification

**Date:** 2026-09-18.
**Hardware and software:** RTX 5090 (32 GB, cudaMallocAsync), Windows, ComfyUI 0.36.0, torch 2.12 + cu130.
**Weights:** `minimax_h3_hyperflow_8step_v1.0.safetensors` (HyperFlow 1.0: 316 modules, rank 256, gate 0.25, base
MiniMax-H3 @ `83db0c0`), loaded onto the non-pruned `minimax_h3_{fl2va,ref2va}_int8_convrot` checkpoints.

**How it was run:** a second ComfyUI instance loading only this pack
(`--disable-all-custom-nodes --whitelist-custom-nodes ComfyUI-HyperFlow`), with the API prompts from
`tools/gpu_verify.py`. Every run used 1344×768, 124 frames, seed 42, the `HyperFlow Sigmas` grid, `euler` and
`BasicGuider` (cfg 1.0).

## Results

| Workflow | Checkpoint | Conditioning | Result | Wall time |
|---|---|---|---|---|
| T2V | fl2va int8_convrot | text only | ✅ | 476 s (includes first load) |
| FL2V | fl2va int8_convrot | first + last frame | ✅ | 155 s |
| Audio-guided AV | fl2va int8_convrot | first frame + external voice anchored at frame 0 (`MiniMaxH3AddGuide`) | ✅ | 135 s |
| Ref2VA | ref2va int8_convrot | reference image + reference audio (`MiniMaxH3ReferenceToVideo`) | ✅ | 391 s (includes checkpoint swap) |

What held in every run:
- The loader logged `HyperFlow 1.0: 314 weight patches + two-time embedder (gate 0.25, rank 256, strength 1)`.
- The server log had no traceback, CUDA or allocator error, OOM, or `lora key not loaded` line.
- Each output had 124 video frames (5.167 s) and a 32 kHz stereo soundtrack of 5.167 s. The two streams are the
  same length.
- **Sampling speed:** the first step of a fresh load included about 5 minutes of model initialization. That covers
  loading the int8 checkpoint and dequantizing, patching and requantizing the 314 targets. After that, steps took
  9–24 s each.

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
| Audio-guided AV | **+0.974** | 0 ms | 0 / 0 ms |
| Ref2VA | **+0.980** | 0 ms | 0 / 0 ms |
| T2V (control: unrelated audio) | +0.136 | — | — |

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
