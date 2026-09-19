# Copyright 2026 Addis Pulse Studio. Licensed under the Apache License, Version 2.0.
"""Audio and frame operations for the dialogue pipeline: placing lines, locking the drive audio into an H3 latent,
and cutting one character's frames out for lip-sync correction and back in again.

The lock works on ComfyUI's native MiniMax H3 AV latent, a nested ``(video [B, 24, T, H, W], audio [B, 32, 2, Ta])``
with ``Ta = round(seconds * 40)``. A nested ``noise_mask`` gives each stream its own mask: an audio mask of 0 keeps the
supplied audio latent as it is at every step, and H3 labels those rows with its conditioning timestep, so HyperFlow
pins them (``r = t``) like any other anchor. A mask between 0 and 1 re-noises the audio to that strength.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("HyperFlow")

LOCK_MODES = ("lock_source", "remix_source", "native")


# -- audio ---------------------------------------------------------------------------------------------------------
def conform(audio: dict, sample_rate: int, channels: int) -> torch.Tensor:
    """The first batch item of an AUDIO as float32 ``[channels, samples]`` at ``sample_rate``."""
    wave = audio["waveform"]
    if wave.ndim == 2:
        wave = wave.unsqueeze(0)
    wave = wave[0].to(torch.float32)
    rate = int(audio["sample_rate"])
    if rate != int(sample_rate):
        import torchaudio

        wave = torchaudio.functional.resample(wave, rate, int(sample_rate))
    if wave.shape[0] == channels:
        return wave
    if wave.shape[0] == 1:
        return wave.expand(channels, -1).clone()
    if channels == 1:
        return wave.mean(dim=0, keepdim=True)
    return wave[:channels]


def seconds_of(audio: dict) -> float:
    return audio["waveform"].shape[-1] / float(audio["sample_rate"])


def fit_duration(audio: dict, seconds: float) -> dict:
    """Pad with silence or cut so the track lasts exactly ``seconds`` (to the sample)."""
    wave = audio["waveform"]
    rate = int(audio["sample_rate"])
    target = round(seconds * rate)
    if wave.shape[-1] > target:
        wave = wave[..., :target]
    elif wave.shape[-1] < target:
        wave = torch.nn.functional.pad(wave, (0, target - wave.shape[-1]))
    return {"waveform": wave, "sample_rate": rate}


def place_lines(plan, clips: dict, sample_rate: int = 0) -> tuple[dict, list[str]]:
    """One track, ``plan.seconds`` long, with every line's clip at its window and silence everywhere else.

    ``clips`` maps a line's ``index`` to its AUDIO (or None for silence). Each clip is read from its ``source_start``
    for exactly the line's duration: a longer clip is cut, a shorter one is followed by silence (and reported).
    Overlapping lines (overlap 'mix') are summed.
    """
    present = [c for c in clips.values() if c is not None]
    if not present:
        raise ValueError("None of the dialogue lines has audio.")
    rate = int(sample_rate) or int(present[0]["sample_rate"])
    channels = 2 if any(c["waveform"].shape[-2] >= 2 for c in present) else 1
    total = round(plan.seconds * rate)
    track = torch.zeros(channels, total, dtype=torch.float32)
    notes = []
    for line in plan.lines:
        clip = clips.get(line.index)
        if clip is None:
            continue
        wave = conform(clip, rate, channels)
        a, b = round(line.start * rate), min(total, round(line.end * rate))
        s0 = round(line.source_start * rate)
        piece = wave[:, s0:s0 + (b - a)]
        if piece.shape[-1] < b - a:
            notes.append(
                f"line {line.index + 1} ({line.speaker_id} {line.speaker}): the clip has "
                f"{piece.shape[-1] / rate:.3f}s from {line.source_start:.3f}s for a {line.duration:.3f}s window; "
                "the rest of the window is silence."
            )
        track[:, a:a + piece.shape[-1]] += piece
    return {"waveform": track.unsqueeze(0), "sample_rate": rate}, notes


# -- latent lock ---------------------------------------------------------------------------------------------------
def av_parts(latent: dict) -> tuple[torch.Tensor, torch.Tensor]:
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if samples is None or not getattr(samples, "is_nested", False):
        raise ValueError("Expected a MiniMax H3 audio+video latent (from 'MiniMax H3 Reference to Video', "
                         "'MiniMax H3 Image to Video' or 'Empty MiniMax H3 Latent AV').")
    parts = samples.unbind()
    if len(parts) != 2 or parts[0].ndim != 5 or parts[1].ndim != 4:
        raise ValueError("Expected a MiniMax H3 latent of (video [B, C, T, H, W], audio [B, C, 2, T]); got "
                         f"{[tuple(p.shape) for p in parts]}.")
    return parts[0], parts[1]


def encode_audio(audio_vae, audio: dict) -> torch.Tensor:
    rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    wave = conform(audio, rate, 2)
    latent = audio_vae.encode(wave.unsqueeze(0).movedim(1, -1))
    if not isinstance(latent, torch.Tensor) or latent.ndim != 4:
        raise ValueError("The audio VAE did not return a [B, C, 2, T] latent; connect the MiniMax H3 audio VAE.")
    return latent


def lock_audio_latent(latent: dict, audio_vae, audio: dict, mode: str = "lock_source",
                      strength: float = 0.35) -> tuple[dict, dict]:
    """Put ``audio`` into the latent's audio stream and mask it. Returns ``(latent, info)``."""
    if mode not in LOCK_MODES:
        raise ValueError(f"mode must be one of {LOCK_MODES}, got {mode!r}.")
    video, target = av_parts(latent)
    info = {"mode": mode, "audio_latent_frames": int(target.shape[-1])}
    if mode == "native":
        return latent, info

    import comfy.nested_tensor

    encoded = encode_audio(audio_vae, audio)
    if encoded.shape[1:3] != target.shape[1:3]:
        raise ValueError(f"The audio VAE gave a {tuple(encoded.shape)} latent; this H3 latent's audio stream is "
                         f"{tuple(target.shape)}. Connect the MiniMax H3 audio VAE.")
    frames = encoded.shape[-1]
    info["drive_latent_frames"] = int(frames)
    if frames > target.shape[-1]:
        encoded = encoded[..., :target.shape[-1]]
    elif frames < target.shape[-1]:
        encoded = torch.nn.functional.pad(encoded, (0, target.shape[-1] - frames))
    if abs(frames - target.shape[-1]) > 1:
        logger.warning("HyperFlow audio lock: the drive audio is %d audio-latent frames, the video's audio stream %d; "
                       "it was %s. Build the drive track with 'HyperFlow Dialogue Track' to match exactly.",
                       frames, target.shape[-1], "cut" if frames > target.shape[-1] else "padded")
    encoded = encoded.to(device=target.device, dtype=target.dtype).expand(target.shape[0], -1, -1, -1).contiguous()

    masks = latent.get("noise_mask")
    if masks is not None and getattr(masks, "is_nested", False):
        video_mask = masks.unbind()[0]
    else:
        if masks is not None:
            logger.warning("HyperFlow audio lock: ignoring a non-nested noise_mask on an H3 latent.")
        video_mask = torch.ones_like(video)
    value = 0.0 if mode == "lock_source" else float(strength)
    info["audio_mask"] = value

    out = dict(latent)
    out["samples"] = comfy.nested_tensor.NestedTensor((video, encoded))
    out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, torch.full_like(encoded, value)))
    return out, info


# -- per-character correction ----------------------------------------------------------------------------------------
def box_pixels(height: int, width: int, box) -> tuple[int, int, int, int]:
    """Normalized ``(x, y, w, h)`` -> pixel ``(x0, y0, x1, y1)``, even-sized, inside the frame. None = full frame."""
    if box is None:
        return 0, 0, width, height
    x, y, w, h = (float(v) for v in box)
    x0 = min(max(0, round(x * width)), width - 2)
    y0 = min(max(0, round(y * height)), height - 2)
    x1 = min(width, max(x0 + 2, round((x + w) * width)))
    y1 = min(height, max(y0 + 2, round((y + h) * height)))
    x1 -= (x1 - x0) % 2
    y1 -= (y1 - y0) % 2
    return x0, y0, x1, y1


def span_indices(spans) -> list[int]:
    return [i for a, b in spans for i in range(a, b)]


def cut_segment(images: torch.Tensor, spans, box=None) -> tuple[torch.Tensor, dict]:
    """The frames of ``spans`` (``[first, end)`` pairs), cropped to ``box``, plus what pasting them back needs."""
    n, height, width = images.shape[0], images.shape[1], images.shape[2]
    spans = [(max(0, a), min(n, b)) for a, b in spans if min(n, b) > max(0, a)]
    indices = span_indices(spans)
    if not indices:
        raise ValueError("This speaker has no frames in the video.")
    x0, y0, x1, y1 = box_pixels(height, width, box)
    segment = {"indices": indices, "spans": spans, "box": (x0, y0, x1, y1), "frames": n,
               "height": height, "width": width}
    return images[indices, y0:y1, x0:x1, :].clone(), segment


def cut_audio(audio: dict, spans, fps: float) -> dict:
    """The audio under ``spans`` (frame pairs at ``fps``), joined in order -- the same seconds as ``cut_segment``."""
    rate = int(audio["sample_rate"])
    wave = audio["waveform"]
    pieces = []
    for a, b in spans:
        s0, s1 = round(a / fps * rate), round(b / fps * rate)
        piece = wave[..., s0:s1]
        if piece.shape[-1] < s1 - s0:
            piece = torch.nn.functional.pad(piece, (0, s1 - s0 - piece.shape[-1]))
        pieces.append(piece)
    return {"waveform": torch.cat(pieces, dim=-1), "sample_rate": rate}


def _feather_mask(h: int, w: int, feather: int, edges: tuple[bool, bool, bool, bool]) -> torch.Tensor:
    """``[h, w, 1]`` weights ramping 0 -> 1 over ``feather`` px from each edge flagged in (left, top, right, bottom)."""
    mask = torch.ones(h, w)
    if feather <= 0:
        return mask.unsqueeze(-1)
    ramp_x = torch.ones(w)
    ramp_y = torch.ones(h)
    fx, fy = min(feather, w // 2), min(feather, h // 2)
    steps_x = (torch.arange(fx, dtype=torch.float32) + 1) / (fx + 1)
    steps_y = (torch.arange(fy, dtype=torch.float32) + 1) / (fy + 1)
    left, top, right, bottom = edges
    if left and fx:
        ramp_x[:fx] = torch.minimum(ramp_x[:fx], steps_x)
    if right and fx:
        ramp_x[w - fx:] = torch.minimum(ramp_x[w - fx:], steps_x.flip(0))
    if top and fy:
        ramp_y[:fy] = torch.minimum(ramp_y[:fy], steps_y)
    if bottom and fy:
        ramp_y[h - fy:] = torch.minimum(ramp_y[h - fy:], steps_y.flip(0))
    return (ramp_y[:, None] * ramp_x[None, :] * mask).unsqueeze(-1)


def paste_segment(images: torch.Tensor, corrected: torch.Tensor, segment: dict, feather: int = 0) -> tuple[torch.Tensor, list[str]]:
    """Write ``corrected`` back over the frames and region ``segment`` was cut from.

    A corrector that returns a different number of frames (LatentSync pads to its own grid) is mapped back by nearest
    index; one that returns another size is resized to the region. Edges that touch the frame border are not feathered.
    """
    notes = []
    if images.shape[0] != segment["frames"] or images.shape[1:3] != (segment["height"], segment["width"]):
        raise ValueError(f"These frames ({tuple(images.shape[:3])}) are not the ones the segment was cut from "
                         f"({segment['frames']}, {segment['height']}, {segment['width']}).")
    indices = segment["indices"]
    x0, y0, x1, y1 = segment["box"]
    h, w = y1 - y0, x1 - x0
    n, m = len(indices), corrected.shape[0]
    if m != n:
        if abs(m - n) > 2:
            notes.append(f"the corrector returned {m} frames for {n}; mapped back by nearest frame.")
        pick = torch.tensor([min(m - 1, (i * m) // n) for i in range(n)])
        corrected = corrected[pick]
    corrected = corrected[..., :images.shape[-1]].to(images.dtype)
    if corrected.shape[1:3] != (h, w):
        notes.append(f"resized the corrected frames from {tuple(corrected.shape[1:3])} to {(h, w)}.")
        corrected = torch.nn.functional.interpolate(
            corrected.movedim(-1, 1), size=(h, w), mode="bilinear", align_corners=False).movedim(1, -1)
    corrected = corrected.to(images.device)
    edges = (x0 > 0, y0 > 0, x1 < segment["width"], y1 < segment["height"])
    mask = _feather_mask(h, w, int(feather), edges).to(images.device, images.dtype)
    out = images.clone()
    region = out[indices, y0:y1, x0:x1, :]
    out[indices, y0:y1, x0:x1, :] = region + mask * (corrected - region)
    return out, notes
