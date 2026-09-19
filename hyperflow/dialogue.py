# Copyright 2026 Addis Pulse Studio. Licensed under the Apache License, Version 2.0.
"""Multi-character dialogue timing for MiniMax H3. Pure Python: no torch, no ComfyUI.

A scene's dialogue is a list of lines. Each line belongs to one speaker, who is bound to one reference picture and one
reference voice (the ``<Picture i>`` / ``<Audio j>`` ordinals of ``MiniMax H3 Reference to Video``), and occupies one
window of the film clock. The plan turns those lines into:

* the H3 frame count (24 fps, snapped up to the model's 17k + 5 grid), which fixes the scene's exact duration,
  ``frame_count / 24`` seconds -- the length of the drive track, the audio latent and the final mux alike;
* each line's window in seconds and in frames, the frames being what a per-character lip-sync pass rewrites;
* stable speaker ids (S1, S2, ...) in order of first appearance, and a timing schedule for the prompt.

The audio itself is placed by :mod:`hyperflow.dialogue_audio`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

FPS = 24
#: MiniMax H3's audio latent runs at 40 frames per second (32 kHz, 800 samples per latent frame).
AUDIO_LATENT_FPS = 40
#: A start of -1 means "right after the previous line"; a duration of -1 means "the clip's own length".
AUTO = -1.0
OVERLAP_MODES = ("error", "mix")
_EPS = 1e-6


def align_frames(frames: int) -> int:
    """The smallest H3 frame count >= ``frames`` (and >= 5) on the 17k + 5 grid."""
    n = max(5, int(frames))
    return n + (5 - n % 17) % 17


def frames_for_seconds(seconds: float, fps: float = FPS) -> int:
    """Frames needed to cover ``seconds`` (rounded up), snapped to the 17k + 5 grid."""
    return align_frames(math.ceil(float(seconds) * fps - _EPS))


def merge_spans(spans) -> list[tuple[int, int]]:
    """Sort ``[first, end)`` spans and join the ones that touch or overlap; drop empty ones."""
    out: list[tuple[int, int]] = []
    for a, b in sorted(s for s in spans if s[1] > s[0]):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def subtract_spans(spans, remove) -> list[tuple[int, int]]:
    """``spans`` with every frame in ``remove`` taken out."""
    out = merge_spans(spans)
    for ra, rb in merge_spans(remove):
        cut = []
        for a, b in out:
            if rb <= a or ra >= b:
                cut.append((a, b))
                continue
            if a < ra:
                cut.append((a, ra))
            if rb < b:
                cut.append((rb, b))
        out = cut
    return out


def speaker_key(name: str) -> str:
    """How speakers are matched: '@Ada', 'ada ' and 'Ada' are one speaker."""
    return (name or "").strip().lstrip("@").strip().casefold()


@dataclass(frozen=True)
class DialogueLine:
    """One line as authored: whose it is and where it goes. Times are seconds on the scene clock."""

    speaker: str
    start: float = AUTO
    duration: float = AUTO
    source_start: float = 0.0
    clip_seconds: float | None = None
    picture: int = 0
    voice: int = 0


@dataclass(frozen=True)
class PlannedLine:
    index: int          # position in the authored chain; indexes the line's audio clips
    speaker: str
    speaker_id: str     # S1, S2, ...
    picture: int
    voice: int
    start: float
    end: float
    source_start: float
    first_frame: int
    end_frame: int      # exclusive

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Speaker:
    speaker_id: str
    name: str
    picture: int
    voice: int


class DialoguePlan:
    def __init__(self, lines: tuple[PlannedLine, ...], speakers: tuple[Speaker, ...], frame_count: int,
                 overlap: str, fps: float = FPS):
        self.lines = lines
        self.speakers = speakers
        self.frame_count = frame_count
        self.overlap = overlap
        self.fps = fps

    @property
    def seconds(self) -> float:
        """The scene's exact duration: every track built from this plan is this long."""
        return self.frame_count / self.fps

    @property
    def audio_latent_frames(self) -> int:
        return round(self.seconds * AUDIO_LATENT_FPS)

    def speaker(self, name: str) -> Speaker:
        key = speaker_key(name)
        for s in self.speakers:
            if speaker_key(s.name) == key or s.speaker_id.casefold() == key:
                return s
        known = ", ".join(f"{s.speaker_id} {s.name}" for s in self.speakers) or "none"
        raise ValueError(f"No speaker '{name}' in this dialogue (speakers: {known}).")

    def lines_of(self, name: str) -> list[PlannedLine]:
        sid = self.speaker(name).speaker_id
        return [line for line in self.lines if line.speaker_id == sid]

    def speaker_frames(self, name: str, handle_frames: int = 0) -> list[tuple[int, int]]:
        """``[first, end)`` frame spans where ``name`` speaks, widened by ``handle_frames`` and merged.

        A handle only reaches into silence: it stops where another speaker's line begins, so a correction pass never
        repaints the other character's mouth at a cut between them.
        """
        sid = self.speaker(name).speaker_id
        own = [(ln.first_frame, ln.end_frame) for ln in self.lines if ln.speaker_id == sid]
        others = [(ln.first_frame, ln.end_frame) for ln in self.lines if ln.speaker_id != sid]
        widened = [(max(0, a - handle_frames), min(self.frame_count, b + handle_frames)) for a, b in own]
        return merge_spans(subtract_spans(widened, others) + own)

    def prompt_schedule(self) -> str:
        out = ["Dialogue schedule (exact timing of the supplied dialogue track):"]
        for line in self.lines:
            refs = []
            if line.picture:
                refs.append(f"<Picture {line.picture}>")
            if line.voice:
                refs.append(f"voice <Audio {line.voice}>")
            who = f"({line.speaker_id}) {line.speaker}" + (f" [{', '.join(refs)}]" if refs else "")
            out.append(f"[{line.start:.2f}-{line.end:.2f}] {who} speaks; only this speaker's lips move.")
        out.append("No other voices. Nobody speaks outside these windows.")
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {
            "fps": self.fps, "frame_count": self.frame_count, "seconds": round(self.seconds, 6),
            "audio_latent_frames": self.audio_latent_frames, "overlap": self.overlap,
            "speakers": [{"id": s.speaker_id, "name": s.name, "picture": s.picture, "voice": s.voice}
                         for s in self.speakers],
            "lines": [{"index": ln.index, "speaker": ln.speaker_id, "name": ln.speaker, "start": round(ln.start, 6),
                       "end": round(ln.end, 6), "source_start": round(ln.source_start, 6),
                       "frames": [ln.first_frame, ln.end_frame]} for ln in self.lines],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def plan_dialogue(lines, scene_seconds: float = 0.0, overlap: str = "error", fps: float = FPS) -> DialoguePlan:
    """Resolve authored lines into a :class:`DialoguePlan`.

    Lines are resolved in chain order, so a start of -1 follows the line authored before it (sequential
    concatenation). ``scene_seconds`` of 0 ends the scene with the last line; the frame count is then snapped up to
    the 17k + 5 grid, which is why a 15 s scene is 362 frames (15.083 s) long.
    """
    if overlap not in OVERLAP_MODES:
        raise ValueError(f"overlap must be one of {OVERLAP_MODES}, got {overlap!r}.")
    lines = list(lines)
    if not lines:
        raise ValueError("A dialogue needs at least one line.")

    resolved = []
    cursor = 0.0
    for i, line in enumerate(lines):
        name = (line.speaker or "").strip().lstrip("@").strip()
        if not name:
            raise ValueError(f"Line {i + 1} has no speaker name.")
        source_start = max(0.0, float(line.source_start))
        start = cursor if float(line.start) < 0 else float(line.start)
        duration = float(line.duration)
        if duration < 0:
            if line.clip_seconds is None:
                raise ValueError(f"Line {i + 1} ({name}): duration -1 needs the clip's length.")
            duration = float(line.clip_seconds) - source_start
        if duration <= _EPS:
            raise ValueError(f"Line {i + 1} ({name}) has no duration (source_start {source_start:.3f}s is at or "
                             "past the end of its clip, or duration is 0).")
        resolved.append((i, name, line, start, start + duration, source_start))
        cursor = start + duration

    ordered = sorted(resolved, key=lambda r: (r[3], r[0]))
    for prev, nxt in zip(ordered, ordered[1:]):
        if nxt[3] < prev[4] - _EPS and overlap == "error":
            raise ValueError(
                f"Line {nxt[0] + 1} ({nxt[1]}) starts at {nxt[3]:.3f}s, before line {prev[0] + 1} ({prev[1]}) ends "
                f"at {prev[4]:.3f}s. Move it, shorten the earlier line, or set overlap to 'mix'."
            )

    last_end = max(r[4] for r in resolved)
    scene = float(scene_seconds) if scene_seconds and scene_seconds > 0 else last_end
    if last_end > scene + _EPS:
        raise ValueError(f"The dialogue runs to {last_end:.3f}s but the scene is {scene:.3f}s long.")
    frame_count = frames_for_seconds(scene, fps)

    speakers: dict[str, Speaker] = {}
    planned = []
    for i, name, line, start, end, source_start in ordered:
        key = speaker_key(name)
        spk = speakers.get(key)
        picture, voice = int(line.picture or 0), int(line.voice or 0)
        if spk is None:
            spk = Speaker(f"S{len(speakers) + 1}", name, picture, voice)
            speakers[key] = spk
        else:
            # one character, one face, one voice: a line may leave them at 0 to inherit, never contradict them
            if (picture and spk.picture and picture != spk.picture) or (voice and spk.voice and voice != spk.voice):
                raise ValueError(
                    f"{name} is bound to <Picture {spk.picture}> / <Audio {spk.voice}> by an earlier line but line "
                    f"{i + 1} says <Picture {picture}> / <Audio {voice}>. A speaker keeps one picture and one voice."
                )
            if (picture and not spk.picture) or (voice and not spk.voice):
                spk = Speaker(spk.speaker_id, spk.name, spk.picture or picture, spk.voice or voice)
                speakers[key] = spk
        planned.append((i, spk.speaker_id, key, start, end, source_start))

    lines_out = tuple(
        PlannedLine(i, speakers[key].name, sid, speakers[key].picture, speakers[key].voice, start, end, source_start,
                    min(frame_count, round(start * fps)), min(frame_count, round(end * fps)))
        for i, sid, key, start, end, source_start in planned
    )
    return DialoguePlan(lines_out, tuple(speakers.values()), frame_count, overlap, fps)
