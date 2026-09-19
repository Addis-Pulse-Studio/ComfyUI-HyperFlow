"""Multi-character dialogue: timing (pure), audio placement and lip-sync segments (torch), and the audio lock sampled
through HyperFlow on a tiny MiniMax H3 (ComfyUI)."""

import json
import tempfile
import unittest

from tests.support import needs_comfy, needs_torch, tiny_model, write_hyperflow_file

from hyperflow.dialogue import DialogueLine, align_frames, frames_for_seconds, plan_dialogue


def two_speakers(**overrides):
    woman = DialogueLine("Woman", start=0.0, duration=7.5, clip_seconds=9.0, picture=1, voice=1)
    man = DialogueLine("Man", start=-1.0, duration=7.5, clip_seconds=8.0, picture=2, voice=2)
    return plan_dialogue([woman, man], **overrides)


class TimingTest(unittest.TestCase):
    def test_grid(self):
        self.assertEqual([align_frames(n) for n in (0, 5, 6, 22, 23, 360)], [5, 5, 22, 22, 39, 362])
        self.assertEqual(frames_for_seconds(15.0), 362)
        self.assertEqual(frames_for_seconds(5.0), 124)

    def test_two_speaker_scene(self):
        plan = two_speakers(scene_seconds=15.0)
        self.assertEqual(plan.frame_count, 362)
        self.assertAlmostEqual(plan.seconds, 362 / 24)
        self.assertEqual(plan.audio_latent_frames, 603)
        a, b = plan.lines
        self.assertEqual((a.speaker_id, a.start, a.end, a.first_frame, a.end_frame), ("S1", 0.0, 7.5, 0, 180))
        self.assertEqual((b.speaker_id, b.start, b.end, b.first_frame, b.end_frame), ("S2", 7.5, 15.0, 180, 360))
        self.assertEqual(plan.speaker_frames("@man"), [(180, 360)])
        # a handle never reaches into the other speaker's line: at a cut it stops dead
        self.assertEqual(plan.speaker_frames("S1", handle_frames=6), [(0, 180)])
        self.assertEqual(plan.speaker_frames("S2", handle_frames=6), [(180, 362)])
        schedule = plan.prompt_schedule()
        self.assertIn("[0.00-7.50] (S1) Woman [<Picture 1>, voice <Audio 1>]", schedule)
        self.assertIn("[7.50-15.00] (S2) Man [<Picture 2>, voice <Audio 2>]", schedule)
        self.assertEqual(json.loads(plan.to_json())["lines"][1]["frames"], [180, 360])

    def test_scene_defaults_to_last_line_and_clip_length(self):
        plan = plan_dialogue([DialogueLine("A", start=1.0, clip_seconds=3.0, source_start=0.5)])
        self.assertEqual((plan.lines[0].start, plan.lines[0].end), (1.0, 3.5))
        self.assertEqual(plan.frame_count, align_frames(84))

    def test_handles_fill_silence_only(self):
        plan = plan_dialogue([DialogueLine("A", 0.0, 2.0), DialogueLine("B", 3.0, 2.0)])
        self.assertEqual(plan.speaker_frames("A", handle_frames=12), [(0, 60)])
        self.assertEqual(plan.speaker_frames("A", handle_frames=36), [(0, 72)])
        self.assertEqual(plan.speaker_frames("B", handle_frames=36), [(48, 124)])

    def test_overlap(self):
        lines = [DialogueLine("A", 0.0, 4.0), DialogueLine("B", 3.0, 2.0)]
        with self.assertRaisesRegex(ValueError, "starts at 3.000s"):
            plan_dialogue(lines)
        self.assertEqual(len(plan_dialogue(lines, overlap="mix").lines), 2)

    def test_speaker_binding(self):
        plan = plan_dialogue([DialogueLine("Ada", 0, 1, picture=1, voice=1), DialogueLine("Ben", -1, 1, picture=2),
                              DialogueLine("@ada", -1, 1), DialogueLine("Ben", -1, 1, voice=2)])
        self.assertEqual([ln.speaker_id for ln in plan.lines], ["S1", "S2", "S1", "S2"])
        self.assertEqual((plan.speaker("ben").picture, plan.speaker("ben").voice), (2, 2))
        self.assertEqual(plan.speaker_frames("Ada"), [(0, 24), (48, 72)])
        with self.assertRaisesRegex(ValueError, "keeps one picture"):
            plan_dialogue([DialogueLine("Ada", 0, 1, picture=1), DialogueLine("Ada", -1, 1, picture=2)])
        with self.assertRaisesRegex(ValueError, "No speaker"):
            plan.speaker("Cy")

    def test_bad_lines(self):
        with self.assertRaisesRegex(ValueError, "runs to"):
            two_speakers(scene_seconds=10.0)
        with self.assertRaisesRegex(ValueError, "needs the clip"):
            plan_dialogue([DialogueLine("A", 0.0)])
        with self.assertRaisesRegex(ValueError, "no duration"):
            plan_dialogue([DialogueLine("A", 0.0, clip_seconds=1.0, source_start=2.0)])


def _tone(seconds, rate=16000, channels=1, value=1.0):
    import torch

    return {"waveform": torch.full((1, channels, round(seconds * rate)), float(value)), "sample_rate": rate}


@needs_torch
class PlacementTest(unittest.TestCase):
    def test_lines_land_on_their_samples(self):
        from hyperflow.dialogue_audio import place_lines

        plan = two_speakers(scene_seconds=15.0)
        track, notes = place_lines(plan, {0: _tone(9.0, value=1.0), 1: _tone(6.0, channels=2, value=2.0)})
        wave, rate = track["waveform"][0], track["sample_rate"]
        self.assertEqual(wave.shape, (2, round(362 / 24 * rate)))
        self.assertEqual(float(wave[0, round(7.5 * rate) - 1]), 1.0)   # woman's last sample, both channels
        self.assertEqual(float(wave[1, 0]), 1.0)
        self.assertEqual(float(wave[0, round(7.5 * rate)]), 2.0)       # man starts exactly at 7.5 s
        self.assertEqual(float(wave[0, round(13.5 * rate) - 1]), 2.0)  # his clip is 6 s ...
        self.assertEqual(float(wave[0, round(13.5 * rate)]), 0.0)      # ... then silence
        self.assertEqual(len(notes), 1)
        self.assertIn("S2 Man", notes[0])

    def test_source_start_and_resample(self):
        import torch
        from hyperflow.dialogue_audio import place_lines

        clip = {"waveform": torch.arange(32000, dtype=torch.float32).view(1, 1, -1), "sample_rate": 32000}
        plan = plan_dialogue([DialogueLine("A", 0.0, 0.5, source_start=0.25, clip_seconds=1.0)])
        track, _ = place_lines(plan, {0: clip})
        self.assertEqual(float(track["waveform"][0, 0, 0]), 8000.0)
        resampled, _ = place_lines(plan, {0: clip}, 16000)
        self.assertEqual(resampled["sample_rate"], 16000)
        self.assertEqual(resampled["waveform"].shape[-1], round(plan.seconds * 16000))

    def test_segment_paste_round_trip(self):
        import torch
        from hyperflow.dialogue_audio import cut_audio, cut_segment, paste_segment

        images = torch.rand(48, 32, 40, 3)
        frames, segment = cut_segment(images, [(4, 10), (20, 22)], box=(0.5, 0.25, 0.5, 0.5))
        self.assertEqual(frames.shape, (8, 16, 20, 3))
        self.assertEqual(segment["box"], (20, 8, 40, 24))
        same, _ = paste_segment(images, frames, segment, feather=4)
        self.assertTrue(torch.allclose(same, images))

        out, notes = paste_segment(images, torch.zeros(8, 16, 20, 3), segment, feather=0)
        self.assertEqual(float(out[4, 8:24, 20:40].abs().sum()), 0.0)
        self.assertTrue(torch.equal(out[3], images[3]))
        self.assertTrue(torch.equal(out[4, :8], images[4, :8]))
        self.assertEqual(notes, [])

        # a corrector that returns more frames at another size is mapped back
        out, notes = paste_segment(images, torch.zeros(12, 32, 40, 3), segment)
        self.assertEqual(float(out[21, 8:24, 20:40].abs().sum()), 0.0)
        self.assertEqual(len(notes), 2)

        audio = cut_audio({"waveform": torch.arange(48 * 100, dtype=torch.float32).view(1, 1, -1),
                           "sample_rate": 2400}, segment["spans"], 24)
        self.assertEqual(audio["waveform"].shape[-1], 8 * 100)
        self.assertEqual(float(audio["waveform"][0, 0, 600]), 2000.0)


class _FakeAudioVAE:
    """Stands in for the H3 audio VAE: 32 kHz in, [1, 32, 2, T] out at 40 latent frames per second."""

    audio_sample_rate = 32000

    def encode(self, wave):
        import torch

        frames = wave.shape[1] // 800
        gen = torch.Generator().manual_seed(11)
        return torch.randn(1, 32, 2, frames, generator=gen)


@needs_comfy
class AudioLockTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = write_hyperflow_file(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _latent(self, audio_t=3):
        import torch
        import comfy.nested_tensor

        return {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, 2, 4, 4),
                                                             torch.zeros(1, 32, 2, audio_t)))}

    def test_lock_sets_the_stream_and_mask(self):
        from hyperflow.dialogue_audio import lock_audio_latent

        drive = _tone(3 / 40, rate=32000)
        out, info = lock_audio_latent(self._latent(), _FakeAudioVAE(), drive, "lock_source")
        video, audio = out["samples"].unbind()
        vmask, amask = out["noise_mask"].unbind()
        self.assertEqual(audio.shape, (1, 32, 2, 3))
        self.assertEqual(float(amask.abs().max()), 0.0)
        self.assertEqual(float(vmask.min()), 1.0)
        remix, info = lock_audio_latent(self._latent(), _FakeAudioVAE(), drive, "remix_source", 0.35)
        self.assertAlmostEqual(float(remix["noise_mask"].unbind()[1].mean()), 0.35, places=6)
        same, _ = lock_audio_latent(self._latent(), _FakeAudioVAE(), drive, "native")
        self.assertNotIn("noise_mask", same)
        with self.assertRaisesRegex(ValueError, "audio\\+video latent"):
            lock_audio_latent({"samples": video}, _FakeAudioVAE(), drive)

    def test_locked_audio_survives_hyperflow_sampling_pinned(self):
        import torch
        import comfy.sample
        import comfy.samplers
        from hyperflow.dialogue_audio import lock_audio_latent
        from hyperflow.embedder import HyperFlowController
        from hyperflow.patch import apply_hyperflow
        from tests.test_comfy import TEXT_LEN, _load_nodes

        model = apply_hyperflow(tiny_model(), self.path)
        (sigmas,) = _load_nodes().NODE_CLASS_MAPPINGS["HyperFlowSigmas"]().get_sigmas(model)
        locked, _ = lock_audio_latent(self._latent(), _FakeAudioVAE(), _tone(3 / 40, rate=32000), "lock_source")
        source = locked["samples"].unbind()[1].clone()

        calls, original = [], HyperFlowController.endpoints

        def spy(controller, t_vals, ctx):
            r = original(controller, t_vals, ctx)
            calls.append((t_vals.tolist(), r.tolist()))
            return r

        gen = torch.Generator().manual_seed(5)
        samples = locked["samples"]
        noise = type(samples)(tuple(torch.randn(t.shape, generator=gen) for t in samples.unbind()))
        cond = [[torch.randn(1, TEXT_LEN, 48, generator=gen), {}]]
        HyperFlowController.endpoints = spy
        try:
            out = comfy.sample.sample_custom(model, noise, 1.0, comfy.samplers.sampler_object("euler"), sigmas,
                                             cond, cond, samples, noise_mask=locked["noise_mask"],
                                             disable_pbar=True, seed=0)
        finally:
            HyperFlowController.endpoints = original
        video, audio = out.unbind()
        self.assertTrue(torch.isfinite(video).all())
        self.assertTrue(torch.allclose(audio, source, atol=1e-5))  # the dialogue is kept, not regenerated
        self.assertEqual(len(calls), 8)
        for t_vals, r_vals in calls:  # the locked audio rows sit at t = 1 and are pinned, r = t
            pinned = [(t, r) for t, r in zip(t_vals, r_vals) if t >= 1.0 - 1e-6]
            self.assertTrue(pinned)
            self.assertTrue(all(abs(t - r) < 1e-6 for t, r in pinned))

    def test_nodes_end_to_end(self):
        import torch

        nodes = __import__("tests.test_comfy", fromlist=["_load_nodes"])._load_nodes().NODE_CLASS_MAPPINGS
        line = nodes["HyperFlowDialogueLine"]()
        (chain,) = line.add(_tone(8.0), "Woman", 1, 1, 0.0, 7.5, 0.0)
        (chain,) = line.add(_tone(8.0, value=0.5), "Man", 2, 2, -1.0, 7.5, 0.0, dialogue=chain,
                            final_audio=_tone(8.0, value=0.25))
        drive, final, length, prompt, plan, report = nodes["HyperFlowDialogueTrack"]().build(
            chain, 15.0, "error", "auto", "append", prompt="A two-shot.")
        self.assertEqual(length, 362)
        self.assertTrue(prompt.startswith("A two-shot.\n\nDialogue schedule"))
        self.assertEqual(float(drive["waveform"][0, 0, round(8 * 16000)]), 0.5)
        self.assertEqual(float(final["waveform"][0, 0, round(8 * 16000)]), 0.25)
        self.assertEqual(float(final["waveform"][0, 0, 100]), 1.0)  # an unset final falls back to drive
        self.assertEqual(json.loads(report)["frame_count"], 362)

        images = torch.rand(362, 32, 48, 3)
        seg_images, seg_audio, segment, _ = nodes["HyperFlowSpeakerSegment"]().cut(
            images, plan, "S2", "box", 0.5, 0.0, 0.5, 1.0, 0.25, "final", 25.0)
        self.assertEqual(seg_images.shape, (362 - 180, 32, 24, 3))
        self.assertEqual(seg_audio["sample_rate"], round(16000 * 25 / 24))
        (merged, _) = nodes["HyperFlowSpeakerPaste"]().paste(images, seg_images, segment, 8)
        self.assertTrue(torch.allclose(merged, images))

        import comfy.nested_tensor

        latent = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, 107, 2, 3),
                                                               torch.zeros(1, 32, 2, 603)))}
        _, mux, info = nodes["HyperFlowAudioLock"]().lock(latent, _FakeAudioVAE(), drive, "native", 0.35, final)
        self.assertEqual(mux["waveform"].shape[-1], round(362 / 24 * 16000))
        self.assertAlmostEqual(json.loads(info)["video_seconds"], 362 / 24, places=5)


if __name__ == "__main__":
    unittest.main()
