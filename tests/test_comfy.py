"""HyperFlow on a real (tiny) ComfyUI MiniMax H3: numerical equivalence against an independent reference, and guards."""

import copy
import importlib.util
import math
import os
import sys
import tempfile
import unittest

from tests.support import (
    ALPHA, GATE, RANK, RAW_SIGMAS, ROOT, hyperflow_state_dict, needs_comfy, tiny_model, write_hyperflow_file,
)

LATENT_T, LATENT_H, LATENT_W, AUDIO_T, TEXT_LEN = 2, 4, 4, 3, 5


def _load_nodes():
    """Import the pack the way ComfyUI does: as a package from its directory."""
    name = "comfyui_hyperflow_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "__init__.py"),
                                                  submodule_search_locations=[ROOT])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _inputs(seed=3, keyframe=True):
    import torch

    gen = torch.Generator().manual_seed(seed)
    video = torch.randn(1, 24, LATENT_T, LATENT_H, LATENT_W, generator=gen)
    audio = torch.randn(1, 32, 2, AUDIO_T, generator=gen)
    context = torch.randn(1, TEXT_LEN, 48, generator=gen)
    payload = {"seed": 0}
    if keyframe:
        cond = torch.randn(1, 24, 1, LATENT_H, LATENT_W, generator=gen)
        payload["keyframes"] = [{"resolved_frame_index": 0, "latent": cond}]
        payload["cond_video_latents"] = [cond]
    return video, audio, context, payload


def _time_shift(sigma, from_shift, to_shift):
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def _grid():
    return [12 * b / (1 + 11 * b) for b in RAW_SIGMAS]


def _reference_model(patcher, sd):
    """The base diffusion model with the LoRA merged in diffusers layout, re-fused by hand, plus a two-time embedder
    that is given explicit (t, r) pairs per stream -- written independently of the pack's code paths."""
    import torch

    dm = copy.deepcopy(patcher.model.diffusion_model)
    scale = ALPHA / RANK

    def delta(module):
        a = sd[f"transformer.{module}.lora_A.weight"]
        b = sd[f"transformer.{module}.lora_B.weight"]
        return scale * (b @ a)

    with torch.no_grad():
        for prefix, native, n in (("transformer_blocks", "blocks", 2), ("token_refiner.refiner_blocks", "token_refiner.blocks", 2)):
            for i in range(n):
                blk = dm.get_submodule(f"{native}.{i}")
                q, k, v = blk.attn.qkv_proj.weight.chunk(3, dim=0)          # diffusers splits contiguous thirds
                q, k, v = (w + delta(f"{prefix}.{i}.attn.{n_}") for w, n_ in ((q, "to_q"), (k, "to_k"), (v, "to_v")))
                blk.attn.qkv_proj.weight.copy_(torch.cat([q, k, v]))
                blk.attn.out_proj.weight.add_(delta(f"{prefix}.{i}.attn.to_out.0"))
                gate, value = blk.mlp.fc1.weight.chunk(2, dim=0)            # native [gate; value]
                diffusers_fc1 = torch.cat([value, gate]) + delta(f"{prefix}.{i}.ff.net.0.proj")
                value, gate = diffusers_fc1.chunk(2, dim=0)
                blk.mlp.fc1.weight.copy_(torch.cat([gate, value]))
                blk.mlp.fc2.weight.add_(delta(f"{prefix}.{i}.ff.net.2"))
        endpoint = copy.deepcopy(dm.time_embedder)
        endpoint.proj_in.weight.add_(delta("endpoint_time_embedder.linear_1"))
        endpoint.proj_out.weight.add_(delta("endpoint_time_embedder.linear_2"))
        dm.time_embedder.proj_in.weight.add_(delta("time_embedder.linear_1"))
        dm.time_embedder.proj_out.weight.add_(delta("time_embedder.linear_2"))

    base_forward = dm.time_embedder.forward
    pairs = {}

    def forward(t_vals):
        emb_t = base_forward(t_vals)
        r = []
        for t in t_vals.tolist():
            key = min(pairs, key=lambda k: abs(k - t))
            assert abs(key - t) < 1e-5, (t, sorted(pairs))
            r.append(pairs[key] if pairs[key] is not None else t)
        r = torch.tensor(r, dtype=torch.float32)
        return emb_t + GATE * (endpoint(r) - emb_t)

    dm.time_embedder.forward = forward
    return dm, pairs


def _run_reference(dm, pairs, step, inputs):
    """One reference forward at grid step ``step``, with the per-stream pairs of upstream's build_row_time_pairs."""
    import torch

    video, audio, context, payload = inputs
    grid = _grid()
    sigma = grid[step]
    if sigma >= 1.0:
        sigma = sigma * (1.0 - 1e-4)  # the documented step-0 separation of the video and audio rows
    sigma = float(torch.tensor(sigma, dtype=torch.float32))
    nxt = grid[step + 1]
    t_v, r_v = 1.0 - sigma, 1.0 - nxt
    t_a, r_a = 1.0 - _time_shift(sigma, 12.0, 3.0), 1.0 - _time_shift(nxt, 12.0, 3.0)
    t_cond = max(t_v, 0.999)
    pairs.clear()
    pairs.update({t_v: r_v, t_a: r_a, t_cond: t_cond, 1.0: 1.0})
    ts = torch.tensor([sigma * 1000.0], dtype=torch.float32)
    with torch.no_grad():
        return dm([video, audio], ts, context, transformer_options={}, minimax_payload=dict(payload))


def _run_patched(patched, step, inputs, sigmas=None):
    import torch

    video, audio, context, payload = inputs
    grid = sigmas if sigmas is not None else _grid()
    ts = torch.tensor([grid[step] * 1000.0], dtype=torch.float32)
    options = {"wrappers": patched.wrappers, "sample_sigmas": torch.tensor(grid, dtype=torch.float32)}
    with torch.no_grad():
        return patched.model.diffusion_model([video, audio], ts, context, transformer_options=options,
                                             minimax_payload=dict(payload))


@needs_comfy
class EquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.tmp = tempfile.TemporaryDirectory()
        cls.sd = hyperflow_state_dict()
        cls.path = write_hyperflow_file(cls.tmp.name, sd={k: v.clone() for k, v in cls.sd.items()})
        cls.base = tiny_model()
        from hyperflow.patch import apply_hyperflow

        cls.reference, cls.pairs = _reference_model(cls.base, cls.sd)
        cls.patched = apply_hyperflow(cls.base, cls.path, 1.0)
        cls.patched.patch_model(device_to=torch.device("cpu"))

    @classmethod
    def tearDownClass(cls):
        cls.patched.unpatch_model(device_to=None)
        cls.tmp.cleanup()

    def _compare(self, step, keyframe=True):
        import torch

        inputs = _inputs(keyframe=keyframe)
        got = _run_patched(self.patched, step, inputs)
        want = _run_reference(self.reference, self.pairs, step, inputs)
        for g, w, name in zip(got, want, ("video", "audio")):
            err = (g - w).abs().max().item()
            self.assertLess(err, 1e-4 * max(1.0, w.abs().max().item()), f"step {step} {name}: max abs err {err}")
        return got

    def test_every_step_fl2va(self):
        for step in range(8):
            self._compare(step)

    def test_t2va_steps(self):
        for step in (0, 4, 7):
            self._compare(step, keyframe=False)

    def test_endpoint_matters(self):
        """With r = t everywhere the output must differ, or the equivalence says nothing about the endpoint."""
        import torch

        inputs = _inputs()
        got = _run_patched(self.patched, 3, inputs)
        want = _run_reference(self.reference, self.pairs, 3, inputs)  # also fills self.pairs for step 3
        for key in list(self.pairs):
            self.pairs[key] = key
        with torch.no_grad():
            wrong = self.reference([inputs[0], inputs[1]], torch.tensor([_grid()[3] * 1000.0]), inputs[2],
                                   transformer_options={}, minimax_payload=dict(inputs[3]))
        self.assertLess((got[0] - want[0]).abs().max().item(), 1e-4)
        self.assertGreater((got[0] - wrong[0]).abs().max().item(), 1e-3)

    def test_step0_rows_are_separated(self):
        """At sigma 1 video and audio need two embedding rows; the nudge moves the sinusoid input by < 1e-3 rad."""
        import torch
        import comfy.ldm.minimax.model as h3

        sigma = float(torch.tensor(1.0 - 1e-4, dtype=torch.float32))
        t_v = 1.0 - sigma
        t_a = float(1.0 - h3.time_shift_sigma(torch.tensor(sigma), 12.0, 3.0))
        self.assertGreater(abs(t_a - t_v), 1e-4)
        self.assertLess(max(t_v, t_a), 1e-3)  # freqs <= 1, so the phase shift is < 1e-3 rad


@needs_comfy
class GuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = write_hyperflow_file(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_curve_form_checkpoint_refused(self):
        from hyperflow.patch import apply_hyperflow

        with self.assertRaisesRegex(ValueError, "curve-form"):
            apply_hyperflow(tiny_model(adaln_curve_grid=8), self.path)

    def test_non_hyperflow_file_refused(self):
        from hyperflow.patch import apply_hyperflow

        other = write_hyperflow_file(self.tmp.name, metadata={"lora_rank": "4"}, name="other.safetensors")
        with self.assertRaisesRegex(ValueError, "Not a HyperFlow"):
            apply_hyperflow(tiny_model(), other)

    def test_double_load_refused(self):
        from hyperflow.patch import apply_hyperflow

        once = apply_hyperflow(tiny_model(), self.path)
        with self.assertRaisesRegex(ValueError, "already carries"):
            apply_hyperflow(once, self.path)

    def test_wrong_schedule_and_missing_context(self):
        import torch
        from hyperflow.patch import apply_hyperflow

        patched = apply_hyperflow(tiny_model(), self.path)
        patched.patch_model(device_to=torch.device("cpu"))
        try:
            inputs = _inputs()
            twenty = [12 * b / (1 + 11 * b) for b in torch.linspace(1, 0, 21).tolist()]
            with self.assertRaisesRegex(ValueError, "fixed 8-step grid"):
                _run_patched(patched, 0, inputs, sigmas=twenty)
            video, audio, context, payload = inputs
            with self.assertRaisesRegex(RuntimeError, "endpoint"):
                with torch.no_grad():
                    patched.model.diffusion_model([video, audio], torch.tensor([500.0]), context,
                                                  transformer_options={}, minimax_payload=payload)
            # a split-off tail of the grid is still the grid
            _run_patched(patched, 0, inputs, sigmas=_grid()[4:])
        finally:
            patched.unpatch_model(device_to=None)

    def test_sigmas_node(self):
        from hyperflow.patch import apply_hyperflow

        nodes = _load_nodes()
        patched = apply_hyperflow(tiny_model(), self.path)
        (sigmas,) = nodes.NODE_CLASS_MAPPINGS["HyperFlowSigmas"]().get_sigmas(patched)
        self.assertEqual(len(sigmas), 9)
        for got, want in zip(sigmas.tolist(), _grid()):
            self.assertTrue(math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-7))
        with self.assertRaisesRegex(ValueError, "no HyperFlow"):
            nodes.NODE_CLASS_MAPPINGS["HyperFlowSigmas"]().get_sigmas(tiny_model())



@needs_comfy
class SamplingTest(unittest.TestCase):
    """Through ComfyUI's real sampler: wrappers merged by sampler_helpers, audio carried at scale 12/3, euler."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = write_hyperflow_file(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _sample(self, model, sigmas):
        import torch
        import comfy.nested_tensor
        import comfy.sample
        import comfy.samplers

        gen = torch.Generator().manual_seed(5)
        video, audio = torch.zeros(1, 24, LATENT_T, LATENT_H, LATENT_W), torch.zeros(1, 32, 2, AUDIO_T)
        latent = comfy.nested_tensor.NestedTensor((video, audio))
        noise = comfy.nested_tensor.NestedTensor((torch.randn(video.shape, generator=gen),
                                                  torch.randn(audio.shape, generator=gen)))
        cond = [[torch.randn(1, TEXT_LEN, 48, generator=gen), {}]]
        return comfy.sample.sample_custom(model, noise, 1.0, comfy.samplers.sampler_object("euler"), sigmas,
                                          cond, cond, latent, disable_pbar=True, seed=0)

    def test_eight_steps_chain_their_endpoints(self):
        import torch
        from hyperflow.embedder import HyperFlowController
        from hyperflow.patch import apply_hyperflow

        model = apply_hyperflow(tiny_model(), self.path)
        (sigmas,) = _load_nodes().NODE_CLASS_MAPPINGS["HyperFlowSigmas"]().get_sigmas(model)
        calls, original = [], HyperFlowController.endpoints

        def spy(controller, t_vals, ctx):
            r = original(controller, t_vals, ctx)
            calls.append((t_vals.tolist(), r.tolist()))
            return r

        HyperFlowController.endpoints = spy
        try:
            out = self._sample(model, sigmas)
        finally:
            HyperFlowController.endpoints = original
        self.assertTrue(all(torch.isfinite(o).all() for o in out.unbind()))
        self.assertEqual(len(calls), 8)
        for (t_now, r_now), (t_next, _) in zip(calls, calls[1:]):
            self.assertEqual(len(t_now), 2)  # video and audio rows stay separate, step 0 included
            for r, t in zip(r_now, t_next):  # r_i = 1 - sigma_{i+1} = t_{i+1}, per stream
                self.assertAlmostEqual(r, t, places=5)
        self.assertEqual(calls[-1][1], [1.0, 1.0])

    def test_other_schedule_refused(self):
        import comfy.samplers
        from hyperflow.patch import apply_hyperflow

        model = apply_hyperflow(tiny_model(), self.path)
        twenty = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "simple", 20)
        with self.assertRaisesRegex(ValueError, "fixed 8-step grid"):
            self._sample(model, twenty)


if __name__ == "__main__":
    unittest.main()
