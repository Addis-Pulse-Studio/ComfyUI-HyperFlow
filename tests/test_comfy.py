"""HyperFlow on a real (tiny) ComfyUI MiniMax H3: numerical equivalence against an independent reference, and guards."""

import copy
import importlib.util
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

from tests.support import (
    ALPHA, GATE, RANK, RAW_SIGMAS, ROOT, hyperflow_metadata, hyperflow_state_dict, native_state_dict, needs_comfy,
    needs_int8, quantize_lora_targets, tiny_model, write_hyperflow_file,
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

    def test_preconverted_file_matches(self):
        """The same weights as a drbaph-style file (native names, fused q|k|v, both fc1 orders) patch identically."""
        import torch
        from hyperflow import keys
        from hyperflow.patch import apply_hyperflow

        inputs = _inputs()
        for layout in ("value_gate", "gate_value"):
            header = hyperflow_metadata(**({keys.FC1_LAYOUT_KEY: layout} if layout == "gate_value" else {}))
            path = write_hyperflow_file(self.tmp.name, sd=native_state_dict(self.sd, layout), metadata=header,
                                        name=f"native_{layout}.safetensors")
            patched = apply_hyperflow(tiny_model(), path, 1.0)
            patched.patch_model(device_to=torch.device("cpu"))
            try:
                for step in (0, 3, 7):
                    got = _run_patched(patched, step, inputs)
                    want = _run_reference(self.reference, self.pairs, step, inputs)
                    for g, w in zip(got, want):
                        self.assertLess((g - w).abs().max().item(), 1e-4 * max(1.0, w.abs().max().item()), layout)
            finally:
                patched.unpatch_model(device_to=None)

    def test_bypass_matches(self):
        """apply_mode 'bypass' (forward hooks, fused q|k|v stacked by rows) equals the merged weights at every step."""
        import torch
        from hyperflow.patch import INJECTION_KEY, apply_hyperflow

        base = tiny_model()
        before = {k: v.clone() for k, v in base.model.diffusion_model.state_dict().items()}
        bypass = apply_hyperflow(base, self.path, 1.0, "bypass")
        self.assertFalse(bypass.patches)
        self.assertEqual(len(bypass.get_injections(INJECTION_KEY)), 1)
        # the hooks move adapters to ComfyUI's compute device; this test computes on CPU
        with mock.patch("comfy.model_management.get_torch_device", return_value=torch.device("cpu")):
            bypass.patch_model(device_to=torch.device("cpu"))
        try:
            after = bypass.model.diffusion_model.state_dict()
            self.assertTrue(all(torch.equal(after[k], v) for k, v in before.items()), "bypass changed a weight")
            inputs = _inputs()
            for step in range(8):
                got = _run_patched(bypass, step, inputs)
                want = _run_reference(self.reference, self.pairs, step, inputs)
                for g, w, name in zip(got, want, ("video", "audio")):
                    err = (g - w).abs().max().item()
                    self.assertLess(err, 1e-4 * max(1.0, w.abs().max().item()), f"bypass step {step} {name}: {err}")
        finally:
            bypass.unpatch_model(device_to=None)
        # ejected on unpatch: the plain model is untouched again
        self.assertEqual(type(base.model.diffusion_model.blocks[0].attn.qkv_proj).forward,
                         base.model.diffusion_model.blocks[0].attn.qkv_proj.forward.__func__)

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
class BypassAdapterTest(unittest.TestCase):
    """Bypass against the merged weight on the Linear calls ComfyUI actually makes.

    ``EquivalenceTest.test_bypass_matches`` runs the whole model, which on CPU in fp32 only ever exercises the
    eager path. These cover what it cannot see: the activation ComfyUI folds into ``mlp.fc2``'s own kernel, a
    folded residual, an argument the adapter does not understand, and the auto / merge-fallback routing.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.sd = hyperflow_state_dict()
        cls.path = write_hyperflow_file(cls.tmp.name, sd={k: v.clone() for k, v in cls.sd.items()})
        cls.reference, cls.pairs = _reference_model(tiny_model(), cls.sd)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _delta(self, module):
        """The merged delta of one diffusers module, as the reference computes it."""
        a = self.sd[f"transformer.{module}.lora_A.weight"]
        b = self.sd[f"transformer.{module}.lora_B.weight"]
        return (ALPHA / RANK) * (b @ a)

    def _bypassed(self, model=None, mode="bypass"):
        """A bypass-mode model with its hooks injected on CPU (they otherwise move to the compute device)."""
        import torch
        from hyperflow.patch import apply_hyperflow

        patched = apply_hyperflow(model if model is not None else tiny_model(), self.path, 1.0, mode)
        with mock.patch("comfy.model_management.get_torch_device", return_value=torch.device("cpu")):
            patched.patch_model(device_to=torch.device("cpu"))
        self.addCleanup(patched.unpatch_model, device_to=None)
        return patched

    def test_folded_swiglu_is_exact(self):
        """ComfyUI calls fc2(fc1(x), input_act="swiglu"), so the hook sees the pre-SwiGLU tensor (twice as wide)."""
        import torch
        import comfy.ops

        fc2 = self._bypassed().model.diffusion_model.blocks[0].mlp.fc2
        merged = fc2.weight + self._delta("transformer_blocks.0.ff.net.2")
        x = torch.randn(7, fc2.weight.shape[1] * 2, generator=torch.Generator().manual_seed(11))
        got = fc2(x, input_act="swiglu")
        want = torch.nn.functional.linear(comfy.ops.INPUT_ACT_EAGER["swiglu"](x.clone()), merged)
        self.assertEqual(tuple(got.shape), tuple(want.shape))
        self.assertLess((got - want).abs().max().item(), 1e-4 * max(1.0, want.abs().max().item()))

    def test_folded_residual_is_exact(self):
        """linear(..., residual=r, residual_scale=s) returns r + s * linear(x), so the delta carries the same s."""
        import torch

        gen = torch.Generator().manual_seed(12)
        out_proj = self._bypassed().model.diffusion_model.blocks[0].attn.out_proj
        merged = out_proj.weight + self._delta("transformer_blocks.0.attn.to_out.0")
        x = torch.randn(5, out_proj.weight.shape[1], generator=gen)
        residual = torch.randn(5, out_proj.weight.shape[0], generator=gen)
        scale = torch.tensor(0.37)
        got = out_proj(x, residual=residual, residual_scale=scale)
        want = residual + scale * torch.nn.functional.linear(x, merged)
        self.assertLess((got - want).abs().max().item(), 1e-4 * max(1.0, want.abs().max().item()))

    def test_unknown_folded_argument_raises(self):
        """An argument the adapter cannot mirror must say so, not compute a wrong delta or raise a TypeError."""
        import torch

        fc1 = self._bypassed().model.diffusion_model.blocks[0].mlp.fc1
        x = torch.randn(3, fc1.weight.shape[1])
        with self.assertRaisesRegex(ValueError, "patch"):
            fc1(x, act_eps=1e-6)

    def test_matches_with_the_activation_applied_outside(self):
        """Before ComfyUI PR #16816 the MLP applied SwiGLU itself; the adapter must be right either way."""
        import torch
        import comfy.ldm.minimax.model as h3

        def forward(self, x):  # the pre-#16816 shape of MLP.forward
            import comfy.ops

            return self.fc2(comfy.ops.INPUT_ACT_EAGER["swiglu"](self.fc1(x)))

        inputs = _inputs()
        bypass = self._bypassed()
        with mock.patch.object(h3.MLP, "forward", forward):
            for step in (0, 4, 7):
                got = _run_patched(bypass, step, inputs)
                want = _run_reference(self.reference, self.pairs, step, inputs)
                for g, w, name in zip(got, want, ("video", "audio")):
                    err = (g - w).abs().max().item()
                    self.assertLess(err, 1e-4 * max(1.0, w.abs().max().item()), f"step {step} {name}: {err}")

    def test_auto_merges_an_unquantized_checkpoint(self):
        from hyperflow.patch import INJECTION_KEY, OPTIONS_KEY, apply_hyperflow

        model = apply_hyperflow(tiny_model(), self.path, 1.0, "auto")
        self.assertEqual(model.model_options[OPTIONS_KEY]["apply_mode"], "patch")
        self.assertTrue(model.patches)
        self.assertIsNone(model.get_injections(INJECTION_KEY))

    def test_auto_bypasses_when_a_target_is_quantized(self):
        """One quantized target is enough: merging would requantize the whole adapter away."""
        from hyperflow.patch import INJECTION_KEY, OPTIONS_KEY, apply_hyperflow

        base = tiny_model()
        base.model.diffusion_model.blocks[0].mlp.fc2.quant_format = "int8_tensorwise"
        model = apply_hyperflow(base, self.path, 1.0, "auto")
        self.assertEqual(model.model_options[OPTIONS_KEY]["apply_mode"], "bypass")
        self.assertFalse(model.patches)
        self.assertEqual(len(model.get_injections(INJECTION_KEY)), 1)

    def test_unhookable_target_falls_back_to_merge(self):
        """A target ComfyUI cannot hook is merged instead of silently dropped, and the output still matches."""
        import comfy.weight_adapter

        target = "diffusion_model.blocks.0.mlp.fc2"
        original = comfy.weight_adapter.BypassInjectionManager._get_module_by_key

        def unhookable(manager, model, key):
            return None if key == target else original(manager, model, key)

        with mock.patch.object(comfy.weight_adapter.BypassInjectionManager, "_get_module_by_key", unhookable):
            model = self._bypassed()
        self.assertEqual(sorted(str(k) for k in model.patches), [f"{target}.weight"])
        inputs = _inputs()
        for step in (0, 7):
            got = _run_patched(model, step, inputs)
            want = _run_reference(self.reference, self.pairs, step, inputs)
            for g, w, name in zip(got, want, ("video", "audio")):
                self.assertLess((g - w).abs().max().item(), 1e-4 * max(1.0, w.abs().max().item()), f"{step} {name}")


@needs_comfy
@needs_int8
class QuantizedBypassTest(unittest.TestCase):
    """Bypass on genuinely INT8 weights -- the configuration the pack is actually used in.

    An INT8 Linear folds ``mlp.fc2``'s SwiGLU into its own kernel, which the fp32 tests never reach: there ComfyUI
    takes its eager path and fc2 is called with the activation already applied.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sd = hyperflow_state_dict()
        self.path = write_hyperflow_file(self.tmp.name, sd={k: v.clone() for k, v in self.sd.items()})
        self.addCleanup(self.tmp.cleanup)

    def test_auto_bypasses_and_the_lora_reaches_the_int8_kernel(self):
        import torch
        from hyperflow.patch import OPTIONS_KEY, apply_hyperflow

        quantized = tiny_model()
        dequantized = quantize_lora_targets(quantized)
        inputs = _inputs()
        with torch.no_grad():  # the same INT8 weights, no adapter: what a dropped LoRA would look like
            plain = quantized.model.diffusion_model(
                [inputs[0], inputs[1]], torch.tensor([_grid()[4] * 1000.0]), inputs[2],
                transformer_options={}, minimax_payload=dict(inputs[3]))

        # the reference runs the exact LoRA on the dequantized values, in fp32
        base = tiny_model()
        with torch.no_grad():
            for name, weight in dequantized.items():
                base.model.diffusion_model.get_submodule(name).weight.copy_(weight)
        reference, pairs = _reference_model(base, self.sd)

        model = apply_hyperflow(quantized, self.path, 1.0, "auto")
        self.assertEqual(model.model_options[OPTIONS_KEY]["apply_mode"], "bypass")
        self.assertFalse(model.patches, "a target fell back to merge on a plain INT8 checkpoint")
        with mock.patch("comfy.model_management.get_torch_device", return_value=torch.device("cpu")):
            model.patch_model(device_to=torch.device("cpu"))
        try:
            for step in (0, 4, 7):
                got = _run_patched(model, step, inputs)
                want = _run_reference(reference, pairs, step, inputs)
                for g, w, name in zip(got, want, ("video", "audio")):
                    # loose: the INT8 GEMM quantizes activations too, which the fp32 reference does not
                    err = (g - w).abs().max().item() / max(1.0, w.abs().max().item())
                    self.assertLess(err, 2e-2, f"step {step} {name}: {err:.2e}")
            # and the adapter must actually be doing something: a silently dropped fc2 LoRA would look like `plain`
            adapted = _run_patched(model, 4, inputs)
            self.assertGreater((adapted[0] - plain[0]).abs().max().item(), 1e-2)
        finally:
            model.unpatch_model(device_to=None)


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



@needs_comfy
class SchedulerAndRetimeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = write_hyperflow_file(self.tmp.name)
        self.nodes = _load_nodes()  # importing the pack registers the scheduler

    def tearDown(self):
        self.tmp.cleanup()

    def test_scheduler_name_gives_the_grid(self):
        import comfy.samplers
        from hyperflow.patch import apply_hyperflow

        self.assertIn("hyperflow", comfy.samplers.SCHEDULER_NAMES)
        self.assertIn("hyperflow", comfy.samplers.KSampler.SCHEDULERS)
        model = apply_hyperflow(tiny_model(), self.path)
        by_name = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "hyperflow", 8)
        (by_node,) = self.nodes.NODE_CLASS_MAPPINGS["HyperFlowSigmas"]().get_sigmas(model)
        self.assertEqual([round(x, 6) for x in by_name.tolist()], [round(x, 6) for x in by_node.tolist()])
        with self.assertRaisesRegex(ValueError, "8-step"):
            comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "hyperflow", 20)

    def test_scheduler_name_samples_through_the_guard(self):
        import comfy.samplers
        from hyperflow.patch import apply_hyperflow

        model = apply_hyperflow(tiny_model(), self.path)
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "hyperflow", 8)
        out = SamplingTest._sample(self, model, sigmas)
        self.assertEqual(len(out.unbind()), 2)

    def test_retime_relabels_the_rate_only(self):
        import torch

        wave = torch.randn(1, 2, 48000)
        (out,) = self.nodes.NODE_CLASS_MAPPINGS["HyperFlowRetimeAudio"]().retime(
            {"waveform": wave, "sample_rate": 48000}, 24.0, 25.0)
        self.assertEqual(out["sample_rate"], 50000)
        self.assertIs(out["waveform"], wave)
        self.assertAlmostEqual(wave.shape[-1] / out["sample_rate"], 0.96)  # 24 frames' audio now lasts 24/25 s


if __name__ == "__main__":
    unittest.main()
