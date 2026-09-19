"""Schedule, header and key conversion: no ComfyUI needed."""

import json
import unittest

from tests.support import RAW_SIGMAS, diffusers_modules, hyperflow_metadata, hyperflow_state_dict, needs_torch

from hyperflow import keys
from hyperflow.header import HyperFlowMetadata
from hyperflow.schedule import DEFAULT_SIGMAS_8STEP, is_subgrid, shift_sigmas, validate_sigmas


class ScheduleTest(unittest.TestCase):
    def test_default_grid_is_eight_steps(self):
        self.assertEqual(len(validate_sigmas(DEFAULT_SIGMAS_8STEP)), 9)

    def test_shift_matches_formula_and_keeps_endpoints(self):
        shifted = shift_sigmas(RAW_SIGMAS, 12.0)
        self.assertEqual(shifted[0], 1.0)
        self.assertEqual(shifted[-1], 0.0)
        b = RAW_SIGMAS[-2]
        self.assertAlmostEqual(shifted[-2], 12 * b / (1 + 11 * b), places=12)

    def test_invalid_grids_raise(self):
        for bad in ([1.0], [1.0, 0.5, 0.6, 0.0], [1.0, 0.5, 0.1], [1.2, 0.5, 0.0]):
            with self.assertRaises(ValueError):
                validate_sigmas(bad)

    def test_subgrid(self):
        grid = shift_sigmas(RAW_SIGMAS, 12.0)
        self.assertTrue(is_subgrid(grid, grid))
        self.assertTrue(is_subgrid(grid[3:], grid))
        self.assertFalse(is_subgrid(shift_sigmas([1.0, 0.75, 0.5, 0.25, 0.0], 12.0), grid))
        self.assertFalse(is_subgrid(grid[:1], grid))


class HeaderTest(unittest.TestCase):
    def test_parses(self):
        meta = HyperFlowMetadata.from_dict(hyperflow_metadata())
        self.assertEqual(meta.gate, 0.7)
        self.assertEqual(meta.lora_rank, 4)
        self.assertEqual(meta.sigmas, tuple(RAW_SIGMAS))
        self.assertEqual((meta.video_shift, meta.audio_shift), (12.0, 3.0))

    def test_rejects_other_loras(self):
        with self.assertRaises(ValueError):
            HyperFlowMetadata.from_dict({"lora_rank": "32", "base_model": "minimax-h3-fl2va"})
        with self.assertRaises(ValueError):
            HyperFlowMetadata.from_dict(hyperflow_metadata(hyperflow="false"))

    def test_rejects_bad_grid(self):
        with self.assertRaises(ValueError):
            HyperFlowMetadata.from_dict(hyperflow_metadata(hyperflow_sigmas=json.dumps([1.0, 0.5])))


class KeyMapTest(unittest.TestCase):
    def test_full_size_file_maps_every_module(self):
        modules = diffusers_modules(num_layers=50)
        self.assertEqual(len(modules), 316)  # the published file's module count
        targets = {m: keys.map_module(m) for m in modules}
        self.assertEqual(sum(t.endpoint is not None for t in targets.values()), 2)
        natives = {t.native for t in targets.values() if t.endpoint is None}
        self.assertIn("blocks.49.attn.qkv_proj", natives)
        self.assertIn("token_refiner.blocks.1.mlp.fc1", natives)
        self.assertIn("time_embedder.proj_in", natives)
        self.assertIn("time_embedder.proj_out", natives)

    def test_renames(self):
        cases = {
            "transformer_blocks.7.attn.to_q": ("blocks.7.attn.qkv_proj", 0, False),
            "transformer_blocks.7.attn.to_k": ("blocks.7.attn.qkv_proj", 1, False),
            "transformer_blocks.7.attn.to_v": ("blocks.7.attn.qkv_proj", 2, False),
            "transformer_blocks.7.attn.to_out.0": ("blocks.7.attn.out_proj", None, False),
            "transformer_blocks.7.ff.net.0.proj": ("blocks.7.mlp.fc1", None, True),
            "transformer_blocks.7.ff.net.2": ("blocks.7.mlp.fc2", None, False),
            "token_refiner.refiner_blocks.1.attn.to_v": ("token_refiner.blocks.1.attn.qkv_proj", 2, False),
            "time_embedder.linear_1": ("time_embedder.proj_in", None, False),
        }
        for module, (native, qkv, swap) in cases.items():
            t = keys.map_module(module)
            self.assertEqual((t.native, t.qkv_index, t.swap_halves), (native, qkv, swap), module)
        self.assertEqual(keys.map_module("endpoint_time_embedder.linear_2").endpoint, "proj_out")

    def test_unknown_modules_raise(self):
        for bad in ("transformer_blocks.0.adaln_proj.linear", "proj_out", "time_embedder.linear_3"):
            with self.assertRaises(KeyError):
                keys.map_module(bad)

    def test_stray_keys_raise(self):
        with self.assertRaises(ValueError):
            keys.plan_from_keys(["transformer.proj_out.weight"])
        with self.assertRaises(ValueError):
            keys.plan_from_keys(["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"])


@needs_torch
class ConvertTest(unittest.TestCase):
    def test_convert(self):
        import torch

        sd = hyperflow_state_dict()
        out = keys.convert(sd, lora_alpha=8.0, qkv_rows=128)
        self.assertEqual(out.rank, 4)
        self.assertEqual(len(out.key_map), len(diffusers_modules()) - 2)
        self.assertEqual(out.key_map["transformer_blocks.1.attn.to_q"],
                         ("diffusion_model.blocks.1.attn.qkv_proj.weight", (0, 0, 128)))
        self.assertEqual(out.key_map["transformer_blocks.1.attn.to_v"][1], (0, 256, 128))
        self.assertEqual(out.key_map["transformer_blocks.1.ff.net.2"], "diffusion_model.blocks.1.mlp.fc2.weight")
        b = sd["transformer.transformer_blocks.0.ff.net.0.proj.lora_B.weight"]
        swapped = out.lora_sd["transformer_blocks.0.ff.net.0.proj.lora_B.weight"]
        self.assertTrue(torch.equal(swapped[:64], b[64:]))
        self.assertTrue(torch.equal(swapped[64:], b[:64]))
        self.assertEqual(float(out.lora_sd["transformer_blocks.0.ff.net.2.alpha"]), 8.0)
        self.assertEqual(set(out.endpoint), {"proj_in", "proj_out"})

    def test_qkv_row_mismatch_raises(self):
        with self.assertRaises(ValueError):
            keys.convert(hyperflow_state_dict(), lora_alpha=8.0, qkv_rows=64)

    def test_missing_endpoint_raises(self):
        sd = {k: v for k, v in hyperflow_state_dict().items() if "endpoint_time_embedder" not in k}
        with self.assertRaises(ValueError):
            keys.convert(sd, lora_alpha=8.0)


if __name__ == "__main__":
    unittest.main()
