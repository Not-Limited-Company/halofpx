import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from halofpx.registry import ModelRegistry

class ModelRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = ModelRegistry()

    def test_registered_models_loaded(self):
        models = self.registry.list_models()
        model_ids = [m["model_id"] for m in models]
        self.assertIn("qwen38-27b", model_ids)
        self.assertIn("ornith-1.5-35b", model_ids)
        self.assertIn("nemotron-3.5-30b", model_ids)

    def test_deepseek_v4_flash_dspark_config(self):
        model = self.registry.get_model("deepseek-v4-flash")
        self.assertIsNotNone(model)
        # DSpark drafter asset + the servable variants, each with provenance
        self.assertIn("draft", model)
        self.assertEqual(model["draft"]["filename"], "DeepSeek-V4-Flash-0731-DSpark-draft-bf16.gguf")
        self.assertIn("source", model["draft"])
        self.assertEqual(model["default_variant"], "IQ2_XXS")
        self.assertIn("ROCMFP2_STRIX", model["variants"])
        for name in ("IQ2_XXS", "ROCMFP2_STRIX"):
            self.assertIn("source", model["variants"][name])
        # IQ2_XXS is the fidelity-neutral lane: standard GGUF quant types, pinned sha
        self.assertEqual(
            model["variants"]["IQ2_XXS"]["sha256"],
            "73b86c93e7b0be37f68c4b3a5361c7e17f9b278647aa335ea304ac5dfd784867",
        )
        # Lucebox MIX (expert types 105/106 + in-file codebook sidecars) and the
        # affine vision build are not loadable by this engine, so they are not offered
        self.assertNotIn("ROCMFPX_MIX_STRIX", model["variants"])
        self.assertNotIn("VISION_EXP_STRIX_LEAN", model["variants"])
        # the lane targets the model's declared maximum window (1M) by default
        self.assertEqual(model["run_config"]["ctx_size"], 1048576)
        # run_config drives draft-dspark (block_size 5) on the HIP backend
        self.assertEqual(model["run_config"]["spec_type"], "draft-dspark")
        self.assertEqual(model["run_config"]["draft_n"], 5)
        self.assertEqual(model["run_config"]["backend"], "ROCm0")

    def test_ornith_15_config(self):
        model = self.registry.get_model("ornith-1.5-35b")
        self.assertIsNotNone(model)
        self.assertEqual(model["hf_repo"], "julianmb/Ornith-1.5-35B-A3B-ROCmFP4-GGUF")
        self.assertIn("ROCmFP4", model["variants"])
        self.assertNotIn("ROCmFP4_FAST", model["variants"])
        self.assertNotIn("ROCmFP4_STRIX_LEAN", model["variants"])
        self.assertIn("Q4_K_M", model["variants"])
        # aug-24 MTP refresh: 87.98% draft acceptance (mean 3.70/4), 105.6
        # tok/s effective vs 75.6 undrafted, measured on gfx1151 — receipt
        # glm5nextrocm/results/2026-08-28-ornith-mtp-graft-ab.md
        self.assertEqual(model["run_config"]["draft_n"], 4)
        self.assertEqual(model["run_config"]["draft_p"], 0.6)
        self.assertTrue(model["run_config"]["mtp_enabled"])
        self.assertEqual(
            model["variants"]["ROCmFP4"]["sha256"],
            "0f907917a1bfe4e0ca0d281e5709dcf34b6277063e94fab29491bb5c80fda696",
        )
        self.assertEqual(model["mmproj"]["filename"], "mmproj-Ornith-1.5-35B-BF16.gguf")
        self.assertEqual(
            model["mmproj"]["hf_repo"],
            "julianmb/Ornith-1.5-35B-A3B-ROCmFP4-GGUF",
        )

    def test_vision_readiness_is_reported_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_path = root / "models.json"
            presets_path = root / "presets.json"
            models_path.write_text(json.dumps({
                "vision-model": {
                    "default_variant": "ROCmFP4",
                    "variants": {"ROCmFP4": {"filename": "model.gguf"}},
                    "mmproj": {"filename": "mmproj.gguf", "size_gib": 0.5},
                }
            }))
            presets_path.write_text("{}")
            (root / "model.gguf").touch()

            with patch("halofpx.registry.HF_CACHE_DIRS", [root]):
                registry = ModelRegistry(models_path, presets_path)
                status = registry.list_models()[0]
                self.assertTrue(status["is_ready"])
                self.assertTrue(status["vision_capable"])
                self.assertFalse(status["vision_ready"])

                (root / "mmproj.gguf").touch()
                status = registry.list_models()[0]
                self.assertTrue(status["vision_ready"])
                self.assertTrue(status["mmproj_status"]["downloaded"])

    def test_variant_level_mmproj_is_strict_per_variant(self):
        """A projector on one variant must not leak to text-only siblings."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_path = root / "models.json"
            presets_path = root / "presets.json"
            models_path.write_text(json.dumps({
                "mixed-model": {
                    "default_variant": "TEXT",
                    "variants": {
                        "TEXT": {"filename": "text.gguf"},
                        "VISION": {
                            "filename": "vision.gguf",
                            "mmproj": {"filename": "mmproj.gguf", "size_gib": 0.5},
                        },
                    },
                }
            }))
            presets_path.write_text("{}")
            (root / "text.gguf").touch()
            (root / "vision.gguf").touch()

            with patch("halofpx.registry.HF_CACHE_DIRS", [root]):
                registry = ModelRegistry(models_path, presets_path)

                # Aggregated model view advertises vision; strict paths do not leak.
                status = registry.list_models()[0]
                self.assertTrue(status["vision_capable"])
                self.assertTrue(status["variants_status"]["VISION"]["vision_capable"])
                self.assertFalse(status["variants_status"]["TEXT"]["vision_capable"])
                self.assertIsNone(registry.get_mmproj_file_path("mixed-model", "TEXT"))
                self.assertIsNone(registry.get_mmproj_file_path("mixed-model", "VISION"))

                (root / "mmproj.gguf").touch()
                status = registry.list_models()[0]
                self.assertTrue(status["variants_status"]["VISION"]["vision_ready"])
                self.assertFalse(status["variants_status"]["TEXT"]["vision_ready"])
                self.assertEqual(
                    registry.get_mmproj_file_path("mixed-model", "VISION").name,
                    "mmproj.gguf",
                )
                self.assertIsNone(registry.get_mmproj_file_path("mixed-model", "TEXT"))

    def test_no_exist_marker_is_not_a_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker_dir = root / "models--x--y" / ".no_exist" / "abc"
            marker_dir.mkdir(parents=True)
            (marker_dir / "model.gguf").write_text("")
            models_path = root / "models.json"
            presets_path = root / "presets.json"
            models_path.write_text(json.dumps({
                "m": {"default_variant": "V", "variants": {"V": {"filename": "model.gguf"}}}
            }))
            presets_path.write_text("{}")
            with patch("halofpx.registry.HF_CACHE_DIRS", [root]):
                registry = ModelRegistry(models_path, presets_path)
                status = registry.list_models()[0]
                self.assertFalse(status["is_ready"])
                self.assertIsNone(status["variants_status"]["V"]["local_path"])

    def test_resolve_local_file_in_model_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "custom-model"
            model_dir.mkdir(parents=True)
            quant_file = model_dir / "custom-quant.gguf"
            quant_file.write_text("model_data")

            models_path = root / "models.json"
            presets_path = root / "presets.json"
            models_path.write_text(json.dumps({
                "custom-model": {
                    "default_variant": "Q4",
                    "variants": {"Q4": {"filename": "custom-quant.gguf"}}
                }
            }))
            presets_path.write_text("{}")

            with patch("halofpx.registry.HF_CACHE_DIRS", [root]):
                registry = ModelRegistry(models_path, presets_path)
                resolved = registry.get_model_file_path("custom-model", "Q4")
                self.assertIsNotNone(resolved)
                self.assertEqual(resolved, quant_file)

    def test_lemonade_cache_discovery(self):
        from halofpx.config import get_lemonade_cache_dirs
        with tempfile.TemporaryDirectory() as tmp:
            custom_lemonade = Path(tmp) / "custom_lemonade_cache"
            custom_lemonade.mkdir(parents=True)
            with patch.dict("os.environ", {"LEMONADE_CACHE_DIR": str(custom_lemonade)}):
                dirs = get_lemonade_cache_dirs()
                self.assertIn(custom_lemonade.resolve(), dirs)

if __name__ == "__main__":
    unittest.main()

