import tempfile
import unittest
from unittest import mock
from pathlib import Path

from halofpx.engine_manager import build_cache_args, get_cache_profile, get_amd_env, EngineManager, health_poll_attempts


class CacheProfileTests(unittest.TestCase):
    def test_32gb_profile(self):
        self.assertEqual(
            get_cache_profile(32),
            {"name": "32GB", "cache_ram_mib": 8192, "ctx_checkpoints": 16},
        )

    def test_64gb_profile(self):
        self.assertEqual(
            get_cache_profile(64),
            {"name": "64GB", "cache_ram_mib": 16384, "ctx_checkpoints": 32},
        )

    def test_128gb_profile(self):
        self.assertEqual(
            get_cache_profile(128),
            {"name": "128GB", "cache_ram_mib": 32768, "ctx_checkpoints": 64},
        )

    def test_engine_environment_function_is_imported(self):
        self.assertTrue(callable(get_amd_env))

    def test_128gb_cache_command(self):
        profile = get_cache_profile(128)
        profile.update({"cache_reuse": 256, "checkpoint_every": 4096})
        args = build_cache_args(profile, Path("/tmp/slots"), mmap_enabled=False, mlock=True)
        self.assertEqual(args[args.index("-cram") + 1], "32768")
        self.assertEqual(args[args.index("-ctxcp") + 1], "64")
        self.assertIn("--no-mmap", args)
        self.assertIn("--cont-batching", args)
        self.assertIn("--kv-unified", args)
        self.assertIn("--mlock", args)

    @mock.patch("halofpx.engine_manager.urllib.request.urlopen")
    @mock.patch("halofpx.engine_manager.subprocess.Popen")
    def test_auto_optimization_enables_both_mtp_and_prompt_cache(self, mock_popen, mock_urlopen):
        mock_resp = mock.Mock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        mock_proc = mock.Mock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        manager = EngineManager()
        with tempfile.NamedTemporaryFile() as fake_model, tempfile.NamedTemporaryFile() as fake_bin:
            with mock.patch.object(manager.registry, "get_model_file_path", return_value=Path(fake_model.name)), \
                 mock.patch("halofpx.engine_manager.get_engine_binary", return_value=Path(fake_bin.name)):
                res = manager.load_model("qwen38-27b", optimization_mode="auto")
                self.assertEqual(res["status"], "success")
                self.assertEqual(res["optimization_mode"], "hybrid")

                cmd = mock_popen.call_args[0][0]
                self.assertIn("--spec-type", cmd)
                self.assertIn("draft-mtp", cmd)
                self.assertIn("-ctxcp", cmd)
                self.assertIn("--cache-prompt", cmd)

                manager.unload_model()

    @mock.patch("halofpx.engine_manager.urllib.request.urlopen", side_effect=Exception("refused"))
    @mock.patch("halofpx.engine_manager.subprocess.Popen")
    def test_load_failure_captures_engine_log_tail(self, mock_popen, mock_urlopen):
        mock_proc = mock.Mock()
        mock_proc.poll.return_value = 1  # process exited immediately
        mock_popen.return_value = mock_proc

        manager = EngineManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_log = Path(tmpdir) / "engine.log"
            fake_log.write_text("ggml_vulkan: error loading tensor\nCUDA OOM simulated\n")
            manager.log_path = fake_log

            with tempfile.NamedTemporaryFile() as fake_model, tempfile.NamedTemporaryFile() as fake_bin:
                with mock.patch.object(manager.registry, "get_model_file_path", return_value=Path(fake_model.name)), \
                     mock.patch("halofpx.engine_manager.get_engine_binary", return_value=Path(fake_bin.name)):
                    res = manager.load_model("qwen38-27b")
                    self.assertEqual(res["status"], "error")
                    self.assertIn("log_tail", res)
                    self.assertIn("CUDA OOM simulated", res["log_tail"])



    @mock.patch("halofpx.engine_manager.urllib.request.urlopen")
    @mock.patch("halofpx.engine_manager.subprocess.Popen")
    def test_dspark_draft_flags_emitted_on_rocm(self, mock_popen, mock_urlopen):
        """deepseek-v4-flash (spec_type: draft-dspark) with a local drafter on ROCm0."""
        mock_resp = mock.Mock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        mock_proc = mock.Mock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        manager = EngineManager()
        with tempfile.NamedTemporaryFile() as fake_model, \
             tempfile.NamedTemporaryFile() as fake_bin, \
             tempfile.NamedTemporaryFile() as fake_draft:
            with mock.patch.object(manager.registry, "get_model_file_path", return_value=Path(fake_model.name)), \
                 mock.patch.object(manager.registry, "get_draft_file_path", return_value=Path(fake_draft.name)), \
                 mock.patch("halofpx.engine_manager.get_engine_binary", return_value=Path(fake_bin.name)):
                res = manager.load_model("deepseek-v4-flash")
                self.assertEqual(res["status"], "success")

                cmd = mock_popen.call_args[0][0]
                self.assertIn("--spec-type", cmd)
                self.assertIn("draft-dspark", cmd)
                self.assertEqual(cmd[cmd.index("-md") + 1], fake_draft.name)
                self.assertEqual(cmd[cmd.index("--spec-draft-n-max") + 1], "5")
                self.assertEqual(cmd[cmd.index("--spec-draft-p-min") + 1], "0.0")
                self.assertEqual(cmd[cmd.index("--spec-draft-ngl") + 1], "99")
                # run_config sets backend ROCm0 for this model
                self.assertEqual(cmd[cmd.index("-dev") + 1], "ROCm0")

                manager.unload_model()

    @mock.patch("halofpx.engine_manager.urllib.request.urlopen")
    @mock.patch("halofpx.engine_manager.subprocess.Popen")
    def test_dspark_vulkan_auto_upgrades_to_rocm0(self, mock_popen, mock_urlopen):
        """A user forcing --device Vulkan0 with DSpark is moved to ROCm0 automatically."""
        mock_resp = mock.Mock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        mock_proc = mock.Mock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        manager = EngineManager()
        dev_run = mock.Mock()
        dev_run.stdout = "Vulkan0\nROCm0\n"
        dev_run.stderr = ""

        with tempfile.NamedTemporaryFile() as fake_model, \
             tempfile.NamedTemporaryFile() as fake_bin, \
             tempfile.NamedTemporaryFile() as fake_draft:
            with mock.patch.object(manager.registry, "get_model_file_path", return_value=Path(fake_model.name)), \
                 mock.patch.object(manager.registry, "get_draft_file_path", return_value=Path(fake_draft.name)), \
                 mock.patch("halofpx.engine_manager.get_engine_binary", return_value=Path(fake_bin.name)), \
                 mock.patch("halofpx.engine_manager.subprocess.run", return_value=dev_run):
                res = manager.load_model("deepseek-v4-flash", device="Vulkan0")
                self.assertEqual(res["status"], "success")

                cmd = mock_popen.call_args[0][0]
                self.assertEqual(cmd[cmd.index("-dev") + 1], "ROCm0")
                self.assertIn("--spec-type", cmd)
                self.assertIn("draft-dspark", cmd)

                manager.unload_model()

    @mock.patch("halofpx.engine_manager.urllib.request.urlopen")
    @mock.patch("halofpx.engine_manager.subprocess.Popen")
    def test_dspark_disabled_when_no_hip_backend(self, mock_popen, mock_urlopen):
        """Vulkan-only host: speculation is disabled instead of crashing the engine."""
        mock_resp = mock.Mock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        mock_proc = mock.Mock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        manager = EngineManager()
        dev_run = mock.Mock()
        dev_run.stdout = "Vulkan0\n"
        dev_run.stderr = ""

        with tempfile.NamedTemporaryFile() as fake_model, \
             tempfile.NamedTemporaryFile() as fake_bin, \
             tempfile.NamedTemporaryFile() as fake_draft:
            with mock.patch.object(manager.registry, "get_model_file_path", return_value=Path(fake_model.name)), \
                 mock.patch.object(manager.registry, "get_draft_file_path", return_value=Path(fake_draft.name)), \
                 mock.patch("halofpx.engine_manager.get_engine_binary", return_value=Path(fake_bin.name)), \
                 mock.patch("halofpx.engine_manager.subprocess.run", return_value=dev_run):
                res = manager.load_model("deepseek-v4-flash", device="Vulkan0")
                self.assertEqual(res["status"], "success")

                cmd = mock_popen.call_args[0][0]
                self.assertNotIn("--spec-type", cmd)
                self.assertEqual(cmd[cmd.index("-dev") + 1], "Vulkan0")

                manager.unload_model()

    @mock.patch("halofpx.engine_manager.urllib.request.urlopen")
    @mock.patch("halofpx.engine_manager.subprocess.Popen")
    def test_dspark_skipped_without_local_drafter(self, mock_popen, mock_urlopen):
        """No local drafter: bare decode with a warning, engine still starts."""
        mock_resp = mock.Mock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        mock_proc = mock.Mock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        manager = EngineManager()
        with tempfile.NamedTemporaryFile() as fake_model, \
             tempfile.NamedTemporaryFile() as fake_bin:
            with mock.patch.object(manager.registry, "get_model_file_path", return_value=Path(fake_model.name)), \
                 mock.patch.object(manager.registry, "get_draft_file_path", return_value=None), \
                 mock.patch("halofpx.engine_manager.get_engine_binary", return_value=Path(fake_bin.name)):
                res = manager.load_model("deepseek-v4-flash")
                self.assertEqual(res["status"], "success")

                cmd = mock_popen.call_args[0][0]
                self.assertNotIn("--spec-type", cmd)

                manager.unload_model()

    def test_health_poll_attempts_floor(self):
        self.assertEqual(health_poll_attempts(0), 60)
        self.assertEqual(health_poll_attempts(4.9), 60)

    def test_health_poll_attempts_scales_with_size(self):
        self.assertEqual(health_poll_attempts(63.8), 765)
        self.assertEqual(health_poll_attempts(115.5), 1386)


if __name__ == "__main__":
    unittest.main()
