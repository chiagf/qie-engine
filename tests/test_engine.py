"""CPU-only tests for qie-engine. No torch, no gradio, no network. Run from the repo root:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import ast
import builtins
import os
import pathlib
import py_compile
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import qie_engine  # noqa: E402
from qie_engine import runtime, settings, wan  # noqa: E402
from PIL import Image  # noqa: E402

from qie_engine.rife import RifeInterpolator, blend_interpolate  # noqa: E402


def _frames(count: int):
    """A real blendable sequence: a black frame ramping to white. Strings cannot be blended."""
    out = []
    for index in range(count):
        value = 0 if count == 1 else round(255 * index / (count - 1))
        out.append(Image.new("RGB", (8, 8), (value, value, value)))
    return out

WAN_ENV = (
    "QIE_WAN_STACK",
    "QIE_WAN_DTYPE",
    "QIE_WAN_OFFLOAD",
    "QIE_ATTN",
    "QIE_WAN_STEPS",
    "QIE_WAN_SHIFT",
    "QIE_WAN_FPS",
    "QIE_WAN_MIN_FRAMES",
    "QIE_WAN_MAX_FRAMES",
    "QIE_WAN_MAX_MP",
    "QIE_WAN_SCHEDULER",
    "QIE_WAN_FLUIDITY",
    "QIE_WAN_QUALITY",
    "QIE_WAN_RIFE_ENABLED",
    "QIE_WAN_EXPERT_HIGH",
    "QIE_WAN_EXPERT_LOW",
    "QIE_WAN_TRANSFORMER",
    "QIE_WAN_PIPELINE",
    "QIE_WAN_LORA",
    "QIE_WAN_LORA_SUBDIR",
    "QIE_WAN_LORA_ENABLED",
    "QIE_WAN_TIMEOUT",
    "QIE_WAN_NEGATIVE",
    "QIE_WAN_MODELS_ROOT",
    "QIE_WAN_GUIDANCE",
    "QIE_WAN_GUIDANCE_2",
)


def clean_env():
    """Drop every knob so a test sees the defaults even if the shell exported one."""
    for key in WAN_ENV:
        os.environ.pop(key, None)


class TestSettings(unittest.TestCase):
    def setUp(self):
        clean_env()

    def tearDown(self):
        clean_env()

    def test_defaults_are_the_48gb_pair(self):
        cfg = settings.load()
        self.assertEqual(cfg.dtype, "fp8")
        self.assertEqual(cfg.stack, "download")
        self.assertEqual(cfg.attn, "sdpa")
        self.assertEqual(cfg.steps, 8)
        self.assertEqual(cfg.guidance, 1.0)
        self.assertEqual(cfg.shift, 8.0)
        self.assertEqual(cfg.fps, 16)
        self.assertEqual(cfg.min_frames, 8)
        self.assertEqual(cfg.max_frames, 81)
        self.assertEqual(cfg.scheduler, "FlowMatchEulerDiscrete")
        self.assertEqual(cfg.fluidity, 1)
        self.assertEqual(cfg.quality, 8)
        self.assertTrue(cfg.rife_enabled)
        self.assertIn("最差质量", cfg.negative)

    def test_pipeline_and_expert_defaults(self):
        cfg = settings.load()
        self.assertEqual(cfg.pipeline, "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
        self.assertIn("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", cfg.expert_high)
        self.assertIn("wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", cfg.expert_high)
        self.assertIn("wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", cfg.expert_low)
        self.assertEqual(cfg.transformer_repo, "cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers")
        self.assertEqual(cfg.lora_repo, "lightx2v/Wan2.2-Lightning")
        self.assertTrue(cfg.lora_enabled)

    def test_env_overrides(self):
        os.environ["QIE_WAN_DTYPE"] = "bf16"
        os.environ["QIE_WAN_STEPS"] = "4"
        os.environ["QIE_WAN_SCHEDULER"] = "UniPCMultistep"
        os.environ["QIE_WAN_FLUIDITY"] = "4"
        os.environ["QIE_WAN_QUALITY"] = "10"
        os.environ["QIE_WAN_RIFE_ENABLED"] = "off"
        cfg = settings.load()
        self.assertEqual(cfg.dtype, "bf16")
        self.assertEqual(cfg.steps, 4)
        self.assertEqual(cfg.scheduler, "UniPCMultistep")
        self.assertEqual(cfg.fluidity, 4)
        self.assertEqual(cfg.quality, 10)
        self.assertFalse(cfg.rife_enabled)

    def test_bad_values_fall_back(self):
        os.environ["QIE_WAN_DTYPE"] = "int4"
        os.environ["QIE_WAN_STEPS"] = "four"
        os.environ["QIE_WAN_SCHEDULER"] = "NotAScheduler"
        os.environ["QIE_WAN_FLUIDITY"] = "7"
        cfg = settings.load()
        self.assertEqual(cfg.dtype, "fp8")
        self.assertEqual(cfg.steps, 8)
        self.assertEqual(cfg.scheduler, "FlowMatchEulerDiscrete")
        self.assertEqual(cfg.fluidity, 1)
        # The accessor must agree with the field even if a caller hand-builds a bad value.
        self.assertEqual(cfg.default_scheduler, "FlowMatchEulerDiscrete")
        self.assertEqual(cfg.default_fluidity, 1)

    def test_scheduler_case_insensitive(self):
        os.environ["QIE_WAN_SCHEDULER"] = "unipcMULTISTEP"
        self.assertEqual(settings.load().scheduler, "UniPCMultistep")

    def test_expert_spec_splitting(self):
        repo, subpath = settings.wan_expert_file("split_files/x.safetensors@owner/repo")
        self.assertEqual((repo, subpath), ("owner/repo", "split_files/x.safetensors"))
        self.assertEqual(settings.wan_expert_file("/models/x.safetensors"), ("", "/models/x.safetensors"))
        self.assertEqual(settings.wan_expert_file("owner/repo:sub/file.bin"), ("owner/repo", "sub/file.bin"))

    def test_fit_size_caps_megapixels_and_aligns(self):
        width, height = settings.fit_size(4000, 3000, max_mp=0.40)
        self.assertEqual(width % 16, 0)
        self.assertEqual(height % 16, 0)
        self.assertLessEqual(width * height / 1_000_000, 0.40 + 0.03)

    def test_public_has_no_secrets_and_lists_knobs(self):
        public = settings.load().public()
        for key in ("dtype", "pipeline", "scheduler", "fluidity", "quality", "fps", "negative"):
            self.assertIn(key, public)
        self.assertNotIn("token", str(public).lower())

    def test_every_scheduler_is_a_string_choice(self):
        self.assertGreaterEqual(len(settings.SCHEDULERS), 6)
        self.assertIn("FlowMatchEulerDiscrete", settings.SCHEDULERS)
        self.assertIn("UniPCMultistep", settings.SCHEDULERS)
        self.assertEqual(settings.DEFAULT_SCHEDULER, "FlowMatchEulerDiscrete")
        self.assertEqual(settings.SHIFT_KEY_SCHEDULERS, frozenset({"FlowMatchEulerDiscrete"}))

    def test_fluidity_choices(self):
        self.assertEqual(settings.FLUIDITY_CHOICES, (1, 2, 4, 8))


class TestRuntime(unittest.TestCase):
    def test_offload_auto(self):
        self.assertEqual(runtime.choose_offload("auto", 48.0), "model")
        self.assertEqual(runtime.choose_offload("auto", 96.0), "none")
        self.assertEqual(runtime.choose_offload("auto", 0.0), "model")

    def test_offload_explicit_wins(self):
        for mode in ("model", "sequential", "none"):
            self.assertEqual(runtime.choose_offload(mode, 96.0), mode)

    def test_plan_fp8_resident_fits_48gb_with_swap_spike(self):
        plan = runtime.wan_memory_plan("fp8", "none", 48.0)
        self.assertEqual(plan["offload"], "none")
        self.assertFalse(plan["encoder_offloaded"])
        # One resident expert + encoder, no offload.
        self.assertEqual(plan["expert_gb"], 14.0)
        self.assertEqual(plan["peak_gb"], round(14.0 + 11.4 + 1.4 + 0.5 + 6.0, 1))
        # The transient swap spike is reported but does not gate `fits`.
        self.assertEqual(plan["swap_spike_gb"], 14.0)
        self.assertEqual(plan["swap_peak_gb"], round(plan["peak_gb"] + 14.0, 1))
        self.assertTrue(plan["fits"])

    def test_offload_plan_reports_no_swap_spike(self):
        plan = runtime.wan_memory_plan("fp8", "model", 48.0)
        self.assertEqual(plan["swap_spike_gb"], 0.0)
        self.assertEqual(plan["swap_peak_gb"], plan["peak_gb"])

    def test_plan_fp8_auto_offloads_encoder_on_48gb(self):
        plan = runtime.wan_memory_plan("fp8", "auto", 48.0)
        self.assertEqual(plan["offload"], "model")
        self.assertTrue(plan["encoder_offloaded"])
        self.assertTrue(plan["fits"])

    def test_plan_bf16_resident_does_not_fit_48gb(self):
        self.assertFalse(runtime.wan_memory_plan("bf16", "none", 48.0)["fits"])
        self.assertTrue(runtime.wan_memory_plan("bf16", "none", 96.0)["fits"])

    def test_unknown_card_is_not_failed(self):
        self.assertTrue(runtime.wan_memory_plan("fp8", "auto", 0.0)["fits"])

    def test_budget_is_the_92_percent_line(self):
        plan = runtime.wan_memory_plan("fp8", "model", 48.0)
        self.assertEqual(plan["budget_gb"], round(48.0 * 0.92, 1))

    def test_gpu_info_returns_a_shape(self):
        info = runtime.gpu_info()
        for key in ("name", "total_gb", "free_gb", "source"):
            self.assertIn(key, info)

    def test_env_summary_reports_token_presence_without_leaking_it(self):
        os.environ["HF_TOKEN"] = "hf_notarealtoken"
        try:
            summary = runtime.env_summary()
            self.assertTrue(summary["HF_TOKEN"])
            self.assertNotIn("notarealtoken", str(summary))
        finally:
            os.environ.pop("HF_TOKEN", None)


class TestPlanHelpers(unittest.TestCase):
    def setUp(self):
        clean_env()

    def tearDown(self):
        clean_env()

    def test_frames_clamped(self):
        cfg = settings.load()
        self.assertEqual(wan.frames_for(0.1, cfg), cfg.min_frames)
        self.assertEqual(wan.frames_for(99.0, cfg), cfg.max_frames)
        self.assertEqual(wan.frames_for(2.0, cfg), 32)

    def test_frames_round_half_up(self):
        cfg = settings.load()
        self.assertEqual(wan.frames_for(1.0, cfg), 16)
        self.assertEqual(wan.frames_for(3.0, cfg), 48)

    def test_size_capped_area_and_aligned(self):
        cfg = settings.load()
        width, height = wan.pick_video_size(4000, 3000, cfg)
        self.assertEqual(width % 16, 0)
        self.assertEqual(height % 16, 0)
        self.assertLessEqual(width * height / 1_000_000, cfg.max_mp + 0.03)

    def test_expert_specs_default_to_download_pair(self):
        cfg = settings.load()
        self.assertTrue(wan.expert_fp8_ready(str(cfg.expert_high)))
        self.assertTrue(wan.expert_fp8_ready(str(cfg.expert_low)))

    def test_expert_fp8_ready_false_for_missing_local_path(self):
        self.assertFalse(wan.expert_fp8_ready("/nope/missing.safetensors"))

    def test_engine_stats_shape(self):
        stats = wan.WanEngine(settings.load()).stats()
        for key in ("loaded", "error", "dtype", "offload", "attn", "lora", "engine", "scheduler", "rife"):
            self.assertIn(key, stats)
        self.assertEqual(stats["engine"], wan.ENGINE_NAME)

    def test_engine_starts_unloaded_with_no_error(self):
        engine = wan.WanEngine(settings.load())
        self.assertFalse(engine.loaded)
        self.assertEqual(engine.error, "")

    def test_run_before_load_raises_with_the_error_string(self):
        engine = wan.WanEngine(settings.load())
        engine.error = "boom"
        with self.assertRaises(RuntimeError) as caught:
            engine.run("a.png", None, "prompt")
        self.assertIn("boom", str(caught.exception))

    def test_scheduler_names_match_settings(self):
        self.assertEqual(wan.scheduler_names(), settings.SCHEDULERS)

    def test_result_dataclass_defaults(self):
        result = wan.VideoResult(path="x.mp4", seed=1, width=624, height=832, frames=81, fps=16, ms=1000, peak_gb=20.0)
        self.assertEqual(result.scheduler, "")
        self.assertEqual(result.interpolated, "")
        self.assertEqual(result.quality, 0)


class TestRife(unittest.TestCase):
    def test_multiplier_one_is_a_noop(self):
        frames = [object(), object()]
        out, backend = blend_interpolate(frames, 1), "none"
        self.assertEqual(out[0], frames[0])
        self.assertEqual(len(out), 2)

    def test_blend_doubles_frame_count(self):
        out = blend_interpolate(_frames(3), 2)
        self.assertEqual(len(out), 5)
        self.assertEqual(out[0].getpixel((0, 0))[0], 0)
        self.assertEqual(out[-1].getpixel((0, 0))[0], 255)

    def test_blend_frame_count_is_n_times_m_minus_one_plus_one(self):
        # n source frames at multiplier m give n*(m-1)+1: the m-1 inserts land *between* the
        # n-1 gaps, and the endpoints are always kept.
        self.assertEqual(len(blend_interpolate(_frames(3), 2)), 5)
        self.assertEqual(len(blend_interpolate(_frames(3), 4)), 9)
        self.assertEqual(len(blend_interpolate(_frames(2), 4)), 5)
        self.assertEqual(len(blend_interpolate(_frames(2), 8)), 9)
        self.assertEqual(len(blend_interpolate(_frames(81), 2)), 161)

    def test_blend_endpoints_are_the_original_frames(self):
        source = _frames(3)
        out = blend_interpolate(source, 4)
        self.assertEqual(len(out), 9)
        self.assertEqual(out[0], source[0])
        self.assertEqual(out[-1], source[-1])

    def test_blend_works_on_pil_images(self):
        left = Image.new("RGB", (32, 32), (0, 0, 0))
        right = Image.new("RGB", (32, 32), (255, 255, 255))
        out = blend_interpolate([left, right], 2)
        self.assertEqual(len(out), 3)
        middle = out[1]
        self.assertEqual(middle.size, (32, 32))
        # Halfway between black and white; PIL rounds the channel down to 127.
        self.assertEqual(middle.getpixel((0, 0)), (127, 127, 127))

    def test_blend_works_on_float_arrays(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed (CI runs the CPU suite torch- and numpy-free)")

        left = np.zeros((8, 8, 3), dtype="float32")
        right = np.ones((8, 8, 3), dtype="float32")
        out = blend_interpolate([left, right], 2)
        self.assertEqual(len(out), 3)
        self.assertAlmostEqual(float(out[1].mean()), 0.5, places=5)

    def test_interpolate_noop_when_disabled(self):
        interpolator = RifeInterpolator(enabled=False)
        frames = _frames(2)
        out, backend = interpolator.interpolate(frames, 4)
        self.assertEqual(backend, "blend")
        self.assertEqual(len(out), 5)
        self.assertEqual(interpolator.error, "")
        self.assertFalse(interpolator.model)

    def test_interpolate_noop_for_short_input(self):
        interpolator = RifeInterpolator(enabled=False)
        frames = _frames(1)
        out, backend = interpolator.interpolate(frames, 4)
        self.assertEqual(backend, "none")
        self.assertEqual(len(out), 1)

    def test_stats_shape(self):
        stats = RifeInterpolator(enabled=False).stats()
        for key in ("enabled", "backend", "loaded", "error", "repo"):
            self.assertIn(key, stats)
        self.assertEqual(stats["repo"], "r3gm/RIFE")

    def test_support_dir_exists_with_warplayer(self):
        support = pathlib.Path(qie_engine.__file__).parent / "_rife_support"
        self.assertTrue((support / "model" / "warplayer.py").exists())
        self.assertTrue((support / "model" / "loss.py").exists())


class TestNoUndefinedNames(unittest.TestCase):
    """The engine never imports torch at module level, so this suite can import it - but a
    NameError in a method only shows up at render time on the box, so check statically too."""

    def _undefined(self, path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bound = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bound.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    bound.add(alias.asname or alias.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, ast.arg):
                bound.add(node.arg)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                bound.add(node.id)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                bound.update(node.names)
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        # `dir()` on the builtins *module* (not the dict that `unittest` exposes as
        # `__builtins__`), so this holds whether the suite or a plain interpreter loads it.
        known = bound | set(dir(builtins)) | {"__name__", "__file__", "__doc__", "__package__"}
        return sorted(name for name in used if name not in known)

    def test_no_undefined_names(self):
        for path in sorted((ROOT / "qie_engine").rglob("*.py")):
            with self.subTest(module=str(path.relative_to(ROOT))):
                self.assertEqual(self._undefined(path), [])


class TestTree(unittest.TestCase):
    def test_every_module_compiles(self):
        for path in sorted((ROOT / "qie_engine").rglob("*.py")):
            py_compile.compile(str(path), doraise=True)

    def test_package_exports_the_engine(self):
        self.assertTrue(hasattr(qie_engine, "WanEngine"))
        self.assertTrue(hasattr(qie_engine, "WanSettings"))
        self.assertEqual(qie_engine.__version__, "0.1.0")

    def test_no_hardcoded_token(self):
        import re

        for path in (ROOT / "qie_engine" / "settings.py", ROOT / "qie_engine" / "wan.py"):
            self.assertIsNone(re.search(r"hf_[A-Za-z0-9]{30,}", path.read_text(encoding="utf-8")))

    def test_pyproject_exists(self):
        self.assertTrue((ROOT / "pyproject.toml").exists())


if __name__ == "__main__":
    unittest.main()
