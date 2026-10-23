"""Wan2.2 video engine: first/last frame in, an mp4 out.

Import-light on purpose - no torch, diffusers or PIL at module level - so a CPU test suite can
exercise every helper. Two backends:

    "fp8"    the canonical Comfy fp8-scaled expert checkpoints via `from_single_file`
    "bf16"   the official diffusers repo, torchao-requantized as the automatic fallback

The engine depends only on `qie_engine.settings.WanSettings`; it never imports an app module.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from qie_engine.rife import RifeInterpolator
from qie_engine.runtime import choose_offload, gpu_info
from qie_engine.settings import (
    MAX_SEED,
    MAX_STEPS,
    SCHEDULERS,
    SHIFT_KEY_SCHEDULERS,
    WanSettings,
    fit_size,
    wan_expert_file,
)

ENGINE_NAME = "wan2.2-i2v-a14b"


@dataclass
class VideoResult:
    path: str
    seed: int
    width: int
    height: int
    frames: int
    fps: int
    ms: int
    peak_gb: float
    scheduler: str = ""
    interpolated: str = ""
    quality: int = 0


def frames_for(duration: float, cfg: WanSettings) -> int:
    """Map a duration in seconds (>= 0.5) onto clamped model frames at the pipeline fps."""
    fps = int(cfg.fps)
    low = int(cfg.min_frames)
    high = int(cfg.max_frames)
    dur = max(0.5, float(duration))
    return max(low, min(high, int(round(dur * fps))))


def pick_video_size(width: int, height: int, cfg: WanSettings) -> tuple[int, int]:
    """Cap a source frame at the wan_max_mp area; 16-aligned to match the VAE stride."""
    return fit_size(width, height, max_mp=cfg.max_mp, align=16)


def expert_fp8_ready(spec: str) -> bool:
    """True when the spec is a local fp8 file, or a `subpath@repo` pair that can be fetched."""
    if os.path.exists(spec):
        return True
    repo, subpath = wan_expert_file(spec)
    return bool(repo and subpath)


def scheduler_names() -> tuple[str, ...]:
    return SCHEDULERS


def apply_scheduler(pipe, name: str, flow_shift: float) -> str:
    """Swap the pipeline's scheduler, carrying the base config over. Returns the active name.

    FlowMatchEulerDiscrete names its shift knob `shift`; every other Wan scheduler names it
    `flow_shift`. Writing the wrong key is silent - the pipeline just runs at the default shift.
    """
    from diffusers import (
        DEISMultistepScheduler,
        DPMSolverMultistepInverseScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler,
        FlowMatchEulerDiscreteScheduler,
        SASolverScheduler,
        UniPCMultistepScheduler,
    )

    classes = {
        "FlowMatchEulerDiscrete": FlowMatchEulerDiscreteScheduler,
        "UniPCMultistep": UniPCMultistepScheduler,
        "DEISMultistep": DEISMultistepScheduler,
        "DPMSolverMultistep": DPMSolverMultistepScheduler,
        "DPMSolverSinglestep": DPMSolverSinglestepScheduler,
        "DPMSolverMultistepInverse": DPMSolverMultistepInverseScheduler,
        "SASolver": SASolverScheduler,
    }
    import copy

    key = "shift" if name in SHIFT_KEY_SCHEDULERS else "flow_shift"
    config = copy.deepcopy(pipe.scheduler.config)
    config[key] = float(flow_shift)
    pipe.scheduler = classes[name].from_config(config)
    return name


class WanEngine:
    """Same surface every app expects: `loaded`, `error`, `offload`, `load()`, `run()`, `stats()`."""

    def __init__(self, cfg: WanSettings):
        self.cfg = cfg
        self.pipe = None
        self.error = ""
        self.loaded = False
        self.offload = ""
        self.attn = ""
        self.lora = ""
        self.dtype = ""
        self.scheduler = ""
        self.rife = RifeInterpolator(cache_dir=cfg.hf_home, enabled=cfg.rife_enabled)

    def _torch(self):
        import torch

        return torch

    def load(self) -> str | None:
        """Build the pipeline. Returns an error string instead of raising."""
        if self.loaded:
            return None
        cfg = self.cfg
        try:
            torch = self._torch()
            from diffusers import WanImageToVideoPipeline

            total = gpu_info().get("total_gb", 0.0)
            self.offload = choose_offload(cfg.offload or "auto", total)
            self.dtype = cfg.dtype

            high, low = self._experts(torch)
            pipe = WanImageToVideoPipeline.from_pretrained(
                cfg.pipeline,
                transformer=high,
                transformer_2=low,
                torch_dtype=torch.bfloat16,
            )
            self.scheduler = apply_scheduler(pipe, cfg.default_scheduler, cfg.shift)
            self._attention(pipe)
            self._lora(pipe)
            if cfg.dtype == "fp8":
                self._requantize(pipe)
            if self.offload in {"model", "sequential"}:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
            self.pipe = pipe
            self.loaded = True
            return None
        except Exception as error:  # noqa: BLE001 - the app shows this string, never a traceback
            self.error = f"{type(error).__name__}: {error}"
            return self.error

    def _experts(self, torch):
        """fp8 single-file pair first (baked path or `subpath@repo`); bf16 repos as fallback."""
        from diffusers.models import WanTransformer3DModel

        cfg = self.cfg
        high_spec, low_spec = str(cfg.expert_high), str(cfg.expert_low)
        fp8_specs = cfg.dtype == "fp8" and (
            expert_fp8_ready(high_spec) and expert_fp8_ready(low_spec)
        )
        if fp8_specs:
            try:
                return (
                    WanTransformer3DModel.from_single_file(self._local(high_spec)),
                    WanTransformer3DModel.from_single_file(self._local(low_spec)),
                )
            except Exception as error:  # noqa: BLE001 - old diffusers or unsupported format
                print(f"[wan] fp8 single-file experts failed ({error}); using bf16 + torchao")
        return (
            WanTransformer3DModel.from_pretrained(
                cfg.transformer_repo, subfolder="transformer", dtype=torch.bfloat16
            ),
            WanTransformer3DModel.from_pretrained(
                cfg.transformer_repo, subfolder="transformer_2", dtype=torch.bfloat16
            ),
        )

    def _local(self, spec: str) -> str:
        """A plain path wins; otherwise `subpath@repo` resolves the file through hf_hub."""
        if os.path.exists(spec):
            return spec
        repo, subpath = wan_expert_file(spec)
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=repo, filename=subpath.strip("/"))

    def _attention(self, pipe) -> None:
        if self.cfg.attn != "flash":
            self.attn = "sdpa"
            return
        try:
            pipe.transformer.set_attention_backend("flash_attention_2")
            pipe.transformer_2.set_attention_backend("flash_attention_2")
            self.attn = "flash"
        except Exception as error:  # noqa: BLE001 - no wheel or unsupported arch: degrade
            print(f"[wan] flash attention unavailable ({error}); using sdpa")
            self.attn = "sdpa"

    def _lora(self, pipe) -> None:
        """Wan2.2 has two experts, so the Lightning pair loads as two adapters (high ->
        transformer, low -> transformer_2). Best-effort: speed only, never a blocker."""
        cfg = self.cfg
        if not cfg.lora_enabled or not cfg.lora_repo:
            return
        sub = cfg.lora_subdir or None
        loaded: list[str] = []
        try:
            pipe.load_lora_weights(
                cfg.lora_repo,
                subfolder=sub,
                weight_name="high_noise_model.safetensors",
                adapter_name="lightning_high",
            )
            loaded.append("lightning_high")
        except Exception as error:  # noqa: BLE001
            print(f"[wan] high-noise Lightning LoRA not loaded ({error})")
        if getattr(pipe, "transformer_2", None) is not None:
            try:
                pipe.load_lora_weights(
                    cfg.lora_repo,
                    subfolder=sub,
                    weight_name="low_noise_model.safetensors",
                    adapter_name="lightning_low",
                    load_into_transformer_2=True,
                )
                loaded.append("lightning_low")
            except Exception as error:  # noqa: BLE001
                print(f"[wan] low-noise Lightning LoRA not loaded ({error})")
        if not loaded:
            return
        try:
            pipe.set_adapters(loaded, adapter_weights=[1.0] * len(loaded))
        except Exception as error:  # noqa: BLE001 - adapters stay active at their default weight
            print(f"[wan] set_adapters left at defaults ({error})")
        self.lora = f"{cfg.lora_repo} [{' + '.join(loaded)}]"

    def _requantize(self, pipe) -> None:
        """torchao fp8-dynamic on the bf16 fallback; on experts already fp8 it is simply cheap."""
        try:
            from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_

            quantize_(pipe.text_encoder.to("cpu"), Float8DynamicActivationFloat8WeightConfig())
            quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
            quantize_(pipe.transformer_2, Float8DynamicActivationFloat8WeightConfig())
        except Exception as error:  # noqa: BLE001 - bf16 weights still run, they just use more
            print(f"[wan] torchao fp8 quantization skipped ({error})")

    def stats(self) -> dict:
        return {
            "loaded": self.loaded,
            "error": self.error,
            "dtype": self.dtype,
            "offload": self.offload,
            "attn": self.attn,
            "lora": self.lora,
            "pipeline": self.cfg.pipeline,
            "engine": ENGINE_NAME,
            "scheduler": self.scheduler,
            "rife": self.rife.stats(),
        }

    def run(
        self,
        first: str,
        last: str | None,
        prompt: str,
        *,
        duration: float = 4.0,
        steps: int | None = None,
        seed: int = 0,
        guidance: float | None = None,
        guidance_2: float | None = None,
        negative: str | None = None,
        scheduler: str | None = None,
        flow_shift: float | None = None,
        fluidity: int | None = None,
        quality: int | None = None,
        progress=None,
        out_dir: str = ".",
    ) -> VideoResult:
        if not self.loaded:
            raise RuntimeError(self.error or "video model not loaded")
        torch = self._torch()
        cfg = self.cfg
        from PIL import Image

        from diffusers.utils import export_to_video

        first_image = Image.open(first).convert("RGB")
        width, height = pick_video_size(*first_image.size, cfg)
        first_image = first_image.resize((width, height), Image.LANCZOS)
        last_image = None
        if last:
            source = Image.open(last).convert("RGB")
            ratio = max(width / source.width, height / source.height)
            crop = source.resize(
                (round(source.width * ratio), round(source.height * ratio)), Image.LANCZOS
            )
            left = max(0, (crop.width - width) // 2)
            top = max(0, (crop.height - height) // 2)
            last_image = crop.crop((left, top, left + width, top + height))

        steps = max(1, min(MAX_STEPS, int(steps or cfg.steps)))
        guidance = cfg.guidance if guidance is None else float(guidance)
        guidance_2 = cfg.guidance_2 if guidance_2 is None else float(guidance_2)
        seed = int(seed) % MAX_SEED
        frames = frames_for(duration, cfg)
        fluidity = cfg.default_fluidity if fluidity is None else int(fluidity)
        quality = cfg.quality if quality is None else int(quality)
        name = scheduler or cfg.default_scheduler
        shift = cfg.shift if flow_shift is None else float(flow_shift)
        if name != self.scheduler or shift != float(cfg.shift):
            apply_scheduler(self.pipe, name, shift)
            self.scheduler = name

        out_dir = os.path.join(out_dir, time.strftime("%Y%m%d-%H%M%S") + f"-v{seed}")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "clip.mp4")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.time()
        output = self.pipe(
            image=first_image,
            last_image=last_image,
            prompt=prompt,
            negative_prompt=(negative or cfg.negative or " "),
            height=height,
            width=width,
            num_frames=frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            guidance_scale_2=guidance_2,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            callback_on_step_end=self._callback(progress, steps),
        ).frames[0]

        # RIFE raises the export fps rather than the model frame count, so the clip is the same
        # motion rendered at 2/4/8x the sampling rate.
        fps = int(cfg.fps)
        interpolated = ""
        if fluidity > 1 and len(output) > 1:
            output, interpolated = self.rife.interpolate(output, fluidity)
            fps = int(cfg.fps) * int(fluidity)

        export_to_video(output, path, fps=fps, quality=quality)
        peak = (
            round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if torch.cuda.is_available()
            else 0.0
        )
        return VideoResult(
            path=path,
            seed=seed,
            width=width,
            height=height,
            frames=len(output),
            fps=fps,
            ms=int((time.time() - started) * 1000),
            peak_gb=peak,
            scheduler=self.scheduler,
            interpolated=interpolated,
            quality=quality,
        )

    @staticmethod
    def _callback(progress, total):
        if progress is None:
            return None

        def callback(_pipe, step, _timestep, callback_kwargs):
            try:
                progress((step + 1) / total, desc=f"step {step + 1}/{total}")
            except Exception:  # noqa: BLE001 - progress must never break a render
                pass
            return callback_kwargs

        return callback
