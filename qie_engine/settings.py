"""Wan2.2 settings: the knobs every Wan app agrees on, read from the environment.

An app owns its own `Config` and hands a `WanSettings` to the engine, so the engine never
imports an app module. Env names keep the `QIE_WAN_*` spelling used by qie-studio, so a box
configured for one app configures the other.

`public()` is what a `/config` route prints: it never contains a token.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

ALIGN = 16
MAX_STEPS = 30
MAX_SEED = 2**31 - 1

# Wan2.2 I2V-A14B: the Chinese negative prompt every Wan Space ships with.
WAN_NEGATIVE = (
    "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, "
    "低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, "
    "毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走"
)

# 160 is Wan2.2's own ceiling; 81 keeps a 48 GB card clear of the activation spike.
WAN_MIN_FRAMES = 8
WAN_MAX_FRAMES = 81
WAN_MAX_FRAMES_LONG = 160
WAN_FPS = 16

WAN_DTYPES = ("fp8", "bf16")
ATTN_MODES = ("sdpa", "flash")
OFFLOAD_MODES = ("auto", "model", "sequential", "none")

# The schedulers diffusers ships for Wan2.2. FlowMatchEulerDiscrete reads `shift`, the rest
# read `flow_shift`, so `apply_scheduler` has to know which key to write.
SCHEDULERS = (
    "FlowMatchEulerDiscrete",
    "UniPCMultistep",
    "DEISMultistep",
    "DPMSolverMultistep",
    "DPMSolverSinglestep",
    "DPMSolverMultistepInverse",
    "SASolver",
)
SHIFT_KEY_SCHEDULERS = frozenset({"FlowMatchEulerDiscrete"})
DEFAULT_SCHEDULER = "FlowMatchEulerDiscrete"

# Fluidity multipliers: the RIFE pass generates (multiplier - 1) frames between each pair, so
# the export fps becomes WAN_FPS * multiplier. 1 means "no interpolation".
FLUIDITY_CHOICES = (1, 2, 4, 8)
DEFAULT_FLUIDITY = 1

# A ~22 MB zip holding train_log/ (RIFE_HDv3, IFNet_HDv3, flownet.pkl). The `model/` support
# code it imports ships with qie-engine in `_rife_support`.
RIFE_REPO = "r3gm/RIFE"
RIFE_ZIP = "RIFEv4.26_0921.zip"

PIPELINE_DEFAULT = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
EXPERT_HIGH_DEFAULT = (
    "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
    "@Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
)
EXPERT_LOW_DEFAULT = (
    "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
    "@Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
)
TRANSFORMER_REPO_DEFAULT = "cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers"


def wan_expert_file(value: str) -> tuple[str, str]:
    """Split `subpath@repo` (or `repo:path`, or a plain path on disk) into (repo, subpath)."""
    if "@" in value and "://" not in value:
        subpath, repo = value.rsplit("@", 1)
        return repo, subpath
    if ":" in value and "://" not in value:
        repo, subpath = value.split(":", 1)
        return repo, subpath
    return "", value


def fit_size(width: int, height: int, max_mp: float = 2.0, align: int = ALIGN) -> tuple[int, int]:
    """Shrink (width, height) to at most `max_mp` megapixels, then align both sides down."""
    width = max(align, int(width))
    height = max(align, int(height))
    mp = (width * height) / 1_000_000
    if mp > max_mp:
        scale = (max_mp / mp) ** 0.5
        width = max(align, int(width * scale))
        height = max(align, int(height * scale))
    width = max(align, int(width // align) * align)
    height = max(align, int(height // align) * align)
    return width, height


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def _int(name: str, default: int, allowed: tuple[int, ...] | None = None) -> int:
    try:
        value = int(_env(name, str(default)))
    except ValueError:
        return default
    return value if allowed is None or value in allowed else default


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    return _env(name, "on" if default else "off").lower() in {"1", "on", "true", "yes"}


def _pick(name: str, default: str, allowed: tuple[str, ...]) -> str:
    """Case-insensitive pick that returns the *canonical* spelling, not the user's casing."""
    value = _env(name, default)
    for item in allowed:
        if value.lower() == item.lower():
            return item
    return default


@dataclass(frozen=True)
class WanSettings:
    pipeline: str
    expert_high: str
    expert_low: str
    transformer_repo: str
    dtype: str
    stack: str
    offload: str
    attn: str
    lora_repo: str
    lora_subdir: str
    lora_enabled: bool
    steps: int
    guidance: float
    guidance_2: float
    shift: float
    max_mp: float
    fps: int
    min_frames: int
    max_frames: int
    negative: str
    timeout: int
    scheduler: str
    fluidity: int
    quality: int
    rife_enabled: bool
    hf_home: str

    @property
    def default_scheduler(self) -> str:
        return self.scheduler if self.scheduler in SCHEDULERS else DEFAULT_SCHEDULER

    @property
    def default_fluidity(self) -> int:
        return self.fluidity if self.fluidity in FLUIDITY_CHOICES else DEFAULT_FLUIDITY

    def public(self) -> dict:
        return asdict(self)


def load() -> WanSettings:
    """Build settings from `QIE_WAN_*` (and the shared `QIE_*` names) in the environment."""
    stack = _env("QIE_WAN_STACK", "download").lower()
    baked = stack == "baked"
    root = _env("QIE_WAN_MODELS_ROOT", "/models")
    repo = _env("QIE_WAN_PIPELINE", PIPELINE_DEFAULT)
    baked_pipeline = os.path.join(root, "pipeline")
    # A baked image ships only the fp8 experts + LoRA; the umt5/VAE/CLIP half stays a download,
    # so `baked` uses /models/pipeline only when the image actually carried it.
    pipeline = (
        baked_pipeline
        if baked and os.path.exists(os.path.join(baked_pipeline, "model_index.json"))
        else repo
    )
    return WanSettings(
        pipeline=pipeline,
        expert_high=(
            os.path.join(root, "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors")
            if baked
            else _env("QIE_WAN_EXPERT_HIGH", EXPERT_HIGH_DEFAULT)
        ),
        expert_low=(
            os.path.join(root, "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors")
            if baked
            else _env("QIE_WAN_EXPERT_LOW", EXPERT_LOW_DEFAULT)
        ),
        transformer_repo=_env("QIE_WAN_TRANSFORMER", TRANSFORMER_REPO_DEFAULT),
        dtype=_pick("QIE_WAN_DTYPE", "fp8", WAN_DTYPES),
        stack=stack,
        offload=_pick("QIE_WAN_OFFLOAD", "auto", OFFLOAD_MODES),
        attn=_pick("QIE_ATTN", "sdpa", ATTN_MODES),
        lora_repo=_env("QIE_WAN_LORA", "lightx2v/Wan2.2-Lightning"),
        lora_subdir=_env("QIE_WAN_LORA_SUBDIR", "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1"),
        lora_enabled=_bool("QIE_WAN_LORA_ENABLED", True),
        steps=_int("QIE_WAN_STEPS", 8),
        guidance=_float("QIE_WAN_GUIDANCE", 1.0),
        guidance_2=_float("QIE_WAN_GUIDANCE_2", 1.0),
        shift=_float("QIE_WAN_SHIFT", 8.0),
        max_mp=_float("QIE_WAN_MAX_MP", 0.40),
        fps=_int("QIE_WAN_FPS", WAN_FPS),
        min_frames=_int("QIE_WAN_MIN_FRAMES", WAN_MIN_FRAMES),
        max_frames=_int("QIE_WAN_MAX_FRAMES", WAN_MAX_FRAMES),
        negative=_env("QIE_WAN_NEGATIVE", WAN_NEGATIVE),
        timeout=_int("QIE_WAN_TIMEOUT", 600),
        scheduler=_pick("QIE_WAN_SCHEDULER", DEFAULT_SCHEDULER, SCHEDULERS),
        fluidity=_int("QIE_WAN_FLUIDITY", DEFAULT_FLUIDITY, allowed=FLUIDITY_CHOICES),
        quality=_int("QIE_WAN_QUALITY", 8, allowed=tuple(range(1, 11))),
        rife_enabled=_bool("QIE_WAN_RIFE_ENABLED", True),
        hf_home=_env("HF_HOME", ""),
    )
