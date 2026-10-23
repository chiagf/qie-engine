"""qie-engine: a portable Wan2.2 I2V engine shared by the qie apps.

    from qie_engine import WanEngine, WanSettings
    from qie_engine.settings import load as load_wan_settings

Nothing here imports an app module, and nothing imports torch at module level, so the CPU test
suite can exercise every helper without a GPU.
"""

from qie_engine.rife import RifeInterpolator, blend_interpolate
from qie_engine.runtime import choose_offload, gpu_info, wan_memory_plan
from qie_engine.settings import (
    FLUIDITY_CHOICES,
    SCHEDULERS,
    WanSettings,
    fit_size,
    wan_expert_file,
)
from qie_engine.wan import WanEngine, VideoResult, expert_fp8_ready, frames_for, pick_video_size

__all__ = [
    "FLUIDITY_CHOICES",
    "SCHEDULERS",
    "RifeInterpolator",
    "VideoResult",
    "WanEngine",
    "WanSettings",
    "blend_interpolate",
    "choose_offload",
    "expert_fp8_ready",
    "fit_size",
    "frames_for",
    "gpu_info",
    "pick_video_size",
    "wan_expert_file",
    "wan_memory_plan",
]

__version__ = "0.1.0"
