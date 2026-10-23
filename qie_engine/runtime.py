"""Where am I running, how big is the card, and how much of Wan fits on it?

Wan-specific only: stills apps keep their own memory planner. `gpu_info` prefers torch (it
knows the real total) and falls back to `nvidia-smi` on a box where torch never imported.
"""

from __future__ import annotations

import os
import subprocess

# Wan2.2 I2V-A14B: umt5-xxl text encoder, wan VAE, CLIP-vision. `WAN_EXPERT_FP8_GB` is the one
# expert resident on the card at a time - the pipeline keeps the other on CPU and swaps it in
# mid-run, so the denoise peak is a single expert, not the pair.
#
# fp8 assumes the Comfy fp8-scaled expert checkpoints; bf16 assumes the diffusers repo.
WAN_EXPERT_FP8_GB = 14.0
WAN_EXPERT_BF16_GB = 28.0
# The swap briefly holds both experts while the incoming one materialises; this is the true
# worst case on a card that is not offloading.
WAN_SWAP_SPIKE_GB = 14.0
WAN_TEXT_ENCODER_GB = 11.4
WAN_CLIP_VISION_GB = 1.4
WAN_VAE_GB = 0.5
WAN_ACTIVATION_GB = 6.0

# The card keeps this much headroom; activations and the expert swap both spike past the
# resident weight total.
WAN_BUDGET_RATIO = 0.92

# Below this the auto plan stops trying to keep the text encoder resident.
WAN_AUTO_RESIDENT_GB = 64.0


def gpu_info() -> dict:
    """(name, total_gb, free_gb) for the first CUDA device, or a CPU placeholder."""
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return {
                "name": torch.cuda.get_device_properties(0).name,
                "total_gb": round(total / 2**30, 1),
                "free_gb": round(free / 2**30, 1),
                "source": "torch",
            }
    except Exception:  # noqa: BLE001 - torch missing or no driver; fall through to nvidia-smi
        pass
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        if out:
            name, total, free = [part.strip() for part in out.splitlines()[0].split(",")]
            return {
                "name": name,
                "total_gb": round(int(total) / 1024, 1),
                "free_gb": round(int(free) / 1024, 1),
                "source": "nvidia-smi",
            }
    except Exception:  # noqa: BLE001
        pass
    return {"name": "cpu", "total_gb": 0.0, "free_gb": 0.0, "source": "none"}


def choose_offload(requested: str, total_gb: float) -> str:
    """`auto` = keep everything resident on a big card, offload the encoder on a small one."""
    if requested in {"model", "sequential", "none"}:
        return requested
    if not total_gb:
        return "model"
    return "none" if total_gb >= WAN_AUTO_RESIDENT_GB else "model"


def wan_memory_plan(dtype_mode: str, offload: str, total_gb: float) -> dict:
    """Weights resident at the same time; the text encoder is the only optional one."""
    offload = choose_offload(offload, total_gb)
    resident_experts = WAN_EXPERT_FP8_GB if dtype_mode == "fp8" else WAN_EXPERT_BF16_GB
    encoder = WAN_TEXT_ENCODER_GB if offload == "none" else 0.0
    base = resident_experts + WAN_CLIP_VISION_GB + WAN_VAE_GB + WAN_ACTIVATION_GB
    # The swap spike is transient (the outgoing expert is freed as the incoming one lands), so
    # it is reported for visibility but never decides `fits` - the denoise peak does.
    denoise_peak = base + encoder
    swap_peak = denoise_peak + (WAN_SWAP_SPIKE_GB if offload == "none" else 0.0)
    fits = not total_gb or denoise_peak <= total_gb * WAN_BUDGET_RATIO
    return {
        "expert_gb": resident_experts,
        "swap_spike_gb": WAN_SWAP_SPIKE_GB if offload == "none" else 0.0,
        "dtype": dtype_mode,
        "offload": offload,
        "encoder_offloaded": offload != "none",
        "text_encoder_gb": WAN_TEXT_ENCODER_GB,
        "clip_vision_gb": WAN_CLIP_VISION_GB,
        "vae_gb": WAN_VAE_GB,
        "activations_gb": WAN_ACTIVATION_GB,
        "peak_gb": round(denoise_peak, 1),
        "swap_peak_gb": round(swap_peak, 1),
        "budget_gb": round(total_gb * WAN_BUDGET_RATIO, 1) if total_gb else None,
        "fits": fits,
    }


def disk_hint(total_gb: float = 0.0) -> str:
    """Rough instance-disk advice: weights download is ~45 GB on first boot."""
    need = 60.0
    return f"~{need:.0f} GB of instance disk recommended (weights ~45 GB + image + clips)"


def env_summary() -> dict:
    """The knobs that decide what the engine will build; handy in /status."""
    return {
        "HF_HOME": os.environ.get("HF_HOME", ""),
        "HF_TOKEN": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "QIE_WAN_STACK": os.environ.get("QIE_WAN_STACK", "download"),
    }
