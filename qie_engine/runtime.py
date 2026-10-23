"""Where am I running, how big is the card, and how much of Wan fits on it?

Wan-specific only: stills apps keep their own memory planner. `gpu_info` prefers torch (it
knows the real total) and falls back to `nvidia-smi` on a box where torch never imported.
"""

from __future__ import annotations

import os
import shutil
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

# Host RAM the load step peaks at: BOTH experts live while the text encoder materialises
# (2 x 14 / 2 x 28 GB) plus encoder + VAE + CLIP + the state-dict transient. Below these the
# Linux OOM killer would SIGKILL the process mid-build - with no traceback to explain it.
WAN_LOAD_RAM_FP8_GB = 52.0
WAN_LOAD_RAM_BF16_GB = 80.0

# Free disk a download needs before it starts; failing fast beats dying at 90%.
WAN_DISK_DOWNLOAD_GB = 15.0
WAN_DISK_EXPERT_GB = 16.0


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


def disk_free_gb(path: str = "") -> float:
    """Free space (GB) on the filesystem holding `path`, or -1.0 when it cannot be measured.

    Walks up to the nearest existing ancestor: `HF_HOME=/data/hf` legitimately does not exist
    yet on first boot, but `/data` does.
    """
    target = os.path.abspath(path or ".")
    while not os.path.exists(target):
        parent = os.path.dirname(target)
        if parent == target:
            break
        target = parent
    try:
        return shutil.disk_usage(target).free / 2**30
    except OSError:
        return -1.0


def disk_guard(path: str, need_gb: float, what: str) -> None:
    """Refuse to start a download that cannot fit. Raises OSError on purpose: it is the one
    exception class the expert loader must never swallow into its bf16 fallback."""
    free = disk_free_gb(path)
    if free < 0 or free >= need_gb:
        return
    raise OSError(
        f"not enough disk for {what}: need ~{need_gb:.0f} GB free below "
        f"{os.path.abspath(path or '.')}, have {free:.1f} GB ({disk_hint()})"
    )


def parse_meminfo(text: str) -> tuple[float, float]:
    """(total_gb, available_gb) parsed out of /proc/meminfo text; missing keys become 0."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            values[key] = int(parts[0]) / (1024 * 1024)  # kB -> GB
    available = values.get("MemAvailable", values.get("MemFree", 0.0))
    return values.get("MemTotal", 0.0), available


def host_memory() -> tuple[float, float]:
    """(total_gb, available_gb) from /proc/meminfo; (0.0, 0.0) where it is not readable
    (Windows/macOS dev boxes), which makes `ram_preflight` pass rather than guess."""
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            return parse_meminfo(handle.read())
    except OSError:
        return 0.0, 0.0


def rss_gb() -> float:
    """Resident set size of this process in GB; 0.0 where /proc is unavailable."""
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)  # kB -> GB
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def ram_preflight(dtype: str, total_gb: float, avail_gb: float) -> str | None:
    """Error string when host RAM cannot survive `WanEngine.load()`; None when it looks fine.

    This is the check that turns "process vanished mid-boot with no traceback" into a boot
    message a renter can read.
    """
    if not total_gb or not avail_gb:
        return None  # unknown platform - say nothing rather than block a working box
    need = WAN_LOAD_RAM_FP8_GB if dtype == "fp8" else WAN_LOAD_RAM_BF16_GB
    if avail_gb >= need:
        return None
    return (
        f"host RAM too low for a {dtype} load: {avail_gb:.0f} GB available, "
        f"the build peaks near {need:.0f} GB - rent a box with more RAM; refusing to start "
        f"rather than being killed by the OOM killer"
    )


def env_summary() -> dict:
    """The knobs that decide what the engine will build; handy in /status."""
    return {
        "HF_HOME": os.environ.get("HF_HOME", ""),
        "HF_TOKEN": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "QIE_WAN_STACK": os.environ.get("QIE_WAN_STACK", "download"),
    }
