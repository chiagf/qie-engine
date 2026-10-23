"""RIFE frame interpolation: make a 16 fps Wan clip look like 32/64/128 fps.

Two backends, picked at runtime:

    "rife"  - the real optical-flow model (r3gm/RIFE v4.26), weights fetched once and cached.
              Needs torch + CUDA; the `model/` support code it imports ships in `_rife_support`.
    "blend" - linear cross-fade between neighbours. No model, no GPU, always available.

`interpolate()` returns the frames plus the backend name it used, so the UI can say which one
produced the clip instead of silently shipping the cheaper option.
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

from qie_engine.settings import RIFE_REPO, RIFE_ZIP

SUPPORT_DIR = Path(__file__).resolve().parent / "_rife_support"


def blend_interpolate(frames, multiplier: int) -> list:
    """Cross-fade (multiplier - 1) frames between each pair. Works on PIL images or ndarrays."""
    if multiplier < 2 or len(frames) < 2:
        return list(frames)
    steps = int(multiplier) - 1
    out = []
    for index in range(len(frames) - 1):
        left, right = frames[index], frames[index + 1]
        out.append(left)
        for step in range(1, steps + 1):
            out.append(_blend(left, right, step / (steps + 1)))
    out.append(frames[-1])
    return out


def _blend(left, right, alpha: float):
    if hasattr(left, "point") and hasattr(right, "point"):
        from PIL import Image

        return Image.blend(left.convert("RGB"), right.convert("RGB"), alpha)
    import numpy as np

    a = np.asarray(left, dtype="float32")
    b = np.asarray(right, dtype="float32")
    return (a * (1.0 - alpha) + b * alpha).astype("float32")


def _to_pil(frame):
    """diffusers hands back PIL images, RIFE hands back ndarrays - normalise on the way out."""
    if hasattr(frame, "convert"):
        return frame
    import numpy as np
    from PIL import Image

    array = np.asarray(frame)
    if array.dtype != "uint8":
        array = (np.clip(array, 0.0, 1.0) * 255).round().astype("uint8")
    return Image.fromarray(array)


def _to_array(frame):
    import numpy as np

    array = np.asarray(frame.convert("RGB") if hasattr(frame, "convert") else frame, dtype="float32")
    if array.max() > 1.5:
        array = array / 255.0
    return np.ascontiguousarray(array)


class RifeInterpolator:
    """Lazily downloads and loads RIFE; degrades to `blend` rather than raising."""

    def __init__(self, cache_dir: str = "", enabled: bool = True):
        self.enabled = bool(enabled)
        self.cache = Path(cache_dir or os.environ.get("HF_HOME") or "") / "rife"
        self.model = None
        self.error = ""
        self.backend = "blend"

    def ensure(self) -> str:
        """Download + load on first use. Returns the active backend name ("rife" or "blend")."""
        if self.backend == "rife":
            return self.backend
        if not self.enabled:
            return self.backend
        try:
            self.model = self._load()
            self.backend = "rife"
            self.error = ""
        except Exception as error:  # noqa: BLE001 - a missing model must never block a render
            self.error = f"{type(error).__name__}: {error}"
            self.backend = "blend"
            print(f"[rife] unavailable ({self.error}); using blend interpolation")
        return self.backend

    def _root(self) -> Path:
        """Extract the zip once, then reuse the directory."""
        root = self.cache / "extracted"
        marker = root / "train_log" / "RIFE_HDv3.py"
        if marker.exists():
            return root
        self.cache.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import hf_hub_download

        archive = hf_hub_download(repo_id=RIFE_REPO, filename=RIFE_ZIP)
        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            for member in handle.namelist():
                # __MACOSX/ and .DS_Store entries are junk the zip ships with.
                if member.startswith("__MACOSX") or member.endswith(".DS_Store"):
                    continue
                handle.extract(member, root)
        if not marker.exists():
            raise RuntimeError(f"RIFE archive layout unexpected: no {marker.name} in {RIFE_REPO}")
        return root

    def _load(self):
        import torch

        root = str(self._root())
        for path in (root, str(SUPPORT_DIR)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from train_log.RIFE_HDv3 import Model

        model = Model()
        model.load_model(os.path.join(root, "train_log"), -1)
        model.eval()
        if torch.cuda.is_available():
            model.device()
            model.flownet = model.flownet.half()
        return model

    def interpolate(self, frames, multiplier: int) -> tuple[list, str]:
        """Return (frames, backend). Multiplier 1 (or fewer than 2 frames) is a no-op."""
        if multiplier < 2 or len(frames) < 2:
            return list(frames), "none"
        backend = self.ensure()
        if backend != "rife":
            return blend_interpolate(frames, multiplier), backend
        try:
            return self._rife(frames, multiplier), backend
        except Exception as error:  # noqa: BLE001 - fall back rather than lose the clip
            self.error = f"{type(error).__name__}: {error}"
            print(f"[rife] interpolation failed ({self.error}); using blend")
            return blend_interpolate(frames, multiplier), "blend"

    def _rife(self, frames, multiplier: int) -> list:
        import torch
        import torch.nn.functional as functional

        steps = int(multiplier) - 1
        arrays = [_to_array(frame) for frame in frames]
        height, width, _ = arrays[0].shape
        # RIFE's flow net needs dimensions divisible by 128 at scale 1.0.
        block = 128
        pad_h = ((height - 1) // block + 1) * block - height
        pad_w = ((width - 1) // block + 1) * block - width
        padding = (0, pad_w, 0, pad_h)
        half = torch.cuda.is_available()

        def to_tensor(array):
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
            tensor = functional.pad(tensor, padding)
            return tensor.half() if half else tensor.float()

        def to_frame(tensor):
            return _to_pil(tensor[0, :, :height, :width].permute(1, 2, 0).float().cpu().numpy())

        model = self.model
        out = [to_frame(to_tensor(arrays[0]))]
        left = to_tensor(arrays[0])
        for index in range(len(arrays) - 1):
            right = to_tensor(arrays[index + 1])
            for step in range(steps):
                t = (step + 1) / (steps + 1)
                out.append(to_frame(model.inference(left, right, t, 1.0)))
            out.append(to_frame(right))
            left = right
        return out

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "loaded": self.model is not None,
            "error": self.error,
            "repo": RIFE_REPO,
        }
