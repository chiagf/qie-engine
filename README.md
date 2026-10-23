# qie-engine

A portable **Wan2.2 I2V** engine: first frame (optionally last frame) in, an mp4 out. Extracted
from [qie-studio](https://github.com/chiagf/qie-studio) so more than one app can share one
engine instead of forking it.

```
from qie_engine import WanEngine
from qie_engine.settings import load as load_wan

engine = WanEngine(load_wan())
error = engine.load()          # None on success, or the error string to show the user
result = engine.run("first.png", "last.png", "slow dolly in", duration=4.0, seed=7)
```

## What it is

- **Engine** - `WanEngine` builds a `WanImageToVideoPipeline` from the two Wan2.2-A14B experts.
  Two backends, tried in order: the canonical **Comfy fp8-scaled** expert checkpoints via
  `from_single_file`, then the official **bf16** diffusers repo with a torchao fp8-dynamic
  requantize as the automatic fallback.
- **Settings** - one `WanSettings` dataclass, every knob an env var (`QIE_WAN_*`). The engine
  never imports an app module, so any app that owns its own `Config` can hand it one of these.
- **Schedulers** - FlowMatchEulerDiscrete (default, `shift=8.0`), UniPC, DEIS and the three
  DPM++ variants. Written per-call, so switching mid-session needs no reload.
- **RIFE** - optional optical-flow interpolation (`r3gm/RIFE` v4.26) that renders a 16 fps clip
  at 32/64/128 fps. Falls back to linear cross-fade rather than failing.

## On a 48 GB card (RTX 4090)

At the default 480x832 / ~81 frames the resident denoise peak is **~33 GiB**:

| Component | GiB |
| --- | --- |
| umt5-XXE text encoder (bf16) | 11.4 |
| Active fp8 expert (14B) | 14.0 |
| CLIP-vision | 1.4 |
| wan VAE (tiled) | 0.5 |
| Activations | 6.0 |

The plan keeps one expert resident and swaps the other in mid-run; the swap briefly holds both
(+14 GiB, reported as `swap_peak_gb` and never used to decide `fits`). `QIE_WAN_OFFLOAD=model_cpu`
gives headroom if it ever gets tight.

720x1280 at 81 frames needs `QIE_WAN_OFFLOAD=model_cpu`.

## Weights

Downloaded on first boot into `$HF_HOME` (~45 GB). Give the box **60 GB** of disk.

## Env vars

| Var | Default | |
| --- | --- | --- |
| `QIE_WAN_DTYPE` | `fp8` | `fp8` \| `bf16` |
| `QIE_WAN_OFFLOAD` | `auto` | `auto` \| `model_cpu` \| `sequential` \| `none` |
| `QIE_WAN_STACK` | `download` | `download` \| `baked` (`/models` layout) |
| `QIE_WAN_STEPS` | `8` | Lightning is tuned for 4-8 |
| `QIE_WAN_SHIFT` | `8.0` | FlowMatchEuler shift |
| `QIE_WAN_MAX_MP` | `0.40` | output ceiling (480x832) |
| `QIE_WAN_SCHEDULER` | `FlowMatchEulerDiscrete` | see `qie_engine.settings.SCHEDULERS` |
| `QIE_WAN_FLUIDITY` | `1` | `1` \| `2` \| `4` \| `8` (RIFE) |
| `QIE_WAN_RIFE_ENABLED` | `on` | off = never interpolate |
| `QIE_WAN_LORA_ENABLED` | `on` | lightx2v Lightning pair |

## Tests

CPU-only, no torch, no network:

```
python -m unittest discover -s tests -v
```

## Credits

The engine was extracted from qie-studio's phase-2 video mode. RIFE inference comes from
[r3gm/RIFE](https://huggingface.co/r3gm/RIFE) (Apache-2.0, originally megvii-research/RIFE); the
`model/` support code it imports is vendored under `qie_engine/_rife_support/`, trimmed to the
inference path. Wan2.2 weights are under their own licenses on Hugging Face.
