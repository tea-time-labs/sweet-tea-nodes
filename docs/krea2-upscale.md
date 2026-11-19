# Krea2 Ultimate Upscale

`STKrea2UltimateUpscale` uses the existing Sweet Tea Ultimate SD Upscale pipeline,
including overlap locking and normalized blending. Output dimensions follow the
source aspect ratio, rounded to a 32-pixel lattice. The requested tile size is a
maximum core size; rows and columns divide the output into similar-sized tiles.
Padding adds context beyond that core size.

Control modes are alternatives:

| Mode | Weights and nodes |
| --- | --- |
| Off | No control weights or extra control node pack. Strength zero also bypasses controls. |
| Depth | `Patil/Krea-2-depth-controlnet` / `depth-control-lora.safetensors`; `facok/comfyui-krea2-controlnet`; `Fannovel16/comfyui_controlnet_aux` with Depth Anything V2 Large. This estimator matches the control model's documented training/inference recipe. |
| Canny | `nynxz/NK2E` / `comfy/canny_v0.1/NK2E-canny-v0.1.safetensors`; `Nynxz/ComfyUI-NK2E`. Experimental in-context edge control, not a channel-concatenated facok ControlLoRA. |
| Tile | A compatible facok ControlLoRA. `AdwolfCzar/krea2-tile-controllora` requires author approval. Compatibility and image quality of those gated weights have not been tested. |

Place control weights in `models/loras`. Filename inputs can include subdirectories.
The selected mode fails explicitly if its weights or node pack are missing. Off
does not inspect or load them. Depth is estimated once for the full input to keep
global depth consistent, then every padded crop is resized and VAE-encoded to
exactly the tile's sampling size. Canny references are similarly encoded per tile;
the legacy local-reference NK2E node avoids the global-reference side channel.

Use text-only conditioning and the unpatched base/LoRA model for this node. If an
Ostris edit reference is wanted, perform that optional edit first, then feed its
image into the tiled pass. Combining those reference patches with the facok
expanded projection is unsupported.

An optional upscale-model input performs neural upscaling before the diffusion
pass. Without it, the initial resize uses Lanczos. Typical turbo starting values:
8 steps, CFG 1, denoise 0.25, 1024px core tiles, 64px padding, 16px blend blur.
CFG 1 normally ignores negative text. Use the original zero-conditioning path
when a negative prompt is empty.

Validation: CPU tests exercise the complete tile/composite loop with a coordinate
ramp, fake VAE and sampler, covering Off, zero strength, local hint alignment,
single full-frame tiles, Canny reference crops, and missing selected weights.
These checks do not establish real Kroma image quality, VRAM use, or gated Tile
weight compatibility.
