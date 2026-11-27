# Krea 2 upscaling

## Krea2 Guided Ultimate Upscale (Sweet Tea)

Add `STKrea2GuidedUpscale` from **Sweet Tea → Krea2**. Connect the original image,
native Krea 2 model, Qwen3-VL-4B CLIP loaded as `krea2`, and the Qwen Image VAE.
Enter the original creative prompt. An optional `UPSCALE_MODEL` runs before diffusion.
Without one the node enlarges the source with Lanczos. Both paths return the requested
dimensions after trimming working-canvas padding.

| Input | Effect |
| --- | --- |
| Vision | On by default. Qwen sees the prompt, a marked overview, the tile plus padding, and the core. It captions the core. Krea receives that caption and the matching padded source crop. With Vision off, all tiles receive the supplied text. |
| Canny | Uses edges detected on the original frame. Its strength controls attention to the tile’s aligned edge reference. Install `NK2E-canny-v0.1.safetensors` in `models/loras`. |
| Depth | Estimates one Depth Anything V2 Large map per frame. Strength scales the depth LoRA and the depth contribution to the target image tokens. Install the facok Krea 2 control node pack, comfyui_controlnet_aux and `depth-control-lora.safetensors`. |
| Reference | Uses the original image as an aligned colour reference, including when using a neural upscaler. Install `NK2E-v0.3.safetensors` in `models/loras`. Strength controls how much attention can go to that reference. |
| Denoise | Controls how far img2img resamples the initial upscale. Zero returns the resized image directly. |
| Tile count and padding | The defaults are a 3× upscale, nine tiles, 256 px padding per edge and eight Euler/simple steps at denoise 0.8. Choose Auto or a square grid from 4 through 100 tiles. Padding ranges from 0 to 512 px. |

Canny, Depth and Reference each have an enable switch. Setting a strength to zero
bypasses that guide and its weights. RGB and Canny keep their respective edit adapters
at 0.7 while enabled. Their sliders scale the attention odds at every step. A value
of 0.5 halves those odds within a layer. It does not specify a fraction of output pixels.
Turning additional guides on adds computation within each tile prediction. The
sampler combines overlapping predictions on the shared canvas at each step.

The original `STKrea2UltimateUpscale` remains available for existing workflows.

## Krea2 Ultimate Upscale (single-mode node)


`STKrea2UltimateUpscale` uses the existing Sweet Tea Ultimate SD Upscale pipeline,
including overlap locking and normalized blending. Output dimensions follow the
source aspect ratio, rounded to a 32-pixel lattice. In Auto mode, Tile size sets
the maximum core size, and the node chooses the row and column counts separately.
Use Tile count to select 4, 9, 16, 25, 36, 49, 64, 81, or 100 tiles. For example,
9 gives exactly 3 columns and 3 rows across the output, including on a wide or
tall canvas. Each axis is divided into 32-pixel-aligned cores. Padding adds
context around each core, clipped to the canvas at the edges.

## Choose the grid and refinement controls

| Control | What it changes |
| --- | --- |
| Tile count | Total cores in an N×N grid: 4 = 2×2, 9 = 3×3, 16 = 4×4, 25 = 5×5. An explicit count overrides Tile size for grid selection; it does not change the output dimensions. |
| Tile size | Maximum core size in pixels when Tile count is Auto. The output dimensions determine the row and column counts independently. |
| Upscale factor (`upscale_by`) | Output width and height relative to the input, rounded to the 32-pixel lattice. It is separate from the number of tiles. |
| Tile padding | Extra source context around each core, clipped at the canvas edges. It does not add cores to the grid. |
| Denoise | Img2img denoising amount passed to the sampler, separate from control-model strength and output scale. |
| Control strength | Weight strength for Depth, Canny or Tile control. Zero bypasses those controls; this input does not adjust Vision mode. |
| Sampler and scheduler | Sampling algorithm and noise schedule. Choosing a tile count does not select either of them. |

A square grid has equal row and column counts; its cores can be rectangular on a
wide or tall output. Each core needs at least one 32-pixel block on each axis. If
the requested grid is too dense for the output, the node asks for fewer tiles or
a larger upscale factor. Existing workflows that omit Tile count keep Auto.

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
pass. Without it, the initial resize uses Lanczos. The node defaults are
8 steps, CFG 1, denoise 0.25, Tile count Auto, Tile size 1024px, padding 64px and
blend blur 16px. These are input defaults, not a tested quality preset.
CFG 1 normally ignores negative text. Use the original zero-conditioning path
when a negative prompt is empty.

Validation: CPU tests exercise the complete tile/composite loop with a coordinate
ramp, fake VAE and sampler, covering Off, zero strength, local hint alignment,
single full-frame tiles, Canny reference crops, and missing selected weights.
These checks do not establish real Kroma image quality, VRAM use, or gated Tile
weight compatibility.

## Vision mode

Select `Vision` and connect the same Krea 2 Qwen3-VL CLIP used by the Pipe.
The optional vision prompt guides refinement. Each overlapping tile gets image
tokens from its matching crop of the original, encoded once before sampling.
Krea receives canvas-relative positions and the node blends predictions at each
denoising step, then decodes the shared latent. No control weights or extra
node pack are needed. Higher CFG accepts plain text negative conditioning.

Vision samples one shared canvas per image. Its sampler progress counts denoising
steps, not completed tiles: an eight-step display does not mean that only eight
tiles are being processed. Each model evaluation runs through the selected tile
regions and blends their predictions. Tile image conditioning is encoded before
sampling begins; this node does not emit separate per-tile sampling progress.

A Pipe can expose Tile count and keep Tile size and padding at their defaults.
The same grid selection applies in Vision and the other control modes. Smaller
tile regions reduce per-call diffusion work at the cost of more model calls;
Vision still keeps a shared full-canvas latent. `mask_blur` is for the legacy pixel
composite path. Vision uses cosine feathering over `tile_padding`.
