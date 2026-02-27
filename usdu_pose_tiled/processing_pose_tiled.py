import math
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter

from comfy import model_management
from nodes import common_ksampler

from .utils import tensor_to_pil, get_crop_region, expand_crop, crop_cond

# flip this on if you actually want the verbose per-tile logs
USDU_DEBUG = False


def _pil_list_to_bhwc(tiles):
    """
    Convert list[PIL] -> torch float tensor [B,H,W,C] in [0,1] on the active device.
    We avoid extra conversions and send straight to the current torch device.
    """
    device = model_management.get_torch_device()
    arr_t = []
    for p in tiles:
        # np.array(copy=False) is fine here, then cast/scale
        np_img = np.array(p, copy=False).astype(np.float32) / 255.0  # HWC
        t = torch.from_numpy(np_img)  # CPU
        arr_t.append(t)
    batch = torch.stack(arr_t, dim=0).to(device, non_blocking=True)
    return batch


def _extract_latent_samples(ksampler_result):
    """
    Normalize common_ksampler(...) output into a BCHW latent tensor for vae.decode().

    Handles:
      - {"samples": <tensor>}
      - (<same dict>,)
      - object with .samples
      - raw tensor
    """
    out = ksampler_result

    # If tuple/list, grab the first element
    if isinstance(out, (list, tuple)):
        if len(out) == 0:
            raise RuntimeError("KSampler returned empty result")
        out = out[0]

    # Dict with 'samples'
    if isinstance(out, dict):
        if "samples" in out:
            return out["samples"]

    # Object with .samples
    if hasattr(out, "samples"):
        return out.samples

    # Raw tensor
    if torch.is_tensor(out):
        return out

    raise RuntimeError(f"Unsupported ksampler output type: {type(out)}")


@torch.no_grad()
def _make_latent_noise_mask_from_tile_mask(hard_mask_full, crop_region_full, latent_hw, blur_px):
    """
    Build a latent-space noise mask for inpainting-style sampling.

    hard_mask_full: PIL 'L' mask in full canvas coords. White=editable (core), black=locked (padding/context).
    crop_region_full: (x1,y1,x2,y2) bbox in full canvas coords that we are encoding/sampling.
    latent_hw: (latent_h, latent_w) spatial size of the latent samples tensor.
    blur_px: blur radius in *full-res pixels* applied before downsampling. Use 0 for a hard edge.

    Returns: torch.float32 [1, latent_h, latent_w] in [0,1], where 1 means "denoise here".
    """
    try:
        core_crop = hard_mask_full.crop(crop_region_full).convert("L")
    except Exception:
        # If something goes wrong, fall back to a full-ones mask (no locking)
        lh, lw = int(latent_hw[0]), int(latent_hw[1])
        return torch.ones((1, lh, lw), dtype=torch.float32)

    # Binarize first (keeps the editable region crisp)
    core_bin = core_crop.point(lambda p: 255 if p > 127 else 0)

    # Blur for a soft falloff, but clamp spill back inside the core
    if blur_px and blur_px > 0:
        blurred = core_bin.filter(ImageFilter.GaussianBlur(int(blur_px)))
    else:
        blurred = core_bin

    lh, lw = int(latent_hw[0]), int(latent_hw[1])

    # Downsample to latent resolution
    # IMPORTANT: if blur_px == 0 we want a truly hard edge in latent space too.
    # Using BILINEAR would introduce a thin fractional transition ring even when blur is 0.
    resample = Image.Resampling.BILINEAR if (blur_px and blur_px > 0) else Image.Resampling.NEAREST
    blurred_rs = blurred.resize((lw, lh), resample)
    core_rs = core_bin.resize((lw, lh), Image.Resampling.NEAREST)

    m = (np.array(blurred_rs, dtype=np.float32) / 255.0)
    c = (np.array(core_rs, dtype=np.float32) / 255.0)

    m = (m * c).clip(0.0, 1.0)  # ensure outside-core stays 0

    return torch.from_numpy(m).unsqueeze(0)  # [1,lh,lw]

def process_images_pose_tiled(
    model,
    positive,
    negative,
    vae,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    init_images,
    image_mask,
    tile_size,
    uniform_tile_mode,
    mask_blur,
    denoise,
    custom_sampler=None,
    custom_sigmas=None,
    progress_bar_enabled=False,
    tiled_decode=False,
    tile_padding=0,
    lock_padding=True,
    noise_mask_blur=0,
    return_layer=False,
):
    """
    Core tiled diffusion pass (1 tile mask at a time).

    Big things this function guarantees:
    - The bbox / crop region for this tile is computed from the *hard* mask
      BEFORE blur. That bbox is what we send to UNet and what we use to crop
      ControlNet hints.
    - We only blur the mask AFTER bbox math, when we feather-alpha the paste.
      This prevents ControlNet conditioning from being "shrunk" and misaligned.
    - We upsample/crop ControlNet conditioning to EXACTLY the latent UNet size
      (latent_size), which fixes the classic ghost-offset problem.
    """

    # lightweight progress accounting (not super expensive if False)
    pbar = None
    if progress_bar_enabled:
        pbar = tqdm(total=1, desc="USDU PoseTiled", unit="tile")

    if not init_images or init_images[0] is None:
        if pbar:
            pbar.close()
        raise ValueError("No init image provided")

    # -------------------------------------------------------------------------
    # 1. Prep hard mask for bbox math, and also prep feathered alpha for later
    # -------------------------------------------------------------------------
    # image_mask can come in as any mode. We want:
    #   - hard_mask: binary-ish (unblurred). Used to compute crop region.
    #   - feather_mask: blurred alpha, used ONLY for compositing.
    hard_mask = image_mask.convert("L")
    if mask_blur > 0:
        feather_mask = hard_mask.filter(ImageFilter.GaussianBlur(mask_blur))
    else:
        feather_mask = hard_mask

    init_image = init_images[0]
    full_w, full_h = init_image.width, init_image.height

    # Bounding box of nonzero alpha region on the *hard* mask (+ tile_padding inflation)
    crop_region0 = get_crop_region(hard_mask, tile_padding)
    x1_0, y1_0, x2_0, y2_0 = crop_region0
    crop_w0 = x2_0 - x1_0
    crop_h0 = y2_0 - y1_0

    # ---------------------------------------------------------------------
    # Fast-path: "single full-image tile" with no padding and no blurs.
    #
    # When the mask covers the entire canvas and there is no overlap/padding
    # and no feathering, the tiled algorithm should behave like a plain
    # Encode -> KSampler -> Decode pass (i.e. no post-composite artifacts).
    #
    # We only enable this when the image is already /8-aligned so we don't
    # need any resizes that could diverge from a basic pipeline.
    # ---------------------------------------------------------------------
    is_full_canvas = (crop_region0 == (0, 0, full_w, full_h))
    can_passthrough = (
        is_full_canvas
        and tile_padding == 0
        and mask_blur == 0
        and (full_w % 8 == 0)
        and (full_h % 8 == 0)
    )

    if can_passthrough:
        # No cropping / resizing / compositing. Just run a full-frame img2img pass.
        tiles_for_model = [img.convert("RGB") for img in init_images]
        bhwc = _pil_list_to_bhwc(tiles_for_model)

        with torch.inference_mode():
            encoded = vae.encode(bhwc)
            if isinstance(encoded, dict) and "samples" in encoded:
                latents = encoded["samples"]
            elif hasattr(encoded, "samples"):
                latents = encoded.samples
            elif hasattr(encoded, "latent"):
                latents = encoded.latent
            else:
                latents = encoded

            latent = {"samples": latents}

            # If lock_padding is enabled but the editable region is the full canvas,
            # a noise mask would be all-ones anyway, so we skip it.

            ksampler_out = common_ksampler(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent=latent,
                denoise=denoise,
            )

            latent_samples_after = _extract_latent_samples(ksampler_out)
            decoded = vae.decode(latent_samples_after)

        results = [tensor_to_pil(decoded, i).convert("RGB") for i in range(decoded.shape[0])]

        if pbar:
            pbar.update(1)
            pbar.close()

        return results

    # Nothing to do if mask is effectively empty
    if crop_w0 <= 0 or crop_h0 <= 0:
        if pbar:
            pbar.close()
        # Just hand original back, untouched
        return ([], None) if return_layer else init_images

    # -------------------------------------------------------------------------
    # 2. Decide target aspect box for this tile
    # -------------------------------------------------------------------------
    # tile_size is "intended" tile shape incl padding (usually rounded to /64).
    # If uniform_tile_mode=True, we force aspect to match tile_size. Else we keep bbox aspect.
    if uniform_tile_mode:
        ratio_crop = crop_w0 / max(1, crop_h0)
        ratio_tile = tile_size[0] / max(1, tile_size[1])
        if ratio_crop > ratio_tile:
            target_w = crop_w0
            target_h = round(crop_w0 / max(1e-6, ratio_tile))
        else:
            target_h = crop_h0
            target_w = round(crop_h0 * ratio_tile)
    else:
        target_w = crop_w0
        target_h = crop_h0

    # Expand original bbox to match that aspect, clamped to canvas
    crop_region1, paste_size = expand_crop(
        crop_region0,
        full_w,
        full_h,
        target_w,
        target_h,
    )
    x1, y1, x2, y2 = crop_region1
    paste_w, paste_h = paste_size  # pixel size in canvas coords (no rounding to /8 yet)

    # UNet / VAE latents must be multiples of 8
    latent_w = int(math.ceil(paste_w / 8) * 8)
    latent_h = int(math.ceil(paste_h / 8) * 8)
    latent_size = (latent_w, latent_h)

    # -------------------------------------------------------------------------
    # 3. Prep per-tile input images for diffusion
    # -------------------------------------------------------------------------
    # We crop the source canvas to crop_region1 (paste_w x paste_h), then resize
    # to latent_size so UNet sees the correct resolution.
    tiles_for_model = []
    for img in init_images:
        cropped_canvas_region = img.crop(crop_region1)  # paste_w x paste_h
        resized_for_model = cropped_canvas_region.resize(
            latent_size, Image.Resampling.LANCZOS
        )
        tiles_for_model.append(resized_for_model)

    # -------------------------------------------------------------------------
    # 4. Crop ControlNet / cond for THIS TILE (alignment-critical)
    # -------------------------------------------------------------------------
    # We crop & resize spatial hints (pose maps, tile maps, depth, etc.) to:
    #   - the SAME bbox (crop_region1)
    #   - the SAME UNet spatial size (latent_size)
    # This matches UNet receptive field and kills the "shifted smaller overlay" bug.
    try:
        positive_cropped = crop_cond(
            positive,
            crop_region1,
            paste_size,          # not used internally anymore, keep for interface compat
            init_image.size,     # full canvas (W,H)
            latent_size,         # UNet spatial size
        )
        negative_cropped = crop_cond(
            negative,
            crop_region1,
            paste_size,
            init_image.size,
            latent_size,
        )
    except Exception as e:
        if USDU_DEBUG:
            print(f"[USDU] crop_cond failed, using original conditioning: {e}")
        positive_cropped, negative_cropped = positive, negative

    if USDU_DEBUG:
        try:
            print("\n[USDU DEBUG] TILE START =========================")
            print(f"[USDU DEBUG] crop_region0={crop_region0}, crop_region1={crop_region1}")
            print(f"[USDU DEBUG] paste_size={paste_size}, latent_size={latent_size}")
            print("[USDU DEBUG] TILE END ===========================\n")
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # 5. Encode -> sample -> decode
    # -------------------------------------------------------------------------
    bhwc = _pil_list_to_bhwc(tiles_for_model)  # [B, latent_h, latent_w, C] float32 [0,1]

    with torch.inference_mode():
        # Encode tile(s) to latent
        encoded = vae.encode(bhwc)  # Comfy VAE expects BHWC in [0,1]
        if isinstance(encoded, dict) and "samples" in encoded:
            latents = encoded["samples"]
        elif hasattr(encoded, "samples"):
            latents = encoded.samples
        elif hasattr(encoded, "latent"):
            latents = encoded.latent
        else:
            latents = encoded

        latent = {"samples": latents}

        # ---------------------------------------------------------------------
        # 5b. OPTIONAL: lock padding/context during sampling (inpainting-style)
        # ---------------------------------------------------------------------
        # If enabled, we attach a latent noise mask so the sampler ONLY denoises
        # inside the tile core (white region of hard_mask). The overlap/padding
        # stays frozen and cannot be "solved" / recolored by the model.
        if lock_padding:
            # 0 means "use the same blur as the composite mask"
            # noise_mask_blur controls *diffusion* falloff at the core boundary.
            #   -1 : follow mask_blur (legacy convenience)
            #    0 : truly hard edge (no diffusion falloff)
            #   >0 : explicit blur radius in full-res pixels
            if noise_mask_blur is None:
                blur_px = 0
            else:
                nmb = int(noise_mask_blur)
                blur_px = mask_blur if nmb < 0 else nmb

            lh, lw = int(latents.shape[-2]), int(latents.shape[-1])
            nm = _make_latent_noise_mask_from_tile_mask(
                hard_mask,
                crop_region1,          # crop bbox in full canvas coords
                (lh, lw),              # latent spatial size
                blur_px,
            )

            # match batch
            if nm.shape[0] != latents.shape[0]:
                nm = nm.repeat(latents.shape[0], 1, 1)

            latent["noise_mask"] = nm.to(latents.device, non_blocking=True)

        # Diffusion step
        ksampler_out = common_ksampler(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive_cropped,
            negative=negative_cropped,
            latent=latent,
            denoise=denoise,
        )

        # Normalize sampler output into tensor for decode
        latent_samples_after = _extract_latent_samples(ksampler_out)

        # Decode latent back to BHWC float [0,1]
        decoded = vae.decode(latent_samples_after)

    # -------------------------------------------------------------------------
    # 6. Resize decoded tile(s) BACK DOWN to paste_size, composite with feather
    # -------------------------------------------------------------------------
    tiles_sampled_fullres = [
        tensor_to_pil(decoded, i).resize(
            paste_size,
            Image.Resampling.LANCZOS,
        )
        for i in range(decoded.shape[0])
    ]

    out_layers_or_images = []
    for i, tile_sampled in enumerate(tiles_sampled_fullres):
        base_img = init_images[i].copy().convert("RGBA")

        #paste decoded tile where it belongs (full canvas sized RGBA layer)
        tile_rgba = Image.new("RGBA", base_img.size)
        tile_rgba.paste(tile_sampled, (x1, y1))

        # feather alpha uses the blurred mask we prepared
        # tile_rgba.putalpha(feather_mask)
        # Build an alpha mask that is ZERO outside the tile's paste bbox.
        # This prevents any blur spill from affecting pixels where the tile has no RGB content.
        paste_w, paste_h = paste_size
        bbox = (x1, y1, x1 + paste_w, y1 + paste_h)

        alpha_full = Image.new("L", base_img.size, 0)
        alpha_crop = feather_mask.crop(bbox)
        alpha_full.paste(alpha_crop, (x1, y1))
        tile_rgba.putalpha(alpha_full)

        if return_layer:
            out_layers_or_images.append(tile_rgba)
        else:
            # sequential alpha composite onto the running canvas
            base_img.alpha_composite(tile_rgba)
            out_layers_or_images.append(base_img.convert("RGB"))

    if pbar:
        pbar.update(1)
        pbar.close()

    # NOTE: we intentionally do NOT call gc.collect() or torch.cuda.empty_cache()
    # here. Doing that per tile was adding ~20s overhead across a big upscale.
    # We let higher-level passes clean memory once per pass instead.

    if return_layer:
        return out_layers_or_images, (x1, y1, x1 + paste_size[0], y1 + paste_size[1])

    return out_layers_or_images
