import math
import torch
import gc
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from enum import Enum

from comfy import model_management
from .processing_pose_tiled import process_images_pose_tiled
from .utils import tensor_to_pil, pil_to_tensor


class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2


class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3


def load_upscaler(upscale_model):
    """Return the upscaler model wrapper if provided, else None."""
    return upscale_model if upscale_model is not None else None


def upscale_image(image, upscaler, scale_factor):
    """
    Try model upscaler first, fallback to Lanczos.
    """
    if upscaler is None:
        w = int(image.width * scale_factor)
        h = int(image.height * scale_factor)
        return image.resize((w, h), Image.LANCZOS)

    try:
        model = getattr(upscaler, 'model', upscaler)
        in_t = pil_to_tensor(image).unsqueeze(0)  # [1,C,H,W]
        in_t = in_t.to(model_management.get_torch_device(), non_blocking=True)

        if hasattr(model, 'upscale'):
            upscaled = model.upscale(in_t, scale_factor)
        else:
            upscaled = model(in_t, scale=scale_factor)

        return tensor_to_pil(upscaled)
    except Exception:
        # fallback to CPU resize if custom model call fails
        w = int(image.width * scale_factor)
        h = int(image.height * scale_factor)
        return image.resize((w, h), Image.LANCZOS)


class USDURedraw:
    """
    Pass 1: redraw tiles with overlap padding.
    - We build a rectangular mask per tile.
    - We do NOT blur that mask ourselves anymore.
      process_images_pose_tiled() will blur AFTER it figures out the bbox,
      so the bbox math sees a crisp region and ControlNet stays aligned.
    - We no longer call empty_cache() for every tile; that was a big slowdown.
    """

    def __init__(self, tile_width, tile_height, padding, mask_blur, mode, lock_padding=True, noise_mask_blur=0, blend_mode="normalized"):
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.padding = padding            # neighbor context to include
        self.mask_blur = mask_blur        # feather radius for final composite
        self.lock_padding = lock_padding
        self.noise_mask_blur = noise_mask_blur
        self.blend_mode = blend_mode
        self.mode = USDUMode(mode)
        self.enabled = self.mode != USDUMode.NONE

    def _run_single_masked_tile(
        self,
        processed_images,
        model, positive, negative, vae, seed, steps, cfg,
        sampler_name, scheduler, denoise,
        mask_img,
        uniform_mode, tiled_decode, return_layer=False
    ):
        tile_size = (
            math.ceil((self.tile_width  + 2*self.padding) / 64) * 64,
            math.ceil((self.tile_height + 2*self.padding) / 64) * 64,
        )

        return process_images_pose_tiled(
            model, positive, negative, vae, seed, steps, cfg,
            sampler_name, scheduler,
            processed_images, mask_img,
            tile_size, uniform_mode, self.mask_blur, denoise,
            tiled_decode=tiled_decode,
            tile_padding=self.padding,
            lock_padding=self.lock_padding,
            noise_mask_blur=self.noise_mask_blur,
            return_layer=return_layer,
        )
    def _normalized_grid_process(
        self,
        model, positive, negative, vae, seed, steps, cfg,
        sampler_name, scheduler, denoise,
        images, uniform_mode, tiled_decode,
        chess_even=None,
    ):
        # Normalized blending (tile-only, no base bleed-through):
        #- Accumulate premultiplied RGB and alpha weights.
        #- Compose as accum/weight where weight>0, else fall back to base.

        # This prevents the original/base image from showing through in feather
        # bands when mask_blur > 0 (the classic \"third-color seam band\").

        # NOTE: In this scheme, alpha is used to weight *tile vs tile* in overlaps.
        #In regions covered by only one tile, the tile wins regardless of alpha
        #(since accum/weight cancels alpha). This is what eliminates base bleed.
        
        processed_images = [img.copy().convert("RGB") for img in images]
        base_images = [img.copy().convert("RGB") for img in images]

        base_w, base_h = base_images[0].size
        rows = math.ceil(base_h / self.tile_height)
        cols = math.ceil(base_w / self.tile_width)

        accums = [np.zeros((base_h, base_w, 3), dtype=np.float32) for _ in base_images]
        weight = np.zeros((base_h, base_w), dtype=np.float32)

        for yi in range(rows):
            for xi in range(cols):
                if chess_even is not None:
                    if (yi + xi) % 2 != (0 if chess_even else 1):
                        continue

                x1 = xi * self.tile_width
                y1 = yi * self.tile_height
                x2 = min(x1 + self.tile_width, base_w)
                y2 = min(y1 + self.tile_height, base_h)
                if x2 <= x1 or y2 <= y1:
                    continue

                mask_img = Image.new("L", (base_w, base_h), "black")
                draw = ImageDraw.Draw(mask_img)
                draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)

                layers, bbox = self._run_single_masked_tile(
                    processed_images,
                    model, positive, negative, vae, seed, steps, cfg,
                    sampler_name, scheduler, denoise,
                    mask_img,
                    uniform_mode, tiled_decode,
                    return_layer=True,
                )

                if bbox is None:
                    continue
                bx1, by1, bx2, by2 = bbox
                if bx2 <= bx1 or by2 <= by1:
                    continue

                a0 = np.array(layers[0].crop(bbox), dtype=np.uint8)[..., 3].astype(np.float32) / 255.0
                if a0.max() <= 0.0:
                    continue

                weight[by1:by2, bx1:bx2] += a0
                wgt = weight[by1:by2, bx1:bx2]

                for bi, layer in enumerate(layers):
                    crop = np.array(layer.crop(bbox), dtype=np.uint8)
                    rgb = crop[..., :3].astype(np.float32) / 255.0
                    a = crop[..., 3].astype(np.float32) / 255.0

                    accums[bi][by1:by2, bx1:bx2, :] += rgb * a[..., None]

                    out = accums[bi][by1:by2, bx1:bx2, :] / np.maximum(wgt[..., None], 1e-6)
                    base_crop = np.array(base_images[bi].crop(bbox), dtype=np.uint8).astype(np.float32) / 255.0
                    out = np.where(wgt[..., None] > 1e-6, out, base_crop)

                    out_u8 = (out * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
                    processed_images[bi].paste(Image.fromarray(out_u8, mode="RGB"), (bx1, by1))

        return processed_images

    def linear_process(
        self,
        model, positive, negative, vae, seed, steps, cfg,
        sampler_name, scheduler, denoise,
        images, uniform_mode, tiled_decode
    ):

        if self.blend_mode == "normalized" and self.lock_padding and self.mask_blur > 0:
            return self._normalized_grid_process(
                model, positive, negative, vae, seed, steps, cfg,
                sampler_name, scheduler, denoise,
                images, uniform_mode, tiled_decode,
                chess_even=None,
            )

        processed_images = [img.copy() for img in images]

        base_w, base_h = images[0].size
        rows = math.ceil(base_h / self.tile_height)
        cols = math.ceil(base_w / self.tile_width)

        for yi in range(rows):
            for xi in range(cols):
                x1 = xi * self.tile_width
                y1 = yi * self.tile_height
                x2 = min(x1 + self.tile_width, base_w)
                y2 = min(y1 + self.tile_height, base_h)
                if x2 <= x1 or y2 <= y1:
                    continue

                mask_img = Image.new("L", (base_w, base_h), "black")
                draw = ImageDraw.Draw(mask_img)
                draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)

                processed_images = self._run_single_masked_tile(
                    processed_images,
                    model, positive, negative, vae, seed, steps, cfg,
                    sampler_name, scheduler, denoise,
                    mask_img,
                    uniform_mode, tiled_decode
                )

        return processed_images

    def _process_chess_pass(
        self,
        images, model, positive, negative, vae, seed, steps, cfg,
        sampler_name, scheduler, denoise,
        uniform_mode, tiled_decode, even
    ):

        if self.blend_mode == "normalized" and self.lock_padding and self.mask_blur > 0:
            return self._normalized_grid_process(
                model, positive, negative, vae, seed, steps, cfg,
                sampler_name, scheduler, denoise,
                images, uniform_mode, tiled_decode,
                chess_even=even,
            )
        processed_images = [img.copy() for img in images]
        base_w, base_h = images[0].size
        rows = math.ceil(base_h / self.tile_height)
        cols = math.ceil(base_w / self.tile_width)

        for yi in range(rows):
            for xi in range(cols):
                if (yi + xi) % 2 == (0 if even else 1):
                    x1 = xi * self.tile_width
                    y1 = yi * self.tile_height
                    x2 = min(x1 + self.tile_width, base_w)
                    y2 = min(y1 + self.tile_height, base_h)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    mask_img = Image.new("L", (base_w, base_h), "black")
                    draw = ImageDraw.Draw(mask_img)
                    draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)

                    processed_images = self._run_single_masked_tile(
                        processed_images,
                        model, positive, negative, vae, seed, steps, cfg,
                        sampler_name, scheduler, denoise,
                        mask_img,
                        uniform_mode, tiled_decode
                    )

        return processed_images

    def chess_process(
        self,
        model, positive, negative, vae, seed, steps, cfg,
        sampler_name, scheduler, denoise,
        images, uniform_mode, tiled_decode
    ):
        even_images = self._process_chess_pass(
            images, model, positive, negative, vae, seed, steps,
            cfg, sampler_name, scheduler, denoise,
            uniform_mode, tiled_decode,
            even=True
        )
        odd_images = self._process_chess_pass(
            even_images, model, positive, negative, vae, seed, steps,
            cfg, sampler_name, scheduler, denoise,
            uniform_mode, tiled_decode,
            even=False
        )
        return odd_images

    def start(
        self,
        model, positive, negative, vae, seed, steps, cfg,
        sampler_name, scheduler, denoise,
        images, uniform_mode, tiled_decode
    ):
        if not self.enabled:
            return images

        # do a single GC/empty_cache() here, not per tile
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.mode == USDUMode.LINEAR:
            out = self.linear_process(
                model, positive, negative, vae, seed, steps, cfg,
                sampler_name, scheduler, denoise,
                images, uniform_mode, tiled_decode
            )
        elif self.mode == USDUMode.CHESS:
            out = self.chess_process(
                model, positive, negative, vae, seed, steps, cfg,
                sampler_name, scheduler, denoise,
                images, uniform_mode, tiled_decode
            )
        else:
            out = images

        return out


class USDUSeamsFix:
    """
    Pass 2: targeted seam repair.
    We build thin/gradient masks along tile seams and re-run tiled diffusion
    only on those bands. Major speed wins come from:
    - vectorized mask generation (NumPy) instead of pixel-for-loops
    - not doing gc/empty_cache() for every tiny band
    """

    def __init__(self, width, padding, denoise, mask_blur, mode, lock_padding=True, noise_mask_blur=0, blend_mode="normalized"):
        self.width = width                # seam band width (px)
        self.padding = padding            # context for seam repair
        self.denoise = denoise
        self.mask_blur = mask_blur
        self.lock_padding = lock_padding
        self.noise_mask_blur = noise_mask_blur
        self.mode = USDUSFMode(mode)
        self.enabled = self.mode != USDUSFMode.NONE
        self.tile_width = 512
        self.tile_height = 512

    # ----- fast mask builders -----

    def _mask_horizontal_band(self, full_w, full_h, x_start, x_end, y_start):
        """
        Horizontal band starting at y_start with thickness self.width.
        Alpha ramps from 0->255 downward.
        """
        arr = np.zeros((full_h, full_w), dtype=np.uint8)

        y0 = max(0, y_start)
        y1 = min(full_h, y_start + self.width)
        if y1 <= y0:
            return Image.fromarray(arr, mode="L")

        x0 = max(0, x_start)
        x1c = min(full_w, x_end)
        if x1c <= x0:
            return Image.fromarray(arr, mode="L")

        band_h = y1 - y0
        grad = (np.arange(band_h, dtype=np.float32) / max(1, self.width)) * 255.0
        grad = grad.clip(0, 255).astype(np.uint8)[:, None]  # (band_h,1)
        band = np.repeat(grad, x1c - x0, axis=1)            # (band_h, band_w)
        arr[y0:y1, x0:x1c] = band

        return Image.fromarray(arr, mode="L")

    def _mask_vertical_band(self, full_w, full_h, x_start, y_start, y_end):
        """
        Vertical band starting at x_start with thickness self.width.
        Alpha ramps 0->255 to the right.
        """
        arr = np.zeros((full_h, full_w), dtype=np.uint8)

        y0 = max(0, y_start)
        y1 = min(full_h, y_end)
        if y1 <= y0:
            return Image.fromarray(arr, mode="L")

        x0 = max(0, x_start)
        x1c = min(full_w, x_start + self.width)
        if x1c <= x0:
            return Image.fromarray(arr, mode="L")

        band_w = x1c - x0
        grad = (np.arange(band_w, dtype=np.float32) / max(1, self.width)) * 255.0
        grad = grad.clip(0, 255).astype(np.uint8)[None, :]  # (1, band_w)
        band = np.repeat(grad, y1 - y0, axis=0)             # (band_h, band_w)
        arr[y0:y1, x0:x1c] = band

        return Image.fromarray(arr, mode="L")

    def _mask_vertical_band_pass(self, full_w, full_h, seam_center_x):
        """
        BAND_PASS mode:
        vertical strip centered on seam_center_x, fading out toward edges.
        """
        arr = np.zeros((full_h, full_w), dtype=np.uint8)

        half_w = self.width // 2
        x0 = seam_center_x - half_w
        x1c = x0 + self.width

        xs = max(0, x0)
        xe = min(full_w, x1c)
        if xe <= xs:
            return Image.fromarray(arr, mode="L")

        band_w = xe - xs
        center = self.width / 2.0
        rel = np.arange(band_w, dtype=np.float32) + (xs - x0)
        dist = np.abs(rel - center) / max(1.0, center)
        vals = (1.0 - dist).clip(0.0, 1.0) * 255.0
        vals = vals.astype(np.uint8)[None, :]  # (1,band_w)
        band = np.repeat(vals, full_h, axis=0)  # full height
        arr[:, xs:xe] = band

        return Image.fromarray(arr, mode="L")

    def _mask_corner_patch(self, full_w, full_h, cx, cy):
        """
        HALF_TILE_PLUS_INTERSECTIONS mode:
        little square patch around a tile corner, radial falloff alpha.
        """
        arr = np.zeros((full_h, full_w), dtype=np.uint8)

        half_wid = self.width // 2
        x0 = cx - half_wid
        y0 = cy - half_wid
        x1 = x0 + self.width
        y1 = y0 + self.width

        xs = max(0, x0)
        ys = max(0, y0)
        xe = min(full_w, x1)
        ye = min(full_h, y1)
        if xe <= xs or ye <= ys:
            return Image.fromarray(arr, mode="L")

        sub_w = xe - xs
        sub_h = ye - ys

        yy, xx = np.mgrid[0:sub_h, 0:sub_w]
        cx_local = (self.width / 2.0) - (xs - x0)
        cy_local = (self.width / 2.0) - (ys - y0)
        dist = np.sqrt((xx - cx_local) ** 2 + (yy - cy_local) ** 2)
        norm = dist / max(1.0, (self.width / 2.0))
        vals = (1.0 - norm).clip(0.0, 1.0) * 255.0
        vals = vals.astype(np.uint8)

        arr[ys:ye, xs:xe] = vals

        return Image.fromarray(arr, mode="L")

    def _run_mask_pass(
        self,
        processed_images,
        mask_img,
        model, positive, negative, vae, seed, steps, cfg,
        sampler_name, scheduler,
        uniform_mode, tiled_decode
    ):
        # IMPORTANT:
        # We do NOT blur here. process_images_pose_tiled() will:
        #   - compute bbox from this unblurred mask (good for alignment)
        #   - blur internally for alpha feather at composite time.
        tile_size = (self.tile_width, self.tile_height)

        new_tiles = process_images_pose_tiled(
            model, positive, negative, vae, seed, steps, cfg,
            sampler_name, scheduler,
            processed_images, mask_img,
            tile_size, uniform_mode, self.mask_blur, self.denoise,
            tiled_decode=tiled_decode,
            tile_padding=self.padding,
            lock_padding=self.lock_padding,
            noise_mask_blur=self.noise_mask_blur,
        )

        for idx, new_tile in enumerate(new_tiles):
            processed_images[idx] = new_tile
        return processed_images

    # ----- seam strategies -----

    def half_tile_process(
        self,
        model, positive, negative, vae, seed, steps, cfg,
        sampler_name, scheduler,
        images, uniform_mode, tiled_decode
    ):
        processed_images = [img.copy() for img in images]
        base_w, base_h = images[0].size
        rows = math.ceil(base_h / self.tile_height)
        cols = math.ceil(base_w / self.tile_width)

        # Horizontal seams
        for yi in range(rows - 1):
            seam_y = (yi + 1) * self.tile_height - self.width // 2
            for xi in range(cols):
                x_start = xi * self.tile_width
                x_end = x_start + self.tile_width
                mask_img = self._mask_horizontal_band(
                    base_w, base_h, x_start, x_end, seam_y
                )

                processed_images = self._run_mask_pass(
                    processed_images,
                    mask_img,
                    model, positive, negative, vae, seed, steps, cfg,
                    sampler_name, scheduler,
                    uniform_mode, tiled_decode
                )

        # Vertical seams
        for yi in range(rows):
            for xi in range(cols - 1):
                seam_x = (xi + 1) * self.tile_width - self.width // 2
                y_start = yi * self.tile_height
                y_end = y_start + self.tile_height
                mask_img = self._mask_vertical_band(
                    base_w, base_h, seam_x, y_start, y_end
                )

                processed_images = self._run_mask_pass(
                    processed_images,
                    mask_img,
                    model, positive, negative, vae, seed, steps, cfg,
                    sampler_name, scheduler,
                    uniform_mode, tiled_decode
                )

        return processed_images

    def band_pass_process(
        self,
        model, positive, negative, vae, seed, steps, cfg,
        sampler_name, scheduler,
        images, uniform_mode, tiled_decode
    ):
        processed_images = [img.copy() for img in images]
        base_w, base_h = images[0].size
        cols = math.ceil(base_w / self.tile_width)

        # Vertical "bands" spanning full height
        for xi in range(1, cols):
            seam_center_x = xi * self.tile_width
            mask_img = self._mask_vertical_band_pass(
                base_w, base_h, seam_center_x
            )

            processed_images = self._run_mask_pass(
                processed_images,
                mask_img,
                model, positive, negative, vae, seed, steps, cfg,
                sampler_name, scheduler,
                uniform_mode, tiled_decode
            )

        return processed_images

    def half_tile_process_corners(
        self,
        model, positive, negative, vae, seed, steps, cfg,
        sampler_name, scheduler,
        images, uniform_mode, tiled_decode
    ):
        # First standard seam pass
        images = self.half_tile_process(
            model, positive, negative, vae, seed, steps, cfg,
            sampler_name, scheduler,
            images, uniform_mode, tiled_decode
        )

        processed_images = [img.copy() for img in images]
        base_w, base_h = images[0].size
        rows = math.ceil(base_h / self.tile_height)
        cols = math.ceil(base_w / self.tile_width)

        # Patch tile intersections
        for yi in range(rows - 1):
            for xi in range(cols - 1):
                cx = (xi + 1) * self.tile_width
                cy = (yi + 1) * self.tile_height
                mask_img = self._mask_corner_patch(base_w, base_h, cx, cy)

                processed_images = self._run_mask_pass(
                    processed_images,
                    mask_img,
                    model, positive, negative, vae, seed, steps, cfg,
                    sampler_name, scheduler,
                    uniform_mode, tiled_decode
                )

        return processed_images

    def start(
        self,
        model, positive, negative, vae, seed, steps, cfg,
        sampler_name, scheduler, denoise,
        images, uniform_mode, tiled_decode
    ):
        if not self.enabled:
            return images

        # single GC/empty_cache() before seam sweep instead of per tiny band
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.mode == USDUSFMode.BAND_PASS:
            return self.band_pass_process(
                model, positive, negative, vae, seed, steps, cfg,
                sampler_name, scheduler,
                images, uniform_mode, tiled_decode
            )
        elif self.mode == USDUSFMode.HALF_TILE:
            return self.half_tile_process(
                model, positive, negative, vae, seed, steps, cfg,
                sampler_name, scheduler,
                images, uniform_mode, tiled_decode
            )
        elif self.mode == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            return self.half_tile_process_corners(
                model, positive, negative, vae, seed, steps, cfg,
                sampler_name, scheduler,
                images, uniform_mode, tiled_decode
            )

        return images


class USDUpscaler:
    """
    Full pipeline:
      1. (optional) learned/image-model upscale
      2. Redraw tiles with overlap (USDURedraw)
      3. Repair seams (USDUSeamsFix)

    We centralize cache cleanup at pass boundaries, not inside inner loops.
    """

    def __init__(
        self,
        upscale_by,
        upscale_model,
        redraw_mode,
        tile_width,
        tile_height,
        mask_blur,
        tile_padding,
        seam_fix_mode,
        seam_fix_denoise,
        seam_fix_width,
        seam_fix_mask_blur,
        seam_fix_padding,
        force_uniform_tiles,
        tiled_decode,
        lock_padding=True,
        noise_mask_blur=0,
        blend_mode="normalized"
    ):
        self.scale_factor = upscale_by
        self.upscaler = load_upscaler(upscale_model)

        # redraw / repaint pass
        self.redraw = USDURedraw(
            tile_width,
            tile_height,
            tile_padding,       # overlap context
            mask_blur,
            redraw_mode,
            lock_padding=lock_padding,
            noise_mask_blur=noise_mask_blur,
            blend_mode=blend_mode
        )

        # seam repair pass
        self.seams_fix = USDUSeamsFix(
            seam_fix_width,
            seam_fix_padding,   # overlap context for seam fixes
            seam_fix_denoise,
            seam_fix_mask_blur,
            seam_fix_mode,
            lock_padding=lock_padding,
            noise_mask_blur=noise_mask_blur,
            blend_mode=blend_mode
        )

        self.force_uniform_tiles = force_uniform_tiles
        self.tiled_decode = tiled_decode

    def upscale(self, images):
        if self.upscaler:
            return [
                upscale_image(img, self.upscaler, self.scale_factor)
                for img in images
            ]
        else:
            w = int(images[0].width * self.scale_factor)
            h = int(images[0].height * self.scale_factor)
            return [img.resize((w, h), Image.LANCZOS) for img in images]

    def process(
        self,
        model, positive, negative, vae, seed, steps, cfg,
        sampler_name, scheduler,
        denoise, images
    ):
        # 1. initial (cheap) upscale
        images = self.upscale(images)

        # 2. redraw tiles with padding
        images = self.redraw.start(
            model, positive, negative, vae, seed, steps, cfg,
            sampler_name, scheduler, denoise,
            images, self.force_uniform_tiles, self.tiled_decode
        )

        # 3. seam repair
        images = self.seams_fix.start(
            model, positive, negative, vae, seed, steps, cfg,
            sampler_name, scheduler, denoise,
            images, self.force_uniform_tiles, self.tiled_decode
        )

        # Return list of [C,H,W] tensors for Comfy "IMAGE"
        return [pil_to_tensor(img) for img in images]
