import torch
import numpy as np
import copy
import torch.nn.functional as F
from PIL import Image

# -----------------------------
# Basic tensor <-> PIL helpers
# -----------------------------

def pil_to_tensor(pil_image):
    """Convert PIL to unbatched tensor: [C,H,W] float32 [0,1]."""
    arr = np.array(pil_image, copy=True).astype(np.float32) / 255.0
    if arr.ndim == 3:  # HWC -> CHW
        arr = arr.transpose(2, 0, 1)
    else:  # HW -> [1,H,W]
        arr = np.expand_dims(arr, axis=0)
    return torch.from_numpy(arr)

def tensor_to_pil(tensor, index=0):
    """
    Convert [B,H,W,C] or [B,C,H,W] or [C,H,W] or [H,W] float[0,1] to PIL.
    We'll try to do the obvious channel ordering fixups.
    """
    if tensor.ndim == 4:
        t = tensor[index].detach().float().cpu().numpy()
    else:
        t = tensor.detach().float().cpu().numpy()
    t = (t * 255.0).clip(0, 255).astype(np.uint8)

    if t.ndim == 3:
        # could be CHW or HWC
        if t.shape[0] in (1, 3, 4):  # CHW
            if t.shape[0] == 1:
                t = t[0]
            else:
                t = np.transpose(t, (1, 2, 0))[:, :, :3]
        elif t.shape[-1] in (1, 3, 4):  # HWC
            if t.shape[-1] == 1:
                t = t[:, :, 0]
            else:
                t = t[:, :, :3]
    elif t.ndim > 3:
        t = np.squeeze(t)
        if t.ndim == 3 and t.shape[0] in (1, 3, 4):
            if t.shape[0] == 1:
                t = t[0]
            else:
                t = np.transpose(t, (1, 2, 0))[:, :, :3]

    return Image.fromarray(t)

# -----------------------------
# Region helpers
# -----------------------------

def get_crop_region(mask, padding=0):
    """
    Find bbox of nonzero mask, inflated by padding, clamped to image bounds.
    """
    mask_np = np.array(mask, copy=False)
    coords = np.argwhere(mask_np > 0)
    if len(coords) == 0:
        return (0, 0, mask_np.shape[1], mask_np.shape[0])
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)
    # coords.max() returns inclusive indices (last nonzero pixel). Cropping bboxes
    # are conventionally exclusive on the max corner, so we must +1 to include
    # the last row/col. Without this, we systematically under-crop by 1px.
    y2 += 1
    x2 += 1

    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(mask_np.shape[1], x2 + padding),
        min(mask_np.shape[0], y2 + padding),
    )

def expand_crop(crop_region, img_width, img_height, target_width, target_height):
    """
    Adjust bbox to match the desired aspect ratio, clamped to canvas.
    Returns (new_bbox, new_size).
    """
    x1, y1, x2, y2 = crop_region
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half_w = target_width // 2
    half_h = target_height // 2
    new_x1 = max(0, cx - half_w)
    new_x2 = min(img_width, cx + half_w)
    new_y1 = max(0, cy - half_h)
    new_y2 = min(img_height, cy + half_h)
    return (new_x1, new_y1, new_x2, new_y2), (new_x2 - new_x1, new_y2 - new_y1)

# -----------------------------
# Spatial crop/resize helpers
# -----------------------------

@torch.no_grad()
def _scale_crop_coords(region_fullspace, full_img_size, spatial_w, spatial_h):
    """
    Map a bbox in full-image pixel space (W_full,H_full) down to coordinates in
    a spatial tensor that's width=spatial_w, height=spatial_h.
    """
    x1, y1, x2, y2 = region_fullspace
    full_w, full_h = full_img_size
    sx = spatial_w / float(full_w)
    sy = spatial_h / float(full_h)

    cx1 = int(round(x1 * sx))
    cy1 = int(round(y1 * sy))
    cx2 = int(round(x2 * sx))
    cy2 = int(round(y2 * sy))

    cx1 = max(0, min(spatial_w, cx1))
    cx2 = max(0, min(spatial_w, cx2))
    cy1 = max(0, min(spatial_h, cy1))
    cy2 = max(0, min(spatial_h, cy2))

    return cx1, cy1, cx2, cy2

@torch.no_grad()
def _crop_resize_bhwc(t, region_fullspace, full_img_size, out_wh):
    """
    t: [B,H,W,C] or [H,W,C]
    Returns same layout but cropped+resized to out_wh (W,H).
    """
    added_batch = False
    if t.ndim == 3:  # HWC -> BHWC
        t = t.unsqueeze(0)
        added_batch = True

    B, H, W, C = t.shape
    cx1, cy1, cx2, cy2 = _scale_crop_coords(region_fullspace, full_img_size, W, H)

    cropped = t[:, cy1:cy2, cx1:cx2, :]          # [B,h,w,C]
    nchw = cropped.permute(0, 3, 1, 2)           # [B,C,h,w]
    resized = F.interpolate(
        nchw, size=(out_wh[1], out_wh[0]), mode='bilinear', align_corners=False
    )
    out_bhwc = resized.permute(0, 2, 3, 1)       # [B,Ht,Wt,C]

    if added_batch:
        out_bhwc = out_bhwc[0]                   # [Ht,Wt,C]

    return out_bhwc

@torch.no_grad()
def _crop_resize_bchw(t, region_fullspace, full_img_size, out_wh):
    """
    t: [B,C,H,W] or [C,H,W]
    Returns BCHW or CHW matching input batch-ness.
    """
    added_batch = False
    if t.ndim == 3:  # CHW -> BCHW
        t = t.unsqueeze(0)
        added_batch = True

    B, C, H, W = t.shape
    cx1, cy1, cx2, cy2 = _scale_crop_coords(region_fullspace, full_img_size, W, H)

    cropped = t[:, :, cy1:cy2, cx1:cx2]          # [B,C,h,w]
    resized = F.interpolate(
        cropped, size=(out_wh[1], out_wh[0]), mode='bilinear', align_corners=False
    )

    if added_batch:
        resized = resized[0]                     # [C,Ht,Wt]

    return resized

@torch.no_grad()
def _crop_resize_hw(t, region_fullspace, full_img_size, out_wh):
    """
    t: [H,W] (single-channel)
    Returns [Ht,Wt].
    """
    H, W = t.shape
    cx1, cy1, cx2, cy2 = _scale_crop_coords(region_fullspace, full_img_size, W, H)

    cropped = t[cy1:cy2, cx1:cx2]                # [h,w]
    cropped = cropped.unsqueeze(0).unsqueeze(0).float()  # [1,1,h,w]
    resized = F.interpolate(
        cropped, size=(out_wh[1], out_wh[0]), mode='bilinear', align_corners=False
    )[0, 0]

    return resized

def _looks_spatial(val):
    """
    Heuristic: does this value look like an image/map (pose map,
    ControlNet hint, etc.) that should be cropped spatially?
    """
    if isinstance(val, Image.Image):
        return True
    if isinstance(val, np.ndarray):
        if val.ndim in (2, 3, 4):
            if val.ndim == 2:
                H, W = val.shape
                # Treat 2D tensors as spatial only when both axes are image-like.
                # This avoids misclassifying vectors like pooled_output [1, 1280].
                return H >= 32 and W >= 32
            if val.ndim == 3:
                # could be CHW or HWC
                if val.shape[0] in (1, 3, 4):  # CHW
                    H, W = val.shape[1], val.shape[2]
                else:                          # HWC
                    H, W = val.shape[0], val.shape[1]
                return max(H, W) >= 32
            if val.ndim == 4:
                if val.shape[1] in (1, 3, 4):  # BCHW
                    H, W = val.shape[-2], val.shape[-1]
                else:                          # BHWC
                    H, W = val.shape[1], val.shape[2]
                return max(H, W) >= 32
    if torch.is_tensor(val):
        if val.ndim in (2, 3, 4):
            if val.ndim == 2:
                H, W = val.shape
                # Treat 2D tensors as spatial only when both axes are image-like.
                # This avoids misclassifying vectors like pooled_output [1, 1280].
                return H >= 32 and W >= 32
            if val.ndim == 3:
                if val.shape[0] in (1, 3, 4):  # CHW
                    H, W = val.shape[1], val.shape[2]
                else:                          # HWC
                    H, W = val.shape[0], val.shape[1]
                return max(H, W) >= 32
            if val.ndim == 4:
                if val.shape[1] in (1, 3, 4):  # BCHW
                    H, W = val.shape[-2], val.shape[-1]
                else:                          # BHWC
                    H, W = val.shape[1], val.shape[2]
                return max(H, W) >= 32
    return False

@torch.no_grad()
def _crop_spatial_value(val, region_fullspace, full_img_size, tile_size):
    """
    Crop+resample a spatial value (pose map, tile control, depth, etc.) to this tile.
    Do resize on CPU then move it back to original device/dtype.
    """
    # PIL -> tensor -> recurse -> back to PIL
    if isinstance(val, Image.Image):
        t = pil_to_tensor(val)                     # [C,H,W] CPU
        cropped = _crop_spatial_value(t, region_fullspace, full_img_size, tile_size)
        return tensor_to_pil(cropped)

    # numpy -> torch CPU -> recurse
    if isinstance(val, np.ndarray):
        t = torch.from_numpy(np.array(val, copy=True)).float()  # CPU
        cropped_t = _crop_spatial_value(t, region_fullspace, full_img_size, tile_size)
        return cropped_t.detach().cpu().numpy()

    # torch tensor path
    if torch.is_tensor(val):
        orig_device = val.device
        orig_dtype = val.dtype

        t = val.detach().to("cpu", non_blocking=True).float()  # CPU for resize

        if t.ndim == 4:
            if t.shape[1] in (1, 3, 4):   # BCHW
                out = _crop_resize_bchw(t, region_fullspace, full_img_size, tile_size)
            else:                         # BHWC
                out = _crop_resize_bhwc(t, region_fullspace, full_img_size, tile_size)
        elif t.ndim == 3:
            if t.shape[0] in (1, 3, 4):   # CHW
                out = _crop_resize_bchw(t, region_fullspace, full_img_size, tile_size)
            else:                         # HWC
                out = _crop_resize_bhwc(t, region_fullspace, full_img_size, tile_size)
        elif t.ndim == 2:
            out = _crop_resize_hw(t, region_fullspace, full_img_size, tile_size)
        else:
            return val

        # restore dtype/device
        target_dtype = torch.float16 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
        out = out.to(orig_device, dtype=target_dtype, non_blocking=True)
        return out

    # default passthrough
    return val

def _shallow_clone_list(l):
    return list(l)

def _shallow_clone_dict(d):
    return {k: v for k, v in d.items()}

def _shallow_clone_obj(o):
    try:
        return copy.copy(o)
    except Exception:
        return o

def _should_force_crop_key(k: str):
    """
    Keys in ControlNet dicts that usually carry spatial data.
    Even if _looks_spatial() can't tell, we crop them.
    """
    lk = (k or "").lower()
    hot = [
        "control", "hint", "image", "map", "hr_hint", "tile",
        "canny", "depth", "pose", "openpose", "seg", "normal", "edge", "mask",
        "input_image", "control_hint_original", "control_hint"
    ]
    return any(word in lk for word in hot)

def _is_non_spatial_conditioning_key(k: str):
    """
    Keys that carry non-spatial conditioning vectors and must never be cropped.
    """
    lk = (k or "").lower()
    blocked = {
        "pooled_output",
        "pooled",
        "text_embeds",
        "time_ids",
        "clip_embeds",
        "clip_embedding",
        "clip_pooled",
    }
    return lk in blocked

def _crop_control_recursive(ctrl_obj, region_fullspace, full_img_size, tile_size):
    """
    Recursively walk a ControlNet structure (lists, dicts, objects) and crop/resize
    any spatial tensors/images to this tile.
    """

    # list/tuple
    if isinstance(ctrl_obj, (list, tuple)):
        out = _shallow_clone_list(ctrl_obj)
        for i, v in enumerate(out):
            if (
                isinstance(v, (list, tuple, dict))
                or (hasattr(v, "__dict__") and not isinstance(v, (torch.Tensor, np.ndarray, Image.Image)))
            ):
                out[i] = _crop_control_recursive(v, region_fullspace, full_img_size, tile_size)
            else:
                if _looks_spatial(v):
                    out[i] = _crop_spatial_value(v, region_fullspace, full_img_size, tile_size)
        return out

    # dict
    if isinstance(ctrl_obj, dict):
        new_ctrl = _shallow_clone_dict(ctrl_obj)
        for k, v in list(new_ctrl.items())[:]:
            if _is_non_spatial_conditioning_key(k):
                continue
            if (
                isinstance(v, (list, tuple, dict))
                or (hasattr(v, "__dict__") and not isinstance(v, (torch.Tensor, np.ndarray, Image.Image)))
            ):
                new_ctrl[k] = _crop_control_recursive(v, region_fullspace, full_img_size, tile_size)
            else:
                if _looks_spatial(v) or _should_force_crop_key(k):
                    new_ctrl[k] = _crop_spatial_value(v, region_fullspace, full_img_size, tile_size)
        return new_ctrl

    # object-like
    obj = _shallow_clone_obj(ctrl_obj)
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        if _is_non_spatial_conditioning_key(attr):
            continue
        try:
            v = getattr(obj, attr)
        except Exception:
            continue

        if (
            isinstance(v, (list, tuple, dict))
            or (hasattr(v, "__dict__") and not isinstance(v, (torch.Tensor, np.ndarray, Image.Image)))
        ):
            try:
                setattr(obj, attr, _crop_control_recursive(v, region_fullspace, full_img_size, tile_size))
            except Exception:
                pass
        else:
            if _looks_spatial(v) or _should_force_crop_key(attr):
                try:
                    setattr(obj, attr, _crop_spatial_value(v, region_fullspace, full_img_size, tile_size))
                except Exception:
                    pass
    return obj

def crop_cond(conditioning, crop_region, init_size, img_size, tile_size):
    """
    Conditioning cropper:
    - Takes the tile bbox (crop_region) which ALREADY includes tile_padding.
    - Crops and resizes any spatial hints (pose maps, tiled control, etc.)
      down to tile_size, which is EXACTLY what UNet will see for that tile.
    - We do *not* inflate crop_region again. Inflating here was the source
      of the "conditioning is smaller & shifted" artifact.
    """

    region_exact = crop_region  # <-- no extra inflate

    # conditioning is often a list like [[cond_tensor, { ...control... }], ...]
    if isinstance(conditioning, list):
        out = []
        for entry in conditioning:
            if (
                isinstance(entry, (list, tuple))
                and len(entry) >= 2
                and isinstance(entry[1], dict)
            ):
                cond_tensor = entry[0]
                cond_dict   = entry[1]
                new_dict    = _shallow_clone_dict(cond_dict)

                if "control" in new_dict and new_dict["control"] is not None:
                    new_dict["control"] = _crop_control_recursive(
                        new_dict["control"],
                        region_exact,
                        img_size,
                        tile_size,
                    )
                else:
                    # Fallback: crop dict inline only when key/value look spatial.
                    # This prevents pooled_output vectors from being resized.
                    should_crop_inline = any(
                        (not _is_non_spatial_conditioning_key(k))
                        and (_looks_spatial(v) or _should_force_crop_key(k))
                        for k, v in new_dict.items()
                    )
                    if should_crop_inline:
                        new_dict = _crop_control_recursive(
                            new_dict,
                            region_exact,
                            img_size,
                            tile_size,
                        )

                out.append([cond_tensor, new_dict])

            elif isinstance(entry, dict):
                new_entry = _shallow_clone_dict(entry)
                if "control" in new_entry and new_entry["control"] is not None:
                    new_entry["control"] = _crop_control_recursive(
                        new_entry["control"],
                        region_exact,
                        img_size,
                        tile_size,
                    )
                else:
                    should_crop_inline = any(
                        (not _is_non_spatial_conditioning_key(k))
                        and (_looks_spatial(v) or _should_force_crop_key(k))
                        for k, v in new_entry.items()
                    )
                    if should_crop_inline:
                        new_entry = _crop_control_recursive(
                            new_entry,
                            region_exact,
                            img_size,
                            tile_size,
                        )
                out.append(new_entry)

            else:
                out.append(entry)
        return out

    # dict-style conditioning
    if isinstance(conditioning, dict):
        new_entry = _shallow_clone_dict(conditioning)
        if "control" in new_entry and new_entry["control"] is not None:
            new_entry["control"] = _crop_control_recursive(
                new_entry["control"],
                region_exact,
                img_size,
                tile_size,
            )
        else:
            should_crop_inline = any(
                (not _is_non_spatial_conditioning_key(k))
                and (_looks_spatial(v) or _should_force_crop_key(k))
                for k, v in new_entry.items()
            )
            if should_crop_inline:
                new_entry = _crop_control_recursive(
                    new_entry,
                    region_exact,
                    img_size,
                    tile_size,
                )
        return new_entry

    # object w/ .control attr
    if hasattr(conditioning, "control"):
        cond_copy = _shallow_clone_obj(conditioning)
        try:
            ctrl_val = getattr(cond_copy, "control")
            setattr(
                cond_copy,
                "control",
                _crop_control_recursive(
                    ctrl_val,
                    region_exact,
                    img_size,
                    tile_size,
                ),
            )
        except Exception:
            pass
        return cond_copy

    # passthrough for non-spatial stuff
    return conditioning
