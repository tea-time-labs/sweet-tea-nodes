"""Optional Canny and Depth guides for the Krea 2 Turbo test node."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch
import torch.nn.functional as F

from comfy.patcher_extension import WrappersMP
import comfy.model_management as mm
import comfy.utils
import folder_paths
from nodes import LoraLoaderModelOnly
from usdu_pose_tiled import krea2_vision as vision


def load_adapter(source):
    spec = spec_from_file_location(source.stem, source)
    adapter = module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def prepare_guide(mode, source, weights, to_pil):
    if mode == "Canny":
        from comfy_extras.nodes_canny import Canny
        return Canny.detect_edge(source, .1, .4)[0].cpu().float()

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    weights = weights / "depth-anything-v2-large"
    mm.unload_all_models()
    device = mm.get_torch_device()
    processor = AutoImageProcessor.from_pretrained(weights, local_files_only=True)
    estimator = AutoModelForDepthEstimation.from_pretrained(weights, local_files_only=True).to(device).eval()
    inputs = processor(images=to_pil(source), return_tensors="pt").to(device)
    predicted = estimator(**inputs).predicted_depth.unsqueeze(1)
    depth = F.interpolate(predicted, size=source.shape[1:3], mode="bilinear", align_corners=False)[0, 0].cpu()
    depth = (depth - depth.min()) / (depth.max() - depth.min()).clamp_min(1e-8)
    return depth[None, :, :, None].expand(-1, -1, -1, 3).contiguous()


def prepare_control(mode, strength, source, model, vae, final, work, tiles, patch, to_pil):
    root = Path(__file__).resolve().parent.parent
    if not (root / "controls/setup-complete").is_file():
        raise FileNotFoundError("Run setup_controls.sh on the test runtime for Depth or Canny.")
    weights = root / "weights"
    folder_paths.add_model_folder_path("loras", str(weights / "loras"), is_default=True)
    guide = prepare_guide(mode, source, weights, to_pil)
    frame = comfy.utils.common_upscale(guide.movedim(-1, 1), *final, "bicubic", "disabled").clamp_(0, 1)
    frame = F.pad(frame, (0, work[0] - final[0], 0, work[1] - final[1]), mode="replicate")
    frame = frame.movedim(1, -1)
    raw_hints = {}
    for tile in tiles:
        x1, y1, x2, y2 = (v * vision.LATENT_SCALE for v in (tile.left, tile.top, tile.right, tile.bottom))
        raw_hints[tile.top // patch, tile.left // patch] = vae.encode(frame[:, y1:y2, x1:x2, :].contiguous()).cpu()

    if mode == "Depth":
        facok = load_adapter(root / "controls/facok/nodes.py")
        controlled, = facok.Krea2ControlLoRALoader().load_lora(model, "depth-control-lora.safetensors", strength)
        apply_depth, = controlled.get_wrappers(WrappersMP.DIFFUSION_MODEL, facok.WRAPPER_KEY)
        controlled.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, facok.WRAPPER_KEY)
        hints = {pos: facok._process_control_latent_for_model(controlled, value)
                 for pos, value in raw_hints.items()}

        def apply(executor, x, timesteps, context, mask, refs, options, hint, **inner):
            options = {**options, facok.CONTROL_LATENT_KEY: hint}
            return apply_depth(executor, x, timesteps, context, mask, refs, options, **inner)
    else:
        nk2e = load_adapter(root / "controls/nk2e/nk2e_nodes.py")
        controlled, = LoraLoaderModelOnly().load_lora_model_only(model, "NK2E-canny-v0.1.safetensors", strength)
        hints = {pos: controlled.model.process_latent_in(value) for pos, value in raw_hints.items()}

        def apply(executor, x, timesteps, context, mask, refs, options, hint, **inner):
            return nk2e._nk2e_incontext_forward(executor.class_obj, x, timesteps, context, options, hint)

    class GuidedVision(vision.TiledVision):
        calls = 0

        def __call__(self, executor, *args, **kwargs):
            def execute_tile(x, timesteps, context, mask, refs, options, **inner):
                hint = hints[options[vision.CANVAS_ORIGIN]]
                self.calls += 1
                return apply(executor, x, timesteps, context, mask, refs, options, hint, **inner)
            return super().__call__(execute_tile, *args, **kwargs)

    return controlled, GuidedVision, guide
