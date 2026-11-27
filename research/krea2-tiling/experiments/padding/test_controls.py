import importlib.util
import os
from pathlib import Path
import sys
from unittest.mock import patch

# Import the same run harness without rerunning its other cases.
os.environ["ST_KREA_EXPERIMENT_RUNS"] = "controls_only"
import experiment as exp
from experiment import model as base_model, mm, nodes, out, root, source, torch, vae, vision
from usdu_pose_tiled.krea2_control import crop_hint
from comfy.patcher_extension import WrappersMP

import folder_paths

folder_paths.add_model_folder_path("loras", str(root / "weights" / "loras"), is_default=True)

def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

facok = load_file("st_test_facok", root / "controls" / "facok" / "nodes.py")
nk2e = load_file("st_test_nk2e", root / "controls" / "nk2e" / "nk2e_nodes.py")
NativeVision = vision.TiledVision

def prepare_hint(kind):
    if kind == "canny":
        from comfy_extras.nodes_canny import Canny
        return Canny.detect_edge(source, .1, .4)[0].cpu().float()
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    model_path = root / "weights" / "depth-anything-v2-large"
    preprocessor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    estimator = AutoModelForDepthEstimation.from_pretrained(model_path, local_files_only=True).to("cuda").eval()
    inputs = preprocessor(images=exp.pil(source), return_tensors="pt").to("cuda")
    with torch.no_grad():
        predicted = estimator(**inputs).predicted_depth.unsqueeze(1)
        depth = torch.nn.functional.interpolate(predicted, size=source.shape[1:3],
                                                mode="bilinear", align_corners=False)[0, 0].cpu()
    depth = ((depth - depth.min()) / (depth.max() - depth.min()).clamp_min(1e-8))
    return depth[None, :, :, None].expand(-1, -1, -1, 3).contiguous()

def test_control(kind):
    case_name = f"p256_grounded_local_{kind}"
    if exp.report["runs"].get(case_name, {}).get("status") == "ok":
        return
    guidance = prepare_hint(kind)
    exp.png(guidance, out / f"{kind}_guide.png")
    mm.soft_empty_cache()
    tiles = vision._tiles(3072, 3072, 1024, 1024, 256, 2, exp.layout)
    hints = {}
    for t in tiles:
        pixel_box = (t.left * 8, t.top * 8, t.right * 8, t.bottom * 8)
        x1, y1, x2, y2 = pixel_box
        aligned = crop_hint(guidance, pixel_box, (3072, 3072), (x2-x1, y2-y1))
        raw = vae.encode(aligned).cpu()
        hints[(t.top // 2, t.left // 2)] = raw
    exp.announce("Aligned guide latents", kind, [tuple(x.shape) for x in hints.values()])

    if kind == "depth":
        controlled, = facok.Krea2ControlLoRALoader().load_lora(base_model, "depth-control-lora.safetensors", .7)
        (apply_depth,) = controlled.get_wrappers(WrappersMP.DIFFUSION_MODEL, facok.WRAPPER_KEY)
        controlled.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, facok.WRAPPER_KEY)
        hints = {key: facok._process_control_latent_for_model(controlled, value)
                 for key, value in hints.items()}
    else:
        controlled, = nodes.LoraLoaderModelOnly().load_lora_model_only(base_model, "NK2E-canny-v0.1.safetensors", .7)
        hints = {key: controlled.model.process_latent_in(value) for key, value in hints.items()}

    calls = 0
    class ControlVision(NativeVision):
        def __call__(self, executor, *args, **kwargs):
            def execute_tile(x, timesteps, context, attention_mask, ref_latents, options, **inner):
                nonlocal calls
                hint = hints[options[vision.CANVAS_ORIGIN]]
                calls += 1
                if kind == "depth":
                    options = {**options, facok.CONTROL_LATENT_KEY: hint}
                    return apply_depth(executor, x, timesteps, context, attention_mask,
                                       ref_latents, options, **inner)
                return nk2e._nk2e_incontext_forward(executor.class_obj, x, timesteps, context, options, hint)
            return super().__call__(execute_tile, *args, **kwargs)

    # The control is applied inside the tiled wrapper using the same bounds and sampler.
    exp.model = controlled
    with patch.object(vision, "TiledVision", ControlVision):
        exp.run(case_name, 256, True, "local")
    row = exp.report["runs"][case_name]
    if row["status"] == "ok" and not row["recovered_latent"]:
        assert calls == 72, f"Expected 72 controlled tile evaluations, got {calls}"
    row.update(control=kind, control_lora_strength=.7, verified_control_calls=calls)
    exp.write_report()
    exp.announce("CONTROL RESULT", row)
    exp.model = base_model
    mm.unload_all_models()

for mode in os.environ.get("ST_KREA_CONTROL_MODE", "depth,canny").split(","):
    test_control(mode)
