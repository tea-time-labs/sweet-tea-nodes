# Sweet Tea Nodes

ComfyUI custom node pack containing:

- `Ultimate SD Upscale (Pose Tiled ControlNet)` (`UltimateSDUpscalePoseTiled`)
- `Wan VACE -> Video (dual strength)` (`WanVaceToVideoPlus`)
- `Wan VACE -> Video (caps inpaint)` (`WanVaceToVideoCapsInpaint`)

## Source Provenance

This repository packages node code synced from the `sweettea` R2 bucket paths:

- `custom_nodes/usdu_pose_tiled/*`
- `custom_nodes/wan_vace_dual_strength.py`
- `custom_nodes/wan_vace_caps_inpaint.py`

## Install (ComfyUI)

1. Open ComfyUI Manager.
2. Search for `Sweet Tea Nodes` in the registry (after publish), then install.
3. Restart ComfyUI.

Manual install from Git:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/tea-time-labs/sweet-tea-nodes.git
```

## Local Validation Commands

Run from repo root:

```bash
python3 -m py_compile __init__.py wan_vace_dual_strength.py wan_vace_caps_inpaint.py usdu_pose_tiled/*.py
comfy --skip-prompt --here node validate
comfy --skip-prompt --here node pack
```

## Notes

- `WanVaceToVideoPlus` is intentionally hardcoded for `length=81` and will raise an error for other lengths.
- `WanVaceToVideoCapsInpaint` supports variable lengths and cap sizes.
