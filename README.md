# Sweet Tea Nodes

ComfyUI node pack focused on high-control video generation and high-fidelity tiled upscaling.

## Node Highlights

- `Ultimate SD Upscale (Pose Tiled ControlNet)` (`UltimateSDUpscalePoseTiled`)
  - Major tiled-upscaling upgrade with lockable padding, ControlNet tile alignment fixes, normalized overlap blending, and seam-fix passes designed to remove bleed and ghost offsets.
- `Wan VACE -> Video (caps inpaint)` (`WanVaceToVideoCapsInpaint`)
  - Temporal cap-stitching node that treats start/end caps as known regions and forces middle-region generation with explicit inpaint mask semantics.
  - Built for precise clip stitching and transition reconstruction workflows.
- `Wan VACE -> Video (hybrid subject swap)` (`WanVaceToVideoHybridSubjectSwap`)
  - Depth-first hybrid subject-swap node for Wan VACE workflows.
  - Combines structural control with a character reference and optional source-RGB luma/detail/background contexts for controllable illustrated/live-action blending.
- `Sweet Tea Preview Video` (`SweetTeaPreviewVideo`)
  - No-save output sink for ComfyUI's native `VIDEO` type.
  - Exposes an existing temp-backed video in place, or byte-copies an otherwise unservable source into Comfy temp, without decoding or re-encoding its video or audio streams.
- `Sweet Tea Execution Receipt` (`SweetTeaExecutionReceipt`)
  - No-save metadata sink for allowlisted external API execution facts.
  - Publishes the provider, request, endpoint, operation, and optional estimated cost for Sweet Tea Studio provenance without exposing API credentials.

## Source Provenance

This repository packages node code synced from the `sweettea` R2 bucket paths:

- `custom_nodes/usdu_pose_tiled/*`
- `custom_nodes/wan_vace_caps_inpaint/*`
- `custom_nodes/wan_vace_hybrid_subject/*`

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
python3 -m unittest tests.test_wan_vace_hybrid_subject
python3 -m py_compile __init__.py wan_vace_caps_inpaint/*.py wan_vace_hybrid_subject/*.py usdu_pose_tiled/*.py
comfy --skip-prompt --here node validate
comfy --skip-prompt --here node pack
```

## Release Automation

This repo includes two GitHub workflows:

- `Validate Node Pack`: four public hybrid-subject contracts, compile checks, and `comfy node validate` on push and PR.
- `Release and Publish`: automatic semantic version bump from commit history, git tag creation, GitHub release creation, and Comfy Registry publish.

Autoversion rules in `main` pushes:

- `BREAKING CHANGE` or `type!:` -> major bump
- `feat:` -> minor bump
- everything else -> patch bump

Required secret for registry publish:

- `REGISTRY_ACCESS_TOKEN`

## Notes

- `WanVaceToVideoCapsInpaint` supports variable lengths and cap sizes.
