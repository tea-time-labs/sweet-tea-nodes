from .nodes_pose_tiled import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .krea2_control import NODE_CLASS_MAPPINGS as KREA2_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as KREA2_NAMES

NODE_CLASS_MAPPINGS.update(KREA2_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(KREA2_NAMES)

from .krea2_guided_upscale import STKrea2GuidedUpscale

NODE_CLASS_MAPPINGS["STKrea2GuidedUpscale"] = STKrea2GuidedUpscale
NODE_DISPLAY_NAME_MAPPINGS["STKrea2GuidedUpscale"] = "Krea2 Guided Ultimate Upscale (Sweet Tea)"
