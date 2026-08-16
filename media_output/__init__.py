"""Temporary media output nodes for Sweet Tea Studio transports."""

from .nodes_media_output import SweetTeaExecutionReceipt, SweetTeaPreviewImage, SweetTeaPreviewVideo

NODE_CLASS_MAPPINGS = {
    "SweetTeaPreviewImage": SweetTeaPreviewImage,
    "SweetTeaPreviewVideo": SweetTeaPreviewVideo,
    "SweetTeaExecutionReceipt": SweetTeaExecutionReceipt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SweetTeaPreviewImage": "Sweet Tea Preview Image",
    "SweetTeaPreviewVideo": "Sweet Tea Preview Video",
    "SweetTeaExecutionReceipt": "Sweet Tea Execution Receipt",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SweetTeaPreviewImage",
    "SweetTeaPreviewVideo",
    "SweetTeaExecutionReceipt",
]
