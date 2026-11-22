"""Jeveration model and optional training helpers.

Do not import Qwen, transformers or flash-linear-attention merely to import
jeveration.rgbd_gt_train. Full-model inference is loaded on demand.
"""
__all__ = ["JeverationPipeline"]

def __getattr__(name):
    if name == "JeverationPipeline":
        from .pipeline import JeverationPipeline
        return JeverationPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
