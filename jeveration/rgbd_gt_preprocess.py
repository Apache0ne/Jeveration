"""Official TA-TiTok image-encoder preprocessing.

The upstream TA-TiTok README_MaskGen.md reconstruction example passes
torch.float32 RGB image / 255.0 directly to tokenizer.encode(...).
It does NOT normalize [0,1] pixels to [-1,1].
"""
import torch
import torch.nn.functional as F


def tokenizer_pixels(rgb: torch.Tensor) -> torch.Tensor:
    """Convert Jeveration's [B,3,512,512] RGB [0,1] to TA-TiTok's 256px [0,1]."""
    if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (3, 512, 512):
        raise ValueError(f"Expected [B,3,512,512] RGB, got {tuple(rgb.shape)}")
    if not torch.isfinite(rgb).all() or rgb.amin() < 0 or rgb.amax() > 1:
        raise ValueError("TA-TiTok RGB input must be finite and in [0,1]")
    return F.interpolate(
        rgb.float(), size=(256, 256), mode="bicubic", align_corners=False
    ).clamp(0.0, 1.0)
