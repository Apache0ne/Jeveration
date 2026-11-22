"""Exact-source LiDeRe readout adapted to Jeveration's cached decoder features.

The upstream LiDeRe core is vendored verbatim in jeveration.vendor.lidere,
from timojl/lidere (MIT). DepthBinPrediction is sourced verbatim from
mbanani/probe3d (MIT). This module ONLY adapts I/O and task target dimensions,
and does not replace the core interpolation, implicit attention, or CNN.

Training follows Probe3D's loss code in jeveration.vendor.probe3d_losses.
"""
from __future__ import annotations
from torch import nn
from torch.nn import functional as F
import torch

from .vendor.lidere import LiDeRe
from .vendor.probe3d_probes import DepthBinPrediction


class CachedDecoderBackbone(nn.Module):
    """Frozen cache interface for LiDeRe's expected backbone API.

    Original cached tensors are [B,C,32,32]; LiDeRe receives
    [B,C,1,32,32] without projection or retraining the decoder.
    """
    def __init__(self, width: int):
        super().__init__()
        self.feat_dim = int(width)
        self.feat_stride = 16
        self.img_size = 512

    def feature_dim(self):
        return self.feat_dim

    def base_size(self, _img_size):
        return (32, 32)

    def freeze(self):
        pass


class SourceLiDeReRGBD(nn.Module):
    """Three separate UNMODIFIED LiDeRe readouts with upstream Probe3D depth bins.

    base_size 64 gives upstream LiDeRe 4x decoded resolution 256px.
    base_size 128 gives direct 512px, with much higher attention-memory cost.
    """
    def __init__(self, in_channels: int, dim=64, base_size=64, max_depth=80.0):
        super().__init__()
        if dim < 8 or dim % 8:
            raise ValueError("LiDeRe dim must be divisible by 8")
        if base_size not in (16,32,64,128):
            raise ValueError("base_size must be 16, 32, 64 or 128")
        self.in_channels = int(in_channels)
        self.dim = int(dim)
        self.base_size = int(base_size)
        self.max_depth = float(max_depth)
        # No shared trainable parameters; each readout has its own backbone
        # adapter with zero trainable parameters.
        self.heads = nn.ModuleDict({
            "depth": LiDeRe(dim, base_size, CachedDecoderBackbone(in_channels),
                            key_name="depth", n_classes=256, no_freeze=True),
            "normal": LiDeRe(dim, base_size, CachedDecoderBackbone(in_channels),
                             key_name="normal", n_classes=3, no_freeze=True),
            "seg": LiDeRe(dim, base_size, CachedDecoderBackbone(in_channels),
                          key_name="seg", n_classes=13, no_freeze=True)
        })
        self.depth_bins = DepthBinPrediction(
            min_depth=0.001, max_depth=max_depth, n_bins=256,
            bins_strategy="UD", norm_strategy="softmax")
        # DepthBinPrediction is parameter-free as published.

    def task(self, task, features):
        if task not in self.heads:
            raise ValueError(task)
        if features.ndim != 4 or features.shape[1:] != (
                self.in_channels,32,32):
            raise ValueError(f"Expected [B,{self.in_channels},32,32], got {tuple(features.shape)}")
        x = features.float().detach().unsqueeze(2)
        raw = self.heads[task].forward_feats(x)[task][:,:,0]
        if task == "depth":
            return self.depth_bins(raw)
        return raw

    def forward(self, features):
        return {task:self.task(task,features) for task in ("depth","normal","seg")}


def rescale_maps(out, resolution=(512,512)):
    """Probe3D: bilinear for depth, bicubic for normals; semantic bilinear."""
    depth = F.interpolate(out["depth"].float(),resolution,mode="bilinear",align_corners=True)
    normal = F.interpolate(out["normal"].float(),resolution,mode="bicubic",align_corners=True)
    normal = F.normalize(normal,dim=1,eps=1e-6)
    seg_logits = F.interpolate(out["seg"].float(),resolution,mode="bilinear",align_corners=False)
    return dict(depth=depth,normal=normal,seg_logits=seg_logits,
                seg=seg_logits.argmax(1))


@torch.inference_mode()
def text_to_rgbd(pipeline,model,prompts,generator=None):
    """Generate RGB once and decode spatial-query geometry via LiDeRe."""
    from .rgbd_sidecar import tap_decoder
    from .images import to_uint8
    if pipeline.codec is None or pipeline.codec.decoder is None:
        raise RuntimeError("Plain TA-TiTok decoder must be loaded")
    if model.in_channels != pipeline.codec.decoder.width:
        raise ValueError("Sidecar input width does not match decoder width")
    ids, mask = pipeline.ptok(prompts)
    codes = pipeline.model.sample_codes(
        ids.to(pipeline.device), mask.to(pipeline.device),
        generator, heads_in_autocast=True)
    with tap_decoder(pipeline.codec.decoder) as tap:
        rgb = pipeline.codec.decode(codes,prompts)
    maps=rescale_maps(model(tap["features"]))
    return dict(rgb=to_uint8(rgb),depth=maps["depth"][:,0].cpu().numpy(),
                normal=maps["normal"].permute(0,2,3,1).cpu().numpy(),
                seg=maps["seg"].cpu().numpy())
