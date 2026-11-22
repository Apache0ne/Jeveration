"""Frozen public TA-TiTok 256-px encoder -> actual Jeveration decoder codes.

No Qwen text-generation pass is used when preparing ground-truth paired frames.
The decoder reconstructs the 512px image from encoded real RGB. Every sample
is checked for structural alignment BEFORE its true 3D labels are used.
"""
from __future__ import annotations
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import config
from .tokenizer import ImageCodec, _fetch
from .rgbd_gt_preprocess import tokenizer_pixels
from modeling.modules.blocks import TiTokEncoder


class FrozenTATiTokEncoder(nn.Module):
    """The same TA-TiTok VQ-128 encoder and 8192-way codebook used by Jeveration."""
    def __init__(self,codec:ImageCodec,source=None):
        super().__init__()
        from omegaconf import OmegaConf
        from safetensors import safe_open
        cfg=deepcopy(codec.cfg)
        # The tokenizer encoder is pretrained at 256px; the Jeveration
        # fine-tuned decoder is 512px. These resolutions must not be confused.
        cfg.dataset.preprocessing.crop_size=256
        self.encoder=TiTokEncoder(cfg)
        self.latent_tokens=nn.Parameter(torch.zeros(
            config.NUM_CODES,self.encoder.width),requires_grad=False)
        filename=_fetch(source,config.TATITOK,"model.safetensors")
        with safe_open(filename,framework="pt",device="cpu") as f:
            keys=set(f.keys())
            prefix="encoder."
            weights={key[len(prefix):]:f.get_tensor(key)
                     for key in keys if key.startswith(prefix)}
            if not weights:raise RuntimeError("Tokenizer checkpoint lacks encoder.* weights")
            missing,unexpected=self.encoder.load_state_dict(weights,strict=False)
            if missing or unexpected:
                raise RuntimeError(f"Tokenizer encoder mismatch: missing={missing}, extra={unexpected}")
            if "latent_tokens" not in keys:
                raise RuntimeError("Tokenizer checkpoint lacks latent_tokens")
            self.latent_tokens.data.copy_(f.get_tensor("latent_tokens"))
            original=f.get_tensor("quantize.embedding.weight")
        if not torch.allclose(original.float(),codec.codebook.detach().cpu().float(),atol=1e-5):
            raise RuntimeError("Encoder and decoder codebooks do not match exactly")
        self.register_buffer("codebook",original.float())
        self.l2_norm=bool(cfg.model.vq_model.get("use_l2_norm",False))
        self.requires_grad_(False).eval()

    @torch.inference_mode()
    def forward(self,rgb):
        """Input actual RGB [B,3,512,512] in [0,1], output token ids [B,128]."""
        if rgb.ndim!=4 or rgb.shape[1:]!=(3,512,512):
            raise ValueError(f"Expected [B,3,512,512], got {tuple(rgb.shape)}")
        # Official TA-TiTok example passes float RGB / 255.0: values [0,1].
        # The old [-1,1] normalization corrupted VQ IDs and scene alignment.
        x=tokenizer_pixels(rgb)
        z=self.encoder(x.to(self.latent_tokens.dtype),self.latent_tokens)
        # [B,C,1,128] -> [B,128,C], nearest public VQ code.
        z=z.squeeze(2).transpose(1,2).float()
        code=self.codebook.float()
        if self.l2_norm:
            z=F.normalize(z,dim=-1)
            code=F.normalize(code,dim=-1)
        d=(z.square().sum(-1,keepdim=True)
           +code.square().sum(-1)[None,None,:]
           -2*torch.matmul(z,code.T))
        ids=d.argmin(-1)
        if ids.shape[1]!=config.NUM_CODES:raise RuntimeError(f"Code count mismatch {ids.shape}")
        return ids


def image_alignment(original,recon):
    """PSNR and global SSIM between REAL image and its teacher-forced decode."""
    from skimage.metrics import structural_similarity
    a=np.asarray(original,dtype=np.uint8)
    b=np.asarray(recon,dtype=np.uint8)
    if a.shape!=b.shape or a.shape!=(512,512,3):
        raise ValueError(f"Unmatched images {a.shape} and {b.shape}")
    mse=np.mean((a.astype(np.float64)-b.astype(np.float64))**2)
    psnr=10*np.log10(255**2/max(mse,1e-12))
    ssim=structural_similarity(a,b,channel_axis=2,data_range=255)
    return dict(psnr=float(psnr),ssim=float(ssim))
