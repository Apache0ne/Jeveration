"""Decoder-side depth/normal/segmentation head trained from Jeveration's frozen dense heads."""
from contextlib import contextmanager
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self, ni, no):
        super().__init__()
        self.main=nn.Sequential(nn.Conv2d(ni,no,3,padding=1),nn.GroupNorm(8,no),nn.SiLU(),
                                nn.Conv2d(no,no,3,padding=1),nn.GroupNorm(8,no))
        self.skip=nn.Conv2d(ni,no,1) if ni!=no else nn.Identity()
    def forward(self,x): return F.silu(self.main(x)+self.skip(x))

class RGBDHead(nn.Module):
    """Takes post-transformer [B,C,32,32] and the SAME decoder's [B,3,512,512] RGB."""
    VERSION=1
    def __init__(self, in_channels, classes=13):
        super().__init__()
        self.in_channels=int(in_channels);self.classes=int(classes)
        self.stem=nn.Sequential(nn.Conv2d(self.in_channels,128,1),nn.SiLU(),Block(128,128))
        self.up=nn.ModuleList([Block(128,96),Block(96,64),Block(64,48),Block(48,24)])
        self.rgb=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.SiLU(),
                               nn.Conv2d(16,16,3,padding=1),nn.SiLU())
        self.out=nn.Sequential(Block(40,32),nn.Conv2d(32,4+self.classes,1))
    def forward(self,features,rgb):
        if features.ndim!=4 or features.shape[1:]!=(self.in_channels,32,32):
            raise ValueError(f"Expected [B,{self.in_channels},32,32] decoder features, got {tuple(features.shape)}")
        if rgb.ndim!=4 or rgb.shape[1:]!=(3,512,512):
            raise ValueError(f"Expected RGB [B,3,512,512], got {tuple(rgb.shape)}")
        x=self.stem(features.float())
        for stage in self.up:
            x=stage(F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False))
        raw=self.out(torch.cat((x,self.rgb(rgb.float().clamp(0,1))),dim=1))
        return dict(log_depth=raw[:,0],normal_raw=raw[:,1:4].permute(0,2,3,1),seg_logits=raw[:,4:])

def decode_maps(raw):
    return dict(depth=raw["log_depth"].clamp(-5,5).exp(),
                normal=F.normalize(raw["normal_raw"],dim=-1,eps=1e-6),
                seg=raw["seg_logits"].argmax(1))

@contextmanager
def tap_decoder(decoder):
    """Use original TA-TiTok forward, no replacement of decoder weights."""
    cache={}
    def capture(_module,inp): cache["features"]=inp[0]
    hook=decoder.ffn.register_forward_pre_hook(capture)
    try:
        yield cache
        if "features" not in cache: raise RuntimeError("TA-TiTok decoder feature hook was not reached")
    finally: hook.remove()

def save_head(path,head,**metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(version=head.VERSION,in_channels=head.in_channels,classes=head.classes,
        state_dict={k:v.detach().cpu() for k,v in head.state_dict().items()},metadata=metadata),path)

def load_head(path,device="cuda"):
    s=torch.load(path,map_location="cpu",weights_only=True)
    if s["version"]!=RGBDHead.VERSION: raise RuntimeError("Unsupported RGBD head version")
    h=RGBDHead(s["in_channels"],s["classes"])
    h.load_state_dict(s["state_dict"],strict=True)
    return h.to(device).eval().requires_grad_(False)
