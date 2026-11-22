"""RGBD v3: three mathematically independent decoder-side networks.

Each task gets a private feature trunk, RGB stem, upsampler, output layer,
optimizer and checkpoint. No multi-task gradients are shared. A small fusion
module consumes DETACHED predictions; it cannot update the three predictors.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from .rgbd_sidecar import Block

TASK_CHANNELS={"depth":1,"normal":3,"seg":13}
VERSION=3

class IndependentHead(nn.Module):
    def __init__(self,task,in_channels):
        super().__init__()
        if task not in TASK_CHANNELS:raise ValueError(f"Unknown head task {task}")
        self.task=task
        self.in_channels=int(in_channels)
        self.features=nn.Sequential(nn.Conv2d(in_channels,128,1),nn.SiLU(),Block(128,128))
        self.ups=nn.ModuleList([Block(128,96),Block(96,64),Block(64,48),Block(48,24)])
        self.rgb=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.SiLU(),
                               nn.Conv2d(16,16,3,padding=1),nn.SiLU())
        self.output=nn.Sequential(Block(40,32),
                                  nn.Conv2d(32,TASK_CHANNELS[task],1))
        # Depth values are log meters; normal outputs are camera-frame xyz.
        # Segmentation outputs are class logits, not softmax probabilities.

    def forward(self,features,rgb):
        if features.ndim!=4 or tuple(features.shape[1:])!=(self.in_channels,32,32):
            raise ValueError(f"{self.task}: expected features [B,{self.in_channels},32,32], got {tuple(features.shape)}")
        if rgb.ndim!=4 or tuple(rgb.shape[1:])!=(3,512,512):
            raise ValueError(f"{self.task}: expected RGB [B,3,512,512], got {tuple(rgb.shape)}")
        x=self.features(features.float())
        for block in self.ups:
            x=block(F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False))
        y=self.output(torch.cat((x,self.rgb(rgb.float().clamp(0,1))),dim=1))
        if self.task=="depth":return y[:,0]
        if self.task=="normal":return y.permute(0,2,3,1)
        return y

class GeometryFusion(nn.Module):
    """Conservative, separately learned residual in log-depth and normals.

    Input channels: log-depth (1), normalized normal (3),
    semantic softmax (13), RGB (3). Segmentation labels are NEVER modified.
    Small residual bounds discourage unfounded geometry hallucination.
    """
    def __init__(self,width=24):
        super().__init__()
        if width%8:raise ValueError("Fusion width must be divisible by 8")
        self.net=nn.Sequential(
            nn.Conv2d(20,width,3,padding=1),nn.GroupNorm(8,width),nn.SiLU(),
            Block(width,width),nn.Conv2d(width,4,3,padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,raw,rgb):
        # stop-gradient by construction: fusion parameters ONLY
        d=raw["log_depth"].detach().float()
        n=F.normalize(raw["normal_raw"].detach().float(),dim=-1,eps=1e-6)
        seg=raw["seg_logits"].detach().float().softmax(1)
        x=torch.cat((d[:,None],n.permute(0,3,1,2),seg,
                     rgb.detach().float().clamp(0,1)),dim=1)
        residual=self.net(x)
        # Bound per-pixel log-depth correction to about +/-22%.
        log_depth=d+0.2*torch.tanh(residual[:,0])
        n_delta=0.20*torch.tanh(residual[:,1:4]).permute(0,2,3,1)
        normals=F.normalize(n+n_delta,dim=-1,eps=1e-6)
        return dict(log_depth=log_depth,normal_raw=normals,
                    seg_logits=raw["seg_logits"].detach())

def independent_outputs(heads,features,rgb,detach=False):
    raw=dict(log_depth=heads["depth"](features,rgb),
             normal_raw=heads["normal"](features,rgb),
             seg_logits=heads["seg"](features,rgb))
    return {k:v.detach() for k,v in raw.items()} if detach else raw

def decoded(raw):
    return dict(depth=raw["log_depth"].float().clamp(-5,5).exp(),
                normal=F.normalize(raw["normal_raw"].float(),dim=-1,eps=1e-6),
                seg=raw["seg_logits"].argmax(1))

def save_component(path,component,task,in_channels,step,metrics=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name("."+path.name+".partial")
    torch.save(dict(version=VERSION,task=task,in_channels=in_channels,
                    model={k:v.detach().cpu() for k,v in component.state_dict().items()},
                    step=int(step),metrics=dict(metrics or {})),tmp)
    tmp.replace(path)

def load_component(path,device="cpu"):
    record=torch.load(path,map_location="cpu",weights_only=True)
    if record.get("version")!=VERSION:raise RuntimeError("Not an RGBD v3 component")
    task=record["task"]
    component=(GeometryFusion() if task=="fusion" else
               IndependentHead(task,record["in_channels"]))
    component.load_state_dict(record["model"],strict=True)
    return component.to(device).eval().requires_grad_(False)

def load_bundle(directory,device="cuda"):
    directory=Path(directory)
    heads=nn.ModuleDict({
        task:load_component(directory/f"best_{task}.pt",device)
        for task in TASK_CHANNELS
    })
    fusion_path=directory/"best_fusion.pt"
    fusion=load_component(fusion_path,device) if fusion_path.exists() else None
    return heads,fusion

class JeverationRGBDv3Pipeline:
    """MIXIN: use with JeverationPipeline for one-pass native RGBD inference."""
    def load_v3_components(self,checkpoint_dir):
        if self.codec is None or self.codec.decoder is None:
            raise RuntimeError("JeverationPipeline must load decoder before RGBDv3 heads")
        self.v3_heads,self.v3_fusion=load_bundle(checkpoint_dir,self.device)
        for task,head in self.v3_heads.items():
            if head.in_channels!=self.codec.decoder.width:
                raise RuntimeError(f"Head {task} decoder-feature width mismatch")
        return self

    @torch.no_grad()
    def text_to_rgbd_v3(self,prompts,generator=None,use_fusion=True):
        from .rgbd_sidecar import tap_decoder
        from .images import to_uint8
        if not hasattr(self,"v3_heads"):raise RuntimeError("Call load_v3_components first")
        ids,mask=self.ptok(prompts)
        codes=self.model.sample_codes(ids.to(self.device),mask.to(self.device),
                                      generator,heads_in_autocast=True)
        with tap_decoder(self.codec.decoder) as tap:
            rgb=self.codec.decode(codes,prompts)
        raw=independent_outputs(self.v3_heads,tap["features"],rgb)
        if use_fusion and self.v3_fusion is not None:
            raw=self.v3_fusion(raw,rgb)
        maps=decoded(raw)
        return dict(rgb=to_uint8(rgb),
                    depth=maps["depth"].cpu().numpy(),
                    normal=maps["normal"].cpu().numpy(),
                    seg=maps["seg"].cpu().numpy())

def make_pipeline():
    from .pipeline import JeverationPipeline
    return type("JeverationRGBDv3Pipeline", (JeverationRGBDv3Pipeline,JeverationPipeline),{})
