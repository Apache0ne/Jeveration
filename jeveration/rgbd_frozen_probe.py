"""Small-data geometry readouts from a frozen Jeveration decoder.

Original adaptation of LiDeRe/Probe3D/few-shot prototypes and guided filtering;
NOT an implementation of AnyUp or a reproduction of the referenced papers.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

SIZE=128
CLASSES=13

def box_mean(x,radius=2):
    if radius<0: raise ValueError("negative radius")
    if radius==0:return x
    return F.avg_pool2d(F.pad(x,(radius,)*4,mode="replicate"),2*radius+1,stride=1)

def luminance(rgb):
    return rgb[:,0:1]*.299+rgb[:,1:2]*.587+rgb[:,2:3]*.114

class FrozenReadoutFeatures(nn.Module):
    """Fixed projection of 32x32 decoder state plus image/color/edge/position."""
    def __init__(self,width,projected=48,seed=42):
        super().__init__()
        if width<1 or projected<4:raise ValueError("invalid dimensions")
        self.width=int(width);self.projected=int(projected)
        g=torch.Generator(device="cpu").manual_seed(seed)
        random=torch.randn(width,projected,generator=g)
        mat=(torch.linalg.qr(random,mode="reduced").Q.T
             if width>=projected else F.normalize(random.T,dim=1))
        self.register_buffer("projection",mat.float().contiguous())
        self.register_buffer("mean",torch.zeros(width))
        self.register_buffer("inv_std",torch.ones(width))
        self.channels=projected+3+3+2+2+1

    @torch.no_grad()
    def fit_statistics(self,dataset):
        n=0;total=torch.zeros(self.width,dtype=torch.float64)
        sq=torch.zeros(self.width,dtype=torch.float64)
        for sample in dataset:
            v=sample["features"].double().reshape(self.width,-1)
            total+=v.sum(1);sq+=v.square().sum(1);n+=v.shape[1]
        if n<2:raise RuntimeError("No training features")
        mean=total/n
        var=(sq/n-mean.square()).clamp_min(1e-6)
        self.mean.copy_(mean.float());self.inv_std.copy_(var.rsqrt().clamp(max=100).float())
        return {"feature_samples":n,"min_variance":float(var.min())}

    def forward(self,features,rgb):
        if features.ndim!=4 or tuple(features.shape[1:])!=(self.width,32,32):
            raise ValueError(f"Expected features [B,{self.width},32,32], got {features.shape}")
        if rgb.ndim!=4 or tuple(rgb.shape[1:])!=(3,512,512):
            raise ValueError(f"Expected RGB [B,3,512,512], got {rgb.shape}")
        v=(features.float()-self.mean[None,:,None,None])*self.inv_std[None,:,None,None]
        v=F.conv2d(v,self.projection[:,:,None,None])
        v=F.interpolate(v,size=(SIZE,SIZE),mode="bilinear",align_corners=False)
        color=F.interpolate(rgb.float(),size=(SIZE,SIZE),mode="area").clamp(0,1)
        smooth=box_mean(color,3);lum=luminance(color)
        gx=torch.zeros_like(lum);gy=torch.zeros_like(lum)
        gx[:,:,:,1:-1]=.5*(lum[:,:,:,2:]-lum[:,:,:,:-2])
        gy[:,:,1:-1,:]=.5*(lum[:,:,2:,:]-lum[:,:,:-2,:])
        uv=torch.linspace(-1,1,SIZE,device=v.device,dtype=v.dtype)
        y,x=torch.meshgrid(uv,uv,indexing="ij")
        xy=torch.stack((x,y))[None].expand(v.shape[0],-1,-1,-1)
        variance=(box_mean(lum.square(),2)-box_mean(lum,2).square()).clamp_min(0)
        return F.normalize(torch.cat((v,color,smooth,gx,gy,xy,variance),1),dim=1,eps=1e-6)

class DepthDistribution(nn.Module):
    """Log-depth bins, with analytic Gaussian-logit initialization from ridge."""
    def __init__(self,channels,bins=48,min_depth=.1,max_depth=80.):
        super().__init__()
        self.linear=nn.Conv2d(channels,bins,1)
        self.register_buffer("centers",torch.linspace(math.log(min_depth),math.log(max_depth),bins))
        self.temperature=.12
        self.local=nn.Sequential(nn.Conv2d(channels,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,1,1))
        nn.init.zeros_(self.local[-1].weight);nn.init.zeros_(self.local[-1].bias)

    def forward(self,z):
        logits=self.linear(z).float()
        prob=torch.softmax(logits,1)
        log_d=(prob*self.centers[None,:,None,None]).sum(1)
        return {"log_depth":log_d+.15*torch.tanh(self.local(z)[:,0].float()),
                "bin_logits":logits}

    @torch.no_grad()
    def init_ridge(self,weight,bias):
        c=self.centers
        self.linear.weight.copy_((2*c[:,None]/self.temperature*weight[None,:])[:,:,None,None])
        self.linear.bias.copy_((2*c*bias-c.square())/self.temperature)

class NormalProbe(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.linear=nn.Conv2d(channels,3,1)
        self.local=nn.Sequential(nn.Conv2d(channels,12,3,padding=1),nn.SiLU(),nn.Conv2d(12,3,1))
        nn.init.zeros_(self.local[-1].weight);nn.init.zeros_(self.local[-1].bias)

    def forward(self,z):
        return F.normalize((self.linear(z)+.12*torch.tanh(self.local(z))).float(),dim=1,eps=1e-6)

    @torch.no_grad()
    def init_ridge(self,weight,bias):
        self.linear.weight.copy_(weight[:,:,None,None]);self.linear.bias.copy_(bias)

class SemanticPrototypes(nn.Module):
    def __init__(self,channels,num_classes=CLASSES):
        super().__init__()
        self.linear=nn.Conv2d(channels,num_classes,1)
        self.local=nn.Sequential(nn.Conv2d(channels,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,num_classes,1))
        nn.init.zeros_(self.local[-1].weight);nn.init.zeros_(self.local[-1].bias)

    def forward(self,z):
        return self.linear(z).float()+.25*self.local(z).float()

    @torch.no_grad()
    def init_prototypes(self,prototypes,counts):
        p=F.normalize(prototypes.float(),dim=1,eps=1e-8)
        self.linear.weight.copy_(12*p[:,:,None,None])
        prior=(counts.float()+1)/(counts.sum()+counts.numel())
        bias=prior.log().clamp(-12,-1);bias[counts==0]=-12
        self.linear.bias.copy_(bias)

class RGBDProbe(nn.Module):
    def __init__(self,width,projected=48,seed=42):
        super().__init__()
        self.encoder=FrozenReadoutFeatures(width,projected,seed)
        c=self.encoder.channels
        self.depth=DepthDistribution(c)
        self.normal=NormalProbe(c)
        self.seg=SemanticPrototypes(c)
        for p in self.encoder.parameters():p.requires_grad=False

    def feature_map(self,features,rgb):return self.encoder(features,rgb)
    def forward_low(self,z):
        return {"depth":self.depth(z),"normal":self.normal(z),"seg":self.seg(z)}
    def forward(self,features,rgb):
        return self.forward_low(self.feature_map(features,rgb))

def fast_guided_upsample(low,rgb,radius=2,eps=.012):
    """Local grayscale guided filter, not the AnyUp neural architecture."""
    if tuple(low.shape[-2:])!=(SIZE,SIZE) or tuple(rgb.shape[-2:])!=(512,512):
        raise ValueError("Expected 128x128 low and 512x512 RGB")
    ih=luminance(rgb.float())
    il=F.interpolate(ih,size=(SIZE,SIZE),mode="area")
    mi=box_mean(il,radius);mp=box_mean(low.float(),radius)
    a=(box_mean(il*low.float(),radius)-mi*mp)/(
        (box_mean(il.square(),radius)-mi.square()).clamp_min(0)+eps)
    b=mp-a*mi
    ah=F.interpolate(box_mean(a,radius),size=(512,512),mode="bilinear",align_corners=False)
    bh=F.interpolate(box_mean(b,radius),size=(512,512),mode="bilinear",align_corners=False)
    return ah*ih+bh

def predictions_512(out,rgb):
    depth=fast_guided_upsample(out["depth"]["log_depth"][:,None],rgb)[:,0].clamp(-5,5).exp()
    normals=F.normalize(fast_guided_upsample(out["normal"],rgb).float(),dim=1,eps=1e-6)
    logits=fast_guided_upsample(out["seg"],rgb)
    return {"depth":depth,"normal":normals.permute(0,2,3,1),
            "seg":logits.argmax(1),"seg_logits":logits}

@torch.inference_mode()
def generate_rgbd(pipeline,probe,prompts,generator=None):
    """RGB+geometry from one frozen Jeveration image-decoder pass."""
    from .rgbd_sidecar import tap_decoder
    from .images import to_uint8
    if pipeline.codec is None or pipeline.codec.decoder is None:
        raise RuntimeError("Load the plain Jeveration decoder first")
    ids,mask=pipeline.ptok(prompts)
    codes=pipeline.model.sample_codes(ids.to(pipeline.device),mask.to(pipeline.device),
                                      generator,heads_in_autocast=True)
    with tap_decoder(pipeline.codec.decoder) as tap:
        rgb=pipeline.codec.decode(codes,prompts).float().clamp(0,1)
    out=predictions_512(probe(tap["features"],rgb),rgb)
    return {"rgb":to_uint8(rgb),"depth":out["depth"].cpu().numpy(),
            "normal":out["normal"].cpu().numpy(),"seg":out["seg"].cpu().numpy()}
