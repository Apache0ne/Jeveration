"""Few-shot geometry training on EXISTING 95/20 Jeveration cache.

Analytic ridge/prototype initialization, three private readouts, 1200 steps,
mask-aware supervision, local SSD staging, Drive snapshots and 20-image report.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,random,zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm
from .rgbd_gt_v2_train import GeometryCache
from .rgbd_gt_stage import stage_groundtruth_cache
from .rgbd_frozen_probe import RGBDProbe,SIZE,CLASSES,predictions_512
from .rgbd_tracking import atomic_json,atomic_torch_save,checkpoint_dir,save_steps

TRAIN_COUNT,VAL_COUNT,TOTAL_STEPS=95,20,1200

def check_splits(root):
    root=Path(root);names=[]
    for split,amount in (("train",TRAIN_COUNT),("val",VAL_COUNT)):
        paths=sorted((root/"cache"/split).glob("*.npz"))
        if len(paths)!=amount:raise RuntimeError(f"Expected {amount} {split} examples; found {len(paths)}")
        names.extend([f"{split}/{p.name}" for p in paths])
    manifest=(root/"source_manifest.json").read_bytes()
    return hashlib.sha256(manifest+b"\n"+"\n".join(names).encode()).hexdigest()

def to_batch(sample,device):
    return {k:v.unsqueeze(0).to(device,non_blocking=True)
            for k,v in sample.items() if torch.is_tensor(v) and k!="alignment_ssim"}

def data128(b):
    """Area targets with strict validity masks and correspondence confidence."""
    d=b["depth"].float().unsqueeze(1)
    v=b["depth_valid"].float().unsqueeze(1)
    coverage=F.adaptive_avg_pool2d(v,(SIZE,SIZE))
    depth=F.adaptive_avg_pool2d(d*v,(SIZE,SIZE))/coverage.clamp_min(1e-4)
    n=b["normal"].float().permute(0,3,1,2)
    nv=b["normal_valid"].float().unsqueeze(1)
    ncov=F.adaptive_avg_pool2d(nv,(SIZE,SIZE))
    normal=F.normalize(F.adaptive_avg_pool2d(n*nv,(SIZE,SIZE))/ncov.clamp_min(1e-4),dim=1)
    seg=F.interpolate(b["seg"][:,None].float(),size=(SIZE,SIZE),mode="nearest")[:,0].long()
    sv=F.interpolate(b["seg_valid"][:,None].float(),size=(SIZE,SIZE),mode="nearest")[:,0]>.5
    dv=(coverage[:,0]>.98)&(depth[:,0]>.05)&(depth[:,0]<80)
    orig=b.get("original_rgb",b["rgb"])
    original=F.interpolate(orig.float(),(SIZE,SIZE),mode="area")
    generated=F.interpolate(b["rgb"].float(),(SIZE,SIZE),mode="area")
    conf=torch.exp(-(original-generated).square().mean(1)/.025).clamp(.05,1.)
    return {"depth":depth[:,0],"normal":normal,"seg":seg,
            "dv":dv,"nv":ncov[:,0]>.98,"sv":sv&(seg!=255),"confidence":conf}

def linear_ridge(x,y,weights,reg=5e-3):
    """Closed-form weighted ridge; never penalize the bias coordinate."""
    x=x.double();y=y.double();weights=weights.double().clamp_min(0)
    if x.ndim!=2 or y.ndim!=2 or y.shape[0]!=x.shape[0]:
        raise ValueError("Incorrect ridge shapes")
    x=torch.cat((x,torch.ones(x.shape[0],1,dtype=x.dtype,device=x.device)),1)
    gram=x.T@(x*weights[:,None]);xy=x.T@(y*weights[:,None])
    penalty=reg*torch.eye(gram.shape[0],dtype=gram.dtype,device=gram.device)
    penalty[-1,-1]=0
    result=torch.linalg.solve(gram+penalty,xy).float()
    return result[:-1].T.contiguous(),result[-1].contiguous()

def analytic_init(model,ds,device,stride=4,reg=5e-3):
    """Train-only least squares initialization; no validation label leakage."""
    model.encoder.fit_statistics(ds)
    model.to(device)
    width=model.encoder.channels
    dx=[];dy=[];dw=[];nx=[];ny=[];nw=[]
    proto=torch.zeros(CLASSES,width,dtype=torch.float64)
    counts=torch.zeros(CLASSES,dtype=torch.float64)
    print("[probe] Initializing readouts analytically from cached TRAIN geometry",flush=True)
    with torch.no_grad():
        for i in tqdm(range(len(ds)),desc="Ridge/prototypes"):
            b=to_batch(ds[i],device)
            z=model.feature_map(b["features"],b["rgb"])
            gt=data128(b)
            z=z[0,:,::stride,::stride].permute(1,2,0).reshape(-1,width).cpu()
            depth=gt["depth"][0,::stride,::stride].flatten().cpu()
            normal=gt["normal"][0,:,::stride,::stride].permute(1,2,0).reshape(-1,3).cpu()
            valid_d=gt["dv"][0,::stride,::stride].flatten().cpu()
            valid_n=gt["nv"][0,::stride,::stride].flatten().cpu()
            valid_s=gt["sv"][0,::stride,::stride].flatten().cpu()
            seg=gt["seg"][0,::stride,::stride].flatten().cpu()
            confidence=gt["confidence"][0,::stride,::stride].flatten().cpu()
            if valid_d.any():
                dx.append(z[valid_d]);dy.append(depth[valid_d].clamp(.1,80).log()[:,None]);dw.append(confidence[valid_d])
            if valid_n.any():
                nx.append(z[valid_n]);ny.append(normal[valid_n]);nw.append(confidence[valid_n])
            for c in range(CLASSES):
                m=valid_s&(seg==c)
                if m.any():
                    proto[c]+=z[m].double().sum(0);counts[c]+=m.sum()
    if not dx or not nx:raise RuntimeError("No valid geometry pixels")
    w,b=linear_ridge(torch.cat(dx),torch.cat(dy),torch.cat(dw),reg)
    model.depth.init_ridge(w[0],b[0])
    w,b=linear_ridge(torch.cat(nx),torch.cat(ny),torch.cat(nw),reg)
    model.normal.init_ridge(w,b)
    model.seg.init_prototypes((proto/counts[:,None].clamp_min(1)).float().to(device),
                              counts.float().to(device))
    return {"ridge_depth_rows":sum(map(len,dx)),"ridge_normal_rows":sum(map(len,nx)),
            "prototype_counts":counts.long().tolist(),"ridge_reg":reg,"sample_stride":stride}

def task_training_loss(task,pred,gt,model,semantic_weights):
    """One task gradient path at a time. No shared trainable trunk."""
    if task=="depth":
        y=gt["depth"].clamp(.1,80).log();w=gt["confidence"]*gt["dv"]
        metric=(F.smooth_l1_loss(pred["log_depth"].float(),y,reduction="none",beta=.18)*w).sum()/w.sum().clamp_min(1)
        bins=model.depth.centers
        t=((y-bins[0])/(bins[1]-bins[0])).clamp(0,len(bins)-1-1e-5)
        lo=t.floor().long();hi=(lo+1).clamp_max(len(bins)-1);alpha=t-lo.float()
        logp=F.log_softmax(pred["bin_logits"].float(),1)
        ce=-((1-alpha)*logp.gather(1,lo[:,None])[:,0]+alpha*logp.gather(1,hi[:,None])[:,0])
        return metric+.05*(ce*w).sum()/w.sum().clamp_min(1)
    if task=="normal":
        w=gt["confidence"]*gt["nv"]
        cosine=1-(pred.float()*gt["normal"]).sum(1).clamp(-1,1)
        return (cosine*w).sum()/w.sum().clamp_min(1)
    if task=="seg":
        w=gt["confidence"]*gt["sv"]
        pix=F.cross_entropy(pred.float(),gt["seg"],weight=semantic_weights,ignore_index=255,reduction="none")
        wgt=semantic_weights[gt["seg"].clamp(0,CLASSES-1)]
        return (pix*w).sum()/(wgt*w).sum().clamp_min(1)
    raise ValueError(task)

