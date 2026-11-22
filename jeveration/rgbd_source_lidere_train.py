"""Train source-faithful LiDeRe (CVPR 2026) on the existing 95/20 cache.

The underlying architecture is the UNMODIFIED upstream LiDeRe implementation
vendored under jeveration/vendor/lidere.py (MIT). DepthBinPrediction, DepthLoss
and angular_loss are UNMODIFIED Probe3D source (MIT). This file provides the
Hypersim cache adapter, task targets, independent AdamW training, and evaluation.

No scratch CNN alternatives, random projections, custom prototype substitutes,
or deterministic guided-filtering in this experiment.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from .rgbd_gt_v2_train import GeometryCache
from .rgbd_gt_stage import stage_groundtruth_cache
from .rgbd_source_lidere import SourceLiDeReRGBD,rescale_maps
from .vendor.probe3d_losses import DepthLoss,angular_loss
from .rgbd_tracking import atomic_json,atomic_torch_save,checkpoint_dir,save_steps
from .rgbd_frozen_probe_train import check_splits

TASKS=("depth","normal","seg")
COUNT_TRAIN=95
COUNT_VAL=20
TOTAL=1200

def move(sample,device):
    return {k:v.unsqueeze(0).to(device,non_blocking=True)
            for k,v in sample.items() if torch.is_tensor(v) and k!="alignment_ssim"}

def amp(device):
    return torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda")

def training_inputs(b,task,model):
    """Prepare original GT exactly at cached 512px; Probe3D handles resampling."""
    out=model.task(task,b["features"])
    if task=="depth":
        result=F.interpolate(out.float(),size=(512,512),mode="bilinear",align_corners=True)
        target=b["depth"].float().unsqueeze(1).clone()
        good=b["depth_valid"].bool()&(target[:,0]>.001)&(target[:,0]<=model.max_depth)
        target[:,0].masked_fill_(~good,0)
        return result,target,good
    if task=="normal":
        result=F.interpolate(out.float(),size=(512,512),mode="bicubic",align_corners=True)
        # Source Probe3D compares angular cosine of arbitrary predicted vectors,
        # so do not normalize the prediction before the published loss.
        target=b["normal"].float().permute(0,3,1,2).contiguous()
        good=b["normal_valid"].bool()
        return result,target,good
    result=F.interpolate(out.float(),size=(512,512),mode="bilinear",align_corners=False)
    target=b["seg"].long().clone()
    good=b["seg_valid"].bool()&(target!=255)
    target[~good]=255
    return result,target,good

def task_loss(task,result,target,mask,depth_fn):
    if not bool(mask.any()):
        # Entirely unlabeled frames must not poison cross-entropy or crash a
        # deterministic 95-image training run. Preserve a valid zero gradient.
        return result.float().sum()*0
    if task=="depth":
        # Exact Probe3D DepthLoss: sigmoid-style scale-invariant log error and
        # multiscale spatial gradient matching, max-depth exclusion.
        return depth_fn(result.float(),target.clone().float())
    if task=="normal":
        # Exact Probe3D angular_loss with [B,1,H,W] valid mask.
        return angular_loss(result.float(),target.float(),mask[:,None])
    return F.cross_entropy(result.float(),target,ignore_index=255)

@torch.inference_mode()
def evaluate(model,dataset,device,per_image=False):
    model.eval()
    absrel=0.;depth_count=0;angle_sum=0.;norm_count=0
    confusion=torch.zeros((13,13),dtype=torch.float64)
    rows=[]
    for i in range(len(dataset)):
        b=move(dataset[i],device)
        with amp(device):
            out=model(b["features"])
        p=rescale_maps(out)
        # Same all-valid GT mask used by v1-v3. max_depth is a *training*
        # exclusion only; do not silently make the held-out evaluation easier.
        valid_d=b["depth_valid"].bool()&(b["depth"]>0)
        valid_n=b["normal_valid"].bool()
        valid_s=b["seg_valid"].bool()&(b["seg"]!=255)
        delta=(p["depth"][:,0]-b["depth"]).abs()/b["depth"].clamp_min(.001)
        dot=(p["normal"]*F.normalize(
            b["normal"].float().permute(0,3,1,2),dim=1,eps=1e-6)).sum(1)
        angle=dot.clamp(-1+1e-6,1-1e-6).acos()*(180/math.pi)
        depth_values=delta[valid_d];normal_values=angle[valid_n]
        absrel+=float(depth_values.sum());depth_count+=int(depth_values.numel())
        angle_sum+=float(normal_values.sum());norm_count+=int(normal_values.numel())
        gt=b["seg"][valid_s].long();ps=p["seg"][valid_s].long()
        if gt.numel():
            confusion+=torch.bincount(
                (gt*13+ps).cpu(),minlength=169).reshape(13,13)
        if per_image:
            rows.append({
                "file":dataset.files[i].name,
                "depth_absrel":float(depth_values.mean()) if depth_values.numel() else None,
                "normal_deg":float(normal_values.mean()) if normal_values.numel() else None
            })
    present=confusion.sum(1)>0
    union=confusion.sum(0)+confusion.sum(1)-confusion.diag()
    miou=(confusion.diag()[present]/union[present].clamp_min(1)).mean()
    result={
        "depth_absrel":absrel/max(1,depth_count),
        "normal_deg":angle_sum/max(1,norm_count),
        "seg_acc":float(confusion.diag().sum()/confusion.sum().clamp_min(1)),
        "seg_miou":float(miou),
        "depth_valid_pixels":depth_count,
        "normal_valid_pixels":norm_count,
        "val_samples":len(dataset)
    }
    return (result,rows) if per_image else result

def trainable_state(module):
    """Do not store LiDeRe's huge deterministic spatial-attention buffers.

    Save ONLY trainable parameters. Position/attention geometry is rebuilt from
    the exact source and base_size, greatly reducing Drive I/O and file size.
    """
    return {k:p.detach().cpu().clone()
            for k,p in module.named_parameters() if p.requires_grad}

@torch.no_grad()
def load_trainable(module,weights):
    expected={k:p for k,p in module.named_parameters() if p.requires_grad}
    if set(weights)!=set(expected):
        raise RuntimeError(f"Checkpoint keys differ: {set(weights)^set(expected)}")
    for key,p in expected.items():
        if p.shape!=weights[key].shape:
            raise RuntimeError(f"Shape mismatch for {key}")
        p.copy_(weights[key].to(p.device,dtype=p.dtype))

def state_path(folder,step):
    return folder/f"trainer_source_step_{step:06d}.pt"

def checkpoint(folder,step,model,optimizers,bests,settings,manifest):
    state=dict(version="source-lidere-probe3d-v1",step=int(step),
               manifest=manifest,settings=settings,
               trainable=trainable_state(model),
               optimizers={k:v.state_dict() for k,v in optimizers.items()},
               best=bests,python_rng=random.getstate(),
               torch_rng=torch.get_rng_state(),
               cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    path=state_path(folder,step)
    atomic_torch_save(path,state)
    atomic_json(folder/"latest_source.json",{"step":step,"checkpoint":path.name})

def resume_state(folder,model,optims,settings,manifest,device):
    latest=json.loads((folder/"latest_source.json").read_text())
    state=torch.load(folder/latest["checkpoint"],map_location="cpu",weights_only=True)
    if state.get("manifest")!=manifest or state.get("settings")!=settings:
        raise RuntimeError("Refusing resume: cache or training configuration changed")
    load_trainable(model,state["trainable"])
    for task,opt in optims.items():
        opt.load_state_dict(state["optimizers"][task])
        for item in opt.state.values():
            for key,value in item.items():
                if torch.is_tensor(value):item[key]=value.to(device)
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if device.type=="cuda" and state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["step"]),state["best"]

def create_model(folder,device):
    meta=json.loads((Path(folder)/"architecture.json").read_text())
    model=SourceLiDeReRGBD(
        meta["in_channels"],dim=meta["dim"],
        base_size=meta["base_size"],max_depth=meta["max_depth"])
    return model.to(device)

def load_best(folder,device="cuda"):
    folder=Path(folder)
    model=create_model(folder,device)
    for task in TASKS:
        ck=torch.load(folder/f"best_{task}.pt",map_location="cpu",weights_only=True)
        load_trainable(model.heads[task],ck["weights"])
    return model.eval().requires_grad_(False)

def train(args):
    if args.steps!=TOTAL:raise ValueError("Controlled experiment must retain 1200 steps")
    if args.batch_size!=1:
        raise ValueError("Use batch_size=1 for LiDeRe dense-query attention on L4")
    if args.grad_accum<1 or args.eval_every<1 or args.save_every<0 or args.warmup<1:
        raise ValueError("Invalid run configuration")
    if args.base_size not in (16,32,64,128):
        raise ValueError("Invalid LiDeRe query grid")
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Use BF16-enabled CUDA (L4/A100)")
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    root=Path(args.root).expanduser().resolve()
    local=stage_groundtruth_cache(root,args.local_cache_dir)
    identity=check_splits(local)
    train_ds=GeometryCache(local,"train",preload=args.cache_mode=="ram")
    val_ds=GeometryCache(local,"val",preload=args.cache_mode=="ram")
    if (len(train_ds),len(val_ds))!=(COUNT_TRAIN,COUNT_VAL):
        raise RuntimeError("Do not change the 95/20 split")
    width=int(train_ds[0]["features"].shape[0])
    folder=checkpoint_dir(args.save_dir)
    settings=dict(in_channels=width,dim=args.dim,
                  base_size=args.base_size,max_depth=args.max_depth,
                  seed=args.seed,lr_depth=args.lr_depth,lr_normal=args.lr_normal,
                  lr_seg=args.lr_seg,grad_accum=args.grad_accum,
                  warmup=args.warmup,weight_decay=args.weight_decay,
                  max_grad_norm=args.max_grad_norm)
    metadata=dict(in_channels=width,dim=args.dim,
                  base_size=args.base_size,max_depth=args.max_depth,
                  source="LiDeRe vendor 5aafadc4eb21b8c69ad2d8bc863fac35d4528d58",
                  loss_source="Probe3D vendor df4e9ace89ec6a353746c309a05646175c0d27ae")
    if args.resume:
        existing=json.loads((folder/"architecture.json").read_text())
        if existing!=metadata:
            raise RuntimeError("Architecture differs from existing training run")
    else:
        if (folder/"latest_source.json").exists():
            raise RuntimeError("Existing run detected: use --resume or a new --save-dir")
        atomic_json(folder/"architecture.json",metadata)
    model=SourceLiDeReRGBD(width,args.dim,args.base_size,args.max_depth).to(device)
    optims={task:torch.optim.AdamW(
        (p for p in model.heads[task].parameters() if p.requires_grad),
        lr=getattr(args,"lr_"+task),weight_decay=args.weight_decay
        ) for task in TASKS}
    depth_fn=DepthLoss(weight_sig=10.0,weight_grad=.5,max_depth=args.max_depth)
    best=dict(depth=float("inf"),normal=float("inf"),seg=float("-inf"))
    start=0
    if args.resume:
        start,best=resume_state(folder,model,optims,settings,identity,device)
        print(f"[LiDeRe] Resumed step {start}/{TOTAL}",flush=True)
    print(f"[LiDeRe] SOURCE-FAITHFUL 95/20 dataset; dim={args.dim}, grid={args.base_size}, 1200 updates",flush=True)
    print(f"[LiDeRe] Actual upstream implicit spatial attention / learned upsampling",flush=True)
    print(f"[LiDeRe] Probe3D DepthBinPrediction + DepthLoss + angular_loss",flush=True)
    journal=(folder/"train_source.jsonl").open("a",buffering=1)
    saves=save_steps(args.save_at,TOTAL)
    progress=tqdm(total=TOTAL*args.grad_accum,
                  initial=start*args.grad_accum,
                  desc="LiDeRe 3-head microbatches",dynamic_ncols=True)
    try:
        for step in range(start+1,TOTAL+1):
            factor=(step/args.warmup if step<=args.warmup else
                    .10+.90*.5*(1+math.cos(math.pi*(step-args.warmup)/
                                            max(1,TOTAL-args.warmup))))
            for task,opt in optims.items():
                for group in opt.param_groups:
                    group["lr"]=getattr(args,"lr_"+task)*factor
                opt.zero_grad(set_to_none=True)
            model.train()
            losses={k:0.0 for k in TASKS}
            for micro in range(args.grad_accum):
                # Deterministic with-replacement sampling of the fixed
                # training split. This is not a balanced epoch traversal.
                rng=random.Random(args.seed+step*1000003+micro*997)
                index=rng.randrange(COUNT_TRAIN)
                b=move(train_ds[index],device)
                if rng.random()<0.5:
                    for field in ("features","depth","seg","depth_valid",
                                  "normal_valid","seg_valid"):
                        b[field]=b[field].flip(-1)
                    b["normal"]=b["normal"].flip(-2).clone()
                    b["normal"][...,0]*=-1
                for task in TASKS:
                    with amp(device):
                        p,t,valid=training_inputs(b,task,model)
                        loss=task_loss(task,p,t,valid,depth_fn)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError(f"Nonfinite {task} loss step={step} micro={micro}")
                    (loss/args.grad_accum).backward()
                    losses[task]+=float(loss.detach())/args.grad_accum
                progress.update(1)
                if step==start+1 and micro==0:
                    print("[LiDeRe] First three independent backwards complete",flush=True)
            for task,opt in optims.items():
                total_norm=nn.utils.clip_grad_norm_(
                    (p for p in model.heads[task].parameters() if p.requires_grad),
                    args.max_grad_norm)
                if not bool(torch.isfinite(total_norm)):
                    raise RuntimeError(f"Nonfinite {task} gradient at step {step}")
                opt.step()
            progress.set_postfix({k:f"{v:.3f}" for k,v in losses.items()})
            if step==start+1 or step%args.log_every==0:
                journal.write(json.dumps(dict(step=step,**losses))+"\n")
            if step%args.eval_every==0 or step==TOTAL:
                metrics=evaluate(model,val_ds,device)
                with (folder/"validation_source.jsonl").open("a") as fd:
                    fd.write(json.dumps({"step":step,**metrics})+"\n")
                print(f"[LiDeRe] Step {step}: AbsRel={metrics['depth_absrel']:.4f}, "
                      f"normals={metrics['normal_deg']:.2f}, mIoU={metrics['seg_miou']:.4f}",flush=True)
                scores=dict(depth=metrics["depth_absrel"],
                            normal=metrics["normal_deg"],
                            seg=metrics["seg_miou"])
                for task,score in scores.items():
                    improved=score>best[task] if task=="seg" else score<best[task]
                    if improved:
                        best[task]=float(score)
                        atomic_torch_save(folder/f"best_{task}.pt",{
                            "step":step,"metric":score,"weights":trainable_state(model.heads[task]),
                            "source":"UNMODIFIED LiDeRe readout"})
            if (args.save_every and step%args.save_every==0) or step in saves or step==TOTAL:
                checkpoint(folder,step,model,optims,best,settings,identity)
                print(f"[LiDeRe] Saved optimizer state at step {step}",flush=True)
    finally:
        progress.close();journal.close()
    print(f"[LiDeRe] Complete: {folder}",flush=True)

def report(args):
    from PIL import Image
    from .dense import depth_view,normal_view,seg_view
    root=Path(args.root).expanduser().resolve()
    local=stage_groundtruth_cache(root,args.local_cache_dir)
    if not (len(list((local/"cache"/"train").glob("*.npz")))==95 and
            len(list((local/"cache"/"val").glob("*.npz")))==20):
        raise RuntimeError("Expected same 95/20 dataset")
    current_id=check_splits(local)
    latest=Path(args.save_dir)/"latest_source.json"
    if latest.is_file():
        pointer=json.loads(latest.read_text())
        state=torch.load(Path(args.save_dir)/pointer["checkpoint"],
                         map_location="cpu",weights_only=True)
        if state["manifest"]!=current_id:
            raise RuntimeError("Refusing evaluation on a different GT cache/split")
    data=GeometryCache(local,"val",preload=False)
    model=load_best(args.save_dir,args.device)
    metrics,rows=evaluate(model,data,torch.device(args.device),per_image=True)
    metrics.update(method="upstream LiDeRe core + upstream Probe3D losses",
                   train_samples=95,steps=TOTAL,
                   baseline_v3=dict(depth_absrel=.58,normal_deg=38.67,seg_miou=.2043))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/"metrics.json",metrics)
    with (out/"per_image.csv").open("w",newline="") as fd:
        wr=csv.DictWriter(fd,fieldnames=["file","depth_absrel","normal_deg"])
        wr.writeheader();wr.writerows(rows)
    for i in range(len(data)):
        b=move(data[i],torch.device(args.device))
        with torch.inference_mode():
            with amp(torch.device(args.device)):
                pred=rescale_maps(model(b["features"]))
        gt_sem=b["seg"][0].cpu().numpy()
        gt_color=seg_view(gt_sem.clip(0,12)).copy();gt_color[gt_sem==255]=0
        rgb=(b["rgb"][0].permute(1,2,0).cpu().numpy()*255).clip(0,255).astype("uint8")
        panels=[rgb,depth_view(b["depth"][0].cpu().numpy()),
                depth_view(pred["depth"][0,0].cpu().numpy()),
                normal_view(b["normal"][0].cpu().numpy()),
                normal_view(pred["normal"][0].permute(1,2,0).cpu().numpy()),
                gt_color,seg_view(pred["seg"][0].cpu().numpy())]
        canvas=Image.new("RGB",(7*256,256))
        for j,panel in enumerate(panels):
            canvas.paste(Image.fromarray(np.asarray(panel,dtype=np.uint8)).resize((256,256)),(j*256,0))
        canvas.save(out/f"val_{i:02d}.jpg",quality=88)
    folder=Path(args.save_dir)
    for name in ("train_source.jsonl","validation_source.jsonl","architecture.json","latest_source.json"):
        if (folder/name).exists():
            (out/name).write_bytes((folder/name).read_bytes())
    destination=out.parent/"Jeveration_Source_LiDeRe_Probe3D_Report.zip"
    with zipfile.ZipFile(destination,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            if f.is_file():z.write(f,f.name)
    print(f"Report: {destination}",flush=True)
    print(json.dumps(metrics,indent=2),flush=True)
    return destination

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    for command in ("fit","report"):
        x=sub.add_parser(command)
        x.add_argument("--root",default="/content/drive/MyDrive/Jeveration_GT/pilot")
        x.add_argument("--local-cache-dir",default="/content/jeveration_gt_pilot_cache")
        x.add_argument("--save-dir",default="/content/drive/MyDrive/Jeveration_GT/pilot/checkpoints_source_lidere")
        x.add_argument("--cache-mode",choices=("ram","local"),default="ram")
        x.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
        if command=="fit":
            x.add_argument("--steps",type=int,default=1200)
            x.add_argument("--batch-size",type=int,default=1)
            x.add_argument("--grad-accum",type=int,default=16)
            x.add_argument("--dim",type=int,default=64)
            x.add_argument("--base-size",type=int,default=64)
            x.add_argument("--max-depth",type=float,default=80.0)
            x.add_argument("--lr-depth",type=float,default=1.5e-4)
            x.add_argument("--lr-normal",type=float,default=1.5e-4)
            x.add_argument("--lr-seg",type=float,default=1.5e-4)
            x.add_argument("--weight-decay",type=float,default=.01)
            x.add_argument("--warmup",type=int,default=80)
            x.add_argument("--max-grad-norm",type=float,default=1.)
            x.add_argument("--eval-every",type=int,default=100)
            x.add_argument("--save-every",type=int,default=200)
            x.add_argument("--save-at",default="400,700,900")
            x.add_argument("--log-every",type=int,default=10)
            x.add_argument("--seed",type=int,default=42)
            x.add_argument("--resume",action="store_true")
        else:
            x.add_argument("--output",default="/content/Jeveration_Source_LiDeRe_Report")
    return p

def main():
    a=parse_args()
    {"fit":train,"report":report}[a.command](a)

if __name__=="__main__":main()
