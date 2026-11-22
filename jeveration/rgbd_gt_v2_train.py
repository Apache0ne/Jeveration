"""Train the SAME cached 95 Hypersim frames for 1200 steps; no new data.

Run: python -u -m jeveration.rgbd_gt_v2_train fit \
   --root /content/drive/MyDrive/Jeveration_GT/pilot \
   --save-dir /content/drive/MyDrive/Jeveration_GT/pilot/checkpoints_v2

Key differences from v1: geometrically consistent loss, class balancing,
fixed camera convention, randomized but synchronized RGB/geometry transforms,
weight EMA, late sharpness-aware optimization, and depth/normal-based model
selection. Source dataset + scene split + decoder architecture do not change.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import math
import os
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .rgbd_gt_train import GTCache
from .rgbd_gt_stage import stage_groundtruth_cache
from .rgbd_gt_v2_losses import (
    NUM_CLASSES, geometry_aware_loss, validation_metrics, finalize_metrics,
)
from .rgbd_sidecar import RGBDHead, save_head, load_head
from .rgbd_tracking import (LiveTraining, atomic_json, atomic_torch_save,
                             checkpoint_dir, save_steps)

# Previous report, for normalization ONLY; not used in training gradients.
BASELINE_DEPTH = 0.6608039541962262
BASELINE_NORMAL = 41.07050489052095
EXPECTED_TRAIN = 95
EXPECTED_VAL = 20


class GeometryCache(GTCache):
    """Read original RGB from the EXISTING NPZ without reprocessing inputs."""
    def _read(self, i):
        x = super()._read(i)
        with np.load(self.files[i]) as z:
            # Old successful v2 ground-truth cache includes both RGB images.
            if "original_rgb" not in z:
                raise RuntimeError(f"Missing original_rgb: {self.files[i]}")
            original = z["original_rgb"].copy()
            x["alignment_ssim"] = torch.tensor(float(z["alignment_ssim"]))
        if original.shape != (512, 512, 3):
            raise RuntimeError(f"Original RGB shape mismatch: {self.files[i]}")
        x["original_rgb"] = torch.from_numpy(original).permute(2,0,1).float()/255.
        return x


def class_weights_from_training_set(dataset):
    """Sqrt-frequency balancing, bounded; average train-pixel weight stays 1."""
    counts = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    for sample in dataset:
        seg = sample["seg"]
        valid = sample["seg_valid"] & (seg != 255)
        if bool(valid.any()):
            counts += torch.bincount(seg[valid].long(), minlength=NUM_CLASSES).double()
    present = counts > 0
    if not bool(present.any()):
        raise RuntimeError("No valid semantic labels in cache")
    freq = counts / counts.sum()
    weights = torch.zeros_like(freq)
    weights[present] = torch.sqrt(freq[present].median()/freq[present]).clamp(.35,3.)
    weights /= (weights*freq).sum().clamp_min(1e-8)
    return weights.float().clamp(0.2,5.0), counts.long()


def augment(b, original_probability=0.20, feature_drop=0.06):
    """Aligned 2D flip and photometric jitter. No untracked camera movement."""
    b={key: value.clone() for key,value in b.items()}
    if random.random() < .50:
        for key in ("features","rgb","original_rgb","depth","seg",
                    "depth_valid","normal_valid","seg_valid"):
            b[key] = b[key].flip(-1)
        b["normal"] = b["normal"].flip(-2)
        b["normal"][...,0] *= -1  # camera-space X changes on horizontal mirror
    if random.random() < original_probability:
        # True RGB is aligned with true depth; decoder RGB is used otherwise.
        b["rgb"] = b["original_rgb"]
    rgb=b["rgb"]
    b["rgb"] = (rgb * random.uniform(.85,1.15)
                +random.uniform(-.055,.055)).clamp(0.,1.)
    # Channel dropout of the 32x32 decoder features limits memorization.
    if feature_drop > 0:
        keep=(torch.rand((b["features"].shape[0], b["features"].shape[1],1,1))
              >=feature_drop).float()
        b["features"]=b["features"] * (keep / (1-feature_drop))
    return b


@torch.no_grad()
def ema_step(ema, model, decay):
    for target, source in zip(ema.parameters(),model.parameters()):
        target.lerp_(source,1.0-decay)


@torch.no_grad()
def evaluate(model, dataset, device, class_weights):
    model.eval()
    total_loss=0.;n=0
    acc=dict(absrel_sum=0.,depth_pixels=0,normal_deg_sum=0.,
             normal_pixels=0,confusion=torch.zeros(NUM_CLASSES,NUM_CLASSES))
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)
    for sample in loader:
        b={k:v.to(device) for k,v in sample.items()
           if k not in ("original_rgb","alignment_ssim")}
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
            out=model(b["features"],b["rgb"])
        loss,_=geometry_aware_loss(out,b,class_weights)
        metrics=validation_metrics(out,b)
        total_loss+=float(loss.item());n+=1
        for key in ("absrel_sum","depth_pixels","normal_deg_sum","normal_pixels"):
            acc[key]+=metrics[key]
        acc["confusion"]+=metrics["confusion"]
    result=finalize_metrics(acc)
    result["loss"]=total_loss/max(1,n)
    result["samples"]=n
    return result


def selection_score(metrics):
    """Depth and normals drive checkpoint choice; segmentation remains auxiliary."""
    return (.60 * metrics["depth_absrel"]/BASELINE_DEPTH
            +.30 * metrics["normal_deg"]/BASELINE_NORMAL
            +.10 * (1.0-metrics["seg_miou"]))


def _best_file(folder,name,model,step,metrics):
    path=folder/name
    tmp=folder/("."+name+".partial")
    save_head(tmp,model,step=int(step),metrics=metrics,
              trainer="ground_truth_v2",source="95 train / 20 held-out Hypersim")
    os.replace(tmp,path)


def _save_full(folder,step,model,ema,optimizer,width,bests,a):
    model_path=folder/f"rgbd_v2_step_{step:06d}.pt"
    trainer_path=folder/f"trainer_v2_step_{step:06d}.pt"
    _best_file(folder,model_path.name,ema,step,dict(step=step))
    state=dict(
        step=int(step),steps_target=int(a.steps),width=int(width),
        training_count=EXPECTED_TRAIN,validation_count=EXPECTED_VAL,
        model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        ema={k:v.detach().cpu() for k,v in ema.state_dict().items()},
        optimizer=optimizer.state_dict(),bests=bests,
        torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
    )
    atomic_torch_save(trainer_path,state)
    atomic_json(folder/"latest_v2.json",
                dict(step=step,trainer=trainer_path.name,model=model_path.name))


def _sam_perturb(model,rho):
    """SAM ascent: epsilon = rho * grad / ||grad||2. Returns perturbations."""
    parameters=[p for p in model.parameters() if p.grad is not None]
    if not parameters:
        raise RuntimeError("No parameters received gradients")
    with torch.no_grad():
        norm=torch.linalg.vector_norm(
            torch.stack([p.grad.detach().float().norm(2) for p in parameters]))
        if not bool(torch.isfinite(norm)) or norm.item() <= 0:
            raise RuntimeError("Invalid gradient norm before SAM ascent")
        scale=rho/(norm+1e-12)
        offsets=[]
        for p in parameters:
            offset=p.grad.detach()*scale
            p.add_(offset)
            offsets.append((p,offset))
    return offsets


def _sam_restore(offsets):
    with torch.no_grad():
        for p,e in offsets:p.sub_(e)


def train(a):
    if a.steps!=1200:
        raise ValueError("The controlled v2 experiment requires exactly 1200 optimizer steps")
    if (a.batch_size<1 or a.grad_accum<1 or a.eval_every<1 or a.save_every<0
        or not 0<a.lr<=.001 or a.ema_decay<0 or a.ema_decay>=1
        or a.sam_start<1 or a.sam_start>a.steps
        or a.sam_rho<0 or not 0<=a.original_prob<=1
        or not 0<=a.feature_drop<1):
        raise ValueError("Invalid training hyperparameters")
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    device=torch.device(a.device)
    if device.type=="cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is required on CUDA")
    if device.type=="cuda":
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
    root=Path(a.root).expanduser().resolve()
    data_root=stage_groundtruth_cache(root,a.local_cache_dir)
    trainset=GeometryCache(data_root,"train",preload=a.cache_mode=="ram")
    valset=GeometryCache(data_root,"val",preload=a.cache_mode=="ram")
    if (len(trainset),len(valset))!=(EXPECTED_TRAIN,EXPECTED_VAL):
        raise RuntimeError(f"Expected same 95/20 samples; got {len(trainset)}/{len(valset)}")
    weights,counts=class_weights_from_training_set(trainset)
    weights=weights.to(device)
    width=trainset[0]["features"].shape[0]
    model=RGBDHead(width).to(device).train()
    ema=copy.deepcopy(model).to(device).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,
                                betas=(.9,.999),eps=1e-8,weight_decay=.035)
    folder=checkpoint_dir(a.save_dir)
    saves=save_steps(a.save_at,a.steps)
    start=0
    bests=dict(joint=float("inf"),depth=float("inf"),
               normal=float("inf"),miou=float("-inf"))
    if a.resume:
        pointer=folder/"latest_v2.json"
        if not pointer.exists(): raise RuntimeError(f"No v2 resume state: {pointer}")
        state_path=folder/json.loads(pointer.read_text())["trainer"]
        ck=torch.load(state_path,map_location="cpu",weights_only=True)
        if ck["width"]!=width or ck["steps_target"]!=a.steps:
            raise RuntimeError("v2 resume incompatible with decoder width or total steps")
        model.load_state_dict(ck["model"],strict=True)
        ema.load_state_dict(ck["ema"],strict=True)
        optimizer.load_state_dict(ck["optimizer"])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.to(device)
        start=int(ck["step"]);bests=ck["bests"]
        if "python_rng" in ck:random.setstate(ck["python_rng"])
        if "torch_rng" in ck:torch.set_rng_state(ck["torch_rng"])
        print(f"[v2] Resumed from step {start}",flush=True)

    print(f"[v2] SAME cache: train={len(trainset)}, val={len(valset)}, steps={a.steps}",flush=True)
    print(f"[v2] class counts={counts.tolist()} class weights={weights.cpu().tolist()}",flush=True)
    print(f"[v2] geometry losses; BF16={device.type=='cuda'}; EMA={a.ema_decay}; "
          f"SAM start={a.sam_start} rho={a.sam_rho}",flush=True)
    print(f"[v2] old checkpoints untouched; saving new run under {folder}",flush=True)
    loader=DataLoader(trainset,batch_size=a.batch_size,shuffle=True,
                      num_workers=0,pin_memory=device.type=="cuda")
    iterator=iter(loader)
    tracker=LiveTraining(folder,a.steps,start,log_every=a.log_every,
                         keep_last=0,progress=True,tensorboard=True,
                         microbatches_per_step=a.grad_accum)
    details=(folder/"v2_training_details.jsonl").open("a",buffering=1)
    try:
        for step in range(start+1,a.steps+1):
            # Small-data schedule: warmup then cosine decay, without short-circuiting.
            if step<=a.warmup:
                factor=step/a.warmup
            else:
                factor=.08+.92*.5*(1+math.cos(math.pi*(step-a.warmup)/
                                              max(1,a.steps-a.warmup)))
            for g in optimizer.param_groups:g["lr"]=a.lr*factor
            model.train()
            optimizer.zero_grad(set_to_none=True)
            use_sam= a.sam_rho>0 and step>=a.sam_start
            saved_batches=[]
            component={k:0. for k in ("depth","normal","geometry","seg",
                                       "seg_ce","seg_dice","silog")}
            train_total=0.
            for micro in range(a.grad_accum):
                try:batch=next(iterator)
                except StopIteration:
                    iterator=iter(loader);batch=next(iterator)
                batch=augment(batch,a.original_prob,a.feature_drop)
                # Save exact sampled batch+augmentation on CPU for SAM second pass.
                if use_sam:saved_batches.append(batch)
                b={k:v.to(device,non_blocking=True) for k,v in batch.items()
                   if k not in ("original_rgb","alignment_ssim")}
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
                    out=model(b["features"],b["rgb"])
                    loss,parts=geometry_aware_loss(out,b,weights)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"Nonfinite loss at step={step}, micro={micro}")
                (loss/a.grad_accum).backward()
                train_total+=float(loss.detach())/a.grad_accum
                for k in component:component[k]+=float(parts[k].detach())/a.grad_accum
                tracker.microstep(step,micro+1)
            if use_sam:
                offsets=_sam_perturb(model,a.sam_rho)
                try:
                    optimizer.zero_grad(set_to_none=True)
                    for batch in saved_batches:
                        b={k:v.to(device,non_blocking=True) for k,v in batch.items()
                           if k not in ("original_rgb","alignment_ssim")}
                        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
                            out=model(b["features"],b["rgb"])
                            loss,_=geometry_aware_loss(out,b,weights)
                        (loss/a.grad_accum).backward()
                finally:_sam_restore(offsets)
            grad_norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not bool(torch.isfinite(grad_norm)):
                raise RuntimeError(f"Nonfinite gradient at step {step}")
            optimizer.step()
            ema_step(ema,model,a.ema_decay)
            # Retain existing logger output, plus geometry-specific v2 history.
            gpu_gib=(torch.cuda.memory_allocated(device)/2**30
                     if device.type=="cuda" else None)
            tracker.step(step,train_total,optimizer.param_groups[0]["lr"],
                         component,grad_norm,gpu_gib)
            if step==1 or step%a.log_every==0 or step==a.steps:
                details.write(json.dumps(dict(step=step,sam=use_sam,
                    loss=train_total,lr=optimizer.param_groups[0]["lr"],
                    **component))+"\n")
                if tracker.writer:
                    for k in ("geometry","seg_ce","seg_dice","silog"):
                        tracker.writer.add_scalar("train_v2/"+k,component[k],step)
            if step%a.eval_every==0 or step==a.steps:
                metrics=evaluate(ema,valset,device,weights)
                metrics["joint_score"]=selection_score(metrics)
                metrics["model"]="ema"
                tracker.validation(step,metrics)
                # All checkpoints selected by held-out GEOMETRY, not training loss.
                score_map=dict(joint=metrics["joint_score"],
                               depth=metrics["depth_absrel"],
                               normal=metrics["normal_deg"],
                               miou=metrics["seg_miou"])
                for key,score in score_map.items():
                    improved=(score>bests[key] if key=="miou" else score<bests[key])
                    if improved:
                        bests[key]=float(score)
                        _best_file(folder,f"rgbd_v2_best_{key}.pt",ema,step,metrics)
                        if key=="joint":
                            _best_file(folder,"rgbd_best.pt",ema,step,metrics)
                        print(f"[v2] BEST {key} @ {step}: {score:.5f}",flush=True)
            if ((a.save_every>0 and step%a.save_every==0)
                 or step in saves or step==a.steps):
                _save_full(folder,step,model,ema,optimizer,width,bests,a)
                print(f"[v2] saved complete optimizer + EMA checkpoint at step {step}",flush=True)
    finally:
        tracker.close()
        details.close()
    print(f"[v2] Finished 1200 steps; best geometry checkpoints: {folder}",flush=True)


def make_report(a):
    from PIL import Image
    from .dense import depth_view,normal_view,seg_view
    root=Path(a.root)
    data_root=stage_groundtruth_cache(root,a.local_cache_dir)
    dataset=GeometryCache(data_root,"val",preload=False)
    if len(dataset)!=20:raise RuntimeError("Validation set must be exactly 20")
    folder=Path(a.save_dir)
    model=load_head(folder/"rgbd_best.pt",device=a.device)
    weights,_=class_weights_from_training_set(
        GeometryCache(data_root,"train",preload=False))
    result=evaluate(model,dataset,torch.device(a.device),weights.to(a.device))
    result["checkpoint_selection"]="depth 60%, normals 30%, semantic mIoU 10%"
    result["baseline_v1"]=dict(depth_absrel=BASELINE_DEPTH,
                               normal_deg=BASELINE_NORMAL,
                               seg_acc=.5953934439289714)
    result["train_images"]=95;result["val_images"]=20;result["steps"]=1200
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/"v2_report.json",result)
    image_rows=[]
    for i,f in enumerate(dataset.files):
        sample=dataset[i]
        with torch.inference_mode():
            b={k:v[None].to(a.device) for k,v in sample.items()
               if k not in ("alignment_ssim","original_rgb")}
            with torch.autocast("cuda",dtype=torch.bfloat16,
                                enabled=str(a.device).startswith("cuda")):
                out=model(b["features"],b["rgb"])
        depth=out["log_depth"][0].float().clamp(-5,5).exp().cpu().numpy()
        normal=torch.nn.functional.normalize(out["normal_raw"][0].float(),
                                             dim=-1).cpu().numpy()
        pred_seg=out["seg_logits"][0].argmax(0).cpu().numpy()
        gt_z=sample["depth"].cpu().numpy()
        gt_n=sample["normal"].cpu().numpy()
        gt_seg=sample["seg"].cpu().numpy()
        gt_sem_color=seg_view(gt_seg.clip(0,12)).copy()
        gt_sem_color[gt_seg==255]=0
        rgb=(sample["rgb"].permute(1,2,0).cpu().numpy()*255).clip(0,255).astype("uint8")
        panels=[rgb,depth_view(gt_z),depth_view(depth),
                normal_view(gt_n),normal_view(normal),
                gt_sem_color,seg_view(pred_seg)]
        image=Image.new("RGB",(7*512,512))
        for j,panel in enumerate(panels):
            image.paste(Image.fromarray(np.asarray(panel,dtype=np.uint8)),(j*512,0))
        image.resize((7*256,256)).save(output/f"val_{i:02d}.jpg",quality=87)
        m=sample["depth_valid"].cpu().numpy().astype(bool)
        n=sample["normal_valid"].cpu().numpy().astype(bool)
        score=float(np.mean((np.abs(depth-gt_z)/np.maximum(.01,gt_z))[m]))
        dots=np.sum(normal*gt_n,axis=-1).clip(-1,1)
        angle=float(np.degrees(np.arccos(dots[n])).mean())
        image_rows.append(dict(file=f.name,depth_absrel=score,normal_deg=angle))
    with (output/"per_image_v2.csv").open("w",newline="") as fp:
        writer=csv.DictWriter(fp,fieldnames=["file","depth_absrel","normal_deg"])
        writer.writeheader();writer.writerows(image_rows)
    for name in ("validation.jsonl","training_metrics.jsonl",
                 "v2_training_details.jsonl","latest_v2.json","events.jsonl"):
        file=folder/name
        if file.exists():
            (output/name).write_bytes(file.read_bytes())
    for name in ("source_manifest.json","preparation_summary.json"):
        file=root/name
        if file.exists():(output/name).write_bytes(file.read_bytes())
    path=output.parent/"Jeveration_RGBD_SmallData_V2_Report.zip"
    with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_DEFLATED) as arc:
        for file in sorted(output.iterdir()):
            if file.is_file():arc.write(file,file.name)
    print(f"v2 evaluation report: {path}",flush=True)
    return path


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest="command",required=True)
    for mode in ("fit","report"):
        x=subs.add_parser(mode)
        x.add_argument("--root",default="/content/drive/MyDrive/Jeveration_GT/pilot")
        x.add_argument("--save-dir",default="/content/drive/MyDrive/Jeveration_GT/pilot/checkpoints_v2")
        x.add_argument("--local-cache-dir",default="/content/jeveration_gt_pilot_cache")
        x.add_argument("--cache-mode",choices=("ram","local"),default="ram")
        x.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
        if mode=="fit":
            x.add_argument("--steps",type=int,default=1200)
            x.add_argument("--batch-size",type=int,default=1)
            x.add_argument("--grad-accum",type=int,default=16)
            x.add_argument("--lr",type=float,default=0.0001)
            x.add_argument("--warmup",type=int,default=60)
            x.add_argument("--ema-decay",type=float,default=.995)
            x.add_argument("--sam-start",type=int,default=500)
            x.add_argument("--sam-rho",type=float,default=.025)
            x.add_argument("--original-prob",type=float,default=.20)
            x.add_argument("--feature-drop",type=float,default=.06)
            x.add_argument("--eval-every",type=int,default=100)
            x.add_argument("--save-every",type=int,default=200)
            x.add_argument("--save-at",default="400,700,900")
            x.add_argument("--log-every",type=int,default=10)
            x.add_argument("--seed",type=int,default=42)
            x.add_argument("--resume",action="store_true")
        else:
            x.add_argument("--output",default="/content/Jeveration_V2_Report")
    return p


def main():
    args=parser().parse_args()
    if args.command=="fit":train(args)
    else:make_report(args)


if __name__=="__main__":main()
