"""Ground-truth RGBD v3: independent depth / normal / semantic training.

Same immutable 95 train + 20 scene-held-out Hypersim cache, 1200 outer steps.
Each head receives one independent AdamW step per outer iteration. Fusion is
trained separately on DETACHED predictions from step --fusion-start onward,
within the SAME 1200-iteration timeline (no extra head optimization steps).
"""
import argparse
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
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate

from .rgbd_gt_v2_train import GeometryCache, class_weights_from_training_set, augment
from .rgbd_gt_stage import stage_groundtruth_cache
from .rgbd_gt_v2_losses import validation_metrics, finalize_metrics
from .rgbd_v3_model import (TASK_CHANNELS, IndependentHead, GeometryFusion,
                             independent_outputs, decoded, save_component,
                             load_component,load_bundle)
from .rgbd_v3_losses import depth_loss,normal_loss,segmentation_loss,fusion_loss
from .rgbd_tracking import atomic_json,atomic_torch_save,checkpoint_dir,save_steps

N_TRAIN=95
N_VAL=20
N_STEPS=1200
BASELINE_V2=dict(depth_absrel=.5614,normal_deg=42.02,seg_acc=.5690)
BASELINE_V1=dict(depth_absrel=.6608,normal_deg=41.07,seg_acc=.5954)


def amp(device):
    return torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda")


def _clone_cpu(model):
    return {name:weight.detach().cpu().clone()
            for name,weight in model.state_dict().items()}


def _save_all(out,step,heads,fusion,optims,choices,args):
    # Full optimizer state must be on disk before the pointer changes.
    state=dict(version=3,steps=N_STEPS,step=step,width=heads["depth"].in_channels,
               train_count=N_TRAIN,val_count=N_VAL,
               heads={k:_clone_cpu(v) for k,v in heads.items()},
               fusion=_clone_cpu(fusion),
               optimizers={k:v.state_dict() for k,v in optims.items()},
               best=choices,random_state=random.getstate(),
               cpu_rng=torch.get_rng_state(),
               cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               config=dict(fusion_start=args.fusion_start,grad_accum=args.grad_accum,
                           batch_size=args.batch_size,seed=args.seed))
    name=f"trainer_v3_step_{step:06d}.pt"
    atomic_torch_save(out/name,state)
    atomic_json(out/"latest_v3.json",dict(step=step,trainer=name))


def _restore(out,heads,fusion,optims,args,device):
    pointer=out/"latest_v3.json"
    if not pointer.is_file():raise RuntimeError(f"Missing v3 resume pointer: {pointer}")
    path=out/json.loads(pointer.read_text())["trainer"]
    st=torch.load(path,map_location="cpu",weights_only=True)
    if (st.get("version")!=3 or st["steps"]!=N_STEPS
        or st["width"]!=heads["depth"].in_channels
        or st.get("config")!=dict(fusion_start=args.fusion_start,
                 grad_accum=args.grad_accum,batch_size=args.batch_size,
                 seed=args.seed)):
        raise RuntimeError("Resume configuration differs from checkpoint")
    for key,head in heads.items():head.load_state_dict(st["heads"][key],strict=True)
    fusion.load_state_dict(st["fusion"],strict=True)
    for key,opt in optims.items():
        opt.load_state_dict(st["optimizers"][key])
        for v in opt.state.values():
            for name,item in v.items():
                if isinstance(item,torch.Tensor):v[name]=item.to(device)
    random.setstate(st["random_state"])
    torch.set_rng_state(st["cpu_rng"])
    if device.type=="cuda" and st["cuda_rng"]:
        torch.cuda.set_rng_state_all(st["cuda_rng"])
    return int(st["step"]),st["best"]


def _to_gpu(b,device):
    return {key:value.to(device,non_blocking=True)
            for key,value in b.items()
            if key not in ("original_rgb","alignment_ssim")}


def _sample(ds,step,micro,batch_size,seed):
    # Deterministic sample indices, independent of dataloader iterator position.
    generator=torch.Generator(device="cpu").manual_seed(
        int(seed)+step*1000003+micro*733)
    indices=torch.randint(len(ds),(batch_size,),generator=generator).tolist()
    return default_collate([ds[i] for i in indices])


def _single_loss(task,pred,b,weights):
    if task=="depth":return depth_loss(pred,b)
    if task=="normal":return normal_loss(pred,b)
    return segmentation_loss(pred,b,weights)


@torch.no_grad()
def evaluate(heads,fusion,ds,device,with_fusion=True,include_per_image=False):
    for head in heads.values():head.eval()
    fusion.eval()
    a={key:dict(absrel_sum=0.,depth_pixels=0,normal_deg_sum=0.,
                normal_pixels=0,confusion=torch.zeros(13,13))
       for key in ("raw","fused")}
    rows=[]
    for index in range(len(ds)):
        original=ds[index]
        b=_to_gpu(default_collate([original]),device)
        with amp(device):
            raw=independent_outputs(heads,b["features"],b["rgb"],detach=True)
            fused=fusion(raw,b["rgb"]) if with_fusion else raw
        for key,raw_out in (("raw",raw),("fused",fused)):
            scores=validation_metrics(raw_out,b)
            for metric in ("absrel_sum","depth_pixels","normal_deg_sum","normal_pixels"):
                a[key][metric]+=scores[metric]
            a[key]["confusion"]+=scores["confusion"]
            if include_per_image:
                rows.append(dict(name=ds.files[index].name,version=key,
                    depth_absrel=scores["absrel_sum"]/max(scores["depth_pixels"],1),
                    normal_deg=scores["normal_deg_sum"]/max(scores["normal_pixels"],1)))
    results={key:finalize_metrics(value) for key,value in a.items()}
    results["samples"]=len(ds)
    if include_per_image:return results,rows
    return results


def train(args):
    if args.steps!=N_STEPS:raise ValueError("v3 comparison requires exactly 1200 outer optimizer iterations")
    if (args.batch_size<1 or args.grad_accum<1 or args.eval_every<1 or args.save_every<0
        or args.fusion_start<1 or args.fusion_start>N_STEPS
        or args.warmup<1 or args.warmup>=N_STEPS):
        raise ValueError("Invalid training configuration")
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Use BF16-capable CUDA (L4/A100)")
    root=Path(args.root).expanduser().resolve()
    local=stage_groundtruth_cache(root,args.local_cache_dir)
    ds=GeometryCache(local,"train",preload=args.cache_mode=="ram")
    val=GeometryCache(local,"val",preload=args.cache_mode=="ram")
    if (len(ds),len(val))!=(N_TRAIN,N_VAL):
        raise RuntimeError(f"Refusing changed data/split: got {len(ds)} train, {len(val)} val; expected 95/20")
    weights,counts=class_weights_from_training_set(ds)
    weights=weights.to(device)
    width=ds[0]["features"].shape[0]
    heads=nn.ModuleDict({task:IndependentHead(task,width)
                         for task in TASK_CHANNELS}).to(device)
    fusion=GeometryFusion().to(device)
    opts={
        "depth":torch.optim.AdamW(heads["depth"].parameters(),lr=args.lr_depth,
                                  weight_decay=.05),
        "normal":torch.optim.AdamW(heads["normal"].parameters(),lr=args.lr_normal,
                                   weight_decay=.04),
        "seg":torch.optim.AdamW(heads["seg"].parameters(),lr=args.lr_seg,
                                weight_decay=.025),
        "fusion":torch.optim.AdamW(fusion.parameters(),lr=args.lr_fusion,
                                   weight_decay=.01),
    }
    lrs=dict(depth=args.lr_depth,normal=args.lr_normal,
             seg=args.lr_seg,fusion=args.lr_fusion)
    folder=checkpoint_dir(args.save_dir)
    start=0
    best=dict(depth=float("inf"),normal=float("inf"),
              seg=float("-inf"),fusion=float("inf"))
    if args.resume:
        start,best=_restore(folder,heads,fusion,opts,args,device)
        print(f"[v3] Restored independent head + fusion optimizers at step {start}",flush=True)
    extras=save_steps(args.save_at,N_STEPS)
    log=(folder/"metrics_v3.jsonl").open("a",encoding="utf8",buffering=1)
    from tqdm.auto import tqdm
    progress=tqdm(total=N_STEPS*args.grad_accum,initial=start*args.grad_accum,
                  desc="v3 independent microbatches",dynamic_ncols=True,mininterval=.5)
    print("[v3] three PRIVATE networks, each has its OWN optimizer and 1200 updates",flush=True)
    print(f"[v3] train=95 val=20 width={width}, effective batch={args.batch_size*args.grad_accum}",flush=True)
    print(f"[v3] class counts={counts.tolist()}",flush=True)
    print(f"[v3] fusion starts at outer step {args.fusion_start}; its input is always detached",flush=True)
    try:
        for step in range(start+1,N_STEPS+1):
            if step<=args.warmup:lr_factor=step/args.warmup
            else:lr_factor=.04+.96*.5*(1+math.cos(
                math.pi*(step-args.warmup)/(N_STEPS-args.warmup)))
            for task,opt in opts.items():
                for group in opt.param_groups:group["lr"]=lrs[task]*lr_factor
                opt.zero_grad(set_to_none=True)
            for h in heads.values():h.train()
            fusion.train()
            total={key:0. for key in ("depth","normal","seg","fusion")}
            fusion_active=step>=args.fusion_start
            for micro in range(args.grad_accum):
                if step==start+1 and micro==0:
                    print("[v3] first microbatch: depth -> normals -> segmentation -> optional fusion",flush=True)
                batch=_sample(ds,step,micro,args.batch_size,args.seed)
                batch=augment(batch,original_probability=0.0,
                              feature_drop=args.feature_drop)
                b=_to_gpu(batch,device)
                raw={}
                # Sequential fwd/back: never retain 3 full 512px training graphs.
                for task in TASK_CHANNELS:
                    with amp(device):
                        result=heads[task](b["features"],b["rgb"])
                        loss,_=_single_loss(task,result,b,weights)
                    if not bool(torch.isfinite(loss)):raise RuntimeError(f"NaN {task} at step {step}")
                    (loss/args.grad_accum).backward()
                    total[task]+=float(loss.detach())/args.grad_accum
                    raw[{"depth":"log_depth","normal":"normal_raw","seg":"seg_logits"}[task]]=result.detach()
                    del result,loss
                if fusion_active:
                    with amp(device):
                        fused=fusion(raw,b["rgb"])
                        f_loss,_=fusion_loss(fused,raw,b)
                    if not bool(torch.isfinite(f_loss)):
                        raise RuntimeError(f"Nonfinite fusion loss at {step}")
                    (f_loss/args.grad_accum).backward()
                    total["fusion"]+=float(f_loss.detach())/args.grad_accum
                progress.update(1)
                if step==start+1 and micro==0:
                    print("[v3] first microbatch completed; independent backward passes succeeded",flush=True)
            for task in TASK_CHANNELS:
                norm=nn.utils.clip_grad_norm_(heads[task].parameters(),1.)
                if not bool(torch.isfinite(norm)):raise RuntimeError(f"Invalid gradient {task}")
                opts[task].step()
            if fusion_active:
                norm=nn.utils.clip_grad_norm_(fusion.parameters(),1.)
                if not bool(torch.isfinite(norm)):raise RuntimeError("Invalid fusion gradient")
                opts["fusion"].step()
            progress.set_postfix(depth=f"{total['depth']:.3f}",
                                 normal=f"{total['normal']:.3f}",
                                 seg=f"{total['seg']:.3f}",
                                 fusion=f"{total['fusion']:.3f}")
            if step==1 or step%args.log_every==0 or step==N_STEPS:
                log.write(json.dumps(dict(step=step,**total))+"\n")
            if step%args.eval_every==0 or step==N_STEPS:
                scores=evaluate(heads,fusion,val,device,with_fusion=fusion_active)
                raw=scores["raw"];fused=scores["fused"]
                record=dict(step=step,raw=raw,fused=fused,
                            fusion_active=fusion_active)
                with (folder/"validation_v3.jsonl").open("a",encoding="utf8") as fp:
                    fp.write(json.dumps(record)+"\n")
                message=(f"[v3] val {step}: raw depth={raw['depth_absrel']:.4f} "
                         f"normal={raw['normal_deg']:.2f}deg "
                         f"seg={raw['seg_acc']:.3f}/mIoU={raw['seg_miou']:.3f}")
                print("\n"+message,flush=True)
                vals=dict(depth=raw["depth_absrel"],normal=raw["normal_deg"],
                          seg=raw["seg_miou"])
                for task,valscore in vals.items():
                    improved=(valscore>best[task] if task=="seg"
                              else vals[task]<best[task])
                    if improved:
                        best[task]=float(valscore)
                        save_component(folder/f"best_{task}.pt",
                            heads[task],task,width,step,raw)
                        print(f"[v3] new BEST {task}: {valscore:.4f} at {step}",flush=True)
                if fusion_active:
                    # Fusion is ranked on BOTH depth and normals; never
                    # allow segmentation to mask deteriorating geometry.
                    fused_score=(.65*fused["depth_absrel"]/.56+
                                 .35*fused["normal_deg"]/42.)
                    if fused_score<best["fusion"]:
                        best["fusion"]=float(fused_score)
                        save_component(folder/"best_fusion.pt",
                                       fusion,"fusion",width,step,fused)
                        print(f"[v3] new BEST fusion: {fused_score:.4f}",flush=True)
            if ((args.save_every and step%args.save_every==0)
                 or step in extras or step==N_STEPS):
                _save_all(folder,step,heads,fusion,opts,best,args)
                print(f"[v3] saved four optimizer states at step {step} -> {folder}",flush=True)
    finally:
        progress.close()
        log.close()
    print("[v3] Complete. Separate task checkpoints and best fusion in",folder,flush=True)


def report(args):
    from PIL import Image
    from .dense import depth_view,normal_view,seg_view
    root=Path(args.root)
    local=stage_groundtruth_cache(root,args.local_cache_dir)
    val=GeometryCache(local,"val",preload=False)
    if len(val)!=20:raise RuntimeError("Expected 20 held-out images")
    heads,fusion=load_bundle(args.save_dir,args.device)
    device=torch.device(args.device)
    results,rows=evaluate(heads,fusion,val,device,with_fusion=fusion is not None,
                          include_per_image=True)
    results["training_images"]=95
    results["heldout_images"]=20
    results["optimizer_steps_per_head"]=1200
    results["v1_baseline"]=BASELINE_V1
    results["v2_joint_baseline"]=BASELINE_V2
    results["architecture"]="three private heads + separately detached fusion"
    # Check if fused prediction of selected best heads improves both metrics.
    r,f=results["raw"],results["fused"]
    results["fusion_improves_both"]=(f["depth_absrel"]<r["depth_absrel"]
                                      and f["normal_deg"]<r["normal_deg"])
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/"report_v3.json",results)
    with (out/"per_image_v3.csv").open("w",newline="",encoding="utf8") as fp:
        wr=csv.DictWriter(fp,fieldnames=["name","version","depth_absrel","normal_deg"])
        wr.writeheader();wr.writerows(rows)
    for idx in range(len(val)):
        sample=val[idx]
        b=_to_gpu(default_collate([sample]),device)
        with torch.inference_mode():
            with amp(device):
                raw=independent_outputs(heads,b["features"],b["rgb"])
                fused=fusion(raw,b["rgb"]) if fusion is not None else raw
        gt_z=sample["depth"].numpy();gt_n=sample["normal"].numpy()
        gt_s=sample["seg"].numpy()
        src=(sample["rgb"].permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
        gts=seg_view(gt_s.clip(0,12)).copy();gts[gt_s==255]=0
        panels=[src,depth_view(gt_z),normal_view(gt_n),gts]
        for raw_out in (raw,fused):
            maps=decoded(raw_out)
            panels.extend([
                depth_view(maps["depth"][0].float().cpu().numpy()),
                normal_view(maps["normal"][0].float().cpu().numpy()),
                seg_view(maps["seg"][0].cpu().numpy())
            ])
        canvas=Image.new("RGB",(len(panels)*256,256))
        for i,img in enumerate(panels):
            canvas.paste(Image.fromarray(np.asarray(img,dtype=np.uint8)).
                         resize((256,256)),(i*256,0))
        canvas.save(out/f"val_{idx:02d}.jpg",quality=88)
    folder=Path(args.save_dir)
    for name in ("metrics_v3.jsonl","validation_v3.jsonl","latest_v3.json"):
        if (folder/name).exists():(out/name).write_bytes((folder/name).read_bytes())
    for name in ("source_manifest.json","preparation_summary.json"):
        if (root/name).exists():(out/name).write_bytes((root/name).read_bytes())
    report=out.parent/"Jeveration_RGBD_V3_IndependentHeads_Report.zip"
    with zipfile.ZipFile(report,"w",compression=zipfile.ZIP_DEFLATED) as arc:
        for p in sorted(out.iterdir()):
            if p.is_file():arc.write(p,p.name)
    print("[v3] Report ZIP:",report,flush=True)
    print("[v3] Raw:",r,flush=True)
    print("[v3] Fused:",f,flush=True)
    print("[v3] Fusion better in both depth and normals:",results["fusion_improves_both"],flush=True)
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest="command",required=True)
    for task in ("fit","report"):
        x=sp.add_parser(task)
        x.add_argument("--root",default="/content/drive/MyDrive/Jeveration_GT/pilot")
        x.add_argument("--save-dir",default="/content/drive/MyDrive/Jeveration_GT/pilot/checkpoints_v3")
        x.add_argument("--local-cache-dir",default="/content/jeveration_gt_pilot_cache")
        x.add_argument("--cache-mode",choices=("ram","local"),default="ram")
        x.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
        if task=="fit":
            x.add_argument("--steps",type=int,default=1200)
            x.add_argument("--batch-size",type=int,default=1)
            x.add_argument("--grad-accum",type=int,default=16)
            x.add_argument("--lr-depth",type=float,default=7e-5)
            x.add_argument("--lr-normal",type=float,default=1.2e-4)
            x.add_argument("--lr-seg",type=float,default=1.8e-4)
            x.add_argument("--lr-fusion",type=float,default=6e-5)
            x.add_argument("--feature-drop",type=float,default=.03)
            x.add_argument("--fusion-start",type=int,default=400)
            x.add_argument("--warmup",type=int,default=50)
            x.add_argument("--eval-every",type=int,default=100)
            x.add_argument("--save-every",type=int,default=200)
            x.add_argument("--save-at",default="400,700,900")
            x.add_argument("--log-every",type=int,default=10)
            x.add_argument("--seed",type=int,default=42)
            x.add_argument("--resume",action="store_true")
        else:
            x.add_argument("--output",default="/content/Jeveration_V3_Report")
    return p


def main():
    a=parser().parse_args()
    {"fit":train,"report":report}[a.command](a)

if __name__=="__main__":main()
