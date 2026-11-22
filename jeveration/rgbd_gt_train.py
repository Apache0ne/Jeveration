"""Train a Jeveration decoder-side geometry head on REAL Hypersim labels.

Subcommands:
  prepare --root /content/drive/MyDrive/Jeveration_GT/pilot
  fit --root /content/drive/MyDrive/Jeveration_GT/pilot --save-dir ...
  report --root ... --save-dir ... --output ...
This pipeline never calls Jeveration's pretrained dense teacher. Ground truth
is Hypersim camera range and NYU40 semantics; normals come from ground-truth
planar depth and the verified pinhole camera.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from .rgbd_gt_data import (SOURCE_REPO, paired_files, download_paired_files,
                           load_hypersim_pair, scene_disjoint_split)
from .rgbd_gt_encoder import FrozenTATiTokEncoder,image_alignment
from .rgbd_gt_cache import ensure_cache_manifest, CACHE_VERSION, ENCODER_INPUT_RANGE
from .rgbd_sidecar import RGBDHead, tap_decoder, decode_maps, save_head,load_head
from .rgbd_tracking import LiveTraining,checkpoint_dir,latest_state_path,save_steps

def _atomic_json(p, obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+".partial")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding="utf8")
    os.replace(tmp,p)


def _row_to_cache(row, codec,encoder,device,prompt):
    from .images import to_uint8
    actual_rgb=torch.from_numpy(row["rgb"].copy()).permute(2,0,1)[None].to(device).float()/255.
    with torch.inference_mode():
        codes=encoder(actual_rgb)
        with tap_decoder(codec.decoder) as captured:
            rec=codec.decode(codes,[prompt])
        result=to_uint8(rec)[0]
        feature=captured["features"].float().cpu().numpy()[0].astype(np.float16)
    quality=image_alignment(row["rgb"],result)
    return feature,result,quality


def prepare(a):
    from .tokenizer import ImageCodec
    from huggingface_hub import hf_hub_download
    root=Path(a.root).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    manifest=root/"source_manifest.json"
    pairs=paired_files(a.dataset)
    train,val,val_scenes=scene_disjoint_split(pairs,20,a.seed)
    spec=dict(dataset=a.dataset,seed=a.seed,
              cache_schema_version=CACHE_VERSION,
              encoder_input_range=ENCODER_INPUT_RANGE,
              validation_scenes=val_scenes,
              source_camera="pinhole 60deg 1024x768, square-center-crop 512",
              depth="ground-truth range converted to camera-axis Z",
              normals="derived by finite differences from camera-axis depth; discontinuity-masked",
              semantics="official NYU40 labels to 13-class conservative mapping; ambiguous ignored",
              min_ssim=a.min_ssim,train_count_candidate=len(train),
              validation_count_candidate=len(val))
    ensure_cache_manifest(root,spec)
    # Only fetch the selected train and validation scene frames, not the full Hypersim archive.
    download_paired_files(train+val,root/"downloads",a.dataset)
    checkpoints=root/"cache"
    for split in ("train","val"):(checkpoints/split).mkdir(parents=True,exist_ok=True)
    prepared=root/"cache_complete.json"
    if prepared.is_file():
        cached=json.loads(prepared.read_text(encoding="utf8"))
        if cached.get("validation_count")==20 and cached.get("training_count",0)>=40:
            print("Reusing completed ground-truth cache.",flush=True)
            return
    from .rgbd_train import download_weights
    _,decoder=download_weights(a.weights_dir,a.model)
    codec=ImageCodec().to(a.device)
    codec.load_decoder(decoder)
    codec.eval()
    encoder=FrozenTATiTokEncoder(codec).to(a.device).eval()
    print(f"Paired train candidates={len(train)}, validation={len(val)};"
          " no pseudo-labels; decoder alignment checked per image",flush=True)
    from tqdm.auto import tqdm
    preview_dir=root/"alignment_previews"
    preview_dir.mkdir(exist_ok=True)
    for split,entries in (("train",train),("val",val)):
        accepted=0
        for candidate_index,entry in enumerate(tqdm(entries,desc=f"GT decoder cache {split}")):
            if split=="val" and accepted>=20: break
            ident=f"{entry['scene']}_{entry['frame']}"
            path=checkpoints/split/f"{ident}.npz"
            reject=path.with_suffix(".rejected.json")
            if path.exists():
                accepted+=1
                continue
            if reject.exists():continue
            raw=load_hypersim_pair(entry,root/"downloads")
            prompt="A realistic indoor photograph of a furnished room."
            feat,rgb,alignment=_row_to_cache(raw,codec,encoder,a.device,prompt)
            # Inspect what the 128 codes reconstruct, even for rejected pairs.
            # Source and decode are aligned only when scene geometry is retained.
            if candidate_index%10==0:
                import PIL.ImageDraw
                sheet=Image.new("RGB",(1024,544),(30,30,30))
                sheet.paste(Image.fromarray(raw["rgb"]),(0,32))
                sheet.paste(Image.fromarray(rgb),(512,32))
                drawer=PIL.ImageDraw.Draw(sheet)
                drawer.text((8,8),"Hypersim source RGB",fill="white")
                drawer.text((520,8),f"Jeveration decode, SSIM {alignment['ssim']:.3f}",fill="white")
                sheet.save(preview_dir/f"{split}_{ident}.jpg",quality=85)
            if not np.isfinite(feat).all():
                raise RuntimeError(f"Non-finite decoder features: {ident}")
            if alignment["ssim"]<a.min_ssim:
                print(f"SKIP {ident}: decoder reconstruction SSIM {alignment['ssim']:.3f}"
                      f" below threshold {a.min_ssim}",flush=True)
                _atomic_json(reject,dict(reason="decoder_alignment",**alignment))
                continue
            valid=raw["depth_valid"].mean()
            valid_n=raw["normal_valid"].mean()
            if valid<0.5 or valid_n<0.4:
                print(f"SKIP {ident}: insufficient true geometric coverage d={valid:.3f}, n={valid_n:.3f}")
                _atomic_json(reject,dict(reason="ground_truth_coverage",depth_valid=float(valid),normal_valid=float(valid_n)))
                continue
            payload=dict(features=feat,rgb=rgb,
                original_rgb=raw["rgb"],
                depth=raw["depth"].astype(np.float16),
                normal=raw["normal"].astype(np.float16),
                seg=raw["seg"].astype(np.uint8),
                depth_valid=raw["depth_valid"].astype(np.uint8),
                normal_valid=raw["normal_valid"].astype(np.uint8),
                seg_valid=raw["seg_valid"].astype(np.uint8),
                alignment_ssim=np.float32(alignment["ssim"]),
                alignment_psnr=np.float32(alignment["psnr"]),
                scene=entry["scene"])
            tmp=path.with_suffix(".partial.npz")
            np.savez_compressed(tmp,**payload);os.replace(tmp,path)
            accepted+=1
        print(f"{split}: {accepted}/{len(entries)} aligned samples",flush=True)
    train_count=len(list((checkpoints/"train").glob("*.npz")))
    val_count=len(list((checkpoints/"val").glob("*.npz")))
    scores=[]
    for split in ("train","val"):
        d=checkpoints/split
        for fp in d.glob("*.rejected.json"):
            reason=json.loads(fp.read_text(encoding="utf8"))
            if "ssim" in reason:scores.append(float(reason["ssim"]))
        for fp in d.glob("*.npz"):
            with np.load(fp) as loaded:scores.append(float(loaded["alignment_ssim"]))
    _atomic_json(root/"preparation_summary.json",
                 dict(train=train_count,val=val_count,required_validation=20,
                      dataset=a.dataset,minimum_ssim=a.min_ssim,
                      ssims=dict(n=len(scores),median=float(np.median(scores)) if scores else None,
                                 p10=float(np.percentile(scores,10)) if scores else None,
                                 p90=float(np.percentile(scores,90)) if scores else None),
                      alignment_preview_folder=str(preview_dir)))
    if val_count!=20:
        raise RuntimeError(f"Got {val_count}/20 accepted validation examples. "
                           "Decoder alignment insufficient. Do not train until resolved.")
    if train_count<40:
        raise RuntimeError(f"Only {train_count} aligned training frames; need >=40 for pilot.")
    _atomic_json(prepared,dict(training_count=train_count,validation_count=val_count,
                               dataset=a.dataset,min_ssim=a.min_ssim))


class GTCache(Dataset):
    """The few hundred source examples can be RAM-resident on Colab's 53 GiB host.

    Previously every gradient-accumulation microbatch decompressed an NPZ
    directly from Google Drive's FUSE mount, leaving the GPU idle for minutes.
    Load once with a visible progress bar; only checkpoints stay on Drive.
    """
    def __init__(self,root,split,preload=False):
        self.files=sorted((Path(root)/"cache"/split).glob("*.npz"))
        if not self.files:raise FileNotFoundError(f"No {split} cache. Run prepare first.")
        self.ram=None
        if preload:
            from tqdm.auto import tqdm
            print(f"[GTCache] Caching {len(self.files)} {split} compressed examples into system RAM once...",flush=True)
            self.ram=[]
            for index in tqdm(range(len(self.files)),desc=f"Load {split} -> RAM",
                              mininterval=.5):
                self.ram.append(self._read(index))
            print(f"[GTCache] {split}: {len(self.ram)} samples ready in RAM; "
                  "no more Google Drive NPZ reads during optimizer updates.",flush=True)
    def __len__(self):return len(self.files)
    def _read(self,i):
        with np.load(self.files[i]) as z:
            arrays={key:z[key].copy() for key in (
                "features","rgb","depth","normal","seg",
                "depth_valid","normal_valid","seg_valid")}
        return dict(features=torch.from_numpy(arrays["features"].astype(np.float32)),
             rgb=torch.from_numpy(arrays["rgb"]).permute(2,0,1).float()/255.,
             depth=torch.from_numpy(arrays["depth"].astype(np.float32)),
             normal=torch.from_numpy(arrays["normal"].astype(np.float32)),
             seg=torch.from_numpy(arrays["seg"].astype(np.int64)),
             depth_valid=torch.from_numpy(arrays["depth_valid"].astype(bool)),
             normal_valid=torch.from_numpy(arrays["normal_valid"].astype(bool)),
             seg_valid=torch.from_numpy(arrays["seg_valid"].astype(bool)))
    def __getitem__(self,i):
        if self.ram is not None:
            return self.ram[i]
        return self._read(i)


def masked_mean(x,valid):
    if valid.shape!=x.shape:raise ValueError(f"Loss/mask shapes {x.shape}, {valid.shape}")
    return (x*valid.to(x.dtype)).sum()/valid.sum().clamp(min=1)


def gt_loss(raw,b,seg_weight=.3):
    depth_valid=b["depth_valid"].bool()
    normal_valid=b["normal_valid"].bool()
    sem_valid=b["seg_valid"].bool()
    true_log=b["depth"].float().clamp(min=.01,max=100).log()
    log_pred=raw["log_depth"].float()
    dloss=masked_mean(F.smooth_l1_loss(log_pred,true_log,reduction="none",beta=.2),depth_valid)
    # Validity masked gradients; no gradient penalties at invalid pixels.
    vx=depth_valid[:,1:,:]&depth_valid[:,:-1,:]
    vy=depth_valid[:,:,1:]&depth_valid[:,:,:-1]
    dx=log_pred[:,1:,:]-log_pred[:,:-1,:]
    dy=log_pred[:,:,1:]-log_pred[:,:,:-1]
    tx=true_log[:,1:,:]-true_log[:,:-1,:]
    ty=true_log[:,:,1:]-true_log[:,:,:-1]
    dloss+=.15*(masked_mean(F.smooth_l1_loss(dx,tx,reduction="none",beta=.1),vx)
                +masked_mean(F.smooth_l1_loss(dy,ty,reduction="none",beta=.1),vy))
    normal=F.normalize(raw["normal_raw"].float(),dim=-1,eps=1e-6)
    ref=F.normalize(b["normal"].float(),dim=-1,eps=1e-6)
    nloss=masked_mean(1-(normal*ref).sum(-1).clamp(-1,1),normal_valid)
    sloss=F.cross_entropy(raw["seg_logits"].float(),b["seg"].long(),
                          ignore_index=255,reduction="none")
    sloss=masked_mean(sloss,sem_valid)
    return dloss+nloss+seg_weight*sloss,dict(depth=dloss,normal=nloss,seg=sloss)


@torch.no_grad()
def evaluate(head,ds,device,seg_weight):
    from .rgbd_gt_data import SEM_IGNORE
    head.eval()
    total={key:0. for key in ("loss","depth","normal","seg","depth_absrel","normal_deg","seg_acc")}
    nvalid={key:0. for key in ("depth_absrel","normal_deg","seg_acc")}
    for b in DataLoader(ds,batch_size=1,shuffle=False,num_workers=0):
        b={k:v.to(device) for k,v in b.items()}
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
            raw=head(b["features"],b["rgb"])
        loss,components=gt_loss(raw,b,seg_weight)
        m=decode_maps(raw)
        v=b["depth_valid"]
        rel=(m["depth"].float()-b["depth"]).abs()/b["depth"].clamp(min=.01)
        v_norm=b["normal_valid"]
        target=F.normalize(b["normal"],dim=-1,eps=1e-6)
        angle=(m["normal"].float()*target).sum(-1).clamp(-1+1e-5,1-1e-5).acos()*180/math.pi
        vm=b["seg_valid"]
        results=dict(depth_absrel=masked_mean(rel,v),
                     normal_deg=masked_mean(angle,v_norm),
                     seg_acc=masked_mean((m["seg"]==b["seg"]).float(),vm))
        for key,t in dict(loss=loss,**components).items():total[key]+=float(t.item())
        for key,t in results.items():
            count=int(b[{"depth_absrel":"depth_valid","normal_deg":"normal_valid","seg_acc":"seg_valid"}[key]].sum())
            total[key]+=float(t.item())*count;nvalid[key]+=count
    for key in ("loss","depth","normal","seg"):total[key]/=len(ds)
    for key in nvalid:total[key]/=max(1,nvalid[key])
    total["samples"]=len(ds)
    return total


def fit(a):
    root=Path(a.root)
    print(f"[FIT] Starting model training; cache={root}, device={a.device}",flush=True)
    print("[FIT] Preloading NPZ data into system RAM so Drive latency is paid only once.",flush=True)
    ds=GTCache(root,"train",preload=a.cache_mode=="ram")
    val=GTCache(root,"val",preload=a.cache_mode=="ram")
    if len(val)!=20:raise RuntimeError(f"Expected 20 validation examples; found {len(val)}")
    if a.batch_size<1 or a.grad_accum<1 or a.steps<1 or a.warmup_steps<1:
        raise ValueError("Invalid optimizer settings")
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    device=torch.device(a.device)
    if device.type=="cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable GPU needed")
    width=ds[0]["features"].shape[0]
    print(f"[FIT] Loaded {len(ds)} training + {len(val)} held-out samples. "
          f"Decoder feature width={width}; building RGBD head on {device}...",flush=True)
    model=RGBDHead(width).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,
                                weight_decay=.01,betas=(.9,.999),eps=1e-8)
    saves=save_steps(a.save_at,a.steps)
    folder=checkpoint_dir(a.save_dir or root/"checkpoints")
    start=0;best=float("inf")
    if a.resume:
        from_file=latest_state_path(folder)
        x=torch.load(from_file,map_location="cpu",weights_only=True)
        if x["width"]!=width or x["steps_target"]!=a.steps:
            raise ValueError("Resume target/width mismatch")
        model.load_state_dict(x["model"]);optimizer.load_state_dict(x["optimizer"])
        for state in optimizer.state.values():
            for key,t in state.items():
                if isinstance(t,torch.Tensor):state[key]=t.to(device)
        start=int(x["step"]);best=float(x["best"])
        print(f"Resumed at step {start}",flush=True)
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=True,
                      num_workers=0,pin_memory=device.type=="cuda")
    iterator=iter(loader)
    print(f"[FIT] Ready: {a.steps} optimizer steps, {a.grad_accum} microbatches/step, "
          f"effective batch {a.batch_size*a.grad_accum}. "
          "The progress bar now advances every microbatch.",flush=True)
    tracker=LiveTraining(folder,a.steps,start,log_every=a.log_every,
                         keep_last=a.keep_last,progress=True,tensorboard=True,
                         microbatches_per_step=a.grad_accum)
    try:
        for step in range(start+1,a.steps+1):
            warm=min(a.warmup_steps,max(1,a.steps//10))
            scale=(step/warm if step<=warm else
                   .05+.95*.5*(1+math.cos(math.pi*(step-warm)/max(1,a.steps-warm))))
            for g in optimizer.param_groups:g["lr"]=a.lr*scale
            model.train();optimizer.zero_grad(set_to_none=True)
            loss_sum=0;part_sum={k:0. for k in ("depth","normal","seg")}
            for micro in range(a.grad_accum):
                if step==start+1 and micro==0:
                    print("[FIT] First microbatch: fetching cached sample...",flush=True)
                try:b=next(iterator)
                except StopIteration:
                    iterator=iter(loader);b=next(iterator)
                if step==start+1 and micro==0:
                    print("[FIT] First batch fetched. Transferring tensors to GPU...",flush=True)
                b={k:v.to(device,non_blocking=True) for k,v in b.items()}
                if random.random()<.5:
                    for key in ("features","rgb","depth","seg","depth_valid","normal_valid","seg_valid"):
                        b[key]=b[key].flip(-1)
                    b["normal"]=b["normal"].flip(-2);b["normal"][...,0]*=-1
                if step==start+1 and micro==0:
                    print("[FIT] First microbatch: model forward + geometry losses...",flush=True)
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
                    raw=model(b["features"],b["rgb"])
                    loss,parts=gt_loss(raw,b,a.seg_weight)
                if not torch.isfinite(loss):raise RuntimeError(f"Non-finite loss at {step}")
                if step==start+1 and micro==0:
                    print("[FIT] First microbatch: backward pass...",flush=True)
                (loss/a.grad_accum).backward()
                if step==start+1 and micro==0:
                    print("[FIT] First microbatch complete; future microbatches update the progress bar.",flush=True)
                tracker.microstep(step,micro+1)
                loss_sum+=float(loss.item())/a.grad_accum
                for k in part_sum:part_sum[k]+=float(parts[k].item())/a.grad_accum
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            tracker.step(step,loss_sum,optimizer.param_groups[0]["lr"],
                         part_sum,grad,(torch.cuda.memory_allocated(device)/2**30 if device.type=="cuda" else None))
            if step%a.eval_every==0 or step==a.steps:
                metrics=evaluate(model,val,device,a.seg_weight)
                tracker.validation(step,metrics)
                if metrics["loss"]<best:
                    best=float(metrics["loss"]);tracker.best(step,model,metrics,save_head)
            if (a.save_every and step%a.save_every==0) or step in saves or step==a.steps:
                tracker.checkpoint(step,model,optimizer,width,best,save_head)
    finally:tracker.close()
    print(f"Ground-truth run complete: {folder}",flush=True)


def report(a):
    import zipfile
    from .dense import depth_view,normal_view,seg_view
    root=Path(a.root);folder=Path(a.save_dir or root/"checkpoints")
    head=load_head(folder/"rgbd_best.pt",device=a.device)
    ds=GTCache(root,"val")
    metrics=evaluate(head,ds,torch.device(a.device),a.seg_weight)
    from .rgbd_gt_data import NYU13
    export=Path(a.output);export.mkdir(parents=True,exist_ok=True)
    metrics["evaluation"]="scene-held-out ground truth Hypersim"
    metrics["source_dataset"]=SOURCE_REPO
    metrics["source_license"]="CC BY-SA 3.0"
    metrics["normal_source"]="geometry-derived from real metric depth, not direct native Hypersim normal_cam"
    _atomic_json(export/"metrics.json",metrics)
    rows=[]
    for i,f in enumerate(ds.files):
        b=ds[i]
        with torch.no_grad():
            raw=head(b["features"][None].to(a.device),b["rgb"][None].to(a.device))
            pred=decode_maps(raw)
        def npy(x):return x.detach().float().cpu().numpy()
        pred_z=npy(pred["depth"][0]);pred_n=npy(pred["normal"][0])
        gt_z=npy(b["depth"]);gt_n=npy(b["normal"])
        normal_mask=b["normal_valid"].numpy().astype(bool)
        depth_mask=b["depth_valid"].numpy().astype(bool)
        e=float(np.mean((np.abs(pred_z-gt_z)/np.maximum(.01,gt_z))[depth_mask]))
        cos=np.sum(pred_n*gt_n,axis=-1).clip(-1,1)
        angle=float(np.mean(np.degrees(np.arccos(cos[normal_mask]))))
        rgb=(b["rgb"].permute(1,2,0).numpy()*255).clip(0,255).astype("uint8")
        sem=b["seg"].numpy()
        gt_seg_image=seg_view(sem.clip(0,12)).copy()
        gt_seg_image[~b["seg_valid"].numpy().astype(bool)]=0
        images=[rgb,depth_view(gt_z),depth_view(pred_z),
                normal_view(gt_n),normal_view(pred_n),
                gt_seg_image,seg_view(npy(pred["seg"][0]).astype("int"))]
        sheet=Image.new("RGB",(512*len(images),512))
        for j,img in enumerate(images):sheet.paste(Image.fromarray(img),(512*j,0))
        sheet.resize((512*len(images)//2,256)).save(export/f"sample_{i:02d}.jpg",quality=84)
        rows.append(dict(file=f.name,depth_absrel=e,normal_deg=angle))
    with (export/"per_image.csv").open("w",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    for name in ("source_manifest.json","preparation_summary.json"):
        p=root/name
        if p.exists():
            (export/name).write_bytes(p.read_bytes())
    for name in ("training_metrics.jsonl","validation.jsonl","events.jsonl","latest.json"):
        p=folder/name
        if p.exists():(export/name).write_bytes(p.read_bytes())
    reportzip=export.parent/"Jeveration_Hypersim_GT_Report.zip"
    with zipfile.ZipFile(reportzip,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for file in export.glob("*"):
            if file.is_file():z.write(file,file.name)
    print(f"Ground-truth validation report ZIP: {reportzip}",flush=True)
    return reportzip


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest="command",required=True)
    for name in ("prepare","fit","report"):
        x=subs.add_parser(name)
        x.add_argument("--root",required=True)
        x.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
        x.add_argument("--seed",type=int,default=42)
        if name=="prepare":
            x.add_argument("--dataset",default=SOURCE_REPO)
            x.add_argument("--min-ssim",type=float,default=.4)
            x.add_argument("--weights-dir",default="/content/weights")
            x.add_argument("--model",default="linlinlinlinsdada/Jeveration")
        else:
            x.add_argument("--save-dir",default=None)
            x.add_argument("--seg-weight",type=float,default=.3)
            if name=="fit":
                x.add_argument("--steps",type=int,default=1200)
                x.add_argument("--batch-size",type=int,default=1)
                x.add_argument("--cache-mode",choices=("ram","disk"),default="ram",
                               help="RAM caches samples once; disk repeatedly reads NPZs (slow on Drive)")
                x.add_argument("--grad-accum",type=int,default=16)
                x.add_argument("--lr",type=float,default=1.5e-4)
                x.add_argument("--warmup-steps",type=int,default=100)
                x.add_argument("--eval-every",type=int,default=100)
                x.add_argument("--save-every",type=int,default=200)
                x.add_argument("--save-at",default="500,900")
                x.add_argument("--log-every",type=int,default=10)
                x.add_argument("--keep-last",type=int,default=0)
                x.add_argument("--resume",action="store_true")
            else:x.add_argument("--output",required=True)
    return p


def main():
    a=parser().parse_args()
    {"prepare":prepare,"fit":fit,"report":report}[a.command](a)

if __name__=="__main__":main()
