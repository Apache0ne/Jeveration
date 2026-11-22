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

@torch.no_grad()
def evaluate(model,ds,device):
    """Pixel-weighted depth, normal and mIoU on exactly 20 held-out frames."""
    model.eval();dtotal=0.;dn=0;ntotal=0.;nn=0
    cm=torch.zeros((CLASSES,CLASSES),dtype=torch.float64)
    rows=[]
    for i in range(len(ds)):
        b=to_batch(ds[i],device)
        pred=predictions_512(model(b["features"],b["rgb"]),b["rgb"])
        dv=b["depth_valid"].bool();nv=b["normal_valid"].bool()
        sv=b["seg_valid"].bool()&(b["seg"]!=255)
        rel=(pred["depth"]-b["depth"]).abs()/b["depth"].clamp_min(.01)
        dot=(pred["normal"]*F.normalize(b["normal"].float(),dim=-1)).sum(-1)
        angle=dot.clamp(-1+1e-6,1-1e-6).acos()*(180/math.pi)
        dr=float(rel[dv].mean()) if bool(dv.any()) else float("nan")
        nr=float(angle[nv].mean()) if bool(nv.any()) else float("nan")
        dtotal+=float(rel[dv].sum());dn+=int(dv.sum())
        ntotal+=float(angle[nv].sum());nn+=int(nv.sum())
        label=b["seg"][sv].long();predicted=pred["seg"][sv]
        if label.numel():
            cm+=torch.bincount((label*CLASSES+predicted).cpu(),minlength=CLASSES**2).reshape(CLASSES,CLASSES)
        rows.append({"file":ds.files[i].name,"depth_absrel":dr,"normal_deg":nr})
    present=cm.sum(1)>0
    union=cm.sum(0)+cm.sum(1)-cm.diag()
    miou=(cm.diag()[present]/union[present].clamp_min(1)).mean()
    metrics={"depth_absrel":dtotal/max(1,dn),"normal_deg":ntotal/max(1,nn),
             "seg_acc":float(cm.diag().sum()/cm.sum().clamp_min(1)),
             "seg_miou":float(miou),"validation_images":len(ds),
             "depth_pixels":dn,"normal_pixels":nn}
    return metrics,rows

def save_full(folder,step,model,optimizers,best,config,identifier):
    state={"format":"frozen-probe-v1","step":step,"target_steps":TOTAL_STEPS,
           "cache_id":identifier,"config":config,
           "model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
           "optimizers":{k:o.state_dict() for k,o in optimizers.items()},
           "best":best,"python_random":random.getstate(),"torch_random":torch.get_rng_state()}
    path=folder/f"trainer_probe_step_{step:06d}.pt"
    atomic_torch_save(path,state)
    atomic_json(folder/"latest_probe.json",{"trainer":path.name,"step":step})

def restore(folder,model,optims,config,identifier,device):
    pointer=json.loads((folder/"latest_probe.json").read_text())
    state=torch.load(folder/pointer["trainer"],map_location="cpu",weights_only=True)
    if state.get("cache_id")!=identifier or state.get("config")!=config:
        raise RuntimeError("Cannot resume: training parameters or dataset changed")
    model.load_state_dict(state["model"],strict=True)
    for task,opt in optims.items():
        opt.load_state_dict(state["optimizers"][task])
        for v in opt.state.values():
            for key,item in v.items():
                if torch.is_tensor(item):v[key]=item.to(device)
    random.setstate(state["python_random"])
    torch.set_rng_state(state["torch_random"])
    return int(state["step"]),state["best"]

def load_best(folder,width,device):
    folder=Path(folder)
    init=torch.load(folder/"probe_initializer.pt",map_location="cpu",weights_only=True)
    model=RGBDProbe(width,init["projected"],init["seed"]).to(device)
    model.encoder.load_state_dict(init["encoder"])
    for task in ("depth","normal","seg"):
        state=torch.load(folder/f"best_{task}.pt",map_location="cpu",weights_only=True)
        getattr(model,task).load_state_dict(state["weights"],strict=True)
    return model.eval().requires_grad_(False)

def train(a):
    if a.steps!=TOTAL_STEPS:raise ValueError("Same-cache comparison requires 1200 steps")
    if a.batch_size!=1 or a.grad_accum<1 or a.eval_every<1 or a.log_every<1:
        raise ValueError("Invalid optimizer settings")
    torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed)
    device=torch.device(a.device)
    if device.type=="cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requires BF16-capable GPU")
    root=Path(a.root).expanduser().resolve()
    local=stage_groundtruth_cache(root,a.local_cache_dir)
    cache_id=check_splits(local)
    trainset=GeometryCache(local,"train",preload=a.cache_mode=="ram")
    valset=GeometryCache(local,"val",preload=a.cache_mode=="ram")
    width=trainset[0]["features"].shape[0]
    model=RGBDProbe(width,a.projected,a.seed).to(device)
    out=checkpoint_dir(a.save_dir)
    config={"seed":a.seed,"width":width,"projected":a.projected,
            "lr_depth":a.lr_depth,"lr_normal":a.lr_normal,"lr_seg":a.lr_seg,
            "grad_accum":a.grad_accum,"init_stride":a.init_stride,"ridge_reg":a.ridge_reg}
    if not a.resume:
        meta=analytic_init(model,trainset,device,a.init_stride,a.ridge_reg)
        atomic_torch_save(out/"probe_initializer.pt",{
            "encoder":{k:v.detach().cpu().clone() for k,v in model.encoder.state_dict().items()},
            "projected":a.projected,"seed":a.seed,"metadata":meta})
        atomic_json(out/"initialization.json",meta)
    else:
        init=torch.load(out/"probe_initializer.pt",map_location="cpu",weights_only=True)
        model.encoder.load_state_dict(init["encoder"])
    opts={
        "depth":torch.optim.AdamW(model.depth.parameters(),lr=a.lr_depth,weight_decay=.04),
        "normal":torch.optim.AdamW(model.normal.parameters(),lr=a.lr_normal,weight_decay=.04),
        "seg":torch.optim.AdamW(model.seg.parameters(),lr=a.lr_seg,weight_decay=.03)}
    start=0;best={"depth":float("inf"),"normal":float("inf"),"seg":float("-inf")}
    if a.resume:start,best=restore(out,model,opts,config,cache_id,device)
    class_counts=torch.zeros(CLASSES,dtype=torch.float64)
    for sample in trainset:
        valid=sample["seg_valid"]&(sample["seg"]!=255)
        if bool(valid.any()):
            class_counts+=torch.bincount(sample["seg"][valid].long(),minlength=CLASSES).double()
    probability=class_counts/class_counts.sum().clamp_min(1)
    weights=probability.clamp_min(1e-8).rsqrt().clamp(max=8.)
    weights[class_counts==0]=0
    weights=weights.float().to(device)
    print(f"[probe] Same 95 train / 20 validation, 1200 updates; decoder width {width}",flush=True)
    print("[probe] Frozen features, analytic ridge depth/normal, semantic prototypes",flush=True)
    print(f"[probe] Checkpoints: {out}",flush=True)
    if start==0:
        baseline,_=evaluate(model,valset,device)
        atomic_json(out/"analytic_validation.json",baseline)
        print("[probe] Step 0 (analytical baseline):",baseline,flush=True)
        best={"depth":float(baseline["depth_absrel"]),
              "normal":float(baseline["normal_deg"]),
              "seg":float(baseline["seg_miou"])}
        for task,score in best.items():
            atomic_torch_save(out/f"best_{task}.pt",{
                "step":0,"metric":score,
                "weights":{k:v.detach().cpu().clone()
                           for k,v in getattr(model,task).state_dict().items()}})
    log=(out/"training_probe.jsonl").open("a",encoding="utf8",buffering=1)
    bar=tqdm(total=TOTAL_STEPS*a.grad_accum,initial=start*a.grad_accum,
             desc="Frozen probe microbatches",dynamic_ncols=True)
    saves=save_steps(a.save_at,TOTAL_STEPS)
    try:
        for step in range(start+1,TOTAL_STEPS+1):
            scale=(step/max(1,a.warmup) if step<=a.warmup else
                   .12+.88*.5*(1+math.cos(math.pi*(step-a.warmup)/
                                           max(1,TOTAL_STEPS-a.warmup))))
            for task,opt in opts.items():
                for group in opt.param_groups:
                    group["lr"]=getattr(a,"lr_"+task)*scale
                opt.zero_grad(set_to_none=True)
            total={"depth":0.,"normal":0.,"seg":0.}
            model.train();model.encoder.eval()
            for micro in range(a.grad_accum):
                rand=random.Random(a.seed+step*100003+micro*127)
                b=to_batch(trainset[rand.randrange(len(trainset))],device)
                if rand.random()<.5:
                    for k in ("features","rgb","original_rgb","depth","seg",
                              "depth_valid","normal_valid","seg_valid"):
                        b[k]=b[k].flip(-1)
                    b["normal"]=b["normal"].flip(-2)
                    b["normal"][...,0]*=-1
                with torch.no_grad():
                    feature=model.feature_map(b["features"],b["rgb"]).detach()
                    gt=data128(b)
                for task in ("depth","normal","seg"):
                    with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
                        prediction=getattr(model,task)(feature)
                        loss=task_training_loss(task,prediction,gt,model,weights)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError(f"Nonfinite {task} loss at step {step}")
                    (loss/a.grad_accum).backward()
                    total[task]+=float(loss.detach())/a.grad_accum
                bar.update(1)
            for task,opt in opts.items():
                norm=nn.utils.clip_grad_norm_(getattr(model,task).parameters(),1.)
                if not bool(torch.isfinite(norm)):
                    raise RuntimeError(f"Nonfinite {task} gradients")
                opt.step()
            bar.set_postfix(**{k:f"{v:.3f}" for k,v in total.items()})
            if step==1 or step%a.log_every==0 or step==TOTAL_STEPS:
                log.write(json.dumps({"step":step,**total})+"\n")
            if step%a.eval_every==0 or step==TOTAL_STEPS:
                metrics,_=evaluate(model,valset,device)
                with (out/"validation_probe.jsonl").open("a",encoding="utf8") as stream:
                    stream.write(json.dumps({"step":step,**metrics})+"\n")
                print(f"[probe] validation {step}: AbsRel={metrics['depth_absrel']:.4f}, "
                      f"normals={metrics['normal_deg']:.2f}°, "
                      f"seg acc={metrics['seg_acc']:.4f}, mIoU={metrics['seg_miou']:.4f}",flush=True)
                for task,value in {"depth":metrics["depth_absrel"],
                                   "normal":metrics["normal_deg"],
                                   "seg":metrics["seg_miou"]}.items():
                    improved=(value>best[task] if task=="seg" else value<best[task])
                    if improved:
                        best[task]=float(value)
                        atomic_torch_save(out/f"best_{task}.pt",{
                            "step":step,"metric":value,
                            "weights":{k:v.detach().cpu().clone()
                                       for k,v in getattr(model,task).state_dict().items()}})
                        print(f"[probe] Best {task} updated: step {step}",flush=True)
            if ((a.save_every and step%a.save_every==0)
                 or step in saves or step==TOTAL_STEPS):
                save_full(out,step,model,opts,best,config,cache_id)
                print(f"[probe] Saved three optimizers and weights at step {step}",flush=True)
    finally:
        log.close();bar.close()
    print("[probe] Training complete; individual best readouts:",out,flush=True)

