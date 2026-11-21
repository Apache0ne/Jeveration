"""Distill Jeveration's trained dense predictions into the same-pass RGB decoder head.

Example:
  python -m jeveration.rgbd_train prepare --root /content/jev_rgbd
  python -m jeveration.rgbd_train fit --root /content/jev_rgbd
  python -m jeveration.rgbd_train infer --root /content/jev_rgbd --prompt "A red car"

The 20-image validation set measures agreement with the released NYUv2-trained
teacher, NOT metric 3D ground-truth accuracy.
"""
import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader

from .rgbd_sidecar import RGBDHead,tap_decoder,decode_maps,save_head
from .rgbd_tracking import LiveTraining, checkpoint_dir, latest_state_path, save_steps

MODEL="linlinlinlinsdada/Jeveration"
DATASET="nateraw/parti-prompts"

def download_weights(folder,repo=MODEL):
    from huggingface_hub import hf_hub_download
    folder=Path(folder).expanduser().resolve()
    folder.mkdir(parents=True,exist_ok=True)
    return tuple(hf_hub_download(repo,filename=f,local_dir=str(folder))
                 for f in ("jeveration.pt","decoder512.pt"))

def select_prompts(dataset, n_train, seed=42):
    from datasets import load_dataset
    ds=load_dataset(dataset,split="train")
    if "Prompt" not in ds.column_names:
        raise ValueError(f"{dataset} lacks the required 'Prompt' column: {ds.column_names}")
    prompts=list(dict.fromkeys(str(x).strip() for x in ds["Prompt"] if x and str(x).strip()))
    if n_train<1 or n_train+20>len(prompts):
        raise ValueError(f"Need >= {n_train+20} UNIQUE prompts; available={len(prompts)}")
    random.Random(seed).shuffle(prompts)
    return prompts[:n_train],prompts[n_train:n_train+20]

def _save_json(path,x):
    tmp=Path(str(path)+".tmp")
    tmp.write_text(json.dumps(x,indent=2,sort_keys=True),encoding="utf8")
    os.replace(tmp,path)

def prepare(a):
    from .pipeline import JeverationPipeline
    from .images import to_uint8
    if a.val_images!=20 or a.seeds_per_prompt<1 or a.seed<0:
        raise ValueError("Validation must be EXACTLY 20 images; positive train seeds; seed>=0")
    root=Path(a.root).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    train,val=select_prompts(a.dataset,a.train_prompts,a.seed)
    manifest=dict(dataset=a.dataset,teacher=a.model,train_prompts=train,val_prompts=val,
                  seeds_per_train_prompt=a.seeds_per_prompt,validation_images=20,seed=a.seed,
                  geometry="pseudo-labels from released dense model, not physical ground truth")
    manifest_file=root/"manifest.json"
    if manifest_file.exists():
        if json.loads(manifest_file.read_text(encoding="utf8"))!=manifest:
            raise RuntimeError("Different cache configuration; use another --root")
    else: _save_json(manifest_file,manifest)
    pending=[]
    for split,prompts in (("train",train),("val",val)):
        (root/split).mkdir(exist_ok=True)
        for index,prompt in enumerate(prompts):
            for variation in range(a.seeds_per_prompt if split=="train" else 1):
                path=root/split/f"{index:05d}_{variation:02d}.npz"
                if not path.exists(): pending.append((split,index,variation,prompt,path))
    if not pending:
        print("All samples already cached. No model download needed.")
        return
    print(f"Preparing {len(pending)} samples; {len(train)*a.seeds_per_prompt} train and 20 validation total",flush=True)
    w,d=download_weights(a.weights_dir,a.model)
    pipe=JeverationPipeline(w,decoder=d,device=a.device)
    pipe.model.eval()
    for done,(split,index,variation,prompt,path) in enumerate(pending,1):
        digest=hashlib.sha256(f"{a.seed}|{split}|{index}|{variation}|{prompt}".encode()).digest()
        sample_seed=int.from_bytes(digest[:4],"little")
        with torch.inference_mode():
            ids,mask=pipe.ptok([prompt])
            codes=pipe.model.sample_codes(ids.to(pipe.device),mask.to(pipe.device),
                         pipe.generator(sample_seed),heads_in_autocast=True)
            with tap_decoder(pipe.codec.decoder) as captured:
                image=pipe.codec.decode(codes,[prompt])
            rgb=to_uint8(image)[0]
            features=captured["features"].float().cpu().numpy()[0].astype(np.float16)
            labels={}
            # One task at a time reduces peak GPU memory.
            for task in ("depth","normal","seg"):
                labels[task]=pipe.dense_predict(rgb[None],tasks=(task,))[task][0].cpu().numpy()
        if rgb.shape!=(512,512,3) or features.ndim!=3 or features.shape[-2:]!=(32,32):
            raise RuntimeError(f"Unexpected decoder output shapes at {path}")
        if not (np.isfinite(features).all() and np.isfinite(labels["depth"]).all()
                and np.isfinite(labels["normal"]).all() and (labels["depth"]>0).all()):
            raise RuntimeError(f"Non-finite teacher output at {path}")
        if not (labels["seg"].min()>=0 and labels["seg"].max()<13):
            raise RuntimeError("NYUv2 segmentation class outside [0,12]")
        tmp=Path(str(path)+".partial.npz")
        np.savez_compressed(tmp,features=features,rgb=rgb,
             depth=labels["depth"].astype(np.float16),
             normal=labels["normal"].astype(np.float16),
             seg=labels["seg"].astype(np.uint8))
        os.replace(tmp,path)
        if done==1 or done%25==0 or done==len(pending):
            print(f"cached {done}/{len(pending)} [{split}]",flush=True)

class Cache(Dataset):
    def __init__(self,root,split):
        self.files=sorted((Path(root)/split).glob("*.npz"))
        if not self.files: raise FileNotFoundError(f"No {split} examples; first run prepare")
    def __len__(self):return len(self.files)
    def __getitem__(self,index):
        with np.load(self.files[index]) as z:
            features,rgb,depth,normal,seg=(z[k].copy() for k in ("features","rgb","depth","normal","seg"))
        if (features.ndim!=3 or features.shape[-2:]!=(32,32)
            or rgb.shape!=(512,512,3) or depth.shape!=(512,512)
            or normal.shape!=(512,512,3) or seg.shape!=(512,512)):
            raise RuntimeError(f"Invalid cache: {self.files[index]}")
        return dict(features=torch.from_numpy(features.astype(np.float32)),
                    rgb=torch.from_numpy(rgb).permute(2,0,1).float()/255.,
                    depth=torch.from_numpy(depth.astype(np.float32)),
                    normal=torch.from_numpy(normal.astype(np.float32)),
                    seg=torch.from_numpy(seg.astype(np.int64)))

def loss_parts(raw,b,seg_weight):
    td=b["depth"].float().clamp(min=math.exp(-5),max=math.exp(5)).log()
    pd=raw["log_depth"].float()
    ld=F.smooth_l1_loss(pd,td,beta=0.2)
    # Preserve teacher's depth boundaries; teacher dense output itself is low-resolution.
    gx=F.smooth_l1_loss(pd[:,:,1:]-pd[:,:,:-1],td[:,:,1:]-td[:,:,:-1],beta=0.1)
    gy=F.smooth_l1_loss(pd[:,1:,:]-pd[:,:-1,:],td[:,1:,:]-td[:,:-1,:],beta=0.1)
    ld=ld+0.15*(gx+gy)
    normal=F.normalize(raw["normal_raw"].float(),dim=-1,eps=1e-6)
    target=F.normalize(b["normal"].float(),dim=-1,eps=1e-6)
    ln=(1-(normal*target).sum(-1).clamp(-1,1)).mean()
    ls=F.cross_entropy(raw["seg_logits"].float(),b["seg"].long())
    return ld+ln+seg_weight*ls,dict(depth=ld,normal=ln,seg=ls)

@torch.no_grad()
def validate(head,ds,device,seg_weight):
    head.eval();tot={k:0. for k in ("loss","depth","normal","seg","normal_deg","depth_absrel","seg_acc")}
    for b in DataLoader(ds,batch_size=1,shuffle=False,num_workers=0):
        b={k:v.to(device) for k,v in b.items()}
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
            raw=head(b["features"],b["rgb"])
        loss,p=loss_parts(raw,b,seg_weight)
        maps=decode_maps(raw)
        norm=F.normalize(b["normal"],dim=-1,eps=1e-6)
        angle=(maps["normal"].float()*norm).sum(-1).clamp(-1+1e-6,1-1e-6).acos().mean()*180/math.pi
        rel=((maps["depth"]-b["depth"]).abs()/b["depth"].clamp_min(0.01)).mean()
        acc=(maps["seg"]==b["seg"]).float().mean()
        for k,v in dict(loss=loss,depth=p["depth"],normal=p["normal"],seg=p["seg"],
                        normal_deg=angle,depth_absrel=rel,seg_acc=acc).items():
            tot[k]+=float(v.item())
    return {**{k:v/len(ds) for k,v in tot.items()},"samples":len(ds)}

def fit(a):
    if a.steps < 1 or a.batch_size < 1 or a.grad_accum < 1 or a.lr <= 0 or a.warmup_steps < 1:
        raise ValueError("Invalid optimizer configuration")
    if (a.eval_every < 1 or a.save_every < 0 or a.log_every < 1
            or a.keep_last < 0 or not 0 < a.seg_weight <= 2):
        raise ValueError("Invalid validation, saving or logging configuration")
    requested_saves = save_steps(a.save_at, a.steps)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA support required; use a Colab A100 or L4")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    root = Path(a.root).expanduser().resolve()
    train, val = Cache(root, "train"), Cache(root, "val")
    if len(val) != 20:
        raise RuntimeError(f"Exactly 20 validation images required; found {len(val)}")
    width = train[0]["features"].shape[0]
    if width != val[0]["features"].shape[0]:
        raise RuntimeError("Train and validation decoder widths differ")
    head = RGBDHead(width).to(device).train()
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    out = checkpoint_dir(a.save_dir if a.save_dir else root / "checkpoints")
    start, best = 0, float("inf")
    if a.resume or a.resume_from:
        resume_path = Path(a.resume_from).expanduser().resolve() if a.resume_from else latest_state_path(out)
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        if saved["width"] != width or saved["steps_target"] != a.steps:
            raise RuntimeError("Resume checkpoint decoder width or target number of steps differs")
        head.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        start, best = int(saved["step"]), float(saved["best"])
        if not 0 <= start <= a.steps:
            raise RuntimeError(f"Invalid resume step {start}")
        print(f"Resumed optimizer and decoder geometry head from {resume_path} at {start}/{a.steps}", flush=True)
    loader = DataLoader(
        train, batch_size=a.batch_size, shuffle=True, num_workers=0,
        drop_last=False, pin_memory=device.type == "cuda"
    )
    iterator = iter(loader)
    warmup = min(a.warmup_steps, max(1, a.steps // 10))
    print(
        f"BF16={device.type=='cuda'} | training={len(train)} images | validation=20 "
        f"| effective batch={a.batch_size * a.grad_accum} | checkpoints={out} "
        f"| save every={a.save_every} | extra save steps={sorted(requested_saves)}",
        flush=True
    )
    tracker = LiveTraining(out, total=a.steps, start=start, log_every=a.log_every,
                           keep_last=a.keep_last, progress=a.progress,
                           tensorboard=a.tensorboard)
    try:
        for step in range(start + 1, a.steps + 1):
            if step <= warmup:
                scale = step / warmup
            else:
                scale = 0.05 + 0.95 * 0.5 * (
                    1 + math.cos(math.pi * (step - warmup) / max(1, a.steps - warmup))
                )
            for group in optimizer.param_groups:
                group["lr"] = a.lr * scale
            head.train()
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            component_total = {"depth": 0.0, "normal": 0.0, "seg": 0.0}
            for _ in range(a.grad_accum):
                try:
                    b = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    b = next(iterator)
                b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
                if a.flip and random.random() < 0.5:
                    b["features"] = b["features"].flip(-1)
                    b["rgb"] = b["rgb"].flip(-1)
                    b["depth"] = b["depth"].flip(-1)
                    b["seg"] = b["seg"].flip(-1)
                    b["normal"] = b["normal"].flip(-2)
                    b["normal"][..., 0] *= -1
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    raw = head(b["features"], b["rgb"])
                    loss, parts = loss_parts(raw, b, a.seg_weight)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at step {step}")
                (loss / a.grad_accum).backward()
                total += float(loss.item()) / a.grad_accum
                for name in component_total:
                    component_total[name] += float(parts[name].item()) / a.grad_accum
            grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()
            gpu_gib = (torch.cuda.memory_allocated(device) / (1024 ** 3)
                       if device.type == "cuda" else None)
            tracker.step(step, total, optimizer.param_groups[0]["lr"],
                         component_total, grad_norm, gpu_gib)
            if step % a.eval_every == 0 or step == a.steps:
                metrics = validate(head, val, device, a.seg_weight)
                tracker.validation(step, metrics)
                if metrics["loss"] < best:
                    best = float(metrics["loss"])
                    tracker.best(step, head, metrics, save_head)
            if ((a.save_every > 0 and step % a.save_every == 0)
                    or step in requested_saves or step == a.steps):
                tracker.checkpoint(step, head, optimizer, width, best, save_head)
    finally:
        tracker.close()
    print(f"Done. Drive/local checkpoints and metrics: {out}", flush=True)
    print("Teacher agreement is not ground-truth 3D accuracy.", flush=True)


def infer(a):
    from .rgbd_pipeline import JeverationRGBDPipeline
    from .dense import depth_view,normal_view,seg_view
    from PIL import Image
    path=Path(a.save_dir if a.save_dir else Path(a.root)/"checkpoints")/"rgbd_best.pt"
    if not path.exists(): raise FileNotFoundError(f"Missing trained sidecar at {path}")
    w,d=download_weights(a.weights_dir,a.model)
    pipe=JeverationRGBDPipeline(w,decoder=d,device=a.device).load_rgbd_head(path)
    result=pipe.text_to_rgbd([a.prompt],generator=pipe.generator(a.seed))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rgb,depth,normal,seg=(result[k][0] for k in ("rgb","depth","normal","seg"))
    Image.fromarray(rgb).save(out/"rgb.png")
    for k,v in (("depth",depth),("normal",normal),("seg",seg)):np.save(out/(k+".npy"),v)
    Image.fromarray(depth_view(depth)).save(out/"depth.png")
    Image.fromarray(normal_view(normal)).save(out/"normal.png")
    Image.fromarray(seg_view(seg,rgb)).save(out/"seg.png")
    print("Saved:",out.resolve())

def make_parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest="command",required=True)
    for name in ("prepare","fit","infer"):
        s=sub.add_parser(name)
        s.add_argument("--root",default="/content/jev_rgbd")
        s.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
        s.add_argument("--seed",type=int,default=42)
        if name!="fit":
            s.add_argument("--weights-dir",default="/content/weights")
            s.add_argument("--model",default=MODEL)
        if name=="prepare":
            s.add_argument("--dataset",default=DATASET)
            s.add_argument("--train-prompts",type=int,default=1400)
            s.add_argument("--seeds-per-prompt",type=int,default=3)
            s.add_argument("--val-images",type=int,default=20)
        elif name=="fit":
            s.add_argument("--steps",type=int,default=4000)
            s.add_argument("--batch-size",type=int,default=2)
            s.add_argument("--grad-accum",type=int,default=8)
            s.add_argument("--lr",type=float,default=1.5e-4)
            s.add_argument("--warmup-steps",type=int,default=200)
            s.add_argument("--seg-weight",type=float,default=0.3)
            s.add_argument("--eval-every",type=int,default=200)
            s.add_argument("--save-every",type=int,default=200)
            s.add_argument("--save-at",default="",help="Extra comma-separated checkpoint steps, e.g. 500,1200,3000")
            s.add_argument("--save-dir",default=None,help="Checkpoint destination, e.g. /content/drive/MyDrive/Jeveration_RGBD/checkpoints")
            s.add_argument("--log-every",type=int,default=10,help="Persist training metrics every N steps; tqdm updates each step")
            s.add_argument("--keep-last",type=int,default=0,help="Keep most recent N periodic checkpoints; 0 keeps all")
            s.add_argument("--progress",action=argparse.BooleanOptionalAction,default=True)
            s.add_argument("--tensorboard",action=argparse.BooleanOptionalAction,default=True)
            s.add_argument("--resume-from",default=None,help="Specific trainer_step_XXXXXX.pt checkpoint")
            s.add_argument("--flip",action=argparse.BooleanOptionalAction,default=True)
            s.add_argument("--resume",action="store_true")
        else:
            s.add_argument("--prompt",required=True)
            s.add_argument("--save-dir",default=None,help="Folder holding rgbd_best.pt")
            s.add_argument("--output",default="/content/jev_rgbd_preview")
    return p

def main():
    a=make_parser().parse_args()
    {"prepare":prepare,"fit":fit,"infer":infer}[a.command](a)

if __name__=="__main__":main()
