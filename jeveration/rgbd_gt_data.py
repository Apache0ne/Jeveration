"""Ground-truth Hypersim preprocessing; source is rendered geometry, not model pseudo-labels.

Source subset: aaronyee/hypersim-train-subset, CC BY-SA 3.0.
The subset has ~150 paired frames in 15 scenes. Normals are derived from
ground-truth depth using the documented standard pinhole camera.
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

SOURCE_REPO = "aaronyee/hypersim-train-subset"
NYU13 = ("bed", "books", "ceiling", "chair", "floor", "furniture",
         "objects", "picture", "sofa", "table", "tv", "wall", "window")
# Official Hypersim code/cpp/tools/scene_annotation_tool/semantic_label_descs.csv;
# map only categories with an unambiguous target; 255 means ignore.
NYU40_TO_13 = {
    1: 11, 2: 4, 3: 5, 4: 0, 5: 3, 6: 8, 7: 9,
    9: 12, 10: 5, 11: 7, 12: 5, 14: 9, 15: 5,
    17: 5, 22: 2, 23: 1, 25: 10, 32: 5, 39: 5,
}
SEM_IGNORE = 255
SOURCE_W, SOURCE_H = 1024, 768
FOV_X_DEG = 60.0


def map_nyu40_to_nyu13(sem):
    sem = np.asarray(sem)
    result = np.full(sem.shape, SEM_IGNORE, dtype=np.uint8)
    for label_id, target in NYU40_TO_13.items():
        result[sem == label_id] = target
    return result


def center_square(a, size=512, mode="bilinear"):
    """Same image crop for RGB and all geometry layers, no shape distortion."""
    if a.shape[:2] != (SOURCE_H, SOURCE_W):
        raise ValueError(f"Expected Hypersim 768x1024 images, received {a.shape}")
    offset=(SOURCE_W-SOURCE_H)//2
    crop=np.ascontiguousarray(a[:,offset:offset+SOURCE_H])
    im=Image.fromarray(crop)
    resize=Image.Resampling.NEAREST if mode=="nearest" else Image.Resampling.BILINEAR
    out=np.array(im.resize((size,size),resample=resize))
    return out


def crop_float(a,size=512):
    import torch
    import torch.nn.functional as F
    a=np.asarray(a,dtype=np.float32)
    if a.shape!=(SOURCE_H,SOURCE_W):
        raise ValueError(f"Expected float map 768x1024, got {a.shape}")
    crop=a[:,(SOURCE_W-SOURCE_H)//2:(SOURCE_W+SOURCE_H)//2].copy()
    x=torch.from_numpy(crop).reshape(1,1,SOURCE_H,SOURCE_H)
    return F.interpolate(x,size=(size,size),mode="bilinear",align_corners=False)[0,0].numpy()


def distance_to_planar_depth(distance, valid=None, size=512):
    """Convert Hypersim Euclidean camera range to positive forward-Z depth.
    
    Hypersim standard pinhole camera has 1024x768 resolution and 60-degree
    horizontal FOV. Crop to central 768x768 then resize to 512x512. The
    focal length is scaled with the image. Do not mistake ray range for z.
    """
    d=np.asarray(distance,dtype=np.float32)
    if d.shape!=(SOURCE_H,SOURCE_W): raise ValueError(f"Unexpected range shape {d.shape}")
    m=np.isfinite(d)&(d>0)&(d<1000)
    if valid is not None:m&=np.asarray(valid,bool)
    safe=np.where(m,d,0)
    weighted=crop_float(safe,size)
    coverage=crop_float(m.astype(np.float32),size)
    valid_out=coverage>0.98
    distance_out=np.divide(weighted,coverage,out=np.zeros_like(weighted),where=coverage>0)
    focal=SOURCE_W/(2*math.tan(math.radians(FOV_X_DEG)/2))*(size/SOURCE_H)
    y,x=np.mgrid[:size,:size].astype(np.float32)
    ray=np.sqrt(1+((x+0.5-size/2)/focal)**2+((y+0.5-size/2)/focal)**2)
    z=distance_out/ray
    good=valid_out&np.isfinite(z)&(z>0.1)&(z<100)
    return np.where(good,z,0).astype(np.float32),good,focal


def depth_to_normals(z, valid, focal):
    """Finite-difference surface normals from ACTUAL depth, camera coords +Z forward.

    Normals face the optical center (-Z). Suppress discontinuities and borders.
    """
    z=np.asarray(z,np.float32)
    valid=np.asarray(valid,bool)
    h,w=z.shape
    y,x=np.mgrid[:h,:w].astype(np.float32)
    X=(x+.5-w/2)*z/focal
    Y=(y+.5-h/2)*z/focal
    pts=np.stack([X,Y,z],-1)
    dx=np.zeros_like(pts);dy=np.zeros_like(pts)
    dx[:,1:-1]=pts[:,2:]-pts[:,:-2]
    dy[1:-1]=pts[2:]-pts[:-2]
    normals=np.cross(dy,dx)
    norm=np.linalg.norm(normals,axis=-1)
    good=valid.copy()
    good[:,1:-1]&=valid[:,:-2]&valid[:,2:]
    good[1:-1]&=valid[:-2]&valid[2:]
    good[0]=False;good[-1]=False;good[:,0]=False;good[:,-1]=False
    # Mask surfaces across sharp depth jumps: prevent false edge normals.
    good[:,1:-1]&=(np.abs(z[:,2:]-z[:,:-2])/(z[:,1:-1]+1e-6))<.08
    good[1:-1]&=(np.abs(z[2:]-z[:-2])/(z[1:-1]+1e-6))<.08
    good&=norm>1e-7
    normals=np.divide(normals,norm[...,None],out=np.zeros_like(normals),where=norm[...,None]>1e-7)
    normals[~good]=0
    return normals.astype(np.float32),good


def read_h5(path):
    import h5py
    with h5py.File(path,"r") as file:
        return np.asarray(file["dataset"])


def paired_files(repo_id=SOURCE_REPO, token=None):
    """Discover exact public sample paths, and pair RGB, depth and semantics."""
    from huggingface_hub import list_repo_files
    paths=list_repo_files(repo_id,repo_type="dataset",token=token)
    import re
    pat=re.compile(r"^cam00_depth_semantic/(ai_\d{3}_\d{3})/images/(scene_cam_\d{2})_geometry_hdf5/(frame\.\d{4})\.depth_meters\.hdf5$")
    pairs=[]
    names=set(paths)
    for depth in sorted(paths):
        match=pat.fullmatch(depth)
        if not match:continue
        scene,cam,frame=match.groups()
        semantic=depth.replace("depth_meters.hdf5","semantic.hdf5")
        rgb=f"cam00_full/{scene}/images/{cam}_final_preview/{frame}.tonemap.jpg"
        if rgb in names and semantic in names:
            pairs.append(dict(scene=scene,frame=frame,rgb=rgb,depth=depth,semantic=semantic))
    if not pairs:
        examples=[p for p in paths if "depth_meters.hdf5" in p][:3]
        raise RuntimeError(f"No matched Hypersim RGB/depth/semantic frames. First depth names={examples}")
    return pairs


def download_paired_files(pairs,destination,repo_id=SOURCE_REPO):
    from huggingface_hub import hf_hub_download
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    from tqdm.auto import tqdm
    reused = 0
    downloaded = 0
    for pair in tqdm(pairs,desc="Ground-truth Hypersim files"):
        for kind in ("rgb","depth","semantic"):
            local_file = destination / pair[kind]
            # The existing 150-pair run saved source files on Drive. Reuse
            # these exact paths without another Hub HEAD request per file.
            if local_file.is_file() and local_file.stat().st_size > 0:
                reused += 1
                continue
            hf_hub_download(repo_id=repo_id,repo_type="dataset",filename=pair[kind],
                            local_dir=str(destination))
            downloaded += 1
    print(f"Source file reuse: {reused} previously downloaded, "
          f"{downloaded} newly fetched; none of the ground-truth sources were discarded.",
          flush=True)


def load_hypersim_pair(pair,root):
    root=Path(root)
    rgb=np.asarray(Image.open(root/pair["rgb"]).convert("RGB"),dtype=np.uint8)
    rgb=center_square(rgb)
    depth_raw=read_h5(root/pair["depth"])
    sem_raw=read_h5(root/pair["semantic"])
    depth,depth_mask,focal=distance_to_planar_depth(depth_raw)
    normals,normal_mask=depth_to_normals(depth,depth_mask,focal)
    sem=map_nyu40_to_nyu13(center_square(np.asarray(sem_raw,np.int16),mode="nearest"))
    return dict(rgb=rgb,depth=depth,normal=normals,seg=sem,
                depth_valid=depth_mask,normal_valid=normal_mask,
                seg_valid=(sem!=SEM_IGNORE),focal_px=float(focal))


def scene_disjoint_split(pairs,val_images=20,seed=42):
    """Hold out complete scenes, then select exactly 20 validation images."""
    import random
    if val_images!=20:raise ValueError("Exactly 20 held-out images are required")
    scenes=sorted({x["scene"] for x in pairs})
    random.Random(seed).shuffle(scenes)
    groups={scene:sorted((x for x in pairs if x["scene"]==scene),
                         key=lambda x:x["frame"]) for scene in scenes}
    # Hold out at least four entire scenes: ~40 candidate frames gives a
    # buffer for decoder-alignment rejection while keeping train/val disjoint.
    val_scenes=[]
    n=0
    for scene in scenes:
        val_scenes.append(scene);n+=len(groups[scene])
        if n>=2*val_images and len(val_scenes)>=4:break
    if len(val_scenes)==len(scenes):
        raise RuntimeError("Need at least one distinct training scene")
    validation=[x for sc in val_scenes for x in groups[sc]]
    training=[x for sc in scenes if sc not in val_scenes for x in groups[sc]]
    if len(validation)<val_images or not training:
        raise RuntimeError("Insufficient scene-disjoint pairs")
    assert not {x["scene"] for x in training}&{x["scene"] for x in validation}
    return training,validation,val_scenes
