"""Task-specific losses for three private RGBD heads and detached geometry fusion.

Depth and normal objectives are differentiated ONLY through their own head.
Segmentation has no gradient path into depth/normal, or vice versa.
Fusion is separately supervised and receives detached predictions.
"""
import torch
from torch.nn import functional as F

from .rgbd_gt_v2_losses import (masked_mean,masked_pool,depth_grad_loss,
                                camera_normals_from_log_depth,
                                segmentation_dice)

def depth_loss(depth_logits,b):
    target=b["depth"].float().clamp(.01,100).log()
    mask=b["depth_valid"].bool()
    pred=depth_logits.float()
    error=pred-target
    metric=masked_mean(F.smooth_l1_loss(pred,target,reduction="none",beta=.20),mask)
    mean=masked_mean(error,mask)
    mean_sq=masked_mean(error.square(),mask)
    silog=(mean_sq-.5*mean.square()).clamp_min(1e-8).sqrt()
    grad=depth_grad_loss(pred,target,mask)
    coarse=pred.new_zeros(())
    for factor,weight in ((2,.12),(4,.08)):
        p,v=masked_pool(pred,mask,factor)
        t,_=masked_pool(target,mask,factor)
        coarse=coarse+weight*masked_mean(F.smooth_l1_loss(p,t,reduction="none",beta=.2),v)
    loss=metric+.15*silog+.08*grad+coarse
    return loss,dict(metric=metric,silog=silog,grad=grad,coarse=coarse)


def normal_loss(normals,b):
    pred=F.normalize(normals.float(),dim=-1,eps=1e-6)
    target=F.normalize(b["normal"].float(),dim=-1,eps=1e-6)
    mask=b["normal_valid"].bool()
    cosine=masked_mean(1-(pred*target).sum(-1).clamp(-1,1),mask)
    # Edge supervision: preserve normal discontinuities, not blanket smoothing.
    vx=mask[:,:,1:] & mask[:,:,:-1]
    vy=mask[:,1:,:] & mask[:,:-1,:]
    gx=(pred[:,:,1:]-pred[:,:,:-1] -
        target[:,:,1:]+target[:,:,:-1]).abs().mean(-1)
    gy=(pred[:,1:,:]-pred[:,:-1,:] -
        target[:,1:,:]+target[:,:-1,:]).abs().mean(-1)
    grad=.5*(masked_mean(gx,vx)+masked_mean(gy,vy))
    loss=cosine+.08*grad
    return loss,dict(cosine=cosine,grad=grad)


def segmentation_loss(logits,b,weights=None):
    mask=b["seg_valid"].bool()&(b["seg"]!=255)
    if not bool(mask.any()):
        zero=logits.float().sum()*0
        return zero,dict(ce=zero,dice=zero)
    pixel=F.cross_entropy(logits.float(),b["seg"].long(),
                          weight=weights,ignore_index=255,reduction="none")
    # Normalize by valid pixel weights, not all valid pixels, to keep the CE
    # task scale stable as semantic class composition changes.
    if weights is None:denom=mask.sum().clamp_min(1)
    else:
        ids=b["seg"].long().clamp(0,weights.shape[0]-1)
        denom=(weights[ids]*mask).sum().clamp_min(1)
    ce=(pixel*mask).sum()/denom
    dice=segmentation_dice(logits,b["seg"].long(),mask)
    return .75*ce+.25*dice,dict(ce=ce,dice=dice)


def fusion_loss(refined,raw,b):
    """Only fused model's trainable corrections receive gradients."""
    d,_=depth_loss(refined["log_depth"],b)
    n,_=normal_loss(refined["normal_raw"],b)
    # Depth-derived normals versus labeled 3D normals at quarter resolution.
    lp,valid=masked_pool(refined["log_depth"].float(),
                         b["depth_valid"].bool(),4)
    from_depth=camera_normals_from_log_depth(lp)
    gt=F.normalize(b["normal"].float(),dim=-1,eps=1e-6)
    gt_mask=b["normal_valid"].bool()
    masked=gt*gt_mask[...,None]
    ngt=F.avg_pool2d(masked.permute(0,3,1,2),4).permute(0,2,3,1)
    ngt=F.normalize(ngt,dim=-1,eps=1e-6)
    mn=F.avg_pool2d(gt_mask[:,None].float(),4)[:,0]>.98
    valid=valid&mn
    m=(valid[:,1:-1,1:-1]&valid[:,2:,1:-1]&valid[:,:-2,1:-1]
       &valid[:,1:-1,2:]&valid[:,1:-1,:-2])
    geometry=masked_mean(1-(from_depth*ngt[:,1:-1,1:-1]).sum(-1).clamp(-1,1),m)
    # Constrain learned fusion to modest correction of independently trained heads.
    shift=(refined["log_depth"].float()-raw["log_depth"].detach().float()).square().mean()
    rn=F.normalize(refined["normal_raw"].float(),dim=-1,eps=1e-6)
    nn=F.normalize(raw["normal_raw"].detach().float(),dim=-1,eps=1e-6)
    correction=(1-(rn*nn).sum(-1).clamp(-1,1)).mean()
    total=d+n+.10*geometry+.35*shift+.30*correction
    return total,dict(depth=d,normal=n,geometry=geometry,shift=shift,correction=correction)
