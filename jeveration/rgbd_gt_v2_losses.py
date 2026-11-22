"""Geometry-aware multi-task losses for the frozen-decoder RGBD pilot.

References:
- Metric3D v2 (2024): coupled depth/normal geometry.
- Virtual Normal Loss (2019): normal constraints from 3D points.
- Multi-scale depth gradients and scale-invariant log depth losses.
- CE + class-balanced soft Dice for minority semantic categories.

All labels are the SAME existing Hypersim 95/20 cache. No pseudo-labels.
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F

NUM_CLASSES = 13
# From the existing cached Hypersim pinhole geometry metadata.
FOCAL_512 = 1024.0 / (2.0 * math.tan(math.pi / 6.0)) * (512.0 / 768.0)


def masked_mean(x, mask):
    if x.shape != mask.shape:
        raise ValueError(f"Mask {tuple(mask.shape)} does not match {tuple(x.shape)}")
    mask = mask.to(x.dtype)
    return (x * mask).sum() / mask.sum().clamp_min(1)


def masked_pool(x, valid, factor):
    """Area-pool values WITHOUT leaking invalid depth through image borders."""
    v = valid.to(x.dtype)
    a = F.avg_pool2d(x * v, factor)
    w = F.avg_pool2d(v, factor)
    return a / w.clamp_min(1e-6), w > 0.98


def depth_grad_loss(pred, target, valid):
    vx = valid[:, :, 1:] & valid[:, :, :-1]
    vy = valid[:, 1:, :] & valid[:, :-1, :]
    dx = (pred[:, :, 1:] - pred[:, :, :-1])
    tx = (target[:, :, 1:] - target[:, :, :-1])
    dy = (pred[:, 1:, :] - pred[:, :-1, :])
    ty = (target[:, 1:, :] - target[:, :-1, :])
    return 0.5 * (masked_mean((dx - tx).abs(), vx) +
                  masked_mean((dy - ty).abs(), vy))


def camera_normals_from_log_depth(log_depth, focal=FOCAL_512):
    """Differentiable 3D surface normals from camera-axis depth.

    Input log_depth: [B,H,W]. Output: [B,H-2,W-2,3].
    +Z points away from camera, so front-facing planes have normal -Z.
    """
    b, h, w = log_depth.shape
    z = log_depth.float().clamp(-5, 5).exp()
    y, x = torch.meshgrid(
        torch.arange(h, device=z.device, dtype=torch.float32),
        torch.arange(w, device=z.device, dtype=torch.float32),
        indexing="ij"
    )
    intrinsics_scale = w / 512.0
    f = float(focal) * intrinsics_scale
    points = torch.stack((
        z * (x[None] + 0.5 - w / 2.0) / f,
        z * (y[None] + 0.5 - h / 2.0) / f,
        z
    ), dim=-1)
    dx = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
    dy = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
    return F.normalize(torch.cross(dy, dx, dim=-1), dim=-1, eps=1e-6)


def segmentation_dice(logits, seg, valid, scale=4):
    """Dice only over present GT classes, ignoring Hypersim unmapped 255."""
    if scale != 1:
        logits = F.avg_pool2d(logits, scale)
        seg = F.interpolate(seg[:, None].float(), size=logits.shape[-2:],
                            mode="nearest")[:, 0].long()
        valid = F.interpolate(valid[:, None].float(), size=logits.shape[-2:],
                              mode="nearest")[:, 0] > 0.5
    probs = F.softmax(logits.float(), dim=1)
    targets = F.one_hot(seg.clamp(0, NUM_CLASSES - 1), NUM_CLASSES).permute(0, 3, 1, 2)
    targets = targets.to(probs.dtype)
    v = valid[:, None].to(probs.dtype)
    intersection = (probs * targets * v).sum(dim=(0, 2, 3))
    union = ((probs + targets) * v).sum(dim=(0, 2, 3))
    present = (targets * v).sum(dim=(0, 2, 3)) > 0
    if not bool(present.any()):
        return logits.sum() * 0
    dice = (2 * intersection + 1) / (union + 1)
    return (1 - dice[present]).mean()


def geometry_aware_loss(raw, batch, class_weights=None):
    """Multi-task loss, with physically consistent geometry on 1/4-scale maps."""
    lp = raw["log_depth"].float().clamp(-5, 5)
    d = batch["depth"].float().clamp(0.01, 100).log()
    valid = batch["depth_valid"].bool()
    residual = lp - d
    base = masked_mean(F.smooth_l1_loss(lp, d, reduction="none", beta=.20), valid)

    # Dense SILog: complementary scale-invariant shape term, retain metric Huber.
    mean_d = masked_mean(residual, valid)
    mean_d2 = masked_mean(residual.square(), valid)
    silog = (mean_d2 - 0.5 * mean_d.square()).clamp_min(1e-8).sqrt()
    depth = base + 0.15 * silog

    for factor, weight in ((2, 0.12), (4, 0.08)):
        po, vm = masked_pool(lp, valid, factor)
        tg, _ = masked_pool(d, valid, factor)
        depth = depth + weight * masked_mean(
            F.smooth_l1_loss(po, tg, reduction="none", beta=.20), vm
        )
    depth = depth + 0.08 * depth_grad_loss(lp, d, valid)

    npred = F.normalize(raw["normal_raw"].float(), dim=-1, eps=1e-6)
    ngt = F.normalize(batch["normal"].float(), dim=-1, eps=1e-6)
    normal_valid = batch["normal_valid"].bool()
    normal = masked_mean(
        1 - (npred * ngt).sum(-1).clamp(-1, 1), normal_valid
    )

    # 1/4-resolution differentiable geometry: directly couples both heads.
    # We require valid depth and normal pixels in every pooled footprint.
    lp4, dv4 = masked_pool(lp, valid, 4)
    dpn = camera_normals_from_log_depth(lp4)  # [B,126,126,3]
    n4 = F.avg_pool2d(
        npred.permute(0, 3, 1, 2), 4
    ).permute(0, 2, 3, 1)
    n4 = F.normalize(n4.float(), dim=-1, eps=1e-6)
    ngt4 = F.avg_pool2d(
        (ngt * normal_valid[..., None]).permute(0, 3, 1, 2), 4
    ).permute(0, 2, 3, 1)
    ngt4 = F.normalize(ngt4.float(), dim=-1, eps=1e-6)
    nv4 = F.avg_pool2d(normal_valid[:, None].float(), 4)[:, 0] > 0.98
    both = dv4 & nv4
    geom_mask = (both[:, 1:-1, 1:-1] &
                 both[:, 2:, 1:-1] & both[:, :-2, 1:-1] &
                 both[:, 1:-1, 2:] & both[:, 1:-1, :-2])
    d_geom = masked_mean(
        1 - (dpn * ngt4[:, 1:-1, 1:-1]).sum(-1).clamp(-1, 1), geom_mask
    )
    consistency = masked_mean(
        1 - (dpn * n4[:, 1:-1, 1:-1]).sum(-1).clamp(-1, 1), geom_mask
    )
    geom = 0.16 * d_geom + 0.06 * consistency

    target = batch["seg"].long()
    seg_valid = batch["seg_valid"].bool() & (target != 255)
    if not bool(seg_valid.any()):
        semantic = raw["seg_logits"].float().sum() * 0
        ce, dice = semantic, semantic
    else:
        ce = F.cross_entropy(
            raw["seg_logits"].float(), target, weight=class_weights,
            ignore_index=255, reduction="none"
        )
        ce = masked_mean(ce, seg_valid)
        dice = segmentation_dice(raw["seg_logits"], target, seg_valid)
        semantic = 0.75 * ce + 0.25 * dice

    # Geometry first; segmentation remains auxiliary.
    total = depth + normal + geom + 0.25 * semantic
    return total, {
        "depth": depth, "normal": normal, "geometry": geom, "seg": semantic,
        "seg_ce": ce, "seg_dice": dice, "silog": silog
    }


@torch.no_grad()
def validation_metrics(raw, batch):
    """Accumulate valid-pixel statistics in a reproducible 20-image evaluation."""
    pred_z = raw["log_depth"].float().clamp(-5, 5).exp()
    valid = batch["depth_valid"].bool()
    gt_z = batch["depth"].float()
    absrel_num = ((pred_z - gt_z).abs() / gt_z.clamp_min(0.01))[valid].sum()
    nmask = batch["normal_valid"].bool()
    npred = F.normalize(raw["normal_raw"].float(), dim=-1, eps=1e-6)
    ngt = F.normalize(batch["normal"].float(), dim=-1, eps=1e-6)
    degrees = (npred * ngt).sum(-1).clamp(-1 + 1e-5, 1 - 1e-5).acos() * (180 / math.pi)
    seg = batch["seg"].long()
    smask = batch["seg_valid"].bool() & (seg != 255)
    pred = raw["seg_logits"].argmax(dim=1)
    g = seg[smask]
    p = pred[smask]
    confusion = torch.bincount(
        g * NUM_CLASSES + p, minlength=NUM_CLASSES * NUM_CLASSES
    ).view(NUM_CLASSES, NUM_CLASSES).float()
    return {
        "absrel_sum": float(absrel_num.item()), "depth_pixels": int(valid.sum()),
        "normal_deg_sum": float(degrees[nmask].sum().item()),
        "normal_pixels": int(nmask.sum()), "confusion": confusion.cpu()
    }


def finalize_metrics(acc):
    cm = acc["confusion"]
    inter = cm.diag()
    union = cm.sum(0) + cm.sum(1) - inter
    present = cm.sum(1) > 0
    miou = (inter[present] / union[present].clamp_min(1)).mean().item()
    acc_seg = inter.sum() / cm.sum().clamp_min(1)
    return dict(
        depth_absrel=acc["absrel_sum"] / max(1, acc["depth_pixels"]),
        normal_deg=acc["normal_deg_sum"] / max(1, acc["normal_pixels"]),
        seg_miou=float(miou), seg_acc=float(acc_seg),
        valid_depth_pixels=acc["depth_pixels"], valid_normal_pixels=acc["normal_pixels"]
    )
