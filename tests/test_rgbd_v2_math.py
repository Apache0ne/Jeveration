"""Mathematical regression tests for actual RGBD geometry v2 losses."""
import importlib.util
import math
import unittest
from pathlib import Path

import torch


def load_file(name):
    path=Path(__file__).resolve().parents[1]/"jeveration"/name
    spec=importlib.util.spec_from_file_location(name.replace(".py",""),path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


losses=load_file("rgbd_gt_v2_losses.py")


class TestV2GeometryLosses(unittest.TestCase):
    def make_batch(self):
        h=w=32
        depth=torch.full((1,h,w),3.,dtype=torch.float32)
        normal=torch.zeros((1,h,w,3),dtype=torch.float32)
        normal[...,2]=-1.
        seg=torch.zeros((1,h,w),dtype=torch.int64)
        seg[:,:,w//2:]=11
        valid=torch.ones((1,h,w),dtype=torch.bool)
        return dict(depth=depth,normal=normal,seg=seg,
                    depth_valid=valid.clone(),normal_valid=valid.clone(),
                    seg_valid=valid.clone())

    def test_camera_normals_on_frontoparallel_plane(self):
        z=torch.full((1,32,32),math.log(3.0))
        n=losses.camera_normals_from_log_depth(z)
        self.assertEqual(tuple(n.shape),(1,30,30,3))
        self.assertLess(float((n[...,2]+1).abs().max()),1e-5)

    def test_perfect_geometry_has_low_depth_normal_loss(self):
        b=self.make_batch()
        raw=dict(
            log_depth=torch.full((1,32,32),math.log(3.),requires_grad=True),
            normal_raw=torch.tensor([0.,0.,-1.]).reshape(1,1,1,3).expand(1,32,32,3)
                               .clone().requires_grad_(True),
            seg_logits=torch.randn((1,13,32,32),requires_grad=True),
        )
        total,parts=losses.geometry_aware_loss(raw,b)
        self.assertTrue(torch.isfinite(total))
        self.assertLess(float(parts["depth"]),.01)
        self.assertLess(float(parts["normal"]),1e-5)
        self.assertLess(float(parts["geometry"]),1e-4)
        total.backward()
        self.assertTrue(torch.isfinite(raw["log_depth"].grad).all())
        self.assertTrue(torch.isfinite(raw["normal_raw"].grad).all())
        self.assertTrue(torch.isfinite(raw["seg_logits"].grad).all())

    def test_missing_depth_pixels_are_ignored(self):
        b=self.make_batch()
        b["depth_valid"][:, :16]=False
        b["normal_valid"][:, :16]=False
        raw=dict(
            log_depth=torch.full((1,32,32),math.log(3.),requires_grad=True),
            normal_raw=torch.tensor([0.,0.,-1.]).reshape(1,1,1,3).expand(1,32,32,3)
                               .clone().requires_grad_(True),
            seg_logits=torch.zeros((1,13,32,32),requires_grad=True))
        baseline,_=losses.geometry_aware_loss(raw,b)
        b["depth"][:, :16]=99.
        b["normal"][:, :16]=torch.tensor([1.,0.,0.])
        changed,_=losses.geometry_aware_loss(raw,b)
        self.assertAlmostEqual(float(baseline),float(changed),places=5)

    def test_missing_seg_labels_are_ignored(self):
        b=self.make_batch()
        b["seg"][:,:16]=255
        b["seg_valid"][:,:16]=False
        pred=torch.zeros((1,13,32,32),requires_grad=True)
        raw=dict(log_depth=torch.full((1,32,32),math.log(3.)),
                 normal_raw=torch.tensor([0.,0.,-1.]).reshape(1,1,1,3).expand(1,32,32,3),
                 seg_logits=pred)
        total,parts=losses.geometry_aware_loss(raw,b)
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(parts["seg"]))

    def test_metric_aggregation_and_miou(self):
        b=self.make_batch()
        raw=dict(log_depth=torch.full((1,32,32),math.log(3.)),
                 normal_raw=torch.tensor([0.,0.,-1.]).reshape(1,1,1,3).expand(1,32,32,3),
                 seg_logits=torch.zeros((1,13,32,32)))
        with torch.no_grad():
            # exactly correct segmentation.
            raw["seg_logits"][:,0,:,:16]=10
            raw["seg_logits"][:,11,:,16:]=10
            m=losses.validation_metrics(raw,b)
            out=losses.finalize_metrics(m)
        self.assertLess(out["depth_absrel"],.001)
        self.assertLess(out["normal_deg"],.5)
        self.assertAlmostEqual(out["seg_acc"],1.,places=5)
        self.assertAlmostEqual(out["seg_miou"],1.,places=5)


if __name__=="__main__":unittest.main()
