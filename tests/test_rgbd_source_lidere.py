"""Regression tests for verbatim-source LiDeRe and Probe3D training integration.

Vendor blob hashes guarantee the upstream algorithms have NOT been rewritten.
Small-grid math tests cover exactly the real upstream tensor path.
"""
import hashlib
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType("jeveration")
pkg.__path__=[str(ROOT/"jeveration")]
sys.modules["jeveration"]=pkg
stub=types.ModuleType("jeveration.rgbd_gt_v2_train")
stub.GeometryCache=object
sys.modules[stub.__name__]=stub
model=importlib.import_module("jeveration.rgbd_source_lidere")
trainer=importlib.import_module("jeveration.rgbd_source_lidere_train")
loss=importlib.import_module("jeveration.vendor.probe3d_losses")

def git_blob(path):
    content=Path(path).read_bytes()
    return hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()

class SourceFaithfulnessTests(unittest.TestCase):
    def test_lidere_source_blob_is_unchanged(self):
        self.assertEqual(git_blob(ROOT/"jeveration/vendor/lidere.py"),
                         "5aafadc4eb21b8c69ad2d8bc863fac35d4528d58")

    def test_probe3d_source_blobs_are_unchanged(self):
        self.assertEqual(git_blob(ROOT/"jeveration/vendor/probe3d_losses.py"),
                         "df4e9ace89ec6a353746c309a05646175c0d27ae")
        self.assertEqual(git_blob(ROOT/"jeveration/vendor/probe3d_probes.py"),
                         "d926f10a746021efa1f904c0772776f2bad1ed4b")

    def test_native_core_shapes_and_no_task_gradient_leakage(self):
        torch.manual_seed(13)
        m=model.SourceLiDeReRGBD(in_channels=8,dim=16,base_size=16)
        x=torch.randn((1,8,32,32))
        expected={"depth":(1,1,64,64),
                  "normal":(1,3,64,64),
                  "seg":(1,13,64,64)}
        all_params={name:{id(p) for p in module.parameters() if p.requires_grad}
                    for name,module in m.heads.items()}
        for a in all_params:
            for b in all_params:
                if a!=b:self.assertFalse(all_params[a]&all_params[b])
        for name in ("depth","normal","seg"):
            m.zero_grad(set_to_none=True)
            out=m.task(name,x)
            self.assertEqual(tuple(out.shape),expected[name])
            self.assertTrue(bool(torch.isfinite(out).all()))
            out.mean().backward()
            for task,head in m.heads.items():
                grads=[p.grad for p in head.parameters() if p.requires_grad]
                if task==name:self.assertTrue(any(g is not None for g in grads))
                else:self.assertTrue(all(g is None for g in grads))

    def test_real_probe3d_depth_and_angular_loss(self):
        depth=torch.full((1,1,32,32),2.,requires_grad=True)
        gt=torch.full_like(depth,2.5)
        d=loss.DepthLoss(max_depth=80)(depth,gt.clone())
        self.assertTrue(bool(torch.isfinite(d)))
        d.backward()
        self.assertTrue(bool(torch.isfinite(depth.grad).all()))
        normals=torch.randn((1,3,32,32),requires_grad=True)
        ref=F.normalize(torch.randn_like(normals),dim=1)
        angular=loss.angular_loss(normals,ref,torch.ones((1,1,32,32),dtype=torch.bool))
        self.assertTrue(bool(torch.isfinite(angular)))
        angular.backward()
        self.assertTrue(bool(torch.isfinite(normals.grad).all()))

    def test_trainable_save_excludes_immutable_large_attention_tables(self):
        m=model.SourceLiDeReRGBD(in_channels=8,dim=16,base_size=16)
        state=trainer.trainable_state(m)
        self.assertTrue(state)
        self.assertFalse(any(".inp" in name or ".q" in name or ".feats_pe" in name
                             for name in state))
        restore=model.SourceLiDeReRGBD(in_channels=8,dim=16,base_size=16)
        trainer.load_trainable(restore,state)
        state2=trainer.trainable_state(restore)
        self.assertEqual(set(state),set(state2))
        self.assertTrue(all(torch.equal(state[k],state2[k]) for k in state))
        names=list(state)
        with self.assertRaises(RuntimeError):
            trainer.load_trainable(restore,{k:v for k,v in state.items() if k!=names[0]})

    def test_loss_wrapper_valid_masks_and_shapes(self):
        m=model.SourceLiDeReRGBD(in_channels=8,dim=16,base_size=16)
        x=torch.randn(1,8,32,32)
        gt_n=torch.zeros((1,512,512,3));gt_n[...,2]=-1
        valid=torch.ones((1,512,512),dtype=torch.bool)
        b={"features":x,"depth":torch.full((1,512,512),2.),
           "normal":gt_n,"seg":torch.zeros((1,512,512),dtype=torch.long),
           "depth_valid":valid,"normal_valid":valid,"seg_valid":valid}
        depth_fn=loss.DepthLoss(max_depth=80)
        for task in ("depth","normal","seg"):
            pred,target,mask=trainer.training_inputs(b,task,m)
            self.assertTrue(bool(mask.any()))
            expected_shape=(1,1,512,512) if task=="depth" else (
                (1,3,512,512) if task=="normal" else (1,13,512,512))
            self.assertEqual(tuple(pred.shape),expected_shape)
            value=trainer.task_loss(task,pred,target,mask,depth_fn)
            self.assertTrue(bool(torch.isfinite(value)))

if __name__=="__main__":
    unittest.main()
