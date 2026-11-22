"""CPU regression checks for v3 task isolation and detached fusion."""
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from jeveration.rgbd_v3_model import (
    IndependentHead, GeometryFusion, decoded,save_component,load_component,
)
from jeveration.rgbd_v3_losses import (
    depth_loss,normal_loss,segmentation_loss,fusion_loss,
)


def batch(size=16):
    target=torch.full((1,size,size),3.,dtype=torch.float32)
    n=torch.zeros((1,size,size,3));n[...,2]=-1
    seg=torch.zeros((1,size,size),dtype=torch.int64)
    seg[:,:,size//2:]=11
    mask=torch.ones((1,size,size),dtype=torch.bool)
    return dict(depth=target,normal=n,seg=seg,
                depth_valid=mask,normal_valid=mask,
                seg_valid=mask,rgb=torch.rand((1,3,size,size)))


def raws(n=16,requires_grad=True):
    return dict(
        log_depth=torch.full((1,n,n),1.09).requires_grad_(requires_grad),
        normal_raw=torch.tensor([0.,0.,-1.]).reshape(1,1,1,3).expand(1,n,n,3)
                        .clone().requires_grad_(requires_grad),
        seg_logits=torch.randn(1,13,n,n,requires_grad=requires_grad),
    )


class TestV3Isolation(unittest.TestCase):
    def test_all_three_heads_have_private_parameter_storage(self):
        heads={t:IndependentHead(t,8) for t in ("depth","normal","seg")}
        ids={t:{id(p) for p in h.parameters()} for t,h in heads.items()}
        for a in ids:
            for b in ids:
                if a!=b:self.assertFalse(ids[a]&ids[b],f"{a} shares params with {b}")
        self.assertEqual(tuple(heads["depth"].output[-1].weight.shape[:1]),(1,))
        self.assertEqual(tuple(heads["normal"].output[-1].weight.shape[:1]),(3,))
        self.assertEqual(tuple(heads["seg"].output[-1].weight.shape[:1]),(13,))

    def test_per_task_losses_only_differentiate_supplied_task_tensor(self):
        b=batch();x=raws()
        ld,_=depth_loss(x["log_depth"],b)
        ld.backward()
        self.assertIsNotNone(x["log_depth"].grad)
        self.assertIsNone(x["normal_raw"].grad)
        self.assertIsNone(x["seg_logits"].grad)
        x=raws()
        ln,_=normal_loss(x["normal_raw"],b)
        ln.backward()
        self.assertIsNone(x["log_depth"].grad)
        self.assertIsNotNone(x["normal_raw"].grad)
        self.assertIsNone(x["seg_logits"].grad)
        x=raws()
        ls,_=segmentation_loss(x["seg_logits"],b)
        ls.backward()
        self.assertIsNone(x["log_depth"].grad)
        self.assertIsNone(x["normal_raw"].grad)
        self.assertIsNotNone(x["seg_logits"].grad)

    def test_fusion_gradient_does_not_reach_independent_heads(self):
        b=batch();x=raws()
        f=GeometryFusion()
        out=f(x,b["rgb"])
        loss,parts=fusion_loss(out,x,b)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNone(x["log_depth"].grad)
        self.assertIsNone(x["normal_raw"].grad)
        self.assertIsNone(x["seg_logits"].grad)
        self.assertTrue(any(p.grad is not None for p in f.parameters()))
        self.assertTrue(torch.isfinite(out["normal_raw"]).all())

    def test_atomic_model_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            h=IndependentHead("normal",8)
            path=Path(td)/"best_normal.pt"
            save_component(path,h,"normal",8,step=100,metrics={"normal_deg":35.})
            restored=load_component(path,"cpu")
            self.assertEqual(restored.task,"normal")
            self.assertEqual(restored.in_channels,8)
            self.assertFalse(any(p.requires_grad for p in restored.parameters()))
            self.assertTrue(all(torch.equal(a,b) for a,b in
                            zip(h.state_dict().values(),restored.state_dict().values())))

if __name__=="__main__":
    unittest.main()
