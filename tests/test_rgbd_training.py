import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import numpy as np
import torch
from torch import nn
from jeveration.rgbd_sidecar import RGBDHead,decode_maps,tap_decoder,save_head,load_head
from jeveration.rgbd_train import Cache,loss_parts,select_prompts,make_parser

class FakeDataset:
    column_names=["Prompt"]
    def __getitem__(self,key):
        assert key=="Prompt"
        return [f"Scene {i}" for i in range(56)]+["Scene 0"]

class TestRGBD(unittest.TestCase):
    def test_split_exact_twenty_no_leak(self):
        with patch.dict(sys.modules,{"datasets":SimpleNamespace(
            load_dataset=lambda *a,**k:FakeDataset())}):
            train,val=select_prompts("fake",30,42)
            self.assertEqual(len(val),20)
            self.assertEqual(len(train),30)
            self.assertFalse(set(train)&set(val))
            self.assertEqual((train,val),select_prompts("fake",30,42))
    def test_split_rejects_insufficient_unique_prompts(self):
        with patch.dict(sys.modules,{"datasets":SimpleNamespace(
            load_dataset=lambda *a,**k:FakeDataset())}):
            with self.assertRaises(ValueError):select_prompts("fake",40,42)
    def test_decoder_capture_and_cleanup(self):
        d=nn.Module();d.ffn=nn.Conv2d(8,3,1)
        with tap_decoder(d) as captured: d.ffn(torch.randn(1,8,32,32))
        self.assertEqual(tuple(captured["features"].shape),(1,8,32,32))
        self.assertFalse(d.ffn._forward_pre_hooks)
    def test_shapes_and_checkpoint(self):
        h=RGBDHead(8)
        with torch.no_grad():
            raw=h(torch.randn(1,8,32,32),torch.rand(1,3,512,512))
            out=decode_maps(raw)
        self.assertEqual(tuple(out["depth"].shape),(1,512,512))
        self.assertEqual(tuple(out["normal"].shape),(1,512,512,3))
        self.assertEqual(tuple(out["seg"].shape),(1,512,512))
        self.assertTrue(torch.isfinite(out["depth"]).all())
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"head.pt";save_head(p,h)
            self.assertEqual(load_head(p,"cpu").in_channels,8)
    def test_cache_loss_and_cli_quality_values(self):
        with tempfile.TemporaryDirectory() as td:
            directory=Path(td)/"train";directory.mkdir()
            np.savez_compressed(directory/"00000.npz",features=np.zeros((8,32,32),np.float16),
                rgb=np.zeros((512,512,3),np.uint8),depth=np.ones((512,512),np.float16),
                normal=np.broadcast_to(np.array([0,0,1],np.float16),(512,512,3)),
                seg=np.zeros((512,512),np.uint8))
            b=Cache(td,"train")[0]
            raw=dict(log_depth=torch.zeros((1,512,512)),
                     normal_raw=torch.tensor([0.,0.,1.]).expand(1,512,512,3),
                     seg_logits=torch.ones((1,13,512,512)))
            total,parts=loss_parts(raw,{k:v.unsqueeze(0) for k,v in b.items()},.3)
            self.assertTrue(torch.isfinite(total))
            self.assertLess(parts["depth"].item(),1e-6)
            self.assertLess(parts["normal"].item(),1e-6)
        prep=make_parser().parse_args(["prepare"])
        fit=make_parser().parse_args(["fit"])
        self.assertEqual(prep.val_images,20)
        self.assertEqual(prep.seeds_per_prompt,3)
        self.assertEqual(fit.batch_size*fit.grad_accum,16)
        self.assertEqual(fit.steps,4000)

if __name__=="__main__":unittest.main()
