"""CPU regression: analytic geometry init, output shapes and task isolation."""
import importlib,sys,tempfile,types,unittest
from pathlib import Path
import torch

root=Path(__file__).resolve().parents[1]
pkg=types.ModuleType("jeveration")
pkg.__path__=[str(root/"jeveration")]
sys.modules["jeveration"]=pkg
v2=types.ModuleType("jeveration.rgbd_gt_v2_train")
v2.GeometryCache=object
sys.modules[v2.__name__]=v2
model=importlib.import_module("jeveration.rgbd_frozen_probe")
train=importlib.import_module("jeveration.rgbd_frozen_probe_train")

class Tiny:
    def __init__(self,n=3):
        gen=torch.Generator().manual_seed(42)
        self.examples=[]
        for i in range(n):
            z=torch.zeros((512,512,3));z[...,2]=-1
            seg=torch.zeros((512,512),dtype=torch.int64);seg[:,256:]=4
            valid=torch.ones((512,512),dtype=torch.bool)
            rgb=torch.rand((3,512,512),generator=gen)
            self.examples.append({
                "features":torch.randn((8,32,32),generator=gen),
                "rgb":rgb,"original_rgb":rgb.clone(),
                "depth":torch.full((512,512),2.0+i/3),
                "normal":z,"seg":seg,
                "depth_valid":valid,"normal_valid":valid,"seg_valid":valid})
    def __len__(self):return len(self.examples)
    def __getitem__(self,i):return self.examples[i]

class ProbeTests(unittest.TestCase):
    def test_ridge_closed_form(self):
        x=torch.arange(20.).reshape(-1,1)/10
        w,b=train.linear_ridge(x,2*x+3,torch.ones(len(x)),1e-7)
        self.assertAlmostEqual(float(w[0,0]),2.,places=3)
        self.assertAlmostEqual(float(b[0]),3.,places=3)

    def test_correspondence_and_analytic_init(self):
        data=Tiny()
        b=train.to_batch(data[0],torch.device("cpu"))
        b["depth_valid"][:,:,0:24]=False
        gt=train.data128(b)
        self.assertFalse(bool(gt["dv"][:,:,0].any()))
        self.assertEqual(tuple(gt["normal"].shape),(1,3,128,128))
        probe=model.RGBDProbe(8,8)
        info=train.analytic_init(probe,data,torch.device("cpu"),stride=8)
        self.assertGreater(info["ridge_depth_rows"],0)
        self.assertGreater(info["prototype_counts"][4],0)
        out=probe(b["features"],b["rgb"])
        p=model.predictions_512(out,b["rgb"])
        self.assertEqual(tuple(p["depth"].shape),(1,512,512))
        self.assertEqual(tuple(p["normal"].shape),(1,512,512,3))
        self.assertEqual(tuple(p["seg"].shape),(1,512,512))
        self.assertTrue(bool(torch.isfinite(p["depth"]).all()))

    def test_independent_gradient_paths(self):
        data=Tiny(1);b=train.to_batch(data[0],torch.device("cpu"))
        gt=train.data128(b);probe=model.RGBDProbe(8,8)
        z=probe.feature_map(b["features"],b["rgb"]).detach()
        semweights=torch.ones(13)
        for task in ("depth","normal","seg"):
            probe.zero_grad(set_to_none=True)
            pred=getattr(probe,task)(z)
            loss=train.task_training_loss(task,pred,gt,probe,semweights)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for other in ("depth","normal","seg"):
                gradients=[v.grad for v in getattr(probe,other).parameters()]
                if task==other:self.assertTrue(any(g is not None for g in gradients))
                else:self.assertTrue(all(g is None for g in gradients))

    def test_snapshot_restore(self):
        net=model.RGBDProbe(8,8)
        opts={k:torch.optim.AdamW(getattr(net,k).parameters(),lr=1e-3)
              for k in ("depth","normal","seg")}
        cfg={"width":8,"seed":42}
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            train.save_full(d,20,net,opts,{"depth":.4},cfg,"cache-A")
            clone=model.RGBDProbe(8,8)
            opt2={k:torch.optim.AdamW(getattr(clone,k).parameters(),lr=1e-3)
                  for k in ("depth","normal","seg")}
            step,best=train.restore(d,clone,opt2,cfg,"cache-A",torch.device("cpu"))
            self.assertEqual(step,20)
            self.assertAlmostEqual(best["depth"],.4)
            self.assertTrue(torch.equal(net.depth.linear.weight,clone.depth.linear.weight))
            with self.assertRaises(RuntimeError):
                train.restore(d,clone,opt2,cfg,"cache-B",torch.device("cpu"))

if __name__=="__main__":unittest.main()
