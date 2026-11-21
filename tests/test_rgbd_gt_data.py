import importlib.util
import math
from pathlib import Path
import unittest
import numpy as np

FILE = Path(__file__).resolve().parents[1] / "jeveration/rgbd_gt_data.py"
spec=importlib.util.spec_from_file_location("rgbd_gt_data",FILE)
gt=importlib.util.module_from_spec(spec)
spec.loader.exec_module(gt)


class GroundTruthGeometryTest(unittest.TestCase):
    def test_source_camera_range_to_true_planar_z(self):
        h,w=768,1024
        focal=w/(2*math.tan(math.radians(60)/2))
        y,x=np.mgrid[:h,:w].astype(np.float32)
        ray=np.sqrt(1+((x+.5-w/2)/focal)**2+((y+.5-h/2)/focal)**2)
        distance=ray*3.0
        z,valid,f=gt.distance_to_planar_depth(distance)
        self.assertEqual(z.shape,(512,512))
        self.assertGreater(valid.mean(),.99)
        self.assertAlmostEqual(f,focal*512/768,places=3)
        self.assertLess(float(np.median(np.abs(z-3))),.005)
        n,vm=gt.depth_to_normals(z,valid,f)
        self.assertGreater(vm.mean(),.9)
        self.assertLess(float(np.median(abs(n[...,2][vm]+1))),.01)

    def test_semantics_are_conservatively_mapped(self):
        mapped=gt.map_nyu40_to_nyu13(np.array([1,2,4,11,22,23,25,31,40,-1]))
        self.assertEqual(mapped.tolist(),[11,4,0,7,2,1,10,255,255,255])

    def test_square_crop_preserves_alignment(self):
        a=np.zeros((768,1024,3),np.uint8)
        a[:,128:896]=10
        a[:,0:128]=255
        result=gt.center_square(a)
        self.assertEqual(result.shape,(512,512,3))
        self.assertTrue(np.all(result==10))

    def test_scene_disjoint_and_validation_candidates(self):
        pairs=[dict(scene=f"ai_001_{i:03d}",frame=f"frame.{j:04d}")
               for i in range(15) for j in range(10)]
        train,val,scenes=gt.scene_disjoint_split(pairs,20,42)
        self.assertGreaterEqual(len(val),40)
        self.assertEqual(len(set(scenes)),4)
        self.assertEqual(len(train),110)
        self.assertFalse({r["scene"] for r in train}&{r["scene"] for r in val})


if __name__=="__main__":
    unittest.main()
