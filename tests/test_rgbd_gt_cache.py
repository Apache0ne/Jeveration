import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1] / "jeveration"


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cache = module("gt_cache", "rgbd_gt_cache.py")
preproc = module("gt_preprocess", "rgbd_gt_preprocess.py")


class TestTATiTokRangeFix(unittest.TestCase):
    def test_expected_pixel_range(self):
        rgb = torch.zeros(1, 3, 512, 512, dtype=torch.float32)
        rgb[:, 0] = 1.0
        rgb[:, 1] = 0.5
        output = preproc.tokenizer_pixels(rgb)
        self.assertEqual(tuple(output.shape), (1, 3, 256, 256))
        self.assertAlmostEqual(float(output[:, 0].mean()), 1.0, places=5)
        self.assertAlmostEqual(float(output[:, 1].mean()), 0.5, places=5)
        self.assertAlmostEqual(float(output[:, 2].mean()), 0.0, places=5)
        self.assertGreaterEqual(float(output.min()), 0)
        self.assertLessEqual(float(output.max()), 1)

    def test_reject_misnormalized_pixels(self):
        with self.assertRaises(ValueError):
            preproc.tokenizer_pixels(torch.full((1,3,512,512), -0.1))
        with self.assertRaises(ValueError):
            preproc.tokenizer_pixels(torch.full((1,3,512,512), 1.1))

    def test_cache_migration_keeps_downloads_and_archives_v1(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            old=dict(dataset="aaronyee/hypersim-train-subset",seed=42,min_ssim=0.4)
            current=dict(**old,cache_schema_version=cache.CACHE_VERSION,
                         encoder_input_range=cache.ENCODER_INPUT_RANGE)
            (root/"source_manifest.json").write_text(json.dumps(old))
            (root/"cache"/"train").mkdir(parents=True)
            (root/"cache"/"train"/"old.npz").write_bytes(b"previous cached features")
            (root/"cache"/"val").mkdir(parents=True)
            (root/"cache"/"val"/"old.rejected.json").write_text("{}")
            (root/"downloads").mkdir()
            (root/"downloads"/"ground_truth.hdf5").write_bytes(b"keep original")
            self.assertEqual(cache.ensure_cache_manifest(root,current),"migrated")
            archive=root/"cache_archived_bad_rgb_range_v1"
            self.assertTrue((archive/"cache"/"train"/"old.npz").exists())
            self.assertTrue((archive/"cache"/"val"/"old.rejected.json").exists())
            self.assertTrue((root/"downloads"/"ground_truth.hdf5").exists())
            self.assertFalse((root/"cache").exists())
            self.assertEqual(cache.ensure_cache_manifest(root,current),"resume")
            altered=dict(current,min_ssim=0.3)
            with self.assertRaises(RuntimeError):
                cache.ensure_cache_manifest(root,altered)


if __name__=="__main__":
    unittest.main()
