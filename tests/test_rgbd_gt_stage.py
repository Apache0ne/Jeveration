"""Check Drive -> /content staging on a small synthetic prepared cache."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

PATH=Path(__file__).resolve().parents[1]/"jeveration/rgbd_gt_stage.py"
spec=importlib.util.spec_from_file_location("rgbd_gt_stage",PATH)
stage=importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


class TestGTStaging(unittest.TestCase):
    def make_cache(self,base):
        drive=base/"drive"/"pilot"
        (drive/"cache"/"train").mkdir(parents=True)
        (drive/"cache"/"val").mkdir()
        (drive/"source_manifest.json").write_text(json.dumps(
            dict(cache_schema_version=2,dataset="hypersim",split_seed=42)))
        # Tiny but structurally valid, compressed NPZs.
        for split,num in (("train",3),("val",20)):
            for i in range(num):
                np.savez_compressed(drive/"cache"/split/f"sample_{i:02d}.npz",
                                    pixels=np.arange(12,dtype=np.uint8),
                                    frame_id=np.int32(i))
        return drive

    def test_stage_resume_and_drive_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            drive=self.make_cache(base)
            local=base/"content"/"gt_pilot_cache"
            result=stage.stage_groundtruth_cache(drive,local)
            self.assertEqual(result,local.resolve())
            self.assertEqual(len(list((local/"cache"/"train").glob("*.npz"))),3)
            self.assertEqual(len(list((local/"cache"/"val").glob("*.npz"))),20)
            self.assertTrue((drive/"source_manifest.json").exists())
            cached=local/"cache"/"train"/"sample_00.npz"
            first_stat=cached.stat().st_mtime_ns
            # Re-running in same runtime should skip already-staged examples.
            stage.stage_groundtruth_cache(drive,local)
            self.assertEqual(cached.stat().st_mtime_ns,first_stat)
            # Lost local example gets restored from Drive, not regenerated.
            cached.unlink()
            stage.stage_groundtruth_cache(drive,local)
            self.assertTrue(cached.exists())
            self.assertTrue((drive/"cache"/"train"/"sample_00.npz").exists())

    def test_reject_different_run_and_bad_validation_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            drive=self.make_cache(base)
            local=base/"fast"
            stage.stage_groundtruth_cache(drive,local)
            source=drive/"source_manifest.json"
            source.write_text(json.dumps(dict(cache_schema_version=3)))
            with self.assertRaises(RuntimeError):
                stage.stage_groundtruth_cache(drive,local)
            (drive/"cache"/"val"/"sample_00.npz").unlink()
            with self.assertRaises(RuntimeError):
                stage.stage_groundtruth_cache(drive,base/"fresh")


if __name__=="__main__":
    unittest.main()
