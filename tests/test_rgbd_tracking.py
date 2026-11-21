"""CPU-only tests for Drive-compatible checkpointing and live metrics."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

TRACKING = Path(__file__).resolve().parents[1] / "jeveration" / "rgbd_tracking.py"
spec = importlib.util.spec_from_file_location("rgbd_tracking", TRACKING)
tracking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracking)


class TestRGBDTracking(unittest.TestCase):
    def test_save_schedule(self):
        self.assertEqual(tracking.save_steps("", 4000), set())
        self.assertEqual(tracking.save_steps("500,1200,3750", 4000), {500, 1200, 3750})
        with self.assertRaises(ValueError):
            tracking.save_steps("0,250", 1000)
        with self.assertRaises(ValueError):
            tracking.save_steps("1001", 1000)
        with self.assertRaises(ValueError):
            tracking.save_steps("1,,2", 1000)
        with self.assertRaises(ValueError):
            tracking.save_steps("abc", 1000)

    def test_snapshot_resume_and_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = nn.Linear(3, 1)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            tracker = tracking.LiveTraining(
                tmp, total=4, log_every=1, keep_last=2,
                progress=False, tensorboard=False,
            )

            def saver(path, head, **metadata):
                torch.save({"weight": head.weight.detach().cpu(),
                            "metadata": metadata}, path)

            try:
                for step in (1, 2, 3):
                    components = {"depth": 0.2, "normal": 0.3, "seg": 0.4}
                    tracker.step(step, 0.5, 1e-4, components, 0.7)
                    tracker.checkpoint(step, model, opt, 3, 0.5, saver)
                metrics = {
                    "loss": 0.4, "depth": 0.1, "normal": 0.1,
                    "seg": 0.2, "normal_deg": 18.0, "depth_absrel": 0.13,
                    "seg_acc": 0.9, "samples": 20,
                }
                tracker.validation(3, metrics)
                tracker.best(3, model, metrics, saver)
            finally:
                tracker.close()
            folder = Path(tmp)
            pointer = json.loads((folder / "latest.json").read_text())
            self.assertEqual(pointer["step"], 3)
            self.assertEqual(tracking.latest_state_path(folder).name, "trainer_step_000003.pt")
            self.assertFalse((folder / "trainer_step_000001.pt").exists())
            self.assertFalse((folder / "rgbd_step_000001.pt").exists())
            self.assertTrue((folder / "trainer_step_000002.pt").exists())
            self.assertTrue((folder / "trainer_step_000003.pt").exists())
            self.assertTrue((folder / "rgbd_best.pt").exists())
            latest = torch.load(tracking.latest_state_path(folder), map_location="cpu",
                                weights_only=True)
            self.assertEqual(latest["step"], 3)
            self.assertIn("optimizer", latest)
            self.assertEqual(len((folder / "training_metrics.jsonl").read_text().splitlines()), 3)
            self.assertEqual(len((folder / "validation.jsonl").read_text().splitlines()), 1)
            self.assertEqual(len((folder / "events.jsonl").read_text().splitlines()), 5)

    def test_empty_resume_folder_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                tracking.latest_state_path(tmp)


if __name__ == "__main__":
    unittest.main()
