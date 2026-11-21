"""Live Colab progress, TensorBoard logs, and crash-safe checkpoint snapshots.

The training cache stays under --root. Checkpoints, evaluation history, and
TensorBoard are written under --save-dir (which can be a mounted Google Drive).
"""
import json
import os
import sys
from pathlib import Path

import torch


def save_steps(value, max_step):
    """Steps explicitly selected by the user; the final step is always saved."""
    if not value.strip():
        return set()
    parts = value.split(",")
    try:
        steps = {int(part.strip()) for part in parts}
    except ValueError as exc:
        raise ValueError("--save-at must be comma-separated positive integer steps") from exc
    if any(not part.strip() for part in parts) or any(step < 1 or step > max_step for step in steps):
        raise ValueError(f"--save-at values must be between 1 and --steps ({max_step})")
    return steps


def checkpoint_dir(path):
    path = Path(path).expanduser().resolve()
    # Do not accidentally write checkpoints to Colab's temporary disk when
    # the user meant a Drive folder but forgot drive.mount(...).
    drive_root = Path("/content/drive")
    if path == drive_root or drive_root in path.parents:
        if not Path("/content/drive/MyDrive").is_dir():
            raise RuntimeError(
                "Google Drive is not mounted: run "
                "from google.colab import drive; drive.mount('/content/drive') first"
            )
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name("." + path.name + ".partial")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf8")
    os.replace(tmp, path)


def atomic_torch_save(path, state):
    path = Path(path)
    tmp = path.with_name("." + path.name + ".partial")
    torch.save(state, tmp)
    os.replace(tmp, path)


def latest_state_path(folder):
    folder = Path(folder)
    pointer = folder / "latest.json"
    if pointer.is_file():
        state = json.loads(pointer.read_text(encoding="utf8"))
        path = folder / state["trainer"]
        if not path.is_file():
            raise FileNotFoundError(f"Latest Drive checkpoint missing: {path}")
        return path
    legacy = folder / "trainer_last.pt"
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(f"No resumable trainer checkpoint found under {folder}")


class LiveTraining:
    """Keep a visible step-level progress bar and durable logs/checkpoints."""
    def __init__(self, folder, total, start=0, log_every=10, keep_last=0,
                 progress=True, tensorboard=True):
        if total < 1 or not 0 <= start <= total:
            raise ValueError("Invalid training progress state")
        if log_every < 1 or keep_last < 0:
            raise ValueError("--log-every must be >=1 and --keep-last must be >=0")
        self.folder = checkpoint_dir(folder)
        self.total = total
        self.log_every = log_every
        self.keep_last = keep_last
        self.jsonl = (self.folder / "training_metrics.jsonl").open("a", encoding="utf8", buffering=1)
        self.val_jsonl = (self.folder / "validation.jsonl").open("a", encoding="utf8", buffering=1)
        self.events = (self.folder / "events.jsonl").open("a", encoding="utf8", buffering=1)
        self.writer = None
        self.pbar = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.folder / "tensorboard"), flush_secs=10)
        if progress:
            from tqdm.auto import tqdm
            self.pbar = tqdm(total=total, initial=start, desc="RGBD training",
                             file=sys.stdout, dynamic_ncols=True, mininterval=0.35,
                             smoothing=0.2)

    def _write_event(self, kind, step, **values):
        event = {"type": kind, "step": int(step), **values}
        self.events.write(json.dumps(event) + "\n")

    def step(self, step, loss, lr, parts, grad_norm, gpu_gib=None):
        metrics = {"step": int(step), "loss": float(loss), "lr": float(lr),
                   "depth": float(parts["depth"]), "normal": float(parts["normal"]),
                   "seg": float(parts["seg"]), "grad_norm": float(grad_norm)}
        if gpu_gib is not None:
            metrics["gpu_allocated_gib"] = float(gpu_gib)
        if self.pbar:
            self.pbar.update(1)
            self.pbar.set_postfix(loss=f"{loss:.4f}", depth=f"{metrics['depth']:.3f}",
                                  normal=f"{metrics['normal']:.3f}",
                                  seg=f"{metrics['seg']:.3f}", lr=f"{lr:.2e}")
        if step == 1 or step % self.log_every == 0 or step == self.total:
            self.jsonl.write(json.dumps(metrics) + "\n")
            if self.writer:
                for key in ("loss", "depth", "normal", "seg", "grad_norm"):
                    self.writer.add_scalar("train/" + key, metrics[key], step)
                self.writer.add_scalar("train/lr", lr, step)
                if gpu_gib is not None:
                    self.writer.add_scalar("train/gpu_allocated_gib", gpu_gib, step)
            if not self.pbar:
                print(f"step {step}/{self.total}: loss={loss:.5f} "
                      f"depth={metrics['depth']:.4f} normal={metrics['normal']:.4f} "
                      f"seg={metrics['seg']:.4f} lr={lr:.2e}", flush=True)

    def validation(self, step, metrics):
        record = dict(metrics, step=int(step))
        self.val_jsonl.write(json.dumps(record) + "\n")
        if self.writer:
            for name, value in record.items():
                if name not in ("step", "samples") and isinstance(value, (int, float)):
                    self.writer.add_scalar("validation/" + name, value, step)
            self.writer.flush()
        message = (f"Validation @ {step}: total={metrics['loss']:.5f} "
                   f"depth_absrel={metrics['depth_absrel']:.4f} "
                   f"normal_deg={metrics['normal_deg']:.2f} "
                   f"seg_acc={metrics['seg_acc']:.4f} (20 images)")
        if self.pbar:
            self.pbar.write(message)
        else:
            print(message, flush=True)
        self._write_event("validation", step, metrics=metrics)

    def best(self, step, head, metrics, save_head):
        path = self.folder / "rgbd_best.pt"
        tmp = path.with_name("." + path.name + ".partial")
        save_head(tmp, head, step=step, metrics=metrics)
        os.replace(tmp, path)
        self._write_event("best", step, path=str(path), validation_loss=float(metrics["loss"]))
        if self.pbar:
            self.pbar.write(f"New best checkpoint: {path}")
        else:
            print(f"New best checkpoint: {path}", flush=True)

    def checkpoint(self, step, head, optimizer, width, best, save_head):
        """Two files per chosen step: model-only and full optimizer resume state."""
        model = self.folder / f"rgbd_step_{step:06d}.pt"
        state = self.folder / f"trainer_step_{step:06d}.pt"
        # Save model and full training state first, then atomically update
        # the pointer. A failed write leaves the previous checkpoint usable.
        tmp = model.with_name("." + model.name + ".partial")
        save_head(tmp, head, step=step, source="Jeveration dense teacher")
        os.replace(tmp, model)
        weights = {key: value.detach().cpu() for key, value in head.state_dict().items()}
        atomic_torch_save(state, {
            "model": weights, "optimizer": optimizer.state_dict(),
            "step": int(step), "width": int(width),
            "steps_target": int(self.total), "best": float(best),
            "torch_rng": torch.get_rng_state()
        })
        atomic_json(self.folder / "latest.json",
                    {"step": int(step), "model": model.name, "trainer": state.name})
        self._write_event("checkpoint", step, model=str(model), trainer=str(state))
        if self.keep_last:
            entries = sorted(self.folder.glob("trainer_step_*.pt"))
            for previous in entries[:-self.keep_last]:
                old_step = previous.stem.removeprefix("trainer_step_")
                previous.unlink()
                partner = self.folder / f"rgbd_step_{old_step}.pt"
                partner.unlink(missing_ok=True)
        if self.writer:
            self.writer.flush()
        msg = f"Saved checkpoint @ {step}: {state}"
        if self.pbar:
            self.pbar.write(msg)
        else:
            print(msg, flush=True)

    def close(self):
        if self.pbar:
            self.pbar.close()
        if self.writer:
            self.writer.flush()
            self.writer.close()
        self.jsonl.close()
        self.val_jsonl.close()
        self.events.close()
