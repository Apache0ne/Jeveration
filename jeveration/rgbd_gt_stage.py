"""Copy only prepared RGBD training examples from a Drive mount to local SSD.

Dataset source and checkpoints remain on Drive; this staging is intentionally
ephemeral and repeatable after a Colab runtime reset.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path


def stage_groundtruth_cache(drive_root, local_root="/content/jeveration_gt_cache"):
    """Incrementally stage validated .npz examples and manifest from Drive.

    The source training cache is untouched. A partial copy is never considered
    complete. The manifest fingerprint prevents mixing different experiments.
    """
    drive_root = Path(drive_root).expanduser().resolve()
    local_root = Path(local_root).expanduser().resolve()
    if drive_root == local_root or drive_root in local_root.parents or local_root in drive_root.parents:
        raise ValueError("Drive dataset root and local training destination must be separate")
    if str(local_root).startswith("/content/drive/"):
        raise ValueError("Local destination must be outside mounted Google Drive")
    source_cache = drive_root / "cache"
    if not (source_cache / "train").is_dir() or not (source_cache / "val").is_dir():
        raise FileNotFoundError("No ready train/val geometry cache in " + str(source_cache))
    source_manifest = drive_root / "source_manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Missing source manifest {source_manifest}")
    manifest_bytes = source_manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    expected = {}
    for split in ("train", "val"):
        expected[split] = sorted(p for p in (source_cache / split).glob("*.npz")
                                 if p.is_file())
        if not expected[split]:
            raise FileNotFoundError(f"No cached {split} samples in Drive")
    if len(expected["val"]) != 20:
        raise RuntimeError(f"Expected exactly 20 held-out samples, found {len(expected['val'])}")
    (local_root / "cache" / "train").mkdir(parents=True, exist_ok=True)
    (local_root / "cache" / "val").mkdir(parents=True, exist_ok=True)
    signature_file = local_root / "stage_source.json"
    if signature_file.is_file():
        old = json.loads(signature_file.read_text())
        if old.get("manifest_sha256") != signature or old.get("drive_root") != str(drive_root):
            raise RuntimeError("The local stage belongs to a different training run. "
                               "Choose another --local-cache-dir.")
    else:
        # If local artifacts already exist, require proof that they're from the same experiment.
        if any(any((local_root / "cache" / split).glob("*.npz")) for split in ("train", "val")):
            raise RuntimeError("Existing unstamped local cache: use an empty --local-cache-dir")
        payload = {"drive_root": str(drive_root), "manifest_sha256": signature}
        tmp = signature_file.with_suffix(".partial")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, signature_file)
    shutil.copyfile(source_manifest, local_root / "source_manifest.json")

    from tqdm.auto import tqdm
    to_copy = [p for split in ("train", "val") for p in expected[split]]
    copied, reused, bytes_written = 0, 0, 0
    print(f"[STAGE] Copying {len(to_copy)} prepared NPZ examples to {local_root}", flush=True)
    for source in tqdm(to_copy, desc="Drive -> /content SSD", unit="file"):
        target = local_root / "cache" / source.parent.name / source.name
        if target.is_file() and target.stat().st_size == source.stat().st_size:
            reused += 1
            continue
        temp = target.with_name("." + target.name + ".partial")
        try:
            with source.open("rb") as fin, temp.open("wb") as fout:
                shutil.copyfileobj(fin, fout, length=8 * 1024 * 1024)
            if temp.stat().st_size != source.stat().st_size:
                raise IOError("Incomplete copy: " + str(source))
            # Corrupt ZIP must not become a training example.
            with zipfile.ZipFile(temp) as zf:
                damaged = zf.testzip()
                if damaged:
                    raise IOError("Corrupt NPZ entry: " + damaged)
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        copied += 1
        bytes_written += target.stat().st_size
    print(f"[STAGE] Ready: {len(expected['train'])} train, "
          f"{len(expected['val'])} validation. "
          f"Newly copied={copied}, reused={reused}, "
          f"transferred={bytes_written/2**30:.2f} GiB", flush=True)
    return local_root
