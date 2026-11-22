"""Safe migration of cached TA-TiTok examples after correcting encoder RGB range.

Only cached decoder features / rejected alignment records are invalid.
Original Hypersim RGB, depth and semantic downloads remain in root/downloads.
"""
import json
import os
import shutil
from pathlib import Path

CACHE_VERSION = 2
ENCODER_INPUT_RANGE = "[0,1] (official TA-TiTok RGB/255)"


def ensure_cache_manifest(root, spec):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "source_manifest.json"
    if not path.exists():
        _write(path, spec)
        return "new"
    old = json.loads(path.read_text(encoding="utf8"))
    if old == spec:
        return "resume"
    # Support precisely the old version 1 format (it used [-1,+1] RGB).
    added = {"cache_schema_version", "encoder_input_range"}
    if "cache_schema_version" in old or {k: v for k, v in spec.items() if k not in added} != old:
        raise RuntimeError(
            "The existing cache has different scene splits or training settings. "
            "Choose another --root instead of mixing data."
        )
    archive = root / "cache_archived_bad_rgb_range_v1"
    suffix = 1
    while archive.exists():
        archive = root / f"cache_archived_bad_rgb_range_v1_{suffix}"
        suffix += 1
    archive.mkdir(parents=True)
    for name in ("cache", "source_manifest.json", "preparation_summary.json",
                 "cache_complete.json"):
        previous = root / name
        if previous.exists():
            shutil.move(str(previous), str(archive / name))
    _write(path, spec)
    print(
        "Rebuilding image-code features using official [0,1] TA-TiTok normalization. "
        f"Archived old invalid .npz/rejected files at {archive}. "
        f"Original RGB/depth/semantics still in {root / 'downloads'}.",
        flush=True
    )
    return "migrated"


def _write(path, obj):
    temp = path.with_name("." + path.name + ".partial")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf8")
    os.replace(temp, path)
