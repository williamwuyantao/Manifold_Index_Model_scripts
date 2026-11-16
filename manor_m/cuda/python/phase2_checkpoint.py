"""Atomic, immutable Phase-2 state checkpoints for bounded large-N jobs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np


SCHEMA = "mfm-phase2-state-checkpoint.v1"


def file_sha256(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_name(iteration: int) -> str:
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("checkpoint iteration must be a nonnegative integer")
    return f"K{iteration:06d}"


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_checkpoint(root: Path, iteration: int, arrays: dict[str, np.ndarray],
                     metadata: dict) -> Path:
    """Write one new checkpoint directory and publish its manifest last.

    Existing checkpoints are never replaced.  An interrupted ``.partial``
    directory remains diagnostic evidence and is never considered resumable.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / checkpoint_name(iteration)
    if final.exists():
        raise FileExistsError(f"checkpoint already exists: {final}")
    if not arrays:
        raise ValueError("checkpoint needs at least one array")
    temporary = root / (final.name + ".partial-" + uuid.uuid4().hex)
    temporary.mkdir()
    records = {}
    for name, source in sorted(arrays.items()):
        if not name or any(token in name for token in ("/", "\\", "..")):
            raise ValueError(f"unsafe checkpoint array name: {name!r}")
        value = np.ascontiguousarray(source)
        if value.dtype.hasobject:
            raise TypeError("object arrays cannot be checkpointed")
        path = temporary / (name + ".bin")
        with path.open("xb") as stream:
            value.tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
        records[name] = {
            "file": path.name,
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    manifest = {
        "schema": SCHEMA,
        "complete": True,
        "iteration": iteration,
        "metadata": metadata,
        "arrays": records,
    }
    manifest_path = temporary / "checkpoint.json"
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(temporary)
    temporary.rename(final)
    _sync_directory(root)
    return final


def read_manifest(path: Path, expected_metadata: dict | None = None) -> dict:
    path = Path(path)
    manifest_path = path / "checkpoint.json"
    before = manifest_path.read_bytes()
    value = json.loads(before)
    if manifest_path.read_bytes() != before:
        raise RuntimeError(f"checkpoint manifest changed while reading: {manifest_path}")
    if (value.get("schema") != SCHEMA or value.get("complete") is not True
            or value.get("iteration") != int(path.name[1:])
            or not isinstance(value.get("arrays"), dict)
            or not value["arrays"]):
        raise ValueError(f"invalid checkpoint manifest: {manifest_path}")
    if expected_metadata is not None:
        for key, expected in expected_metadata.items():
            if value.get("metadata", {}).get(key) != expected:
                raise ValueError(f"checkpoint metadata differs: {key}")
    return value


def load_checkpoint(path: Path, expected_metadata: dict | None = None,
                    verify_hashes: bool = True) -> tuple[dict[str, np.memmap], dict]:
    path = Path(path)
    manifest = read_manifest(path, expected_metadata)
    arrays = {}
    for name, record in sorted(manifest["arrays"].items()):
        file_path = path / record["file"]
        dtype = np.dtype(record["dtype"])
        shape = tuple(int(value) for value in record["shape"])
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if (record["bytes"] != expected_bytes or file_path.stat().st_size != expected_bytes):
            raise ValueError(f"checkpoint byte count differs: {name}")
        if verify_hashes and file_sha256(file_path) != record["sha256"]:
            raise ValueError(f"checkpoint hash differs: {name}")
        arrays[name] = np.memmap(file_path, dtype=dtype, mode="r", shape=shape, order="C")
    return arrays, manifest


def complete_checkpoints(root: Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    result = []
    for path in root.iterdir():
        if not path.is_dir() or not path.name.startswith("K") or ".partial-" in path.name:
            continue
        try:
            read_manifest(path)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            continue
        result.append(path)
    return sorted(result, key=lambda path: int(path.name[1:]))


def latest_checkpoint(root: Path) -> Path | None:
    values = complete_checkpoints(root)
    return values[-1] if values else None
