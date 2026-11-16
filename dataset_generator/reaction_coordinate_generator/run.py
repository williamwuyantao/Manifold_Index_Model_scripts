#!/usr/bin/env python3
"""Generate resumable, large-N reaction-coordinate Monte Carlo datasets.

Each CUDA launch handles one automatically sized chunk of independent initial
conditions. The chunk size is derived from available GPU memory, capped at a
large saturation-friendly block count, and each completed chunk is committed
to the HDF5 dataset before the next one begins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import h5py
import numpy as np


KERNEL_SOURCE = Path(__file__).resolve().parent / "cuda" / "reaction_coordinate_mc_em.cu"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def export_segment_checkpoint(root: Path, dataset_path: Path,
                              start: int, stop: int) -> tuple[Path, str]:
    """Atomically materialize one completed HDF5 segment for off-host backup.

    The growing HDF5 file remains the canonical resume store.  Immutable
    segment files make every completed CUDA batch independently transferable
    without repeatedly copying a multi-million-row HDF5 container.
    """
    with h5py.File(dataset_path, "r") as h5:
        global_index = np.asarray(h5["global_index"][start:stop])
        if global_index.size != stop - start or not np.all(h5["MCDone"][start:stop] == 1):
            raise RuntimeError("cannot export an unfinished checkpoint segment")
        checkpoint = root / "checkpoints" / (
            f"segment-{int(global_index[0]):012d}-{int(global_index[-1]) + 1:012d}.npz")
        if checkpoint.exists():
            return checkpoint, sha256(checkpoint)
        payload = {
            "global_index": global_index,
            "X": np.asarray(h5["X"][start:stop]),
            "XcleanTop": np.asarray(h5["XcleanTop"][start:stop]),
            "TanTop": np.asarray(h5["TanTop"][start:stop]),
            "theta": np.asarray(h5["theta"][start:stop]),
            "phi": np.asarray(h5["phi"][start:stop]),
            "Y": np.asarray(h5["Y"][start:stop]),
            "MCOutputs17": np.asarray(h5["MCOutputs17"][start:stop]),
            "MCHitAnyCount": np.asarray(h5["MCHitAnyCount"][start:stop]),
            "MCPathCount": np.asarray(h5["MCPathCount"][start:stop]),
            "MCDone": np.asarray(h5["MCDone"][start:stop]),
        }
    checkpoint.parent.mkdir(exist_ok=True)
    temporary = checkpoint.with_name(f"{checkpoint.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(checkpoint)
    return checkpoint, sha256(checkpoint)


def truncated_normal(rng: np.random.Generator, count: int, sigma: float) -> np.ndarray:
    values = np.empty(count, dtype=np.float64)
    filled = 0
    while filled < count:
        draws = rng.normal(0.0, sigma, size=max(1024, count - filled))
        accepted = draws[(draws >= -1.0) & (draws <= 1.0)]
        take = min(accepted.size, count - filled)
        values[filled:filled + take] = accepted[:take]
        filled += take
    return values


def truncated_normal_scalar(rng: np.random.Generator, sigma: float) -> float:
    """Draw one truncated normal without consuming a chunk-sized buffer."""
    while True:
        value = float(rng.normal(0.0, sigma))
        if -1.0 <= value <= 1.0:
            return value


def tangent_basis(unit: np.ndarray) -> np.ndarray:
    axis = np.zeros_like(unit)
    least = np.argmin(np.abs(unit), axis=1)
    axis[np.arange(unit.shape[0]), least] = 1.0
    first = axis - unit * np.sum(axis * unit, axis=1, keepdims=True)
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    return np.stack((first, np.cross(unit, first)), axis=1)


def bottom_layout(ambient_dimension: int) -> tuple[int, int, int]:
    d3 = ambient_dimension // 3
    d4 = d3 + 1
    d2 = ambient_dimension - 3 - d3 - d4
    if d2 <= 0:
        raise ValueError("ambient dimension is too small for reaction-coordinate bottom blocks")
    return d2, d3, d4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--ambient-dimension", type=int, required=True)
    parser.add_argument("--n-curve", type=int, required=True,
                        help="number of independent initial conditions")
    parser.add_argument("--n-sims", type=int, default=48_000,
                        help="Monte Carlo paths per training initial condition")
    parser.add_argument("--heldout-n-sims", type=int, default=400_000,
                        help="Monte Carlo paths per validation/testing condition")
    parser.add_argument("--n-t", type=int, default=50_000,
                        help="Euler--Maruyama steps per trajectory")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tangential-strength", type=float, default=1.0)
    parser.add_argument("--normal-strength", type=float, default=1.0)
    parser.add_argument("--input-sigma", type=float, default=0.5)
    parser.add_argument("--diffusion-scale", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--delta-theta", type=float, default=0.2)
    parser.add_argument("--delta-r", type=float, default=0.8)
    parser.add_argument("--tan-el-lo", type=float, default=-0.5)
    parser.add_argument("--tan-el-hi", type=float, default=0.0)
    parser.add_argument("--cos-dtheta", type=float, default=0.8660254)
    parser.add_argument("--tan-dtheta", type=float, default=0.202710)
    parser.add_argument("--c1", type=float, default=0.7)
    parser.add_argument("--c2", type=float, default=0.9)
    parser.add_argument("--c3", type=float, default=1.1)
    parser.add_argument("--c4", type=float, default=1.3)
    parser.add_argument("--heldout-count", type=int, default=5_000)
    parser.add_argument(
        "--path-stride", type=int, default=None,
        help="global Philox path stride; defaults to the largest path count",
    )
    parser.add_argument("--min-hit-fraction", type=float, default=0.8)
    parser.add_argument("--curve-chunk", type=int, default=None,
                        help="override the automatic initial-condition chunk size")
    parser.add_argument("--heldout-curve-chunk", type=int, default=256,
                        help="launch batch for testing and validation rows")
    parser.add_argument("--training-curve-chunk", type=int, default=8192,
                        help="launch batch for training rows")
    parser.add_argument("--checkpoint-path-steps", type=int,
                        default=3_000_000_000_000,
                        help="automatic per-launch Monte-Carlo work budget")
    parser.add_argument("--global-offset", type=int, default=0,
                        help="global row offset for a non-overlapping dataset shard")
    parser.add_argument("--global-rows", type=int, default=None,
                        help="total rows across shards; defaults to n-curve")
    parser.add_argument("--mode", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument(
        "--reuse-shared-reduce", action="store_true",
        help="reuse MODE=0 bottom shared memory for the final reduction",
    )
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--maxrregcount", type=int, default=160)
    parser.add_argument("--cuda-arch", type=int, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--gpu", default=None,
                        help="optional CUDA_VISIBLE_DEVICES override; UUID pinning is supported")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-runtime-upgrade",
        action="store_true",
        help="allow a checked runtime-only migration of an existing partial root",
    )
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    if args.ambient_dimension < 6:
        raise ValueError("ambient dimension must be at least 6")
    if args.global_rows is None:
        args.global_rows = args.n_curve
    if args.global_rows <= 2 * args.heldout_count:
        raise ValueError("global-rows must leave at least one training point")
    if args.global_offset < 0 or args.global_offset + args.n_curve > args.global_rows:
        raise ValueError("global shard lies outside [0, global-rows)")
    if (args.n_sims <= 0 or args.heldout_n_sims <= 0
            or args.n_t <= 0 or args.threads <= 0):
        raise ValueError("path counts, n-t, and threads must be positive")
    if args.threads > 1024 or (args.threads & (args.threads - 1)) != 0:
        raise ValueError("threads must be a power of two no larger than 1024")
    if args.path_stride is None:
        args.path_stride = max(args.n_sims, args.heldout_n_sims)
    if args.path_stride < max(args.n_sims, args.heldout_n_sims):
        raise ValueError("path-stride must cover both training and heldout paths")
    max_subsequence = ((args.global_rows - 1) * args.path_stride
                       + max(args.n_sims, args.heldout_n_sims) - 1)
    if max_subsequence > (1 << 64) - 1:
        raise ValueError("global path numbering overflows uint64")
    finite_values = (
        args.input_sigma, args.diffusion_scale, args.temperature, args.dt,
        args.delta_theta, args.delta_r, args.tan_el_lo, args.tan_el_hi,
        args.cos_dtheta, args.tan_dtheta, args.c1, args.c2, args.c3, args.c4,
    )
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("all simulation parameters must be finite")
    if (args.input_sigma <= 0 or args.diffusion_scale <= 0
            or args.temperature <= 0 or args.dt <= 0):
        raise ValueError("input-sigma, diffusion-scale, and dt must be positive")
    if args.tangential_strength <= 0 or args.normal_strength <= 0:
        raise ValueError("potential strengths must be positive")
    if not 0.0 < args.min_hit_fraction <= 1.0:
        raise ValueError("min-hit-fraction must be in (0, 1]")
    if args.curve_chunk is not None and args.curve_chunk <= 0:
        raise ValueError("curve-chunk must be positive")
    if args.heldout_curve_chunk is not None and args.heldout_curve_chunk <= 0:
        raise ValueError("heldout-curve-chunk must be positive")
    if args.training_curve_chunk is not None and args.training_curve_chunk <= 0:
        raise ValueError("training-curve-chunk must be positive")
    if args.checkpoint_path_steps <= 0:
        raise ValueError("checkpoint-path-steps must be positive")


def visible_free_bytes() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    requested = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    rows: list[tuple[str, int]] = []
    for line in output.splitlines():
        uuid, free_mib = [part.strip() for part in line.split(",", 1)]
        rows.append((uuid, int(float(free_mib))))
    for uuid, free_mib in rows:
        if requested == uuid:
            return free_mib * 1024 * 1024
    return rows[0][1] * 1024 * 1024 if len(rows) == 1 else None


def choose_curve_chunk(args: argparse.Namespace, bytes_per_curve: int, *,
                       n_sims: int | None = None, limit: int | None = None,
                       override: int | None = None) -> int:
    """Choose a safe checkpoint batch without online throughput probing.

    The cap is driven by the exact work per condition and free memory.  For
    the long production trajectories, a few hundred persistent CUDA blocks
    already occupy a whole device; allowing tens of thousands would merely
    turn one checkpoint into a many-hour kernel launch.
    """
    maximum = args.n_curve if limit is None else limit
    free_bytes = visible_free_bytes()
    if free_bytes is None:
        memory_limited = 65_536
    else:
        # Input, output, and staging are accounted with a 3x safety factor.
        memory_limited = max(1, int(free_bytes * 0.55) // max(1, 3 * bytes_per_curve))
    requested = override if override is not None else args.curve_chunk
    paths = args.n_sims if n_sims is None else n_sims
    work_limited = max(
        1, args.checkpoint_path_steps // max(1, paths * args.n_t)
    )
    if requested is not None:
        chunk = min(maximum, requested, memory_limited, work_limited)
        if chunk >= 256:
            return max(256, (chunk // 256) * 256)
        if chunk >= 128:
            return 128
        return max(1, chunk)
    chunk = min(maximum, 65_536, memory_limited, work_limited)
    if chunk >= 256:
        return max(256, (chunk // 256) * 256)
    if chunk >= 128:
        return 128
    return max(1, chunk)


def simulation_config(args: argparse.Namespace, dbot: int, curve_chunk: int) -> dict[str, Any]:
    return {
        "schema": "reaction-coordinate-monte-carlo.v6",
        "production_launch": "once_v1",
        "ambient_dimension": args.ambient_dimension,
        "intrinsic_dimension": 2,
        "dbot": dbot,
        "rows": args.n_curve,
        "global_offset": args.global_offset,
        "global_rows": args.global_rows,
        "n_sims": args.n_sims,
        "heldout_n_sims": args.heldout_n_sims,
        "path_stride": args.path_stride,
        "rng_stream_version": "global_curve_path_stride_v1",
        "input_generation_version": "global_row_seed_v1",
        "n_t": args.n_t,
        "seed": args.seed,
        "tangential_strength": args.tangential_strength,
        "normal_strength": args.normal_strength,
        "input_sigma": args.input_sigma,
        "diffusion_scale": args.diffusion_scale,
        "temperature": args.temperature,
        "dt": args.dt,
        "delta_theta": args.delta_theta,
        "delta_r": args.delta_r,
        "tan_el_lo": args.tan_el_lo,
        "tan_el_hi": args.tan_el_hi,
        "cos_dtheta": args.cos_dtheta,
        "tan_dtheta": args.tan_dtheta,
        "c1": args.c1,
        "c2": args.c2,
        "c3": args.c3,
        "c4": args.c4,
        "heldout_count": args.heldout_count,
        "split_contract": "global[0:H)=testing; global[H:2H)=validation; remainder=training",
        "min_hit_fraction": args.min_hit_fraction,
        "curve_chunk": curve_chunk,
        "checkpoint_path_steps": args.checkpoint_path_steps,
        "checkpoint_policy": "capacity_and_path_steps_v1",
        "mode": args.mode,
        "reuse_shared_reduce": args.reuse_shared_reduce,
        "threads": args.threads,
        "maxrregcount": args.maxrregcount,
        "cuda_arch": args.cuda_arch,
        "kernel_source_sha256": sha256(KERNEL_SOURCE),
    }


def prepare_root(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    config_path = args.out_root / "config.json"
    dataset_path = args.out_root / "dataset.h5"
    if args.out_root.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to reuse an existing variant root: {args.out_root}")
        prior = json.loads(config_path.read_text())
        if prior != config:
            runtime_keys = {"kernel_source_sha256", "cuda_arch", "production_launch"}
            changed = {
                key for key in set(prior) | set(config)
                if prior.get(key) != config.get(key)
            }
            if not args.allow_runtime_upgrade or not changed.issubset(runtime_keys):
                raise RuntimeError("resume configuration differs from the existing variant")
            stamp = time.time_ns()
            atomic_json(
                args.out_root / f"config.before-runtime-upgrade-{stamp}.json",
                prior,
            )
            atomic_json(
                args.out_root / f"runtime-upgrade-{stamp}.json",
                {
                    "schema": "reaction-coordinate-runtime-upgrade.v1",
                    "changed_keys": sorted(changed),
                    "before": prior,
                    "after": config,
                },
            )
            atomic_json(config_path, config)
        if not dataset_path.exists():
            raise RuntimeError("resume root has no dataset.h5")
        (args.out_root / "checkpoints").mkdir(exist_ok=True)
        with h5py.File(dataset_path, "r+") as h5:
            h5.attrs["completed_rows"] = int(np.count_nonzero(h5["MCDone"][:]))
            if prior != config:
                for key in ("kernel_source_sha256", "cuda_arch", "production_launch"):
                    h5.attrs[key] = config[key]
            h5.flush()
        return dataset_path

    args.out_root.mkdir(parents=True)
    (args.out_root / "raw").mkdir()
    (args.out_root / "bin").mkdir()
    (args.out_root / "checkpoints").mkdir()
    atomic_json(config_path, config)
    global_index = np.arange(args.global_offset,
                             args.global_offset + args.n_curve, dtype=np.int64)
    heldout = args.heldout_count
    testing = np.flatnonzero(global_index < heldout).astype(np.int64)
    validation = np.flatnonzero((global_index >= heldout)
                                & (global_index < 2 * heldout)).astype(np.int64)
    training = np.flatnonzero(global_index >= 2 * heldout).astype(np.int64)
    rng = np.random.default_rng(args.seed + args.global_offset + 1)
    rng.shuffle(training)
    chunk_rows = config["curve_chunk"]

    def write_indices(h5: h5py.File, name: str, values: np.ndarray) -> None:
        kwargs: dict[str, Any] = dict(data=values)
        if values.size:
            kwargs["chunks"] = (min(chunk_rows, values.size),)
        h5.create_dataset(name, **kwargs)

    with h5py.File(dataset_path, "w") as h5:
        h5.attrs["schema"] = config["schema"]
        h5.attrs["state"] = "running"
        for key, value in config.items():
            if key != "schema":
                h5.attrs[key] = value
        h5.attrs["label_contract"] = "first-hit count divided by hit-any count"
        h5.attrs["completed_rows"] = 0
        h5.create_dataset("X", shape=(args.n_curve, args.ambient_dimension), dtype="f4",
                          chunks=(chunk_rows, args.ambient_dimension))
        h5.create_dataset("XcleanTop", shape=(args.n_curve, 3), dtype="f8", chunks=(chunk_rows, 3))
        h5.create_dataset("TanTop", shape=(args.n_curve, 2, 3), dtype="f8", chunks=(chunk_rows, 2, 3))
        h5.create_dataset("theta", shape=(args.n_curve,), dtype="f4", chunks=(chunk_rows,))
        h5.create_dataset("phi", shape=(args.n_curve,), dtype="f4", chunks=(chunk_rows,))
        h5.create_dataset("Y", shape=(args.n_curve, 4), dtype="f4", chunks=(chunk_rows, 4))
        h5.create_dataset("MCOutputs17", shape=(args.n_curve, 17), dtype="f4", chunks=(chunk_rows, 17))
        h5.create_dataset("MCHitAnyCount", shape=(args.n_curve,), dtype="f4", chunks=(chunk_rows,))
        h5.create_dataset("MCPathCount", shape=(args.n_curve,), dtype="i4", chunks=(chunk_rows,))
        h5.create_dataset("MCDone", shape=(args.n_curve,), dtype="u1", chunks=(chunk_rows,))
        h5["MCDone"][:] = 0
        h5["MCPathCount"][:] = 0
        h5.create_dataset("global_index", data=global_index, chunks=(chunk_rows,))
        write_indices(h5, "training_indices", training)
        write_indices(h5, "validation_indices", validation)
        write_indices(h5, "testing_indices", testing)
    return dataset_path


def generate_chunk(args: argparse.Namespace, start: int, count: int, d2: int, d3: int, d4: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    global_start = args.global_offset + start
    # Every row owns a deterministic stream.  The input for a global row is
    # therefore invariant to CUDA checkpoint size, shard boundaries, and
    # resume segmentation.  This is intentionally a new input-generation
    # version; it must not be mixed with rows generated by the old chunk seed.
    clean = np.empty((count, 3), dtype=np.float64)
    radial = np.empty(count, dtype=np.float64)
    bot = np.empty((count, d2 + d3 + d4), dtype=np.float64)
    for local, global_row in enumerate(range(global_start, global_start + count)):
        rng = np.random.default_rng(
            np.random.SeedSequence((int(args.seed), int(global_row)))
        )
        unit = rng.normal(size=3)
        unit /= np.linalg.norm(unit)
        clean[local] = unit
        radial[local] = truncated_normal_scalar(rng, args.input_sigma) + 1.0
        bot[local, :d2] = rng.normal(0.0, args.input_sigma, size=d2)
        bot[local, d2:d2 + d3] = rng.normal(0.0, args.input_sigma, size=d3)
        bot[local, d2 + d3:] = rng.normal(0.0, args.input_sigma, size=d4)
    top = np.ascontiguousarray(clean * radial[:, None], dtype=np.float32)
    bot = np.ascontiguousarray(bot, dtype=np.float32)
    normal_scale = 16.0 * args.normal_strength
    c1tmp = normal_scale * (2.0 / (args.input_sigma**2 * 3.0))
    c2tmp = normal_scale * (2.0 / (0.9**2 * args.input_sigma**2 * d2))
    c3tmp = normal_scale * (2.0 / (0.8**2 * args.input_sigma**2 * d3))
    c4tmp = normal_scale * (2.0 / (0.7**2 * args.input_sigma**2 * d4))
    coeff = np.ascontiguousarray(np.concatenate((
        np.full(d2, 1.0 - args.dt * c2tmp, dtype=np.float32),
        np.full(d3, 1.0 - args.dt * c3tmp, dtype=np.float32),
        np.full(d4, 1.0 - args.dt * c4tmp, dtype=np.float32),
    )))
    tangent = tangent_basis(clean)
    theta = np.arctan2(clean[:, 2], np.hypot(clean[:, 0], clean[:, 1])).astype(np.float32)
    phi = np.arctan2(clean[:, 1], clean[:, 0]).astype(np.float32)
    return top, bot, coeff, clean, tangent, theta, phi, c1tmp


def compile_kernel(args: argparse.Namespace, root: Path, dbot: int) -> Path:
    # Bind the cache entry to the source and compile contract.  A source edit
    # must never silently reuse a binary produced from an older source file.
    source_tag = sha256(KERNEL_SOURCE)[:16]
    reuse_tag = "reuse1" if args.reuse_shared_reduce else "reuse0"
    binary = root / "bin" / (
        f"reaction_mc_em_dbot{dbot}_r{args.maxrregcount}_sm{args.cuda_arch}_"
        f"src{source_tag}_{reuse_tag}_once_v2"
    )
    if binary.exists():
        return binary
    temporary = binary.with_name(f"{binary.name}.tmp.{os.getpid()}")
    command = [
        str(args.nvcc), "-O3", "--use_fast_math", "-std=c++17",
        f"-gencode=arch=compute_{args.cuda_arch},code=sm_{args.cuda_arch}",
        f"-maxrregcount={args.maxrregcount}", f"-DMFM_DBOT={dbot}",
        "-DMFM_CONST_COEFF=1",
    ]
    if args.reuse_shared_reduce:
        command.append("-DMFM_REUSE_SHARED_REDUCE=1")
    command.extend([str(KERNEL_SOURCE), "-o", str(temporary)])
    try:
        subprocess.run(command, check=True)
        temporary.replace(binary)
    finally:
        temporary.unlink(missing_ok=True)
    return binary


def record_build(root: Path, args: argparse.Namespace, binary: Path) -> None:
    compiler = subprocess.run(
        [str(args.nvcc), "--version"], check=True,
        capture_output=True, text=True,
    )
    atomic_json(root / "build.json", {
        "schema": "reaction-coordinate-build.v1",
        "source_sha256": sha256(KERNEL_SOURCE),
        "binary": binary.name,
        "binary_sha256": sha256(binary),
        "cuda_arch": args.cuda_arch,
        "maxrregcount": args.maxrregcount,
        "mode": args.mode,
        "reuse_shared_reduce": args.reuse_shared_reduce,
        "nvcc_version": compiler.stdout.strip(),
    })


def run_chunk(args: argparse.Namespace, root: Path, binary: Path, global_start: int,
              top: np.ndarray, bot: np.ndarray, coeff: np.ndarray, c1tmp: float,
              n_sims: int, label_group: str) -> tuple[np.ndarray, dict[str, Any]]:
    scratch = root / "scratch"
    scratch.mkdir(exist_ok=True)
    top_path = scratch / "top.f32"
    bot_path = scratch / "bot.f32"
    coeff_path = scratch / "coeff.f32"
    top.tofile(top_path)
    bot.tofile(bot_path)
    coeff.tofile(coeff_path)
    attempt = f"{global_start:09d}-{label_group}-{time.time_ns()}"
    raw_output = root / "raw" / f"mc_outputs17-{attempt}.f32"
    environment = dict(os.environ)
    if args.gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    environment.update({
        "MFM_NCURVE": str(top.shape[0]),
        "MFM_NSIMS": str(n_sims),
        "MFM_NT": str(args.n_t),
        "MFM_TOP_F32": str(top_path),
        "MFM_BOT_F32": str(bot_path),
        "MFM_COEFF_F32": str(coeff_path),
        "MFM_DT": repr(args.dt),
        "MFM_TERM_T": repr(args.n_t * args.dt),
        "MFM_C1TMP": repr(c1tmp),
        "MFM_SIGMA": repr(math.sqrt(args.dt) * args.diffusion_scale),
        "MFM_DELTA_THETA": repr(args.delta_theta),
        "MFM_DELTA_R": repr(args.delta_r),
        "MFM_TAN_EL_LO": repr(args.tan_el_lo),
        "MFM_TAN_EL_HI": repr(args.tan_el_hi),
        "MFM_COS_DTHETA": repr(args.cos_dtheta),
        "MFM_TAN_DTHETA": repr(args.tan_dtheta),
        "MFM_C1": repr(args.c1),
        "MFM_C2": repr(args.c2),
        "MFM_C3": repr(args.c3),
        "MFM_C4": repr(args.c4),
        "MFM_BASE_SEED": str(args.seed),
        "MFM_CURVE_OFFSET": str(global_start),
        "MFM_PATH_STRIDE": str(args.path_stride),
        "MFM_TEMPERATURE": repr(args.temperature),
        "MFM_TANGENTIAL_STRENGTH": repr(args.tangential_strength),
    })
    completed = subprocess.run(
        [str(binary), str(args.mode), str(args.threads), "once", str(raw_output)],
        env=environment, check=True, text=True, capture_output=True,
    )
    (root / "raw" / f"kernel-{attempt}.stdout.log").write_text(completed.stdout)
    values = np.fromfile(raw_output, dtype=np.float32)
    expected = 17 * top.shape[0]
    if values.size != expected:
        raise RuntimeError(f"expected {expected} output values, found {values.size}")
    return values.reshape(17, top.shape[0]).T.copy(), {
        "raw_output": raw_output.name,
        "raw_output_sha256": sha256(raw_output),
        "global_start": global_start,
        "rows": int(top.shape[0]),
        "n_sims": n_sims,
        "label_group": label_group,
        "kernel_stdout": completed.stdout.strip(),
    }


def check_outputs(args: argparse.Namespace, outputs: np.ndarray,
                  n_sims: int) -> tuple[np.ndarray, np.ndarray, float]:
    if not np.isfinite(outputs).all():
        raise RuntimeError("non-finite Monte Carlo output")
    hit_any = outputs[:, 12]
    if np.min(hit_any) < args.min_hit_fraction * n_sims:
        raise RuntimeError("hit-any count is below the configured minimum fraction")
    labels = outputs[:, 13:17] / hit_any[:, None]
    if not np.isfinite(labels).all():
        raise RuntimeError("non-finite first-hit probability")
    sum_error = float(np.max(np.abs(np.sum(labels, axis=1) - 1.0)))
    if sum_error > 2e-6:
        raise RuntimeError(f"first-hit probability sum error {sum_error} exceeds 2e-6")
    return labels.astype(np.float32), hit_any.astype(np.float32), sum_error


def label_group(args: argparse.Namespace, global_row: int) -> tuple[str, int]:
    if global_row < args.heldout_count:
        return "testing", args.heldout_n_sims
    if global_row < 2 * args.heldout_count:
        return "validation", args.heldout_n_sims
    return "training", args.n_sims


def label_segments(args: argparse.Namespace, start: int, stop: int,
                   bytes_per_curve: int):
    """Yield homogeneous label batches within one stored input chunk."""
    cursor = start
    while cursor < stop:
        global_cursor = args.global_offset + cursor
        group, paths = label_group(args, global_cursor)
        if group == "testing":
            boundary = args.heldout_count
        elif group == "validation":
            boundary = 2 * args.heldout_count
        else:
            boundary = args.global_rows
        segment_stop = min(stop, boundary - args.global_offset)
        override = (args.training_curve_chunk if group == "training"
                    else args.heldout_curve_chunk)
        cap = choose_curve_chunk(args, bytes_per_curve, n_sims=paths,
                                 limit=segment_stop - cursor, override=override)
        while cursor < segment_stop:
            next_cursor = min(segment_stop, cursor + cap)
            yield cursor, next_cursor, group, paths
            cursor = next_cursor


def incomplete_ranges(done: np.ndarray, base: int) -> list[tuple[int, int]]:
    """Return contiguous not-done runs without double-counting a resume."""
    pending: list[tuple[int, int]] = []
    cursor = 0
    while cursor < done.size:
        while cursor < done.size and bool(done[cursor]):
            cursor += 1
        if cursor == done.size:
            break
        begin = cursor
        while cursor < done.size and not bool(done[cursor]):
            cursor += 1
        pending.append((base + begin, base + cursor))
    return pending
def update_progress(root: Path, config: dict[str, Any], done: int, last_chunk: dict[str, Any] | None) -> None:
    payload: dict[str, Any] = {
        "schema": "reaction-coordinate-monte-carlo-progress.v6",
        "rows_total": config["rows"],
        "rows_complete": done,
        "rows_remaining": config["rows"] - done,
        "global_offset": config["global_offset"],
        "global_rows": config["global_rows"],
        "curve_chunk": config["curve_chunk"],
    }
    if last_chunk is not None:
        payload["last_chunk"] = last_chunk
    atomic_json(root / "progress.json", payload)


def finalize(root: Path, dataset_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    min_hit = float("inf")
    max_hit = float("-inf")
    max_sum_error = 0.0
    with h5py.File(dataset_path, "r+") as h5:
        done = np.asarray(h5["MCDone"][:])
        if not np.all(done):
            raise RuntimeError("cannot finalize an incomplete Monte Carlo dataset")
        for start in range(0, config["rows"], config["curve_chunk"]):
            stop = min(config["rows"], start + config["curve_chunk"])
            y = h5["Y"][start:stop]
            hit = h5["MCHitAnyCount"][start:stop]
            min_hit = min(min_hit, float(np.min(hit)))
            max_hit = max(max_hit, float(np.max(hit)))
            max_sum_error = max(max_sum_error, float(np.max(np.abs(np.sum(y, axis=1) - 1.0))))
        h5.attrs["state"] = "complete"
    report = {
        "schema": "reaction-coordinate-monte-carlo-completion.v6",
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "kernel_source_sha256": config["kernel_source_sha256"],
        "rows": config["rows"],
        "ambient_dimension": config["ambient_dimension"],
        "n_sims": config["n_sims"],
        "heldout_n_sims": config["heldout_n_sims"],
        "n_t": config["n_t"],
        "tangential_strength": config["tangential_strength"],
        "normal_strength": config["normal_strength"],
        "curve_chunk": config["curve_chunk"],
        "min_hit_any": min_hit,
        "max_hit_any": max_hit,
        "max_probability_sum_error": max_sum_error,
    }
    atomic_json(root / "completion.json", report)
    update_progress(root, config, config["rows"], None)
    return report


def main() -> None:
    args = parse_args()
    validate(args)
    d2, d3, d4 = bottom_layout(args.ambient_dimension)
    dbot = d2 + d3 + d4
    bytes_per_curve = 4 * (args.ambient_dimension + 17)
    curve_chunk = choose_curve_chunk(args, bytes_per_curve)
    config = simulation_config(args, dbot, curve_chunk)
    dataset_path = prepare_root(args, config)
    binary = compile_kernel(args, args.out_root, dbot)
    record_build(args.out_root, args, binary)

    bytes_per_curve = 4 * (args.ambient_dimension + 17)
    with h5py.File(dataset_path, "r") as h5:
        done = int(h5.attrs.get("completed_rows", np.count_nonzero(h5["MCDone"][:])))
    update_progress(args.out_root, config, done, None)
    for start in range(0, args.n_curve, curve_chunk):
        stop = min(args.n_curve, start + curve_chunk)
        with h5py.File(dataset_path, "r") as h5:
            already_complete = bool(np.all(h5["MCDone"][start:stop]))
        if already_complete:
            export_segment_checkpoint(args.out_root, dataset_path, start, stop)
            continue
        count = stop - start
        top, bot, coeff, clean, tangent, theta, phi, c1tmp = generate_chunk(
            args, start, count, d2, d3, d4)
        # Close the HDF5 file before every CUDA launch.  Each completed label
        # segment is therefore an independently stable backup point.
        with h5py.File(dataset_path, "r+") as h5:
            h5["X"][start:stop] = np.concatenate((top, bot), axis=1)
            h5["XcleanTop"][start:stop] = clean
            h5["TanTop"][start:stop] = tangent
            h5["theta"][start:stop] = theta
            h5["phi"][start:stop] = phi
            h5.flush()
        for segment_start, segment_stop, group, paths in label_segments(
                args, start, stop, bytes_per_curve):
            with h5py.File(dataset_path, "r") as h5:
                done_flags = np.asarray(h5["MCDone"][segment_start:segment_stop], dtype=np.uint8)
            for local_start, local_stop in incomplete_ranges(done_flags, segment_start):
                relative_start = local_start - start
                relative_stop = local_stop - start
                outputs, evidence = run_chunk(
                    args, args.out_root, binary, args.global_offset + local_start,
                    top[relative_start:relative_stop], bot[relative_start:relative_stop],
                    coeff, c1tmp, paths, group)
                labels, hit_any, sum_error = check_outputs(args, outputs, paths)
                with h5py.File(dataset_path, "r+") as h5:
                    prior_done = np.asarray(h5["MCDone"][local_start:local_stop], dtype=np.uint8)
                    if np.any(prior_done):
                        raise RuntimeError("resume race: a pending row became complete")
                    h5["Y"][local_start:local_stop] = labels
                    h5["MCOutputs17"][local_start:local_stop] = outputs
                    h5["MCHitAnyCount"][local_start:local_stop] = hit_any
                    h5["MCPathCount"][local_start:local_stop] = paths
                    h5["MCDone"][local_start:local_stop] = 1
                    new_done = local_stop - local_start
                    done = int(h5.attrs.get("completed_rows", 0)) + new_done
                    h5.attrs["completed_rows"] = done
                    h5.flush()
                checkpoint, checkpoint_sha256 = export_segment_checkpoint(
                    args.out_root, dataset_path, local_start, local_stop)
                update_progress(args.out_root, config, done, {
                    "start": local_start, "stop": local_stop,
                    "global_start": args.global_offset + local_start,
                    "checkpoint": checkpoint.name,
                    "checkpoint_sha256": checkpoint_sha256,
                    "max_probability_sum_error": sum_error, **evidence})
    report = finalize(args.out_root, dataset_path, config)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
