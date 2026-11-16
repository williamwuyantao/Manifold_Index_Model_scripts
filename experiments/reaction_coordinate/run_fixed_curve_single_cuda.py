#!/usr/bin/env python3
"""Evaluate one fixed reaction-coordinate seven-tuple at one N.

This is a narrow continuation runner: it does not re-screen structural or
evaluation parameters and does not regenerate data.  It is used to extend a
completed screening run to a larger N while preserving the same Python/shared
CUDA-core path and its capacity-selected Phase-1/Phase-2 tiling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


THIS = Path(__file__).resolve().parent
if str(THIS) not in sys.path:
    sys.path.insert(0, str(THIS))

import run_mc_fixed7_screen_cuda as screen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--parameters-json", type=Path, required=True)
    args = parser.parse_args()
    if args.n < 1000:
        raise ValueError("N must be at least 1000")
    if args.output.exists():
        raise RuntimeError("output already exists; refusing to overwrite")
    if not args.data.is_file() or not args.library.is_file():
        raise FileNotFoundError("data HDF5 or CUDA library is absent")

    args.output.mkdir(parents=True)
    lock = args.output / ".running.lock"
    lock.mkdir()
    try:
        parameters = json.loads(args.parameters_json.read_text())
        required = {"alpha", "K_offset", "h_scale", "rho_init", "bite",
                    "eval_rho", "eval_h"}
        if set(parameters) != required:
            raise ValueError("parameters JSON must contain exactly the fixed seven parameters")

        core = screen.ManorMCuda(args.library)
        info = core.device_info(0)
        if core.device_count() != 1 or info.compute_capability not in {"8.9", "9.0"}:
            raise RuntimeError("single visible sm_89/sm_90 device is required")

        manifest = dict(
            schema="mfm-reaction-coordinate-fixed-curve-single.v1",
            backend="Python + shared CUDA core",
            program_sha256=screen.sha256(Path(__file__)),
            screen_program_sha256=screen.sha256(Path(screen.__file__)),
            library_sha256=screen.sha256(args.library),
            data=str(args.data.resolve()),
            data_sha256=screen.sha256(args.data),
            N=args.n,
            fixed_parameters=parameters,
            selection="fixed tuple supplied by completed multi-N screen",
            validation_and_testing="independent views; no parameter selection",
            metric_sum_contract="fp32-query256-tree-v1",
            device=dict(name=info.name,
                        visible_device=os.environ.get("CUDA_VISIBLE_DEVICES"),
                        compute_capability=info.compute_capability,
                        total_global_memory_bytes=info.total_global_memory_bytes),
        )
        manifest_sha = hashlib.sha256(
            json.dumps(screen.plain(manifest), sort_keys=True).encode()
        ).hexdigest()
        manifest["manifest_sha256"] = manifest_sha
        screen.write_json(args.output / "manifest.json", manifest)
        screen.write_json(args.output / "status.json",
                          dict(state="loading", observed_at=time.time()))

        data = screen.load_data(args.data, args.n, include_testing=True)
        policy = screen.MemoryPolicy(
            working_bytes=0,
            pinned_bytes=16 * screen.GIB,
            phase1_working_bytes=0,
            phase1_index_heap_bytes=48 * screen.GIB,
        )
        rows = screen.evaluate_curve(
            data, args.library, [args.n], parameters, policy,
            args.output, manifest_sha,
        )
        summary = dict(
            schema="mfm-reaction-coordinate-fixed-curve-single.v1",
            complete=True,
            manifest_sha256=manifest_sha,
            N=args.n,
            fixed_parameters=parameters,
            result=rows[0],
            completed_at=time.time(),
        )
        screen.write_json(args.output / "summary.json", summary)
        screen.write_json(args.output / "status.json",
                          dict(state="complete", observed_at=time.time()))
        return 0
    except BaseException as error:
        screen.write_json(args.output / "status.json",
                          dict(state="failed", error=repr(error), observed_at=time.time()))
        raise
    finally:
        lock.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
