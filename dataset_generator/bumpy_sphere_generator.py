#!/usr/bin/env python3
"""Generate a reach-relative tube-noise bumpy sphere dataset.

This is the bumpy-sphere analogue of the Meyer-Helix relative-noise data:
clean samples live on a 2D bumpy sphere in the first three coordinates,
local reach is estimated from the analytic focal radius, and noisy samples
are produced by projecting ambient random directions onto the normal space
and scaling by local reach.  Thus ||X - Xclean|| <= 0.9 * 0.98 * local reach.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import h5py
import numpy as np

# Canonical noise reach safety factor: noise_reach = 0.98 * loc_reach.
NOISE_REACH_SAFETY = 0.98
TORUS_SCHEMA = "mfm-canonical-torus-dataset.v1"
TORUS_TRAIN_FRACTION = 0.80
TORUS_VALIDATION_FRACTION = 0.10
TORUS_TEST_FRACTION = 0.10

FAMILY_SCHEMA = "mfm-bumpy-sphere-family.v1"
FAMILY_VERSIONS = ("v1", "v2", "v3")
FAMILY_DIMENSIONS = (10, 20)
FAMILY_SPLITS = ("train", "validation", "test")
FAMILY_SPLIT_CODES = {"train": 1, "validation": 2, "test": 3}
FAMILY_LATENT_SEED = 2026082601
FAMILY_X_NOISE_SEED = 2026082602
FAMILY_Y_NOISE_SEED = 2026082603
FAMILY_SELECTION_SEED = 2026082604
FAMILY_RESPONSE_KAPPA = 8.75
FAMILY_RESPONSE_ROTATION_SEED = 20260717
FAMILY_RESPONSE_ROTATION_INDEX = 8
FAMILY_Y_COVARIANCE_COUNT = 262_144
FAMILY_IID_AUDIT_COUNT = 50_000
FAMILY_REACH_CALIBRATION_SEED = 2026082605
FAMILY_CANONICAL_RECIPE_SCHEMA = "mfm-bumpy-sphere-family-recipe.v1"
FAMILY_CANONICAL_CHUNK_SIZE = 200_000
FAMILY_CANONICAL_ENVIRONMENT = {
    "python": "3.9.25",
    "numpy": "2.0.2",
    "scipy": "1.13.1",
    "h5py": "3.14.0",
    "openblas": "0.3.27",
}
FAMILY_CANONICAL_RESPONSE_IDENTITY_SHA256 = (
    "e130404b38dcf9dd6553d23a2983c007ddf26b6b0488c868a175de1a5c6b6f3a"
)
FAMILY_CANONICAL_REACH_SHA256 = {
    "v1": "47ebb4983dc25c16e9312286b30db4a04f8ac28e953b96db127abd954188517c",
    "v2": "fa35fea13ad991a934f26bc1e4d4aadcc618f4587cf1c11c4bcf32e0fb116ce9",
    "v3": "9ff526388254591dec4aeffdbc6cf31443cfcce3c737719a8efa4ff439a8b1a6",
}
FAMILY_CANONICAL_COVARIANCE_ROOT_SHA256 = {
    "v1": "f86c4fcb8be58dfb1966fff45f9ae637bd43c85da8ff2d96c27b5aabdeb9a43d",
    "v2": "198df5bd380590cb9015a54dc826060eead7d7e0872c8103eee6a8e4e11885c6",
    "v3": "fd69793a4b90ff2d1ce2b711531aef9c338edcf9d9606f1599e3760e1f10b0b4",
}

# Frozen conservative unscaled global-reach values.  A 65,536-point
# equal-area screen followed by multi-start float64 continuous focal-reach
# minimization gave 0.1094317819, 0.0947515175, and 0.0589372714.  The N=50k
# exhaustive pairwise audit found a larger pairwise minimum in every case, so
# focal curvature is the global limiter.  The tiny downward guard keeps the
# post-rescale global reach at (numerically, and conservatively) sqrt(D).
FAMILY_RAW_GLOBAL_REACH = {
    "v1": 0.10943175,
    "v2": 0.09475149,
    "v3": 0.05893724,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _array_sha256(value: np.ndarray, dtype: str) -> str:
    canonical = np.asarray(value, dtype=dtype)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _numpy_blas_version() -> str:
    configuration = getattr(np.__config__, "CONFIG", {})
    try:
        return str(configuration["Build Dependencies"]["blas"]["version"])
    except (KeyError, TypeError):
        return "unknown"


def _torus_covariance_root(covariance: np.ndarray, relative: float) -> np.ndarray:
    eigenvalues, vectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return float(relative) * (vectors * np.sqrt(eigenvalues)) @ vectors.T


def _torus_response_stats(
    source: h5py.File, sample_count: int = 200_000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    response = source["Y_clean_physical_torus"]
    block = np.asarray(response[: min(sample_count, response.shape[0])], dtype=np.float64)
    centered = block - block.mean(axis=0)
    covariance = centered.T @ centered / block.shape[0]
    coordinate_std = block.std(axis=0)
    coordinate_range = block.max(axis=0) - block.min(axis=0)
    return covariance, coordinate_std, coordinate_range


def _torus_create_dataset(
    handle: h5py.File,
    name: str,
    shape: tuple[int, ...],
    chunk_rows: int,
) -> h5py.Dataset:
    rows = min(max(1, int(chunk_rows)), max(1, shape[0]), 16_384)
    return handle.create_dataset(
        name,
        shape=shape,
        dtype="f4",
        chunks=(rows, *shape[1:]),
    )


def _torus_partition(pool_count: int) -> dict[str, int]:
    validation_count = int(math.floor(pool_count * TORUS_VALIDATION_FRACTION))
    test_count = int(math.floor(pool_count * TORUS_TEST_FRACTION))
    train_count = pool_count - validation_count - test_count
    if min(train_count, validation_count, test_count) < 1:
        raise ValueError("torus source pool is too small for the fixed 80/10/10 split")
    return {
        "pool_count": int(pool_count),
        "train_begin": 0,
        "train_end": int(train_count),
        "validation_begin": int(train_count),
        "validation_end": int(train_count + validation_count),
        "test_begin": int(train_count + validation_count),
        "test_end": int(pool_count),
    }


def _initialize_torus_output(
    output_path: Path,
    identity: dict[str, object],
    partition: dict[str, int],
    coordinate_std: np.ndarray,
    coordinate_range: np.ndarray,
    covariance_root: np.ndarray,
    chunk_rows: int,
    validation_cap: int,
    eval_cap: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    d = int(identity["ambient_d"])
    train_count = int(partition["train_end"])
    response_width = int(covariance_root.shape[0])
    generator = np.random.default_rng(int(identity["training_noise_seed"]))
    with h5py.File(output_path, "w") as output:
        output.attrs["identity_json"] = _canonical_json(identity)
        output.attrs["complete"] = False
        output.attrs["training_complete"] = False
        output.attrs["training_completed_rows"] = 0
        output.attrs["training_rng_state_json"] = _canonical_json(
            generator.bit_generator.state
        )
        output.attrs["manifold_d"] = 2
        output.attrs["ambient_d"] = d
        output.attrs["source_count"] = int(identity["source_count"])
        output.attrs["cache_policy"] = "no_cross_cell_cache"
        for name, value in partition.items():
            output.attrs[f"partition_{name}"] = int(value)

        _torus_create_dataset(output, "x_train", (train_count, d), chunk_rows)
        _torus_create_dataset(output, "xclean_train", (train_count, d), chunk_rows)
        _torus_create_dataset(
            output, "tangent_train", (train_count, d, 2), chunk_rows
        )
        _torus_create_dataset(
            output, "y_train_clean", (train_count, response_width), chunk_rows
        )
        _torus_create_dataset(
            output, "y_train_observed", (train_count, response_width), chunk_rows
        )

        _torus_create_dataset(output, "x_validation", (validation_cap, d), chunk_rows)
        _torus_create_dataset(
            output,
            "y_validation_observed",
            (validation_cap, response_width),
            chunk_rows,
        )
        _torus_create_dataset(output, "x_eval", (eval_cap, d), chunk_rows)
        _torus_create_dataset(output, "xclean_eval", (eval_cap, d), chunk_rows)
        _torus_create_dataset(output, "tangent_eval", (eval_cap, d, 2), chunk_rows)
        _torus_create_dataset(
            output, "y_eval_clean", (eval_cap, response_width), chunk_rows
        )

        validation_indices = np.arange(
            partition["validation_begin"],
            partition["validation_begin"] + validation_cap,
            dtype=np.int64,
        )
        eval_indices = np.arange(
            partition["test_end"] - eval_cap,
            partition["test_end"],
            dtype=np.int64,
        )
        output.create_dataset("validation_source_indices", data=validation_indices)
        output.create_dataset("eval_source_indices", data=eval_indices)
        output.create_dataset("y_min", data=np.zeros(response_width, dtype=np.float32))
        output.create_dataset("y_span", data=np.ones(response_width, dtype=np.float32))
        output.create_dataset(
            "y_coordinate_std_original", data=coordinate_std.astype(np.float32)
        )
        output.create_dataset(
            "y_noise_covariance_root", data=covariance_root.astype(np.float32)
        )
        noise_rms = float(np.sqrt(np.mean(np.sum(covariance_root**2, axis=1))))
        denominator = float(np.sqrt(np.mean(np.square(coordinate_std))))
        output.attrs["y_noise_absolute_original_units"] = noise_rms
        output.attrs["y_noise_relative_denominator"] = "rms_per_coordinate_std"
        output.attrs["y_noise_denominator_value"] = denominator
        output.attrs["y_response_range_mean"] = float(np.mean(coordinate_range))
        output.attrs["y_noise_as_fraction_of_range"] = noise_rms / max(
            float(np.mean(coordinate_range)), np.finfo(np.float64).eps
        )
        output.flush()


def generate_torus_d10(arguments: argparse.Namespace) -> int:
    if not arguments.source_h5:
        raise ValueError("--source-h5 is required for --dataset torus_d10")
    source_path = Path(arguments.source_h5).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if arguments.n < 10_000:
        raise ValueError("--n is the total torus pool count and must be at least 10,000")
    if arguments.d != 10:
        raise ValueError("the torus_d10 profile requires --d 10")
    if arguments.eval_cap < 1 or arguments.validation_cap < 1:
        raise ValueError("evaluation and validation caps must be positive")
    if arguments.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    if arguments.seed == arguments.validation_noise_seed:
        raise ValueError("training and validation noise seeds must differ")

    output_path = (
        Path(arguments.output_h5).expanduser().resolve()
        if arguments.output_h5
        else Path(__file__).resolve().parents[1]
        / "dataset"
        / "bumpy_sphere"
        / f"torus_d10_N{arguments.n}.h5"
    )
    with h5py.File(source_path, "r") as source:
        required = (
            "X_RelCstDevSig",
            "Xclean_top3",
            "Tan",
            "Y_clean_physical_torus",
        )
        missing = [name for name in required if name not in source]
        if missing:
            raise RuntimeError(f"torus source lacks {missing}")
        if not bool(source.attrs.get("complete", False)):
            raise RuntimeError("torus source is not marked complete")
        source_count = int(source["X_RelCstDevSig"].shape[0])
        if arguments.n > source_count:
            raise ValueError(
                f"requested pool N={arguments.n} exceeds source count {source_count}"
            )
        if source["X_RelCstDevSig"].shape[1] < arguments.d:
            raise RuntimeError("torus source ambient dimension is too small")
        covariance, coordinate_std, coordinate_range = _torus_response_stats(source)

    partition = _torus_partition(arguments.n)
    if arguments.validation_cap > partition["validation_end"] - partition["validation_begin"]:
        raise ValueError("--validation-cap exceeds the fixed validation pool")
    if arguments.eval_cap > partition["test_end"] - partition["test_begin"]:
        raise ValueError("--eval-cap exceeds the fixed testing pool")
    covariance_root = _torus_covariance_root(
        covariance, arguments.y_noise_relative
    )
    source_sha256 = _sha256(source_path)
    identity = {
        "schema": TORUS_SCHEMA,
        "dataset": "torus_d10",
        "source": str(source_path),
        "source_size": source_path.stat().st_size,
        "source_sha256": source_sha256,
        "source_count": int(source_count),
        "pool_count": int(arguments.n),
        "ambient_d": int(arguments.d),
        "manifold_d": 2,
        "train_fraction": TORUS_TRAIN_FRACTION,
        "validation_fraction": TORUS_VALIDATION_FRACTION,
        "test_fraction": TORUS_TEST_FRACTION,
        "validation_cap": int(arguments.validation_cap),
        "eval_cap": int(arguments.eval_cap),
        "training_noise_seed": int(arguments.seed),
        "validation_noise_seed": int(arguments.validation_noise_seed),
        "y_noise_relative": float(arguments.y_noise_relative),
        "x_rule": "first ambient_d coordinates of source X_RelCstDevSig",
        "y_noise_rule": "relative*Cov(Y_clean)^(1/2) float32 Gaussian",
        "nested_prefix_contract": True,
        "cache_policy": "no idx_zero_based or other cross-cell cache",
    }
    identity_text = _canonical_json(identity)

    if output_path.exists():
        with h5py.File(output_path, "r") as existing:
            if str(existing.attrs.get("identity_json", "")) != identity_text:
                raise RuntimeError(f"{output_path}: existing identity differs")
            if bool(existing.attrs.get("complete", False)):
                print(str(existing.attrs["summary_json"]))
                return 0
    else:
        _initialize_torus_output(
            output_path,
            identity,
            partition,
            coordinate_std,
            coordinate_range,
            covariance_root,
            arguments.chunk_size,
            arguments.validation_cap,
            arguments.eval_cap,
        )

    started = time.perf_counter()
    with h5py.File(source_path, "r") as source, h5py.File(output_path, "r+") as output:
        completed = int(output.attrs.get("training_completed_rows", 0))
        generator = np.random.default_rng()
        generator.bit_generator.state = json.loads(
            str(output.attrs["training_rng_state_json"])
        )
        transform = np.asarray(
            output["y_noise_covariance_root"], dtype=np.float32
        ).T
        train_count = int(partition["train_end"])
        for begin in range(completed, train_count, arguments.chunk_size):
            end = min(train_count, begin + arguments.chunk_size)
            x = np.asarray(source["X_RelCstDevSig"][begin:end, : arguments.d], dtype=np.float32)
            clean_top3 = np.asarray(source["Xclean_top3"][begin:end], dtype=np.float32)
            xclean = np.zeros((end - begin, arguments.d), dtype=np.float32)
            xclean[:, :3] = clean_top3
            tangent = np.asarray(source["Tan"][begin:end, : arguments.d, :], dtype=np.float32)
            y_clean = np.asarray(source["Y_clean_physical_torus"][begin:end], dtype=np.float32)
            y_observed = y_clean + generator.standard_normal(
                y_clean.shape, dtype=np.float32
            ) @ transform
            output["x_train"][begin:end] = x
            output["xclean_train"][begin:end] = xclean
            output["tangent_train"][begin:end] = tangent
            output["y_train_clean"][begin:end] = y_clean
            output["y_train_observed"][begin:end] = y_observed
            output.attrs.modify("training_completed_rows", end)
            output.attrs["training_rng_state_json"] = _canonical_json(
                generator.bit_generator.state
            )
            output.flush()
            elapsed = time.perf_counter() - started
            rate = (end - completed) / max(elapsed, 1e-12)
            eta = (train_count - end) / max(rate, 1e-12)
            print(
                f"copied training {end}/{train_count} "
                f"rate={rate:.0f}/s eta={eta/60:.1f}min",
                flush=True,
            )

        validation_indices = np.asarray(output["validation_source_indices"], dtype=np.int64)
        eval_indices = np.asarray(output["eval_source_indices"], dtype=np.int64)
        validation_clean = np.asarray(
            source["Y_clean_physical_torus"][validation_indices], dtype=np.float32
        )
        validation_generator = np.random.default_rng(arguments.validation_noise_seed)
        output["x_validation"][:] = np.asarray(
            source["X_RelCstDevSig"][validation_indices, : arguments.d],
            dtype=np.float32,
        )
        output["y_validation_observed"][:] = validation_clean + validation_generator.standard_normal(
            validation_clean.shape, dtype=np.float32
        ) @ transform
        output["x_eval"][:] = np.asarray(
            source["X_RelCstDevSig"][eval_indices, : arguments.d], dtype=np.float32
        )
        eval_clean_top3 = np.asarray(source["Xclean_top3"][eval_indices], dtype=np.float32)
        eval_clean = np.zeros((arguments.eval_cap, arguments.d), dtype=np.float32)
        eval_clean[:, :3] = eval_clean_top3
        output["xclean_eval"][:] = eval_clean
        output["tangent_eval"][:] = np.asarray(
            source["Tan"][eval_indices, : arguments.d, :], dtype=np.float32
        )
        output["y_eval_clean"][:] = np.asarray(
            source["Y_clean_physical_torus"][eval_indices], dtype=np.float32
        )
        output.attrs["training_complete"] = True
        summary = {
            **identity,
            "output": str(output_path),
            "partition": partition,
            "training_count": int(partition["train_end"]),
            "validation_count": int(arguments.validation_cap),
            "testing_count": int(arguments.eval_cap),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "generator_seconds": time.perf_counter() - started,
        }
        output.attrs["summary_json"] = _canonical_json(summary)
        output.attrs["complete"] = True
        output.flush()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _family_fibonacci_centers(count: int, phase: float) -> np.ndarray:
    index = np.arange(int(count), dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / float(count)
    angle = index * math.pi * (3.0 - math.sqrt(5.0)) + float(phase)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    centers = np.stack(
        (radius * np.cos(angle), radius * np.sin(angle), z), axis=1
    )
    return centers / np.linalg.norm(centers, axis=1, keepdims=True)


def _family_cap_offset(width: np.ndarray) -> np.ndarray:
    width = np.asarray(width, dtype=np.float64)
    return 0.5 * np.square(width) * (
        1.0 - np.exp(-2.0 / np.square(width))
    )


def _family_geometry_parameters(version: str) -> dict[str, object]:
    if version not in FAMILY_VERSIONS:
        raise ValueError(f"unknown bumpy-sphere family version: {version}")
    if version == "v1":
        centers = _family_fibonacci_centers(14, 0.31)
        index = np.arange(14, dtype=np.float64)
        width = 0.50 + 0.06 * np.remainder(2.0 * index + 1.0, 5.0) / 4.0
        amplitude = 6.0 * (
            0.260
            + 0.065 * np.sin(1.65 * index + 0.40)
            + 0.040 * np.cos(2.80 * index + 0.20)
        )
        directions = np.asarray(
            [[0.44, -0.28, 0.85], [-0.62, 0.72, 0.31]],
            dtype=np.float64,
        )
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        linear = np.asarray([6.0 * 0.014, -6.0 * 0.010]) @ directions
        formula = {
            "version": "bumpy_sphere_v1",
            "construction": "legacy_centered_pearls_v3_scale6",
            "center_count": 14,
            "fibonacci_phase": 0.31,
            "width_rule": "0.50+0.06*((2*i+1) mod 5)/4",
            "amplitude_rule": "6*(0.260+0.065*sin(1.65*i+0.40)+0.040*cos(2.80*i+0.20))",
            "cap_centering": "0.5*w^2*(1-exp(-2/w^2))",
            "linear_directions": directions.tolist(),
            "linear_coefficients": [0.084, -0.060],
        }
        return {
            "centers": centers,
            "width": width,
            "amplitude": amplitude,
            "offset": _family_cap_offset(width),
            "linear": linear,
            "formula": formula,
        }

    broad = bump_params()
    micro_centers = _family_fibonacci_centers(32, 0.73)
    index = np.arange(32, dtype=np.float64)
    micro_width = 0.26 + 0.055 * np.remainder(5.0 * index + 1.0, 8.0) / 7.0
    raw = 0.72 * np.sin(1.73 * index + 0.21) + 0.28 * np.cos(
        2.91 * index + 0.47
    )
    centered = raw - np.mean(raw)
    unit_amplitude = centered / np.max(np.abs(centered))
    micro_scale = 0.36 if version == "v2" else 0.45
    centers = np.concatenate((broad["centers"], micro_centers), axis=0)
    width = np.concatenate((broad["width"], micro_width))
    amplitude = np.concatenate(
        (0.75 * broad["amp"], micro_scale * unit_amplitude)
    )
    offset = np.concatenate(
        (np.zeros(broad["width"].shape[0]), _family_cap_offset(micro_width))
    )
    formula = {
        "version": f"bumpy_sphere_{version}",
        "construction": "two_scale_v0_plus_centered_fibonacci_caps",
        "broad_v0_factor": 0.75,
        "micro_center_count": 32,
        "micro_fibonacci_phase": 0.73,
        "micro_width_rule": "0.26+0.055*((5*i+1) mod 8)/7",
        "micro_raw_rule": "0.72*sin(1.73*i+0.21)+0.28*cos(2.91*i+0.47), mean-centered and max-abs normalized",
        "micro_amplitude": micro_scale,
        "micro_cap_centering": "0.5*w^2*(1-exp(-2/w^2))",
        "v0_centers": broad["centers"].tolist(),
        "v0_amplitude": broad["amp"].tolist(),
        "v0_width": broad["width"].tolist(),
        "v0_linear": broad["linear"].tolist(),
    }
    return {
        "centers": centers,
        "width": width,
        "amplitude": amplitude,
        "offset": offset,
        "linear": 0.75 * broad["linear"],
        "formula": formula,
    }


def _family_radius_derivatives(
    unit: np.ndarray, parameters: dict[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unit = np.asarray(unit, dtype=np.float64)
    count = unit.shape[0]
    centers = np.asarray(parameters["centers"], dtype=np.float64)
    width = np.asarray(parameters["width"], dtype=np.float64)
    amplitude = np.asarray(parameters["amplitude"], dtype=np.float64)
    offset = np.asarray(parameters["offset"], dtype=np.float64)
    linear = np.asarray(parameters["linear"], dtype=np.float64)
    radius = 1.0 + unit @ linear
    gradient = np.broadcast_to(linear, (count, 3)).copy()
    hessian = np.zeros((count, 3, 3), dtype=np.float64)
    for center, amp, current_width, current_offset in zip(
        centers, amplitude, width, offset
    ):
        width2 = float(current_width * current_width)
        exponential = np.exp((unit @ center - 1.0) / width2)
        radius += float(amp) * (exponential - float(current_offset))
        gradient += (
            float(amp) / width2 * exponential[:, None] * center[None, :]
        )
        hessian += (
            float(amp)
            / (width2 * width2)
            * exponential[:, None, None]
            * np.outer(center, center)[None, :, :]
        )
    if np.any(~np.isfinite(radius)) or float(np.min(radius)) <= 0.35:
        raise RuntimeError(
            f"invalid family radial field: min radius={float(np.min(radius)):.8g}"
        )
    return radius, gradient, hessian


def _family_geometry(
    unit: np.ndarray, parameters: dict[str, object]
) -> dict[str, np.ndarray]:
    unit = np.asarray(unit, dtype=np.float64)
    unit /= np.maximum(
        np.linalg.norm(unit, axis=1, keepdims=True), np.finfo(np.float64).tiny
    )
    radius, gradient, hessian = _family_radius_derivatives(unit, parameters)
    first, second = tangent_frame(unit)
    dr_first = np.sum(gradient * first, axis=1)
    dr_second = np.sum(gradient * second, axis=1)
    jac_first = radius[:, None] * first + dr_first[:, None] * unit
    jac_second = radius[:, None] * second + dr_second[:, None] * unit
    tangent_first = jac_first / np.maximum(
        np.linalg.norm(jac_first, axis=1, keepdims=True),
        np.finfo(np.float64).tiny,
    )
    residual = jac_second - np.sum(
        tangent_first * jac_second, axis=1, keepdims=True
    ) * tangent_first
    tangent_second = residual / np.maximum(
        np.linalg.norm(residual, axis=1, keepdims=True),
        np.finfo(np.float64).tiny,
    )
    normal = np.cross(tangent_first, tangent_second)
    normal /= np.maximum(
        np.linalg.norm(normal, axis=1, keepdims=True),
        np.finfo(np.float64).tiny,
    )
    normal[np.sum(normal * unit, axis=1) < 0] *= -1.0
    curvature = curvature_value(
        unit,
        first,
        second,
        radius,
        gradient,
        hessian,
        jac_first,
        jac_second,
        normal,
    )
    focal_reach = 1.0 / np.maximum(curvature, 1.0e-12)
    return {
        "unit": unit,
        "radius": radius,
        "tangent_first": tangent_first,
        "tangent_second": tangent_second,
        "curvature": curvature,
        "focal_reach": focal_reach,
    }


def _family_rotation_matrix(index: int) -> np.ndarray:
    generator = np.random.default_rng(FAMILY_RESPONSE_ROTATION_SEED)
    rotations = [np.eye(3, dtype=np.float64)]
    while len(rotations) <= int(index):
        matrix = generator.normal(size=(3, 3))
        q, r = np.linalg.qr(matrix)
        signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
        q = q * signs[None, :]
        if np.linalg.det(q) < 0.0:
            q[:, -1] *= -1.0
        if all(
            not np.allclose(q, old, atol=1.0e-12, rtol=0.0)
            for old in rotations
        ):
            rotations.append(q)
    return rotations[int(index)]


def _family_response(radius: np.ndarray, unit: np.ndarray) -> np.ndarray:
    rotation = _family_rotation_matrix(FAMILY_RESPONSE_ROTATION_INDEX)
    physical = np.asarray(radius, dtype=np.float64)[:, None] * np.asarray(
        unit, dtype=np.float64
    )
    phase = FAMILY_RESPONSE_KAPPA * (physical @ rotation.T)
    raw = np.stack((np.cos(phase), np.sin(phase)), axis=2).reshape(-1, 6)
    return (0.5 * (raw + 1.0)).astype(np.float32)


def _family_equal_area_unit(count: int) -> np.ndarray:
    index = np.arange(int(count), dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / float(count)
    angle = index * math.pi * (3.0 - math.sqrt(5.0)) + 0.417
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.stack((radius * np.cos(angle), radius * np.sin(angle), z), axis=1)


def _family_response_statistics(
    parameters: dict[str, object], relative_noise: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unit = _family_equal_area_unit(FAMILY_Y_COVARIANCE_COUNT)
    radius, _, _ = _family_radius_derivatives(unit, parameters)
    response = _family_response(radius, unit).astype(np.float64)
    centered = response - response.mean(axis=0)
    covariance = centered.T @ centered / response.shape[0]
    covariance_root = _torus_covariance_root(covariance, relative_noise)
    return covariance, covariance_root, response.std(axis=0)


def _family_sobol_unit(count: int) -> np.ndarray:
    try:
        from scipy.stats import qmc
    except ImportError as exception:
        raise RuntimeError(
            "scipy is required for the family reach calibration"
        ) from exception
    power = int(math.ceil(math.log2(max(2, int(count)))))
    sample = qmc.Sobol(
        d=2, scramble=True, seed=FAMILY_REACH_CALIBRATION_SEED
    ).random_base2(power)[: int(count)]
    z = 1.0 - 2.0 * sample[:, 0]
    angle = 2.0 * math.pi * sample[:, 1]
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.stack((radius * np.cos(angle), radius * np.sin(angle), z), axis=1)


def _family_build_reach_calibration(
    version: str,
    parameters: dict[str, object],
    count: int,
    tau: float,
    device_name: str,
    block_size: int,
    neighbors: int,
) -> dict[str, object]:
    if count < 100:
        raise ValueError("reach calibration requires at least 100 points")
    if tau <= 0.0 or block_size < 1 or neighbors < 1:
        raise ValueError("reach calibration tau/block/neighbors are invalid")
    unit = _family_sobol_unit(count)
    geometry = _family_geometry(unit, parameters)
    xclean = geometry["radius"][:, None] * geometry["unit"]
    tangent = np.stack(
        (geometry["tangent_first"], geometry["tangent_second"]), axis=2
    )
    focal = np.asarray(geometry["focal_reach"], dtype=np.float64)
    pairwise = np.empty(count, dtype=np.float64)
    tau2 = float(tau) ** 2
    started = time.perf_counter()
    if device_name == "cpu":
        for begin in range(0, count, block_size):
            end = min(count, begin + block_size)
            difference = xclean[None, :, :] - xclean[begin:end, None, :]
            latent_difference = unit[None, :, :] - unit[begin:end, None, :]
            chord2 = np.sum(np.square(difference), axis=2)
            latent2 = np.sum(np.square(latent_difference), axis=2)
            projection = np.einsum(
                "ijd,idm->ijm", difference, tangent[begin:end], optimize=True
            )
            normal2 = chord2 - np.sum(np.square(projection), axis=2)
            keep = (latent2 >= tau2) & (chord2 > 0.0) & (normal2 > 1.0e-24)
            candidate = chord2 / (
                2.0 * np.sqrt(np.maximum(normal2, 1.0e-24))
            )
            candidate[~keep] = np.inf
            pairwise[begin:end] = np.min(candidate, axis=1)
            elapsed = time.perf_counter() - started
            print(
                f"reach calibration {version} {end}/{count} "
                f"rate={end/max(elapsed, 1.0e-12):.0f}/s",
                flush=True,
            )
    else:
        try:
            import torch
        except ImportError as exception:
            raise RuntimeError("PyTorch is required for GPU reach calibration") from exception
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA calibration requested but unavailable: {device_name}")
        if device.type == "cuda":
            torch.cuda.set_device(device)

        x_all = torch.as_tensor(xclean, dtype=torch.float64, device=device)
        u_all = torch.as_tensor(unit, dtype=torch.float64, device=device)
        tangent_all = torch.as_tensor(tangent, dtype=torch.float64, device=device)
        for begin in range(0, count, block_size):
            end = min(count, begin + block_size)
            xi = x_all[begin:end]
            ui = u_all[begin:end]
            ti = tangent_all[begin:end]
            chord2 = torch.zeros(
                (end - begin, count), dtype=torch.float64, device=device
            )
            latent2 = torch.zeros_like(chord2)
            differences = []
            for coordinate in range(3):
                difference = x_all[:, coordinate][None, :] - xi[:, coordinate : coordinate + 1]
                differences.append(difference)
                chord2.addcmul_(difference, difference)
                latent_difference = (
                    u_all[:, coordinate][None, :]
                    - ui[:, coordinate : coordinate + 1]
                )
                latent2.addcmul_(latent_difference, latent_difference)
            tangent2 = torch.zeros_like(chord2)
            for tangent_axis in range(2):
                projection = torch.zeros_like(chord2)
                for coordinate, difference in enumerate(differences):
                    projection.addcmul_(
                        difference, ti[:, coordinate, tangent_axis : tangent_axis + 1]
                    )
                tangent2.addcmul_(projection, projection)
            normal2 = chord2 - tangent2
            keep = (latent2 >= tau2) & (chord2 > 0.0) & (normal2 > 1.0e-24)
            candidate = chord2 / (
                2.0 * torch.sqrt(torch.clamp(normal2, min=1.0e-24))
            )
            candidate.masked_fill_(~keep, float("inf"))
            pairwise[begin:end] = torch.min(candidate, dim=1).values.cpu().numpy()
            elapsed = time.perf_counter() - started
            print(
                f"reach calibration {version} {end}/{count} "
                f"rate={end/max(elapsed, 1.0e-12):.0f}/s",
                flush=True,
            )
            del xi, ui, ti, chord2, latent2, differences, tangent2, normal2, keep, candidate
        if device.type == "cuda":
            torch.cuda.empty_cache()
    bad = ~np.isfinite(pairwise) | (pairwise <= 0.0)
    pairwise[bad] = focal[bad]
    local_reach = np.minimum(focal, pairwise)
    from scipy.spatial import cKDTree

    tree = cKDTree(unit)
    digest = hashlib.sha256()
    digest.update(np.asarray(unit, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(local_reach, dtype="<f8").tobytes(order="C"))
    metadata = {
        "method": "float64 exhaustive sampled pairwise bottleneck, then min with analytic focal radius",
        "sample_rule": "scrambled Sobol equal-area sphere coordinates",
        "seed": FAMILY_REACH_CALIBRATION_SEED,
        "count": int(count),
        "latent_tau": float(tau),
        "neighbors": int(neighbors),
        "interpolation": "inverse latent-distance squared kNN IDW; pointwise min with analytic focal radius",
        "calibration_sha256": digest.hexdigest(),
        "focal_limiter_fraction": float(np.mean(focal <= pairwise)),
        "local_reach_quantiles": np.percentile(
            local_reach, [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 100]
        ).tolist(),
        "focal_reach_quantiles": np.percentile(
            focal, [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 100]
        ).tolist(),
        "pairwise_reach_quantiles": np.percentile(
            pairwise, [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 100]
        ).tolist(),
    }
    return {
        "unit": unit,
        "local_reach": local_reach,
        "tree": tree,
        "neighbors": int(neighbors),
        "metadata": metadata,
    }


def _family_recipe_path(directory: Path, version: str) -> Path:
    return directory / f"bumpy_sphere_{version}_canonical_recipe.npz"


def _family_write_recipe(
    path: Path,
    version: str,
    parameters: dict[str, object],
    calibration: dict[str, object],
    response_statistics: tuple[np.ndarray, np.ndarray, np.ndarray],
    y_noise_relative: float,
) -> None:
    unit = np.asarray(calibration["unit"], dtype="<f8")
    local_reach = np.asarray(calibration["local_reach"], dtype="<f8")
    covariance, covariance_root, coordinate_std = response_statistics
    metadata = dict(calibration["metadata"])
    identity = {
        "schema": FAMILY_CANONICAL_RECIPE_SCHEMA,
        "version": version,
        "geometry_formula_sha256": _family_formula_hash(parameters),
        "reach_calibration_sha256": metadata["calibration_sha256"],
        "covariance_sha256": _array_sha256(covariance, "<f8"),
        "covariance_root_sha256": _array_sha256(covariance_root, "<f8"),
        "coordinate_std_sha256": _array_sha256(coordinate_std, "<f8"),
        "response_identity_sha256": _family_response_identity()[
            "identity_sha256"
        ],
        "y_noise_relative": float(y_noise_relative),
        "unit_count": int(unit.shape[0]),
        "neighbors": int(calibration["neighbors"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            identity_json=np.asarray(_canonical_json(identity)),
            calibration_metadata_json=np.asarray(_canonical_json(metadata)),
            unit=unit,
            local_reach=local_reach,
            covariance=np.asarray(covariance, dtype="<f8"),
            covariance_root=np.asarray(covariance_root, dtype="<f8"),
            coordinate_std=np.asarray(coordinate_std, dtype="<f8"),
        )
    temporary.replace(path)
    print(
        json.dumps(
            {"canonical_recipe_written": str(path), "identity": identity},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def _family_load_recipe(
    path: Path,
    version: str,
    parameters: dict[str, object],
    y_noise_relative: float,
) -> tuple[dict[str, object], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    from scipy.spatial import cKDTree

    with np.load(path, allow_pickle=False) as recipe:
        identity = json.loads(str(recipe["identity_json"].item()))
        metadata = json.loads(str(recipe["calibration_metadata_json"].item()))
        unit = np.asarray(recipe["unit"], dtype=np.float64)
        local_reach = np.asarray(recipe["local_reach"], dtype=np.float64)
        covariance = np.asarray(recipe["covariance"], dtype=np.float64)
        covariance_root = np.asarray(recipe["covariance_root"], dtype=np.float64)
        coordinate_std = np.asarray(recipe["coordinate_std"], dtype=np.float64)

    if identity.get("schema") != FAMILY_CANONICAL_RECIPE_SCHEMA:
        raise RuntimeError(f"{path}: unsupported family recipe schema")
    if identity.get("version") != version:
        raise RuntimeError(
            f"{path}: recipe version {identity.get('version')} does not match {version}"
        )
    if identity.get("geometry_formula_sha256") != _family_formula_hash(parameters):
        raise RuntimeError(f"{path}: geometry formula hash differs")
    if float(identity.get("y_noise_relative", -1.0)) != float(y_noise_relative):
        raise RuntimeError(
            f"{path}: recipe y_noise_relative differs from {y_noise_relative}"
        )
    if unit.ndim != 2 or unit.shape[1] != 3 or local_reach.shape != unit.shape[:1]:
        raise RuntimeError(f"{path}: invalid reach-calibration array shapes")
    digest = hashlib.sha256()
    digest.update(np.asarray(unit, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(local_reach, dtype="<f8").tobytes(order="C"))
    reach_hash = digest.hexdigest()
    if reach_hash != identity.get("reach_calibration_sha256"):
        raise RuntimeError(f"{path}: reach-calibration payload hash differs")
    covariance_hash = _array_sha256(covariance_root, "<f8")
    if covariance_hash != identity.get("covariance_root_sha256"):
        raise RuntimeError(f"{path}: covariance-root payload hash differs")
    if _array_sha256(covariance, "<f8") != identity.get("covariance_sha256"):
        raise RuntimeError(f"{path}: covariance payload hash differs")
    if _array_sha256(coordinate_std, "<f8") != identity.get(
        "coordinate_std_sha256"
    ):
        raise RuntimeError(f"{path}: coordinate-std payload hash differs")
    if identity.get(
        "response_identity_sha256"
    ) != FAMILY_CANONICAL_RESPONSE_IDENTITY_SHA256:
        raise RuntimeError(f"{path}: response identity hash differs")
    if metadata.get("calibration_sha256") != reach_hash:
        raise RuntimeError(f"{path}: calibration metadata hash differs")
    if int(identity.get("unit_count", -1)) != unit.shape[0]:
        raise RuntimeError(f"{path}: calibration row count differs")
    neighbors = int(identity.get("neighbors", -1))
    if neighbors < 1:
        raise RuntimeError(f"{path}: invalid neighbor count")

    calibration = {
        "unit": unit,
        "local_reach": local_reach,
        "tree": cKDTree(unit),
        "neighbors": neighbors,
        "metadata": metadata,
    }
    print(
        json.dumps(
            {"canonical_recipe_loaded": str(path), "identity": identity},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return calibration, (covariance, covariance_root, coordinate_std)


def _family_require_canonical_reproduction(
    version: str,
    arguments: argparse.Namespace,
    calibration: dict[str, object],
    response_statistics: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    import scipy

    actual_environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "h5py": h5py.__version__,
        "openblas": _numpy_blas_version(),
    }
    failures: list[str] = []
    for name, expected in FAMILY_CANONICAL_ENVIRONMENT.items():
        actual = actual_environment[name]
        if actual != expected:
            failures.append(f"{name}={actual}, expected {expected}")
    if arguments.chunk_size != FAMILY_CANONICAL_CHUNK_SIZE:
        failures.append(
            f"chunk_size={arguments.chunk_size}, expected {FAMILY_CANONICAL_CHUNK_SIZE}"
        )
    if arguments.reach_calibration_count != 50_000:
        failures.append("reach_calibration_count must be 50000")
    if arguments.reach_calibration_neighbors != 32:
        failures.append("reach_calibration_neighbors must be 32")
    if arguments.reach_calibration_tau != 0.02:
        failures.append("reach_calibration_tau must be 0.02")
    reach_hash = str(calibration["metadata"].get("calibration_sha256", ""))
    if reach_hash != FAMILY_CANONICAL_REACH_SHA256[version]:
        failures.append(
            f"reach calibration hash {reach_hash}, expected "
            f"{FAMILY_CANONICAL_REACH_SHA256[version]}"
        )
    covariance_hash = _array_sha256(response_statistics[1], "<f8")
    if covariance_hash != FAMILY_CANONICAL_COVARIANCE_ROOT_SHA256[version]:
        failures.append(
            f"covariance-root hash {covariance_hash}, expected "
            f"{FAMILY_CANONICAL_COVARIANCE_ROOT_SHA256[version]}"
        )
    response_identity_hash = str(
        _family_response_identity().get("identity_sha256", "")
    )
    if response_identity_hash != FAMILY_CANONICAL_RESPONSE_IDENTITY_SHA256:
        failures.append(
            f"response identity hash {response_identity_hash}, expected "
            f"{FAMILY_CANONICAL_RESPONSE_IDENTITY_SHA256}"
        )
    if failures:
        raise RuntimeError(
            "canonical family reproduction refused:\n  - " + "\n  - ".join(failures)
        )
    print(
        json.dumps(
            {
                "canonical_reproduction_check": "passed",
                "version": version,
                "environment": actual_environment,
                "reach_calibration_sha256": reach_hash,
                "covariance_root_sha256": covariance_hash,
                "response_identity_sha256": response_identity_hash,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def _family_predict_local_reach(
    unit: np.ndarray,
    focal_reach: np.ndarray,
    calibration: dict[str, object],
) -> np.ndarray:
    neighbors = min(
        int(calibration["neighbors"]),
        int(np.asarray(calibration["local_reach"]).size),
    )
    distance, index = calibration["tree"].query(
        np.asarray(unit, dtype=np.float64), k=neighbors, workers=-1
    )
    distance = np.asarray(distance, dtype=np.float64)
    index = np.asarray(index, dtype=np.int64)
    if neighbors == 1:
        distance = distance[:, None]
        index = index[:, None]
    values = np.asarray(calibration["local_reach"], dtype=np.float64)[index]
    exact = distance[:, 0] <= 1.0e-14
    weight = 1.0 / np.maximum(distance, 1.0e-12) ** 2
    interpolated = np.sum(weight * values, axis=1) / np.sum(weight, axis=1)
    interpolated[exact] = values[exact, 0]
    return np.minimum(np.asarray(focal_reach, dtype=np.float64), interpolated)


def _family_formula_hash(parameters: dict[str, object]) -> str:
    formula = _canonical_json(parameters["formula"])
    return hashlib.sha256(formula.encode("utf-8")).hexdigest()


def _family_response_identity() -> dict[str, object]:
    rotation = _family_rotation_matrix(FAMILY_RESPONSE_ROTATION_INDEX)
    identity = {
        "candidate_id": "physical-torus-r08-k8.75",
        "construction": "physical-coordinate-orthogonal-torus-wrap.v1",
        "kappa": FAMILY_RESPONSE_KAPPA,
        "rotation_seed": FAMILY_RESPONSE_ROTATION_SEED,
        "rotation_index": FAMILY_RESPONSE_ROTATION_INDEX,
        "rotation": rotation.tolist(),
        "normalization": "(raw_cos_sin+1)/2; exact coordinate range [0,1]",
        "legacy_formula_sha256": "f6e0c7ad8354045eb56b38aca85da4eb3f138a8523eb8ee6178ea4ad5a6ed001",
    }
    identity["identity_sha256"] = hashlib.sha256(
        _canonical_json(identity).encode("utf-8")
    ).hexdigest()
    return identity


def _family_split_counts(pool_count: int) -> dict[str, int]:
    validation = int(math.floor(0.10 * int(pool_count)))
    test = int(math.floor(0.10 * int(pool_count)))
    train = int(pool_count) - validation - test
    if min(train, validation, test) < 1:
        raise ValueError("family pool is too small for the fixed 80/10/10 split")
    return {"train": train, "validation": validation, "test": test}


def _family_pool_suffix(split: str) -> str:
    return "train" if split == "train" else f"{split}_pool"


def _family_dataset_name(field: str, split: str) -> str:
    suffix = _family_pool_suffix(split)
    if field == "y_clean":
        return f"y_{suffix}_clean"
    if field == "y_observed":
        return f"y_{suffix}_observed"
    return f"{field}_{suffix}"


def _family_rng(
    base_seed: int,
    split: str,
    start: int,
    stream: int,
    ambient_d: int = 0,
) -> np.random.Generator:
    sequence = np.random.SeedSequence(
        [
            int(base_seed),
            int(FAMILY_SPLIT_CODES[split]),
            int(start),
            int(stream),
            int(ambient_d),
        ]
    )
    return np.random.default_rng(sequence)


def _family_truncated_gamma(
    generator: np.random.Generator,
    count: int,
    shape: float,
    upper: float,
) -> np.ndarray:
    output = np.empty(int(count), dtype=np.float32)
    completed = 0
    while completed < output.size:
        draw = generator.gamma(
            shape=float(shape),
            scale=2.0,
            size=max(4096, output.size - completed),
        )
        draw = draw[draw <= float(upper)]
        take = min(draw.size, output.size - completed)
        if take:
            output[completed : completed + take] = draw[:take]
            completed += take
    return output


def _family_create_dataset(
    output: h5py.File,
    name: str,
    shape: tuple[int, ...],
    chunk_size: int,
    dtype: str = "f4",
) -> h5py.Dataset:
    rows = min(max(1, int(chunk_size)), max(1, int(shape[0])), 16_384)
    return output.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        chunks=(rows, *shape[1:]),
    )


def _family_identity(
    version: str,
    ambient_d: int,
    pool_count: int,
    split_counts: dict[str, int],
    chunk_size: int,
    sigma: float,
    y_noise_relative: float,
    validation_cap: int,
    eval_cap: int,
    parameters: dict[str, object],
    calibration: dict[str, object],
) -> dict[str, object]:
    raw_global_reach = float(FAMILY_RAW_GLOBAL_REACH[version])
    target_global_reach = math.sqrt(float(ambient_d))
    return {
        "schema": FAMILY_SCHEMA,
        "dataset": f"bumpy_sphere_{version}",
        "geometry_formula": parameters["formula"],
        "geometry_formula_sha256": _family_formula_hash(parameters),
        "response": _family_response_identity(),
        "pool_count": int(pool_count),
        "ambient_d": int(ambient_d),
        "manifold_d": 2,
        "split_counts": split_counts,
        "split_rule": "three independent RNG streams with exact 80/10/10 counts",
        "split_iid": True,
        "shared_pairing_rule": "same split/row latent draws across v1/v2/v3 and D10/D20",
        "latent_seed": FAMILY_LATENT_SEED,
        "x_noise_seed": FAMILY_X_NOISE_SEED,
        "y_noise_seed": FAMILY_Y_NOISE_SEED,
        "selection_seed": FAMILY_SELECTION_SEED,
        "rng_chunk_key": "SeedSequence(base_seed,split_code,row_start,stream,ambient_d); ambient_d=0 for latent/Y",
        "chunk_size": int(chunk_size),
        "nested_training_prefix_contract": True,
        "raw_global_reach": raw_global_reach,
        "target_global_reach": target_global_reach,
        "geometry_scale": target_global_reach / raw_global_reach,
        "global_reach_calibration": "65536-point equal-area screen plus multi-start float64 continuous focal minimization; N=50000 exhaustive pairwise audit was nonlimiting",
        "local_reach_rule": "kNN-IDW of float64 pairwise+focal calibration, then pointwise min with analytic focal radius",
        "reach_calibration": calibration["metadata"],
        "x_noise_sigma": float(sigma),
        "x_noise_reach_safety": NOISE_REACH_SAFETY,
        "x_noise_rule": "target-D Gaussian direction; truncated chi-square_D radius; tangent projection; local focal-reach scaling",
        "y_noise_relative": float(y_noise_relative),
        "y_noise_rule": "relative*Cov(Y_clean)^(1/2) Gaussian; covariance from deterministic 262144-point equal-area sphere rule",
        "validation_cap": int(validation_cap),
        "testing_cap": int(eval_cap),
        "cache_policy": "no cross-cell or cross-dataset cache",
    }


def _initialize_family_output(
    output_path: Path,
    identity: dict[str, object],
    covariance: np.ndarray,
    covariance_root: np.ndarray,
    coordinate_std: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ambient_d = int(identity["ambient_d"])
    chunk_size = int(identity["chunk_size"])
    split_counts = {
        key: int(value) for key, value in dict(identity["split_counts"]).items()
    }
    with h5py.File(output_path, "w", libver="latest") as output:
        output.attrs["identity_json"] = _canonical_json(identity)
        output.attrs["complete"] = False
        output.attrs["selected_views_complete"] = False
        output.attrs["ambient_d"] = ambient_d
        output.attrs["manifold_d"] = 2
        output.attrs["global_reach"] = float(identity["target_global_reach"])
        output.attrs["cache_policy"] = "no_cross_cell_cache"
        for split in FAMILY_SPLITS:
            output.attrs[f"completed_{split}"] = 0
            count = split_counts[split]
            for field, trailing in (
                ("latent", (3,)),
                ("xclean", (ambient_d,)),
                ("x", (ambient_d,)),
                ("tangent", (ambient_d, 2)),
                ("local_reach", ()),
                ("noise_reach", ()),
                ("y_clean", (6,)),
                ("y_observed", (6,)),
            ):
                _family_create_dataset(
                    output,
                    _family_dataset_name(field, split),
                    (count, *trailing),
                    chunk_size,
                )
        output.create_dataset("y_covariance_clean", data=covariance.astype(np.float64))
        output.create_dataset(
            "y_noise_covariance_root", data=covariance_root.astype(np.float64)
        )
        output.create_dataset(
            "y_coordinate_std_clean", data=coordinate_std.astype(np.float64)
        )
        output.create_dataset("y_min", data=np.zeros(6, dtype=np.float32))
        output.create_dataset("y_span", data=np.ones(6, dtype=np.float32))
        output.create_dataset(
            "response_rotation",
            data=_family_rotation_matrix(FAMILY_RESPONSE_ROTATION_INDEX),
        )
        output.flush()


def _family_selection_indices(split: str, count: int, cap: int) -> np.ndarray:
    generator = np.random.default_rng(
        np.random.SeedSequence(
            [FAMILY_SELECTION_SEED, FAMILY_SPLIT_CODES[split], int(count), int(cap)]
        )
    )
    return np.sort(generator.choice(int(count), size=int(cap), replace=False)).astype(
        np.int64
    )


def _family_read_selected_rows(
    dataset: h5py.Dataset,
    indices: np.ndarray,
    *,
    read_rows: int = 262_144,
) -> np.ndarray:
    """Read sorted sparse rows through large contiguous HDF5 slices.

    h5py's point-selection path is extremely slow when a few thousand indices
    span a multi-million-row dataset.  The family pools are uncompressed, so a
    sequential slice-and-filter pass is much faster and returns exactly the
    same rows in exactly the same order.
    """
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if selected.size == 0:
        return np.empty((0, *dataset.shape[1:]), dtype=dataset.dtype)
    if np.any(selected[1:] <= selected[:-1]):
        raise ValueError("indices must be strictly increasing")
    if selected[0] < 0 or selected[-1] >= dataset.shape[0]:
        raise IndexError("selected row lies outside the HDF5 dataset")

    result = np.empty((selected.size, *dataset.shape[1:]), dtype=dataset.dtype)
    output_start = 0
    while output_start < selected.size:
        block_start = (int(selected[output_start]) // read_rows) * read_rows
        block_stop = min(dataset.shape[0], block_start + read_rows)
        output_stop = int(np.searchsorted(selected, block_stop, side="left"))
        block = np.asarray(dataset[block_start:block_stop])
        result[output_start:output_stop] = block[
            selected[output_start:output_stop] - block_start
        ]
        output_start = output_stop
    return result


def _family_write_selected_views(
    output: h5py.File, identity: dict[str, object]
) -> None:
    mappings = {
        "validation": {
            "latent": "latent_validation",
            "xclean": "xclean_validation",
            "x": "x_validation",
            "tangent": "tangent_validation",
            "local_reach": "local_reach_validation",
            "noise_reach": "noise_reach_validation",
            "y_clean": "y_validation_clean",
            "y_observed": "y_validation_observed",
        },
        "test": {
            "latent": "latent_eval",
            "xclean": "xclean_eval",
            "x": "x_eval",
            "tangent": "tangent_eval",
            "local_reach": "local_reach_eval",
            "noise_reach": "noise_reach_eval",
            "y_clean": "y_eval_clean",
            "y_observed": "y_eval_observed",
        },
    }
    split_counts = {
        key: int(value) for key, value in dict(identity["split_counts"]).items()
    }
    for split, fields in mappings.items():
        cap = int(
            identity["validation_cap"] if split == "validation" else identity["testing_cap"]
        )
        indices = _family_selection_indices(split, split_counts[split], cap)
        index_name = (
            "validation_pool_indices" if split == "validation" else "eval_pool_indices"
        )
        if index_name in output:
            output[index_name][:] = indices
        else:
            output.create_dataset(index_name, data=indices)
        for source_field, destination in fields.items():
            source = output[_family_dataset_name(source_field, split)]
            values = _family_read_selected_rows(source, indices)
            if destination in output:
                output[destination][:] = values
            else:
                output.create_dataset(destination, data=values)
    output.attrs["selected_views_complete"] = True
    output.flush()


def _family_split_iid_audit(
    output: h5py.File, identity: dict[str, object]
) -> dict[str, object]:
    split_counts = {
        key: int(value) for key, value in dict(identity["split_counts"]).items()
    }
    rows: dict[str, dict[str, object]] = {}
    samples: dict[str, np.ndarray] = {}
    for split in FAMILY_SPLITS:
        count = split_counts[split]
        audit_count = min(FAMILY_IID_AUDIT_COUNT, count)
        generator = np.random.default_rng(
            np.random.SeedSequence(
                [FAMILY_SELECTION_SEED, 99, FAMILY_SPLIT_CODES[split], count]
            )
        )
        indices = np.sort(generator.choice(count, size=audit_count, replace=False))
        latent = np.asarray(
            _family_read_selected_rows(
                output[_family_dataset_name("latent", split)], indices
            ),
            dtype=np.float64,
        )
        local_reach = np.asarray(
            _family_read_selected_rows(
                output[_family_dataset_name("local_reach", split)], indices
            ),
            dtype=np.float64,
        )
        y_clean = np.asarray(
            _family_read_selected_rows(
                output[_family_dataset_name("y_clean", split)], indices
            ),
            dtype=np.float64,
        )
        features = np.column_stack((latent, local_reach, y_clean))
        samples[split] = features
        rows[split] = {
            "sample_count": int(audit_count),
            "latent_mean": latent.mean(axis=0).tolist(),
            "latent_covariance": np.cov(latent, rowvar=False).tolist(),
            "local_reach_mean": float(local_reach.mean()),
            "local_reach_std": float(local_reach.std()),
            "y_clean_mean": y_clean.mean(axis=0).tolist(),
            "y_clean_std": y_clean.std(axis=0).tolist(),
        }
    comparisons: dict[str, object] = {}
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        a = samples[first]
        b = samples[second]
        mean_difference = np.abs(a.mean(axis=0) - b.mean(axis=0))
        pooled_scale = np.maximum(
            0.5 * (a.std(axis=0) + b.std(axis=0)), np.finfo(np.float64).tiny
        )
        comparisons[f"{first}_vs_{second}"] = {
            "maximum_mean_difference_in_pooled_std_units": float(
                np.max(mean_difference / pooled_scale)
            ),
            "latent_covariance_frobenius_difference": float(
                np.linalg.norm(np.cov(a[:, :3], rowvar=False) - np.cov(b[:, :3], rowvar=False))
            ),
        }
    return {
        "construction_guarantee": "independent split-specific latent RNG streams; split assignment cannot depend on X/Y values",
        "paired_across_datasets": True,
        "splits": rows,
        "comparisons": comparisons,
    }


def _family_generate_file(
    output_path: Path,
    version: str,
    ambient_d: int,
    arguments: argparse.Namespace,
    calibration: dict[str, object],
    response_statistics: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[str, object]:
    parameters = _family_geometry_parameters(version)
    split_counts = _family_split_counts(arguments.n)
    if arguments.validation_cap > split_counts["validation"]:
        raise ValueError("--validation-cap exceeds the validation pool")
    if arguments.eval_cap > split_counts["test"]:
        raise ValueError("--eval-cap exceeds the testing pool")
    if response_statistics is None:
        response_statistics = _family_response_statistics(
            parameters, arguments.y_noise_relative
        )
    covariance, covariance_root, coordinate_std = response_statistics
    identity = _family_identity(
        version,
        ambient_d,
        arguments.n,
        split_counts,
        arguments.chunk_size,
        arguments.sigma,
        arguments.y_noise_relative,
        arguments.validation_cap,
        arguments.eval_cap,
        parameters,
        calibration,
    )
    identity_text = _canonical_json(identity)
    if output_path.exists():
        with h5py.File(output_path, "r") as existing:
            if str(existing.attrs.get("identity_json", "")) != identity_text:
                raise RuntimeError(f"{output_path}: existing identity differs")
            if bool(existing.attrs.get("complete", False)):
                print(f"complete; skipping {output_path}", flush=True)
                return json.loads(str(existing.attrs["summary_json"]))
    else:
        _initialize_family_output(
            output_path, identity, covariance, covariance_root, coordinate_std
        )

    scale = float(identity["geometry_scale"])
    target_reach = float(identity["target_global_reach"])
    raw_global_reach = float(identity["raw_global_reach"])
    covariance_transform = covariance_root.astype(np.float32).T
    started = time.perf_counter()
    initial_completed = 0
    with h5py.File(output_path, "r+") as output:
        initial_completed = sum(
            int(output.attrs.get(f"completed_{split}", 0)) for split in FAMILY_SPLITS
        )
        for split in FAMILY_SPLITS:
            split_count = split_counts[split]
            completed = int(output.attrs.get(f"completed_{split}", 0))
            for start in range(completed, split_count, arguments.chunk_size):
                stop = min(split_count, start + arguments.chunk_size)
                current = stop - start
                latent_generator = _family_rng(
                    FAMILY_LATENT_SEED, split, start, stream=1
                )
                unit = latent_generator.standard_normal((current, 3))
                unit /= np.maximum(
                    np.linalg.norm(unit, axis=1, keepdims=True),
                    np.finfo(np.float64).tiny,
                )
                geometry = _family_geometry(unit, parameters)
                raw_focal_reach = geometry["focal_reach"]
                raw_local_reach = _family_predict_local_reach(
                    geometry["unit"], raw_focal_reach, calibration
                )
                if float(np.min(raw_local_reach)) < raw_global_reach * (1.0 - 2.0e-5):
                    raise RuntimeError(
                        f"{version}: generated local reach {float(np.min(raw_local_reach)):.10g} "
                        f"falls below frozen global reach {raw_global_reach:.10g}"
                    )
                local_reach = (scale * raw_local_reach).astype(np.float32)
                noise_reach = (NOISE_REACH_SAFETY * local_reach).astype(np.float32)

                xclean = np.zeros((current, ambient_d), dtype=np.float32)
                xclean[:, :3] = (
                    scale
                    * geometry["radius"][:, None]
                    * geometry["unit"]
                ).astype(np.float32)
                tangent = np.zeros((current, ambient_d, 2), dtype=np.float32)
                tangent[:, :3, 0] = geometry["tangent_first"].astype(np.float32)
                tangent[:, :3, 1] = geometry["tangent_second"].astype(np.float32)

                direction_generator = _family_rng(
                    FAMILY_X_NOISE_SEED,
                    split,
                    start,
                    stream=1,
                    ambient_d=ambient_d,
                )
                direction = direction_generator.standard_normal(
                    (current, ambient_d), dtype=np.float32
                )
                direction /= np.maximum(
                    np.linalg.norm(direction, axis=1, keepdims=True),
                    np.finfo(np.float32).tiny,
                )
                gamma_generator = _family_rng(
                    FAMILY_X_NOISE_SEED,
                    split,
                    start,
                    stream=2,
                    ambient_d=ambient_d,
                )
                gamma = _family_truncated_gamma(
                    gamma_generator,
                    current,
                    ambient_d / 2.0,
                    0.81 * target_reach * target_reach / (arguments.sigma**2),
                )
                radius_draw = (
                    np.float32(arguments.sigma)
                    * noise_reach
                    / np.float32(target_reach)
                    * np.sqrt(gamma)
                )
                noise = direction * radius_draw[:, None]
                coefficient = np.einsum("ndm,nd->nm", tangent, noise, optimize=True)
                noise -= np.einsum("ndm,nm->nd", tangent, coefficient, optimize=True)
                ratio_max = float(
                    np.max(
                        np.linalg.norm(noise, axis=1)
                        / np.maximum(local_reach, np.finfo(np.float32).tiny)
                    )
                )
                if ratio_max > 0.9 * NOISE_REACH_SAFETY + 2.0e-5:
                    raise RuntimeError(
                        f"local-reach tube support failed: max ratio={ratio_max:.8g}"
                    )
                x = xclean + noise

                y_clean = _family_response(
                    geometry["radius"], geometry["unit"]
                )
                y_generator = _family_rng(
                    FAMILY_Y_NOISE_SEED, split, start, stream=1
                )
                y_observed = y_clean + y_generator.standard_normal(
                    y_clean.shape, dtype=np.float32
                ) @ covariance_transform

                values = {
                    "latent": geometry["unit"].astype(np.float32),
                    "xclean": xclean,
                    "x": x,
                    "tangent": tangent,
                    "local_reach": local_reach,
                    "noise_reach": noise_reach,
                    "y_clean": y_clean,
                    "y_observed": y_observed,
                }
                for field, value in values.items():
                    output[_family_dataset_name(field, split)][start:stop] = value
                output.attrs.modify(f"completed_{split}", stop)
                output.attrs["progress"] = f"{split} {stop}/{split_count}"
                output.flush()

                elapsed = time.perf_counter() - started
                generated = (
                    sum(
                        int(output.attrs.get(f"completed_{name}", 0))
                        for name in FAMILY_SPLITS
                    )
                    - initial_completed
                )
                remaining = arguments.n - sum(
                    int(output.attrs.get(f"completed_{name}", 0))
                    for name in FAMILY_SPLITS
                )
                rate = generated / max(elapsed, 1.0e-12)
                eta = remaining / max(rate, 1.0e-12)
                print(
                    f"{output_path.name} {split} {stop}/{split_count} "
                    f"total={sum(int(output.attrs.get(f'completed_{name}', 0)) for name in FAMILY_SPLITS)}/{arguments.n} "
                    f"rate={rate:.0f}/s eta={eta/60:.1f}min "
                    f"local=[{float(np.min(raw_local_reach)):.6g},{float(np.max(raw_local_reach)):.6g}] "
                    f"noise_ratio_max={ratio_max:.6f}",
                    flush=True,
                )

        _family_write_selected_views(output, identity)
        iid_audit = _family_split_iid_audit(output, identity)
        summary = {
            "schema": FAMILY_SCHEMA,
            "file": str(output_path),
            "dataset": identity["dataset"],
            "ambient_d": ambient_d,
            "pool_count": arguments.n,
            "split_counts": split_counts,
            "validation_selected_count": arguments.validation_cap,
            "testing_selected_count": arguments.eval_cap,
            "global_reach": target_reach,
            "geometry_scale": scale,
            "x_noise_sigma": arguments.sigma,
            "y_noise_relative": arguments.y_noise_relative,
            "split_iid": True,
            "iid_audit": iid_audit,
            "generator_seconds_this_invocation": time.perf_counter() - started,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        output.attrs["split_iid_audit_json"] = _canonical_json(iid_audit)
        output.attrs["summary_json"] = _canonical_json(summary)
        output.attrs["progress"] = "complete"
        output.attrs["complete"] = True
        output.flush()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def _family_output_name(version: str, ambient_d: int, count: int) -> str:
    count_label = f"{count // 1_000_000}M" if count % 1_000_000 == 0 else str(count)
    return (
        f"bumpy_sphere_{version}__torus_iso_r08_k8p75__"
        f"D{ambient_d}_N{count_label}.h5"
    )


def _parse_csv_values(text: str, cast: type) -> tuple:
    values = tuple(cast(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise ValueError("comma-separated value list is empty")
    return values


def _family_reach_audit(versions: tuple[str, ...], count: int) -> dict[str, object]:
    unit = _family_equal_area_unit(count)
    result: dict[str, object] = {}
    for version in versions:
        parameters = _family_geometry_parameters(version)
        geometry = _family_geometry(unit, parameters)
        focal = geometry["focal_reach"]
        result[version] = {
            "sample_count": int(count),
            "formula_sha256": _family_formula_hash(parameters),
            "frozen_raw_global_reach": FAMILY_RAW_GLOBAL_REACH[version],
            "sample_focal_reach_quantiles": np.percentile(
                focal, [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 100]
            ).tolist(),
            "sample_radius_quantiles": np.percentile(
                geometry["radius"], [0, 1, 5, 25, 50, 75, 95, 99, 100]
            ).tolist(),
        }
    return result


def generate_bumpy_sphere_family(arguments: argparse.Namespace) -> int:
    versions = _parse_csv_values(arguments.versions, str)
    dimensions = _parse_csv_values(arguments.dimensions, int)
    if any(version not in FAMILY_VERSIONS for version in versions):
        raise ValueError(f"--versions must be drawn from {FAMILY_VERSIONS}")
    if any(dimension not in FAMILY_DIMENSIONS for dimension in dimensions):
        raise ValueError(f"--dimensions must be drawn from {FAMILY_DIMENSIONS}")
    if arguments.n < 10:
        raise ValueError("--n must be at least 10")
    if arguments.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    if arguments.sigma <= 0.0:
        raise ValueError("--sigma must be positive")
    if arguments.y_noise_relative < 0.0:
        raise ValueError("--y-noise-relative must be nonnegative")
    if arguments.family_audit_only:
        print(
            json.dumps(
                _family_reach_audit(versions, arguments.family_audit_count),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output_dir = (
        Path(arguments.out_dir).expanduser().resolve()
        if arguments.out_dir
        else Path(__file__).resolve().parents[1] / "dataset" / "bumpy_sphere"
    )
    if arguments.output_h5 and (len(versions) != 1 or len(dimensions) != 1):
        raise ValueError("--output-h5 requires exactly one version and one dimension")
    recipe_dir = (
        Path(arguments.family_recipe_dir).expanduser().resolve()
        if arguments.family_recipe_dir
        else None
    )
    if arguments.write_family_recipe and recipe_dir is None:
        raise ValueError("--write-family-recipe requires --family-recipe-dir")
    for version in versions:
        parameters = _family_geometry_parameters(version)
        recipe_path = (
            _family_recipe_path(recipe_dir, version)
            if recipe_dir is not None
            else None
        )
        if recipe_path is not None and recipe_path.is_file():
            calibration, response_statistics = _family_load_recipe(
                recipe_path,
                version,
                parameters,
                arguments.y_noise_relative,
            )
        else:
            if recipe_path is not None and not arguments.write_family_recipe:
                raise FileNotFoundError(
                    f"canonical recipe is missing: {recipe_path}; "
                    "use --write-family-recipe only on the canonical source host"
                )
            calibration = _family_build_reach_calibration(
                version,
                parameters,
                arguments.reach_calibration_count,
                arguments.reach_calibration_tau,
                arguments.reach_calibration_device,
                arguments.reach_calibration_block_size,
                arguments.reach_calibration_neighbors,
            )
            response_statistics = _family_response_statistics(
                parameters, arguments.y_noise_relative
            )
            if recipe_path is not None:
                _family_write_recipe(
                    recipe_path,
                    version,
                    parameters,
                    calibration,
                    response_statistics,
                    arguments.y_noise_relative,
                )
        if arguments.require_canonical_family:
            _family_require_canonical_reproduction(
                version, arguments, calibration, response_statistics
            )
        print(
            json.dumps(
                {"version": version, "reach_calibration": calibration["metadata"]},
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        if arguments.family_recipe_only:
            continue
        for ambient_d in dimensions:
            output_path = (
                Path(arguments.output_h5).expanduser().resolve()
                if arguments.output_h5
                else output_dir
                / _family_output_name(version, ambient_d, arguments.n)
            )
            _family_generate_file(
                output_path,
                version,
                ambient_d,
                arguments,
                calibration,
                response_statistics,
            )
    return 0


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "setup_paths.m").is_file() and (path / "datasets").is_dir():
            return path
    return start.parents[2]


def bump_params() -> dict[str, np.ndarray]:
    centers = np.asarray(
        [
            [0.83, -0.22, 0.51],
            [-0.31, 0.91, 0.27],
            [0.15, -0.62, 0.77],
            [-0.76, -0.44, -0.48],
            [0.55, 0.39, -0.74],
            [-0.18, 0.10, 0.98],
        ],
        dtype=np.float64,
    )
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    return {
        "centers": centers,
        "amp": np.asarray([0.18, -0.12, 0.10, 0.075, -0.065, 0.055], dtype=np.float64),
        "width": np.asarray([0.36, 0.52, 0.43, 0.61, 0.39, 0.56], dtype=np.float64),
        "linear": np.asarray([0.035, -0.025, 0.045], dtype=np.float64),
    }


def bumpy_radius(u: np.ndarray, bump: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = u.shape[0]
    r = 1.0 + u @ bump["linear"]
    grad = np.repeat(bump["linear"][None, :], n, axis=0)
    hess = np.zeros((n, 3, 3), dtype=np.float64)
    for c, a, w in zip(bump["centers"], bump["amp"], bump["width"]):
        w2 = float(w * w)
        g = np.exp((u @ c - 1.0) / w2)
        r += a * g
        grad += (a / w2) * g[:, None] * c[None, :]
        hess += (a / (w2 * w2)) * g[:, None, None] * np.outer(c, c)[None, :, :]
    if float(np.min(r)) <= 0.35:
        raise RuntimeError(f"radius field became too small: min(r)={np.min(r):.5g}")
    return r, grad, hess


def tangent_frame(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.repeat(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64), u.shape[0], axis=0)
    near = np.abs(u[:, 2]) > 0.88
    a[near] = np.asarray([0.0, 1.0, 0.0])
    e1 = np.cross(a, u)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), np.finfo(np.float64).eps)
    e2 = np.cross(u, e1)
    e2 /= np.maximum(np.linalg.norm(e2, axis=1, keepdims=True), np.finfo(np.float64).eps)
    return e1, e2


def curvature_value(
    u: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    r: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    j1: np.ndarray,
    j2: np.ndarray,
    normal_top: np.ndarray,
) -> np.ndarray:
    e = np.sum(j1 * j1, axis=1)
    f = np.sum(j1 * j2, axis=1)
    g = np.sum(j2 * j2, axis=1)
    det_i = np.maximum(e * g - f * f, np.finfo(np.float64).eps)

    h11 = np.einsum("ni,nij,nj->n", e1, hess, e1)
    h22 = np.einsum("ni,nij,nj->n", e2, hess, e2)
    h12 = np.einsum("ni,nij,nj->n", e1, hess, e2)
    gu = np.sum(grad * u, axis=1)
    r11 = h11 - gu
    r22 = h22 - gu
    r12 = h12
    r1 = np.sum(grad * e1, axis=1)
    r2 = np.sum(grad * e2, axis=1)

    x11 = r11[:, None] * u + 2.0 * r1[:, None] * e1 - r[:, None] * u
    x22 = r22[:, None] * u + 2.0 * r2[:, None] * e2 - r[:, None] * u
    x12 = r12[:, None] * u + r1[:, None] * e2 + r2[:, None] * e1

    ll = np.sum(normal_top * x11, axis=1)
    mm = np.sum(normal_top * x12, axis=1)
    nn = np.sum(normal_top * x22, axis=1)

    aa = (g * ll - f * mm) / det_i
    bb = (g * mm - f * nn) / det_i
    cc = (-f * ll + e * mm) / det_i
    dd = (-f * mm + e * nn) / det_i
    tr = aa + dd
    det_s = aa * dd - bb * cc
    disc = np.maximum(0.0, tr * tr - 4.0 * det_s)
    k1 = 0.5 * (tr + np.sqrt(disc))
    k2 = 0.5 * (tr - np.sqrt(disc))
    return np.maximum(np.abs(k1), np.abs(k2))


def gamma_truncated(rng: np.random.Generator, shape: float, scale: float, upper: float, n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.float64)
    filled = 0
    batch = max(4096, n)
    while filled < n:
        draws = rng.gamma(shape=shape, scale=scale, size=batch)
        draws = draws[draws <= upper]
        take = min(draws.size, n - filled)
        if take > 0:
            out[filled : filled + take] = draws[:take]
            filled += take
    return out


def bumpy_y(u: np.ndarray, freq: float) -> np.ndarray:
    x = u[:, 0].astype(np.float64)
    y = u[:, 1].astype(np.float64)
    z = u[:, 2].astype(np.float64)
    f = float(freq)

    amp1 = 1.00 + 0.18 * z - 0.10 * x * y
    amp2 = 0.92 - 0.14 * x + 0.12 * y * z
    amp3 = 0.85 + 0.16 * y - 0.08 * z**2
    amp4 = 0.78 + 0.10 * x * z - 0.06 * y

    phase1 = f * (2.10 * x + 1.25 * y - 0.70 * z + 0.35 * x * y)
    phase2 = f * (1.65 * y - 1.05 * x * z + 0.55 * z + 0.25 * y**2)
    phase3 = f * (1.30 * z + 0.95 * x - 0.85 * y * z + 0.22 * x**2) + 0.20 * x * y
    phase4 = f * (1.10 * (x - y) + 0.75 * z + 0.40 * x * y * z)

    return np.stack(
        [
            amp1 * np.sin(np.pi * phase1) + 0.18 * (x**2 - y**2),
            amp2 * np.cos(np.pi * phase2) + 0.15 * x * z - 0.06 * y**2,
            amp3 * np.sin(np.pi * phase3 + 0.35 * np.cos(np.pi * f * y)) + 0.10 * x * y,
            amp4 * np.cos(np.pi * phase4 + 0.25 * np.sin(np.pi * f * x * z)) + 0.12 * z**2 - 0.05 * x,
        ],
        axis=1,
    ).astype(np.float32)


def write_h5(path: Path, gamma: dict[str, np.ndarray | float | int | str], y_freq: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        g = h5.create_group("gamma")
        for key, value in gamma.items():
            if isinstance(value, str):
                g.attrs[key] = value
            else:
                g.create_dataset(key, data=value)
        h5.create_dataset("idxGood", data=np.arange(1, int(gamma["N"]) + 1, dtype=np.uint32))
        h5.create_dataset("Y", data=bumpy_y(np.asarray(gamma["Xlatent"]), y_freq))


def write_matlab_v5(path: Path, gamma: dict[str, np.ndarray | float | int | str], y_freq: float) -> None:
    import scipy.io

    xlatent = np.asarray(gamma["Xlatent"])
    gamma_mat = dict(gamma)
    gamma_mat["Xlatent"] = xlatent.T
    gamma_mat["Xclean"] = np.asarray(gamma["Xclean"]).T
    gamma_mat["Tan"] = np.asarray(gamma["Tan"]).transpose(1, 2, 0)
    gamma_mat["X_RelCstDevSig"] = np.asarray(gamma["X_RelCstDevSig"]).T
    gamma_mat["X_UniCstDevSig"] = np.asarray(gamma["X_UniCstDevSig"]).T
    gamma_mat["NoiseRel"] = np.asarray(gamma["NoiseRel"]).T
    for key in ("loc_reach_arr", "CurvVal", "Kapval"):
        gamma_mat[key] = np.asarray(gamma[key]).reshape(1, -1)

    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(
        path,
        {
            "gamma": gamma_mat,
            "idxGood": np.arange(1, int(gamma["N"]) + 1, dtype=np.uint32).reshape(1, -1),
            "Y": bumpy_y(xlatent, y_freq).T,
        },
        do_compression=False,
        long_field_names=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate canonical manifold datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=("bumpy_sphere", "torus_d10", "bumpy_sphere_family"),
        default="bumpy_sphere",
    )
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-rescale-reach", action="store_true")
    parser.add_argument("--y-freq", type=float, default=1.0)
    parser.add_argument("--matlab-v5", action="store_true", help="write a MATLAB v5 MAT file in D x N layout")
    parser.add_argument("--source-h5", default="")
    parser.add_argument("--output-h5", default="")
    parser.add_argument("--validation-cap", type=int, default=5_000)
    parser.add_argument("--eval-cap", type=int, default=5_000)
    parser.add_argument("--y-noise-relative", type=float, default=0.003)
    parser.add_argument("--validation-noise-seed", type=int, default=2026082101)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--versions", default="v1,v2,v3")
    parser.add_argument("--dimensions", default="10,20")
    parser.add_argument("--family-audit-only", action="store_true")
    parser.add_argument("--family-audit-count", type=int, default=262_144)
    parser.add_argument("--reach-calibration-count", type=int, default=50_000)
    parser.add_argument("--reach-calibration-neighbors", type=int, default=32)
    parser.add_argument("--reach-calibration-tau", type=float, default=0.02)
    parser.add_argument("--reach-calibration-device", default="cuda:0")
    parser.add_argument("--reach-calibration-block-size", type=int, default=1024)
    parser.add_argument(
        "--family-recipe-dir",
        default="",
        help=(
            "directory containing per-version canonical reach/covariance recipes; "
            "these small files replace cross-machine recalibration"
        ),
    )
    parser.add_argument(
        "--write-family-recipe",
        action="store_true",
        help="write missing recipes; use only in the canonical source environment",
    )
    parser.add_argument(
        "--family-recipe-only",
        action="store_true",
        help="write or validate recipes without generating an H5 manifold pool",
    )
    parser.add_argument(
        "--require-canonical-family",
        action="store_true",
        help=(
            "refuse generation unless Python/numerical versions, chunking, reach "
            "calibration, and response covariance match the canonical D20 files"
        ),
    )
    args = parser.parse_args()

    if args.dataset == "bumpy_sphere_family":
        return generate_bumpy_sphere_family(args)

    if args.dataset == "torus_d10":
        if args.seed is None:
            args.seed = 2026080801
        return generate_torus_d10(args)

    if args.seed is None:
        args.seed = 20260509

    if args.d < 3:
        raise ValueError("--d must be at least 3")
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir) if args.out_dir else (
        find_repo_root(Path(__file__).resolve()) / "dataset" / "bumpy_sphere"
    )

    u = rng.normal(size=(args.n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    bump = bump_params()
    radial, grad, hess = bumpy_radius(u, bump)
    e1, e2 = tangent_frame(u)
    dr1 = np.sum(grad * e1, axis=1)
    dr2 = np.sum(grad * e2, axis=1)
    j1 = radial[:, None] * e1 + dr1[:, None] * u
    j2 = radial[:, None] * e2 + dr2[:, None] * u

    tan1 = j1 / np.maximum(np.linalg.norm(j1, axis=1, keepdims=True), np.finfo(np.float64).eps)
    j2p = j2 - np.sum(tan1 * j2, axis=1, keepdims=True) * tan1
    tan2 = j2p / np.maximum(np.linalg.norm(j2p, axis=1, keepdims=True), np.finfo(np.float64).eps)
    normal_top = np.cross(tan1, tan2)
    normal_top /= np.maximum(np.linalg.norm(normal_top, axis=1, keepdims=True), np.finfo(np.float64).eps)
    flip = np.sum(normal_top * u, axis=1) < 0
    normal_top[flip] *= -1.0

    curv = curvature_value(u, e1, e2, radial, grad, hess, j1, j2, normal_top)
    loc_reach = 1.0 / np.maximum(curv, 1.0e-8)
    reach = float(np.min(loc_reach[np.isfinite(loc_reach) & (loc_reach > 0)]))

    xclean = np.zeros((args.n, args.d), dtype=np.float64)
    xclean[:, :3] = radial[:, None] * u
    tan = np.zeros((args.n, args.d, 2), dtype=np.float64)
    tan[:, :3, 0] = tan1
    tan[:, :3, 1] = tan2

    scale = 1.0
    if not args.no_rescale_reach:
        scale = math.sqrt(float(args.d)) / reach
        xclean *= scale
        loc_reach *= scale
        curv /= scale
        reach *= scale

    global_reach = math.sqrt(float(args.d))
    noise_reach = NOISE_REACH_SAFETY * loc_reach
    upper = 0.81 * global_reach * global_reach / (args.sigma * args.sigma)
    radius_draw = gamma_truncated(rng, shape=args.d / 2.0, scale=2.0, upper=upper, n=args.n)
    rndrad = args.sigma * noise_reach / global_reach * np.sqrt(radius_draw)
    ambient_noise = rng.normal(size=(args.n, args.d))
    ambient_noise /= np.maximum(np.linalg.norm(ambient_noise, axis=1, keepdims=True), np.finfo(np.float64).eps)
    ambient_noise *= rndrad[:, None]
    coeff = np.einsum("nd,ndm->nm", ambient_noise, tan)
    tangent_part = np.einsum("nm,ndm->nd", coeff, tan)
    noise_rel = ambient_noise - tangent_part
    xnoisy = xclean + noise_rel
    ratio = np.linalg.norm(noise_rel, axis=1) / np.maximum(loc_reach, np.finfo(np.float64).eps)
    if float(np.max(ratio)) > 0.9 * NOISE_REACH_SAFETY + 2.0e-5:
        raise RuntimeError(f"noise/local-reach guarantee failed: max ratio={np.max(ratio):.8g}")

    stem = f"BumpySphereReachRelative_D={args.d}_N={args.n}_sigma={args.sigma:g}_seed={args.seed}"
    gamma = {
        "Name": "BumpySphereReachRelative",
        "D": np.int32(args.d),
        "m": np.int32(2),
        "N": np.int64(args.n),
        "Xlatent": u.astype(np.float32),
        "Xclean": xclean.astype(np.float32),
        "Tan": tan.astype(np.float32),
        "X_RelCstDevSig": xnoisy.astype(np.float32),
        "X_UniCstDevSig": xnoisy.astype(np.float32),
        "NoiseRel": noise_rel.astype(np.float32),
        "loc_reach_arr": loc_reach.astype(np.float32),
        "Reach": np.float32(reach),
        "CurvVal": curv.astype(np.float32),
        "Kapval": curv.astype(np.float32),
        "normalSigma": np.float32(args.sigma),
        "rescale": "Reach" if not args.no_rescale_reach else "None",
        "type_noise_off_curve": "relative_orthogonal_tube",
    }
    out_file = out_dir / f"{stem}.mat"
    if args.matlab_v5:
        write_matlab_v5(out_file, gamma, args.y_freq)
    else:
        write_h5(out_file, gamma, args.y_freq)

    summary = {
        "file": str(out_file),
        "D": args.d,
        "N": args.n,
        "sigma": args.sigma,
        "seed": args.seed,
        "rescale_factor": scale,
        "Reach": reach,
        "matlab_v5": bool(args.matlab_v5),
        "local_reach_quantiles": np.percentile(loc_reach, [0, 1, 5, 25, 50, 75, 95, 99, 100]).tolist(),
        "noise_norm_over_local_reach_quantiles": np.percentile(ratio, [0, 1, 5, 25, 50, 75, 95, 99, 100]).tolist(),
        "max_noise_norm_over_local_reach": float(np.max(ratio)),
    }
    summary_file = out_dir / f"{stem}_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
