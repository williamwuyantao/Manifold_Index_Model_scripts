"""Phase 1 cut construction, hybrid initialization and iteration schedules."""
from __future__ import annotations
from dataclasses import asdict
import math
import gc
import time
import numpy as np


def _cuts(ambient_dimension: int, neighbor_count: int) -> np.ndarray:
    count = 128
    low = 3 * ambient_dimension
    high = math.floor(0.8 * neighbor_count)
    span = high - low + 1
    if span < count:
        raise ValueError(f"cut interval [{low},{high}] is too short")
    values = low + np.floor(
        np.arange(count, dtype=np.float64) * (span - 1) / (count - 1)
    ).astype(np.int32)
    if np.any(np.diff(values) <= 0):
        raise RuntimeError("production cut construction produced duplicates")
    return np.ascontiguousarray(values[None, :])


def _phase2_schedules(
    phase1_radius: np.ndarray,
    owner: np.ndarray,
    axial_tangent: np.ndarray,
    h_scale: float,
    rho_init: float,
    bite: int,
    ambient_dimension: int,
    manifold_dimension: int,
    iteration_count: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    finite_owner = (
        owner
        & np.isfinite(phase1_radius)
        & np.isfinite(axial_tangent)
        & (phase1_radius > 0)
        & (axial_tangent > 0)
    )
    if not np.array_equal(finite_owner, owner):
        raise RuntimeError("Phase 1 returned invalid owner diagnostics")
    phase1_h = max(float(np.median(phase1_radius[finite_owner].astype(np.float64))), 1e-6)
    base_h = h_scale * phase1_h
    powers = np.arange(iteration_count, dtype=np.float64)
    h_growth = math.exp(1.0 / (2.0 * max(4, ambient_dimension)))
    rho_decay = math.exp(-1.0 / (2.0 * manifold_dimension + 8.0))
    h_schedule = np.asarray(
        base_h * np.power(h_growth, np.minimum(powers, bite)), dtype=np.float32
    )
    rho_schedule = np.asarray(
        rho_init * np.power(rho_decay, powers), dtype=np.float32
    )
    return phase1_h, base_h, h_schedule, rho_schedule


def _hybrid_phase1(
    context,
    x: np.ndarray,
    index: np.ndarray | None,
    counts: np.ndarray,
    cuts: np.ndarray,
    manifold_dimension: int,
    reserve_bytes: int,
    maximum_query_block: int | None = None,
    capacity_options=None,
    response: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    query_limit = int(x.shape[0]) if maximum_query_block is None else int(maximum_query_block)
    if query_limit < 1:
        raise ValueError("maximum_query_block must be positive")
    common = {
        "gpu_memory_fraction": 0.88,
        "reserve_memory_fraction": 0.08,
        "reserve_memory_bytes": reserve_bytes,
        "heavy_factor": 0.10,
        "jitter": 1.0e-6,
        "minimum_query_block": 1,
        "maximum_query_block": min(int(x.shape[0]), query_limit),
        "stable_window": 7,
        "neighbor_scaled_max": 1.0,
        "split_scaled_max": 6.0,
        "split_seed": 730021,
        "cut_max_fraction": 0.8,
        "h_null_correction": False,
        "return_debug": True,
    }
    if capacity_options is not None:
        common['capacity_options'] = capacity_options
    if index is None:
        if response is None:
            raise ValueError("generated Phase1 requires response coordinates")
        common['response'] = response
    started = time.perf_counter()
    if hasattr(context._core._library, "manor_m_cuda_phase1_hybrid_precomputed"):
        phase1, fused_stats = context.phase1_precomputed(
            x,
            index,
            counts,
            cuts,
            manifold_dimension,
            selection_mode=0,
            hybrid=True,
            **common,
        )
        fused_capacity = getattr(context, "last_phase1_capacity", None)
        context.last_hybrid_capacity = {"fused": fused_capacity}
        wall = time.perf_counter() - started
        owner_count = int(
            phase1["owner_mask0"].astype(bool, copy=False).sum()
        )
        return phase1, {
            "execution_mode": "fused_stable_top5",
            "wall_seconds": wall,
            "requested_maximum_query_block": query_limit,
            "owner_count": owner_count,
            "fused_stats": asdict(fused_stats),
        }
    primary, primary_stats = context.phase1_precomputed(
        x,
        index,
        counts,
        cuts,
        manifold_dimension,
        selection_mode=0,
        **common,
    )
    primary_capacity = getattr(context, 'last_phase1_capacity', None)
    fallback, fallback_stats = context.phase1_precomputed(
        x,
        index,
        counts,
        cuts,
        manifold_dimension,
        selection_mode=2,
        **common,
    )
    context.last_hybrid_capacity = dict(primary=primary_capacity,
        fallback=getattr(context, 'last_phase1_capacity', None))
    wall = time.perf_counter() - started

    primary_owner = primary["owner_mask0"].astype(bool, copy=False)
    fallback_owner = (~primary_owner) & fallback["owner_mask0"].astype(bool, copy=False)
    owner = primary_owner | fallback_owner

    # Mirror local_phase1_hybrid exactly.  The primary route keeps its
    # aggregated output; only the fallback route consumes raw tangent/center.
    primary["Tan0"][fallback_owner] = fallback["TanRaw0"][fallback_owner]
    primary["Cen0"][fallback_owner] = fallback["CenRaw0"][fallback_owner]
    for name in (
        "Rad0",
        "axial_tan0",
        "num_neigh0",
        "selected_cut0",
        "geom0",
        "stable_h0",
        "neighbor_scaled0",
        "split_scaled0",
        "split_h_a0",
        "split_h_b0",
        "rejection_code0",
        "selected_index0",
    ):
        primary[name][fallback_owner] = fallback[name][fallback_owner]
    primary["Tan0"][~owner] = 0
    primary["Cen0"][~owner] = 0
    primary["owner_mask0"] = np.ascontiguousarray(owner, dtype=np.uint8)

    summary = {
        "wall_seconds": wall,
        "requested_maximum_query_block": query_limit,
        "primary_owner_count": int(primary_owner.sum()),
        "fallback_owner_count": int(fallback_owner.sum()),
        "owner_count": int(owner.sum()),
        "primary_stats": asdict(primary_stats),
        "fallback_stats": asdict(fallback_stats),
    }
    del fallback
    gc.collect()
    return primary, summary
