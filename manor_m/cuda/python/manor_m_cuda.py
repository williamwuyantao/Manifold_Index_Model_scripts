"""Thin Python adapter for the shared Manor-M CUDA C ABI.

This module intentionally has no dependency on MATLAB or PyTorch.  The phase
algorithms will enter through the same C ABI after their current monolithic MEX
implementations have been separated from MATLAB parsing and mxArray ownership.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


_ABI_VERSION = 3
_SUCCESS = 0


class ManorMCudaError(RuntimeError):
    """Raised when the shared CUDA core returns a non-success status."""


class _CDeviceInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("compute_capability_major", ctypes.c_int32),
        ("compute_capability_minor", ctypes.c_int32),
        ("multiprocessor_count", ctypes.c_int32),
        ("warp_size", ctypes.c_int32),
        ("max_threads_per_block", ctypes.c_int32),
        ("cuda_driver_version", ctypes.c_int32),
        ("cuda_runtime_version", ctypes.c_int32),
        ("total_global_memory_bytes", ctypes.c_uint64),
        ("free_global_memory_bytes", ctypes.c_uint64),
        ("name", ctypes.c_char * 256),
    ]


class _CTilePlanRequest(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("staging_slot_count", ctypes.c_int32),
        ("usable_memory_fraction", ctypes.c_double),
        ("reserve_memory_bytes", ctypes.c_uint64),
        ("persistent_memory_bytes", ctypes.c_uint64),
        ("reference_row_count", ctypes.c_int64),
        ("reference_bytes_per_row", ctypes.c_uint64),
        ("query_row_count", ctypes.c_int64),
        ("query_bytes_per_row", ctypes.c_uint64),
        ("scratch_bytes_per_query_row", ctypes.c_uint64),
        ("minimum_reference_tile_rows", ctypes.c_int64),
        ("minimum_query_tile_rows", ctypes.c_int64),
        ("preferred_reference_tile_rows", ctypes.c_int64),
        ("preferred_query_tile_rows", ctypes.c_int64),
    ]


class _CTilePlan(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("execution_mode", ctypes.c_int32),
        ("staging_slot_count", ctypes.c_int32),
        ("device_free_memory_bytes", ctypes.c_uint64),
        ("usable_memory_bytes", ctypes.c_uint64),
        ("full_resident_memory_bytes", ctypes.c_uint64),
        ("estimated_peak_memory_bytes", ctypes.c_uint64),
        ("reference_tile_rows", ctypes.c_int64),
        ("query_tile_rows", ctypes.c_int64),
        ("reference_tile_count", ctypes.c_int64),
        ("query_tile_count", ctypes.c_int64),
    ]


class _CContextOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("stream_count", ctypes.c_int32),
        ("memory_pool_release_threshold_bytes", ctypes.c_uint64),
    ]


class _CTiledStats(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("staging_slot_count", ctypes.c_int32),
        ("element_count", ctypes.c_int64),
        ("tile_elements", ctypes.c_int64),
        ("tile_count", ctypes.c_int64),
        ("peak_device_staging_bytes", ctypes.c_uint64),
        ("peak_pinned_host_staging_bytes", ctypes.c_uint64),
        ("wall_milliseconds", ctypes.c_double),
    ]


class _CResponseKnnOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("scratch_memory_bytes", ctypes.c_uint64),
        ("max_retry_rounds", ctypes.c_int32),
        ("collect_stats", ctypes.c_int32),
    ]


class _CResponseKnnStats(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("response_dimension", ctypes.c_int32),
        ("point_count", ctypes.c_int64),
        ("query_count", ctypes.c_int64),
        ("neighbor_count", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
        ("bootstrap_milliseconds", ctypes.c_double),
        ("threshold_milliseconds", ctypes.c_double),
        ("scan_milliseconds", ctypes.c_double),
        ("retry_milliseconds", ctypes.c_double),
        ("select_milliseconds", ctypes.c_double),
        ("refine_milliseconds", ctypes.c_double),
        ("sort_milliseconds", ctypes.c_double),
        ("emit_milliseconds", ctypes.c_double),
        ("total_milliseconds", ctypes.c_double),
        ("retry_rounds", ctypes.c_int64),
        ("retry_queries", ctypes.c_int64),
        ("owner_misplaced", ctypes.c_uint64),
        ("exact_ties", ctypes.c_uint64),
        ("compact_overflow", ctypes.c_uint64),
        ("mean_candidates", ctypes.c_double),
        ("max_candidates", ctypes.c_double),
    ]


class _CP1CapacityOptions(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("extension_version", ctypes.c_uint32)] + [
        (name, ctypes.c_uint64) for name in ("working_bytes", "pinned_bytes", "retained_limit_bytes",
            "knn_scratch_bytes", "reserve_bytes")] + [
        ("memory_fraction", ctypes.c_float), ("reserve_fraction", ctypes.c_float),
        ("mode", ctypes.c_int32), ("maximum_query_block", ctypes.c_int32)]


class _CP1CapacityPlan(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("extension_version", ctypes.c_uint32),
        ("mode", ctypes.c_int32), ("query_block", ctypes.c_int32)] + [
        (name, ctypes.c_uint64) for name in ("free_bytes", "total_bytes", "usable_bytes", "full_index_bytes",
            "index_device_bytes", "pinned_bytes", "knn_scratch_bytes", "knn_required_bytes",
            "phase1_required_bytes", "global_state_bytes", "estimated_peak_bytes", "projected_retained_bytes")]


def _p1_capacity_options(values):
    for name in ('working_bytes', 'pinned_bytes', 'retained_limit_bytes', 'knn_scratch_bytes', 'reserve_bytes'):
        value = values.get(name)
        if type(value) is not int or not 0 <= value <= 2**64 - 1:
            raise ValueError('P1 capacity byte counts must be nonnegative uint64 integers')
    if (type(values.get('mode')) is not int or values['mode'] not in (0, 1, 2, 3, 4)
            or type(values.get('maximum_query_block')) is not int
            or not 0 <= values['maximum_query_block'] <= 2**31 - 1
            or not 0 < values.get('memory_fraction', 0) < 1
            or not 0 <= values.get('reserve_fraction', -1) < 1
            or not values['retained_limit_bytes'] or not values['knn_scratch_bytes']):
        raise ValueError('Invalid P1 capacity fractions/mode/limits')
    return _CP1CapacityOptions(struct_size=ctypes.sizeof(_CP1CapacityOptions), extension_version=1, **values)


def _p1_capacity_dict(raw):
    if raw.struct_size != ctypes.sizeof(raw) or raw.extension_version != 1:
        raise ManorMCudaError("P1 capacity descriptor mismatch")
    return {name: int(getattr(raw, name)) for name, _ in raw._fields_}


class _CPhase1Options(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("gpu_memory_fraction", ctypes.c_float),
        ("reserve_memory_fraction", ctypes.c_float),
        ("reserve_memory_bytes", ctypes.c_uint64),
        ("heavy_factor", ctypes.c_float),
        ("jitter", ctypes.c_float),
        ("neighbor_scaled_max", ctypes.c_float),
        ("split_scaled_max", ctypes.c_float),
        ("minimum_query_block", ctypes.c_int32),
        ("maximum_query_block", ctypes.c_int32),
        ("stable_window", ctypes.c_int32),
        ("selection_mode", ctypes.c_int32),
        ("split_seed", ctypes.c_uint64),
        ("cut_max_fraction", ctypes.c_double),
        ("index_is_shared_slice", ctypes.c_int32),
        ("h_null_correction", ctypes.c_int32),
        ("collect_profile", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
    ]


_FloatPointer = ctypes.POINTER(ctypes.c_float)
_Int32Pointer = ctypes.POINTER(ctypes.c_int32)
_Uint32Pointer = ctypes.POINTER(ctypes.c_uint32)
_Uint8Pointer = ctypes.POINTER(ctypes.c_uint8)


class _CPhase1Output(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("tangent_point_major", _FloatPointer),
        ("center_point_major", _FloatPointer),
        ("radius", _FloatPointer),
        ("axial_tangent", _FloatPointer),
        ("neighbor_count", _Int32Pointer),
        ("owner_mask", _Uint8Pointer),
        ("stable_covered", _Uint8Pointer),
        ("selected_scale_one_based", _Uint32Pointer),
        ("selected_response_count", _Uint32Pointer),
        ("selected_cut", _Uint32Pointer),
        ("geometry_score", _FloatPointer),
        ("stable_h", _FloatPointer),
        ("neighbor_scaled", _FloatPointer),
        ("split_scaled", _FloatPointer),
        ("split_h_a", _FloatPointer),
        ("split_h_b", _FloatPointer),
        ("rejection_code", _Uint8Pointer),
        ("raw_tangent_point_major", _FloatPointer),
        ("raw_center_point_major", _FloatPointer),
        ("selected_index_one_based", _Uint32Pointer),
    ]


class _CPhase1Stats(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("ambient_dimension", ctypes.c_int32),
        ("manifold_dimension", ctypes.c_int32),
        ("query_block", ctypes.c_int32),
        ("query_block_upper_bound", ctypes.c_int32),
        ("scale_count", ctypes.c_int32),
        ("point_count", ctypes.c_int64),
        ("active_anchor_count", ctypes.c_int64),
        ("stable_covered_count", ctypes.c_uint64),
        ("owner_count", ctypes.c_uint64),
        ("support_threshold", ctypes.c_double),
        ("total_milliseconds", ctypes.c_double),
        ("upload_milliseconds", ctypes.c_double),
        ("distance_milliseconds", ctypes.c_double),
        ("sort_milliseconds", ctypes.c_double),
        ("scale_topk_milliseconds", ctypes.c_double),
        ("scale_select_milliseconds", ctypes.c_double),
        ("scale_stream_milliseconds", ctypes.c_double),
        ("final_extract_milliseconds", ctypes.c_double),
        ("owner_rule_milliseconds", ctypes.c_double),
        ("owner_aggregate_milliseconds", ctypes.c_double),
        ("mask_nonowners_milliseconds", ctypes.c_double),
        ("download_milliseconds", ctypes.c_double),
        ("memory_free_bytes", ctypes.c_uint64),
        ("memory_total_bytes", ctypes.c_uint64),
        ("memory_required_bytes", ctypes.c_uint64),
        ("memory_next_required_bytes", ctypes.c_uint64),
    ]


class _CPhase2Input(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("x_point_major", _FloatPointer),
        ("response_point_major", _FloatPointer),
        ("index_zero_based", _Int32Pointer),
        ("tangent_point_major", _FloatPointer),
        ("center_point_major", _FloatPointer),
        ("owner_mask", _Uint8Pointer),
        ("point_count", ctypes.c_int64),
        ("ambient_dimension", ctypes.c_int32),
        ("manifold_dimension", ctypes.c_int32),
        ("response_dimension", ctypes.c_int32),
        ("neighbor_count", ctypes.c_int32),
        ("base_h", ctypes.c_float),
        ("rho_schedule", _FloatPointer),
        ("rho_schedule_count", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
    ]


class _CPhase2Options(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("lambda_", ctypes.c_float),
        ("jitter", ctypes.c_float),
        ("gpu_memory_fraction", ctypes.c_float),
        ("reserve_memory_fraction", ctypes.c_float),
        ("reserve_memory_bytes", ctypes.c_uint64),
        ("minimum_query_block", ctypes.c_int32),
        ("maximum_query_block", ctypes.c_int32),
        ("regression_mode", ctypes.c_int32),
        ("iteration_count", ctypes.c_int32),
        ("heavy_factor", ctypes.c_float),
        ("temporary_heavy_factor", ctypes.c_float),
        ("h_schedule", _FloatPointer),
        ("h_schedule_count", ctypes.c_int32),
        ("index_is_shared_slice", ctypes.c_int32),
        ("evaluation_uses_owner_mask", ctypes.c_int32),
        ("collect_profile", ctypes.c_int32),
        ("use_async_allocator", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
    ]


class _CPhase2Evaluation(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("test_x_point_major", _FloatPointer),
        ("test_y_point_major", _FloatPointer),
        ("test_tangent_point_major", _FloatPointer),
        ("test_center_point_major", _FloatPointer),
        ("test_point_count", ctypes.c_int64),
        ("validation_x_point_major", _FloatPointer),
        ("validation_y_point_major", _FloatPointer),
        ("validation_point_count", ctypes.c_int64),
        ("rho_scales", _FloatPointer),
        ("rho_scale_count", ctypes.c_int32),
        ("h_scales", _FloatPointer),
        ("h_scale_count", ctypes.c_int32),
        ("evaluated_iterations", _Int32Pointer),
        ("evaluated_iteration_count", ctypes.c_int32),
        ("sum_sparse_threshold", ctypes.c_float),
        ("count_sparse_threshold", ctypes.c_int32),
        ("fallback_exponent_scale", ctypes.c_float),
        ("epsilon_weight", ctypes.c_float),
        ("warps_per_block", ctypes.c_int32),
        ("center_rho_scale", ctypes.c_float),
        ("center_h_scale", ctypes.c_float),
        ("reserved", ctypes.c_int32),
        ("validation_tangent_point_major", _FloatPointer),
    ]


class _CPhase2Output(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("tangent_point_major", _FloatPointer),
        ("center_point_major", _FloatPointer),
        ("neighbor_count", _Int32Pointer),
        ("heavy_mask", _Uint8Pointer),
        ("keep_mask", _Uint8Pointer),
        ("solver_failed", _Uint8Pointer),
        ("neighbor_count_history", _Int32Pointer),
        ("heavy_mask_history", _Uint8Pointer),
        ("keep_mask_history", _Uint8Pointer),
        ("gradient_debug_point_major", _FloatPointer),
        ("eval_mse", _FloatPointer),
        ("eval_mse_coordinate", _FloatPointer),
        ("eval_dev_center", _FloatPointer),
        ("eval_dev_center_tangent", _FloatPointer),
        ("eval_dev_tangent", _FloatPointer),
        ("eval_mse_k0", _FloatPointer),
        ("eval_mse_coordinate_k0", _FloatPointer),
        ("eval_dev_center_k0", _FloatPointer),
        ("eval_dev_center_tangent_k0", _FloatPointer),
        ("eval_dev_tangent_k0", _FloatPointer),
        ("validation_mse", _FloatPointer),
        ("validation_mse_coordinate", _FloatPointer),
        ("validation_mse_k0", _FloatPointer),
        ("validation_mse_coordinate_k0", _FloatPointer),
        ("validation_dev_tangent", _FloatPointer),
        ("validation_dev_tangent_k0", _FloatPointer),
    ]


class _CPhase2Stats(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("ambient_dimension", ctypes.c_int32),
        ("manifold_dimension", ctypes.c_int32),
        ("response_dimension", ctypes.c_int32),
        ("query_block", ctypes.c_int32),
        ("iteration_count", ctypes.c_int32),
        ("valid_iterations", ctypes.c_int32),
        ("stopped_no_heavy", ctypes.c_int32),
        ("evaluation_iterations_executed", ctypes.c_int32),
        ("point_count", ctypes.c_int64),
        ("owner_count", ctypes.c_int64),
        ("validation_point_count", ctypes.c_int64),
        ("memory_required_bytes", ctypes.c_uint64),
        ("memory_usable_bytes", ctypes.c_uint64),
        ("memory_index_bytes", ctypes.c_uint64),
        ("history_bytes", ctypes.c_uint64),
        ("total_milliseconds", ctypes.c_double),
        ("host_prepare_milliseconds", ctypes.c_double),
        ("device_loop_milliseconds", ctypes.c_double),
        ("output_pack_milliseconds", ctypes.c_double),
        ("sparse_pair_count", ctypes.c_int64),
        ("response_evaluated", ctypes.c_int32),
        ("tangent_evaluated", ctypes.c_int32),
        ("center_evaluated", ctypes.c_int32),
        ("center_tangent_evaluated", ctypes.c_int32),
        ("timer_upload_init_milliseconds", ctypes.c_double),
        ("timer_eval_stats_milliseconds", ctypes.c_double),
        ("timer_eval_main_milliseconds", ctypes.c_double),
        ("timer_eval_pair_milliseconds", ctypes.c_double),
        ("timer_valid_main_milliseconds", ctypes.c_double),
        ("timer_valid_pair_milliseconds", ctypes.c_double),
        ("timer_iter_setup_milliseconds", ctypes.c_double),
        ("timer_regression_milliseconds", ctypes.c_double),
        ("timer_mask_chain_milliseconds", ctypes.c_double),
        ("timer_aggregation_accumulate_milliseconds", ctypes.c_double),
        ("timer_aggregation_finalize_milliseconds", ctypes.c_double),
        ("timer_snapshots_milliseconds", ctypes.c_double),
        ("timer_eval_store_milliseconds", ctypes.c_double),
        ("timer_valid_store_milliseconds", ctypes.c_double),
        ("host_cuda_allocate_milliseconds", ctypes.c_double),
        ("host_cuda_free_milliseconds", ctypes.c_double),
        ("host_cuda_event_destroy_milliseconds", ctypes.c_double),
    ]


@dataclass(frozen=True)
class DeviceInfo:
    device_ordinal: int
    name: str
    compute_capability: str
    multiprocessor_count: int
    warp_size: int
    max_threads_per_block: int
    cuda_driver_version: int
    cuda_runtime_version: int
    total_global_memory_bytes: int
    free_global_memory_bytes: int


@dataclass(frozen=True)
class TilePlan:
    execution_mode: str
    staging_slot_count: int
    device_free_memory_bytes: int
    usable_memory_bytes: int
    full_resident_memory_bytes: int
    estimated_peak_memory_bytes: int
    reference_tile_rows: int
    query_tile_rows: int
    reference_tile_count: int
    query_tile_count: int


@dataclass(frozen=True)
class TiledStats:
    device_ordinal: int
    staging_slot_count: int
    element_count: int
    tile_elements: int
    tile_count: int
    peak_device_staging_bytes: int
    peak_pinned_host_staging_bytes: int
    wall_milliseconds: float


@dataclass(frozen=True)
class ResponseKnnStats:
    device_ordinal: int
    response_dimension: int
    point_count: int
    query_count: int
    neighbor_count: int
    bootstrap_milliseconds: float
    threshold_milliseconds: float
    scan_milliseconds: float
    retry_milliseconds: float
    select_milliseconds: float
    refine_milliseconds: float
    sort_milliseconds: float
    emit_milliseconds: float
    total_milliseconds: float
    retry_rounds: int
    retry_queries: int
    owner_misplaced: int
    exact_ties: int
    compact_overflow: int
    mean_candidates: float
    max_candidates: float


def _response_knn_stats(raw):
    if raw.struct_size != ctypes.sizeof(_CResponseKnnStats):
        raise ManorMCudaError(
            "Response-kNN stats struct size mismatch: "
            f"core={raw.struct_size}, Python={ctypes.sizeof(_CResponseKnnStats)}"
        )
    return ResponseKnnStats(
        device_ordinal=int(raw.device_ordinal),
        response_dimension=int(raw.response_dimension),
        point_count=int(raw.point_count),
        query_count=int(raw.query_count),
        neighbor_count=int(raw.neighbor_count),
        bootstrap_milliseconds=float(raw.bootstrap_milliseconds),
        threshold_milliseconds=float(raw.threshold_milliseconds),
        scan_milliseconds=float(raw.scan_milliseconds),
        retry_milliseconds=float(raw.retry_milliseconds),
        select_milliseconds=float(raw.select_milliseconds),
        refine_milliseconds=float(raw.refine_milliseconds),
        sort_milliseconds=float(raw.sort_milliseconds),
        emit_milliseconds=float(raw.emit_milliseconds),
        total_milliseconds=float(raw.total_milliseconds),
        retry_rounds=int(raw.retry_rounds),
        retry_queries=int(raw.retry_queries),
        owner_misplaced=int(raw.owner_misplaced),
        exact_ties=int(raw.exact_ties),
        compact_overflow=int(raw.compact_overflow),
        mean_candidates=float(raw.mean_candidates),
        max_candidates=float(raw.max_candidates),
    )


@dataclass(frozen=True)
class Phase1Stats:
    device_ordinal: int
    ambient_dimension: int
    manifold_dimension: int
    query_block: int
    query_block_upper_bound: int
    scale_count: int
    point_count: int
    active_anchor_count: int
    stable_covered_count: int
    owner_count: int
    support_threshold: float
    total_milliseconds: float
    upload_milliseconds: float
    distance_milliseconds: float
    sort_milliseconds: float
    scale_topk_milliseconds: float
    scale_select_milliseconds: float
    scale_stream_milliseconds: float
    final_extract_milliseconds: float
    owner_rule_milliseconds: float
    owner_aggregate_milliseconds: float
    mask_nonowners_milliseconds: float
    download_milliseconds: float
    memory_free_bytes: int
    memory_total_bytes: int
    memory_required_bytes: int
    memory_next_required_bytes: int


@dataclass(frozen=True)
class Phase2Stats:
    device_ordinal: int
    ambient_dimension: int
    manifold_dimension: int
    response_dimension: int
    query_block: int
    iteration_count: int
    valid_iterations: int
    stopped_no_heavy: int
    evaluation_iterations_executed: int
    point_count: int
    owner_count: int
    validation_point_count: int
    memory_required_bytes: int
    memory_usable_bytes: int
    memory_index_bytes: int
    history_bytes: int
    total_milliseconds: float
    host_prepare_milliseconds: float
    device_loop_milliseconds: float
    output_pack_milliseconds: float
    sparse_pair_count: int
    response_evaluated: int
    tangent_evaluated: int
    center_evaluated: int
    center_tangent_evaluated: int
    timer_upload_init_milliseconds: float
    timer_eval_stats_milliseconds: float
    timer_eval_main_milliseconds: float
    timer_eval_pair_milliseconds: float
    timer_valid_main_milliseconds: float
    timer_valid_pair_milliseconds: float
    timer_iter_setup_milliseconds: float
    timer_regression_milliseconds: float
    timer_mask_chain_milliseconds: float
    timer_aggregation_accumulate_milliseconds: float
    timer_aggregation_finalize_milliseconds: float
    timer_snapshots_milliseconds: float
    timer_eval_store_milliseconds: float
    timer_valid_store_milliseconds: float
    host_cuda_allocate_milliseconds: float
    host_cuda_free_milliseconds: float
    host_cuda_event_destroy_milliseconds: float


def _candidate_libraries() -> Iterable[Path | str]:
    override = os.environ.get("MANOR_M_CUDA_LIBRARY")
    if override:
        yield Path(override).expanduser()

    cuda_root = Path(__file__).resolve().parents[1]
    names = (
        "libmanor_m_cuda.so",
        "libmanor_m_cuda.dylib",
        "manor_m_cuda.dll",
    )
    for directory in (
        cuda_root / "build",
        cuda_root / "build" / "lib",
        cuda_root / "lib",
        Path(__file__).resolve().parent,
    ):
        for name in names:
            yield directory / name

    discovered = ctypes.util.find_library("manor_m_cuda")
    if discovered:
        yield discovered


def _load_library(explicit_path: str | os.PathLike[str] | None = None) -> ctypes.CDLL:
    candidates: Iterable[Path | str]
    if explicit_path is None:
        candidates = _candidate_libraries()
    else:
        candidates = (Path(explicit_path).expanduser(),)

    attempts: list[str] = []
    for candidate in candidates:
        candidate_text = os.fspath(candidate)
        if isinstance(candidate, Path) and not candidate.is_file():
            attempts.append(f"{candidate_text} (missing)")
            continue
        try:
            library = ctypes.CDLL(candidate_text)
        except OSError as error:
            attempts.append(f"{candidate_text} ({error})")
            continue
        _declare_signatures(library)
        abi_version = int(library.manor_m_cuda_abi_version())
        if abi_version != _ABI_VERSION:
            raise ManorMCudaError(
                f"CUDA core ABI {abi_version} does not match Python adapter ABI "
                f"{_ABI_VERSION}: {candidate_text}"
            )
        return library

    details = "\n  ".join(attempts)
    raise ManorMCudaError(
        "Could not load libmanor_m_cuda. Build manor_m/cuda first or set "
        f"MANOR_M_CUDA_LIBRARY. Attempts:\n  {details}"
    )


def _declare_signatures(library: ctypes.CDLL) -> None:
    library.manor_m_cuda_abi_version.argtypes = []
    library.manor_m_cuda_abi_version.restype = ctypes.c_uint32
    library.manor_m_cuda_version.argtypes = []
    library.manor_m_cuda_version.restype = ctypes.c_char_p
    library.manor_m_cuda_last_error.argtypes = []
    library.manor_m_cuda_last_error.restype = ctypes.c_char_p
    library.manor_m_cuda_status_string.argtypes = [ctypes.c_int]
    library.manor_m_cuda_status_string.restype = ctypes.c_char_p
    library.manor_m_cuda_device_count.argtypes = [ctypes.POINTER(ctypes.c_int32)]
    library.manor_m_cuda_device_count.restype = ctypes.c_int
    library.manor_m_cuda_get_device_info.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(_CDeviceInfo),
    ]
    library.manor_m_cuda_get_device_info.restype = ctypes.c_int
    library.manor_m_cuda_plan_tiles.argtypes = [
        ctypes.POINTER(_CTilePlanRequest),
        ctypes.POINTER(_CTilePlan),
    ]
    library.manor_m_cuda_plan_tiles.restype = ctypes.c_int
    library.manor_m_cuda_context_create.argtypes = [
        ctypes.POINTER(_CContextOptions),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.manor_m_cuda_context_create.restype = ctypes.c_int
    library.manor_m_cuda_context_destroy.argtypes = [ctypes.c_void_p]
    library.manor_m_cuda_context_destroy.restype = ctypes.c_int
    library.manor_m_cuda_smoke_axpy.argtypes = [
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
    ]
    library.manor_m_cuda_smoke_axpy.restype = ctypes.c_int
    library.manor_m_cuda_smoke_tiled_axpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(_CTiledStats),
    ]
    library.manor_m_cuda_smoke_tiled_axpy.restype = ctypes.c_int
    library.manor_m_cuda_response_knn.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(_CResponseKnnOptions),
        ctypes.POINTER(_CResponseKnnStats),
    ]
    library.manor_m_cuda_response_knn.restype = ctypes.c_int
    if hasattr(library, "manor_m_cuda_response_knn_memory"):
        library.manor_m_cuda_response_knn_memory.argtypes = [
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        library.manor_m_cuda_response_knn_memory.restype = ctypes.c_int
    if hasattr(library, 'manor_m_cuda_response_knn_host'):
        library.manor_m_cuda_response_knn_host.argtypes = library.manor_m_cuda_response_knn.argtypes
        library.manor_m_cuda_response_knn_host.restype = ctypes.c_int
    if hasattr(library, 'manor_m_cuda_response_knn_host_range'):
        library.manor_m_cuda_response_knn_host_range.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(_CResponseKnnOptions),
            ctypes.POINTER(_CResponseKnnStats),
        ]
        library.manor_m_cuda_response_knn_host_range.restype = ctypes.c_int
    if hasattr(library, 'manor_m_cuda_response_knn_compact_owners_host'):
        library.manor_m_cuda_response_knn_compact_owners_host.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_int32,
            _Uint8Pointer,
            ctypes.c_int64,
            _Int32Pointer,
            ctypes.POINTER(_CResponseKnnOptions),
            ctypes.POINTER(_CResponseKnnStats),
        ]
        library.manor_m_cuda_response_knn_compact_owners_host.restype = ctypes.c_int
    library.manor_m_cuda_phase1_precomputed.argtypes = [
        ctypes.c_void_p,
        _FloatPointer,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_int32,
        _Int32Pointer,
        ctypes.c_int64,
        ctypes.c_int32,
        _Int32Pointer,
        ctypes.c_int32,
        _Int32Pointer,
        ctypes.c_int32,
        ctypes.POINTER(_CPhase1Options),
        ctypes.POINTER(_CPhase1Output),
        ctypes.POINTER(_CPhase1Stats),
    ]
    library.manor_m_cuda_phase1_precomputed.restype = ctypes.c_int
    if hasattr(library, "manor_m_cuda_phase1_hybrid_precomputed"):
        library.manor_m_cuda_phase1_hybrid_precomputed.argtypes = (
            library.manor_m_cuda_phase1_precomputed.argtypes
        )
        library.manor_m_cuda_phase1_hybrid_precomputed.restype = ctypes.c_int
    if hasattr(library, 'manor_m_cuda_plan_phase1'):
        library.manor_m_cuda_plan_phase1.argtypes = [ctypes.c_void_p, _FloatPointer, _FloatPointer,
            ctypes.c_int64] + [ctypes.c_int32] * 7 + [ctypes.POINTER(_CP1CapacityOptions),
                                                    ctypes.POINTER(_CP1CapacityPlan)]
        library.manor_m_cuda_plan_phase1.restype = ctypes.c_int
        if hasattr(library, "manor_m_cuda_plan_phase1_hybrid"):
            library.manor_m_cuda_plan_phase1_hybrid.argtypes = (
                library.manor_m_cuda_plan_phase1.argtypes
            )
            library.manor_m_cuda_plan_phase1_hybrid.restype = ctypes.c_int
        library.manor_m_cuda_phase1_bounded.argtypes = library.manor_m_cuda_phase1_precomputed.argtypes + [
            ctypes.POINTER(_CP1CapacityOptions), ctypes.POINTER(_CP1CapacityPlan)]
        library.manor_m_cuda_phase1_bounded.restype = ctypes.c_int
        if hasattr(library, "manor_m_cuda_phase1_hybrid_bounded"):
            library.manor_m_cuda_phase1_hybrid_bounded.argtypes = (
                library.manor_m_cuda_phase1_bounded.argtypes
            )
            library.manor_m_cuda_phase1_hybrid_bounded.restype = ctypes.c_int
        generated_phase1_arguments = [
            ctypes.c_void_p, _FloatPointer, _FloatPointer, ctypes.c_int64,
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            _Int32Pointer, ctypes.c_int32, _Int32Pointer, ctypes.c_int32,
            ctypes.POINTER(_CPhase1Options), ctypes.POINTER(_CPhase1Output),
            ctypes.POINTER(_CPhase1Stats), ctypes.POINTER(_CP1CapacityOptions),
            ctypes.POINTER(_CP1CapacityPlan),
        ]
        if hasattr(library, "manor_m_cuda_phase1_response_bounded"):
            library.manor_m_cuda_phase1_response_bounded.argtypes = (
                generated_phase1_arguments
            )
            library.manor_m_cuda_phase1_response_bounded.restype = ctypes.c_int
        if hasattr(library, "manor_m_cuda_phase1_response_hybrid_bounded"):
            library.manor_m_cuda_phase1_response_hybrid_bounded.argtypes = (
                generated_phase1_arguments
            )
            library.manor_m_cuda_phase1_response_hybrid_bounded.restype = ctypes.c_int
    library.manor_m_cuda_phase2.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_CPhase2Input),
        ctypes.POINTER(_CPhase2Options),
        ctypes.POINTER(_CPhase2Evaluation),
        ctypes.POINTER(_CPhase2Output),
        ctypes.POINTER(_CPhase2Stats),
    ]
    library.manor_m_cuda_phase2.restype = ctypes.c_int
    if hasattr(library, "manor_m_cuda_phase2_tiled"):
        library.manor_m_cuda_phase2_tiled.argtypes = (
            library.manor_m_cuda_phase2.argtypes + [ctypes.c_int32])
        library.manor_m_cuda_phase2_tiled.restype = ctypes.c_int
    if hasattr(library, "manor_m_cuda_phase2_index_tiled"):
        library.manor_m_cuda_phase2_index_tiled.argtypes = (
            library.manor_m_cuda_phase2.argtypes + [ctypes.c_int32,
                ctypes.c_int32, ctypes.POINTER(_CIndexStreamStats)])
        library.manor_m_cuda_phase2_index_tiled.restype = ctypes.c_int
    if hasattr(library, "manor_m_cuda_phase2_compact_index_tiled"):
        library.manor_m_cuda_phase2_compact_index_tiled.argtypes = (
            library.manor_m_cuda_phase2.argtypes + [ctypes.c_int32,
                ctypes.c_int32, ctypes.POINTER(_CIndexStreamStats)])
        library.manor_m_cuda_phase2_compact_index_tiled.restype = ctypes.c_int
    if hasattr(library, "manor_m_cuda_phase2_response_tiled"):
        library.manor_m_cuda_phase2_response_tiled.argtypes = (
            library.manor_m_cuda_phase2.argtypes + [ctypes.c_int32,
                ctypes.c_uint64, ctypes.POINTER(_CGeneratedIndexStats)])
        library.manor_m_cuda_phase2_response_tiled.restype = ctypes.c_int


class _CIndexStreamStats(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("extension_version", ctypes.c_uint32)] + [
        (name, ctypes.c_uint64) for name in (
            "enabled", "logical_index_bytes", "device_index_bytes", "pinned_host_bytes",
            "tile_owners", "tiles_per_pass", "buffer_count", "upload_calls",
            "host_to_device_bytes")] + [(name, ctypes.c_double) for name in (
            "host_fill_milliseconds", "host_reuse_wait_milliseconds", "copy_milliseconds")]


class _CGeneratedIndexStats(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("extension_version", ctypes.c_uint32),
    ] + [(name, ctypes.c_uint64) for name in (
        "enabled", "logical_full_index_bytes", "logical_compact_index_bytes",
        "materialized_host_index_bytes", "materialized_device_index_bytes",
        "scratch_memory_bytes", "response_soa_bytes", "generation_passes",
        "query_tiles", "owner_rows", "peak_query_rows", "peak_owner_rows",
        "peak_device_index_tile_bytes", "owner_row_map_h2d_bytes",
    )] + [(name, ctypes.c_double) for name in (
        "knn_reported_milliseconds", "generation_wall_milliseconds",
    )]


class _CResidentStats(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("extension_version", ctypes.c_uint32)] + [
        (name, ctypes.c_uint64) for name in (
            "enabled", "buffer_count", "retained_bytes", "host_to_device_bytes",
            "generated_device_bytes", "cache_hits", "skipped_host_to_device_bytes",
            "initial_state_device_copy_bytes")]


class Context:
    """Long-lived, device-affine CUDA execution context."""

    def __init__(self, core: "ManorMCuda", pointer: ctypes.c_void_p):
        self._core = core
        self._pointer = pointer

    def close(self) -> None:
        if not self._pointer:
            return
        pointer = self._pointer
        self._pointer = ctypes.c_void_p()
        self._core._check(
            self._core._library.manor_m_cuda_context_destroy(pointer),
            "context_destroy",
        )

    def __enter__(self) -> "Context":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _resident_api(self):
        if not self._pointer:
            raise ManorMCudaError("CUDA context is already closed")
        lib = self._core._library
        if not hasattr(lib, "manor_m_cuda_resident_begin"):
            raise ManorMCudaError("This core has no resident-input extension; rebuild the isolated candidate.")
        lib.manor_m_cuda_resident_begin.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.manor_m_cuda_resident_put.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint64]
        if hasattr(lib, "manor_m_cuda_resident_validate_index"):
            lib.manor_m_cuda_resident_validate_index.argtypes = [
                ctypes.c_void_p, _Int32Pointer, ctypes.c_int64, ctypes.c_int32]
        if hasattr(lib, "manor_m_cuda_resident_validate_compact_index"):
            lib.manor_m_cuda_resident_validate_compact_index.argtypes = [
                ctypes.c_void_p, _Int32Pointer, ctypes.c_int64,
                ctypes.c_int32, ctypes.c_int64]
        lib.manor_m_cuda_resident_clear.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.manor_m_cuda_resident_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CResidentStats)]
        if hasattr(lib, "manor_m_cuda_resident_prepare_owner_index"):
            lib.manor_m_cuda_resident_prepare_owner_index.argtypes = [
                ctypes.c_void_p, _Int32Pointer, _Uint8Pointer, ctypes.c_int64, ctypes.c_int32]
        return lib

    def _resident_begin(self, maximum_bytes):
        self._core._check(self._resident_api().manor_m_cuda_resident_begin(
            self._pointer, int(maximum_bytes)), "resident_begin")

    def _resident_put(self, name, array):
        if array.flags.writeable or not array.flags.c_contiguous:
            raise ValueError("Resident host inputs must be immutable contiguous snapshots")
        self._core._check(self._resident_api().manor_m_cuda_resident_put(
            self._pointer, name.encode("utf-8"), ctypes.c_void_p(array.ctypes.data),
            array.nbytes), "resident_put")

    def _resident_validate_index(self, index):
        lib = self._resident_api()
        if not hasattr(lib, "manor_m_cuda_resident_validate_index"):
            return False
        if (index.flags.writeable or not index.flags.c_contiguous or
                index.dtype.name != "int32" or index.ndim != 2):
            raise ValueError("Validated resident index must be immutable contiguous int32 [N,k]")
        self._core._check(lib.manor_m_cuda_resident_validate_index(
            self._pointer, index.ctypes.data_as(_Int32Pointer),
            index.shape[0], index.shape[1]), "resident_validate_index")
        return True

    def _resident_validate_compact_index(self, index, point_count):
        lib = self._resident_api()
        if not hasattr(lib, "manor_m_cuda_resident_validate_compact_index"):
            return False
        if (index.flags.writeable or not index.flags.c_contiguous or
                index.dtype.name != "int32" or index.ndim != 2):
            raise ValueError(
                "Validated compact index must be immutable contiguous int32 [M,k]")
        if not 1 <= index.shape[0] <= int(point_count):
            raise ValueError("Compact index row count must lie in [1,N]")
        self._core._check(lib.manor_m_cuda_resident_validate_compact_index(
            self._pointer, index.ctypes.data_as(_Int32Pointer),
            index.shape[0], index.shape[1], int(point_count)),
            "resident_validate_compact_index")
        return True

    def _resident_clear(self, group=""):
        self._core._check(self._resident_api().manor_m_cuda_resident_clear(
            self._pointer, group.encode("utf-8")), "resident_clear")

    def _resident_prepare_owner_index(self, index, owner):
        lib = self._resident_api()
        self._core._check(lib.manor_m_cuda_resident_prepare_owner_index(self._pointer,
            index.ctypes.data_as(_Int32Pointer), owner.ctypes.data_as(_Uint8Pointer),
            index.shape[0], index.shape[1]), "resident_prepare_owner_index")

    def resident_stats(self):
        """Input-transfer counters, not a peak-memory or all-traffic estimate."""
        raw = _CResidentStats()
        self._core._check(self._resident_api().manor_m_cuda_resident_stats(
            self._pointer, ctypes.byref(raw)), "resident_stats")
        if raw.struct_size != ctypes.sizeof(raw) or raw.extension_version != 1:
            raise ManorMCudaError("Unsupported resident-input stats layout")
        return {key: int(getattr(raw, key)) for key, _ in raw._fields_}

    def smoke_tiled_axpy(
        self,
        x,
        y,
        alpha: float = 2.0,
        tile_elements: int = 0,
    ):
        """Run the double-buffered, host-staged out-of-core smoke."""
        try:
            import numpy as np
        except ImportError as error:
            raise ManorMCudaError("smoke_tiled_axpy requires NumPy") from error

        if not self._pointer:
            raise ManorMCudaError("CUDA context is already closed")
        x_array = np.ascontiguousarray(x, dtype=np.float32)
        y_array = np.ascontiguousarray(y, dtype=np.float32)
        if x_array.shape != y_array.shape:
            raise ValueError(
                f"x and y must have the same shape, got {x_array.shape} and "
                f"{y_array.shape}"
            )
        output = np.empty_like(x_array)
        raw_stats = _CTiledStats()
        float_pointer = ctypes.POINTER(ctypes.c_float)
        self._core._check(
            self._core._library.manor_m_cuda_smoke_tiled_axpy(
                self._pointer,
                float(alpha),
                x_array.ctypes.data_as(float_pointer),
                y_array.ctypes.data_as(float_pointer),
                output.ctypes.data_as(float_pointer),
                int(x_array.size),
                int(tile_elements),
                ctypes.byref(raw_stats),
            ),
            "smoke_tiled_axpy",
        )
        if raw_stats.struct_size != ctypes.sizeof(_CTiledStats):
            raise ManorMCudaError(
                "Tiled-stats struct size mismatch: "
                f"core={raw_stats.struct_size}, "
                f"Python={ctypes.sizeof(_CTiledStats)}"
            )
        stats = TiledStats(
            device_ordinal=int(raw_stats.device_ordinal),
            staging_slot_count=int(raw_stats.staging_slot_count),
            element_count=int(raw_stats.element_count),
            tile_elements=int(raw_stats.tile_elements),
            tile_count=int(raw_stats.tile_count),
            peak_device_staging_bytes=int(raw_stats.peak_device_staging_bytes),
            peak_pinned_host_staging_bytes=int(
                raw_stats.peak_pinned_host_staging_bytes
            ),
            wall_milliseconds=float(raw_stats.wall_milliseconds),
        )
        return output, stats

    def response_knn(
        self,
        response,
        neighbor_count: int,
        *,
        query_first: int = 0,
        query_count: int | None = None,
        scratch_memory_bytes: int = 7 * 1024**3,
        max_retry_rounds: int = 12,
        host_output: bool = False,
        output=None,
    ):
        """Build the exact zero-based Phase-1 response-neighbor index.

        ``response`` is a C-contiguous ``[N, DY]`` float32 array.  The returned
        ``[query_count, neighbor_count]`` int32 array uses the MEX's MATLAB
        ``[neighbor_count, query_count]`` column-major ``response_index`` layout.
        ``query_first`` selects a contiguous global-anchor range without
        materializing preceding rows; nonzero ranges require host output.
        The capacity-enabled core and matching new MEX source resolve exact
        FP64 ties self-first, then by global ID; older binaries did not define
        tie order when the sort input came from unordered atomic appends.
        """
        try:
            import numpy as np
        except ImportError as error:
            raise ManorMCudaError("response_knn requires NumPy") from error

        if not self._pointer:
            raise ManorMCudaError("CUDA context is already closed")
        response_array = np.ascontiguousarray(response, dtype=np.float32)
        if response_array.ndim != 2:
            raise ValueError(
                f"response must have shape [N, DY], got {response_array.shape}"
            )
        point_count, response_dimension = response_array.shape
        query_first = int(query_first)
        if not 0 <= query_first < point_count:
            raise ValueError("query_first must lie in [0, N)")
        if query_count is None:
            query_count = point_count - query_first
        if not 1 <= int(query_count) <= point_count - query_first:
            raise ValueError("query range must lie in [0, N)")
        if not 1 <= int(neighbor_count) <= point_count:
            raise ValueError("neighbor_count must lie in [1, N]")

        shape = (int(query_count), int(neighbor_count))
        if output is None:
            output = np.empty(shape, dtype=np.int32)
        elif (not isinstance(output, np.ndarray) or output.shape != shape or output.dtype != np.int32
                or not output.flags.c_contiguous or not output.flags.writeable):
            raise ValueError('kNN output must be writable C-contiguous int32 with the exact shape')
        if query_first:
            if not host_output:
                raise ValueError("nonzero query_first requires bounded host output")
            symbol = 'manor_m_cuda_response_knn_host_range'
        else:
            symbol = 'manor_m_cuda_response_knn_host' if host_output else 'manor_m_cuda_response_knn'
        if not hasattr(self._core._library, symbol):
            raise ManorMCudaError('core does not provide the requested bounded kNN entry point')
        raw_options = _CResponseKnnOptions(
            struct_size=ctypes.sizeof(_CResponseKnnOptions),
            abi_version=_ABI_VERSION,
            scratch_memory_bytes=int(scratch_memory_bytes),
            max_retry_rounds=int(max_retry_rounds),
            collect_stats=1,
        )
        raw_stats = _CResponseKnnStats()
        common_arguments = (
            self._pointer,
            response_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(point_count),
            int(response_dimension),
        )
        if query_first:
            call_arguments = common_arguments + (int(query_first), int(query_count))
        else:
            call_arguments = common_arguments + (int(query_count),)
        call_arguments += (
            int(neighbor_count),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.byref(raw_options),
            ctypes.byref(raw_stats),
        )
        self._core._check(
            getattr(self._core._library, symbol)(*call_arguments),
            "response_knn",
        )
        return output, _response_knn_stats(raw_stats)

    def response_knn_compact_owners(
        self,
        response,
        neighbor_count: int,
        owner_mask,
        *,
        scratch_memory_bytes: int = 7 * 1024**3,
        max_retry_rounds: int = 12,
        output=None,
    ):
        """Build exact kNN rows only for owners, in increasing owner order.

        The native implementation derives its query chunk directly from the
        supplied scratch ceiling and includes its gather workspace in that
        arithmetic.  It never materializes a complete host or device ``N*k``
        index and performs no production-time throughput search.
        """
        try:
            import numpy as np
        except ImportError as error:
            raise ManorMCudaError(
                "response_knn_compact_owners requires NumPy") from error
        if not self._pointer:
            raise ManorMCudaError("CUDA context is already closed")
        symbol = getattr(
            self._core._library,
            "manor_m_cuda_response_knn_compact_owners_host",
            None,
        )
        if symbol is None:
            raise ManorMCudaError(
                "core does not provide compact-owner exact kNN")
        response_array = np.ascontiguousarray(response, dtype=np.float32)
        owner_array = np.ascontiguousarray(owner_mask, dtype=np.uint8)
        if response_array.ndim != 2 or not len(response_array):
            raise ValueError("response must have nonempty shape [N,DY]")
        point_count, response_dimension = response_array.shape
        if owner_array.shape != (point_count,):
            raise ValueError("owner_mask must have shape [N]")
        owner_count = int(np.count_nonzero(owner_array))
        if owner_count < 1:
            raise ValueError("compact-owner kNN requires at least one owner")
        if not 1 <= int(neighbor_count) <= point_count:
            raise ValueError("neighbor_count must lie in [1,N]")
        shape = (owner_count, int(neighbor_count))
        if output is None:
            output = np.empty(shape, dtype=np.int32)
        elif (not isinstance(output, np.ndarray) or output.shape != shape or
                output.dtype != np.int32 or not output.flags.c_contiguous or
                not output.flags.writeable):
            raise ValueError(
                "compact kNN output must be writable C-contiguous int32 [M,k]")
        raw_options = _CResponseKnnOptions(
            struct_size=ctypes.sizeof(_CResponseKnnOptions),
            abi_version=_ABI_VERSION,
            scratch_memory_bytes=int(scratch_memory_bytes),
            max_retry_rounds=int(max_retry_rounds),
            collect_stats=1,
        )
        raw_stats = _CResponseKnnStats()
        self._core._check(symbol(
            self._pointer,
            response_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(point_count), int(response_dimension), int(neighbor_count),
            owner_array.ctypes.data_as(_Uint8Pointer), int(owner_count),
            output.ctypes.data_as(_Int32Pointer), ctypes.byref(raw_options),
            ctypes.byref(raw_stats)), "response_knn_compact_owners")
        return output, _response_knn_stats(raw_stats)

    def plan_phase1(
        self, x, response, k, counts, cuts, mm, *, capacity_options,
        hybrid: bool = False,
    ):
        import numpy as np
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.ascontiguousarray(response, dtype=np.float32)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
            raise ValueError('capacity planning requires matching X[N,D]/Y[N,MY]')
        counts_array = np.ascontiguousarray(counts, dtype=np.int32)
        cuts_array = np.ascontiguousarray(cuts, dtype=np.int32)
        if counts_array.ndim != 1 or cuts_array.ndim != 2:
            raise ValueError("capacity planning requires counts[S] and cuts[S,R]")
        if cuts_array.shape[0] != counts_array.size or cuts_array.shape[1] < 1:
            raise ValueError("capacity planning cuts must have shape [S,R]")
        symbol = self._core._library.manor_m_cuda_plan_phase1
        label = "plan_phase1"
        if hybrid:
            if counts_array.size != 1:
                raise ValueError("fused hybrid Phase1 planning requires one scale")
            if not hasattr(
                self._core._library, "manor_m_cuda_plan_phase1_hybrid"
            ):
                raise ManorMCudaError(
                    "core does not provide fused hybrid Phase1 capacity planning"
                )
            symbol = self._core._library.manor_m_cuda_plan_phase1_hybrid
            label = "plan_phase1_hybrid"
        options = _p1_capacity_options(capacity_options)
        raw = _CP1CapacityPlan()
        self._core._check(symbol(
            self._pointer, x.ctypes.data_as(_FloatPointer), y.ctypes.data_as(_FloatPointer),
            len(x), x.shape[1], y.shape[1], int(mm), int(k), counts_array.size,
            cuts_array.shape[1], int(np.max(cuts_array)), ctypes.byref(options),
            ctypes.byref(raw)), label)
        return _p1_capacity_dict(raw)

    def phase1_precomputed(
        self,
        x,
        index_zero_based,
        response_counts,
        cuts_scale_major,
        manifold_dimension: int,
        *,
        response=None,
        active_anchor_count: int | None = None,
        index_is_shared_slice: bool = False,
        gpu_memory_fraction: float = 0.92,
        reserve_memory_fraction: float = 0.08,
        reserve_memory_bytes: int = 6 * 1024**3,
        heavy_factor: float = 0.10,
        jitter: float = 1.0e-6,
        minimum_query_block: int = 128,
        maximum_query_block: int = 2**31 - 1,
        stable_window: int = 7,
        neighbor_scaled_max: float = 1.0,
        split_scaled_max: float = 6.0,
        split_seed: int = 730021,
        cut_max_fraction: float = 0.8,
        selection_mode: int = 0,
        hybrid: bool = False,
        h_null_correction: bool = False,
        return_debug: bool = False,
        capacity_options=None,
    ):
        """Run unchanged Phase-1 selection/aggregation from an exact index.

        ``x`` is C ``[N,D]`` and a normal index is C ``[A,k]``. If ``index``
        is None, ``response[N,MY]`` and capacity mode 4 generate exact kNN
        tiles directly into P1 without materializing N*k. Cuts are C
        ``[S,R]``. These buffers are byte-identical to the corresponding
        MATLAB column-major arrays, so the adapter does not transpose them.
        """
        try:
            import numpy as np
        except ImportError as error:
            raise ManorMCudaError("phase1_precomputed requires NumPy") from error
        if not self._pointer:
            raise ManorMCudaError("CUDA context is already closed")

        x_array = np.ascontiguousarray(x, dtype=np.float32)
        if x_array.ndim != 2:
            raise ValueError(f"x must have shape [N,D], got {x_array.shape}")
        point_count, ambient_dimension = x_array.shape
        generated_index = index_zero_based is None
        response_array = None
        if generated_index:
            if response is None:
                raise ValueError("mode-4 Phase1 requires response[N,MY]")
            response_array = np.ascontiguousarray(response, dtype=np.float32)
            if response_array.ndim != 2 or len(response_array) != point_count:
                raise ValueError("response must have shape [N,MY]")
            if index_is_shared_slice:
                raise ValueError("generated Phase1 cannot use a shared index slice")
            active_anchor_count = point_count
            index_array = None
        else:
            index_array = np.ascontiguousarray(index_zero_based, dtype=np.int32)
        if not generated_index and index_is_shared_slice:
            if index_array.ndim != 1:
                raise ValueError("a shared response index must have shape [k]")
            neighbor_count = int(index_array.shape[0])
            if active_anchor_count is None:
                active_anchor_count = point_count
        elif not generated_index:
            if index_array.ndim != 2:
                raise ValueError(
                    f"index must have shape [A,k], got {index_array.shape}"
                )
            index_anchors, neighbor_count = index_array.shape
            if active_anchor_count is None:
                active_anchor_count = index_anchors
            if int(active_anchor_count) != index_anchors:
                raise ValueError(
                    "active_anchor_count must equal the first index dimension"
                )

        counts_array = np.ascontiguousarray(response_counts, dtype=np.int32)
        cuts_array = np.ascontiguousarray(cuts_scale_major, dtype=np.int32)
        if counts_array.ndim != 1 or counts_array.size < 1:
            raise ValueError("response_counts must be one-dimensional")
        if cuts_array.ndim != 2 or cuts_array.shape[0] != counts_array.size:
            raise ValueError("cuts must have shape [scale_count,cuts_per_scale]")
        if generated_index:
            neighbor_count = int(np.max(counts_array))
            if capacity_options is None or int(capacity_options.get("mode", 0)) != 4:
                raise ValueError("generated Phase1 requires explicit capacity mode 4")
        if hybrid:
            if int(selection_mode) != 0 or counts_array.size != 1:
                raise ValueError(
                    "fused hybrid Phase1 requires selection_mode=0 and one scale"
                )
            if not generated_index and not hasattr(
                self._core._library, "manor_m_cuda_phase1_hybrid_precomputed"
            ):
                raise ManorMCudaError("core does not provide fused hybrid Phase1")

        tangent = np.empty(
            (point_count, int(manifold_dimension), ambient_dimension),
            dtype=np.float32,
        )
        center = np.empty((point_count, ambient_dimension), dtype=np.float32)
        radius = np.empty(point_count, dtype=np.float32)
        axial = np.empty(point_count, dtype=np.float32)
        neighbors = np.empty(point_count, dtype=np.int32)
        owner = np.empty(point_count, dtype=np.uint8)
        covered = np.empty(point_count, dtype=np.uint8)
        selected_scale = np.empty(point_count, dtype=np.uint32)
        selected_response_count = np.empty(point_count, dtype=np.uint32)
        selected_cut = np.empty(point_count, dtype=np.uint32)
        geometry = np.empty(point_count, dtype=np.float32)
        stable_h = np.empty(point_count, dtype=np.float32)
        neighbor_scaled = np.empty(point_count, dtype=np.float32)
        split_scaled = np.empty(point_count, dtype=np.float32)
        split_h_a = np.empty(point_count, dtype=np.float32)
        split_h_b = np.empty(point_count, dtype=np.float32)
        rejection = np.empty(point_count, dtype=np.uint8)
        raw_tangent = (
            np.empty_like(tangent) if return_debug else None
        )
        raw_center = np.empty_like(center) if return_debug else None
        selected_index = (
            np.empty(point_count, dtype=np.uint32) if return_debug else None
        )

        def pointer(array, pointer_type):
            return pointer_type() if array is None else array.ctypes.data_as(pointer_type)

        raw_options = _CPhase1Options(
            struct_size=ctypes.sizeof(_CPhase1Options),
            abi_version=_ABI_VERSION,
            gpu_memory_fraction=float(gpu_memory_fraction),
            reserve_memory_fraction=float(reserve_memory_fraction),
            reserve_memory_bytes=int(reserve_memory_bytes),
            heavy_factor=float(heavy_factor),
            jitter=float(jitter),
            neighbor_scaled_max=float(neighbor_scaled_max),
            split_scaled_max=float(split_scaled_max),
            minimum_query_block=int(minimum_query_block),
            maximum_query_block=int(maximum_query_block),
            stable_window=int(stable_window),
            selection_mode=int(selection_mode),
            split_seed=int(split_seed),
            cut_max_fraction=float(cut_max_fraction),
            index_is_shared_slice=int(index_is_shared_slice),
            h_null_correction=int(h_null_correction),
            collect_profile=1,
            reserved=0,
        )
        raw_output = _CPhase1Output(
            struct_size=ctypes.sizeof(_CPhase1Output),
            abi_version=_ABI_VERSION,
            tangent_point_major=pointer(tangent, _FloatPointer),
            center_point_major=pointer(center, _FloatPointer),
            radius=pointer(radius, _FloatPointer),
            axial_tangent=pointer(axial, _FloatPointer),
            neighbor_count=pointer(neighbors, _Int32Pointer),
            owner_mask=pointer(owner, _Uint8Pointer),
            stable_covered=pointer(covered, _Uint8Pointer),
            selected_scale_one_based=pointer(selected_scale, _Uint32Pointer),
            selected_response_count=pointer(
                selected_response_count, _Uint32Pointer
            ),
            selected_cut=pointer(selected_cut, _Uint32Pointer),
            geometry_score=pointer(geometry, _FloatPointer),
            stable_h=pointer(stable_h, _FloatPointer),
            neighbor_scaled=pointer(neighbor_scaled, _FloatPointer),
            split_scaled=pointer(split_scaled, _FloatPointer),
            split_h_a=pointer(split_h_a, _FloatPointer),
            split_h_b=pointer(split_h_b, _FloatPointer),
            rejection_code=pointer(rejection, _Uint8Pointer),
            raw_tangent_point_major=pointer(raw_tangent, _FloatPointer),
            raw_center_point_major=pointer(raw_center, _FloatPointer),
            selected_index_one_based=pointer(selected_index, _Uint32Pointer),
        )
        raw_stats = _CPhase1Stats()
        symbol = (
            self._core._library.manor_m_cuda_phase1_hybrid_precomputed
            if hybrid
            else self._core._library.manor_m_cuda_phase1_precomputed
        )
        extension = ()
        raw_capacity = _CP1CapacityPlan()
        if capacity_options is not None:
            if not hasattr(self._core._library, 'manor_m_cuda_phase1_bounded'):
                raise ManorMCudaError('core does not provide budgeted Phase1')
            capacity = _p1_capacity_options(capacity_options)
            if hybrid:
                if not hasattr(
                    self._core._library, "manor_m_cuda_phase1_hybrid_bounded"
                ):
                    raise ManorMCudaError(
                        "core does not provide bounded fused hybrid Phase1"
                    )
                symbol = self._core._library.manor_m_cuda_phase1_hybrid_bounded
            else:
                symbol = self._core._library.manor_m_cuda_phase1_bounded
            extension = (ctypes.byref(capacity), ctypes.byref(raw_capacity))
        self.last_phase1_capacity = None
        if generated_index:
            generated_symbol = (
                "manor_m_cuda_phase1_response_hybrid_bounded"
                if hybrid else "manor_m_cuda_phase1_response_bounded"
            )
            if not hasattr(self._core._library, generated_symbol):
                raise ManorMCudaError(
                    "core does not provide generated exact-kNN Phase1"
                )
            symbol = getattr(self._core._library, generated_symbol)
            call_arguments = (
                self._pointer,
                x_array.ctypes.data_as(_FloatPointer),
                response_array.ctypes.data_as(_FloatPointer),
                int(point_count), int(ambient_dimension),
                int(response_array.shape[1]), int(manifold_dimension),
                int(neighbor_count),
                counts_array.ctypes.data_as(_Int32Pointer),
                int(counts_array.size),
                cuts_array.ctypes.data_as(_Int32Pointer),
                int(cuts_array.shape[1]), ctypes.byref(raw_options),
                ctypes.byref(raw_output), ctypes.byref(raw_stats),
                *extension,
            )
        else:
            call_arguments = (
                self._pointer,
                x_array.ctypes.data_as(_FloatPointer),
                int(point_count),
                int(ambient_dimension),
                int(manifold_dimension),
                index_array.ctypes.data_as(_Int32Pointer),
                int(active_anchor_count),
                int(neighbor_count),
                counts_array.ctypes.data_as(_Int32Pointer),
                int(counts_array.size),
                cuts_array.ctypes.data_as(_Int32Pointer),
                int(cuts_array.shape[1]),
                ctypes.byref(raw_options),
                ctypes.byref(raw_output),
                ctypes.byref(raw_stats),
                *extension,
            )
        self._core._check(
            symbol(*call_arguments),
            "phase1_hybrid_precomputed" if hybrid else "phase1_precomputed",
        )
        if capacity_options is not None:
            self.last_phase1_capacity = _p1_capacity_dict(raw_capacity)
        if raw_stats.struct_size != ctypes.sizeof(_CPhase1Stats):
            raise ManorMCudaError(
                "Phase-1 stats struct size mismatch: "
                f"core={raw_stats.struct_size}, Python={ctypes.sizeof(_CPhase1Stats)}"
            )

        result = {
            "Tan0": tangent,
            "Cen0": center,
            "Rad0": radius,
            "axial_tan0": axial,
            "num_neigh0": neighbors,
            "owner_mask0": owner,
            "stable_covered": covered,
            "selected_scale0": selected_scale,
            "selected_response_count0": selected_response_count,
            "selected_cut0": selected_cut,
            "geom0": geometry,
            "stable_h0": stable_h,
            "neighbor_scaled0": neighbor_scaled,
            "split_scaled0": split_scaled,
            "split_h_a0": split_h_a,
            "split_h_b0": split_h_b,
            "rejection_code0": rejection,
        }
        if return_debug:
            result["TanRaw0"] = raw_tangent
            result["CenRaw0"] = raw_center
            result["selected_index0"] = selected_index

        floating_stats = {
            "support_threshold",
            "total_milliseconds",
            "upload_milliseconds",
            "distance_milliseconds",
            "sort_milliseconds",
            "scale_topk_milliseconds",
            "scale_select_milliseconds",
            "scale_stream_milliseconds",
            "final_extract_milliseconds",
            "owner_rule_milliseconds",
            "owner_aggregate_milliseconds",
            "mask_nonowners_milliseconds",
            "download_milliseconds",
        }
        stats = Phase1Stats(
            **{
                name: (
                    float(getattr(raw_stats, name))
                    if name in floating_stats
                    else int(getattr(raw_stats, name))
                )
                for name in Phase1Stats.__annotations__
            }
        )
        return result, stats

    def phase2(
        self,
        x,
        response,
        index_zero_based,
        tangent,
        center,
        owner_mask,
        base_h: float,
        rho_schedule,
        manifold_dimension: int,
        *,
        iteration_count: int = 1,
        h_schedule=None,
        index_is_shared_slice: bool = False,
        lambda_: float = 1.0e-5,
        jitter: float = 1.0e-6,
        gpu_memory_fraction: float = 0.92,
        reserve_memory_fraction: float = 0.08,
        reserve_memory_bytes: int = 0,
        minimum_query_block: int = 1,
        maximum_query_block: int = 131072,
        regression_mode: int = 0,
        heavy_factor: float = 0.10,
        temporary_heavy_factor: float = 0.20,
        evaluation_uses_owner_mask: bool = True,
        test_x=None,
        test_y=None,
        test_tangent=None,
        test_center=None,
        validation_x=None,
        validation_y=None,
        validation_tangent=None,
        eval_rho_scales=(1.0,),
        eval_h_scales=(1.0,),
        evaluated_iterations=None,
        sum_sparse_threshold: float = 10.0,
        count_sparse_threshold: int = 0,
        fallback_exponent_scale: float = 2.0,
        epsilon_weight: float = 1.0e-20,
        warps_per_block: int = 4,
        center_rho_scale: float = 4.0,
        center_h_scale: float = 1.25,
        return_histories: bool = True,
        return_debug: bool = False,
        use_async_allocator: bool = False,
        direct_output_buffers: bool = False,
        evaluation_only: bool = False,
        aggregation_tile_owners: int = 0,
        index_tile_owners: int = 0,
        index_is_compact_owner_order: bool = False,
        generated_neighbor_count: int | None = None,
        knn_scratch_memory_bytes: int = 0,
    ):
        """Run the shared production Phase-2 execution body.

        Every array is C point-major: ``x``/``center`` are ``[N,D]``,
        ``response`` is ``[N,MY]``, and ``tangent`` is ``[N,MM,D]``.  Truth
        arrays are independently optional.  Omitting test ``Y`` skips relMSE,
        omitting test tangent skips DevTan, omitting test center skips DevCen
        and DevCenTan, and omitting the validation pair skips validation.
        Optional index tiling requires explicit eviction of only the resident
        response/compact index first; all host inputs still undergo validation.
        ``index_is_compact_owner_order`` accepts exactly the rows selected by
        ``owner_mask`` in increasing global-owner order, avoiding a complete
        host ``N*k`` array. Its extra index H2D is reported separately and is
        not input-cache reuse.
        Passing ``index_zero_based=None`` selects the exact generated-index
        fallback. It consumes device kNN tiles directly and requires
        ``generated_neighbor_count`` plus a positive exact-kNN scratch budget;
        no complete host or retained device ``N*k``/``M*k`` index is created.
        Its scratch budget selects the largest exact device tile that fits, so
        a request that already fits may intentionally run as one tile.
        ``evaluation_only=True`` scores the supplied checkpoint without
        constructing neighbours or returning duplicated state/status arrays;
        the result contains only requested metrics and extension statistics.
        """
        try:
            import numpy as np
        except ImportError as error:
            raise ManorMCudaError("phase2 requires NumPy") from error
        if not self._pointer:
            raise ManorMCudaError("CUDA context is already closed")
        if not isinstance(direct_output_buffers, (bool, np.bool_)):
            raise TypeError("direct_output_buffers must be boolean")
        if not isinstance(evaluation_only, (bool, np.bool_)):
            raise TypeError("evaluation_only must be boolean")
        if not isinstance(index_is_compact_owner_order, (bool, np.bool_)):
            raise TypeError("index_is_compact_owner_order must be boolean")
        generated_index = index_zero_based is None
        if isinstance(knn_scratch_memory_bytes, (bool, np.bool_)) or not isinstance(
                knn_scratch_memory_bytes, (int, np.integer)):
            raise TypeError("knn_scratch_memory_bytes must be an integer")
        if not 0 <= knn_scratch_memory_bytes <= np.iinfo(np.uint64).max:
            raise ValueError(
                "knn_scratch_memory_bytes must be in [0, UINT64_MAX]")
        if isinstance(aggregation_tile_owners, (bool, np.bool_)) or not isinstance(
                aggregation_tile_owners, (int, np.integer)):
            raise TypeError("aggregation_tile_owners must be an integer")
        if not 0 <= aggregation_tile_owners <= np.iinfo(np.int32).max:
            raise ValueError("aggregation_tile_owners must be in [0, INT32_MAX]")
        if isinstance(index_tile_owners, (bool, np.bool_)) or not isinstance(
                index_tile_owners, (int, np.integer)):
            raise TypeError("index_tile_owners must be an integer")
        if not 0 <= index_tile_owners <= np.iinfo(np.int32).max:
            raise ValueError("index_tile_owners must be in [0, INT32_MAX]")
        phase2_call = self._core._library.manor_m_cuda_phase2
        compact_index_stats = bool(index_is_compact_owner_order)
        generated_neighbor_value = None
        if generated_index:
            if index_is_shared_slice or index_is_compact_owner_order or index_tile_owners:
                raise ValueError(
                    "generated Phase-2 cannot combine with shared, compact, or "
                    "host-streamed indices")
            if knn_scratch_memory_bytes <= 0 and not evaluation_only:
                raise ValueError(
                    "generated Phase-2 requires a positive knn_scratch_memory_bytes")
            if isinstance(generated_neighbor_count, (bool, np.bool_)) or not isinstance(
                    generated_neighbor_count, (int, np.integer)):
                raise TypeError("generated_neighbor_count must be an integer")
            generated_neighbor_value = int(generated_neighbor_count)
            if generated_neighbor_value < 1:
                raise ValueError("generated_neighbor_count must be positive")
            phase2_call = getattr(
                self._core._library, "manor_m_cuda_phase2_response_tiled", None)
            if phase2_call is None:
                raise ManorMCudaError(
                    "This core does not support generated exact-index Phase-2")
        elif generated_neighbor_count is not None or knn_scratch_memory_bytes:
            raise ValueError(
                "generated_neighbor_count and knn_scratch_memory_bytes require "
                "index_zero_based=None")
        elif index_is_compact_owner_order:
            phase2_call = getattr(
                self._core._library, "manor_m_cuda_phase2_compact_index_tiled", None)
            if phase2_call is None:
                raise ManorMCudaError(
                    "This core does not support compact owner-order indices")
        elif index_tile_owners:
            phase2_call = getattr(self._core._library, "manor_m_cuda_phase2_index_tiled", None)
            if phase2_call is None:
                raise ManorMCudaError("This core does not support bounded index streaming")
        elif aggregation_tile_owners:
            phase2_call = getattr(self._core._library, "manor_m_cuda_phase2_tiled", None)
            if phase2_call is None:
                raise ManorMCudaError("This core does not support aggregation scratch tiling")

        x_array = np.ascontiguousarray(x, dtype=np.float32)
        response_array = np.ascontiguousarray(response, dtype=np.float32)
        tangent_array = np.ascontiguousarray(tangent, dtype=np.float32)
        center_array = np.ascontiguousarray(center, dtype=np.float32)
        owner_array = np.ascontiguousarray(owner_mask, dtype=np.uint8)
        if x_array.ndim != 2:
            raise ValueError(f"x must have shape [N,D], got {x_array.shape}")
        point_count, ambient_dimension = x_array.shape
        if response_array.ndim != 2 or response_array.shape[0] != point_count:
            raise ValueError("response must have shape [N,MY]")
        response_dimension = int(response_array.shape[1])
        if tangent_array.shape != (
            point_count,
            int(manifold_dimension),
            ambient_dimension,
        ):
            raise ValueError("tangent must have shape [N,MM,D]")
        if center_array.shape != (point_count, ambient_dimension):
            raise ValueError("center must have shape [N,D]")
        if owner_array.shape != (point_count,):
            raise ValueError("owner_mask must have shape [N]")

        index_array = None if generated_index else np.ascontiguousarray(
            index_zero_based, dtype=np.int32)
        if generated_index:
            neighbor_count = generated_neighbor_value
            if neighbor_count > point_count:
                raise ValueError(
                    "generated_neighbor_count must lie in [1, point_count]")
        elif index_is_shared_slice:
            if index_is_compact_owner_order:
                raise ValueError("a compact owner index cannot be a shared slice")
            if index_array.ndim != 1:
                raise ValueError("a shared Phase-2 index must have shape [k]")
            neighbor_count = int(index_array.size)
        elif index_is_compact_owner_order:
            owner_count = int(np.count_nonzero(owner_array))
            if owner_count < 1:
                raise ValueError("a compact Phase-2 index requires at least one owner")
            if index_array.ndim != 2 or index_array.shape[0] != owner_count:
                raise ValueError(
                    "a compact Phase-2 index must have shape [owner_count,k]")
            neighbor_count = int(index_array.shape[1])
        else:
            if index_array.ndim != 2 or index_array.shape[0] != point_count:
                raise ValueError("a Phase-2 index must have shape [N,k]")
            neighbor_count = int(index_array.shape[1])

        rho_array = np.ascontiguousarray(rho_schedule, dtype=np.float32)
        if rho_array.ndim == 0:
            rho_array = rho_array.reshape(1)
        if rho_array.ndim != 1 or rho_array.size not in (1, int(iteration_count)):
            raise ValueError("rho_schedule must be scalar or iteration-sized")
        h_array = None
        if h_schedule is not None:
            h_array = np.ascontiguousarray(h_schedule, dtype=np.float32)
            if h_array.ndim == 0:
                h_array = h_array.reshape(1)
            if h_array.ndim != 1 or h_array.size not in (1, int(iteration_count)):
                raise ValueError("h_schedule must be scalar or iteration-sized")

        def optional_array(value, shape, name):
            if value is None:
                return None
            result = np.ascontiguousarray(value, dtype=np.float32)
            if result.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
            return result

        test_x_array = None
        test_count = 0
        if test_x is not None:
            test_x_array = np.ascontiguousarray(test_x, dtype=np.float32)
            if test_x_array.ndim != 2 or test_x_array.shape[1] != ambient_dimension:
                raise ValueError("test_x must have shape [Ntest,D]")
            test_count = int(test_x_array.shape[0])
        if any(value is not None for value in (test_y, test_tangent, test_center)) \
                and test_x_array is None:
            raise ValueError("testing truth requires test_x")
        test_y_array = optional_array(
            test_y, (test_count, response_dimension), "test_y"
        )
        test_tangent_array = optional_array(
            test_tangent,
            (test_count, int(manifold_dimension), ambient_dimension),
            "test_tangent",
        )
        test_center_array = optional_array(
            test_center, (test_count, ambient_dimension), "test_center"
        )

        if (validation_x is None) != (validation_y is None):
            raise ValueError("validation_x and validation_y must be supplied together")
        validation_x_array = validation_y_array = None
        validation_count = 0
        if validation_x is not None:
            validation_x_array = np.ascontiguousarray(validation_x, dtype=np.float32)
            validation_y_array = np.ascontiguousarray(validation_y, dtype=np.float32)
            if validation_x_array.ndim != 2 or validation_x_array.shape[1] != ambient_dimension:
                raise ValueError("validation_x must have shape [Nvalid,D]")
            validation_count = int(validation_x_array.shape[0])
            if validation_y_array.shape != (validation_count, response_dimension):
                raise ValueError("validation_y must have shape [Nvalid,MY]")

        validation_tangent_array = optional_array(
            validation_tangent, (validation_count, manifold_dimension, ambient_dimension),
            "validation_tangent")
        extended_validation = hasattr(self._core._library,
            "manor_m_cuda_validation_tangent_version")
        if validation_tangent_array is not None and (not extended_validation or not validation_count):
            raise ValueError("validation tangent requires a supporting core and validation X/Y")

        evaluation_enabled = any(
            value is not None
            for value in (
                test_y_array,
                test_tangent_array,
                test_center_array,
                validation_x_array,
            )
        )
        test_metrics_enabled = any(
            value is not None
            for value in (test_y_array, test_tangent_array, test_center_array)
        )
        eval_rho_array = np.ascontiguousarray(eval_rho_scales, dtype=np.float32)
        eval_h_array = np.ascontiguousarray(eval_h_scales, dtype=np.float32)
        if eval_rho_array.ndim == 0:
            eval_rho_array = eval_rho_array.reshape(1)
        if eval_h_array.ndim == 0:
            eval_h_array = eval_h_array.reshape(1)
        if eval_rho_array.ndim != 1 or eval_rho_array.size < 1:
            raise ValueError("eval_rho_scales must be a nonempty vector")
        if eval_h_array.ndim != 1 or eval_h_array.size < 1:
            raise ValueError("eval_h_scales must be a nonempty vector")
        eval_iteration_array = None
        if evaluated_iterations is not None:
            eval_iteration_array = np.ascontiguousarray(
                evaluated_iterations, dtype=np.int32
            )
            if eval_iteration_array.ndim == 0:
                eval_iteration_array = eval_iteration_array.reshape(1)
            if eval_iteration_array.ndim != 1:
                raise ValueError("evaluated_iterations must be one-dimensional")
        eval_k0 = eval_iteration_array is not None and bool(
            np.any(eval_iteration_array == 0)
        )
        if evaluation_only:
            if not evaluation_enabled:
                raise ValueError("evaluation_only requires evaluation truth")
            if (eval_iteration_array is None or eval_iteration_array.size != 1 or
                    int(eval_iteration_array[0]) != 0):
                raise ValueError(
                    "evaluation_only requires evaluated_iterations=(0,)")
            if return_histories or return_debug:
                raise ValueError(
                    "evaluation_only requires histories and debug output disabled")
            if not evaluation_uses_owner_mask:
                raise ValueError("evaluation_only requires owner-mask scoring")

        # A checkpoint score has no next state.  Keeping these pointers null
        # prevents both an N-sized device duplicate and an N-sized host/D2H
        # result that the caller would immediately discard.
        tangent_next = None if evaluation_only else np.empty_like(tangent_array)
        center_next = None if evaluation_only else np.empty_like(center_array)
        neighbors = None if evaluation_only else np.empty(point_count, dtype=np.int32)
        heavy = None if evaluation_only else np.empty(point_count, dtype=np.uint8)
        keep = None if evaluation_only else np.empty(point_count, dtype=np.uint8)
        solver_failed = None if evaluation_only else np.empty(point_count, dtype=np.uint8)
        count_history_raw = (
            np.empty((int(iteration_count), point_count), dtype=np.int32)
            if return_histories
            else None
        )
        heavy_history_raw = (
            np.empty((int(iteration_count), point_count), dtype=np.uint8)
            if return_histories
            else None
        )
        keep_history_raw = (
            np.empty((int(iteration_count), point_count), dtype=np.uint8)
            if return_histories
            else None
        )
        gradient = (
            np.empty(
                (point_count, response_dimension, ambient_dimension),
                dtype=np.float32,
            )
            if return_debug
            else None
        )

        n_rho = int(eval_rho_array.size)
        n_h = int(eval_h_array.size)
        k_count = int(iteration_count)
        eval_raw = {}
        if evaluation_enabled:
            for name in (
                "EvalMSE",
                "EvalDevCen",
                "EvalDevCenTan",
                "EvalDevTan",
            ):
                eval_raw[name] = np.empty((n_h, n_rho, k_count), dtype=np.float32)
            eval_raw["EvalMSECoordinate"] = np.empty(
                (response_dimension, n_h, n_rho, k_count), dtype=np.float32
            )
            if eval_k0 and test_metrics_enabled:
                for name in (
                    "EvalMSE0",
                    "EvalDevCen0",
                    "EvalDevCenTan0",
                    "EvalDevTan0",
                ):
                    eval_raw[name] = np.empty((n_h, n_rho), dtype=np.float32)
                eval_raw["EvalMSECoordinate0"] = np.empty(
                    (response_dimension, n_h, n_rho), dtype=np.float32
                )
            if validation_tangent_array is not None:
                eval_raw["EvalValidDevTan"] = np.empty((n_h, n_rho, k_count), dtype=np.float32)
                if eval_k0:
                    eval_raw["EvalValidDevTan0"] = np.empty((n_h, n_rho), dtype=np.float32)
            if validation_x_array is not None:
                eval_raw["EvalValidMSE"] = np.empty(
                    (n_h, n_rho, k_count), dtype=np.float32
                )
                eval_raw["EvalValidMSECoordinate"] = np.empty(
                    (response_dimension, n_h, n_rho, k_count), dtype=np.float32
                )
                if eval_k0:
                    eval_raw["EvalValidMSE0"] = np.empty(
                        (n_h, n_rho), dtype=np.float32
                    )
                    eval_raw["EvalValidMSECoordinate0"] = np.empty(
                        (response_dimension, n_h, n_rho), dtype=np.float32
                    )

        def pointer(array, pointer_type):
            return pointer_type() if array is None else array.ctypes.data_as(pointer_type)

        raw_input = _CPhase2Input(
            struct_size=ctypes.sizeof(_CPhase2Input),
            abi_version=_ABI_VERSION,
            x_point_major=pointer(x_array, _FloatPointer),
            response_point_major=pointer(response_array, _FloatPointer),
            index_zero_based=pointer(index_array, _Int32Pointer),
            tangent_point_major=pointer(tangent_array, _FloatPointer),
            center_point_major=pointer(center_array, _FloatPointer),
            owner_mask=pointer(owner_array, _Uint8Pointer),
            point_count=point_count,
            ambient_dimension=ambient_dimension,
            manifold_dimension=int(manifold_dimension),
            response_dimension=response_dimension,
            neighbor_count=neighbor_count,
            base_h=float(base_h),
            rho_schedule=pointer(rho_array, _FloatPointer),
            rho_schedule_count=int(rho_array.size),
            reserved=0,
        )
        raw_options = _CPhase2Options(
            struct_size=ctypes.sizeof(_CPhase2Options),
            abi_version=_ABI_VERSION,
            lambda_=float(lambda_),
            jitter=float(jitter),
            gpu_memory_fraction=float(gpu_memory_fraction),
            reserve_memory_fraction=float(reserve_memory_fraction),
            reserve_memory_bytes=int(reserve_memory_bytes),
            minimum_query_block=int(minimum_query_block),
            maximum_query_block=int(maximum_query_block),
            regression_mode=int(regression_mode),
            iteration_count=int(iteration_count),
            heavy_factor=float(heavy_factor),
            temporary_heavy_factor=float(temporary_heavy_factor),
            h_schedule=pointer(h_array, _FloatPointer),
            h_schedule_count=0 if h_array is None else int(h_array.size),
            index_is_shared_slice=int(index_is_shared_slice),
            evaluation_uses_owner_mask=int(evaluation_uses_owner_mask),
            collect_profile=1,
            use_async_allocator=int(use_async_allocator),
            reserved=(int(direct_output_buffers) | (int(evaluation_only) << 1)),
        )
        raw_evaluation = None
        if evaluation_enabled:
            raw_evaluation = _CPhase2Evaluation(
                struct_size=(ctypes.sizeof(_CPhase2Evaluation) if extended_validation else
                    _CPhase2Evaluation.validation_tangent_point_major.offset),
                abi_version=_ABI_VERSION,
                test_x_point_major=pointer(test_x_array, _FloatPointer),
                test_y_point_major=pointer(test_y_array, _FloatPointer),
                test_tangent_point_major=pointer(test_tangent_array, _FloatPointer),
                test_center_point_major=pointer(test_center_array, _FloatPointer),
                test_point_count=test_count,
                validation_x_point_major=pointer(validation_x_array, _FloatPointer),
                validation_y_point_major=pointer(validation_y_array, _FloatPointer),
                validation_point_count=validation_count,
                validation_tangent_point_major=pointer(validation_tangent_array, _FloatPointer),
                rho_scales=pointer(eval_rho_array, _FloatPointer),
                rho_scale_count=n_rho,
                h_scales=pointer(eval_h_array, _FloatPointer),
                h_scale_count=n_h,
                evaluated_iterations=pointer(eval_iteration_array, _Int32Pointer),
                evaluated_iteration_count=(
                    0 if eval_iteration_array is None else int(eval_iteration_array.size)
                ),
                sum_sparse_threshold=float(sum_sparse_threshold),
                count_sparse_threshold=int(count_sparse_threshold),
                fallback_exponent_scale=float(fallback_exponent_scale),
                epsilon_weight=float(epsilon_weight),
                warps_per_block=int(warps_per_block),
                center_rho_scale=float(center_rho_scale),
                center_h_scale=float(center_h_scale),
                reserved=0,
            )
        raw_output = _CPhase2Output(
            struct_size=(ctypes.sizeof(_CPhase2Output) if extended_validation else
                _CPhase2Output.validation_dev_tangent.offset),
            abi_version=_ABI_VERSION,
            tangent_point_major=pointer(tangent_next, _FloatPointer),
            center_point_major=pointer(center_next, _FloatPointer),
            neighbor_count=pointer(neighbors, _Int32Pointer),
            heavy_mask=pointer(heavy, _Uint8Pointer),
            keep_mask=pointer(keep, _Uint8Pointer),
            solver_failed=pointer(solver_failed, _Uint8Pointer),
            neighbor_count_history=pointer(count_history_raw, _Int32Pointer),
            heavy_mask_history=pointer(heavy_history_raw, _Uint8Pointer),
            keep_mask_history=pointer(keep_history_raw, _Uint8Pointer),
            gradient_debug_point_major=pointer(gradient, _FloatPointer),
            eval_mse=pointer(eval_raw.get("EvalMSE"), _FloatPointer),
            eval_mse_coordinate=pointer(
                eval_raw.get("EvalMSECoordinate"), _FloatPointer
            ),
            eval_dev_center=pointer(eval_raw.get("EvalDevCen"), _FloatPointer),
            eval_dev_center_tangent=pointer(
                eval_raw.get("EvalDevCenTan"), _FloatPointer
            ),
            eval_dev_tangent=pointer(eval_raw.get("EvalDevTan"), _FloatPointer),
            eval_mse_k0=pointer(eval_raw.get("EvalMSE0"), _FloatPointer),
            eval_mse_coordinate_k0=pointer(
                eval_raw.get("EvalMSECoordinate0"), _FloatPointer
            ),
            eval_dev_center_k0=pointer(
                eval_raw.get("EvalDevCen0"), _FloatPointer
            ),
            eval_dev_center_tangent_k0=pointer(
                eval_raw.get("EvalDevCenTan0"), _FloatPointer
            ),
            eval_dev_tangent_k0=pointer(
                eval_raw.get("EvalDevTan0"), _FloatPointer
            ),
            validation_mse=pointer(
                eval_raw.get("EvalValidMSE"), _FloatPointer
            ),
            validation_mse_coordinate=pointer(
                eval_raw.get("EvalValidMSECoordinate"), _FloatPointer
            ),
            validation_mse_k0=pointer(
                eval_raw.get("EvalValidMSE0"), _FloatPointer
            ),
            validation_mse_coordinate_k0=pointer(
                eval_raw.get("EvalValidMSECoordinate0"), _FloatPointer
            ),
            validation_dev_tangent=pointer(eval_raw.get("EvalValidDevTan"), _FloatPointer),
            validation_dev_tangent_k0=pointer(eval_raw.get("EvalValidDevTan0"), _FloatPointer),
        )
        raw_stats = _CPhase2Stats()
        evaluation_pointer = (
            ctypes.POINTER(_CPhase2Evaluation)()
            if raw_evaluation is None
            else ctypes.byref(raw_evaluation)
        )
        call_arguments = [self._pointer, ctypes.byref(raw_input),
            ctypes.byref(raw_options), evaluation_pointer,
            ctypes.byref(raw_output), ctypes.byref(raw_stats)]
        raw_index_stats = _CIndexStreamStats() if (index_tile_owners or compact_index_stats) else None
        raw_generated_stats = _CGeneratedIndexStats() if generated_index else None
        if generated_index:
            call_arguments.extend([int(aggregation_tile_owners),
                int(knn_scratch_memory_bytes), ctypes.byref(raw_generated_stats)])
        elif index_is_compact_owner_order:
            call_arguments.extend([int(aggregation_tile_owners), int(index_tile_owners),
                                   ctypes.byref(raw_index_stats)])
        elif index_tile_owners:
            call_arguments.extend([int(aggregation_tile_owners), int(index_tile_owners),
                                   ctypes.byref(raw_index_stats)])
        elif aggregation_tile_owners:
            call_arguments.append(int(aggregation_tile_owners))
        self._core._check(phase2_call(*call_arguments), "phase2")
        if raw_stats.struct_size != ctypes.sizeof(_CPhase2Stats):
            raise ManorMCudaError(
                "Phase-2 stats struct size mismatch: "
                f"core={raw_stats.struct_size}, Python={ctypes.sizeof(_CPhase2Stats)}"
            )

        result = {} if evaluation_only else {
            "Tan_next": tangent_next,
            "Cen_next": center_next,
            "num_neigh": neighbors,
            "heavy_mask": heavy,
            "keep_mask": keep,
            "solver_failed": solver_failed,
        }
        if raw_index_stats is not None:
            if (raw_index_stats.struct_size != ctypes.sizeof(_CIndexStreamStats)
                    or raw_index_stats.extension_version != 1):
                raise ManorMCudaError("Unsupported index-stream stats layout")
            result["index_stream_stats"] = {name: getattr(raw_index_stats, name)
                                           for name, _ in raw_index_stats._fields_}
        if raw_generated_stats is not None:
            if (raw_generated_stats.struct_size != ctypes.sizeof(_CGeneratedIndexStats)
                    or raw_generated_stats.extension_version != 1):
                raise ManorMCudaError("Unsupported generated-index stats layout")
            result["generated_index_stats"] = {
                name: getattr(raw_generated_stats, name)
                for name, _ in raw_generated_stats._fields_
            }
        if return_histories:
            result["num_neigh_iter"] = count_history_raw.T
            result["heavy_mask_iter"] = heavy_history_raw.T
            result["keep_mask_iter"] = keep_history_raw.T
        if return_debug:
            result["grad_debug"] = gradient
        for name, array in eval_raw.items():
            if array.ndim == 4:
                result[name] = array.transpose(3, 2, 1, 0)
            elif array.ndim == 3 and array.shape[0] == response_dimension:
                result[name] = array.transpose(2, 1, 0)
            elif array.ndim == 3:
                result[name] = array.transpose(2, 1, 0)
            elif array.ndim == 2:
                result[name] = array.T
            else:
                result[name] = array

        stats = Phase2Stats(
            **{
                name: (
                    float(getattr(raw_stats, name))
                    if name.endswith("_milliseconds")
                    else int(getattr(raw_stats, name))
                )
                for name in Phase2Stats.__annotations__
            }
        )
        return result, stats


class ManorMCuda:
    """Loaded shared CUDA core with Python-native result objects."""

    def __init__(self, library_path: str | os.PathLike[str] | None = None):
        self._library = _load_library(library_path)

    @property
    def version(self) -> str:
        return self._library.manor_m_cuda_version().decode("utf-8")

    @property
    def abi_version(self) -> int:
        return int(self._library.manor_m_cuda_abi_version())

    def _check(self, status: int, operation: str) -> None:
        if status == _SUCCESS:
            return
        detail_pointer = self._library.manor_m_cuda_last_error()
        detail = detail_pointer.decode("utf-8") if detail_pointer else ""
        status_pointer = self._library.manor_m_cuda_status_string(status)
        status_name = status_pointer.decode("utf-8") if status_pointer else str(status)
        raise ManorMCudaError(f"{operation}: {status_name}: {detail}")

    def device_count(self) -> int:
        count = ctypes.c_int32()
        self._check(
            self._library.manor_m_cuda_device_count(ctypes.byref(count)),
            "device_count",
        )
        return int(count.value)

    def device_info(self, device_ordinal: int) -> DeviceInfo:
        raw = _CDeviceInfo()
        self._check(
            self._library.manor_m_cuda_get_device_info(
                int(device_ordinal), ctypes.byref(raw)
            ),
            "device_info",
        )
        if raw.struct_size != ctypes.sizeof(_CDeviceInfo):
            raise ManorMCudaError(
                f"Device-info struct size mismatch: core={raw.struct_size}, "
                f"Python={ctypes.sizeof(_CDeviceInfo)}"
            )
        if raw.abi_version != _ABI_VERSION:
            raise ManorMCudaError(
                f"Device-info ABI mismatch: core={raw.abi_version}, "
                f"Python={_ABI_VERSION}"
            )
        return DeviceInfo(
            device_ordinal=int(raw.device_ordinal),
            name=bytes(raw.name).split(b"\0", 1)[0].decode("utf-8"),
            compute_capability=(
                f"{raw.compute_capability_major}.{raw.compute_capability_minor}"
            ),
            multiprocessor_count=int(raw.multiprocessor_count),
            warp_size=int(raw.warp_size),
            max_threads_per_block=int(raw.max_threads_per_block),
            cuda_driver_version=int(raw.cuda_driver_version),
            cuda_runtime_version=int(raw.cuda_runtime_version),
            total_global_memory_bytes=int(raw.total_global_memory_bytes),
            free_global_memory_bytes=int(raw.free_global_memory_bytes),
        )

    def sweep_session(self, *, device_ordinal=0, maximum_retained_bytes=16 * 1024**3):
        """Candidate explicit-residency API; each sweep defaults to reuse=False."""
        from resident_sweep import SweepSession
        return SweepSession(self, device_ordinal=device_ordinal,
                            maximum_retained_bytes=maximum_retained_bytes)

    def context(
        self,
        device_ordinal: int = 0,
        stream_count: int = 2,
        memory_pool_release_threshold_bytes: int = 0,
    ) -> Context:
        options = _CContextOptions(
            struct_size=ctypes.sizeof(_CContextOptions),
            abi_version=_ABI_VERSION,
            device_ordinal=int(device_ordinal),
            stream_count=int(stream_count),
            memory_pool_release_threshold_bytes=int(
                memory_pool_release_threshold_bytes
            ),
        )
        pointer = ctypes.c_void_p()
        self._check(
            self._library.manor_m_cuda_context_create(
                ctypes.byref(options), ctypes.byref(pointer)
            ),
            "context_create",
        )
        return Context(self, pointer)

    def plan_tiles(
        self,
        *,
        device_ordinal: int,
        persistent_memory_bytes: int,
        reference_row_count: int,
        reference_bytes_per_row: int,
        query_row_count: int,
        query_bytes_per_row: int,
        scratch_bytes_per_query_row: int = 0,
        staging_slot_count: int = 2,
        usable_memory_fraction: float = 0.92,
        reserve_memory_bytes: int = 0,
        minimum_reference_tile_rows: int = 1,
        minimum_query_tile_rows: int = 1,
        preferred_reference_tile_rows: int = 0,
        preferred_query_tile_rows: int = 0,
    ) -> TilePlan:
        request = _CTilePlanRequest(
            struct_size=ctypes.sizeof(_CTilePlanRequest),
            abi_version=_ABI_VERSION,
            device_ordinal=int(device_ordinal),
            staging_slot_count=int(staging_slot_count),
            usable_memory_fraction=float(usable_memory_fraction),
            reserve_memory_bytes=int(reserve_memory_bytes),
            persistent_memory_bytes=int(persistent_memory_bytes),
            reference_row_count=int(reference_row_count),
            reference_bytes_per_row=int(reference_bytes_per_row),
            query_row_count=int(query_row_count),
            query_bytes_per_row=int(query_bytes_per_row),
            scratch_bytes_per_query_row=int(scratch_bytes_per_query_row),
            minimum_reference_tile_rows=int(minimum_reference_tile_rows),
            minimum_query_tile_rows=int(minimum_query_tile_rows),
            preferred_reference_tile_rows=int(preferred_reference_tile_rows),
            preferred_query_tile_rows=int(preferred_query_tile_rows),
        )
        raw = _CTilePlan()
        self._check(
            self._library.manor_m_cuda_plan_tiles(
                ctypes.byref(request), ctypes.byref(raw)
            ),
            "plan_tiles",
        )
        if raw.struct_size != ctypes.sizeof(_CTilePlan):
            raise ManorMCudaError(
                f"Tile-plan struct size mismatch: core={raw.struct_size}, "
                f"Python={ctypes.sizeof(_CTilePlan)}"
            )
        mode = {0: "resident", 1: "tiled"}.get(
            int(raw.execution_mode), f"unknown-{raw.execution_mode}"
        )
        return TilePlan(
            execution_mode=mode,
            staging_slot_count=int(raw.staging_slot_count),
            device_free_memory_bytes=int(raw.device_free_memory_bytes),
            usable_memory_bytes=int(raw.usable_memory_bytes),
            full_resident_memory_bytes=int(raw.full_resident_memory_bytes),
            estimated_peak_memory_bytes=int(raw.estimated_peak_memory_bytes),
            reference_tile_rows=int(raw.reference_tile_rows),
            query_tile_rows=int(raw.query_tile_rows),
            reference_tile_count=int(raw.reference_tile_count),
            query_tile_count=int(raw.query_tile_count),
        )

    def response_knn_memory(
        self,
        point_count: int,
        response_dimension: int,
        neighbor_count: int,
        *,
        tiled_output: bool = True,
    ) -> dict[str, int]:
        """Return the core's exact fixed and per-query kNN scratch terms."""
        symbol = getattr(
            self._library, "manor_m_cuda_response_knn_memory", None
        )
        if symbol is None:
            raise ManorMCudaError(
                "core does not expose response-kNN memory arithmetic"
            )
        if not isinstance(tiled_output, bool):
            raise TypeError("tiled_output must be boolean")
        fixed = ctypes.c_uint64()
        per_query = ctypes.c_uint64()
        self._check(
            symbol(
                int(point_count),
                int(response_dimension),
                int(neighbor_count),
                int(tiled_output),
                ctypes.byref(fixed),
                ctypes.byref(per_query),
            ),
            "response_knn_memory",
        )
        full = int(fixed.value) + int(point_count) * int(per_query.value)
        if not fixed.value or not per_query.value or full > 2**64 - 1:
            raise ManorMCudaError(
                "core returned invalid response-kNN memory arithmetic"
            )
        return {
            "fixed_bytes": int(fixed.value),
            "bytes_per_query": int(per_query.value),
            "full_scratch_bytes": full,
        }

    def smoke_axpy(self, x, y, alpha: float = 2.0, device_ordinal: int = 0):
        """Run the shared-core adapter smoke and return a float32 NumPy array."""
        try:
            import numpy as np
        except ImportError as error:
            raise ManorMCudaError("smoke_axpy requires NumPy") from error

        x_array = np.ascontiguousarray(x, dtype=np.float32)
        y_array = np.ascontiguousarray(y, dtype=np.float32)
        if x_array.shape != y_array.shape:
            raise ValueError(
                f"x and y must have the same shape, got {x_array.shape} and "
                f"{y_array.shape}"
            )
        output = np.empty_like(x_array)
        float_pointer = ctypes.POINTER(ctypes.c_float)
        self._check(
            self._library.manor_m_cuda_smoke_axpy(
                int(device_ordinal),
                float(alpha),
                x_array.ctypes.data_as(float_pointer),
                y_array.ctypes.data_as(float_pointer),
                output.ctypes.data_as(float_pointer),
                int(x_array.size),
            ),
            "smoke_axpy",
        )
        return output


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", help="Explicit path to libmanor_m_cuda")
    parser.add_argument(
        "--devices",
        default="all",
        help="Comma-separated CUDA ordinals, or 'all' (default)",
    )
    parser.add_argument(
        "--skip-smoke", action="store_true", help="Report devices without AXPY"
    )
    parser.add_argument(
        "--tiled-smoke-elements",
        type=int,
        default=0,
        help="Also run this many float32 elements through the tiled engine",
    )
    parser.add_argument(
        "--tile-elements",
        type=int,
        default=0,
        help="Tiled-smoke tile size; zero selects the core default",
    )
    arguments = parser.parse_args()

    core = ManorMCuda(arguments.library)
    if arguments.devices == "all":
        devices = list(range(core.device_count()))
    else:
        devices = [int(value) for value in arguments.devices.split(",")]

    report = {
        "schema": "manor-m-cuda-adapter-smoke.v1",
        "core_version": core.version,
        "abi_version": core.abi_version,
        "devices": [],
    }
    for device in devices:
        item = asdict(core.device_info(device))
        if not arguments.skip_smoke:
            import numpy as np

            x = np.linspace(-1.0, 1.0, 4099, dtype=np.float32)
            y = np.linspace(2.0, -2.0, 4099, dtype=np.float32)
            alpha = np.float32(1.75)
            output = core.smoke_axpy(x, y, float(alpha), device)
            expected = alpha * x + y
            max_abs_error = float(np.max(np.abs(output - expected)))
            item["smoke_max_abs_error"] = max_abs_error
            item["smoke_pass"] = max_abs_error <= 2.0e-6
        if arguments.tiled_smoke_elements > 0:
            import numpy as np

            count = arguments.tiled_smoke_elements
            x = np.linspace(-1.0, 1.0, count, dtype=np.float32)
            y = np.linspace(2.0, -2.0, count, dtype=np.float32)
            alpha = np.float32(1.75)
            with core.context(device_ordinal=device, stream_count=2) as context:
                output, stats = context.smoke_tiled_axpy(
                    x,
                    y,
                    float(alpha),
                    arguments.tile_elements,
                )
            expected = alpha * x + y
            max_abs_error = float(np.max(np.abs(output - expected)))
            item["tiled_smoke"] = asdict(stats)
            item["tiled_smoke_max_abs_error"] = max_abs_error
            item["tiled_smoke_pass"] = max_abs_error <= 2.0e-6
        report["devices"].append(item)

    report["pass"] = all(
        item.get("smoke_pass", True) and item.get("tiled_smoke_pass", True)
        for item in report["devices"]
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
