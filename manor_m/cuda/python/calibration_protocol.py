"""Calibration grids, validation selection, atomic output and worker ownership.

The N=1M ``kSlice`` axis is the sampled alpha axis: alpha=kSlice/1,000,000.  At any
other N the effective neighbour count is

    min(N, max(k_floor(D), ceil(alpha*N))).

The floor is an algorithmic safety bound, not a tuned hyperparameter. K=0
remains a selectable initial model; its nullable K_offset is not discarded.
"""
from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid
import numpy as np
from numbers import Real



AXES = {
    "kSlice": ("k_floors", "k"),
    "h_scale": ("h_scales", "positive"),
    "rho_init": ("rho_inits", "positive"),
    "bite": ("bites", "bite"),
    "K_offset": ("K_offsets", "integer"),
    "eval_rho": ("eval_rhos", "positive"),
    "eval_h": ("eval_hs", "positive"),
}
SUBMILLION_N_VALUES = tuple(n * scale for scale in (1000, 10000, 100000)
                           for n in range(1, 10))
REFERENCE_N = 1_000_000
K_ALPHA_BINDING_CONTRACT = (
    "alpha=kSlice(N=1000000)/1000000; "
    "k(N)=min(N,max(k_floor(D),ceil(alpha*N)))")


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("expected a finite number")
    return float(value)


def _positive_integer(value):
    number = _number(value)
    if number < 1 or number != int(number):
        raise ValueError("expected a positive integer")
    return int(number)


def _triplet(grid):
    values = [_number(value) for value in grid]
    if len(values) != 3 or len(set(values)) != 3:
        raise ValueError("expected three distinct finite axis values")
    return sorted(values)


def _nullable_offset(value):
    if value is None or (isinstance(value, Real) and not isinstance(value, bool)
                         and math.isnan(value)):
        return None
    return _number(value)


def matlab_round(value):
    """MATLAB round(): half away from zero, not Python's ties-to-even."""
    value = _number(value)
    fraction, integral = math.modf(abs(value))
    rounded = int(integral) + int(fraction >= 0.5)
    return -rounded if value < 0 else rounded


def minimum_k(D):
    D = _positive_integer(D)
    return max(9 * D, math.ceil((3 * D + 127) / 0.8))


def initial_profile(D):
    return dict(name="N1M_round_01", n_min=1000000, n_max=1000000,
        k_over_N=0, k_min=minimum_k(D), k_floors=[300, 450, 675],
        h_scales=[3.5, 5.0, 7.0], rho_inits=[0.5, 0.725, 1.0],
        bites=[2, 4, 6], K_offsets=[8, 12, 16],
        eval_rhos=[0.5, 0.75, 1.0], eval_hs=[0.5, 0.75, 1.0])


def shift_axis(grid, winner, kind, lower_bound=-math.inf):
    """Shift a three-point axis one step, preserving two previous values."""
    values = _triplet(grid)
    winner = _number(winner)
    if kind not in {"positive", "k", "bite", "integer"}:
        raise ValueError("unknown adaptive axis kind")
    if kind == "positive" and values[0] <= 0:
        raise ValueError("positive axis must be positive")
    position = min(range(3), key=lambda i: abs(values[i] - winner))
    if abs(values[position] - winner) > 1e-9 * max(1, abs(winner)):
        raise ValueError("winner is absent from its grid")
    if position == 1:
        return values
    edge = values[position]
    outward = edge ** 2 / values[1] if kind in {"positive", "k"} else 2 * edge - values[1]
    if kind != "positive":
        outward = matlab_round(outward)
    if position == 0:
        if kind == "k":
            outward = max(lower_bound, outward)
        elif kind == "bite":
            outward = max(0, outward)
        shifted = [outward, *values[:2]]
    else:
        shifted = [*values[1:], outward]
    shifted = _triplet(shifted)
    if kind == "positive" and shifted[0] <= 0:
        raise ValueError("invalid positive grid expansion")
    return shifted


def next_profile(profile, winner, next_round):
    """Return (next profile or None, expandable flags, hard-floor flags).

    Inputs must already come from an independently verified validation-only
    round. A None profile means bracketed, NOT that final testing has passed.
    Never trust caller-supplied edge flags or mutate a stored profile.
    """
    next_round = _positive_integer(next_round)
    if next_round < 2:
        raise ValueError("next round must be at least 2")
    k_min = _positive_integer(profile["k_min"])
    normalized_winner = dict(winner)
    normalized_winner["K_offset"] = _nullable_offset(winner["K_offset"])
    for coordinate, (field, kind) in AXES.items():
        values = _triplet(profile[field])
        if list(profile[field]) != values:
            raise ValueError("profile axes must be sorted")
        if kind == "positive" and values[0] <= 0:
            raise ValueError("positive axis must be positive")
        if kind == "bite" and values[0] < 0:
            raise ValueError("bite cannot be negative")
        if coordinate == "kSlice":
            values = sorted(set(max(value, k_min) for value in values))
        if coordinate == "K_offset" and normalized_winner[coordinate] is None:
            if winner["K"] != 0:
                raise ValueError("nullable K_offset requires the K=0 initial model")
            continue
        value = _number(normalized_winner[coordinate])
        if min(abs(value - point) for point in values) > 1e-9 * max(1, abs(value)):
            raise ValueError("winner is absent from its grid: " + coordinate)
    flags, hard = boundary_flags(profile, normalized_winner)
    if not any(flags.values()):
        return None, flags, hard
    result = copy.deepcopy(profile)
    for coordinate, (field, kind) in AXES.items():
        if flags[coordinate]:
            values = profile[field]
            if coordinate == "kSlice":
                values = [max(value, k_min) for value in values]
            result[field] = shift_axis(values, normalized_winner[coordinate], kind, k_min)
    result["name"] = "N1M_round_%02d" % next_round
    return result, flags, hard


def _ceil_ratio(numerator, denominator):
    """Exact ceil(numerator/denominator) for nonnegative integers."""
    numerator = _positive_integer(numerator)
    denominator = _positive_integer(denominator)
    return (numerator + denominator - 1) // denominator


def _alpha_k_binding(reference_k, N, D, reference_N=REFERENCE_N):
    N, k_min = _positive_integer(N), minimum_k(D)
    reference_N = _positive_integer(reference_N)
    reference_k = _positive_integer(reference_k)
    if N < k_min:
        raise ValueError("N is below the feasible minimum k")
    if reference_k > reference_N:
        raise ValueError("N1M k sample must define alpha in (0,1]")
    raw_k = _ceil_ratio(reference_k * N, reference_N)
    effective_k = min(N, max(k_min, raw_k))
    return dict(alpha=reference_k / reference_N,
                reference_N=reference_N,
                reference_kSlice=reference_k,
                raw_kSlice=raw_k,
                kSlice=effective_k,
                k_floor=k_min,
                k_floor_applied=raw_k < k_min,
                N_cap_applied=raw_k > N)


def alpha_k_bindings(reference, N, D, reference_N=REFERENCE_N):
    """Bind every N1M alpha sample to its exact effective k at N.

    Integer arithmetic avoids a floating-point ceil changing k at an exact
    boundary. Multiple alpha values may legitimately collapse to one k at the
    safety floor; they remain in the returned provenance instead of inventing
    replacement endpoints or midpoints.
    """
    N, k_min = _positive_integer(N), minimum_k(D)
    reference_N = _positive_integer(reference_N)
    if N < k_min:
        raise ValueError("N is below the feasible minimum k")
    reference = [matlab_round(value) for value in _triplet(reference)]
    if reference[0] < 1 or reference[-1] > reference_N:
        raise ValueError("N1M k samples must define alpha in (0,1]")
    return [_alpha_k_binding(reference_k, N, D, reference_N)
            for reference_k in reference]


def bind_k_grid(reference, N, D):
    """Map the three N1M alpha samples to unique effective k values at N.

    The second return value reports a mathematical floor/cap, not the expected
    alpha scaling itself. A collapsed grid intentionally has one or two k
    values: adding synthetic k points would silently introduce new alphas.
    """
    bindings = alpha_k_bindings(reference, N, D)
    values = sorted({item["kSlice"] for item in bindings})
    bounded = any(item["k_floor_applied"] or item["N_cap_applied"]
                  for item in bindings)
    return values, bounded


def fixed_n1m_candidate(reference_winner, N, D):
    """Freeze the N1M winning alpha/coordinates and bind derived k and K.

    The score is unset because a value measured at N1M is not a score at N.
    None serializes the same way as MATLAB's NaN in the JSON schema.

    ``kSlice`` at N1M identifies alpha; it is not an absolute k floor. Iteration
    counts have the independent algorithmic lower bound K=0: a frozen offset
    that requests a negative count at small N therefore evaluates the unchanged
    initial model. The alpha and offset are never rewritten or snapped.
    """
    N, k_min = _positive_integer(N), minimum_k(D)
    if N < k_min:
        raise ValueError("N is below the feasible minimum k")
    reference_k = matlab_round(reference_winner["kSlice"])
    binding = _alpha_k_binding(reference_k, N, D)
    effective_k = binding["kSlice"]
    offset = _nullable_offset(reference_winner["K_offset"])
    K, _, _ = bind_iteration_count(offset, N)
    candidate = dict(validation_relMSE_total=None, kSlice=effective_k,
        alpha=binding["alpha"], alpha_candidates=[binding["alpha"]],
        alpha_identifiable=True, reference_kSlice=reference_k,
        raw_kSlice=binding["raw_kSlice"], k_floor=k_min,
        k_floor_applied=binding["k_floor_applied"],
        N_cap_applied=binding["N_cap_applied"], K_offset=offset, K=K)
    for name in ("h_scale", "rho_init", "bite", "eval_rho", "eval_h"):
        candidate[name] = _number(reference_winner[name])
    return candidate, binding["k_floor_applied"] or binding["N_cap_applied"]


def bind_iteration_count(offset, N):
    """Return ``(effective_K, raw_K, bound)`` without changing the offset.

    This is capacity/feasibility arithmetic, not a parameter search.  K=0 is
    already the explicit initial-model branch in both CUDA and MATLAB.  The
    raw value is retained so downstream records distinguish an intentional
    K=0 candidate from a frozen offset that meets the hard lower bound.
    """
    N = _positive_integer(N)
    offset = _nullable_offset(offset)
    raw = 0 if offset is None else math.ceil(2 * math.log(N) + offset)
    return max(0, raw), raw, raw < 0


SELECTION_AXES = ("kSlice", "h_scale", "rho_init", "bite", "K_offset", "K", "eval_rho", "eval_h")


def select_winner(named_records):
    # MATLAB dir() is lexical; equal scores keep the first file, not last.
    _, record = min(sorted(named_records), key=lambda item: item[1]["best_validation_relMSE_total"])
    return dict(validation_relMSE_total=record["best_validation_relMSE_total"],
        kSlice=record["kSlice"], h_scale=record["h_scale"],
        rho_init=record["rho_init"], bite=record["bite"],
        K_offset=record["validation_selected_K_offset"],
        K=record["validation_selected_K"],
        eval_rho=record["validation_selected_eval_rho"],
        eval_h=record["validation_selected_eval_h"])


def boundary_flags(profile, winner):
    def endpoint(value, grid):
        values = list(dict.fromkeys(grid))
        return len(values) > 1 and any(abs(value - edge) <= 1e-10 * max(1, abs(edge))
                                     for edge in (values[0], values[-1]))
    k_grid = [max(profile["k_min"], value) for value in profile["k_floors"]]
    flags = {"kSlice": endpoint(winner["kSlice"], k_grid),
             "h_scale": endpoint(winner["h_scale"], profile["h_scales"]),
             "rho_init": endpoint(winner["rho_init"], profile["rho_inits"]),
             "bite": endpoint(winner["bite"], profile["bites"]),
             "K_offset": winner["K_offset"] is not None and math.isfinite(winner["K_offset"]) and endpoint(winner["K_offset"], profile["K_offsets"]),
             "eval_rho": endpoint(winner["eval_rho"], profile["eval_rhos"]),
             "eval_h": endpoint(winner["eval_h"], profile["eval_hs"])}
    hard = {"kSlice": winner["kSlice"] == profile["k_min"] == min(k_grid),
            "bite": winner["bite"] == 0 == min(profile["bites"])}
    for key, bound in hard.items():
        if bound:
            flags[key] = False
    return flags, hard


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path, value):
    """Keep a failed write's uniquely named temporary file as evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def worker_lock(root, recover_stale=False):
    """Kernel lock prevents duplicate workers, even after PID reuse/crashes.

    Never unlink the lock inode. A stale active owner record requires explicit
    recovery and is retained; no process is signalled or terminated.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".worker.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another worker owns this output directory") from error
        owner_file = root / "worker_owner.json"
        if owner_file.exists():
            previous = json.loads(owner_file.read_text())
            if previous.get("state") == "running":
                if not recover_stale:
                    raise RuntimeError("stale owner record: explicit recovery required")
                write_atomic(root / "lock_evidence" / (uuid.uuid4().hex + ".json"), previous)
        owner = {"pid": os.getpid(),
                 "state": "running", "started_at": now()}
        write_atomic(owner_file, owner)
        try:
            yield
        except BaseException:
            owner["state"] = "failed"
            raise
        else:
            owner["state"] = "complete"
        finally:
            owner["ended_at"] = now()
            write_atomic(owner_file, owner)


def winner(cube, reference):
    # MATLAB min(rel(:)): first tie in column-major order, not numpy C order.
    position = np.unravel_index(int(np.nanargmin(cube.ravel(order="F"))),
                                cube.shape, order="F")
    k, rho, h = position
    return {
        "best_validation_relMSE_total": float(cube[position]),
        "validation_selected_K_offset": reference["K_offsets"][k],
        "validation_selected_K": reference["K_grid"][k],
        "validation_selected_eval_rho": reference["eval_rho_grid"][rho],
        "validation_selected_eval_h": reference["eval_h_grid"][h],
    }
