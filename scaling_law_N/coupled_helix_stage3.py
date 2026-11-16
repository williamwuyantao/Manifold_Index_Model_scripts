"""Run the coupled-helix fixed-seven Stage 3 curve on a large H5 dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

from cuda_core_data import Stage2Data
import stage2_cuda_core as stage2
import stage3_cuda_core as stage3
from calibration_protocol import fixed_n1m_candidate
from fixed_hyperparameter_screen_cuda import candidate_at_n, normalize_parameters
from stage2_memory import MemoryPolicy


MAP = "natural_exponential3"
SETUP = "coupled_helix_D20_" + MAP
MAPS = ("natural_polynomial3", "natural_rational3", "natural_exponential3")
N_VALUES = [2_000_000, 3_000_000, 4_000_000, 5_000_000, 7_000_000,
            10_000_000, 16_000_000]


def read(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_large(dataset, setup, *, maximum_training_count, include_testing=True):
    """Read only the largest requested prefix and the fixed query views."""
    with h5py.File(dataset, "r") as source:
        if source.attrs.get("state") != "complete":
            raise ValueError("complete coupled-helix large dataset required")
        identity = json.loads(source.attrs["identity"])
        if (identity.get("schema") != "mfm-coupled-helix-data.v1"
                or identity.get("ambient_dimension") != 20
                or identity.get("intrinsic_dimension") != 2
                or identity.get("relative_y_noise") != .01
                or identity.get("train_count", 0) < maximum_training_count):
            raise ValueError("large dataset identity or training prefix is incompatible")
        arrays = {
            "X": source["geometry/train/x_noisy"][:maximum_training_count],
            "Y": source["maps/" + MAP + "/train/observed"][:maximum_training_count],
            "Xvalid": source["geometry/validation/x_noisy"][:],
            "Yvalid": source["maps/" + MAP + "/validation/observed"][:],
            "Tvalid": source["geometry/validation/tangent"][:],
        }
        if include_testing:
            arrays.update({
                "Xtest": source["geometry/testing/x_noisy"][:],
                "Ytest": source["maps/" + MAP + "/testing/clean"][:],
                "Ttest": source["geometry/testing/tangent"][:],
            })
    arrays = {name: np.ascontiguousarray(value, dtype=np.float32)
              for name, value in arrays.items()}
    if (arrays["X"].shape != (maximum_training_count, 20)
            or arrays["Y"].shape != (maximum_training_count, 3)
            or arrays["Xvalid"].shape != (5000, 20)
            or arrays["Yvalid"].shape != (5000, 3)
            or arrays["Tvalid"].shape != (5000, 2, 20)):
        raise ValueError("unexpected coupled-helix large dataset shapes")
    for value in arrays.values():
        if not np.isfinite(value).all():
            raise ValueError("nonfinite coupled-helix large dataset array")
    identity = dict(identity, dataset_file_sha256=file_sha(dataset),
                    loaded_training_count=maximum_training_count,
                    stage3_map=MAP)
    return Stage2Data(identity, arrays)


def make_configuration(calibration, selection, selection_sha, dataset):
    parameters = normalize_parameters(selection["parameters"])
    # The fixed-seven selection stores alpha, whereas the binding
    # helper expects an explicit N1M kSlice. Bind alpha at the anchor N
    # without changing any of the seven selected values.
    candidate, _ = candidate_at_n(parameters, 1_000_000, 20)
    return dict(
        schema="mfm-stage3-fixed-seven-selection.v1",
        status="selected_fixed_seven",
        setup=dict(id=SETUP, D=20, version="coupled_helix",
                   map_name=MAP, dataset=str(Path(dataset).resolve())),
        fixed_parameters=parameters,
        validation_winner=candidate,
        parameter_selection=dict(
            source_sha256=selection_sha,
            selection_data="validation only",
            testing_used_for_selection=False,
            origin="combined_cross_N_fixed_seven",
            binding_reference_N=1_000_000,
            Stage1_recalibrated=False),
        calibration_source_identity=calibration["identity"],
    )


def main(argv=None):
    global MAP, SETUP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--runtime-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase1-index-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-values", default=','.join(map(str, N_VALUES)))
    parser.add_argument("--map", choices=MAPS, default=MAP)
    args = parser.parse_args(argv)
    MAP, SETUP = args.map, "coupled_helix_D20_" + args.map
    ns = [int(value) for value in args.n_values.split(',')]
    if ns != sorted(set(ns)) or any(n <= 1_000_000 or n > 20_000_000 for n in ns):
        parser.error("N values must be increasing, distinct, and in (1M,20M]")
    calibration = read(args.calibration)
    if (calibration.get("status") != "calibrated_bracketed"
            or calibration.get("identity", {}).get("map") != MAP):
        raise ValueError("completed coupled-helix Stage 1 calibration required")
    selection_raw = read(args.selection)
    reports = selection_raw.get("reports", [])
    report = next((item for item in reports if item.get("setup_id") == SETUP), None)
    if report is None or selection_raw.get("testing_used") is not False:
        raise ValueError("validation-only fixed-seven selection for coupled helix required")
    winner = report["combined_winner"]
    selection_sha = file_sha(args.selection)
    selection = dict(parameters=winner["parameters"],
                     anchor_validation=winner["validation_relMSE_total_by_N"])
    runtime = stage2.validate_runtime(args.library, args.runtime_json)
    if runtime.get("production_start_allowed") is not True:
        raise ValueError("qualified production runtime required")
    runtime = dict(runtime, coupled_helix_stage3_entry_sha256=file_sha(Path(__file__)))
    args.phase1_index_dir.mkdir(parents=True, exist_ok=True)
    configuration = make_configuration(calibration, selection, selection_sha, args.dataset)
    policy = MemoryPolicy(phase1_mode=4, phase1_knn_scratch_bytes=0,
                          phase1_index_heap_bytes=0,
                          phase1_spill_directory=None)
    result = stage3.run_fixed_setup(
        configuration, dataset=args.dataset, output=args.output,
        library=args.library, runtime=runtime, n_values=ns, resume=args.resume,
        memory_policy=policy, data_loader=load_large)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
