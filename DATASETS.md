# Data used by the paper

The Git repository contains the measurements and the small geometric inputs
needed to redraw the paper figures. Full training arrays live outside Git.
`config/dataset_inventory.json` records 350 retained HDF5 files, with each
array's shape, dtype and SHA-256 checksum. Paths are relative to `dataset/`.

```bash
python scripts/datasets.py list --match 'curves/meyer_helix/*'
python scripts/datasets.py check --root /path/to/dataset --match 'curves/meyer_helix/*'
python scripts/datasets.py check --root /path/to/dataset --match 'curves/meyer_helix/*' --hash
```

The default check reads array metadata; `--hash` also reads every array byte.
The manifest refers to the retained realizations. Fresh draws from the
generators have different checksums.

## Retrieve a retained dataset

Copy from an existing dataset collection, preserving its relative paths:

```bash
python scripts/datasets.py fetch --source /path/to/full/dataset \
  --match 'curves/meyer_helix/*'
```

For a hosted collection with the same layout, replace `--source` with
`--base-url https://your-data-host/path/to/dataset/`. Downloads and copies are checked
against the array hashes before becoming final files. Existing files are checked
and left in place.

## Generate fresh inputs

| Experiment | Input generation |
|---|---|
| Single-index curves | `experiments/curves/run_curves.m` generates predictors in every repetition. |
| Meyer helix curves | `dataset_generator/curves/generate_meyer_pool.m` produces the HDF5 pool consumed by `run_curves.m`. |
| Bumpy-sphere IID curves | `scaling_law_N/bumpy_iid_repetitions.py prepare` and `worker`, using `config/geometry_recipe.npz` and `config/bumpy_experiment.json`. |
| Coupled-helix IID curves | `scaling_law_N/coupled_helix_iid_repetitions.py prepare` and `worker`, using `data/coupled_helix/` and `config/coupled_experiment.json`. |
| Fixed global tubes | `scaling_law_N/bumpy_global_tube_experiment.py` calls `dataset_generator/bumpy_global_tube.py`; settings are in `config/global_tube_experiment.json`. |
| Reaction coordinates | `dataset_generator/reaction_coordinate_generator/run.py`, using `config/reaction_generator.json`. |

The bumpy and coupled IID workers generate training, validation and testing
arrays without downloading the retained datasets. Their sampling seeds and
selected parameters are included. The full runs can be large; `--n-values`
and `--repetitions` select part of the paper grid.

To generate a new two-million-point Meyer pool in MATLAB:

```matlab
addpath dataset_generator/curves
generate_meyer_pool(18)
generate_meyer_pool(24)
generate_meyer_pool(30)
```

This uses the same curve formula, estimated-reach normalization and constant
normal-displacement scale as the retained pools. It needs the Statistics and
Parallel Computing Toolboxes. The optional arguments are `(D,N,output_file,seed)`.
An existing output is never overwritten.

## curve measurements

The six vector figures, displayed means and error-bar endpoints are retained in
`figures/pdf/curves/` and `results/curves/published_curves.json`. Fresh runs save
individual observations and random states.
