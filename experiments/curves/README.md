# One-dimensional curve experiments

`config/curve_experiments.json` gives the manuscript's settings for the
single-index model and Meyer helix in dimensions 18, 24 and 30. Each experiment
uses three repetitions and a 90% training / 10% testing split.

The one-dimensional implementation is in `backend/`: MATLAB
orchestration and three CUDA-MEX kernels. It is separate from the multidimensional
Python + CUDA estimator in `manor_m/`.

For a fresh run in MATLAB:

```matlab
addpath experiments/curves/backend
build_manor_1_mex
addpath experiments/curves
run_curves('single_index',18)
% For Meyer, retrieve a retained pool (DATASETS.md), or generate a fresh one:
addpath dataset_generator/curves
generate_meyer_pool(18)
run_curves('meyer_helix',18)
plot_curves('single_index',18)
plot_curves('meyer_helix',18)
```

Results go to `reproduced/curves/`. Single-index predictors are generated
directly in each repetition; no input dataset is required. Fresh predictor
realizations are saved alongside the results, together with the random states
and Meyer sample indices. `plot_curves.m` plots prediction and mean tangent-space errors
with means and sample standard deviations across repetitions.

## Retained paper material

The six learning-curve PDFs are in `figures/pdf/curves/`. Their displayed
means and error-bar endpoints are read from vector coordinates in
`results/curves/published_curves.json`.
The two-million-point Meyer predictor pools, including clean points,
tangents, latent coordinates and geometry, are stored under
`dataset/curves/meyer_helix/`. The manuscript's constant-noise predictor array is
`gamma/X_UniCstDevSig`.
