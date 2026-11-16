# The Manifold-Index Model: provably and efficiently avoiding the statistical and computational curse of dimensionality

## Authors and contact

| Author | Affiliation | Email |
|---|---|---|
| Yantao Wu | Department of Applied Mathematics & Statistics, Johns Hopkins University | [ywu212@jhu.edu](mailto:ywu212@jhu.edu) |
| Mauro Maggioni | Department of Mathematics and Department of Applied Mathematics & Statistics, Johns Hopkins University | [mauromaggionijhu@icloud.com](mailto:mauromaggionijhu@icloud.com) |

Johns Hopkins University, Baltimore, MD 21218, USA.

## Abstract

We study vector-valued regression with predictors filling a tube around an unknown $m$-dimensional manifold $\mathcal{M}$ in $\mathbb{R}^D$. In the Manifold-Index Model, the mean response is $F=f\circ\Pi_\mathcal{M}$, where $\Pi_\mathcal{M}$ is the nearest-point projection onto $\mathcal{M}$ and $f:\mathcal{M}\to\mathbb{R}^p$ is a possibly non-injective link function. Multiscale ANisotropic lOcal Regression (MANOR) estimates tangent planes by multiscale PCA of predictors with similar responses, then refines them by pooling outer products of local slopes. The neighborhoods narrow tangentially while retaining complete normal fibers. We assume a Lipschitz link differential and bounded response noise. For $p\ge m$, under our geometric and sampling conditions, $n$ independent observations give mean tangent-space error of order $n^{-2/(m+4)}$ and integrated squared prediction error at the $m$-dimensional nonparametric rate $n^{-4/(m+4)}$, up to logarithmic factors. In a scaling regime with fixed intrinsic dimension, model and scale parameters, and ratio of manifold volume to the $m$-th power of its reach, the required sample size grows polynomially in $D+p$, and computation is polynomial in $n$, $D$ and $p$. Numerical experiments on curves with non-monotone links and surfaces with vector-valued responses compare observed convergence rates with the theoretical bounds; a Langevin dynamics example tests prediction of hitting probabilities under an approximate manifold-index model.

## Repository contents

Python/CUDA and MATLAB implementations of MANOR, experiment configurations,
measurements, and figure sources accompanying the paper.

| Directory | Purpose |
|---|---|
| `manor_m/` | Estimator, exact memory-aware tiling, single-GPU and multi-GPU backends |
| `dataset_generator/` | The paper's curve, bumpy-sphere, coupled-helix and Monte Carlo generators |
| `scaling_law_N/` | Parameter selection, fixed-parameter curves and IID repetitions |
| `experiments/` | One-dimensional experiments, Phase 2 analysis and reaction-coordinate evaluation |
| `config/` | Geometry definitions, fixed parameters and figure settings |
| `results/` | Individual scientific measurements, grouped by experiment |
| `data/` | Small numerical inputs needed to reproduce geometry and response figures |
| `figures/` | Plotting scripts and the corresponding PDFs |
| `scripts/` | Build and repository verification commands |
| `dataset/` | Full predictor datasets stored outside Git |

[FILES.md](FILES.md) describes every included file and the dataset layout.

## Reproduce figures

```bash
python -m pip install -r requirements.txt
python figures/reproduce.py all
python scripts/verify.py
```

Individual targets are `bumpy`, `phase2`, `coupled`, `global-tube`,
`bumpy-maps`, `bumpy-geometry`, `reaction`, `diagrams`, and `curves`.
Outputs go to `reproduced/`. The `diagrams` target (included in `all`) requires
pdfLaTeX with TikZ, AMS fonts and graphicx. It compiles in a temporary directory.
[figures/README.md](figures/README.md) maps the paper figures to their sources.
The six one-dimensional plots and their displayed values are included;
`curves` composes their prediction and tangent-space panels. To plot fresh curve
experiments, use `experiments/curves/plot_curves.m`.

The current bumpy-sphere and coupled-helix curves use ten independent IID fits
at every displayed sample size. Error bars are sample standard deviations.
For the global tubes, only sample sizes with all ten repetitions enter a
mean-and-SD curve; all individual measurements remain available in the result files. `config/record_counts.json` lists the included counts.
`selection/` contains the parameter-selection measurements and single-run
learning curves associated with those parameter choices.

Phase 2 uses the local-reach bumpy-sphere construction at 500,000, 700,000
and 1,000,000 samples, with ten independent repetitions per size. The global
tube experiment uses separate noisy predictors for each radius ratio.

## Build and run

```bash
bash scripts/build_core.sh
```

The default targets the visible GPU (`CUDA_ARCH=native`, CMake 3.24 or newer).
Set `CUDA_ARCH` explicitly when building without a GPU, or for a different
machine; for example, `CUDA_ARCH=90-real` targets an H100. Set `NVCC` for a
different CUDA toolkit. For compute
capability 12.0, use `CUDA_ARCH=120-real` with a compatible compiler.
`config/build.json` records the numerical build options. The library is
`build/libmanor_m_cuda.so`; `BUILD_DIR` changes the build location.
The build also creates `build/runtime.json`, which records the loaded ABI,
compiled metric conventions and library checksum.
Dynamic GNU runtime linking is the default. Set `STATIC_GNU_RUNTIME=ON` only
when the compiler's static `libstdc++` archive is installed.
Python execution does not require MATLAB. The shared CUDA core includes the
algorithm sources under `manor_m/matlab/cuda_mex/`.

`CudaCoreEstimator` in `manor_m/cuda/python/estimator.py` provides the
single-GPU interface. Its capacity planner selects resident or exact tiled
execution from the memory budget and the required array sizes. Phase 1 state is reused within the same dataset and
exact neighbor configuration. Different datasets and sample sizes have
independent state.

For one large fit across several GPUs, see
[manor_m/cuda/multi_gpu/README.md](manor_m/cuda/multi_gpu/README.md).
The GPUs share one exact host kNN table and exchange model states at the
algorithm's synchronization boundaries. The optional Gram cache and
persistent readout buffers are part of this backend.

## Fresh IID experiments

Run from the repository root after building the CUDA library:

```bash
python scaling_law_N/bumpy_iid_repetitions.py prepare
python scaling_law_N/bumpy_iid_repetitions.py worker --check-only
python scaling_law_N/bumpy_iid_repetitions.py worker --n-values 10000 --repetitions 1
python scaling_law_N/bumpy_iid_repetitions.py summary

python scaling_law_N/coupled_helix_iid_repetitions.py prepare
python scaling_law_N/coupled_helix_iid_repetitions.py worker --check-only
python scaling_law_N/coupled_helix_iid_repetitions.py worker --n-values 10000 --repetitions 1
python scaling_law_N/coupled_helix_iid_repetitions.py summary
```

Omit `--n-values` and `--repetitions` to run the full paper grid. Each worker
generates its inputs from the included geometry calibration and fixed
parameters.
The default run folders are `reproduced/bumpy_iid` and `reproduced/coupled_iid`.
Use `--root` to choose another location. For another library build, pass
`--library path/to/libmanor_m_cuda.so`; its adjacent `runtime.json` is used
unless `--runtime-json` is specified. `--check-only` reads the configuration
and validates the library without generating data or fitting.

Use `CUDA_VISIBLE_DEVICES` to select the GPU. Independent workers divide
the sample-size/repetition grid with `--lane` and `--lanes`; they share the same
run root and use per-repetition locks. The default working-memory budget is
24 GiB, adjustable through `--working-gib`. Runs resume completed results only
when their parameters, sampling plan and library checksum match.

## Datasets and optional MATLAB interface

Generator commands expose their configuration through `--help`. The
coupled-helix generator is self-contained with the supplied calibration:

```bash
python dataset_generator/coupled_helix_generator.py \
  --n 1000000 --queries 5000 --output dataset/coupled_helix/example.h5
```

Stored results contain the measurements, fixed parameters and generator seeds. Full training
arrays are listed in [DATASETS.md](DATASETS.md), with commands for checking,
retrieving and regenerating them. `SHA256SUMS` covers every
shared source, figure and scientific data file.

The optional multidimensional MATLAB adapter is included. After the core build:

```bash
cmake -S manor_m/cuda -B build -DMANOR_M_BUILD_MATLAB=ON
cmake --build build --target manor_m_cuda_mex
```

This requires MATLAB headers and a supported C++ compiler. The one-dimensional
curve estimator has its own CUDA-MEX backend and build command in
[experiments/curves/README.md](experiments/curves/README.md).

## Paper figure sources

The manuscript-specific geometry diagrams are under `figures/paper_diagrams/`.
The scaling figures used in the manuscript are rendered with the common two-panel
layout in `figures/bumpy_rates.py`: the left panel overlays prediction errors and
the right panel overlays tangent-space errors.
