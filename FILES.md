# File guide

The shared repository contains the current estimator, the selected paper experiments,
their figure inputs and scientific results. Numerical selection records are kept
only for the explicitly selected bumpy and coupled-helix examples. They are data;
the active plotting commands use the current IID results.

Every shared file is listed below. Files under `manor_m/matlab/cuda_mex/` are
algorithm sources included by the standalone Python backend. The independent
one-dimensional examples retain their MATLAB/CUDA implementation.

## root

| File | Purpose |
|---|---|
| `DATASETS.md` | Retained dataset manifest, retrieval commands and fresh-data generation. |
| `.gitignore` | Excludes datasets, compiled objects, caches and local runtime output. |
| `FILES.md` | Complete file guide and collaborator dataset layout. |
| `README.md` | Main entry: layout, builds, figure reproduction and estimator use. |
| `SHA256SUMS` | Content checksums for every shared file except this checksum list. |
| `requirements.txt` | Python packages required by the runtime and figure scripts. |

## config

| File | Purpose |
|---|---|
| `config/dataset_inventory.json` | Array shapes, dtypes and hashes for the 350 retained full HDF5 datasets. |
| `config/build.json` | Numerical compiler and optimization options used by the build script. |
| `config/bumpy_experiment.json` | Fresh IID sampling rules and displayed sample sizes. |
| `config/bumpy_geometry.json` | Complete bumpy-sphere shape, scales and normal-noise definition. |
| `config/bumpy_parameter_selection.json` | Cross-N validation scores that selected the two fixed configurations. |
| `config/bumpy_parameters.json` | The two fixed seven-parameter configurations for bumpy Y0 and Y2. |
| `config/coupled_experiment.json` | IID grid, all three fixed configurations, response normalization and A/B/C readout policies. |
| `config/coupled_geometry.json` | Surface projection, intrinsic coloring and selected view parameters. |
| `config/curve_experiments.json` | Paper parameters and sample sizes for the six one-dimensional examples. |
| `config/figure_style.json` | Panel sizes used by the Phase 2 figure renderer. |
| `config/geometry_recipe.npz` | Numerical bumpy-sphere geometry coefficients used for exact data regeneration. |
| `config/global_tube_experiment.json` | Global-radius ratios, independent predictor sampling rules and IID sample sizes. |
| `config/paper_scope.json` | The selected experiments and their relation to the paper figures. |
| `config/parula.json` | Fixed color lookup table for the bumpy response-component panels. |
| `config/phase2_experiment.json` | Selected local-reach experiment, iteration grid and three-projection protocol. |
| `config/reaction_generator.json` | Complete D20 Monte Carlo dynamics and first-hit target configuration. |
| `config/reaction_parameters.json` | Selected fixed seven parameters and evaluation size for the reaction example. |
| `config/record_counts.json` | Number of measured records included in each scientific collection. |

## data

| File | Purpose |
|---|---|
| `data/coupled_helix/response_fields.npz` | Shared intrinsic grid and all three components of the A/B/C response maps. |
| `data/coupled_helix/response_summary.json` | Selected map definitions and physical Jacobian/reach measurements. |
| `data/coupled_helix/samples.npz` | 100,000 noisy predictors, intrinsic coordinates, calibrated local reach and noisy-X PCA basis. |
| `data/coupled_helix/summary.json` | Surface scale and geometric calibration measurements. |
| `data/reaction_samples.npz` | The 60,000 sphere coordinates and Monte Carlo response values displayed in the panels. |

## dataset_generator

| File | Purpose |
|---|---|
| `dataset_generator/curves/generate_meyer_pool.m` | Generate a new Meyer-helix HDF5 predictor pool with the paper curve and normal-displacement scale. |
| `dataset_generator/bumpy_global_tube.py` | Separate predictor datasets with fixed global-reach tube radii. |
| `dataset_generator/bumpy_sphere_generator.py` | Bumpy-sphere geometry, normal noise and response sampling. |
| `dataset_generator/coupled_helix_generator.py` | Selected 5-by-4 coupled helix, local normal noise and the three current response maps. |
| `dataset_generator/curves/compute_reach.m` | Curve local-reach computation used by the generator. |
| `dataset_generator/curves/createCurve.m` | Meyer-helix coordinate construction. |
| `dataset_generator/curves/generate_single_index.m` | IID predictor and response draws for the single-index experiment. |
| `dataset_generator/curves/normcols.m` | Column normalization used by the curve generator. |
| `dataset_generator/reaction_coordinate_generator/__init__.py` | Reaction-coordinate generator package entry. |
| `dataset_generator/reaction_coordinate_generator/cuda/reaction_coordinate_mc_em.cu` | Euler-Maruyama Monte Carlo simulation and first-hit response estimation. |
| `dataset_generator/reaction_coordinate_generator/run.py` | Python entry for the reaction-coordinate Monte Carlo generator. |

## experiments

| File | Purpose |
|---|---|
| `experiments/curves/README.md` | curve computation, plotting and input conventions. |
| `experiments/curves/backend/build_manor_1_mex.m` | Compile the one-dimensional MATLAB/CUDA backend. |
| `experiments/curves/backend/manor_1_eval_gpu.m` | One-dimensional prediction interface. |
| `experiments/curves/backend/manor_1_eval_mex.cu` | CUDA kernel-weighted prediction for the one-dimensional examples. |
| `experiments/curves/backend/manor_1_gpu.m` | one-dimensional estimator interface. |
| `experiments/curves/backend/manor_1_performance.m` | One-dimensional error measurements used by the curve scripts. |
| `experiments/curves/backend/manor_1_phase1_cuda_mex.cu` | One-dimensional Phase 1 numerical kernel. |
| `experiments/curves/backend/manor_1_phase2_cuda_mex.cu` | One-dimensional Phase 2 update kernel. |
| `experiments/curves/plot_curves.m` | Draw the six curve-experiment learning figures. |
| `experiments/curves/run_curves.m` | Compute the single-index and Meyer-helix experiments. |
| `experiments/phase2/run.py` | Python entry for the reaction-coordinate Monte Carlo generator. |
| `experiments/reaction_coordinate/run_fixed_curve_single_cuda.py` | Fixed-sample evaluation of the selected Monte Carlo reaction dataset. |
| `experiments/reaction_coordinate/run_mc_fixed7_screen_cuda.py` | Validation-only selection of the reaction example parameters. |

## figures

| File | Purpose |
|---|---|
| `figures/README.md` | Map the current paper figures to rendering targets and source files. |
| `figures/curves.py` | Compose the six retained learning curves into the paper prediction/tangent panels. |
| `figures/diagrams.py` | Rebuild geometry and compile the five manuscript TikZ diagrams in temporary directories. |
| `figures/pdf/Coupled_Meyer_Helix_D20_rational_response_components.pdf` | Current rendered figure: Coupled Meyer Helix D20 rational response components. |
| `figures/pdf/Coupled_Meyer_Helix_D20_response_components.pdf` | Current rendered figure: Coupled Meyer Helix D20 response components. |
| `figures/pdf/curve_link.pdf` | Current rendered figure: curve link. |
| `figures/pdf/local_cylinder.pdf` | Current rendered figure: local cylinder. |
| `figures/pdf/meyer_helix_paper_panels.pdf` | Current rendered figure: meyer helix paper panels. |
| `figures/pdf/phase1_scan.pdf` | Current rendered figure: phase1 scan. |
| `figures/pdf/phase2_refinement.pdf` | Current rendered figure: phase2 refinement. |
| `figures/pdf/single_index_paper_panels.pdf` | Current rendered figure: single index paper panels. |
| `figures/pdf/theory_roadmap.pdf` | Current rendered figure: theory roadmap. |
| `figures/bumpy_maps.py` | Six response components on sphere longitude/latitude coordinates. |
| `figures/bumpy_rates.py` | Common two-panel mean/SD learning curves and log-log regression. |
| `figures/coupled.py` | Selected clean-surface projections and each response-map component. |
| `figures/geometry.py` | Bumpy-sphere geometry reconstruction and selected surface view. |
| `figures/pdf/bumpy_geometry.pdf` | Current vector figure: bumpy geometry. |
| `figures/pdf/bumpy_linear_components.pdf` | Current vector figure: bumpy linear components. |
| `figures/pdf/bumpy_quadratic_components.pdf` | Current vector figure: bumpy quadratic components. |
| `figures/pdf/Bumpy_Sphere_D20_scaling_IID10.pdf` | Current two-panel bumpy-sphere scaling figure used by the manuscript. |
| `figures/pdf/Coupled_Meyer_Helix_D20_AB_scaling_IID10.pdf` | Current two-panel coupled-helix A/B scaling figure used by the manuscript. |
| `figures/pdf/coupled_helix_geometry.pdf` | Current vector figure: coupled helix geometry. |
| `figures/pdf/MFM_model_geometry.pdf` | Current vector figure: Figure 1 geometry diagram. |
| `figures/pdf/MFM_m2_preimage_slabs.pdf` | Current vector figure: Figure 2 response-preimage slabs. |
| `figures/pdf/curves/meyer_helix_D18.pdf` | Current vector figure: meyer helix D18. |
| `figures/pdf/curves/meyer_helix_D24.pdf` | Current vector figure: meyer helix D24. |
| `figures/pdf/curves/meyer_helix_D30.pdf` | Current vector figure: meyer helix D30. |
| `figures/pdf/curves/single_index_D18.pdf` | Current vector figure: single index D18. |
| `figures/pdf/curves/single_index_D24.pdf` | Current vector figure: single index D24. |
| `figures/pdf/curves/single_index_D30.pdf` | Current vector figure: single index D30. |
| `figures/pdf/global_tube_c025_scaling.pdf` | Current vector figure: global tube c025 scaling. |
| `figures/pdf/global_tube_c050_scaling.pdf` | Current vector figure: global tube c050 scaling. |
| `figures/pdf/global_tube_c075_scaling.pdf` | Current vector figure: global tube c075 scaling. |
| `figures/pdf/phase2_iterations.pdf` | Current vector figure: phase2 iterations. |
| `figures/pdf/phase2_projections.pdf` | Current vector figure: phase2 projections. |
| `figures/pdf/reaction_components.pdf` | Current vector figure: reaction components. |
| `figures/pdf/reaction_geometry.pdf` | Current vector figure: reaction geometry. |
| `figures/phase2.py` | Per-iteration and three-projection figure layout. |
| `figures/reaction.py` | Sphere/target geometry, potential and Monte Carlo response-component panels. |
| `figures/reproduce.py` | Recompute ensemble statistics and dispatch every current paper figure. |

## figures/paper_diagrams

| File | Purpose |
|---|---|
| `figures/paper_diagrams/curve_link.tex` | The one-dimensional link curve from the manuscript. |
| `figures/paper_diagrams/local_cylinder.tex` | Local-cylinder geometry and selection diagram. |
| `figures/paper_diagrams/phase1_scan.tex` | Complete Phase 1 diagram with response packets, three radii, spectrum and radius score. |
| `figures/paper_diagrams/phase2_refinement.tex` | Current warm-start and two anisotropic-refinement neighborhoods. |
| `figures/paper_diagrams/source/MFM_m2_diameter_label.pdf` | Vector typesetting of the tube-diameter annotation. |
| `figures/paper_diagrams/source/MFM_m2_geometry_base.pdf` | Retained vector geometry for the cylindrical-patch reconstruction. |
| `figures/paper_diagrams/source/MFM_m2_geometry_outline.json` | Manifold patch coordinates for the Figure 1 reconstruction. |
| `figures/paper_diagrams/source/MFM_m2_tube_outline.json` | Tube boundary coordinates for the Figure 1 reconstruction. |
| `figures/paper_diagrams/theory_roadmap.tex` | Theorem/proposition dependency diagram with manuscript numbering. |
| `figures/paper_diagrams/update_m2_geometry_tube.py` | Reconstruct the cylindrical patch, tube and diameter annotation from retained vector assets. |
| `figures/paper_diagrams/model_geometry.py` | Rebuild Figure 1 from the retained vector base geometry. |
| `figures/paper_diagrams/response_preimage_slabs.py` | Rebuild the response-preimage slabs used in Figure 2. |
| `figures/paper_diagrams/MFM_m2_geometry.pdf` | Retained vector base geometry consumed by `model_geometry.py`. |

## manor_m

| File | Purpose |
|---|---|
| `manor_m/cuda/matlab/manor_m_cuda_mex.cpp` | Optional MATLAB adapter to the current shared CUDA core. |
| `manor_m/cuda/CMakeLists.txt` | Build definition for the standalone shared CUDA core. |
| `manor_m/cuda/core/manor_m_aggregation_gather.cuh` | Warp aggregation gather helper used by the configured kernel paths. |
| `manor_m/cuda/core/manor_m_aggregation_tiles.h` | Capacity and indexing rules for bounded owner aggregation tiles. |
| `manor_m/cuda/core/manor_m_aggregation_warp.cuh` | Covariance accumulation across warp lanes in unchanged neighbor order. |
| `manor_m/cuda/core/manor_m_cuda.cu` | CUDA library implementation and exported entry points. |
| `manor_m/cuda/core/manor_m_cuda.h` | CUDA library implementation and exported entry points. |
| `manor_m/cuda/core/manor_m_cuda_internal.h` | Internal CUDA state and allocation interfaces. |
| `manor_m/cuda/core/manor_m_cuda_resident.h` | Resident session and reusable device-buffer API. |
| `manor_m/cuda/core/manor_m_devtan_orthonormal.cuh` | FP64 orthonormal-basis tangent error evaluation. |
| `manor_m/cuda/core/manor_m_eval_contribution_audit.cuh` | Optional per-query contribution capture for numerical verification. |
| `manor_m/cuda/core/manor_m_eval_fixed_order.cuh` | Deterministic fixed-order metric reduction used by production. |
| `manor_m/cuda/core/manor_m_generated_phase2.cuh` | Exact generated-index Phase 2 execution when full neighbor storage does not fit. |
| `manor_m/cuda/core/manor_m_index_stream.cuh` | Bounded streaming of complete exact neighbor rows. |
| `manor_m/cuda/core/manor_m_index_tiles.h` | Index-tile size and memory planning helpers. |
| `manor_m/cuda/core/manor_m_index_validation.h` | Full integer index validation with vectorizable host scanning. |
| `manor_m/cuda/core/manor_m_knn_ties.cuh` | Deterministic tie ordering for exact neighbor selection. |
| `manor_m/cuda/core/manor_m_minimex.h` | Shared algorithm-source adapter used by the standalone CUDA library. |
| `manor_m/cuda/core/manor_m_phase1_capacity.cuh` | Initialization capacity estimates and tile selection. |
| `manor_m/cuda/core/manor_m_phase1_index_stream.cuh` | Phase 1 streamed exact-index path. |
| `manor_m/cuda/core/manor_m_phase1_port.cu` | Standalone shared-library adapter for the Phase 1 algorithm source. |
| `manor_m/cuda/core/manor_m_phase1_query_local_mask.cuh` | Query-local mask representation for bounded initialization memory. |
| `manor_m/cuda/core/manor_m_phase2_port.cu` | Standalone shared-library adapter for the Phase 2 algorithm source. |
| `manor_m/cuda/core/manor_m_regression_cooperative.cuh` | Cooperative small-system regression solver. |
| `manor_m/cuda/core/manor_m_regression_shuffle.cuh` | Shuffle-based regression input sharing. |
| `manor_m/cuda/core/manor_m_regression_warp_entries.cuh` | Warp-distributed regression matrix entries. |
| `manor_m/cuda/multi_gpu/CMakeLists.txt` | Build definition for the optional multi-GPU backend. |
| `manor_m/cuda/multi_gpu/README.md` | Multi-GPU numerical contract, build command and API usage. |
| `manor_m/cuda/multi_gpu/phase1_multi.cuh` | Initialization kernels for disjoint anchor ranges and exact owner reduction. |
| `manor_m/cuda/multi_gpu/phase1_source.py` | Generate the multi-GPU initialization wrapper from the shared numerical kernels. |
| `manor_m/cuda/multi_gpu/phase2.cu` | Native multi-GPU state exchange, regression, aggregation and shared readout. |
| `manor_m/cuda/multi_gpu/phase2.h` | C interface and buffer contracts for the multi-GPU backend. |
| `manor_m/cuda/multi_gpu/readout_source.py` | Generate the multi-GPU ordered readout kernels from the shared evaluation source. |
| `manor_m/cuda/python/array_state_hash.py` | Full-array hashes for numerical state verification, without temporary array copies. |
| `manor_m/cuda/python/calibration_protocol.py` | Seven-axis search profiles, iteration counts, winner selection and result writing. |
| `manor_m/cuda/python/estimator.py` | Single-GPU estimator: memory planning, exact initialization reuse and evaluation. |
| `manor_m/cuda/python/manor_m_cuda.py` | Typed Python binding to the CUDA shared library. |
| `manor_m/cuda/python/multi_gpu_estimator.py` | One-fit multi-GPU estimator with one shared exact kNN table. |
| `manor_m/cuda/python/phase1.py` | Phase 1 initialization, candidate cut positions and Phase 2 bandwidth schedules. |
| `manor_m/cuda/python/phase2_checkpoint.py` | Save and restore complete numerical states between bounded large fits. |
| `manor_m/cuda/python/resident_sweep.py` | Input sessions, device residency, exact-k state reuse and bounded host index residency. |
| `manor_m/cuda/python/stage2_memory.py` | GPU capacity model and bounded tiling policy. |
| `manor_m/cuda/tests/multi_gpu.py` | Same-data numerical checks for the single/multi-GPU initialization, updates and readout. |
| `manor_m/matlab/cuda_mex/manor_m_h_null_table.h` | Numerical null-reference table used in the Phase 1 anisotropic score. |
| `manor_m/matlab/cuda_mex/manor_m_phase1_mex.cu` | Shared numerical Phase 1 source compiled into the standalone core. |
| `manor_m/matlab/cuda_mex/manor_m_phase2_mex.cu` | Shared numerical Phase 2 and prediction source compiled into the standalone core. |

## results

| File | Purpose |
|---|---|
| `results/bumpy_sphere/iid.jsonl` | One scientific record per independent fixed-parameter fit; N, repetition, parameters and validation/testing errors. |
| `results/bumpy_sphere/selection/bumpy_sphere_v2_D20_Y0/stage1.jsonl` | validation calibration cubes and selected model. |
| `results/bumpy_sphere/selection/bumpy_sphere_v2_D20_Y0/stage2_curve.jsonl` | small-N fixed-parameter single-realization curve. |
| `results/bumpy_sphere/selection/bumpy_sphere_v2_D20_Y0/stage2_selection.jsonl` | Cross-N validation candidate measurements. |
| `results/bumpy_sphere/selection/bumpy_sphere_v2_D20_Y0/stage3_curve.jsonl` | large-N fixed-parameter single-realization curve. |
| `results/bumpy_sphere/selection/bumpy_sphere_v2_D20_Y2/stage1.jsonl` | validation calibration cubes and selected model. |
| `results/bumpy_sphere/selection/bumpy_sphere_v2_D20_Y2/stage2_curve.jsonl` | small-N fixed-parameter single-realization curve. |
| `results/bumpy_sphere/selection/bumpy_sphere_v2_D20_Y2/stage2_selection.jsonl` | Cross-N validation candidate measurements. |
| `results/bumpy_sphere/selection/bumpy_sphere_v2_D20_Y2/stage3_curve.jsonl` | large-N fixed-parameter single-realization curve. |
| `results/bumpy_sphere/selection/counts.json` | Record counts for the retained bumpy Stage 1/2/3 selection collection. |
| `results/bumpy_sphere/selection/scaling.json` | Preserved measured single-realization curves before the ten-IID experiment. |
| `results/coupled_helix/iid.jsonl` | One scientific record per independent fixed-parameter fit; N, repetition, parameters and validation/testing errors. |
| `results/coupled_helix/selection/scaling.json` | Preserved measured single-realization curves before the ten-IID experiment. |
| `results/curves/published_curves.json` | Displayed means and error-bar endpoints from the six curve figures; no invented individual repetitions. |
| `results/global_tube/c025/fixed_parameters.json` | Final cross-N validation selection and the fixed seven parameters for this tube ratio. |
| `results/global_tube/c025/iid.jsonl` | One scientific record per independent fixed-parameter fit; N, repetition, parameters and validation/testing errors. |
| `results/global_tube/c025/selection.jsonl` | Validation calibration and cross-N candidate measurements for this tube ratio. |
| `results/global_tube/c025/selection_summary.json` | Compact count and completeness summary of the parameter-selection records. |
| `results/global_tube/c050/fixed_parameters.json` | Final cross-N validation selection and the fixed seven parameters for this tube ratio. |
| `results/global_tube/c050/iid.jsonl` | One scientific record per independent fixed-parameter fit; N, repetition, parameters and validation/testing errors. |
| `results/global_tube/c050/selection.jsonl` | Validation calibration and cross-N candidate measurements for this tube ratio. |
| `results/global_tube/c050/selection_summary.json` | Compact count and completeness summary of the parameter-selection records. |
| `results/global_tube/c075/fixed_parameters.json` | Final cross-N validation selection and the fixed seven parameters for this tube ratio. |
| `results/global_tube/c075/iid.jsonl` | One scientific record per independent fixed-parameter fit; N, repetition, parameters and validation/testing errors. |
| `results/global_tube/c075/selection.jsonl` | Validation calibration and cross-N candidate measurements for this tube ratio. |
| `results/phase2/direct_states.jsonl` | Per-iteration tangent errors at fixed training owners for all 30 paired fits. |
| `results/phase2/parameter_selection.json` | Measured validation candidates behind the fixed Phase 2 configuration. |
| `results/phase2/plot_data.json` | Verified mean and sample-SD arrays for the current Phase 2 figures. |
| `results/phase2/readout.jsonl` | Per-iteration testing errors and initial/final/true-projection readouts for all 30 fits. |
| `results/reaction_coordinate/fixed_N.json` | Independent testing error and tangent error for the selected N=400,000 reaction example. |

## scaling_law_N

| File | Purpose |
|---|---|
| `scaling_law_N/analysis/phase2_measurements.py` | Per-iteration measurements and paired initial/final/true-tangent readouts. |
| `scaling_law_N/bumpy_global_tube_experiment.py` | Global-tube calibration and shared-trajectory validation/testing evaluation. |
| `scaling_law_N/bumpy_iid_repetitions.py` | Fresh IID bumpy-sphere draws and fixed-parameter repetitions. |
| `scaling_law_N/coupled_helix_iid_repetitions.py` | Fresh IID helix draws, ten fits per N, with the selected per-map readout policy. |
| `scaling_law_N/coupled_helix_stage1.py` | N1M seven-parameter validation calibration for the selected helix responses. |
| `scaling_law_N/coupled_helix_stage2.py` | Cross-N validation selection of one fixed seven-parameter helix configuration. |
| `scaling_law_N/coupled_helix_stage3.py` | Fixed-parameter large-N helix evaluation. |
| `scaling_law_N/cuda_core_data.py` | Load canonical arrays and response/tangent truths into the common experiment interface. |
| `scaling_law_N/fixed_hyperparameter_curve_cuda.py` | Evaluate the selected fixed configuration across sample sizes. |
| `scaling_law_N/fixed_hyperparameter_screen_cuda.py` | Evaluate and rank fixed seven-parameter candidates using validation data. |
| `scaling_law_N/global_tube_small_n_screen.py` | Cross-N validation screening for the fixed global-tube parameters. |
| `scaling_law_N/large_n_multi_gpu.py` | Application adapter and CLI for one large fit across several GPUs. |
| `scaling_law_N/stage2_cuda_core.py` | Small-N grids, validation selection, independent testing and metric normalization. |
| `scaling_law_N/stage3_checkpointed_cuda_core.py` | Large-N runner with complete-state iteration checkpoints. |
| `scaling_law_N/stage3_cuda_core.py` | Large-N model configuration and fixed-parameter runner. |
| `scaling_law_N/stage3_fixed7_cuda_core.py` | Bind the cross-N selected seven parameters to the large-N runner. |

## scripts

| File | Purpose |
|---|---|
| `scripts/datasets.py` | List, retrieve and verify full datasets against the published array manifest. |
| `scripts/write_runtime.py` | Inspect the built ABI and numerical conventions and record the library checksum. |
| `scripts/build_core.sh` | Build the current single-GPU library with the numerical options in config/build.json. |
| `scripts/verify.py` | Check shared-file hashes, imports, privacy and independent-repetition statistics. |

## Result fields and completeness

A line of an IID `.jsonl` file is one completed fit, with `N`, `repetition`,
`fixed_parameters`, the scientific dataset identity and separate `validation`
and `testing` values. Relative mean squared error uses the recorded response
variance normalization. `testing_devTan` is the mean tangent-space error.
Testing data never select parameters. Mean curves use all ten repetitions
and `ddof=1` sample standard deviations. Incomplete groups stay in the
individual data and do not enter the mean curve.

Bumpy Y0/Y2 and helix A/B/C each have 18 sample sizes and ten repetitions per
map. A uses the selected fallback exponent scale 4; B and C use scale 2.
Global-tube selection measurements are separate from the fresh IID fits.
Each tube ratio has distinct noisy predictors. The Phase 2 trajectory experiment
uses the local-reach construction, .

## Full datasets in the collaborator copy


`dataset/curves/meyer_helix/D18.h5`, `D24.h5` and `D30.h5` hold the two-million-point Meyer pools, including noisy/clean predictors, tangents,
latent coordinates and geometry. The paper uses `gamma/X_UniCstDevSig` for
its predictors. MATLAB `h5read` returns a D-by-N matrix; HDF5 tools show N-by-D.
Single-index predictors are generated by the experiment, so there is no input
dataset for that model.

`dataset/bumpy_sphere/` contains the canonical v2-D20 predictor dataset and the
three global-tube datasets. `dataset/bumpy_sphere/iid/` holds the fresh IID
realizations for the bumpy-sphere repetitions.

`dataset/coupled_helix/` contains the base surface, response-map calibration,
polynomial/rational data, and exponential data. Its `iid/` subdirectory holds
the fresh independent realizations. `dataset/reaction_coordinate/D20/`
contains the full Monte Carlo dataset used by the reported example.

`config/dataset_inventory.json` lists available files, array shapes, dtypes and
checksums. `scripts/datasets.py` reads this shared manifest. Large dataset files
are excluded from Git; see `DATASETS.md` for retrieval and generation commands.
