# Large-N multi-GPU backend

This separate library runs one fixed-parameter experiment across several GPUs.
The base single-GPU library and runners remain available alongside this backend.

One process owns one immutable training dataset and one exact response-kNN
index. GPUs construct disjoint query rows of that index. Stable-H selection and extraction then run on disjoint global anchor ranges.
Their raw rows are exchanged before the global owner thresholds are
computed. Initialization aggregation also runs on disjoint anchor ranges.
During Phase 2, GPUs compute disjoint
owner rows and exchange gradients, counts and model states at the algorithm's
synchronization boundaries. A complete ordered neighbor row always stays on
one GPU. Communication copies values; it does not introduce a floating-point
cross-GPU sum. Peer copies are used where available, with a bounded pinned-host
relay otherwise.

At each model state, all admitted GPUs cooperate on one metric family at a
time: validation response, validation tangent, testing response, then testing
tangent. Each view is split into contiguous query ranges. The model is updated
once per iteration and shared by these readouts. Per-query contributions are copied back into the original query
order before the fixed 256-lane metric reduction. No partial metric
sums are added across GPUs. Sparse fallback and the DevTan definition are
unchanged.

Native readout buffers remain allocated through the fit. Training norms and
tangent products are refreshed once per model state. The optional tangent
Gram cache stores `N*D*(D+1)/2` FP32 entries per GPU; allocation uses both the
remaining model budget and actual free memory. If it does not fit, readout
uses the per-tile Gram calculation. No cache changes query weights,
neighbor order or metric reduction order.

GPU capacity determines whether each device retains its assigned index rows
or streams bounded whole-row tiles. No throughput probes run in production.
The host index is a single shared mapping; an RSS window bounds its residency.
All large fits using this backend must use the same shared work directory so
that `large-fit.lock` serializes them. Different N values and IID repetitions
never reuse model or index state.

Build with the same numerical compiler options as the qualified single-GPU
library, using this directory as the CMake source:

```sh
cmake -S manor_m/cuda/multi_gpu -B build/multi_gpu \
  -DCMAKE_CUDA_COMPILER="${NVCC:-nvcc}" \
  -DMANOR_M_CUDA_ARCHITECTURES="${CUDA_ARCH:-90-real}" \
  -DMANOR_M_BUILD_MATLAB=OFF -DMANOR_M_STATIC_GNU_RUNTIME=ON \
  -DMANOR_M_REGRESSION_SHARED_SOLVE=ON -DMANOR_M_REGRESSION_WARP_SYNC=ON \
  -DMANOR_M_REGRESSION_COOPERATIVE_SOLVE=ON -DMANOR_M_REGRESSION_PARALLEL_SYSTEM=ON \
  -DMANOR_M_REGRESSION_SHUFFLE_INPUTS=ON -DMANOR_M_REGRESSION_WARP_ENTRIES=ON \
  -DMANOR_M_AGGREGATION_WARP_ENTRIES=ON -DMANOR_M_AGGREGATION_TILED_MASK=ON \
  -DMANOR_M_HOST_INDEX_VECTOR_CHECK=ON -DMANOR_M_STREAM_OWNER_ROWS=ON \
  -DMANOR_M_DEVTAN_ORTHONORMAL_FP64=ON -DMANOR_M_EVAL_MIN_BLOCKS_PER_SM=5 \
  -DMANOR_M_PHASE1_SELECT_MIN_BLOCKS_PER_SM=24 \
  -DMANOR_M_MULTI_PERSISTENT_READOUT=ON -DMANOR_M_MULTI_CACHED_GRAM=ON \
  '-DCMAKE_CUDA_FLAGS=--ptxas-options=-v -DMANOR_M_JACOBI_READ_BARRIER=1'
cmake --build build/multi_gpu --target manor_m_cuda -j2
```

The output is `single_gpu/libmanor_m_cuda_multi.so` under the build directory.
Use an architecture supported by the actual GPUs. The `MANOR_M_*` numerical
options should match the single-GPU production build.

`scaling_law_N/large_n_multi_gpu.py` provides `run_fixed(...)` for an already
loaded `Stage2Data`, and a CLI for global-tube and coupled-helix datasets.
It requires the qualification manifest matching the library and GPU count.
The current application adapter handles two-dimensional manifolds.

```python
from large_n_multi_gpu import run_fixed

result = run_fixed(
    library, data, fixed_seven_parameters,
    devices=[0, 1, 2, 3],
    workdir=shared_large_fit_workdir,
    qualification=qualification_manifest,
    all_iterations=True,
)
```

Add `scaling_law_N` and `manor_m/cuda/python` to `PYTHONPATH` when importing
the adapter. All applications using this backend share
`shared_large_fit_workdir`; the fit lock is held through model destruction.
The seven parameters, alpha-to-k binding, K-offset schedule, data, independent
testing and numerical thresholds are unchanged.

The build creates a separate Phase-1 translation unit from the base
source, adding range dispatch and synchronization around its unchanged
kernels. Global anchor numbers, split seeds, neighbor order and full-data
owner thresholds are retained. Exact kNN building, initialization, Phase-2
model updates and metric readouts are distributed. Actual speedup is measured
from complete fits; transfer and synchronization remain part of that time.

For a newly compiled library, the bundled test entry can write its matching
qualification manifest. Use an existing canonical global-tube fixture of at
most 100k training points with 5000 queries in each independent view:

```sh
python manor_m/cuda/tests/multi_gpu.py \
  --library MULTI_LIBRARY --baseline SINGLE_LIBRARY --devices 0,1,2,3 \
  --workdir qualification \
  --dataset FIXTURE_H5 --tube-module dataset_generator/bumpy_global_tube.py \
  --parameters FIXED_PARAMETERS_JSON --c 0.5 \
  --qualification-output qualification.json
```

This checks the requested device counts, complete initialization arrays,
model states, metrics, sparse fallback and stopped states, followed by
memcheck, synccheck and racecheck. The resulting manifest is consumed by
`run_fixed`.
