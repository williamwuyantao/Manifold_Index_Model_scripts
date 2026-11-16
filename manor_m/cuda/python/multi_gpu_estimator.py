"""One large-N fit across multiple CUDA devices, with one host kNN index.

The original estimator remains unchanged. Exact response-kNN query ranges and
Phase-2 owner ranges run concurrently. Phase-1's globally coupled owner rule
and initialization still run once through the original implementation.
"""
from __future__ import annotations

import ctypes as C
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from manor_m_cuda import (ManorMCuda, _CPhase2Input, _CPhase2Options, _CPhase2Evaluation,
                          _FloatPointer, _Int32Pointer, _Uint8Pointer,
                          _Uint32Pointer, _CPhase1Options, _CPhase1Output, _CPhase1Stats,
                          _CP1CapacityOptions, _CP1CapacityPlan, _p1_capacity_options,
                          ManorMCudaError, _ABI_VERSION)
from phase1 import _cuts, _hybrid_phase1, _phase2_schedules
from resident_sweep import _snapshot, IndexResidency
from estimator import P1_NAMES
from stage2_memory import MemoryPolicy

GIB = 1024**3


class _Memory(C.Structure):
    _fields_ = [('struct_size', C.c_uint32), ('version', C.c_uint32),
                ('working_bytes', C.c_uint64), ('reserve_bytes', C.c_uint64),
                ('pinned_bytes_per_device', C.c_uint64),
                ('maximum_tile_owners', C.c_int32), ('flags', C.c_int32)]


class _Stats(C.Structure):
    _fields_ = [('struct_size', C.c_uint32), ('version', C.c_uint32),
                *[(k, C.c_int32) for k in ('device_count', 'owner_count',
                                          'valid_iterations', 'stopped_no_heavy')],
                *[(k, C.c_uint64) for k in ('logical_index_bytes', 'uploaded_index_bytes',
                                           'peer_bytes', 'relayed_bytes', 'pinned_bytes')],
                *[(k, C.c_double) for k in ('regression_ms', 'masks_ms',
                                           'aggregation_ms', 'communication_ms')]]


def _pointer(array, kind):
    return array.ctypes.data_as(kind) if array is not None else kind()


def _hash(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def host_capacity_bytes():
    """Available host/cgroup capacity, with 128 GiB of global headroom."""
    fields = {s.split(':')[0]: int(s.split()[1])*1024
              for s in Path('/proc/meminfo').read_text().splitlines()
              if s.startswith(('MemTotal:', 'MemAvailable:'))}
    limits = [fields['MemTotal']]
    for name in ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory.max'):
        p = Path(name)
        if p.exists() and p.read_text().strip() != 'max':
            limits.append(int(p.read_text()))
    anonymous = 0
    for name in ('/sys/fs/cgroup/memory/memory.stat', '/sys/fs/cgroup/memory.stat'):
        p = Path(name)
        if p.exists():
            values = dict((s.split()[0], int(s.split()[1])) for s in p.read_text().splitlines())
            anonymous = values.get('total_rss', values.get('anon', 0))
            break
    return max(0, min(fields['MemAvailable'], min(limits)-anonymous)-128*GIB)


class MultiGpuEstimator:
    """Explicit per-fit ownership; no cross-N, cross-IID or cross-job cache.

    All callers on a machine use the same workdir: its advisory lock serializes
    large fits. ``devices`` are visible CUDA ordinals. Runtime capacity choices
    use allocation arithmetic, never trial throughput kernels.
    """
    def __init__(self, library, x, y, *, devices, workdir,
                 working_bytes=0, reserve_bytes=2*GIB,
                 pinned_bytes_per_device=512*1024**2,
                 host_bytes=None, maximum_tile_owners=65536,
                 force_host_relay=False, force_index_stream=False):
        self.devices = tuple(int(v) for v in devices)
        self.requested_devices = self.devices
        if not self.devices or len(set(self.devices)) != len(self.devices):
            raise ValueError('devices must contain unique CUDA ordinals')
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._lock = (self.workdir/'large-fit.lock').open('a')
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise RuntimeError('Another large-N fit owns this workdir')
        self.contexts, self.pool, self._watch, self._model = [], None, None, C.c_void_p()
        self.index = self.p1 = None
        try:
            self.core = ManorMCuda(library)
            self.lib = self.core._library
            self._bind()
            self.x = _snapshot(x, np.float32)
            self.y = _snapshot(y, np.float32)
            if self.x.ndim != 2 or self.y.ndim != 2 or len(self.x) != len(self.y):
                raise ValueError('X and Y must be matching nonempty matrices')
            if not len(self.x) or not np.isfinite(self.x).all() or not np.isfinite(self.y).all():
                raise ValueError('X/Y must be nonempty and finite')
            self.n, self.d = self.x.shape
            if len(self.devices) > self.n:
                raise ValueError('More devices than samples')
            self.memory = _Memory(C.sizeof(_Memory), 1, int(working_bytes),
                                  int(reserve_bytes), int(pinned_bytes_per_device),
                                  int(maximum_tile_owners),
                                  int(force_host_relay) | (int(force_index_stream) << 1))
            if working_bytes < 0 or reserve_bytes < 0 or pinned_bytes_per_device < 4096 or maximum_tile_owners < 1:
                raise ValueError('Invalid memory settings')
            capacity = host_capacity_bytes()
            self.host_bytes = capacity if host_bytes is None else min(capacity, int(host_bytes))
            if self.host_bytes < self.x.nbytes+self.y.nbytes+2*GIB:
                raise MemoryError('Insufficient host capacity for one large-N fit')
            input_bytes = self.x.nbytes+self.y.nbytes
            available = sorted(((self.core.device_info(d).free_global_memory_bytes, d)
                                for d in self.devices), reverse=True)
            self.devices = tuple(d for free, d in available if free >= input_bytes+reserve_bytes)
            if not self.devices:
                raise MemoryError('No requested GPU can retain the training inputs')
            for device in self.devices:
                context = self.core.context(device_ordinal=device, stream_count=2)
                self.contexts.append(context)
                context._resident_begin(self.x.nbytes+self.y.nbytes+64*1024**2)
                context._resident_put('dataset.x', self.x)
                context._resident_put('dataset.response', self.y)
            self.pool = ThreadPoolExecutor(max_workers=len(self.contexts))
            self.preparation = None
        except Exception:
            self.close()
            raise

    def _bind(self):
        lib = self.lib
        lib.manor_m_multi_error.restype = C.c_char_p
        lib.manor_m_multi_phase1_error.restype = C.c_char_p
        lib.manor_m_multi_phase1.argtypes = [C.POINTER(C.c_void_p), C.c_int32,
            _FloatPointer, C.c_int64, C.c_int32, _Int32Pointer, C.c_int32,
            _Int32Pointer, C.c_int32, C.POINTER(_CPhase1Options),
            C.POINTER(_CPhase1Output), C.POINTER(_CP1CapacityOptions),
            C.POINTER(_CPhase1Stats), C.POINTER(_CP1CapacityPlan)]
        lib.manor_m_multi_minimum_bytes.argtypes = [C.c_int32]*6+[C.POINTER(C.c_uint64)]
        lib.manor_m_multi_create.argtypes = [C.POINTER(C.c_void_p), C.c_int32,
            C.POINTER(_CPhase2Input), C.POINTER(_CPhase2Options), C.POINTER(_Memory),
            C.POINTER(C.c_void_p)]
        lib.manor_m_multi_step.argtypes = [C.c_void_p, C.c_float, C.c_float, C.POINTER(_Stats)]
        lib.manor_m_multi_snapshot.argtypes = [C.c_void_p, _FloatPointer, _FloatPointer,
                                              _Int32Pointer, _Uint8Pointer, _Uint8Pointer, _Uint8Pointer]
        lib.manor_m_multi_stats.argtypes = [C.c_void_p, C.POINTER(_Stats)]
        lib.manor_m_multi_evaluate.argtypes = [C.c_void_p, C.c_int32,
            C.POINTER(_CPhase2Evaluation), C.c_float, C.c_float, C.c_int32,
            _FloatPointer, _FloatPointer, _FloatPointer, _FloatPointer]
        lib.manor_m_multi_reduce_queries.argtypes = [C.c_void_p, C.c_int32,
            _FloatPointer, C.c_int32, C.c_int32, C.c_int32,
            _FloatPointer, _FloatPointer, _FloatPointer]
        lib.manor_m_multi_destroy.argtypes = [C.c_void_p]

    def _check(self, value):
        if value:
            raise RuntimeError(self.lib.manor_m_multi_error().decode('utf-8'))

    def _parallel(self, function):
        futures = [self.pool.submit(function, i) for i in range(len(self.contexts))]
        # Drain every task before propagating an exception or releasing inputs.
        values, error = [], None
        for future in futures:
            try:
                values.append(future.result())
            except Exception as exc:
                values.append(None)
                if error is None:
                    error = exc
        if error is not None:
            raise error
        return values

    def prepare(self, k):
        k = int(k)
        if self.p1 is not None:
            if self.k != k:
                raise ValueError('Close this fit before changing its exact k')
            return dict(reused=True, index_rebuilt=False, source='active', wall_seconds=0.0)
        cuts = _cuts(self.d, k)
        started = time.perf_counter()
        # Capacity-only planning precedes the expensive N*k index build. A
        # device that cannot hold P1's global state contributes no index rows.
        policy = MemoryPolicy(phase1_working_bytes=int(self.memory.working_bytes),
            pinned_bytes=int(self.memory.pinned_bytes_per_device), phase1_mode=3,
            reserve_bytes=int(self.memory.reserve_bytes), reserve_fraction=0)
        survivors, preflight = [], []
        for device, context in zip(self.devices, self.contexts):
            options = dict(policy.phase1_options(self.core.device_info(device).free_global_memory_bytes),
                           reserve_fraction=0)
            try:
                plan = context.plan_phase1(self.x,self.y,k,[k],cuts,2,
                    capacity_options=options,hybrid=True)
            except ManorMCudaError as error:
                if 'Neither resident nor bounded-index P1 fits' not in str(error):
                    raise
                preflight.append(dict(device=device,selected=False,reason=str(error)))
                context.close()
            else:
                survivors.append((device,context))
                preflight.append(dict(device=device,selected=True,plan=plan))
        self.devices = tuple(device for device, _ in survivors)
        self.contexts = [context for _, context in survivors]
        if not survivors:
            raise MemoryError('No requested GPU fits the exact Phase1 state')
        # Exactly one shared mmap. Each GPU writes disjoint complete query rows.
        overhead = self.x.nbytes+self.y.nbytes+len(self.contexts)*self.memory.pinned_bytes_per_device+4*GIB
        window = self.host_bytes-overhead
        if window < GIB:
            raise MemoryError('No host index window remains')
        self._watch = IndexResidency(str(self.workdir), window, self.host_bytes)
        self._watch.__enter__()
        self.index = self._watch.storage(self.n, k, 0, str(self.workdir))
        def build(i):
            first = self.n*i//len(self.contexts)
            last = self.n*(i+1)//len(self.contexts)
            info = self.core.device_info(self.devices[i])
            available = max(0, info.free_global_memory_bytes-self.memory.reserve_bytes)
            scratch = min(available, self.memory.working_bytes or available)
            _, stats = self.contexts[i].response_knn(self.y, k, query_first=first,
                query_count=last-first, scratch_memory_bytes=scratch,
                host_output=True, output=self.index[first:last])
            return dict(first=first, count=last-first, stats=asdict(stats))
        knn = self._parallel(build)
        self.index.flags.writeable = False
        knn_wall = time.perf_counter()-started
        p1, summary = self._phase1(k, cuts)
        self.p1 = {name: _snapshot(p1[name], p1[name].dtype) for name in P1_NAMES}
        self.k = k
        self.preparation = dict(reused=False, index_rebuilt=True,
            source='parallel_exact_knn_and_global_phase1', wall_seconds=time.perf_counter()-started,
            logical_host_index_bytes=self.index.nbytes, host_index_instances=1,
            knn_wall_seconds=knn_wall, knn_partitions=knn, initialization=summary,
            phase1_device_capacity=preflight,
            phase1_state_sha256={name: _hash(value) for name, value in self.p1.items()})
        self._check_host()
        return self.preparation

    def _phase1(self, k, cuts):
        # Each original kernel keeps the full-data coordinate system. Global
        # owner thresholds are computed after copying all selected raw rows.
        started = time.perf_counter()
        policy = MemoryPolicy(phase1_working_bytes=int(self.memory.working_bytes),
            pinned_bytes=int(self.memory.pinned_bytes_per_device), phase1_mode=3,
            reserve_bytes=int(self.memory.reserve_bytes), reserve_fraction=0)
        count = len(self.contexts)
        block = (self.n+count-1)//count
        capacities = (_CP1CapacityOptions*count)(*[
            _p1_capacity_options(dict(policy.phase1_options(
                self.core.device_info(device).free_global_memory_bytes), maximum_query_block=block,
                reserve_fraction=0))
            for device in self.devices])
        options = _CPhase1Options(struct_size=C.sizeof(_CPhase1Options), abi_version=_ABI_VERSION,
            gpu_memory_fraction=.88, reserve_memory_fraction=0,
            reserve_memory_bytes=self.memory.reserve_bytes, heavy_factor=.1, jitter=1e-6,
            neighbor_scaled_max=1, split_scaled_max=6, minimum_query_block=1,
            maximum_query_block=block, stable_window=7, selection_mode=0,
            split_seed=730021, cut_max_fraction=.8, collect_profile=1)
        state = dict(Tan0=np.empty((self.n,2,self.d),np.float32),
            Cen0=np.empty((self.n,self.d),np.float32),Rad0=np.empty(self.n,np.float32),
            axial_tan0=np.empty(self.n,np.float32), num_neigh0=np.empty(self.n,np.int32),
            owner_mask0=np.empty(self.n,np.uint8),selected_cut0=np.empty(self.n,np.uint32))
        output = _CPhase1Output(struct_size=C.sizeof(_CPhase1Output),abi_version=_ABI_VERSION,
            tangent_point_major=_pointer(state['Tan0'],_FloatPointer),
            center_point_major=_pointer(state['Cen0'],_FloatPointer),
            radius=_pointer(state['Rad0'],_FloatPointer),
            axial_tangent=_pointer(state['axial_tan0'],_FloatPointer),
            neighbor_count=_pointer(state['num_neigh0'],_Int32Pointer),
            owner_mask=_pointer(state['owner_mask0'],_Uint8Pointer),
            selected_cut=_pointer(state['selected_cut0'],_Uint32Pointer))
        stats, plans = (_CPhase1Stats*count)(), (_CP1CapacityPlan*count)()
        contexts = (C.c_void_p*count)(*[c._pointer.value for c in self.contexts])
        code = self.lib.manor_m_multi_phase1(contexts,count,_pointer(self.x,_FloatPointer),
            self.n,self.d,_pointer(self.index,_Int32Pointer),k,_pointer(cuts,_Int32Pointer),
            cuts.shape[1],C.byref(options),C.byref(output),capacities,stats,plans)
        if code:
            raise RuntimeError(self.lib.manor_m_multi_phase1_error().decode('utf-8'))
        return state, dict(execution_mode='multi_gpu_fused_stable_top5',
            wall_seconds=time.perf_counter()-started,owner_count=int(np.count_nonzero(state['owner_mask0'])),
            global_owner_thresholds=True,
            per_device_stats=[{key:getattr(s,key) for key,_ in s._fields_} for s in stats],
            per_device_capacity=[{key:getattr(p,key) for key,_ in p._fields_} for p in plans])

    def _check_host(self):
        if self._watch is not None and self._watch.exceeded:
            raise MemoryError('The large-N fit exceeded its host reservation')

    def _create_model(self, *, heavy_factor=.1, temporary_heavy_factor=.2,
                      lambda_=1e-5, jitter=1e-6, regression_mode=0):
        if self._model:
            self._check(self.lib.manor_m_multi_destroy(self._model))
            self._model = C.c_void_p()
        required = C.c_uint64()
        self._check(self.lib.manor_m_multi_minimum_bytes(self.n, self.d, 2,
            self.y.shape[1], int(np.count_nonzero(self.p1['owner_mask0'])), self.k, C.byref(required)))
        survivors, plans = [], []
        for device, context in zip(self.devices, self.contexts):
            free = self.core.device_info(device).free_global_memory_bytes
            usable = max(0, free-self.memory.reserve_bytes)
            if self.memory.working_bytes:
                usable = min(usable, self.memory.working_bytes)
            plans.append(dict(device=device, free_bytes=free, usable_bytes=usable,
                              minimum_model_and_readout_bytes=required.value,
                              selected=usable >= required.value))
            if usable >= required.value:
                survivors.append((device, context))
            else:
                context.close()
        self.devices = tuple(device for device, _ in survivors)
        self.contexts = [context for _, context in survivors]
        self.device_capacity = plans
        if not survivors:
            raise MemoryError('No requested GPU fits the distributed model/readout state')
        self._rho = np.ones(1, np.float32)
        p = self.p1
        raw = _CPhase2Input(struct_size=C.sizeof(_CPhase2Input), abi_version=_ABI_VERSION,
            x_point_major=_pointer(self.x, _FloatPointer), response_point_major=_pointer(self.y, _FloatPointer),
            index_zero_based=_pointer(self.index, _Int32Pointer),
            tangent_point_major=_pointer(p['Tan0'], _FloatPointer),
            center_point_major=_pointer(p['Cen0'], _FloatPointer),
            owner_mask=_pointer(p['owner_mask0'], _Uint8Pointer),
            point_count=self.n, ambient_dimension=self.d, manifold_dimension=2,
            response_dimension=self.y.shape[1], neighbor_count=self.k,
            base_h=1.0, rho_schedule=_pointer(self._rho, _FloatPointer), rho_schedule_count=1)
        opts = _CPhase2Options(struct_size=C.sizeof(_CPhase2Options), abi_version=_ABI_VERSION,
            **{'lambda_': lambda_, 'jitter': jitter, 'gpu_memory_fraction': .88,
               'reserve_memory_fraction': 0, 'reserve_memory_bytes': self.memory.reserve_bytes,
               'minimum_query_block': 1, 'maximum_query_block': 131072,
               'regression_mode': regression_mode, 'iteration_count': 1,
               'heavy_factor': heavy_factor, 'temporary_heavy_factor': temporary_heavy_factor,
               'evaluation_uses_owner_mask': 1})
        contexts = (C.c_void_p*len(self.contexts))(*[c._pointer.value for c in self.contexts])
        self._check(self.lib.manor_m_multi_create(contexts, len(contexts), C.byref(raw),
                    C.byref(opts), C.byref(self.memory), C.byref(self._model)))

    def snapshot(self):
        state = dict(Tan_next=np.empty((self.n, 2, self.d), np.float32),
                     Cen_next=np.empty((self.n, self.d), np.float32),
                     num_neigh=np.empty(self.n, np.int32),
                     heavy_mask=np.empty(self.n, np.uint8), keep_mask=np.empty(self.n, np.uint8),
                     solver_failed=np.empty(self.n, np.uint8))
        self._check(self.lib.manor_m_multi_snapshot(self._model,
            *[_pointer(state[k], kind) for k, kind in (
                ('Tan_next', _FloatPointer), ('Cen_next', _FloatPointer),
                ('num_neigh', _Int32Pointer), ('heavy_mask', _Uint8Pointer),
                ('keep_mask', _Uint8Pointer), ('solver_failed', _Uint8Pointer))]))
        return state

    def evaluate_state(self, *, h, rho, views, eval_rhos, eval_hs,
                       fallback_exponent_scale=2.0, sum_sparse_threshold=10.0,
                       initial_normalization=False):
        """Partition queries, then apply the original full-query reduction tree."""
        jobs = [(name, mode, x, truth) for name, values in views.items()
                for mode, x, truth in (('response', values['x'], values.get('y')),
                                        ('tangent', values['x'], values.get('tangent')))
                if truth is not None]
        if any(len(x) > 65536 for _, _, x, _ in jobs):
            raise ValueError('This fixed-parameter readout supports up to 65536 queries per view')
        hs = np.ascontiguousarray(eval_hs, dtype=np.float32)
        rs = np.ascontiguousarray(eval_rhos, dtype=np.float32)
        if not len(hs) or not len(rs) or len(hs)*len(rs) > 9:
            raise ValueError('Readout supports one to nine fixed evaluation slots')
        if not jobs:
            return {}
        # All devices cooperate on one metric family at a time. This keeps
        # response-only and tangent-only query work from stranding device groups.
        if len(self.contexts)>1 and len(jobs)>1:
            output={}
            for name,mode,x,truth in jobs:
                key='y' if mode=='response' else 'tangent'
                output.update(self.evaluate_state(h=h,rho=rho,
                    views={name:dict(x=x,**{key:truth})},eval_rhos=eval_rhos,eval_hs=eval_hs,
                    fallback_exponent_scale=fallback_exponent_scale,
                    sum_sparse_threshold=sum_sparse_threshold,
                    initial_normalization=initial_normalization))
            return output
        canonical = []
        for name, mode, x, truth in jobs:
            x = np.ascontiguousarray(x, dtype=np.float32)
            truth = np.ascontiguousarray(truth, dtype=np.float32)
            expected = ((len(x), self.y.shape[1]) if mode == 'response'
                        else (len(x), 2, self.d))
            if not len(x) or x.shape != (len(x), self.d) or truth.shape != expected:
                raise ValueError('Query truth layout does not match the model')
            canonical.append((name, mode, x, truth))
        # Every context runs in at most one task at a time. If there are more
        # GPUs than views, each view gets a contiguous group of GPUs and queries.
        count = len(self.contexts)
        assignments = [[] for _ in self.contexts]
        gathered, roots = {}, {}
        for j, (_, _, x, _) in enumerate(canonical):
            group = (list(range(count*j//len(jobs), count*(j+1)//len(jobs)))
                     if count >= len(jobs) else [j % count])
            group = group[:len(x)]
            roots[j] = group[0]
            if len(group) > 1:
                gathered[j] = np.empty((len(rs)*len(hs)*(self.y.shape[1]+4), len(x)), np.float32)
            for rank, device in enumerate(group):
                first, last = len(x)*rank//len(group), len(x)*(rank+1)//len(group)
                assignments[device].append((j, first, last))

        def output_arrays(mode):
            return (np.empty((len(rs), len(hs)), np.float32) if mode == 'response' else None,
                    np.empty((len(rs), len(hs), self.y.shape[1]), np.float32) if mode == 'response' else None,
                    np.empty((len(rs), len(hs)), np.float32) if mode == 'tangent' else None)

        def add_outputs(local, name, mode, mse, coordinate, tangent):
            if mode == 'response':
                local[(name, 'MSE')] = mse
                local[(name, 'MSECoordinate')] = coordinate
            else:
                local[(name, 'DevTan')] = tangent

        def score(device):
            local = {}
            for j, first, last in assignments[device]:
                name, mode, all_x, all_truth = canonical[j]
                x, truth = all_x[first:last], all_truth[first:last]
                mse, coordinate, tangent = output_arrays(mode)
                contributions = (np.empty((gathered[j].shape[0], last-first), np.float32)
                                 if j in gathered else None)
                evaluation = _CPhase2Evaluation(struct_size=C.sizeof(_CPhase2Evaluation), abi_version=_ABI_VERSION,
                    test_x_point_major=_pointer(x, _FloatPointer),
                    test_y_point_major=_pointer(truth if mode == 'response' else None, _FloatPointer),
                    test_tangent_point_major=_pointer(truth if mode == 'tangent' else None, _FloatPointer),
                    test_point_count=len(x), rho_scales=_pointer(rs, _FloatPointer), rho_scale_count=len(rs),
                    h_scales=_pointer(hs, _FloatPointer), h_scale_count=len(hs),
                    sum_sparse_threshold=sum_sparse_threshold, count_sparse_threshold=0,
                    fallback_exponent_scale=fallback_exponent_scale, epsilon_weight=1e-20,
                    warps_per_block=4, center_h_scale=1.25, center_rho_scale=4)
                self._check(self.lib.manor_m_multi_evaluate(self._model, device, C.byref(evaluation),
                    float(h), float(rho), int(initial_normalization), _pointer(mse, _FloatPointer),
                    _pointer(coordinate, _FloatPointer), _pointer(tangent, _FloatPointer),
                    _pointer(contributions, _FloatPointer)))
                if contributions is not None:
                    # Copy, not a partial floating-point sum. Query q returns
                    # to column q before the unchanged 256-lane sum tree.
                    gathered[j][:, first:last] = contributions
                else:
                    add_outputs(local, name, mode, mse, coordinate, tangent)
            return local
        outputs = {}
        for local in self._parallel(score):
            outputs.update(local)
        def reduce(device):
            local = {}
            for j, contributions in gathered.items():
                if roots[j] != device:
                    continue
                name, mode, x, _ = canonical[j]
                mse, coordinate, tangent = output_arrays(mode)
                self._check(self.lib.manor_m_multi_reduce_queries(self._model, device,
                    _pointer(contributions, _FloatPointer), len(x), len(rs)*len(hs),
                    int(initial_normalization), _pointer(mse, _FloatPointer),
                    _pointer(coordinate, _FloatPointer), _pointer(tangent, _FloatPointer)))
                add_outputs(local, name, mode, mse, coordinate, tangent)
            return local
        if gathered:
            for local in self._parallel(reduce):
                outputs.update(local)
        return outputs

    def fit(self, candidate, data, *, all_iterations=True, on_iteration=None,
            heavy_factor=.1, temporary_heavy_factor=.2,
            fallback_exponent_scale=2.0, sum_sparse_threshold=10.0):
        """Return the same fixed-parameter validation/testing measurement fields."""
        started = time.perf_counter()
        if (_hash(data.arrays['X']) != _hash(self.x) or
                _hash(data.arrays['Y']) != _hash(self.y)):
            raise ValueError('The query record belongs to different training data')
        preparation = self.prepare(candidate['kSlice'])
        requested = int(candidate['K'])
        if requested < 0:
            raise ValueError('K must be nonnegative')
        views = {name: dict(x=_snapshot(data.arrays['X'+suffix], np.float32),
                            y=_snapshot(data.arrays['Y'+suffix], np.float32),
                            tangent=_snapshot(data.arrays['T'+suffix], np.float32))
                 for name, suffix in (('validation', 'valid'), ('testing', 'test'))}
        _, _, hs, rhos = _phase2_schedules(self.p1['Rad0'], self.p1['owner_mask0'].astype(bool),
            self.p1['axial_tan0'], candidate['h_scale'], candidate['rho_init'],
            candidate['bite'], self.d, 2, max(1, requested))
        self._create_model(heavy_factor=heavy_factor, temporary_heavy_factor=temporary_heavy_factor)
        measurements = {}
        if all_iterations or requested == 0:
            measurements[0] = self.evaluate_state(h=hs[0], rho=rhos[0], views=views,
                eval_rhos=[candidate['eval_rho']], eval_hs=[candidate['eval_h']],
                fallback_exponent_scale=fallback_exponent_scale, sum_sparse_threshold=sum_sparse_threshold,
                initial_normalization=True)
        stats = _Stats()
        self._check(self.lib.manor_m_multi_stats(self._model, C.byref(stats)))
        for k in range(1, requested+1):
            self._check(self.lib.manor_m_multi_step(self._model, hs[k-1], rhos[k-1], C.byref(stats)))
            self._check_host()
            if stats.stopped_no_heavy:
                break
            if all_iterations or k == requested:
                measurements[k] = self.evaluate_state(h=hs[k-1], rho=rhos[k-1], views=views,
                    eval_rhos=[candidate['eval_rho']], eval_hs=[candidate['eval_h']],
                    fallback_exponent_scale=fallback_exponent_scale, sum_sparse_threshold=sum_sparse_threshold)
            if on_iteration:
                on_iteration(k, {key: getattr(stats, key) for key, _ in stats._fields_})
        valid = int(stats.valid_iterations)
        state = self.snapshot()
        if valid not in measurements:
            position = max(0, valid-1)
            measurements[valid] = self.evaluate_state(h=hs[position], rho=rhos[position], views=views,
                eval_rhos=[candidate['eval_rho']], eval_hs=[candidate['eval_h']],
                fallback_exponent_scale=fallback_exponent_scale, sum_sparse_threshold=sum_sparse_threshold,
                initial_normalization=valid == 0 or (not all_iterations and valid < requested))
        hashes = {key: _hash(state[key]) for key in state}
        diagnostic = dict(multi_gpu={key: getattr(stats, key) for key, _ in stats._fields_},
            preparation=preparation, host_memory=self._watch.stats(), final_state_sha256=hashes,
            model_update_passes=1, metric_sum_contract='fp32-query256-tree-v1',
            device_capacity=self.device_capacity)
        output = {}
        selected = range(requested+1) if all_iterations else [requested]
        for view in views:
            variance = np.var(views[view]['y'].astype(np.float64), axis=0, ddof=1)
            if not np.isfinite(variance).all() or (variance <= 0).any():
                raise ValueError('Nonpositive/nonfinite response variance')
            total = float(variance.sum())
            rows = []
            for k in selected:
                effective = min(k, valid)
                values = measurements[effective]
                mse = np.asarray(values[(view, 'MSE')], float).reshape(-1)
                coordinate = np.asarray(values[(view, 'MSECoordinate')], float).reshape(-1)
                tangent = np.asarray(values[(view, 'DevTan')], float).reshape(-1)
                if not all(np.isfinite(v).all() and (v >= 0).all() for v in (mse, coordinate, tangent)):
                    raise RuntimeError('Nonfinite or negative executed metric')
                rows.append(dict(iteration=k, state_iteration=effective, terminal_noop=k > valid,
                    relMSE_total=float(mse[0]/total), relMSE_coord=float(np.mean(coordinate/variance)),
                    devTan=float(tangent[0])))
            endpoint = rows[-1]
            output[view] = dict(candidate, testing_relMSE_total=endpoint['relMSE_total'],
                testing_relMSE_coord=endpoint['relMSE_coord'], testing_devTan=endpoint['devTan'],
                requested_K=requested, effective_K=valid, stopped_no_heavy=bool(stats.stopped_no_heavy),
                terminal_state_carry_forward=valid < requested, iteration_rows=rows,
                phase1_is_iteration_zero=True, selection_iteration=requested,
                iteration_contract='all-phase2-iterations-separate-metrics.v1', metric_view=view,
                testing_used_for_selection=False, final_state_sha256={'response': hashes, 'tangent': hashes},
                preparation=preparation, python_diagnostics={'shared_trajectory': diagnostic},
                model_update_passes=1, shared_evaluation_groups=['validation', 'testing'])
        return dict(output, diagnostic=diagnostic, wall=time.perf_counter()-started)

    def close(self):
        try:
            if self._model:
                try:
                    self._check(self.lib.manor_m_multi_destroy(self._model))
                finally:
                    self._model = C.c_void_p()
        finally:
            try:
                if self.pool is not None:
                    self.pool.shutdown(wait=True)
                    self.pool = None
                for context in reversed(self.contexts):
                    context.close()
                self.contexts = []
                if self._watch is not None:
                    try:
                        self._watch.__exit__(None, None, None)
                    finally:
                        self._watch = None
            finally:
                self.index = self.p1 = None
                if getattr(self, '_lock', None) is not None:
                    self._lock.close()
                    self._lock = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
