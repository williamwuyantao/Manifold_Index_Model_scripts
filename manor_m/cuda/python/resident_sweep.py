"""Opt-in device residency across sweeps in ONE live CUDA worker process.

Default reuse=False still uploads a dataset once *within* the sweep, then
releases it. reuse=True additionally retains it for the next sweep. It is not
an on-disk cache, CUDA IPC, or persistence across Slurm jobs/process exits.

This is a candidate execution primitive, not a promoted setup-matrix runner.
"""
from contextlib import contextmanager
from collections import OrderedDict
import hashlib
import inspect
import json
import mmap
import os
import shutil
import tempfile
import threading
import time
import weakref

import numpy as np


DATA_NAMES = ("x", "response", "test_x", "test_y", "test_tangent", "test_center",
              "validation_x", "validation_y")


def _digest(array):
    h = hashlib.sha256()
    h.update(str((array.dtype.str, array.shape)).encode())
    h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def _snapshot(value, dtype=np.float32):
    a = np.ascontiguousarray(value, dtype=dtype)
    # bytes backing cannot be made writable with setflags; no caller aliases.
    return np.frombuffer(a.tobytes(order="C"), dtype=a.dtype).reshape(a.shape)


def _index_storage(n, k, heap_bytes, directory=None):
    """Host output for exact kNN, without an unbounded Python heap allocation.

    Larger outputs use an anonymous temporary file, not a result/partial
    file. The mmap owns its lifetime; no manual unmap while an array view exists.
    File-backed page cache remains OS-managed, not a promise about process RSS.
    """
    size = int(n) * int(k) * 4
    if size <= heap_bytes:
        return np.empty((n, k), dtype=np.int32)
    directory = directory or tempfile.gettempdir()
    if shutil.disk_usage(directory).free < size + 1024**3:
        raise MemoryError('Insufficient local scratch disk for the exact kNN index')
    with tempfile.TemporaryFile(prefix='mfm-knn-', dir=directory) as backing:
        # Exact kNN writes every entry before the index becomes readable.
        # Avoid eager zero-fill of the entire file on network filesystems.
        backing.truncate(size)
        return np.memmap(backing, dtype=np.int32, mode='r+', shape=(n, k), order='C')


class IndexResidency:
    """Bound resident mmap pages without discarding reusable kernel file cache.

    The caller sets the process RSS window and limit in bytes. An optional
    on_limit callback can stop a batch worker; otherwise a limit violation
    raises MemoryError when the context exits. Shared dirty pages retain their
    contents across MADV_DONTNEED, and array views own their mapping lifetime.
    """
    def __init__(self, directory, rss_window, reservation, on_limit=None):
        if not 0 < rss_window < reservation:
            raise ValueError('Require 0 < RSS window < process reservation')
        self.directory = str(directory)
        os.makedirs(self.directory, exist_ok=True)
        self.window, self.reservation = int(rss_window), int(reservation)
        self.on_limit = on_limit
        self._allocate = _index_storage
        self.arrays = []
        self._arrays_lock = threading.Lock()
        self.stop = threading.Event()
        self.peak = {'VmRSS': 0, 'RssAnon': 0, 'RssFile': 0}
        self.reclaims, self.errors = 0, []
        self.exceeded = None
        self.thread = None

    @staticmethod
    def memory():
        with open('/proc/self/status') as stream:
            return {line.split(':')[0]: int(line.split()[1])*1024
                    for line in stream if line.split(':')[0] in
                    ('VmRSS', 'RssAnon', 'RssFile')}

    def storage(self, n, k, heap_bytes, directory=None):
        array = self._allocate(n, k, heap_bytes, directory or self.directory)
        if isinstance(array, np.memmap):
            with self._arrays_lock:
                self.arrays.append(weakref.ref(array))
        return array

    def _sample(self):
        while not self.stop.is_set():
            current = self.memory()
            for key, value in current.items():
                self.peak[key] = max(self.peak[key], value)
            if current.get('VmRSS', 0) > self.window:
                with self._arrays_lock:
                    self.arrays = [r for r in self.arrays if r() is not None]
                    references = self.arrays.copy()
                for reference in references:
                    array = reference()
                    if array is None:
                        continue
                    try:
                        array._mmap.madvise(mmap.MADV_DONTNEED)
                        self.reclaims += 1
                    except OSError as error:
                        self.errors.append(str(error))
                remaining = self.memory()
                if remaining.get('VmRSS', 0) > self.reservation:
                    self.exceeded = dict(observed=remaining, **self.stats())
                    if self.on_limit is not None:
                        self.on_limit(self.exceeded)
                    self.stop.set()
            self.stop.wait(.05)

    def __enter__(self):
        if not hasattr(mmap.mmap, 'madvise') or not os.path.isfile('/proc/self/status'):
            raise RuntimeError('RSS-controlled index residency requires Linux madvise')
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, error_type, *_):
        self.stop.set()
        self.thread.join()
        if error_type is None:
            if self.exceeded is not None:
                raise MemoryError('Process exceeded its configured RSS reservation')
            if self.errors:
                raise RuntimeError('Index mapping reclamation failed: '+self.errors[0])

    def stats(self):
        return dict(peak_bytes=self.peak.copy(), rss_window_bytes=self.window,
                    reservation_bytes=self.reservation, page_reclaims=self.reclaims,
                    errors=self.errors.copy(), index_initialization='ftruncate, no eager zero fill',
                    kernel_file_cache='retained; OS-reclaimable')


def _normalize(data, copy):
    if set(data) - set(DATA_NAMES) or any(data.get(k) is None for k in ("x", "response")):
        raise ValueError("Data must contain x/response and only supported truth arrays")
    out = {k: (_snapshot(v) if copy else np.ascontiguousarray(v, dtype=np.float32))
           for k, v in data.items() if v is not None}
    x, y = out["x"], out["response"]
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
        raise ValueError("x/response require nonempty [N,D]/[N,MY] shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Training data must be finite")
    if ("validation_x" in out) != ("validation_y" in out):
        raise ValueError("Validation x/y must be supplied together")
    for prefix in ("test", "validation"):
        truth = [key for key in out if key.startswith(prefix + "_")]
        xx = out.get(prefix + "_x")
        if truth and xx is None:
            raise ValueError("Truth arrays require their query x")
        if xx is not None:
            if xx.ndim != 2 or xx.shape[1] != x.shape[1] or not len(xx):
                raise ValueError("Query x must have nonempty [Q,D] shape")
            for key in truth:
                if not len(out[key]) == len(xx) or not np.isfinite(out[key]).all():
                    raise ValueError("Query/truth count mismatch or nonfinite input")
            yy = out.get(prefix + "_y")
            if yy is not None and yy.shape != (len(xx), y.shape[1]):
                raise ValueError("Truth Y must have shape [Q,MY]")
    return out


class SweepSession:
    """Own all GPU/host inputs; explicitly close or use as a context manager.

    ``with session.sweep(dataset_id, data=..., reuse=True) as sweep: ...``
    followed by ``session.sweep(dataset_id, reuse=True)`` omits dataset upload.
    New supplied data are content-checked against the retained snapshot; a
    different identity/content raises instead of silently changing experiments.
    The owning thread/process is checked before every access. Concurrent use,
    pickling/fork reuse and cross-job reuse are not supported.
    """
    def __init__(self, core, *, device_ordinal=0, maximum_retained_bytes=16 * 1024**3,
                 maximum_phase1_host_bytes=0):
        if maximum_retained_bytes <= 0:
            raise ValueError("A positive retained-memory budget is required")
        if (isinstance(maximum_phase1_host_bytes, bool)
                or not isinstance(maximum_phase1_host_bytes, int) or maximum_phase1_host_bytes < 0):
            raise ValueError("Phase1 host-cache budget must be a nonnegative integer")
        self.core = core
        self.device = int(device_ordinal)
        self.maximum_bytes = int(maximum_retained_bytes)
        self.phase1_host_limit = maximum_phase1_host_bytes
        self._phase1_cache = OrderedDict()
        self._phase1_cache_bytes = self._phase1_cache_hits = self._phase1_cache_evictions = 0
        self.process, self.thread = os.getpid(), threading.get_ident()
        self.context = None
        self.data = {}
        self.identity = self.data_hashes = None
        self.phase1_key = self.phase1 = self.index = None
        self.index_streamed = False
        self.index_compact = False
        self.index_generated = False
        self.generated_neighbor_count = None
        self.generated_knn_scratch_bytes = 0
        self.compact_knn_stats = None
        self.active = False
        self.closed = False
        self.generation = 0
        self.last_stats = {}

    def phase1_cache_stats(self):
        self._check()
        return dict(model_count=len(self._phase1_cache), retained_bytes=self._phase1_cache_bytes,
                    maximum_bytes=self.phase1_host_limit, restores=self._phase1_cache_hits,
                    evictions=self._phase1_cache_evictions, cached_index_bytes=0)

    def _remember_phase1(self, key, p1, index_hash, source_wall):
        # Only immutable initial-model arrays, never P2 output or N*k indices.
        # Unknown mutable metadata is not eligible for this optional cache.
        if not self.phase1_host_limit or not all(isinstance(v, np.ndarray) for v in p1.values()):
            return
        size = sum(v.nbytes for v in p1.values())
        if size > self.phase1_host_limit:
            return
        while self._phase1_cache_bytes + size > self.phase1_host_limit:
            _, old = self._phase1_cache.popitem(last=False)
            self._phase1_cache_bytes -= old['bytes']
            self._phase1_cache_evictions += 1
        self._phase1_cache[key] = dict(model=dict(p1), index_hash=index_hash,
                                      source_wall=source_wall, bytes=size)
        self._phase1_cache_bytes += size

    def _check(self):
        if (os.getpid(), threading.get_ident()) != (self.process, self.thread):
            raise RuntimeError("Resident session belongs to a different process/thread")
        if self.closed:
            raise RuntimeError("Resident session is closed")

    def _clear(self):
        if self.context is not None:
            self.context._resident_clear()
            self.last_stats = self.context.resident_stats()
        self.data = {}
        self.identity = self.data_hashes = None
        self.phase1_key = self.phase1 = self.index = None
        self.index_streamed = False
        self.index_compact = False
        self.index_generated = False
        self.generated_neighbor_count = None
        self.generated_knn_scratch_bytes = 0
        self.compact_knn_stats = None
        self._phase1_cache.clear()
        self._phase1_cache_bytes = self._phase1_cache_hits = self._phase1_cache_evictions = 0
        self.generation += 1

    def clear(self):
        self._check()
        if self.active:
            raise RuntimeError("Cannot clear a running sweep")
        self._clear()

    def close(self):
        if self.closed:
            return
        self._check()
        if self.active:
            raise RuntimeError("Cannot close a running sweep")
        try:
            self._clear()
        finally:
            if self.context is not None:
                self.context.close()
                self.context = None
            self.closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        # Never call CUDA in a fork child or a different thread at destruction.
        try:
            if (os.getpid(), threading.get_ident()) == (self.process, self.thread):
                self.close()
        except Exception:
            pass

    @contextmanager
    def sweep(self, dataset_id, *, data=None, reuse=False):
        self._check()
        if self.active:
            raise RuntimeError("Nested/concurrent sweeps are not supported")
        if not isinstance(reuse, bool) or not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("Explicit boolean reuse and nonempty dataset_id required")
        if data is None and (not reuse or self.identity is None):
            raise ValueError("No retained dataset: supply data on the first sweep; reuse defaults to False")
        if self.identity is not None:
            if dataset_id != self.identity:
                raise ValueError("Dataset identity mismatch; clear explicitly before replacement")
            if data is not None:
                hashes = {k: _digest(v) for k, v in _normalize(data, copy=False).items()}
                if hashes != self.data_hashes:
                    raise ValueError("Dataset content mismatch; retained data were not replaced")
        if not reuse or self.identity is None:
            snapshots = _normalize(data, copy=True)
            hashes = {k: _digest(v) for k, v in snapshots.items()}
            if sum(v.nbytes for v in snapshots.values()) > self.maximum_bytes:
                raise ValueError("Dataset alone exceeds retained-memory budget")
            self._clear()
            if self.context is None:
                self.context = self.core.context(device_ordinal=self.device, stream_count=2)
            try:
                self.context._resident_begin(self.maximum_bytes)
                for name, array in snapshots.items():
                    self.context._resident_put("data." + name, array)
            except BaseException:
                self._clear()
                raise
            self.identity, self.data, self.data_hashes = dataset_id, snapshots, hashes
        self.active = True
        token = self.generation
        try:
            yield Sweep(self, token)
        except BaseException:
            # No half-built/rejected state can be silently reused next time.
            self._clear()
            raise
        finally:
            self.active = False
            if not reuse:
                self._clear()
            elif self.context is not None:
                self.last_stats = self.context.resident_stats()


class Sweep:
    def __init__(self, session, token):
        self.session, self.token = session, token

    def _check(self):
        self.session._check()
        if not self.session.active or self.token != self.session.generation:
            raise RuntimeError("Sweep handle is no longer active")

    @property
    def data(self):
        self._check()
        return dict(self.session.data)

    @property
    def context(self):
        self._check()
        return self.session.context

    def stats(self):
        return self.context.resident_stats()

    def prepare_phase1(self, k, *, configuration, builder, preflight=None):
        """One GPU k at a time, optionally bounded immutable host P1 snapshots.

        builder(context, x, index, configuration) returns the complete initial
        Phase1 dict.  A mode-4 builder additionally declares a keyword-only
        ``response`` argument; this is supplied only when exact kNN tiles are
        generated inside P1 and ``index`` is None.
        configuration must include MM/cuts/seeds/hybrid and all builder options.
        The builder's source is part of the key; hidden mutable closures are
        forbidden. Every changed k/config evicts the active GPU Phase1, not
        the dataset; an optional bounded host cache can retain initial P1.
        """
        self._check()
        if getattr(builder, "__closure__", None):
            raise ValueError("Phase1 builder must not hide mutable closure state")
        if not 1 <= int(k) <= len(self.session.data["x"]):
            raise ValueError("k outside supported range")
        s = self.session
        source = inspect.getsource(builder)
        key = json.dumps(dict(k=int(k), configuration=configuration,
                              dataset_id=s.identity, data_sha256=s.data_hashes,
                              builder_sha256=hashlib.sha256(source.encode()).hexdigest()),
                         sort_keys=True, allow_nan=False)
        if key == s.phase1_key:
            return s.index, dict(s.phase1), {"reused": True, "wall_seconds": 0.0,
                "source": "active", "index_rebuilt": False,
                "index_compact": s.index_compact,
                "index_generated": s.index_generated,
                "generated_neighbor_count": s.generated_neighbor_count,
                "generated_knn_scratch_bytes":
                    s.generated_knn_scratch_bytes}
        cached = s._phase1_cache.get(key)
        s.context._resident_clear("phase1")
        s.phase1_key = s.phase1 = s.index = None
        s.index_streamed = False
        s.index_compact = False
        s.index_generated = False
        s.generated_neighbor_count = None
        s.generated_knn_scratch_bytes = 0
        s.compact_knn_stats = None
        # A host-cache restore performs no new P1; do not report the previous
        # k's native allocation observation as this preparation's measurement.
        s.context.last_hybrid_capacity = None
        started = time.perf_counter()
        try:
            execution = preflight(s.context, s.data, int(k)) if preflight is not None else None
            capacity = execution['capacity_options'] if execution else None
            generated = bool(capacity and capacity['mode'] == 4)
            if generated and not hasattr(
                    s.context._core._library,
                    'manor_m_cuda_phase2_response_tiled'):
                raise RuntimeError(
                    'Generated exact-kNN Phase1 requires direct generated-index '
                    'Phase2 support')
            if execution and not generated:
                output = _index_storage(len(s.data['x']), int(k), execution['heap_bytes'],
                                        execution.get('spill_directory'))
                index, _ = s.context.response_knn(s.data['response'], int(k), output=output,
                    host_output=capacity['mode'] >= 2,
                    scratch_memory_bytes=execution['knn_scratch_bytes'])
                s.index_streamed = capacity['mode'] >= 2
            elif generated:
                index = None
            else:
                index, _ = s.context.response_knn(s.data["response"], int(k))
            if index is not None:
                index.setflags(write=False)
                # Exact kNN output is owned by this session. Validate its complete
                # range once per immutable generation; Phase2 may then skip the
                # repeated O(N*k) scan, but arbitrary borrowed arrays still scan.
                s.context._resident_validate_index(index)
            if cached is not None:
                p1 = dict(cached['model'])
            else:
                # Pass a detached JSON configuration, exactly the value in the key.
                config = json.loads(key)['configuration']
                if execution:
                    config['capacity_options'] = execution['capacity_options']
                builder_options = {}
                if generated:
                    if 'response' not in inspect.signature(builder).parameters:
                        raise TypeError(
                            "A mode-4 Phase1 builder must accept keyword response")
                    builder_options['response'] = s.data['response']
                p1 = builder(
                    s.context, s.data["x"], index, config, **builder_options)
            n, d = s.data["x"].shape
            for name in ("Tan0", "Cen0", "owner_mask0"):
                if name not in p1:
                    raise ValueError("Phase1 builder omitted required initial state")
            if p1["Cen0"].shape != (n, d) or p1["Tan0"].ndim != 3 or p1["Tan0"].shape[::2] != (n, d) or p1["owner_mask0"].shape != (n,):
                raise ValueError("Phase1 initial-state shape mismatch")
            if cached is None:
                p1 = {name: _snapshot(value, value.dtype) if isinstance(value, np.ndarray) else value
                      for name, value in p1.items()}
            for name in ("Tan0", "Cen0"):
                if p1[name].dtype != np.float32:
                    raise ValueError("Phase1 Tan/Cen must be float32")
                s.context._resident_put("phase1." + name, p1[name])
            if cached is None:
                p1["owner_mask0"] = _snapshot(p1["owner_mask0"] != 0, np.uint8)
            s.context._resident_put("phase1.owner", p1["owner_mask0"])
            owners = int(np.count_nonzero(p1['owner_mask0']))
            if generated:
                # P1 already consumed exact generated kNN tiles.  Keep no
                # complete N*k or owner*k host/retained-device index: Phase2
                # regenerates scratch-bounded exact tiles and consumes them
                # directly on device.
                # Dataset/config/source hashes identify a cached immutable P1;
                # there is intentionally no absent index buffer to hash.
                index = None
                s.index_generated = True
                s.generated_neighbor_count = int(k)
                actual_capacity = getattr(
                    s.context, 'last_hybrid_capacity', None)
                records = []
                if isinstance(actual_capacity, dict):
                    if 'knn_scratch_bytes' in actual_capacity:
                        records.append(actual_capacity)
                    else:
                        records.extend(value for value in actual_capacity.values()
                                       if isinstance(value, dict) and
                                       'knn_scratch_bytes' in value)
                actual_scratch = (min(record['knn_scratch_bytes']
                                      for record in records)
                                  if records else execution['knn_scratch_bytes'])
                s.generated_knn_scratch_bytes = int(actual_scratch)
            else:
                extra_compact = owners * int(k) * 4 if owners < n else 0
                if (not s.index_streamed and execution and
                        s.context.resident_stats()['retained_bytes'] + extra_compact > s.maximum_bytes):
                    # P1 was affordable resident, but actual owner compaction would
                    # exceed the explicit retained budget. Keep the host index for P2.
                    s.context._resident_clear('phase1.response_index')
                    s.index_streamed = True
                if not s.index_streamed:
                    s.context._resident_prepare_owner_index(index, p1["owner_mask0"])
            # Recompute the exact input on a revisit. Require exactly the
            # original bytes before restoring P1; a nested N prefix is not reuse.
            if (cached is not None and not generated and
                    _digest(index) != cached['index_hash']):
                raise RuntimeError(
                    'regenerated exact index differs from cached Phase1 input')
            s.index, s.phase1, s.phase1_key = index, p1, key
            eligible = (cached is None and s.phase1_host_limit
                        and all(isinstance(v, np.ndarray) for v in p1.values())
                        and sum(v.nbytes for v in p1.values()) <= s.phase1_host_limit)
            # Avoid a full index hash if an oversized model cannot be cached;
            # when eligible, include that hash in the preparation wall time.
            index_hash = _digest(index) if eligible and not generated else None
            elapsed = time.perf_counter() - started
            if cached is not None:
                s._phase1_cache.move_to_end(key)
                s._phase1_cache_hits += 1
            elif eligible:
                s._remember_phase1(key, p1, index_hash, elapsed)
        except BaseException:
            s.context._resident_clear("phase1")
            raise
        return index, dict(p1), dict(reused=cached is not None, wall_seconds=elapsed,
            source='host_snapshot' if cached is not None else 'built',
            index_rebuilt=not generated,
            source_wall_seconds=cached['source_wall'] if cached is not None else elapsed,
            index_compact=s.index_compact,
            index_generated=s.index_generated,
            generated_neighbor_count=s.generated_neighbor_count,
            generated_knn_scratch_bytes=s.generated_knn_scratch_bytes,
            compact_knn_stats=(None if s.compact_knn_stats is None
                else vars(s.compact_knn_stats)))

    def phase2(self, base_h, rho_schedule, *, manifold_dimension, **options):
        self._check()
        s = self.session
        if s.phase1 is None:
            raise RuntimeError("Prepare the exact Phase1 configuration before Phase2")
        overlap = set(options) & set(DATA_NAMES)
        if overlap:
            raise ValueError("Query/truth data belong to the dataset snapshot, not cell options")
        p1 = s.phase1
        truth = {k: v for k, v in s.data.items() if k not in ("x", "response")}
        return s.context.phase2(s.data["x"], s.data["response"], s.index,
            p1["Tan0"], p1["Cen0"], p1["owner_mask0"], base_h, rho_schedule,
            manifold_dimension, index_is_compact_owner_order=s.index_compact,
            generated_neighbor_count=s.generated_neighbor_count,
            knn_scratch_memory_bytes=s.generated_knn_scratch_bytes,
            **truth, **options)
