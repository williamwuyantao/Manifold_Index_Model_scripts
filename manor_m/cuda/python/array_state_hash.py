"""Bounded parallelism across independent, complete array hashes.

Every batch is blocking, including its error path. No digest/pointer cache,
array copy, per-array chunk hash, or overlap with mutation of the next state.
The caller must keep every array unchanged until the batch returns, just as
for the serial hasher. Each array still uses the original complete SHA-256.
"""

from concurrent.futures import ThreadPoolExecutor, wait
import operator


def hash_arrays(values, names, hash_function):
    """Original serial ordering; also the default for existing callers."""
    return {name: hash_function(values[name]) for name in names}


class ArrayHashPool:
    """Explicit one-to-four-worker, synchronous state-hash batch."""

    def __init__(self, workers):
        if isinstance(workers, bool):
            raise ValueError("hash workers must be an integer from 1 to 4")
        try:
            workers = operator.index(workers)
        except TypeError as error:
            raise ValueError("hash workers must be an integer from 1 to 4") from error
        if not 1 <= workers <= 4:
            raise ValueError("hash workers must be an integer from 1 to 4")
        self.workers = workers
        self._pool = (ThreadPoolExecutor(max_workers=workers,
                      thread_name_prefix="mfm-state-sha") if workers > 1 else None)
        self._closed = False

    def __call__(self, values, names, hash_function):
        if self._closed:
            raise RuntimeError("state-hash pool is closed")
        names = tuple(names)
        if len(set(names)) != len(names):
            raise ValueError("duplicate state-hash names")
        # Resolve every required input before any thread starts reading it.
        arrays = tuple(values[name] for name in names)
        if self._pool is None:
            return dict(zip(names, map(hash_function, arrays)))
        futures = []
        try:
            for value in arrays:
                futures.append(self._pool.submit(hash_function, value))
        finally:
            # Also drain all submitted readers if submission itself fails.
            wait(futures)
        # All readers have finished even when an earlier result raises.
        return {name: future.result() for name, future in zip(names, futures)}

    def close(self):
        if not self._closed:
            self._closed = True
            if self._pool is not None:
                self._pool.shutdown(wait=True)

    def __enter__(self):
        if self._closed:
            raise RuntimeError("state-hash pool is closed")
        return self

    def __exit__(self, *error):
        self.close()
