"""Canonical Stage2 data preparation, without MATLAB or a GPU context.

Array layout is C-contiguous [point, coordinate], and tangent layout is
[point, manifold coordinate, ambient coordinate].  To preserve the existing
Stage2 scientific definition, each N below 1M fits its response-noise
covariance on that exact nested training prefix; 1M and larger use 1M.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


REFERENCE_COUNT = 1_000_000
QUERY_COUNT = 5000
MAPS = (
    dict(id=0, name="torus_iso_k8p75", family="isotropic", parameters={"k": 8.75},
         definition="phi_i=8.75*z_i"),
    dict(id=1, name="torus_aniso_12_15_18", family="anisotropic",
         parameters={"frequencies": [12, 15, 18]}, definition="phi_i=[12,15,18]_i*z_i"),
    dict(id=2, name="torus_twist_k15_b2", family="quadratic_twist",
         parameters={"k": 15, "beta": 2},
         definition="phi_i=15*z_i+2*z_j*z_l, {j,l}=other coordinates"),
    dict(id=3, name="torus_chirp_k14_b2p5", family="cubic_chirp",
         parameters={"k": 14, "beta": 2.5}, definition="phi_i=14*z_i+2.5*z_i^3"),
    dict(id=4, name="torus_chirp_k10p5_b3", family="cubic_chirp",
         parameters={"k": 10.5, "beta": 3}, definition="phi_i=10.5*z_i+3*z_i^3"),
)


def response_map(z, map_id):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != 3 or map_id not in range(5):
        raise ValueError("response map requires [N,3] and map_id in 0..4")
    if map_id == 0:
        phase = 8.75 * z
    elif map_id == 1:
        phase = z * np.asarray([12., 15., 18.])
    elif map_id == 2:
        phase = 15 * z + 2 * z[:, [1, 2, 0]] * z[:, [2, 0, 1]]
    elif map_id == 3:
        phase = 14 * z + 2.5 * z ** 3
    else:
        phase = 10.5 * z + 3 * z ** 3
    y = np.empty((len(z), 6), dtype=np.float32)
    y[:, 0::2], y[:, 1::2] = np.cos(phase), np.sin(phase)
    # Match MATLAB's two SINGLE operations after casting cos/sin, not a
    # fused double expression followed by a final float32 cast.
    np.add(y, np.float32(1), out=y)
    np.multiply(y, np.float32(.5), out=y)
    return y


def covariance_root(clean):
    y = np.asarray(clean, dtype=np.float64)
    centered = y - y.mean(axis=0)
    covariance = centered.T @ centered / len(y)
    covariance = .5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    root = .01 * ((vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T)
    return .5 * (root + root.T)


def noise_product_fixed_order(root, standard):
    """Match MATLAB SINGLE 6-by-6 multiplication term for term.

    NumPy's BLAS-backed matmul may reassociate or fuse this short dot product.
    Stage2's canonical MATLAB construction rounds every multiply and add in a
    left-to-right six-term sequence, so keep that tiny dimension explicit.
    """
    root = np.ascontiguousarray(root, dtype=np.float32)
    standard = np.ascontiguousarray(standard, dtype=np.float32)
    if root.shape != (6, 6) or standard.ndim != 2 or standard.shape[1] != 6:
        raise ValueError("noise product requires root [6,6] and samples [N,6]")
    result = np.empty_like(standard)
    scratch = np.empty(len(standard), dtype=np.float32)
    for output in range(6):
        total = result[:, output]
        np.multiply(standard[:, 0], root[output, 0], out=total)
        for term in range(1, 6):
            np.multiply(standard[:, term], root[output, term], out=scratch)
            np.add(total, scratch, out=total)
    return result


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array)).hexdigest()


@dataclass
class Stage2Data:
    identity: dict
    arrays: dict
    prefix_builder: object = None

    @property
    def hashes(self):
        return {key: array_hash(value) for key, value in self.arrays.items()}

    def variance(self, view):
        key = {"validation": "Yvalid", "testing": "Ytest"}[view]
        coordinate = np.var(self.arrays[key].astype(np.float64), axis=0, ddof=1)
        if not np.isfinite(coordinate).all() or np.any(coordinate <= 0):
            raise ValueError("nonpositive/nonfinite response variance")
        return coordinate, float(coordinate.sum())

    def for_n(self, n):
        """Return the exact Stage2 inputs for one nested training prefix."""
        n = int(n)
        if n <= 0:
            raise ValueError("N must be positive")
        if self.prefix_builder is not None:
            return self.prefix_builder(n)
        if len(self.arrays["X"]) < n or len(self.arrays["Y"]) < n:
            raise ValueError("canonical training prefix is too short")
        arrays = dict(self.arrays)
        arrays["X"] = np.ascontiguousarray(arrays["X"][:n])
        arrays["Y"] = np.ascontiguousarray(arrays["Y"][:n])
        return Stage2Data(self.identity, arrays)


def noisy_prefix(clean, standard, clean_validation, standard_validation, n):
    """Match MATLAB's per-N single-precision response-noise construction."""
    n = int(n)
    if not 0 < n <= len(clean) or len(standard) < n:
        raise ValueError("invalid response-noise prefix")
    count = min(n, REFERENCE_COUNT)
    root = covariance_root(clean[:count]).astype(np.float32)
    y = np.ascontiguousarray(clean[:n] + noise_product_fixed_order(root, standard[:n]),
                             dtype=np.float32)
    yv = np.ascontiguousarray(clean_validation
                              + noise_product_fixed_order(root, standard_validation),
                              dtype=np.float32)
    return y, yv


def load_canonical(dataset, setup, *, maximum_training_count=REFERENCE_COUNT,
                   include_testing=True):
    """Read canonical prefixes on this host; never download or refit per N.

    Stage2 keeps the one-million-row default.  Stage3 may request a
    larger nested prefix, while the response-noise covariance remains frozen
    to the first one million rows exactly as specified by ``noisy_prefix``.
    """
    import h5py

    d, map_id = int(setup["D"]), int(setup["map_id"])
    maximum_training_count = int(maximum_training_count)
    if maximum_training_count < REFERENCE_COUNT:
        raise ValueError("canonical loader must retain the first 1M noise reference")
    if map_id not in range(5):
        raise ValueError("unknown response map")
    with h5py.File(Path(dataset), "r") as source:
        identity = json.loads(source.attrs["identity_json"])
        if (identity.get("schema") != "mfm-bumpy-sphere-family.v1"
                or identity.get("dataset") != "bumpy_sphere_" + setup["version"]
                or identity.get("ambient_d") != d
                or identity.get("manifold_d") != 2
                or not bool(source.attrs["complete"])):
            raise ValueError("wrong or incomplete canonical dataset")
        scale = float(identity["geometry_scale"])
        if not np.isfinite(scale) or scale <= 0 or "y_noise_relative" not in identity:
            raise ValueError("invalid scale/noise identity")

        def read(name, count, columns):
            dataset = source[name]
            if dataset.ndim != 2 or dataset.shape[0] < count or dataset.shape[1] < columns:
                raise ValueError("canonical shape mismatch: " + name)
            return np.asarray(dataset[:count, :columns], dtype=np.float32, order="C")

        x = read("x_train", maximum_training_count, d)
        clean_x = read("xclean_train", maximum_training_count, 3)
        y0 = read("y_train_clean", maximum_training_count, 6)
        y0_noisy = read("y_train_observed", maximum_training_count, 6)
        xv = read("x_validation", QUERY_COUNT, d)
        cxv = read("xclean_validation", QUERY_COUNT, 3)
        yv0 = read("y_validation_clean", QUERY_COUNT, 6)
        xt = cx_t = yt0 = None
        if include_testing:
            xt = read("x_eval", QUERY_COUNT, d)
            cx_t = read("xclean_eval", QUERY_COUNT, 3)
            yt0 = read("y_eval_clean", QUERY_COUNT, 6)
        yv_noisy = read("y_validation_observed", QUERY_COUNT, 6)
        tangent_v = np.asarray(source["tangent_validation"], dtype=np.float32)
        if tangent_v.shape != (QUERY_COUNT, d, 2):
            raise ValueError("canonical tangent shape mismatch")
        tangent_v = np.ascontiguousarray(tangent_v.transpose(0, 2, 1))
        tangent_t = None
        if include_testing:
            tangent_t = np.asarray(source["tangent_eval"], dtype=np.float32)
            if tangent_t.shape != (QUERY_COUNT, d, 2):
                raise ValueError("canonical tangent shape mismatch")
            tangent_t = np.ascontiguousarray(tangent_t.transpose(0, 2, 1))
        # h5py already exposes the numerical matrices used by MATLAB h5read
        # for these repository datasets.  Transposing the response rotation
        # here selects the inverse map and changes every derived Y response.
        stored_rotation = np.asarray(source["response_rotation"], dtype=np.float64)
        old_noise_root = np.asarray(source["y_noise_covariance_root"], dtype=np.float64)

    sample = clean_x[:2048].astype(np.float64) / scale
    rotations = (stored_rotation, stored_rotation.T)
    errors = [float(np.max(np.abs(response_map(sample @ r.T, 0).astype(np.float64)
                                   - y0[:len(sample)]))) for r in rotations]
    which = int(errors[1] < errors[0])
    rotation, error = rotations[which], errors[which]
    if error > 2e-4:
        raise ValueError("stored Y0 response orientation parity failed")
    if map_id == 0:
        clean, cv, yt = y0, yv0, yt0
    else:
        clean = response_map((clean_x.astype(np.float64) / scale) @ rotation.T, map_id)
        cv = response_map((cxv.astype(np.float64) / scale) @ rotation.T, map_id)
        yt = (response_map((cx_t.astype(np.float64) / scale) @ rotation.T, map_id)
              if include_testing else None)
    standard = np.linalg.solve(old_noise_root, (y0_noisy.astype(np.float64) - y0).T).T.astype(np.float32)
    standard_v = np.linalg.solve(old_noise_root, (yv_noisy.astype(np.float64) - yv0).T).T.astype(np.float32)
    combined = dict(schema="mfm-derived-response-setup.v1", base_dataset_identity=identity,
        setup_id=setup["id"], map=MAPS[map_id], relative_y_noise=.01,
        noise_covariance_source="clean responses on the fixed first 1M training rows",
        standard_noise_source="shared canonical Y0 noise reused across maps",
        response_rotation=rotation.tolist(), stored_Y0_parity_max_abs=error,
        validation_count=QUERY_COUNT, testing_count=QUERY_COUNT)
    static = dict(Xvalid=xv, Tvalid=tangent_v)
    if include_testing:
        static.update(Xtest=xt, Ytest=yt, Ttest=tangent_t)

    def build_prefix(n):
        y, yv = noisy_prefix(clean, standard, cv, standard_v, n)
        arrays = dict(X=np.ascontiguousarray(x[:n]), Y=y, Yvalid=yv, **static)
        if any(not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("nonfinite canonical inputs")
        return Stage2Data(combined, arrays)

    arrays = build_prefix(REFERENCE_COUNT).arrays
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("nonfinite canonical inputs")
    return Stage2Data(combined, arrays, prefix_builder=build_prefix)
