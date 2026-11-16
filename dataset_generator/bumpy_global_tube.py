"""Bumpy v2 / quadratic link in a uniform, fixed global normal ball."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np

SEED = 2026101901
SCHEMA = "mfm-bumpy-global-normal-ball.v1"
CS = (0.25, 0.5, 0.75)


def label(c):
    if c not in CS:
        raise ValueError("c must be 0.25, 0.5 or 0.75")
    return "c%03d" % round(100 * c)


def sample_chunk(c, split, start, count, generator, geometry_scale):
    # The c key is deliberately absent: groups are paired in Q, direction,
    # radius fraction and standardized response noise, not in their noisy X.
    def rng(stream):
        return np.random.default_rng(np.random.SeedSequence([SEED, split, start, stream]))
    unit = rng(1).normal(size=(count, 3))
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    g = generator._family_geometry(unit, generator._family_geometry_parameters("v2"))
    tangent = np.zeros((count, 2, 20))
    tangent[:, 0, :3], tangent[:, 1, :3] = g["tangent_first"], g["tangent_second"]
    clean = np.zeros((count, 20))
    clean[:, :3] = geometry_scale * g["radius"][:, None] * unit
    direction = rng(2).normal(size=(count, 20))
    coefficients = np.einsum("nmd,nd->nm", tangent, direction)
    direction -= np.einsum("nm,nmd->nd", coefficients, tangent)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radial_fraction = rng(3).random(count) ** (1.0 / 18)
    noise = direction * (c * math.sqrt(20) * radial_fraction[:, None])
    assert np.max(np.abs(np.einsum("nmd,nd->nm", tangent, noise))) < 1e-10
    assert np.max(np.linalg.norm(noise, axis=1)) <= c * math.sqrt(20) * (1+1e-12)
    return dict(x_noisy=(clean + noise).astype("f4"), tangent=tangent.astype("f4"),
                clean3=clean[:, :3].astype("f4"), latent=unit,
                standard=rng(4).standard_normal((count, 6)).astype("f4"))


def generate(path, c, plan, generator, data_api, train_count=1_000_000):
    identity = dict(schema=SCHEMA, c=c, global_reach=math.sqrt(20),
        global_reach_method=plan["canonical_identity"]["global_reach_calibration"],
        geometry_scale=plan["canonical_identity"]["geometry_scale"],
        geometry_formula_sha256=plan["canonical_identity"]["geometry_formula_sha256"],
        ambient_dimension=20, intrinsic_dimension=2, train_count=train_count,
        validation_count=5000, testing_count=5000, relative_y_noise=.01,
        link="quadratic phase 15*xi_i+2*xi_j*xi_l, six sine/cosine components",
        normal_law="uniform complete 18-dimensional normal ball; radius c*global_reach",
        response_covariance="fixed independent 262144-point equal-area clean-sphere rule",
        seed=SEED, chunk_size=65536, c_groups_paired=True)
    assert generator._family_formula_hash(generator._family_geometry_parameters("v2")) == identity["geometry_formula_sha256"]
    path = Path(path)
    if path.exists():
        with h5py.File(path, "r") as f:
            assert f.attrs["state"] == "complete" and json.loads(f.attrs["identity"]) == identity
        return identity
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.h5")
    rotation = np.asarray(plan["response_rotation"])
    unit = generator._family_equal_area_unit(262144)
    geo = generator._family_geometry(unit, generator._family_geometry_parameters("v2"))
    covariance = data_api.covariance_root(data_api.response_map(
        (geo["radius"][:, None] * unit) @ rotation.T, 2)).astype("f4")
    with h5py.File(temporary, "a") as f:
        encoded = json.dumps(identity, sort_keys=True)
        if "identity" in f.attrs:
            assert f.attrs["identity"] == encoded
        f.attrs.update(identity=encoded, state="generating")
        for split, name in enumerate(("train", "validation", "testing"), 1):
            n = train_count if name == "train" else 5000
            group = f.require_group("geometry/" + name)
            response = f.require_group("maps/quadratic6/" + name)
            fields = {"x_noisy": (20,), "tangent": (2, 20), "clean3": (3,), "latent": (3,)}
            for key, shape in fields.items():
                if key not in group:
                    group.create_dataset(key, (n,)+shape, dtype="f8" if key == "latent" else "f4",
                                         chunks=(min(n, 8192),)+shape)
            for key in ("clean", "observed"):
                if key not in response:
                    response.create_dataset(key, (n, 6), dtype="f4", chunks=(min(n, 8192), 6))
            for start in range(int(group.attrs.get("completed", 0)), n, 65536):
                stop = min(n, start + 65536)
                values = sample_chunk(c, split, start, stop-start, generator, identity["geometry_scale"])
                clean_y = data_api.response_map((values["clean3"].astype(float) /
                    identity["geometry_scale"]) @ rotation.T, 2)
                for key in fields:
                    group[key][start:stop] = values[key]
                response["clean"][start:stop] = clean_y
                response["observed"][start:stop] = clean_y + data_api.noise_product_fixed_order(
                    covariance, values["standard"])
                f.flush()
                group.attrs["completed"] = stop
                f.flush()
        f.attrs["state"] = "complete"
    os.replace(temporary, path)
    return identity


def load_data(dataset, setup=None, *, include_testing=True, expected_c=None):
    from cuda_core_data import Stage2Data
    with h5py.File(dataset, "r") as f:
        identity = json.loads(f.attrs["identity"])
        assert f.attrs["state"] == "complete" and identity["schema"] == SCHEMA
        if expected_c is not None and identity["c"] != expected_c:
            raise ValueError("noisy X belongs to a different tube width")
        if setup is not None and setup.get("dataset_identity", identity) != identity:
            raise ValueError("saved result belongs to a different dataset")
        arrays = dict(X=f["geometry/train/x_noisy"][:], Y=f["maps/quadratic6/train/observed"][:])
        splits = [("validation", "valid", "observed")]
        if include_testing:
            splits.append(("testing", "test", "clean"))
        for name, suffix, truth in splits:
            arrays.update({"X"+suffix: f["geometry/"+name+"/x_noisy"][:],
                           "Y"+suffix: f["maps/quadratic6/"+name+"/"+truth][:],
                           "T"+suffix: f["geometry/"+name+"/tangent"][:]})
    return Stage2Data(identity, arrays)
