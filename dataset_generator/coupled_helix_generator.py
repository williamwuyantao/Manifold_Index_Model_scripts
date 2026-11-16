"""Generate the paper's coupled Meyer helix with its three response maps."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
from contextlib import nullcontext
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAPS = ("natural_polynomial3", "natural_rational3", "natural_exponential3")
STAGE1_MAPS = MAPS
SCHEMA = "mfm-coupled-helix-data.v1"


MAP_SPECS = {'natural_polynomial3': {'family': 'complex_field', 'kind': 'polynomial', 'origin': [-1.0, 0.25], 'extent': [2.0, 1.5], 'numerator': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [0.6, 0.2], 1.0], 'features': '(Re(F/abs(F)), Im(F/abs(F)), log(abs(F)))'}, 'natural_rational3': {'family': 'complex_field', 'kind': 'mobius', 'origin': [-1.0, 0.3], 'extent': [2.0, 1.5], 'degree': 10, 'a': [-1.3, 0.1], 'b': [1.3, 1.8], 'features': '(Re(F/abs(F)), Im(F/abs(F)), log(abs(F)))'}, 'natural_exponential3': {'family': 'complex_field', 'kind': 'exponential', 'origin': [0.0, 0.0], 'extent': [1.0, 1.0], 'h': [0.0, 8.0, 8.0], 'features': '(Re(F/abs(F)), Im(F/abs(F)), log(abs(F)))'}}


def quantiles(x):
    return dict(zip(("min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"),
                    map(float, np.quantile(x, [0, .01, .05, .25, .5, .75, .95, .99, 1]))))


def one_dim(t, d, mode):
    """Exact public MATLAB coordinate formulas, with stable analytic derivatives."""
    i = np.arange(1, d + 1, dtype=np.float64)
    centers = (i - 1) / (d - 1)
    z = t[:, None] - centers
    if mode == "gaussian":
        w = 1 / d
        x = np.exp(-z*z / (4*w*w)) / ((2*np.pi)**.25 * w**.5)
        first = -z / (2*w*w)
        second = np.full_like(z, -1 / (2*w*w))
        return x, x*first, x*(first*first + second)
    w = (1 + .3*np.cos(10*i)) / d
    v = (1 + .3*np.sin(10*i)) / d
    az = np.abs(z)
    envelope = np.exp(-z*z / (2*w*(2*w + az))) / ((2*np.pi)**.25 * w**.25)
    log_first = -z*(az + 4*w) / (2*w*(2*w + az)**2)
    log_second = -4*w / (2*w + az)**3
    envelope_first = envelope * log_first
    envelope_second = envelope * (log_first**2 + log_second)
    if mode == "envelope":
        return envelope, envelope_first, envelope_second, 10*i + z/v, 1/v
    phase = 10*i + z/v
    c, s = np.cos(phase), np.sin(phase)
    return (envelope*c,
            envelope_first*c - envelope*s/v,
            envelope_second*c - 2*envelope_first*s/v - envelope*c/v**2)


def square_surface(uv, kind):
    coupled = kind == "square_coupled_helix"
    mode = "envelope" if coupled else ("gaussian" if kind == "square_gaussian" else "helix")
    left, right = one_dim(uv[:, 0], 5, mode), one_dim(uv[:, 1], 4, mode)
    outer = lambda a, b: (a[:, :, None]*b[:, None, :]).reshape(len(uv), 20)
    x = outer(left[0], right[0])
    xs, xt = outer(left[1], right[0]), outer(left[0], right[1])
    xss, xst, xtt = outer(left[2], right[0]), outer(left[1], right[1]), outer(left[0], right[2])
    if coupled:
        phase = (left[3][:, :, None] + right[3][:, None, :]).reshape(len(uv), 20)
        a = np.broadcast_to(left[4][:, None], (5, 4)).reshape(20)
        b = np.broadcast_to(right[4][None, :], (5, 4)).reshape(20)
        c, s = np.cos(phase), np.sin(phase)
        xss, xst, xtt = (xss*c - 2*xs*s*a - x*c*a*a,
                         xst*c - xs*s*b - xt*s*a - x*c*a*b,
                         xtt*c - 2*xt*s*b - x*c*b*b)
        xs, xt, x = xs*c - x*s*a, xt*c - x*s*b, x*c
    return x, np.stack((xs, xt), axis=2), np.stack((xss, xst, xtt), axis=2)


def geometry(jac, hess, angle_count=128):
    r00 = np.linalg.norm(jac[:, :, 0], axis=1)
    q0 = jac[:, :, 0]/r00[:, None]
    r01 = np.sum(q0*jac[:, :, 1], axis=1)
    residual = jac[:, :, 1] - q0*r01[:, None]
    r11 = np.linalg.norm(residual, axis=1)
    if min(r00.min(), r11.min()) <= 1e-10:
        raise ValueError("surface differential is singular on sampled points")
    tangent = np.stack((q0, residual/r11[:, None]), axis=2)
    normal_hess = hess - np.einsum("nda,nak->ndk", tangent, np.einsum("nda,ndk->nak", tangent, hess))
    a, b, c = 1/r00, -r01/(r00*r11), 1/r11
    b11 = normal_hess[:, :, 0]*a[:, None]**2
    b12 = normal_hess[:, :, 0]*(a*b)[:, None] + normal_hess[:, :, 1]*(a*c)[:, None]
    b22 = (normal_hess[:, :, 0]*b[:, None]**2 +
           2*normal_hess[:, :, 1]*(b*c)[:, None] + normal_hess[:, :, 2]*c[:, None]**2)
    coeff = np.stack((b11, 2*b12, b22), axis=2)
    gram = np.einsum("ndi,ndj->nij", coeff, coeff)
    theta = np.arange(angle_count)*np.pi/angle_count
    v = np.stack((np.cos(theta)**2, np.sin(theta)*np.cos(theta), np.sin(theta)**2))
    maximum = np.zeros(len(jac))
    for lo in range(0, len(jac), 4096):
        values = np.einsum("ia,nij,ja->na", v, gram[lo:lo+4096], v, optimize=True)
        maximum[lo:lo+4096] = np.sqrt(np.maximum(values.max(axis=1), 0))
    frobenius = np.sqrt(np.sum(b11*b11 + 2*b12*b12 + b22*b22, axis=1))
    # ||d B(v,v)/d theta|| <= 2||B||_F, nearest sampled angle <= pi/(2*M).
    curvature_upper = maximum + np.pi*frobenius/angle_count
    singular = np.linalg.svd(jac, compute_uv=False)
    return tangent, maximum, curvature_upper, 1/curvature_upper, r00*r11, singular


def normal_noise(tangent, reach, rng, sigma=.5, safety=.98):
    d = tangent.shape[1]
    radius2 = rng.chisquare(d, len(reach))
    limit = (.9*math.sqrt(d)/sigma)**2
    while np.any(radius2 > limit):
        bad = radius2 > limit
        radius2[bad] = rng.chisquare(d, bad.sum())
    direction = rng.normal(size=(len(reach), d))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    displacement = direction*(sigma*np.sqrt(radius2)/math.sqrt(d)*safety*reach)[:, None]
    displacement -= np.einsum("nda,na->nd", tangent, np.einsum("nda,nd->na", tangent, displacement))
    ratio = np.linalg.norm(displacement, axis=1)/reach
    normality = np.max(np.abs(np.einsum("nda,nd->na", tangent, displacement)))
    assert ratio.max() <= .9*safety*(1+1e-12)
    assert normality <= 1e-10*max(1., reach.max())
    return displacement, ratio, float(normality)


def response(uv, name, basis=None, map_specs=None):
    """Return three response values and their analytic parameter derivatives."""
    uv = np.asarray(uv, dtype=np.float64)
    spec = (MAP_SPECS if map_specs is None else map_specs)[name]
    from numpy.polynomial import Polynomial
    pair = lambda value: complex(*value) if isinstance(value, list) else value
    dz = np.array([spec["extent"][0], 1j*spec["extent"][1]])
    z = complex(*spec["origin"])+uv@dz
    if spec["kind"] == "polynomial":
        polynomial = Polynomial([pair(value) for value in spec["numerator"]])
        f = polynomial(z)
        logder = polynomial.deriv()(z)/f
    elif spec["kind"] == "mobius":
        a, b = complex(*spec["a"]), complex(*spec["b"])
        f = ((z-a)/(z-b))**spec["degree"]
        logder = spec["degree"]*(1/(z-a)-1/(z-b))
    elif spec["kind"] == "exponential":
        polynomial = Polynomial(spec["h"])
        f = np.exp(polynomial(z))
        logder = polynomial.deriv()(z)
    else:
        raise ValueError("unknown complex field")
    unit = f/np.abs(f)
    delta = logder[:, None]*dz
    dunit = 1j*unit[:, None]*delta.imag
    return (np.column_stack((unit.real, unit.imag, np.log(np.abs(f)))),
            np.stack((dunit.real, dunit.imag, delta.real), axis=1))


def map_spec(name):
    return dict(MAP_SPECS[name])


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            sha.update(block)
    return sha.hexdigest()


def calibration(directory):
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text())
    if report["case"] != "square_coupled_helix" or not report["complete"]:
        raise ValueError("a completed coupled-helix geometry calibration is required")
    with np.load(directory / "samples.npz") as values:
        uv = values["latent"][:report["N"]].copy()
        reach = values["local_reach"][:report["N"]].copy()
    raw, jac, _ = square_surface(uv, "square_coupled_helix")
    center = raw.mean(axis=0)
    _, basis = np.linalg.eigh(np.cov(raw, rowvar=False))
    return report, uv, reach, raw, jac, center, basis[:, -3:]


def event(**value):
    print(json.dumps(value), flush=True)


def generate(args):
    import h5py
    from scipy.spatial import cKDTree
    report, uv, reach, raw, jac, center, basis = calibration(args.calibration)
    summary = json.loads(Path(args.screen).read_text())["response_calibration"]
    if summary["calibration_samples_sha256"] != digest(args.calibration/"samples.npz"):
        raise ValueError("response screen and geometry calibration differ")
    scale = summary["scale"]
    names = list(MAPS if not getattr(args, "maps", None) else args.maps.split(","))
    if any(name not in summary["maps"] for name in names):
        raise ValueError("requested response is not in the saved screen")
    tree = cKDTree(uv)
    target = args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial.h5")
    with target.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if target.exists():
            with h5py.File(target, "r") as source:
                if source.attrs.get("state") != "complete" or source.attrs["train_count"] != args.n:
                    raise ValueError("existing final dataset has a different identity")
            event(stage="dataset_already_complete", path=str(target))
            return
        identity = dict(schema=SCHEMA, train_count=args.n, validation_count=args.queries,
                        testing_count=args.queries, seed=args.seed, chunk_rows=args.chunk,
                        ambient_dimension=20, intrinsic_dimension=2, relative_y_noise=.01,
                        parameter_domain="[0,1]^2", parameter_sampling="uniform parameter area",
                        scale=scale, center=center.tolist(), response_maps=names,
                        response_map_specs={name:map_spec(name) for name in names},
                        calibration_samples_sha256=summary["calibration_samples_sha256"],
                        response_screen_sha256=digest(args.screen),
                        noise_rule="sigma=.5; preprojection chi20 truncated at .9 reach; normal projection; safety=.98",
                        reach_rule="32 nearest calibration points in (s,t), inverse-distance-squared interpolation, capped by analytic 128-angle focal bound")
        geometry_source = getattr(args,"geometry_source",None)
        if geometry_source is not None:
            if Path(geometry_source).resolve() == target.resolve():
                raise ValueError("new response data must not overwrite its geometry source")
            with h5py.File(geometry_source,"r") as source:
                parent = json.loads(source.attrs["identity"])
                if (source.attrs["state"] != "complete" or parent["train_count"] < args.n
                        or parent["validation_count"] != args.queries or parent["testing_count"] != args.queries
                        or parent["calibration_samples_sha256"] != identity["calibration_samples_sha256"]):
                    raise ValueError("stored geometry does not match this calibration and split sizes")
            identity["geometry_source"] = dict(file=Path(geometry_source).name,
                sha256=digest(geometry_source),identity=parent,reuse="exact stored geometry; newly evaluated responses on its stored latent coordinates")
        start = time.perf_counter()
        with h5py.File(temporary, "a") as output, (h5py.File(geometry_source,"r") if geometry_source is not None else nullcontext()) as stored:
            encoded = json.dumps(identity, sort_keys=True)
            if "identity" in output.attrs and output.attrs["identity"] != encoded:
                raise ValueError("partial dataset identity differs")
            output.attrs.update(identity=encoded, schema=SCHEMA, state="generating", train_count=args.n)
            for split_code, split in enumerate(("validation", "testing", "train")):
                n = args.n if split == "train" else args.queries
                group = output.require_group("geometry/"+split)
                shapes = dict(x_clean=(n, 20), x_noisy=(n, 20), latent=(n, 2), local_reach=(n,))
                if split != "train":
                    shapes["tangent"] = (n, 2, 20)
                for key, shape in shapes.items():
                    if key not in group:
                        group.create_dataset(key, shape=shape, dtype="f4", chunks=(min(16384, n),)+shape[1:])
                for name in names:
                    response_group = output.require_group("maps/"+name+"/"+split)
                    m = summary["maps"][name]["dimension"]
                    for key in ("clean", "observed"):
                        if key not in response_group:
                            response_group.create_dataset(key, (n, m), dtype="f4", chunks=(min(16384,n),m))
                begin = int(group.attrs.get("completed_rows", 0))
                for lo in range(begin, n, args.chunk):
                    hi = min(lo+args.chunk, n)
                    if stored is not None:
                        fields = {key:stored["geometry/"+split+"/"+key][lo:hi] for key in shapes}
                        latent = fields["latent"].astype(float)
                        ratio = np.linalg.norm(fields["x_noisy"]-fields["x_clean"],axis=1)/fields["local_reach"]
                        normality = None
                    else:
                        rng = np.random.default_rng(np.random.SeedSequence([args.seed, split_code, lo, 0]))
                        latent = rng.random((hi-lo, 2))
                        x, j, h = square_surface(latent, "square_coupled_helix")
                        tangent, _, _, focal, _, _ = geometry(j*scale, h*scale)
                        distance, indices = tree.query(latent, k=32, workers=4)
                        weight = 1/np.maximum(distance, 1e-12)**2
                        local = np.minimum(np.sum(reach[indices]*weight, axis=1)/weight.sum(axis=1), focal)
                        noise_rng = np.random.default_rng(np.random.SeedSequence([args.seed, split_code, lo, 1]))
                        displacement, ratio, normality = normal_noise(tangent, local, noise_rng)
                        clean = (x-center)*scale
                        fields = dict(x_clean=clean, x_noisy=clean+displacement, latent=latent, local_reach=local)
                        if split != "train":
                            fields["tangent"] = tangent.transpose(0, 2, 1)
                    for key, value in fields.items():
                        group[key][lo:hi] = value.astype(np.float32)
                    for number, name in enumerate(names):
                        info = summary["maps"][name]
                        values, _ = response(latent, name, basis, summary.get("map_specs"))
                        values = (values-np.asarray(info["mean"]))*info["normalization_factor"]
                        covariance = np.asarray(info["covariance"])
                        eigenvalue, eigenvector = np.linalg.eigh(covariance)
                        root = eigenvector*np.sqrt(np.maximum(eigenvalue, 0))[None, :]
                        yrng = np.random.default_rng(np.random.SeedSequence([args.seed, split_code, lo, 2+number]))
                        noisy_y = values + .01*yrng.standard_normal(values.shape) @ root.T
                        output["maps/"+name+"/"+split+"/clean"][lo:hi] = values.astype("f4")
                        output["maps/"+name+"/"+split+"/observed"][lo:hi] = noisy_y.astype("f4")
                    output.flush()
                    group.attrs["completed_rows"] = hi
                    output.flush()
                    event(stage="dataset_chunk", split=split, rows=hi, total=n,
                          normality=normality,
                          maximum_noise_reach_ratio=float(ratio.max()))
            output.attrs["state"] = "complete"
            output.flush()
        os.replace(temporary, target)
        save(target.with_suffix(".json"), dict(identity, state="complete", sha256=digest(target),
                                              bytes=target.stat().st_size))
        event(stage="dataset_complete", path=str(target), bytes=target.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n', type=int, default=1000000)
    parser.add_argument('--queries', type=int, default=5000)
    parser.add_argument('--chunk', type=int, default=65536)
    parser.add_argument('--seed', type=int, default=2026100702)
    parser.add_argument('--maps', default=','.join(MAPS))
    parser.add_argument('--calibration', type=Path, default=ROOT/'data/coupled_helix')
    parser.add_argument('--screen', type=Path, default=ROOT/'config/coupled_experiment.json')
    parser.add_argument('--geometry-source', type=Path)
    args = parser.parse_args()
    if min(args.n, args.queries, args.chunk) <= 0:
        parser.error('row counts must be positive')
    generate(args)


if __name__ == '__main__':
    main()
