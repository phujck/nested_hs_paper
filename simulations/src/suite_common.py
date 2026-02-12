import itertools
import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, logm


EPS = 1e-12


@dataclass
class SuiteProfile:
    name: str
    cutoff_list: list
    beta_list: list
    coupling_list: list
    mc_samples: int
    quad_order: int
    edgeworth_order: int


def ensure_dir(path):
    import os

    os.makedirs(path, exist_ok=True)


def kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def partial_trace(rho, dims, keep):
    dims = list(dims)
    keep = list(keep)
    traced = [i for i in range(len(dims)) if i not in keep]

    labels = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if 2 * len(dims) > len(labels):
        raise ValueError("Too many subsystems for einsum trace")

    left = labels[: len(dims)]
    right = labels[len(dims) : 2 * len(dims)]
    for idx in traced:
        right[idx] = left[idx]

    out_labels = [left[i] for i in keep] + [right[i] for i in keep]
    expr = "".join(left + right) + "->" + "".join(out_labels)
    reshaped = rho.reshape(dims + dims)
    traced_rho = np.einsum(expr, reshaped)
    dim_keep = int(np.prod([dims[i] for i in keep]))
    return traced_rho.reshape((dim_keep, dim_keep))


def hermitize(a):
    return 0.5 * (a + a.conj().T)


def normalize_density(rho):
    tr = np.trace(rho)
    if abs(tr) < EPS:
        raise ValueError("Cannot normalize zero-trace matrix")
    return rho / tr


def thermal_density(H, beta):
    return expm(-beta * H)


def hmf_from_rho(rho, beta):
    h = -(1.0 / beta) * logm(rho)
    return hermitize(h)


def trace_distance(rho_a, rho_b):
    vals = np.linalg.eigvalsh(hermitize(rho_a - rho_b))
    return 0.5 * np.sum(np.abs(vals))


def fro_error(rho_a, rho_b):
    return float(np.linalg.norm(rho_a - rho_b, ord="fro"))


def destroy(n):
    a = np.zeros((n, n), dtype=np.complex128)
    for i in range(1, n):
        a[i - 1, i] = math.sqrt(i)
    return a


def oscillator_ops(n, omega):
    a = destroy(n)
    adag = a.conj().T
    h = omega * (adag @ a + 0.5 * np.eye(n, dtype=np.complex128))
    x = (a + adag) / math.sqrt(2.0 * omega)
    return h, x


def qutrit_operators():
    e01 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.complex128)
    e12 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.complex128)
    e02 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.complex128)
    lam2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=np.complex128)
    lam7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=np.complex128)
    lam5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=np.complex128)
    lam3 = np.diag([1.0, -1.0, 0.0]).astype(np.complex128)
    lam8 = (1.0 / math.sqrt(3.0)) * np.diag([1.0, 1.0, -2.0]).astype(np.complex128)
    return {
        "l1": e01,
        "l2": lam2,
        "l3": lam3,
        "l4": e02,
        "l5": lam5,
        "l6": e12,
        "l7": lam7,
        "l8": lam8,
    }


def qutrit_models():
    ops = qutrit_operators()
    hs_comm = np.diag([0.0, 0.7, 1.6]).astype(np.complex128)
    f_comm = np.diag([0.0, 1.0, 2.0]).astype(np.complex128)

    hs_noncomm = np.diag([0.0, 0.9, 1.8]).astype(np.complex128)
    f_noncomm = ops["l1"] + 0.45 * ops["l6"] + 0.2 * ops["l4"]

    return {
        "hs_comm": hs_comm,
        "f_comm": f_comm,
        "hs_noncomm": hs_noncomm,
        "f_noncomm": f_noncomm,
    }


def sigma_x():
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)


def sigma_z():
    return np.array([[1, 0], [0, -1]], dtype=np.complex128)


def spin_bath_ops(num_spins, omega_list, coupling_list):
    if len(omega_list) != num_spins or len(coupling_list) != num_spins:
        raise ValueError("omega_list and coupling_list must match num_spins")

    sx = sigma_x()
    sz = sigma_z()
    eye2 = np.eye(2, dtype=np.complex128)

    hb = np.zeros((2**num_spins, 2**num_spins), dtype=np.complex128)
    b = np.zeros_like(hb)

    for i in range(num_spins):
        mats_h = [eye2] * num_spins
        mats_b = [eye2] * num_spins
        mats_h = list(mats_h)
        mats_b = list(mats_b)
        mats_h[i] = omega_list[i] * sx
        mats_b[i] = coupling_list[i] * sz
        hb = hb + kron_all(mats_h)
        b = b + kron_all(mats_b)

    return hb, b


def exact_reduced_density(Hs, Hb, Hi, beta):
    ds = Hs.shape[0]
    db = Hb.shape[0]
    htot = np.kron(Hs, np.eye(db)) + np.kron(np.eye(ds), Hb) + Hi
    rho_tot = thermal_density(htot, beta)
    rho_s = partial_trace(rho_tot, [ds, db], keep=[0])
    return normalize_density(rho_s)


def exact_reduced_density_bosonic(Hs, f, bath_omegas, bath_cs, cutoff, beta):
    ds = Hs.shape[0]
    h_b_terms = []
    x_terms = []
    for w in bath_omegas:
        hk, xk = oscillator_ops(cutoff, w)
        h_b_terms.append(hk)
        x_terms.append(xk)

    eye_b = [np.eye(cutoff, dtype=np.complex128) for _ in bath_omegas]
    h_b = np.zeros((cutoff ** len(bath_omegas),) * 2, dtype=np.complex128)
    x_b = np.zeros_like(h_b)

    for i, hk in enumerate(h_b_terms):
        mats = list(eye_b)
        mats[i] = hk
        h_b = h_b + kron_all(mats)

    for i, xk in enumerate(x_terms):
        mats = list(eye_b)
        mats[i] = bath_cs[i] * xk
        x_b = x_b + kron_all(mats)

    hi = np.kron(f, x_b)
    return exact_reduced_density(Hs, h_b, hi, beta)


def commuting_gaussian_prediction(Hs, f, bath_omegas, bath_cs, beta):
    lam = 0.0
    for w, c in zip(bath_omegas, bath_cs):
        lam += (c * c) / (2.0 * w * w)
    rho = expm(-beta * (Hs - lam * (f @ f)))
    return normalize_density(rho), lam


def derivative_scalar(fun, order, h=1e-3):
    # Five-point finite-difference formulas for 2nd and 4th derivatives.
    if order == 2:
        return (-fun(2 * h) + 16 * fun(h) - 30 * fun(0.0) + 16 * fun(-h) - fun(-2 * h)) / (12 * h * h)
    if order == 4:
        return (
            fun(-2 * h)
            - 4 * fun(-h)
            + 6 * fun(0.0)
            - 4 * fun(h)
            + fun(2 * h)
        ) / (h**4)
    raise ValueError("Supported orders are 2 and 4")


def cumulants_from_bath_free_energy(Hb, B, beta):
    def free_log(theta):
        z = np.trace(expm(-beta * (Hb + theta * B)))
        return float(np.log(np.real_if_close(z)))

    d2 = derivative_scalar(free_log, 2)
    d4 = derivative_scalar(free_log, 4)
    alpha2 = -(1.0 / beta) * d2 / math.factorial(2)
    alpha4 = -(1.0 / beta) * d4 / math.factorial(4)
    return alpha2, alpha4, d2, d4


def fit_hmf_in_basis(hmf, basis_ops):
    vecs = [op.reshape(-1) for op in basis_ops]
    a = np.stack(vecs, axis=1)
    b = hmf.reshape(-1)
    coeffs, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    recon = sum(coeffs[i] * basis_ops[i] for i in range(len(basis_ops)))
    resid = np.linalg.norm(hmf - recon, ord="fro") / max(np.linalg.norm(hmf, ord="fro"), EPS)
    return coeffs, float(np.real_if_close(resid))


def gauss_hermite_expectation_matrix(Hs, f, beta, sigma, quad_order):
    from numpy.polynomial.hermite import hermgauss

    xs, ws = hermgauss(quad_order)
    acc = np.zeros_like(Hs, dtype=np.complex128)
    norm = 1.0 / math.sqrt(math.pi)
    for x, w in zip(xs, ws):
        val = math.sqrt(2.0) * x
        acc = acc + norm * w * expm(-beta * (Hs + sigma * val * f))
    return normalize_density(acc)


def edgeworth_k2_k4_expectation(Hs, f, beta, k2, k4, quad_order):
    from numpy.polynomial.hermite import hermgauss

    sigma = math.sqrt(max(k2, EPS))
    xs, ws = hermgauss(quad_order)
    norm = 1.0 / math.sqrt(math.pi)

    acc0 = np.zeros_like(Hs, dtype=np.complex128)
    acc4 = np.zeros_like(Hs, dtype=np.complex128)
    for x, w in zip(xs, ws):
        z = math.sqrt(2.0) * x
        h4 = z**4 - 6 * z**2 + 3
        mat = expm(-beta * (Hs + sigma * z * f))
        acc0 = acc0 + norm * w * mat
        acc4 = acc4 + norm * w * h4 * mat

    corr = (k4 / (24.0 * sigma**4)) * acc4
    rho = acc0 + corr
    rho = hermitize(rho)
    evals = np.linalg.eigvalsh(rho)
    if np.min(evals) < -1e-10:
        shift = (-np.min(evals) + 1e-10)
        rho = rho + shift * np.eye(rho.shape[0])
    return normalize_density(rho)


def static_spin_distribution(couplings):
    # For Hb=0 and independent spins at infinite temperature.
    n = len(couplings)
    values = []
    for bits in itertools.product([-1, 1], repeat=n):
        b = 0.0
        for c, s in zip(couplings, bits):
            b += c * s
        values.append(b)
    values = np.array(values, dtype=float)
    probs = np.full(values.shape, 1.0 / values.size)
    return values, probs


def exact_static_mixture(Hs, f, beta, values, probs):
    rho = np.zeros_like(Hs, dtype=np.complex128)
    for v, p in zip(values, probs):
        rho = rho + p * expm(-beta * (Hs + v * f))
    return normalize_density(rho)


def moments_from_distribution(values, probs):
    mean = np.sum(values * probs)
    centered = values - mean
    m2 = np.sum((centered**2) * probs)
    m4 = np.sum((centered**4) * probs)
    k4 = m4 - 3 * (m2**2)
    return float(mean), float(m2), float(m4), float(k4)


def commutator_chain(Hs, f, depth=3):
    chain = [hermitize(f)]
    cur = f.copy()
    for _ in range(depth):
        cur = Hs @ cur - cur @ Hs
        chain.append(hermitize(1j * cur))
    return chain


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    raise TypeError(f"Unsupported type: {type(obj)}")


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=to_jsonable)

