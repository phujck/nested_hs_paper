import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


EPS = 1e-12


@dataclass
class SuiteProfileV2:
    name: str
    cutoff_list: list
    beta_list: list
    coupling_list: list
    mode_count: int
    eta: float
    omega_c: float
    omega_max_factor: float
    path_tau_points: int
    path_samples: int
    scalar_samples: int
    mc_scaling_samples: list
    fd_step: float
    spin_omegas: list
    spin_c_template: list


def ensure_dir(path):
    import os

    os.makedirs(path, exist_ok=True)


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


def kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def hermitize(a):
    return 0.5 * (a + a.conj().T)


def normalize_density(rho):
    rho = hermitize(np.asarray(rho, dtype=np.complex128))
    tr = np.trace(rho)
    if abs(tr) < EPS:
        raise ValueError("Cannot normalize matrix with near-zero trace.")
    rho = rho / tr
    return hermitize(rho)


def thermal_density(H, beta):
    H = hermitize(np.asarray(H, dtype=np.complex128))
    evals = np.linalg.eigvalsh(H)
    shift = float(np.min(np.real(evals)))
    return expm(-beta * (H - shift * np.eye(H.shape[0], dtype=np.complex128)))


def trace_distance(rho_a, rho_b):
    vals = np.linalg.eigvalsh(hermitize(rho_a - rho_b))
    return float(0.5 * np.sum(np.abs(vals)))


def fro_error(rho_a, rho_b):
    return float(np.linalg.norm(rho_a - rho_b, ord="fro"))


def partial_trace(rho, dims, keep):
    dims = list(dims)
    keep = list(keep)
    traced = [i for i in range(len(dims)) if i not in keep]
    labels = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if 2 * len(dims) > len(labels):
        raise ValueError("Too many subsystems for einsum trace.")
    left = labels[: len(dims)]
    right = labels[len(dims) : 2 * len(dims)]
    for idx in traced:
        right[idx] = left[idx]
    out_labels = [left[i] for i in keep] + [right[i] for i in keep]
    expr = "".join(left + right) + "->" + "".join(out_labels)
    traced_rho = np.einsum(expr, rho.reshape(dims + dims))
    dim_keep = int(np.prod([dims[i] for i in keep]))
    return traced_rho.reshape((dim_keep, dim_keep))


def qutrit_commuting_model():
    energies = np.array([0.0, 0.9, 1.8], dtype=float)
    hs = np.diag(energies).astype(np.complex128)
    fvals = np.array([0.0, 1.0, 2.0], dtype=float)
    f = np.diag(fvals).astype(np.complex128)
    return energies, hs, fvals, f


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


def exact_reduced_density(Hs, Hb, Hi, beta):
    ds = Hs.shape[0]
    db = Hb.shape[0]
    htot = np.kron(Hs, np.eye(db, dtype=np.complex128)) + np.kron(np.eye(ds, dtype=np.complex128), Hb) + Hi
    rho_tot = thermal_density(htot, beta)
    rho_s = partial_trace(rho_tot, [ds, db], keep=[0])
    return normalize_density(rho_s)


def exact_reduced_density_bosonic(Hs, f, bath_omegas, bath_cs, cutoff, beta):
    h_b_terms = []
    x_terms = []
    for w in bath_omegas:
        hk, xk = oscillator_ops(cutoff, w)
        h_b_terms.append(hk)
        x_terms.append(xk)
    eye_b = [np.eye(cutoff, dtype=np.complex128) for _ in bath_omegas]
    dim_b = cutoff ** len(bath_omegas)
    h_b = np.zeros((dim_b, dim_b), dtype=np.complex128)
    x_b = np.zeros((dim_b, dim_b), dtype=np.complex128)
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


def ohmic_spectral_density(omega, eta, omega_c, g):
    omega = np.asarray(omega, dtype=float)
    return 2.0 * eta * (g * g) * omega * np.exp(-omega / omega_c)


def lambda_ohmic_continuum(eta, omega_c, g):
    return float((2.0 * eta * omega_c / np.pi) * (g * g))


def discretize_ohmic_bath(num_modes, omega_c, eta, g, omega_max_factor=6.0):
    omega_max = float(omega_max_factor * omega_c)
    edges = np.linspace(0.0, omega_max, num_modes + 1)
    widths = np.diff(edges)
    omegas = 0.5 * (edges[:-1] + edges[1:])
    jvals = ohmic_spectral_density(omegas, eta, omega_c, g)
    c2 = (2.0 / np.pi) * jvals * omegas * widths
    cs = np.sqrt(np.maximum(c2, 0.0))
    return omegas, cs, widths


def lambda_from_discrete(omegas, cs):
    omegas = np.asarray(omegas, dtype=float)
    cs = np.asarray(cs, dtype=float)
    return float(np.sum((cs * cs) / (2.0 * omegas * omegas)))


def commuting_gaussian_prediction(Hs, f, beta, lam):
    rho = thermal_density(Hs - lam * (f @ f), beta)
    return normalize_density(rho)


def _logmeanexp(samples):
    smax = float(np.max(samples))
    return smax + math.log(float(np.mean(np.exp(samples - smax))))


def diagonal_probabilities_from_integrated_field(energies, fvals, beta, x_samples):
    energies = np.asarray(energies, dtype=float)
    fvals = np.asarray(fvals, dtype=float)
    x_samples = np.asarray(x_samples, dtype=float)
    logs = np.array([-beta * e + _logmeanexp(f * x_samples) for e, f in zip(energies, fvals)], dtype=float)
    logs = logs - np.max(logs)
    weights = np.exp(logs)
    probs = weights / np.sum(weights)
    return probs


def density_from_probabilities(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / np.sum(probs)
    return normalize_density(np.diag(probs.astype(np.complex128)))


def scalar_hs_density(energies, fvals, beta, lam, n_samples, rng):
    c_beta = max(2.0 * beta * lam, 0.0)
    x_samples = math.sqrt(max(c_beta, EPS)) * rng.standard_normal(int(n_samples))
    probs = diagonal_probabilities_from_integrated_field(energies, fvals, beta, x_samples)
    rho = density_from_probabilities(probs)
    stats = {
        "x_mean": float(np.mean(x_samples)),
        "x_var": float(np.var(x_samples)),
        "c_beta": float(c_beta),
    }
    return rho, probs, stats


def kernel_discrete(tau, beta, omegas, cs):
    tau = np.asarray(tau, dtype=float)
    omegas = np.asarray(omegas, dtype=float)
    cs = np.asarray(cs, dtype=float)
    coeff = (cs * cs) / (2.0 * omegas)
    denom = np.sinh(0.5 * beta * omegas)
    denom = np.maximum(denom, EPS)
    out = np.zeros_like(tau, dtype=float)
    for k in range(len(omegas)):
        out += coeff[k] * np.cosh(omegas[k] * (0.5 * beta - np.abs(tau))) / denom[k]
    return out


def build_tau_covariance(beta, n_tau, omegas, cs):
    dt = beta / float(n_tau)
    taus = (np.arange(n_tau, dtype=float) + 0.5) * dt
    cov = np.zeros((n_tau, n_tau), dtype=float)
    for i in range(n_tau):
        for j in range(i, n_tau):
            d = abs(taus[i] - taus[j])
            d = min(d, beta - d)
            kij = float(kernel_discrete(np.array([d]), beta, omegas, cs)[0])
            cov[i, j] = kij
            cov[j, i] = kij
    return taus, dt, cov


def sample_gaussian_paths(cov, n_samples, rng):
    evals, evecs = np.linalg.eigh(0.5 * (cov + cov.T))
    evals = np.maximum(evals, 0.0)
    root = evecs @ np.diag(np.sqrt(evals))
    z = rng.standard_normal((int(n_samples), cov.shape[0]))
    return z @ root.T


def path_hs_density(energies, fvals, beta, omegas, cs, n_tau, n_samples, rng):
    _, dt, cov = build_tau_covariance(beta, n_tau, omegas, cs)
    xi_paths = sample_gaussian_paths(cov, n_samples, rng)
    x_samples = dt * np.sum(xi_paths, axis=1)
    probs = diagonal_probabilities_from_integrated_field(energies, fvals, beta, x_samples)
    rho = density_from_probabilities(probs)
    stats = {
        "x_mean": float(np.mean(x_samples)),
        "x_var": float(np.var(x_samples)),
        "c_beta_grid": float(dt * dt * np.sum(cov)),
    }
    return rho, probs, stats


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
        mats_h[i] = float(omega_list[i]) * sx
        mats_b[i] = float(coupling_list[i]) * sz
        hb = hb + kron_all(mats_h)
        b = b + kron_all(mats_b)
    return hb, b


def finite_difference_weights(order, radius):
    offsets = np.arange(-radius, radius + 1, dtype=float)
    n = offsets.size
    a = np.zeros((n, n), dtype=float)
    for i in range(n):
        a[i, :] = offsets**i / math.factorial(i)
    b = np.zeros(n, dtype=float)
    b[order] = 1.0
    w = np.linalg.solve(a, b)
    return offsets, w


def finite_difference_derivative(fun, order, h, radius=4):
    offsets, weights = finite_difference_weights(order, radius)
    vals = np.array([fun(float(o * h)) for o in offsets], dtype=float)
    return float(np.dot(weights, vals) / (h**order))


def source_deformed_log_partition(Hb, B, beta, theta):
    z = np.trace(expm(-beta * (Hb + theta * B)))
    return float(np.log(np.real_if_close(z)))


def integrated_even_cumulants(Hb, B, beta, orders=(2, 4, 6), h=7.5e-3):
    alpha = {}
    derivative = {}
    stability = {}

    fun = lambda th: source_deformed_log_partition(Hb, B, beta, th)
    for order in orders:
        d_h = finite_difference_derivative(fun, order=order, h=h, radius=4)
        d_h2 = finite_difference_derivative(fun, order=order, h=0.5 * h, radius=4)
        derivative[order] = d_h2
        stability_denom = max(abs(d_h2), abs(d_h), 1e-8)
        stability[order] = float(abs(d_h2 - d_h) / stability_denom)
        alpha[order] = float((1.0 / beta) * d_h2 / math.factorial(order))

    return alpha, derivative, stability


def commuting_truncated_prediction(Hs, f, beta, alpha_map, max_order):
    h_eff = np.array(Hs, dtype=np.complex128)
    for order, coeff in alpha_map.items():
        if order <= max_order:
            h_eff = h_eff - float(coeff) * np.linalg.matrix_power(f, int(order))
    rho = thermal_density(h_eff, beta)
    return normalize_density(rho), h_eff


def diag_probs(rho):
    p = np.real_if_close(np.diag(rho)).astype(float)
    p = np.maximum(p, 0.0)
    if np.sum(p) < EPS:
        raise ValueError("Diagonal probabilities are degenerate.")
    return p / np.sum(p)


def level_shift(energies, probs, beta, idx, ref=0):
    probs = np.asarray(probs, dtype=float)
    energies = np.asarray(energies, dtype=float)
    num = (energies[idx] - energies[ref]) + (1.0 / beta) * math.log(max(probs[idx], EPS) / max(probs[ref], EPS))
    return float(num)


def density_diagnostics(rho):
    tr = np.trace(rho)
    herm = np.linalg.norm(rho - rho.conj().T, ord="fro")
    return float(abs(tr - 1.0)), float(herm)
