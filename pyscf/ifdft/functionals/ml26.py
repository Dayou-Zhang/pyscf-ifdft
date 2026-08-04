"""Composite feature evaluator for fixed-density ML26 energies.

The Rung3.5 formulas and stretched-AO construction in this module are adapted
from ``pyscf-r35dft`` (Apache License 2.0). Public calculations use the value
path; derivative helpers remain private functional infrastructure.
"""

from __future__ import annotations

import math
import time
from types import SimpleNamespace

import numpy as np
from pyscf import gto, lib, scf
from pyscf.dft import numint, xc_deriv
from scipy.special import erf

from ..features import build_context
from ..features import FeatureSetInfo
from ..grids import grids_are_equivalent
from ..ifxc import IFXCEvaluator
from ..mixers.base import MixerResult


_BETAS = (0.1, 0.2, 1.0, 3.0)
_BETA_TAGS = ("0p1", "0p2", "1p0", "3p0")
_ML25_NFEATURES = 66
_NMAX = 0.6
_M11PO_C = np.asarray(
    (-0.170265960, 0.32610431, 0.0, 0.173830277, -0.337808383),
    dtype=np.float64,
)
_M11PO_D = np.asarray(
    (0.0, -0.002379857, 2.026450260, -0.943023593, -4.195668115),
    dtype=np.float64,
)


def _stretch_ao_molecule(mol, beta: float):
    if mol.cart:
        raise ValueError("ML26 Rung3.5 features do not support Cartesian AOs")
    for shell in range(mol.nbas):
        if mol.bas_nctr(shell) > 1:
            raise ValueError(
                "ML26 Rung3.5 features require at most one contraction per shell"
            )

    stretched = mol.copy()
    stretched.build()
    original_env = mol._env.copy()
    for shell in range(stretched.nbas):
        angular_momentum = stretched._bas[shell, gto.ANG_OF]
        nprim = stretched._bas[shell, gto.NPRIM_OF]
        exponent_start = stretched._bas[shell, gto.PTR_EXP]
        coefficient_start = stretched._bas[shell, gto.PTR_COEFF]
        for primitive in range(nprim):
            exponent = original_env[exponent_start + primitive]
            coefficient = original_env[coefficient_start + primitive]
            denominator = exponent + beta
            stretched._env[exponent_start + primitive] = exponent * beta / denominator
            stretched._env[coefficient_start + primitive] = (
                coefficient
                * (2.0 * math.pi * beta) ** 0.75
                / denominator**1.5
                * (beta / denominator) ** angular_momentum
            )
    return stretched


def _nga_energy(rho: np.ndarray, beta: float) -> np.ndarray:
    a01 = (4.0 * math.pi / 3.0) ** (2.0 / 3.0)
    a11 = 4.869
    # ML26 is pinned to the coefficient used to generate its development
    # reference vectors.  This predates a later pyscf-r35dft NGA correction.
    coefficients = (
        (1.0, 1.3493, 0.0),
        (0.893696, 3.642, 0.0),
        (0.889355, 0.0, 0.06202),
    )
    cl = -(3.0 / 4.0) * (6.0 / math.pi) ** (1.0 / 3.0)
    y2_factor = beta**-1 * (4.0 * math.pi / 3.0) ** (2.0 / 3.0)

    density = np.maximum(rho[:, 0], 1e-10)
    gradient_squared = np.maximum(np.sum(rho[:, 1:4] ** 2, axis=1), 1e-12)
    x32 = gradient_squared**0.75 / density**2
    y2 = y2_factor * density ** (2.0 / 3.0)
    numerator = a01 * y2 + a11 * x32 * y2
    denominator = np.full_like(numerator, 1e-22)
    for y_power in range(3):
        for x_power in range(3):
            denominator += (
                coefficients[y_power][x_power]
                * x32**x_power
                * y2**y_power
            )
    return numerator / denominator * cl * density ** (4.0 / 3.0)


def _one_minus_elf(rho: np.ndarray, factor: float) -> np.ndarray:
    ctl = 0.3 * (6.0 * math.pi**2) ** (2.0 / 3.0)
    density = np.maximum(rho[:, 0], 1e-10)
    tau = np.maximum(rho[:, 5], 1e-10)
    gradient_squared = np.sum(rho[:, 1:4] ** 2, axis=1)
    tau_weizsaecker = gradient_squared / (8.0 * density)
    tau_lda = ctl * density ** (5.0 / 3.0)
    elf = 1.0 / (1.0 + ((tau - tau_weizsaecker) / tau_lda) ** 2)
    return 1.0 - factor * elf


def _r35_values(rho: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Return pointwise ``(local_hybrid_by_spin, b05)`` values."""
    density = np.maximum(rho[:, 0], 1e-10)
    eta = np.maximum(np.abs(rho[:, 6]), 1e-10)
    e35 = -(density ** (2.0 / 3.0)) * eta ** (4.0 / 3.0)
    enga = _nga_energy(rho, beta)
    enga_for_normalization = np.minimum(enga, -1e-10)
    n35 = (e35 - 2e-6) / (enga_for_normalization - 1e-6)
    fn35 = 0.5 * (1.0 - erf(5.0 * (n35 - _NMAX)))

    ymax = (256.0 / 3.0) ** (1.0 / 3.0)
    yfactor = beta**-0.5 * (4.0 * math.pi / 3.0) ** (1.0 / 3.0)
    sd = 0.5 * (1.0 - erf(yfactor * density ** (1.0 / 3.0) - ymax))

    ra, rb = density
    eb1 = e35[0] * rb / ra + e35[1] * ra / rb
    eb05 = 0.5 * fn35[0] * fn35[1] * eb1 * sd[0] * sd[1]

    omelf = _one_minus_elf(rho, 1.0)
    om2elf = _one_minus_elf(rho, 2.0)
    efit0 = (enga - e35) * sd
    local_hybrid = efit0 * sum(
        coefficient * omelf**power
        for power, coefficient in enumerate(_M11PO_C)
    )
    b05_factor = np.full_like(eb05, _M11PO_D[0])
    for power, coefficient in enumerate(_M11PO_D[1:], start=1):
        b05_factor += coefficient * (om2elf[0] ** power + om2elf[1] ** power)
    return local_hybrid, eb05 * b05_factor


def _r35_values_and_derivatives(
    rho: np.ndarray, beta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Rung3.5 values and analytic derivatives.

    Derivative slots are ``(n, grad_x, grad_y, grad_z, tau, eta)``.  Values
    clipped by the existing regularizers have a zero derivative on the
    clipped branch.
    """
    rho = np.asarray(rho)
    if rho.ndim != 3 or rho.shape[:2] != (2, 7):
        raise ValueError(f"Expected Rung3.5 rho shape (2, 7, ngrid), got {rho.shape}")
    complex_step = np.iscomplexobj(rho)

    nraw = rho[:, 0]
    grad = rho[:, 1:4]
    tauraw = rho[:, 5]
    etaraw = rho[:, 6]
    n_active = np.real(nraw) > 1e-10
    n = (
        np.where(n_active, nraw, 1e-10)
        if complex_step
        else np.maximum(nraw, 1e-10)
    )
    eta_sign = np.sign(np.real(etaraw))
    eta_active = np.abs(np.real(etaraw)) > 1e-10
    eta = (
        np.where(eta_active, eta_sign * etaraw, 1e-10)
        if complex_step
        else np.maximum(np.abs(etaraw), 1e-10)
    )
    ngrid = n.shape[1]

    e35 = -(n ** (2.0 / 3.0)) * eta ** (4.0 / 3.0)
    de35 = np.zeros((2, 6, ngrid), dtype=rho.dtype)
    de35[:, 0] = (
        -(2.0 / 3.0) * n ** (-1.0 / 3.0) * eta ** (4.0 / 3.0)
        * n_active
    )
    de35[:, 5] = (
        -(4.0 / 3.0) * n ** (2.0 / 3.0) * eta ** (1.0 / 3.0)
        * eta_sign
        * eta_active
    )

    p_raw = np.sum(grad * grad, axis=1)
    p_active = np.real(p_raw) > 1e-12
    p = (
        np.where(p_active, p_raw, 1e-12)
        if complex_step
        else np.maximum(p_raw, 1e-12)
    )
    x = p**0.75 / n**2
    y_factor = beta**-1 * (4.0 * math.pi / 3.0) ** (2.0 / 3.0)
    y = y_factor * n ** (2.0 / 3.0)
    dx = np.zeros_like(de35)
    dx[:, 0] = -2.0 * x / n * n_active
    dx[:, 1:4] = (
        1.5 * x[:, None, :] / p[:, None, :]
        * grad
        * p_active[:, None, :]
    )
    dy = np.zeros_like(de35)
    dy[:, 0] = (2.0 / 3.0) * y / n * n_active

    a01 = (4.0 * math.pi / 3.0) ** (2.0 / 3.0)
    a11 = 4.869
    coefficients = (
        (1.0, 1.3493, 0.0),
        (0.893696, 3.642, 0.0),
        (0.889355, 0.0, 0.06202),
    )
    numerator = y * (a01 + a11 * x)
    denominator = np.full_like(numerator, 1e-22)
    denominator_x = np.zeros_like(numerator)
    denominator_y = np.zeros_like(numerator)
    for y_power in range(3):
        for x_power in range(3):
            coefficient = coefficients[y_power][x_power]
            denominator += coefficient * x**x_power * y**y_power
            if x_power:
                denominator_x += (
                    coefficient * x_power * x ** (x_power - 1) * y**y_power
                )
            if y_power:
                denominator_y += (
                    coefficient * y_power * x**x_power * y ** (y_power - 1)
                )
    ratio = numerator / denominator
    ratio_x = (
        a11 * y * denominator - numerator * denominator_x
    ) / denominator**2
    ratio_y = (
        (a01 + a11 * x) * denominator - numerator * denominator_y
    ) / denominator**2
    cl = -(3.0 / 4.0) * (6.0 / math.pi) ** (1.0 / 3.0)
    n43 = n ** (4.0 / 3.0)
    enga = cl * n43 * ratio
    denga = cl * n43[:, None, :] * (
        ratio_x[:, None, :] * dx + ratio_y[:, None, :] * dy
    )
    denga[:, 0] += (
        cl * (4.0 / 3.0) * n ** (1.0 / 3.0) * ratio * n_active
    )

    enga_active = np.real(enga) < -1e-10
    enga_norm = (
        np.where(enga_active, enga, -1e-10)
        if complex_step
        else np.minimum(enga, -1e-10)
    )
    denga_norm = denga * enga_active[:, None, :]
    numerator35 = e35 - 2e-6
    denominator35 = enga_norm - 1e-6
    n35 = numerator35 / denominator35
    dn35 = (
        de35 * denominator35[:, None, :]
        - numerator35[:, None, :] * denga_norm
    ) / denominator35[:, None, :] ** 2
    fn35 = 0.5 * (1.0 - erf(5.0 * (n35 - _NMAX)))
    dfn35 = (
        -5.0
        / math.sqrt(math.pi)
        * np.exp(-25.0 * (n35 - _NMAX) ** 2)[:, None, :]
        * dn35
    )

    ymax = (256.0 / 3.0) ** (1.0 / 3.0)
    sd_factor = beta**-0.5 * (4.0 * math.pi / 3.0) ** (1.0 / 3.0)
    sd_arg = sd_factor * n ** (1.0 / 3.0) - ymax
    sd = 0.5 * (1.0 - erf(sd_arg))
    dsd = np.zeros_like(de35)
    dsd[:, 0] = (
        -np.exp(-(sd_arg**2))
        / math.sqrt(math.pi)
        * (sd_factor / 3.0)
        * n ** (-2.0 / 3.0)
        * n_active
    )

    def one_minus_elf(factor):
        ctl = 0.3 * (6.0 * math.pi**2) ** (2.0 / 3.0)
        tau_active = np.real(tauraw) > 1e-10
        tau = (
            np.where(tau_active, tauraw, 1e-10)
            if complex_step
            else np.maximum(tauraw, 1e-10)
        )
        a = tau - p_raw / (8.0 * n)
        q = a / (ctl * n ** (5.0 / 3.0))
        dq = np.zeros_like(de35)
        dq[:, 0] = (
            p_raw / (8.0 * n**2) * n_active
            - a * (5.0 / 3.0) / n * n_active
        ) / (ctl * n ** (5.0 / 3.0))
        dq[:, 1:4] = (
            -grad / (4.0 * n[:, None, :])
            / (ctl * n[:, None, :] ** (5.0 / 3.0))
        )
        dq[:, 4] = tau_active / (ctl * n ** (5.0 / 3.0))
        elf = 1.0 / (1.0 + q * q)
        value = 1.0 - factor * elf
        derivative = (
            2.0 * factor * q[:, None, :] * elf[:, None, :] ** 2 * dq
        )
        return value, derivative

    omelf, domelf = one_minus_elf(1.0)
    om2elf, dom2elf = one_minus_elf(2.0)

    efit = (enga - e35) * sd
    defit = (denga - de35) * sd[:, None, :] + (
        enga - e35
    )[:, None, :] * dsd
    local_factor = np.zeros_like(efit)
    dlocal_factor = np.zeros_like(de35)
    for power, coefficient in enumerate(_M11PO_C):
        local_factor += coefficient * omelf**power
        if power:
            dlocal_factor += (
                coefficient
                * power
                * omelf[:, None, :] ** (power - 1)
                * domelf
            )
    local_hybrid = efit * local_factor
    dlocal_hybrid = (
        defit * local_factor[:, None, :]
        + efit[:, None, :] * dlocal_factor
    )

    na, nb = n
    ea, eb = e35
    eb1 = ea * nb / na + eb * na / nb
    deb1 = np.zeros_like(de35)
    deb1[0] = nb[None, :] / na[None, :] * de35[0]
    deb1[1] = na[None, :] / nb[None, :] * de35[1]
    deb1[0, 0] += (
        -ea * nb / na**2 + eb / nb
    ) * n_active[0]
    deb1[1, 0] += (
        ea / na - eb * na / nb**2
    ) * n_active[1]

    b05_factor = np.full(
        ngrid, _M11PO_D[0], dtype=rho.dtype
    )
    db05_factor = np.zeros_like(de35)
    for power, coefficient in enumerate(_M11PO_D[1:], start=1):
        b05_factor += coefficient * (
            om2elf[0] ** power + om2elf[1] ** power
        )
        db05_factor += (
            coefficient
            * power
            * om2elf[:, None, :] ** (power - 1)
            * dom2elf
        )

    b05 = (
        0.5
        * fn35[0]
        * fn35[1]
        * eb1
        * sd[0]
        * sd[1]
        * b05_factor
    )
    db05 = np.zeros_like(de35)
    db05[0] = 0.5 * (
        dfn35[0] * fn35[1] * eb1 * sd[0] * sd[1] * b05_factor
        + fn35[0] * fn35[1] * deb1[0] * sd[0] * sd[1] * b05_factor
        + fn35[0] * fn35[1] * eb1 * dsd[0] * sd[1] * b05_factor
        + fn35[0] * fn35[1] * eb1 * sd[0] * sd[1] * db05_factor[0]
    )
    db05[1] = 0.5 * (
        fn35[0] * dfn35[1] * eb1 * sd[0] * sd[1] * b05_factor
        + fn35[0] * fn35[1] * deb1[1] * sd[0] * sd[1] * b05_factor
        + fn35[0] * fn35[1] * eb1 * sd[0] * dsd[1] * b05_factor
        + fn35[0] * fn35[1] * eb1 * sd[0] * sd[1] * db05_factor[1]
    )
    return local_hybrid, b05, dlocal_hybrid, db05


def _r35_rho(
    semilocal_rho: np.ndarray, eta: np.ndarray, *, out=None
) -> np.ndarray:
    shape = (2, 7, semilocal_rho.shape[-1])
    if out is None or out.shape != shape:
        rho = np.empty(shape, dtype=np.float64)
    else:
        rho = out
    rho[:, :4] = semilocal_rho[:, :4]
    rho[:, 4] = 0.0
    rho[:, 5] = semilocal_rho[:, 4]
    rho[:, 6] = eta
    rho[:, 0] = np.maximum(rho[:, 0], 1e-12)
    rho[:, 5] = np.maximum(rho[:, 5], 1e-12)
    return rho


def _r35_ao_dm(
    mol, ao: np.ndarray, dm: np.ndarray, mask, ao_loc=None
) -> np.ndarray:
    if ao_loc is None:
        ao_loc = mol.ao_loc_nr()
    return numint._dot_ao_dm(
        mol,
        ao,
        dm,
        mask,
        (0, mol.nbas),
        ao_loc,
    )


def _eta35_from_ao_dm(
    stretched_ao: np.ndarray, ao_dm: np.ndarray
) -> np.ndarray:
    return np.asarray(numint._contract_rho(stretched_ao, ao_dm), dtype=np.float64)


def _eta35(
    mol, ao: np.ndarray, stretched_ao: np.ndarray, dm: np.ndarray, mask
) -> np.ndarray:
    """Reference-compatible eta helper retained for focused tests."""
    return _eta35_from_ao_dm(
        stretched_ao, _r35_ao_dm(mol, ao[0], dm, mask)
    )


def _collect_total_gga_density(
    ni,
    mol,
    grids,
    dm_total,
    *,
    hermi,
    max_memory,
) -> np.ndarray:
    nao = mol.nao_nr()
    rho = np.empty((4, len(grids.weights)), dtype=np.float64)
    make_rho = ni._gen_rho_evaluator(
        mol, dm_total, hermi, False, grids
    )[0]
    offset = 0
    for ao, mask, weight, _ in ni.block_loop(
        mol, grids, nao, 1, max_memory=max_memory
    ):
        stop = offset + weight.size
        rho[:, offset:stop] = make_rho(0, ao, mask, "GGA")
        offset = stop
    return rho


def _contract_vv10_potential(
    ni,
    mol,
    grids,
    vv10_vxc,
    *,
    max_memory,
) -> np.ndarray:
    nao = mol.nao_nr()
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = numint.NBINS * 2 - int(
        numint.NBINS * np.log(cutoff) / np.log(grids.cutoff)
    )
    pair_mask = mol.get_overlap_cond() < -np.log(ni.cutoff)
    vmat = np.zeros((nao, nao), dtype=np.float64)
    aow = None
    offset = 0
    for ao, mask, weight, _ in ni.block_loop(
        mol, grids, nao, 1, max_memory=max_memory
    ):
        stop = offset + weight.size
        weighted = np.array(
            vv10_vxc[:, offset:stop] * weight, copy=True
        )
        weighted[0] *= 0.5
        aow = numint._scale_ao_sparse(
            ao[:4], weighted[:4], mask, ao_loc, out=aow
        )
        numint._dot_ao_ao_sparse(
            ao[0],
            aow,
            None,
            nbins,
            mask,
            pair_mask,
            ao_loc,
            hermi=0,
            out=vmat,
        )
        offset = stop
    return lib.hermi_sum(vmat)


def _vv10_energy(rho: np.ndarray, coords: np.ndarray, weights: np.ndarray) -> float:
    """Evaluate only the PySCF VV10 nonlocal energy on a total density."""
    energy = 0.0
    vv10_numint = numint.NumInt()
    for parameters, factor in vv10_numint.nlc_coeff("VV10"):
        exc, _ = numint._vv10nlc(
            rho,
            coords,
            rho,
            weights,
            coords,
            parameters,
        )
        energy += factor * np.dot(rho[0] * weights, exc)
    return float(energy)


def _vv10_energy_and_potential(
    rho: np.ndarray, coords: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    """Evaluate VV10 once and return its energy and transformed GGA potential."""
    energy = 0.0
    vxc = np.zeros((2, rho.shape[1]), dtype=np.float64)
    vv10_numint = numint.NumInt()
    for parameters, factor in vv10_numint.nlc_coeff("VV10"):
        exc, raw_vxc = numint._vv10nlc(
            rho,
            coords,
            rho,
            weights,
            coords,
            parameters,
        )
        energy += factor * np.dot(rho[0] * weights, exc)
        vxc += factor * raw_vxc
    return float(energy), np.asarray(
        xc_deriv.transform_vxc(rho, vxc, "GGA", spin=0),
        dtype=np.float64,
    )


def _hf_exchange_energy(mol, dm, *, spin: int, method=None) -> float:
    """Return the Hartree–Fock exchange energy used as ML26's final feature."""
    if method is None:
        method = scf.UHF(mol) if spin else scf.RHF(mol)

    if spin == 0:
        dm = np.asarray(dm, dtype=np.float64)
        vk = np.asarray(method.get_k(mol, dm, hermi=1), dtype=np.float64)
        return float(-0.25 * np.einsum("ij,ji->", dm, vk, optimize=True))

    dma, dmb = numint._format_uks_dm(dm)
    spin_dm = np.asarray((dma, dmb), dtype=np.float64)
    vk = np.asarray(method.get_k(mol, spin_dm, hermi=1), dtype=np.float64)
    return float(
        -0.5
        * (
            np.einsum("ij,ji->", spin_dm[0], vk[0], optimize=True)
            + np.einsum("ij,ji->", spin_dm[1], vk[1], optimize=True)
        )
    )


class _FixedGradientModel:
    """Private linear model used to contract one feature coefficient vector."""

    def __init__(self, coefficients):
        self.coefficients = np.asarray(
            coefficients, dtype=np.float64
        ).reshape(-1)

    @staticmethod
    def select_features(features):
        return np.asarray(features, dtype=np.float64)

    def evaluate(
        self, features, *, order, need_energy, context=None, out=None
    ):
        return MixerResult(
            energy=0.0 if need_energy else None,
            gradient=np.array(self.coefficients, copy=True),
        )


def _density_orbitals(dm):
    values, vectors = np.linalg.eigh(
        0.5 * (np.asarray(dm, dtype=np.float64) + np.asarray(dm).T)
    )
    values[np.abs(values) < 1e-13] = 0.0
    return vectors, values


def _vv10_response(
    mol,
    nlcgrids,
    dm0,
    dms,
    *,
    mo_coeff,
    mo_occ,
    max_memory,
):
    from pyscf.hessian.rks import get_vnlc_resp

    if mo_coeff is None or mo_occ is None:
        dm0_arr = np.asarray(dm0)
        if dm0_arr.ndim == 2:
            mo_coeff, mo_occ = _density_orbitals(dm0_arr)
        else:
            pairs = tuple(_density_orbitals(dm0_arr[s]) for s in range(2))
            mo_coeff = tuple(pair[0] for pair in pairs)
            mo_occ = tuple(pair[1] for pair in pairs)
    proxy = SimpleNamespace(
        nlcgrids=nlcgrids,
        _numint=numint.NumInt(),
        xc="VV10",
        nlc="VV10",
    )
    return np.asarray(
        get_vnlc_resp(
            proxy,
            mol,
            mo_coeff,
            mo_occ,
            np.asarray(dms, dtype=np.float64),
            max_memory,
        ),
        dtype=np.float64,
    )


class ML26FeatureEvaluator:
    feature_set = "ml26"
    requires_nlc_grid = True

    def __init__(self):
        self._semilocal = IFXCEvaluator("ml26")
        self._stretched_cache_key = None
        self._stretched_molecules = None
        self._stretched_pair_masks = None
        self._grid_equivalence_key = None
        self._grid_equivalence_value = None
        self.last_timing = None
        semilocal_labels = self._semilocal.feature_labels()
        if len(semilocal_labels) != _ML25_NFEATURES + 3:
            raise RuntimeError(
                "libifxc ML26 semilocal backend must provide 69 features"
            )
        r35_labels = tuple(
            label
            for tag in _BETA_TAGS
            for label in (
                f"r35_2_beta{tag}_local_hybrid",
                f"r35_2_beta{tag}_b05",
            )
        )
        self._labels = (
            tuple(semilocal_labels[:_ML25_NFEATURES])
            + r35_labels
            + tuple(semilocal_labels[_ML25_NFEATURES:])
            + ("vv10", "e_hf_exchange")
        )
        self.info = FeatureSetInfo(
            key=self.feature_set,
            name="ML26 composite integral features",
            nfeatures=len(self._labels),
            max_deriv_order=0,
            xctype="MGGA",
            labels=self._labels,
        )

    @property
    def nfeatures(self) -> int:
        return self.info.nfeatures

    def feature_labels(self) -> tuple[str, ...]:
        return self._labels

    def new_evaluator(self):
        return type(self)()

    def close(self):
        self._semilocal.close()

    @staticmethod
    def _geometry_key(mol):
        return (
            id(mol),
            bool(mol.cart),
            int(mol.nbas),
            tuple(np.asarray(mol.atom_coords(), dtype=np.float64).reshape(-1)),
        )

    def _stretched_data(self, mol, cutoff):
        key = (self._geometry_key(mol), float(cutoff))
        if key != self._stretched_cache_key:
            stretched = tuple(
                _stretch_ao_molecule(mol, beta) for beta in _BETAS
            )
            threshold = -math.log(float(cutoff))
            original_cond = mol.get_overlap_cond()
            pair_masks = tuple(
                np.logical_or(
                    original_cond < threshold,
                    stretched_mol.get_overlap_cond() < threshold,
                )
                for stretched_mol in stretched
            )
            self._stretched_cache_key = key
            self._stretched_molecules = stretched
            self._stretched_pair_masks = pair_masks
        return self._stretched_molecules, self._stretched_pair_masks

    def _uses_fused_nlc_grid(self, grids, nlcgrids):
        key = (
            id(grids),
            id(getattr(grids, "coords", None)),
            id(getattr(grids, "weights", None)),
            id(nlcgrids),
            id(getattr(nlcgrids, "coords", None)),
            id(getattr(nlcgrids, "weights", None)),
        )
        if key != self._grid_equivalence_key:
            self._grid_equivalence_key = key
            self._grid_equivalence_value = grids_are_equivalent(
                grids, nlcgrids
            )
        return bool(self._grid_equivalence_value)

    @staticmethod
    def _semilocal_coefficients(coefficients):
        coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        if coefficients.size != 79:
            raise ValueError(
                f"ML26 composite potential expects 79 coefficients, got {coefficients.size}"
            )
        return np.concatenate((coefficients[:66], coefficients[74:77]))

    @staticmethod
    def _unsupported(*args, **kwargs):
        raise NotImplementedError(
            "ML26 public feature derivatives are unavailable; its analytic "
            "first derivatives are private to ground-state RIF/UIF SCF"
        )

    eval_integral_features = _unsupported
    eval_local_values = _unsupported
    eval_weighted_first_derivatives = _unsupported
    eval_weighted_first_derivatives_batch = _unsupported
    eval_feature_directional_derivative = _unsupported
    eval_weighted_second_derivative_response = _unsupported
    eval_response_batch = _unsupported

    def _primary_response(
        self,
        ni,
        mol,
        grids,
        dm0,
        dms,
        *,
        spin: int,
        coefficients,
        hermi,
        max_memory,
    ):
        """Contract semilocal and Rung3.5 response and feature directions."""
        coefficients = np.asarray(
            coefficients, dtype=np.float64
        ).reshape(-1)
        semilocal_coefficients = self._semilocal_coefficients(
            coefficients
        )
        nao = mol.nao_nr()
        ao_loc = mol.ao_loc_nr()
        cutoff = grids.cutoff * 1e2
        nbins = numint.NBINS * 2 - int(
            numint.NBINS * np.log(cutoff) / np.log(grids.cutoff)
        )
        pair_mask = mol.get_overlap_cond() < -np.log(ni.cutoff)
        stretched_molecules, stretched_pair_masks = self._stretched_data(
            mol, ni.cutoff
        )

        if spin == 0:
            dm0 = np.asarray(dm0, dtype=np.float64)
            trial_dms = np.asarray(dms, dtype=np.float64)
            if trial_dms.ndim == 2:
                trial_dms = trial_dms[None]
            ntrial = trial_dms.shape[0]
            make_rho0 = ni._gen_rho_evaluator(
                mol, dm0, 1, False, grids
            )[0]
            make_rho1 = ni._gen_rho_evaluator(
                mol, trial_dms, hermi, False, grids
            )[0]
            vmat = np.zeros((ntrial, nao, nao), dtype=np.float64)
            vtau = np.zeros_like(vmat)
        else:
            dm0 = np.asarray(dm0, dtype=np.float64)
            trial_dms = np.asarray(dms, dtype=np.float64)
            if trial_dms.ndim == 3:
                trial_dms = trial_dms[:, None]
            ntrial = trial_dms.shape[1]
            make_rhoa0 = ni._gen_rho_evaluator(
                mol, dm0[0], 1, False, grids
            )[0]
            make_rhob0 = ni._gen_rho_evaluator(
                mol, dm0[1], 1, False, grids
            )[0]
            make_rhoa1 = ni._gen_rho_evaluator(
                mol, trial_dms[0], hermi, False, grids
            )[0]
            make_rhob1 = ni._gen_rho_evaluator(
                mol, trial_dms[1], hermi, False, grids
            )[0]
            vmat = np.zeros((2, ntrial, nao, nao), dtype=np.float64)
            vtau = np.zeros_like(vmat)
        dfeatures = np.zeros((ntrial, 79), dtype=np.float64)
        aow = None

        for ao, mask, weight, coords in ni.block_loop(
            mol, grids, nao, 1, max_memory=max_memory
        ):
            ngrid = weight.size
            if spin == 0:
                rho0 = make_rho0(0, ao, mask, "MGGA")
                rho1 = np.asarray(
                    [
                        make_rho1(index, ao, mask, "MGGA")
                        for index in range(ntrial)
                    ]
                )
                wv_batch, semilocal_dfeatures = (
                    self._semilocal.eval_response_batch(
                        rho0,
                        rho1,
                        weight,
                        semilocal_coefficients,
                        spin=0,
                    )
                )
                spin_rho0 = np.asarray((0.5 * rho0, 0.5 * rho0))
                spin_rho1 = np.repeat(
                    (0.5 * rho1)[:, None], 2, axis=1
                )
                ao_dm0 = _r35_ao_dm(
                    mol, ao[0], dm0, mask, ao_loc
                )
                ao_dm1 = np.asarray(
                    [
                        _r35_ao_dm(
                            mol, ao[0], trial_dms[index], mask, ao_loc
                        )
                        for index in range(ntrial)
                    ]
                )
            else:
                rho_a0 = make_rhoa0(0, ao, mask, "MGGA")
                rho_b0 = make_rhob0(0, ao, mask, "MGGA")
                rho_a1 = np.asarray(
                    [
                        make_rhoa1(index, ao, mask, "MGGA")
                        for index in range(ntrial)
                    ]
                )
                rho_b1 = np.asarray(
                    [
                        make_rhob1(index, ao, mask, "MGGA")
                        for index in range(ntrial)
                    ]
                )
                wv_batch, semilocal_dfeatures = (
                    self._semilocal.eval_response_batch(
                        (rho_a0, rho_b0),
                        (rho_a1, rho_b1),
                        weight,
                        semilocal_coefficients,
                        spin=1,
                    )
                )
                wv_batch = np.moveaxis(wv_batch, 1, 0)
                spin_rho0 = np.asarray((rho_a0, rho_b0))
                spin_rho1 = np.stack((rho_a1, rho_b1), axis=1)
                ao_dm0 = np.asarray(
                    (
                        _r35_ao_dm(
                            mol, ao[0], dm0[0], mask, ao_loc
                        ),
                        _r35_ao_dm(
                            mol, ao[0], dm0[1], mask, ao_loc
                        ),
                    )
                )
                ao_dm1 = np.asarray(
                    [
                        [
                            _r35_ao_dm(
                                mol,
                                ao[0],
                                trial_dms[spin_index, index],
                                mask,
                                ao_loc,
                            )
                            for index in range(ntrial)
                        ]
                        for spin_index in range(2)
                    ]
                )
            dfeatures[:, :66] += semilocal_dfeatures[:, :66]
            dfeatures[:, 74:77] += semilocal_dfeatures[:, 66:]

            eta0 = np.zeros((2, ngrid), dtype=np.float64)
            r35_rho0 = _r35_rho(spin_rho0, eta0)
            for beta_index, (
                beta,
                stretched_mol,
                stretched_pair_mask,
            ) in enumerate(
                zip(_BETAS, stretched_molecules, stretched_pair_masks)
            ):
                stretched_ao = ni.eval_ao(
                    stretched_mol,
                    coords,
                    deriv=0,
                    non0tab=mask,
                    cutoff=grids.cutoff,
                )
                if spin == 0:
                    eta0_total = _eta35_from_ao_dm(
                        stretched_ao, ao_dm0
                    )
                    eta0[:] = 0.5 * eta0_total
                    eta1_total = np.asarray(
                        [
                            _eta35_from_ao_dm(
                                stretched_ao, ao_dm1[index]
                            )
                            for index in range(ntrial)
                        ]
                    )
                    eta1 = np.repeat(
                        (0.5 * eta1_total)[:, None], 2, axis=1
                    )
                else:
                    eta0[0] = _eta35_from_ao_dm(
                        stretched_ao, ao_dm0[0]
                    )
                    eta0[1] = _eta35_from_ao_dm(
                        stretched_ao, ao_dm0[1]
                    )
                    eta1 = np.empty(
                        (ntrial, 2, ngrid), dtype=np.float64
                    )
                    for spin_index in range(2):
                        for index in range(ntrial):
                            eta1[index, spin_index] = (
                                _eta35_from_ao_dm(
                                    stretched_ao,
                                    ao_dm1[spin_index, index],
                                )
                            )
                r35_rho0[:, 6] = eta0
                _, _, dlocal, db05 = _r35_values_and_derivatives(
                    r35_rho0, beta
                )
                r35_direction = np.zeros(
                    (ntrial, 2, 7, ngrid), dtype=np.float64
                )
                r35_direction[:, :, :4] = spin_rho1[:, :, :4]
                r35_direction[:, :, 5] = spin_rho1[:, :, 4]
                r35_direction[:, :, 6] = eta1
                compact_direction = r35_direction[
                    :, :, (0, 1, 2, 3, 5, 6)
                ]
                local_index = 66 + 2 * beta_index
                b05_index = local_index + 1
                dfeatures[:, local_index] += np.einsum(
                    "sxg,tsxg,g->t",
                    dlocal,
                    compact_direction,
                    weight,
                    optimize=True,
                )
                dfeatures[:, b05_index] += np.einsum(
                    "sxg,tsxg,g->t",
                    db05,
                    compact_direction,
                    weight,
                    optimize=True,
                )

                for index in range(ntrial):
                    direction = r35_direction[index]
                    amplitude = float(np.max(np.abs(direction)))
                    if amplitude == 0.0:
                        continue
                    step = 1e-20
                    _, _, dlocal_step, db05_step = (
                        _r35_values_and_derivatives(
                            r35_rho0 + 1j * step * direction,
                            beta,
                        )
                    )
                    response_derivative = (
                        coefficients[local_index]
                        * np.imag(dlocal_step)
                        + coefficients[b05_index]
                        * np.imag(db05_step)
                    ) / step
                    if spin == 0:
                        response_derivative = 0.5 * (
                            response_derivative[0]
                            + response_derivative[1]
                        )
                        wv_batch[index] += (
                            weight[None]
                            * response_derivative[:5]
                        )
                        eta_weight = (
                            0.5
                            * weight
                            * response_derivative[5]
                        )
                        aow = numint._scale_ao_sparse(
                            ao[0], eta_weight, mask, ao_loc, out=aow
                        )
                        numint._dot_ao_ao_sparse(
                            stretched_ao,
                            aow,
                            None,
                            nbins,
                            mask,
                            stretched_pair_mask,
                            ao_loc,
                            hermi=0,
                            out=vmat[index],
                        )
                    else:
                        wv_batch[:, index] += (
                            weight[None, None]
                            * response_derivative[:, :5]
                        )
                        for spin_index in range(2):
                            eta_weight = (
                                0.5
                                * weight
                                * response_derivative[spin_index, 5]
                            )
                            aow = numint._scale_ao_sparse(
                                ao[0],
                                eta_weight,
                                mask,
                                ao_loc,
                                out=aow,
                            )
                            numint._dot_ao_ao_sparse(
                                stretched_ao,
                                aow,
                                None,
                                nbins,
                                mask,
                                stretched_pair_mask,
                                ao_loc,
                                hermi=0,
                                out=vmat[spin_index, index],
                            )

            if spin == 0:
                for index in range(ntrial):
                    weighted = wv_batch[index]
                    weighted[0] *= 0.5
                    weighted[4] *= 0.5
                    aow = numint._scale_ao_sparse(
                        ao[:4], weighted[:4], mask, ao_loc, out=aow
                    )
                    numint._dot_ao_ao_sparse(
                        ao[0],
                        aow,
                        None,
                        nbins,
                        mask,
                        pair_mask,
                        ao_loc,
                        hermi=0,
                        out=vmat[index],
                    )
                    numint._tau_dot_sparse(
                        ao,
                        ao,
                        weighted[4],
                        nbins,
                        mask,
                        pair_mask,
                        ao_loc,
                        out=vtau[index],
                    )
            else:
                for spin_index in range(2):
                    for index in range(ntrial):
                        weighted = wv_batch[spin_index, index]
                        weighted[0] *= 0.5
                        weighted[4] *= 0.5
                        aow = numint._scale_ao_sparse(
                            ao[:4],
                            weighted[:4],
                            mask,
                            ao_loc,
                            out=aow,
                        )
                        numint._dot_ao_ao_sparse(
                            ao[0],
                            aow,
                            None,
                            nbins,
                            mask,
                            pair_mask,
                            ao_loc,
                            hermi=0,
                            out=vmat[spin_index, index],
                        )
                        numint._tau_dot_sparse(
                            ao,
                            ao,
                            weighted[4],
                            nbins,
                            mask,
                            pair_mask,
                            ao_loc,
                            out=vtau[spin_index, index],
                        )
        if spin == 0:
            vmat = lib.hermi_sum(vmat, axes=(0, 2, 1)) + vtau
        else:
            vmat = lib.hermi_sum(
                vmat.reshape(-1, nao, nao), axes=(0, 2, 1)
            ).reshape(2, ntrial, nao, nao)
            vmat += vtau
        return vmat, dfeatures

    def build_response_cache(
        self,
        method,
        dm0,
        *,
        grids,
        nlcgrids,
        spin: int,
        mo_coeff,
        mo_occ,
        max_memory,
    ):
        ni = method._numint
        mol = method.mol
        reference = ni._matching_reference_state(
            mol, grids, dm0, spin=spin
        )
        if reference is None:
            features, _ = ni.integrate_features(
                mol,
                grids,
                dm0,
                spin=spin,
                max_memory=max_memory,
                method=method,
                nlcgrids=nlcgrids,
            )
        else:
            features = reference.features
        result = method.if_model.evaluate(
            features,
            order=2,
            need_energy=False,
            context=build_context(
                mol, spin_polarized=bool(spin), grids=grids
            ),
        )
        raw_k = np.asarray(
            method.get_k(mol, dm0, hermi=1), dtype=np.float64
        )
        dm_total = (
            np.asarray(dm0)
            if spin == 0
            else np.asarray(dm0[0]) + np.asarray(dm0[1])
        )
        vv10_rho = _collect_total_gga_density(
            ni,
            mol,
            nlcgrids,
            dm_total,
            hermi=1,
            max_memory=max_memory,
        )
        _, vv10_vxc = _vv10_energy_and_potential(
            vv10_rho,
            np.asarray(nlcgrids.coords),
            np.asarray(nlcgrids.weights),
        )
        vv10_vmat = _contract_vv10_potential(
            ni,
            mol,
            nlcgrids,
            vv10_vxc,
            max_memory=max_memory,
        )
        ni.last_response_features = np.asarray(
            features, dtype=np.float64
        ).reshape(1, -1)
        ni.last_feature_derivative = np.asarray(
            result.gradient, dtype=np.float64
        ).reshape(1, -1)
        ni.last_feature_hessian = np.asarray(
            result.hessian, dtype=np.float64
        )
        ni.last_feature_labels = tuple(
            method.if_model.feature_labels
        )
        return {
            "dm0": np.array(dm0, dtype=np.float64, copy=True),
            "features": np.asarray(features, dtype=np.float64),
            "gradient": np.asarray(result.gradient, dtype=np.float64),
            "hessian": np.asarray(result.hessian, dtype=np.float64),
            "raw_k": raw_k,
            "vv10_vmat": vv10_vmat,
            "mo_coeff": mo_coeff,
            "mo_occ": mo_occ,
        }

    def response(
        self,
        method,
        cache,
        dms,
        *,
        grids,
        nlcgrids,
        spin: int,
        singlet,
        hermi,
        max_memory,
    ):
        ni = method._numint
        mol = method.mol
        gradient = cache["gradient"]
        dm0 = cache["dm0"]
        if spin == 0:
            trial_dms = np.asarray(dms, dtype=np.float64)
            single = trial_dms.ndim == 2
            if single:
                trial_dms = trial_dms[None]
            if singlet is False:
                response_dm0 = np.asarray((0.5 * dm0, 0.5 * dm0))
                response_dms = np.asarray(
                    (0.5 * trial_dms, -0.5 * trial_dms)
                )
                direct, dfeatures = self._primary_response(
                    ni,
                    mol,
                    grids,
                    response_dm0,
                    response_dms,
                    spin=1,
                    coefficients=gradient,
                    hermi=hermi,
                    max_memory=max_memory,
                )
                direct = direct[0]
                total_trials = np.zeros_like(trial_dms)
            else:
                response_dm0 = dm0
                response_dms = trial_dms
                direct, dfeatures = self._primary_response(
                    ni,
                    mol,
                    grids,
                    response_dm0,
                    response_dms,
                    spin=0,
                    coefficients=gradient,
                    hermi=hermi,
                    max_memory=max_memory,
                )
                total_trials = trial_dms
            if singlet is not False:
                vv10_direct = _vv10_response(
                    mol,
                    nlcgrids,
                    dm0,
                    total_trials,
                    mo_coeff=cache["mo_coeff"],
                    mo_occ=cache["mo_occ"],
                    max_memory=max_memory,
                )
                if vv10_direct.ndim == 2:
                    vv10_direct = vv10_direct[None]
                direct += gradient[77] * vv10_direct
                dfeatures[:, 77] = np.einsum(
                    "ij,tji->t",
                    cache["vv10_vmat"],
                    total_trials,
                    optimize=True,
                )
            for index, dm1 in enumerate(trial_dms):
                k1 = np.asarray(
                    method.get_k(mol, dm1, hermi=hermi),
                    dtype=np.float64,
                )
                direct[index] -= 0.5 * gradient[78] * k1
                if singlet is not False:
                    dfeatures[index, 78] = -0.5 * np.einsum(
                        "ij,ji->", cache["raw_k"], dm1, optimize=True
                    )
        else:
            trial_dms = np.asarray(dms, dtype=np.float64)
            single = trial_dms.ndim == 3
            if single:
                trial_dms = trial_dms[:, None]
            response_dm0 = dm0
            response_dms = trial_dms
            direct, dfeatures = self._primary_response(
                ni,
                mol,
                grids,
                response_dm0,
                response_dms,
                spin=1,
                coefficients=gradient,
                hermi=hermi,
                max_memory=max_memory,
            )
            total_trials = trial_dms[0] + trial_dms[1]
            vv10_direct = _vv10_response(
                mol,
                nlcgrids,
                dm0,
                total_trials,
                mo_coeff=cache["mo_coeff"],
                mo_occ=cache["mo_occ"],
                max_memory=max_memory,
            )
            if vv10_direct.ndim == 2:
                vv10_direct = vv10_direct[None]
            direct += gradient[77] * vv10_direct[None]
            dfeatures[:, 77] = np.einsum(
                "ij,tji->t",
                cache["vv10_vmat"],
                total_trials,
                optimize=True,
            )
            for index in range(trial_dms.shape[1]):
                k1 = np.asarray(
                    method.get_k(
                        mol, trial_dms[:, index], hermi=hermi
                    ),
                    dtype=np.float64,
                )
                direct[:, index] -= gradient[78] * k1
                dfeatures[index, 78] = -sum(
                    np.einsum(
                        "ij,ji->",
                        cache["raw_k"][spin_index],
                        trial_dms[spin_index, index],
                        optimize=True,
                    )
                    for spin_index in range(2)
                )

        outer_coefficients = dfeatures @ cache["hessian"].T
        ntrial = dfeatures.shape[0]
        for index in range(ntrial):
            fixed_model = _FixedGradientModel(
                outer_coefficients[index]
            )
            if spin == 0 and singlet is False:
                raw_k = np.asarray(
                    (0.5 * cache["raw_k"], 0.5 * cache["raw_k"])
                )
                _, _, _, _, outer = self.evaluate_scf(
                    ni,
                    mol,
                    grids,
                    response_dm0,
                    spin=1,
                    raw_k=raw_k,
                    model=fixed_model,
                    context=build_context(
                        mol, spin_polarized=True, grids=grids
                    ),
                    nlcgrids=nlcgrids,
                    max_memory=max_memory,
                    hermi=1,
                )
                direct[index] += outer[0]
            else:
                _, _, _, _, outer = self.evaluate_scf(
                    ni,
                    mol,
                    grids,
                    response_dm0,
                    spin=spin,
                    raw_k=cache["raw_k"],
                    model=fixed_model,
                    context=build_context(
                        mol,
                        spin_polarized=bool(spin),
                        grids=grids,
                    ),
                    nlcgrids=nlcgrids,
                    max_memory=max_memory,
                    hermi=1,
                )
                if spin == 0:
                    direct[index] += outer
                else:
                    direct[:, index] += outer
        ni.last_response_dfeatures = np.asarray(
            dfeatures, dtype=np.float64
        )
        if spin == 0:
            return direct[0] if single else direct
        return direct[:, 0] if single else direct

    def gradient_veff(
        self,
        ks_grad,
        mol,
        dm,
        *,
        grids,
        nlcgrids,
        spin: int,
        max_memory,
    ):
        """Return the private ML26 static-grid gradient potential."""
        from pyscf.grad import rks as pyscf_grad_rks

        mf = ks_grad.base
        ni = mf._numint
        reference = None
        if (
            grids is mf.grids
            and nlcgrids is getattr(mf, "nlcgrids", grids)
        ):
            reference = ni._matching_reference_state(
                mol, grids, dm, spin=spin
            )
        if reference is None:
            features, _ = ni.integrate_features(
                mol,
                grids,
                dm,
                spin=spin,
                max_memory=max_memory,
                method=mf,
                nlcgrids=nlcgrids,
            )
            context = build_context(
                mol, spin_polarized=bool(spin), grids=grids
            )
            coefficients = np.asarray(
                mf.if_model.evaluate(
                    features,
                    order=1,
                    need_energy=False,
                    context=context,
                ).gradient,
                dtype=np.float64,
            ).reshape(-1)
        else:
            coefficients = np.asarray(
                reference.mixer_gradient, dtype=np.float64
            ).reshape(-1)
        semilocal_coefficients = self._semilocal_coefficients(
            coefficients
        )

        if nlcgrids is None:
            nlcgrids = grids
        fused_nlc = self._uses_fused_nlc_grid(grids, nlcgrids)
        if spin == 0:
            dm_total = np.asarray(dm, dtype=np.float64)
        else:
            dma, dmb = numint._format_uks_dm(dm)
            spin_dms = np.asarray((dma, dmb), dtype=np.float64)
            dm_total = spin_dms[0] + spin_dms[1]
        vv10_rho = _collect_total_gga_density(
            ni,
            mol,
            nlcgrids,
            dm_total,
            hermi=1,
            max_memory=max_memory,
        )
        _, vv10_vxc = _vv10_energy_and_potential(
            vv10_rho,
            np.asarray(nlcgrids.coords),
            np.asarray(nlcgrids.weights),
        )

        stretched_molecules = self._stretched_data(mol, ni.cutoff)[0]
        nao = mol.nao_nr()
        ao_loc = mol.ao_loc_nr()
        if spin == 0:
            vxc = np.zeros((3, nao, nao), dtype=np.float64)
            make_rho = ni._gen_rho_evaluator(
                mol, dm_total, 1, False, grids
            )[0]
        else:
            vxc = np.zeros((2, 3, nao, nao), dtype=np.float64)
            make_rhoa = ni._gen_rho_evaluator(
                mol, spin_dms[0], 1, False, grids
            )[0]
            make_rhob = ni._gen_rho_evaluator(
                mol, spin_dms[1], 1, False, grids
            )[0]

        spin_rho = None
        eta = None
        r35_rho = None
        semilocal_wv = None
        weighted_vvars = None
        grid_offset = 0
        for ao, mask, weight, coords in ni.block_loop(
            mol, grids, nao, 2, max_memory=max_memory
        ):
            ngrid = weight.size
            stop = grid_offset + ngrid
            if spin_rho is None or spin_rho.shape != (2, 5, ngrid):
                spin_rho = np.empty((2, 5, ngrid), dtype=np.float64)
                eta = np.empty((2, ngrid), dtype=np.float64)
                r35_rho = None
            if spin == 0:
                rho_total = make_rho(0, ao[:10], mask, "MGGA")
                semilocal_wv = (
                    self._semilocal.eval_weighted_first_derivatives(
                        rho_total,
                        weight,
                        semilocal_coefficients,
                        spin=0,
                        out=(
                            semilocal_wv
                            if semilocal_wv is not None
                            and semilocal_wv.shape == (5, ngrid)
                            else None
                        ),
                    )
                )
                weighted_vvars = np.array(
                    semilocal_wv, dtype=np.float64, copy=True
                )
                np.multiply(rho_total, 0.5, out=spin_rho[0])
                spin_rho[1] = spin_rho[0]
                ao_dm_total = _r35_ao_dm(
                    mol, ao[0], dm_total, mask, ao_loc
                )
            else:
                rho_a = make_rhoa(0, ao[:10], mask, "MGGA")
                rho_b = make_rhob(0, ao[:10], mask, "MGGA")
                semilocal_wv = (
                    self._semilocal.eval_weighted_first_derivatives(
                        (rho_a, rho_b),
                        weight,
                        semilocal_coefficients,
                        spin=1,
                        out=(
                            semilocal_wv
                            if semilocal_wv is not None
                            and semilocal_wv.shape == (2, 5, ngrid)
                            else None
                        ),
                    )
                )
                weighted_vvars = np.array(
                    semilocal_wv, dtype=np.float64, copy=True
                )
                spin_rho[0] = rho_a
                spin_rho[1] = rho_b
                ao_dm_a = _r35_ao_dm(
                    mol, ao[0], spin_dms[0], mask, ao_loc
                )
                ao_dm_b = _r35_ao_dm(
                    mol, ao[0], spin_dms[1], mask, ao_loc
                )

            if fused_nlc:
                vv_weighted = (
                    coefficients[77]
                    * vv10_vxc[:, grid_offset:stop]
                    * weight
                )
                if spin == 0:
                    weighted_vvars[:4] += vv_weighted
                else:
                    weighted_vvars[:, :4] += vv_weighted[None]

            eta.fill(0.0)
            r35_rho = _r35_rho(spin_rho, eta, out=r35_rho)
            for beta_index, (beta, stretched_mol) in enumerate(
                zip(_BETAS, stretched_molecules)
            ):
                stretched_ao = ni.eval_ao(
                    stretched_mol,
                    coords,
                    deriv=1,
                    non0tab=mask,
                    cutoff=grids.cutoff,
                )
                if spin == 0:
                    eta_total = _eta35_from_ao_dm(
                        stretched_ao[0], ao_dm_total
                    )
                    np.multiply(eta_total, 0.5, out=eta[0])
                    eta[1] = eta[0]
                else:
                    eta[0] = _eta35_from_ao_dm(
                        stretched_ao[0], ao_dm_a
                    )
                    eta[1] = _eta35_from_ao_dm(
                        stretched_ao[0], ao_dm_b
                    )
                r35_rho[:, 6] = eta
                _, _, dlocal, db05 = _r35_values_and_derivatives(
                    r35_rho, beta
                )
                derivative = (
                    coefficients[66 + 2 * beta_index] * dlocal
                    + coefficients[67 + 2 * beta_index] * db05
                )
                if spin == 0:
                    derivative = 0.5 * (
                        derivative[0] + derivative[1]
                    )
                    weighted_vvars += weight[None] * derivative[:5]
                    eta_weight = 0.5 * weight * derivative[5]
                    original_weighted = numint._scale_ao(
                        ao[0], eta_weight
                    )
                    pyscf_grad_rks._d1_dot_(
                        vxc,
                        stretched_mol,
                        stretched_ao[1:4],
                        original_weighted,
                        mask,
                        ao_loc,
                        True,
                    )
                    stretched_weighted = numint._scale_ao(
                        stretched_ao[0], eta_weight
                    )
                    pyscf_grad_rks._d1_dot_(
                        vxc,
                        mol,
                        ao[1:4],
                        stretched_weighted,
                        mask,
                        ao_loc,
                        True,
                    )
                else:
                    weighted_vvars += (
                        weight[None, None] * derivative[:, :5]
                    )
                    for spin_index in range(2):
                        eta_weight = (
                            0.5
                            * weight
                            * derivative[spin_index, 5]
                        )
                        original_weighted = numint._scale_ao(
                            ao[0], eta_weight
                        )
                        pyscf_grad_rks._d1_dot_(
                            vxc[spin_index],
                            stretched_mol,
                            stretched_ao[1:4],
                            original_weighted,
                            mask,
                            ao_loc,
                            True,
                        )
                        stretched_weighted = numint._scale_ao(
                            stretched_ao[0], eta_weight
                        )
                        pyscf_grad_rks._d1_dot_(
                            vxc[spin_index],
                            mol,
                            ao[1:4],
                            stretched_weighted,
                            mask,
                            ao_loc,
                            True,
                        )

            if spin == 0:
                weighted_vvars[0] *= 0.5
                weighted_vvars[4] *= 0.5
                pyscf_grad_rks._gga_grad_sum_(
                    vxc, mol, ao, weighted_vvars, mask, ao_loc
                )
                pyscf_grad_rks._tau_grad_dot_(
                    vxc,
                    mol,
                    ao,
                    weighted_vvars[4],
                    mask,
                    ao_loc,
                    True,
                )
            else:
                weighted_vvars[:, 0] *= 0.5
                weighted_vvars[:, 4] *= 0.5
                for spin_index in range(2):
                    weighted = weighted_vvars[spin_index]
                    pyscf_grad_rks._gga_grad_sum_(
                        vxc[spin_index],
                        mol,
                        ao,
                        weighted,
                        mask,
                        ao_loc,
                    )
                    pyscf_grad_rks._tau_grad_dot_(
                        vxc[spin_index],
                        mol,
                        ao,
                        weighted[4],
                        mask,
                        ao_loc,
                        True,
                    )
            grid_offset = stop
        vxc *= -1.0

        if not fused_nlc:
            vnlc = np.zeros((3, nao, nao), dtype=np.float64)
            offset = 0
            for ao, mask, weight, _ in ni.block_loop(
                mol, nlcgrids, nao, 2, max_memory=max_memory
            ):
                stop = offset + weight.size
                weighted = np.zeros(
                    (5, weight.size), dtype=np.float64
                )
                weighted[:4] = (
                    coefficients[77]
                    * vv10_vxc[:, offset:stop]
                    * weight
                )
                weighted[0] *= 0.5
                pyscf_grad_rks._gga_grad_sum_(
                    vnlc, mol, ao, weighted, mask, ao_loc
                )
                offset = stop
            vnlc *= -1.0
            if spin == 0:
                vxc += vnlc
            else:
                vxc += vnlc[None]

        if spin == 0:
            vj, vk = ks_grad.get_jk(mol, dm_total)
            vxc += vj - 0.5 * coefficients[78] * vk
        else:
            vj, vk = ks_grad.get_jk(mol, spin_dms)
            vxc += vj[0] + vj[1] - coefficients[78] * vk
        return lib.tag_array(vxc, exc1_grid=None)

    def evaluate_scf(
        self,
        ni,
        mol,
        grids,
        dm,
        *,
        spin: int,
        raw_k,
        model,
        context,
        nlcgrids=None,
        max_memory=2000,
        xctype="MGGA",
        hermi=1,
        jk_time=0.0,
    ):
        """Evaluate the private ML26 SCF feature-chain potential."""
        if spin not in (0, 1):
            raise ValueError(f"Unsupported spin {spin}")
        if nlcgrids is None:
            nlcgrids = grids
        fused_nlc = self._uses_fused_nlc_grid(grids, nlcgrids)
        stretched_molecules, stretched_pair_masks = self._stretched_data(
            mol, ni.cutoff
        )
        nao = mol.nao_nr()
        ao_loc = mol.ao_loc_nr()
        semilocal_features = np.zeros(
            self._semilocal.nfeatures, dtype=np.float64
        )
        semilocal_block = np.empty_like(semilocal_features)
        r35_features = np.zeros(2 * len(_BETAS), dtype=np.float64)
        vv10_rho = (
            np.empty((4, len(grids.weights)), dtype=np.float64)
            if fused_nlc
            else None
        )

        if spin == 0:
            dm_total = np.asarray(dm, dtype=np.float64)
            make_rho = ni._gen_rho_evaluator(
                mol, dm_total, hermi, False, grids
            )[0]
            nelec = 0.0
        else:
            dma, dmb = numint._format_uks_dm(dm)
            spin_dms = (
                np.asarray(dma, dtype=np.float64),
                np.asarray(dmb, dtype=np.float64),
            )
            dm_total = spin_dms[0] + spin_dms[1]
            make_rhoa = ni._gen_rho_evaluator(
                mol, spin_dms[0], hermi, False, grids
            )[0]
            make_rhob = ni._gen_rho_evaluator(
                mol, spin_dms[1], hermi, False, grids
            )[0]
            nelec = np.zeros(2, dtype=np.float64)

        primary_feature_start = time.perf_counter()
        spin_rho = None
        eta = None
        r35_rho = None
        grid_offset = 0
        for ao, mask, weight, coords in ni.block_loop(
            mol, grids, nao, 1, max_memory=max_memory
        ):
            ngrid = weight.size
            if spin_rho is None or spin_rho.shape != (2, 5, ngrid):
                spin_rho = np.empty((2, 5, ngrid), dtype=np.float64)
                eta = np.empty((2, ngrid), dtype=np.float64)
                r35_rho = None
            if spin == 0:
                rho_total = make_rho(0, ao, mask, xctype)
                nelec += np.dot(weight, rho_total[0])
                semilocal_features += self._semilocal.eval_integral_features(
                    rho_total, weight, spin=0, out=semilocal_block
                )
                np.multiply(rho_total, 0.5, out=spin_rho[0])
                spin_rho[1] = spin_rho[0]
                ao_dm_total = _r35_ao_dm(
                    mol, ao[0], dm_total, mask, ao_loc
                )
            else:
                rho_a = make_rhoa(0, ao, mask, xctype)
                rho_b = make_rhob(0, ao, mask, xctype)
                nelec[0] += np.dot(weight, rho_a[0])
                nelec[1] += np.dot(weight, rho_b[0])
                semilocal_features += self._semilocal.eval_integral_features(
                    (rho_a, rho_b), weight, spin=1, out=semilocal_block
                )
                spin_rho[0] = rho_a
                spin_rho[1] = rho_b
                ao_dm_a = _r35_ao_dm(
                    mol, ao[0], spin_dms[0], mask, ao_loc
                )
                ao_dm_b = _r35_ao_dm(
                    mol, ao[0], spin_dms[1], mask, ao_loc
                )
            stop = grid_offset + ngrid
            if fused_nlc:
                if spin == 0:
                    vv10_rho[:, grid_offset:stop] = rho_total[:4]
                else:
                    vv10_rho[:, grid_offset:stop] = (
                        rho_a[:4] + rho_b[:4]
                    )
            eta.fill(0.0)
            r35_rho = _r35_rho(spin_rho, eta, out=r35_rho)
            for beta_index, (beta, stretched_mol) in enumerate(
                zip(_BETAS, stretched_molecules)
            ):
                stretched_ao = ni.eval_ao(
                    stretched_mol,
                    coords,
                    deriv=0,
                    non0tab=mask,
                    cutoff=grids.cutoff,
                )
                if spin == 0:
                    eta_total = _eta35_from_ao_dm(
                        stretched_ao, ao_dm_total
                    )
                    np.multiply(eta_total, 0.5, out=eta[0])
                    eta[1] = eta[0]
                else:
                    eta[0] = _eta35_from_ao_dm(stretched_ao, ao_dm_a)
                    eta[1] = _eta35_from_ao_dm(stretched_ao, ao_dm_b)
                r35_rho[:, 6] = eta
                local_hybrid, b05 = _r35_values(
                    r35_rho, beta
                )
                r35_features[2 * beta_index] += np.dot(
                    weight, local_hybrid[0] + local_hybrid[1]
                )
                r35_features[2 * beta_index + 1] += np.dot(weight, b05)
            grid_offset = stop
        primary_feature_time = time.perf_counter() - primary_feature_start

        nlc_density_time = 0.0
        if not fused_nlc:
            nlc_density_start = time.perf_counter()
            vv10_rho = _collect_total_gga_density(
                ni,
                mol,
                nlcgrids,
                dm_total,
                hermi=hermi,
                max_memory=max_memory,
            )
            nlc_density_time = time.perf_counter() - nlc_density_start

        vv10_start = time.perf_counter()
        vv10_energy, vv10_vxc = _vv10_energy_and_potential(
            vv10_rho,
            np.asarray(nlcgrids.coords),
            np.asarray(nlcgrids.weights),
        )
        vv10_time = time.perf_counter() - vv10_start
        del vv10_rho

        raw_k = np.asarray(raw_k, dtype=np.float64)
        if spin == 0:
            hf_exchange_energy = -0.25 * np.einsum(
                "ij,ji->", dm_total, raw_k, optimize=True
            )
        else:
            hf_exchange_energy = -0.5 * sum(
                np.einsum(
                    "ij,ji->", spin_dms[s], raw_k[s], optimize=True
                )
                for s in range(2)
            )
        backend_features = np.concatenate(
            (
                semilocal_features[:_ML25_NFEATURES],
                r35_features,
                semilocal_features[_ML25_NFEATURES:],
                np.asarray((vv10_energy, hf_exchange_energy)),
            )
        )
        features = np.asarray(
            model.select_features(backend_features), dtype=np.float64
        )

        mixer_start = time.perf_counter()
        mixer_result = model.evaluate(
            features,
            order=1,
            need_energy=True,
            context=context,
        )
        mixer_time = time.perf_counter() - mixer_start
        coefficients = np.asarray(
            mixer_result.gradient, dtype=np.float64
        ).reshape(-1)
        semilocal_coefficients = self._semilocal_coefficients(coefficients)

        primary_potential_start = time.perf_counter()
        cutoff = grids.cutoff * 1e2
        nbins = numint.NBINS * 2 - int(
            numint.NBINS * np.log(cutoff) / np.log(grids.cutoff)
        )
        pair_mask = mol.get_overlap_cond() < -np.log(ni.cutoff)
        if spin == 0:
            vmat = np.zeros((nao, nao), dtype=np.float64)
            vtau = np.zeros_like(vmat)
            make_rho = ni._gen_rho_evaluator(
                mol, dm_total, hermi, False, grids
            )[0]
        else:
            vmat = np.zeros((2, nao, nao), dtype=np.float64)
            vtau = np.zeros_like(vmat)
            make_rhoa = ni._gen_rho_evaluator(
                mol, spin_dms[0], hermi, False, grids
            )[0]
            make_rhob = ni._gen_rho_evaluator(
                mol, spin_dms[1], hermi, False, grids
            )[0]

        aow = None
        semilocal_wv = None
        weighted_vvars = None
        grid_offset = 0
        for ao, mask, weight, coords in ni.block_loop(
            mol, grids, nao, 1, max_memory=max_memory
        ):
            stop = grid_offset + weight.size
            ngrid = weight.size
            if spin_rho is None or spin_rho.shape != (2, 5, ngrid):
                spin_rho = np.empty((2, 5, ngrid), dtype=np.float64)
                eta = np.empty((2, ngrid), dtype=np.float64)
                r35_rho = None
            if spin == 0:
                rho_total = make_rho(0, ao, mask, xctype)
                semilocal_wv = self._semilocal.eval_weighted_first_derivatives(
                    rho_total,
                    weight,
                    semilocal_coefficients,
                    spin=0,
                    out=(
                        semilocal_wv
                        if semilocal_wv is not None
                        and semilocal_wv.shape == (5, weight.size)
                        else None
                    ),
                )
                if (
                    weighted_vvars is None
                    or weighted_vvars.shape != semilocal_wv.shape
                ):
                    weighted_vvars = np.empty_like(semilocal_wv)
                np.copyto(weighted_vvars, semilocal_wv)
                if fused_nlc:
                    weighted_vvars[:4] += (
                        coefficients[77]
                        * vv10_vxc[:, grid_offset:stop]
                        * weight
                    )
                np.multiply(rho_total, 0.5, out=spin_rho[0])
                spin_rho[1] = spin_rho[0]
                ao_dm_total = _r35_ao_dm(
                    mol, ao[0], dm_total, mask, ao_loc
                )
            else:
                rho_a = make_rhoa(0, ao, mask, xctype)
                rho_b = make_rhob(0, ao, mask, xctype)
                semilocal_wv = self._semilocal.eval_weighted_first_derivatives(
                    (rho_a, rho_b),
                    weight,
                    semilocal_coefficients,
                    spin=1,
                    out=(
                        semilocal_wv
                        if semilocal_wv is not None
                        and semilocal_wv.shape == (2, 5, weight.size)
                        else None
                    ),
                )
                if (
                    weighted_vvars is None
                    or weighted_vvars.shape != semilocal_wv.shape
                ):
                    weighted_vvars = np.empty_like(semilocal_wv)
                np.copyto(weighted_vvars, semilocal_wv)
                if fused_nlc:
                    weighted_vvars[:, :4] += (
                        coefficients[77]
                        * vv10_vxc[:, grid_offset:stop]
                        * weight
                    )[None, :, :]
                spin_rho[0] = rho_a
                spin_rho[1] = rho_b
                ao_dm_a = _r35_ao_dm(
                    mol, ao[0], spin_dms[0], mask, ao_loc
                )
                ao_dm_b = _r35_ao_dm(
                    mol, ao[0], spin_dms[1], mask, ao_loc
                )

            eta.fill(0.0)
            r35_rho = _r35_rho(spin_rho, eta, out=r35_rho)
            for beta_index, (
                beta,
                stretched_mol,
                stretched_pair_mask,
            ) in enumerate(
                zip(_BETAS, stretched_molecules, stretched_pair_masks)
            ):
                stretched_ao = ni.eval_ao(
                    stretched_mol,
                    coords,
                    deriv=0,
                    non0tab=mask,
                    cutoff=grids.cutoff,
                )
                if spin == 0:
                    eta_total = _eta35_from_ao_dm(
                        stretched_ao, ao_dm_total
                    )
                    np.multiply(eta_total, 0.5, out=eta[0])
                    eta[1] = eta[0]
                else:
                    eta[0] = _eta35_from_ao_dm(stretched_ao, ao_dm_a)
                    eta[1] = _eta35_from_ao_dm(stretched_ao, ao_dm_b)
                r35_rho[:, 6] = eta
                _, _, dlocal, db05 = _r35_values_and_derivatives(
                    r35_rho, beta
                )
                r35_derivative = (
                    coefficients[66 + 2 * beta_index] * dlocal
                    + coefficients[67 + 2 * beta_index] * db05
                )
                if spin == 0:
                    r35_derivative = 0.5 * (
                        r35_derivative[0] + r35_derivative[1]
                    )
                    weighted_vvars += (
                        weight[None, :] * r35_derivative[:5]
                    )
                    eta_weight = 0.5 * weight * r35_derivative[5]
                    aow = numint._scale_ao_sparse(
                        ao[0], eta_weight, mask, ao_loc, out=aow
                    )
                    numint._dot_ao_ao_sparse(
                        stretched_ao,
                        aow,
                        None,
                        nbins,
                        mask,
                        stretched_pair_mask,
                        ao_loc,
                        hermi=0,
                        out=vmat,
                    )
                else:
                    weighted_vvars += (
                        weight[None, None, :] * r35_derivative[:, :5]
                    )
                    for spin_index in range(2):
                        eta_weight = (
                            0.5
                            * weight
                            * r35_derivative[spin_index, 5]
                        )
                        aow = numint._scale_ao_sparse(
                            ao[0], eta_weight, mask, ao_loc, out=aow
                        )
                        numint._dot_ao_ao_sparse(
                            stretched_ao,
                            aow,
                            None,
                            nbins,
                            mask,
                            stretched_pair_mask,
                            ao_loc,
                            hermi=0,
                            out=vmat[spin_index],
                        )

            if spin == 0:
                weighted_vvars[0] *= 0.5
                weighted_vvars[4] *= 0.5
                aow = numint._scale_ao_sparse(
                    ao[:4], weighted_vvars[:4], mask, ao_loc, out=aow
                )
                numint._dot_ao_ao_sparse(
                    ao[0],
                    aow,
                    None,
                    nbins,
                    mask,
                    pair_mask,
                    ao_loc,
                    hermi=0,
                    out=vmat,
                )
                numint._tau_dot_sparse(
                    ao,
                    ao,
                    weighted_vvars[4],
                    nbins,
                    mask,
                    pair_mask,
                    ao_loc,
                    out=vtau,
                )
            else:
                weighted_vvars[:, 0] *= 0.5
                weighted_vvars[:, 4] *= 0.5
                for spin_index in range(2):
                    weighted = weighted_vvars[spin_index]
                    aow = numint._scale_ao_sparse(
                        ao[:4], weighted[:4], mask, ao_loc, out=aow
                    )
                    numint._dot_ao_ao_sparse(
                        ao[0],
                        aow,
                        None,
                        nbins,
                        mask,
                        pair_mask,
                        ao_loc,
                        hermi=0,
                        out=vmat[spin_index],
                    )
                    numint._tau_dot_sparse(
                        ao,
                        ao,
                        weighted[4],
                        nbins,
                        mask,
                        pair_mask,
                        ao_loc,
                        out=vtau[spin_index],
                    )
            grid_offset = stop

        if spin == 0:
            vmat = lib.hermi_sum(vmat) + vtau
            vmat -= 0.5 * coefficients[78] * raw_k
        else:
            vmat = lib.hermi_sum(
                vmat, axes=(0, 2, 1)
            ) + vtau
            vmat -= coefficients[78] * raw_k
        primary_potential_time = (
            time.perf_counter() - primary_potential_start
        )

        nlc_potential_time = 0.0
        if not fused_nlc:
            nlc_potential_start = time.perf_counter()
            vv10_vmat = _contract_vv10_potential(
                ni,
                mol,
                nlcgrids,
                vv10_vxc,
                max_memory=max_memory,
            )
            if spin == 0:
                vmat += coefficients[77] * vv10_vmat
            else:
                vmat += coefficients[77] * vv10_vmat[None, :, :]
            nlc_potential_time = (
                time.perf_counter() - nlc_potential_start
            )

        self.last_timing = {
            "feature_integration": (
                primary_feature_time + nlc_density_time
            ),
            "primary_feature_integration": primary_feature_time,
            "nlc_density": nlc_density_time,
            "vv10": vv10_time,
            "mixer": mixer_time,
            "potential_contraction": (
                primary_potential_time + nlc_potential_time
            ),
            "primary_potential_contraction": primary_potential_time,
            "nlc_potential_contraction": nlc_potential_time,
            "jk": float(jk_time),
            "grid_passes": 2 if fused_nlc else 4,
            "primary_grid_passes": 2,
            "nlc_grid_passes": 0 if fused_nlc else 2,
            "vv10_evaluations": 1,
            "jk_builds": 1,
        }
        return (
            nelec,
            features,
            float(mixer_result.energy),
            coefficients,
            vmat,
        )

    def integrate_features(
        self,
        ni,
        mol,
        grids,
        dm,
        *,
        spin: int,
        max_memory=2000,
        xctype="MGGA",
        method=None,
        nlcgrids=None,
    ):
        if spin not in (0, 1):
            raise ValueError(f"Unsupported spin {spin}")
        if nlcgrids is None:
            nlcgrids = grids
        fused_nlc = self._uses_fused_nlc_grid(grids, nlcgrids)
        stretched_molecules = self._stretched_data(mol, ni.cutoff)[0]
        nao = mol.nao_nr()
        ao_loc = mol.ao_loc_nr()
        semilocal_features = np.zeros(self._semilocal.nfeatures, dtype=np.float64)
        semilocal_block = np.empty_like(semilocal_features)
        r35_features = np.zeros(2 * len(_BETAS), dtype=np.float64)
        vv10_rho = (
            np.empty((4, len(grids.weights)), dtype=np.float64)
            if fused_nlc
            else None
        )

        if spin == 0:
            dm_total = np.asarray(dm, dtype=np.float64)
            make_rho = ni._gen_rho_evaluator(mol, dm_total, 1, False, grids)[0]
            nelec = 0.0
        else:
            dma, dmb = numint._format_uks_dm(dm)
            spin_dms = (
                np.asarray(dma, dtype=np.float64),
                np.asarray(dmb, dtype=np.float64),
            )
            dm_total = spin_dms[0] + spin_dms[1]
            make_rhoa = ni._gen_rho_evaluator(mol, spin_dms[0], 1, False, grids)[0]
            make_rhob = ni._gen_rho_evaluator(mol, spin_dms[1], 1, False, grids)[0]
            nelec = np.zeros(2, dtype=np.float64)

        spin_rho = None
        eta = None
        r35_rho = None
        grid_offset = 0
        for ao, mask, weight, coords in ni.block_loop(
            mol, grids, nao, 1, max_memory=max_memory
        ):
            ngrid = weight.size
            if spin_rho is None or spin_rho.shape != (2, 5, ngrid):
                spin_rho = np.empty((2, 5, ngrid), dtype=np.float64)
                eta = np.empty((2, ngrid), dtype=np.float64)
                r35_rho = None
            if spin == 0:
                rho_total = make_rho(0, ao, mask, xctype)
                nelec += np.dot(weight, rho_total[0])
                semilocal_features += self._semilocal.eval_integral_features(
                    rho_total, weight, spin=0, out=semilocal_block
                )
                np.multiply(rho_total, 0.5, out=spin_rho[0])
                spin_rho[1] = spin_rho[0]
                ao_dm_total = _r35_ao_dm(
                    mol, ao[0], dm_total, mask, ao_loc
                )
            else:
                rho_a = make_rhoa(0, ao, mask, xctype)
                rho_b = make_rhob(0, ao, mask, xctype)
                nelec[0] += np.dot(weight, rho_a[0])
                nelec[1] += np.dot(weight, rho_b[0])
                semilocal_features += self._semilocal.eval_integral_features(
                    (rho_a, rho_b), weight, spin=1, out=semilocal_block
                )
                spin_rho[0] = rho_a
                spin_rho[1] = rho_b
                ao_dm_a = _r35_ao_dm(
                    mol, ao[0], spin_dms[0], mask, ao_loc
                )
                ao_dm_b = _r35_ao_dm(
                    mol, ao[0], spin_dms[1], mask, ao_loc
                )
            stop = grid_offset + ngrid
            if fused_nlc:
                if spin == 0:
                    vv10_rho[:, grid_offset:stop] = rho_total[:4]
                else:
                    vv10_rho[:, grid_offset:stop] = (
                        rho_a[:4] + rho_b[:4]
                    )
            eta.fill(0.0)
            r35_rho = _r35_rho(spin_rho, eta, out=r35_rho)
            for beta_index, (beta, stretched_mol) in enumerate(
                zip(_BETAS, stretched_molecules)
            ):
                stretched_ao = ni.eval_ao(
                    stretched_mol,
                    coords,
                    deriv=0,
                    non0tab=mask,
                    cutoff=grids.cutoff,
                )
                if spin == 0:
                    eta_total = _eta35_from_ao_dm(
                        stretched_ao, ao_dm_total
                    )
                    np.multiply(eta_total, 0.5, out=eta[0])
                    eta[1] = eta[0]
                else:
                    eta[0] = _eta35_from_ao_dm(stretched_ao, ao_dm_a)
                    eta[1] = _eta35_from_ao_dm(stretched_ao, ao_dm_b)
                r35_rho[:, 6] = eta
                local_hybrid, b05 = _r35_values(
                    r35_rho, beta
                )
                r35_features[2 * beta_index] += np.dot(
                    weight, local_hybrid[0] + local_hybrid[1]
                )
                r35_features[2 * beta_index + 1] += np.dot(weight, b05)
            grid_offset = stop

        if not fused_nlc:
            vv10_rho = _collect_total_gga_density(
                ni,
                mol,
                nlcgrids,
                dm_total,
                hermi=1,
                max_memory=max_memory,
            )
        vv10_energy = _vv10_energy(
            vv10_rho,
            np.asarray(nlcgrids.coords),
            np.asarray(nlcgrids.weights),
        )
        features = np.concatenate(
            (
                semilocal_features[:_ML25_NFEATURES],
                r35_features,
                semilocal_features[_ML25_NFEATURES:],
                np.asarray((vv10_energy,)),
                np.asarray(
                    (
                        _hf_exchange_energy(
                            mol,
                            dm,
                            spin=spin,
                            method=method,
                        ),
                    )
                ),
            )
        )
        return nelec, features


__all__ = ["ML26FeatureEvaluator"]
