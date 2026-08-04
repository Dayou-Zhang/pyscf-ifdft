import copy
import os

import h5py
import numpy as np
from pyscf import dft, lib, scf

from .features import build_context
from .grids import DEFAULT_ATOM_GRID, IFGridManager, copy_grids
from .numint import IntegralFeatureNumInt
from .registry import get_functional
from .results import IFDFTEnergyResult


def _density_mf_class(spin_polarized: bool):
    return dft.UKS if spin_polarized else dft.RKS


def _clone_density_method(
    source, *, mol, xc, spin_polarized: bool, grids=None, x2c=False
):
    cls = _density_mf_class(spin_polarized)
    density_mf = cls(mol, xc=xc)
    source_x2c = source is not None and getattr(source, "with_x2c", None) is not None
    if x2c or source_x2c:
        density_mf = density_mf.x2c()
    if source is not None:
        for attr in ("conv_tol", "conv_tol_grad", "max_cycle", "max_memory", "verbose", "chkfile", "direct_scf", "diis_space", "level_shift"):
            if hasattr(source, attr):
                setattr(density_mf, attr, getattr(source, attr))
    if grids is not None:
        density_mf.grids = grids
    elif getattr(source, "grids", None) is not None:
        density_mf.grids = copy_grids(source.grids)
    return density_mf


def _density_method_for_mol(mol, spin_polarized: bool):
    cls = _density_mf_class(spin_polarized)
    return cls(mol)


def _default_nlc_grids(mol):
    return dft.RKS(mol).nlcgrids


def _x2c_method(source):
    """Return an X2C wrapper without changing the supplied source method."""
    if hasattr(source, "x2c"):
        return source.x2c()
    return source


def _density_matrix_from_mf(mf):
    return mf.make_rdm1()


def _mo_is_unrestricted(mo_coeff, mo_occ):
    coeff = np.asarray(mo_coeff)
    occ = np.asarray(mo_occ)
    return (coeff.ndim == 3 and coeff.shape[0] == 2) or (occ.ndim == 2 and occ.shape[0] == 2)


def _format_dm_for_spin(dm, spin_polarized: bool):
    if spin_polarized:
        if isinstance(dm, tuple) and len(dm) == 2:
            return dm
        arr = np.asarray(dm)
        if arr.ndim == 3 and arr.shape[0] == 2:
            return arr
        if arr.ndim == 2:
            return np.array((0.5 * arr, 0.5 * arr))
        raise ValueError(f"Expected UKS density matrix, got {arr.shape}")
    arr = np.asarray(dm)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 2:
        return arr[0] + arr[1]
    if isinstance(dm, tuple) and len(dm) == 2:
        return dm[0] + dm[1]
    raise ValueError(f"Expected RKS density matrix, got {arr.shape}")


def _template_or_hf_method(mol, spin_polarized: bool, template=None):
    if template is not None:
        return template
    return scf.UHF(mol) if spin_polarized else scf.RHF(mol)


def _trace_dm_matrix(dm, matrix):
    return float(np.einsum("ij,ji->", np.asarray(dm), np.asarray(matrix), optimize=True))


def _classical_energy_and_hf_exchange(
    mol, dm, spin_polarized: bool, template=None, hf_coeff: float = 1.0
):
    mf = _template_or_hf_method(mol, spin_polarized, template=template)
    hcore = mf.get_hcore() if hasattr(mf, "get_hcore") else scf.hf.get_hcore(mol)
    e_nuc = float(mol.energy_nuc())
    want_hf_exchange = abs(float(hf_coeff)) > 0.0

    if spin_polarized:
        dma, dmb = np.asarray(dm, dtype=np.float64)
        dm_total = dma + dmb
        e1 = _trace_dm_matrix(dm_total, hcore)
        if want_hf_exchange:
            vj, vk = mf.get_jk(mol, np.asarray((dma, dmb)), hermi=1, with_j=True, with_k=True)
        else:
            vj = mf.get_j(mol, np.asarray((dma, dmb)), hermi=1)
            vk = None
        vj = np.asarray(vj, dtype=np.float64)
        vj_total = vj[0] + vj[1] if vj.ndim == 3 else vj
        e_coul = 0.5 * _trace_dm_matrix(dm_total, vj_total)
        if want_hf_exchange:
            vk = np.asarray(vk, dtype=np.float64)
            e_hf_exchange = -0.5 * (
                _trace_dm_matrix(dma, vk[0])
                + _trace_dm_matrix(dmb, vk[1])
            )
        else:
            e_hf_exchange = 0.0
        return e_nuc + e1 + e_coul, e_hf_exchange

    dm = np.asarray(dm, dtype=np.float64)
    e1 = _trace_dm_matrix(dm, hcore)
    if want_hf_exchange:
        vj, vk = mf.get_jk(mol, dm, hermi=1, with_j=True, with_k=True)
    else:
        vj = mf.get_j(mol, dm, hermi=1)
        vk = None
    e_coul = 0.5 * _trace_dm_matrix(dm, vj)
    e_hf_exchange = (
        -0.25 * _trace_dm_matrix(dm, vk) if want_hf_exchange else 0.0
    )
    return e_nuc + e1 + e_coul, e_hf_exchange


def _cached_classical_energy_and_hf_exchange(template, dm, hf_coeff: float):
    cache = getattr(template, "_ifdft_fixed_density_cache", None)
    if not isinstance(cache, dict):
        return None
    cached_dm = np.asarray(cache.get("dm"))
    current_dm = np.asarray(dm)
    if cached_dm.shape != current_dm.shape or not np.array_equal(cached_dm, current_dm):
        return None
    e_hf_exchange = cache.get("e_hf_exchange", cache.get("e_x"))
    if e_hf_exchange is None and abs(float(hf_coeff)) > 0.0:
        return None
    return (
        float(cache["e_classical"]),
        0.0 if e_hf_exchange is None else float(e_hf_exchange),
    )


def _write_feature_file(feature_file, result, *, spin_polarized: bool):
    feature_file = os.fspath(feature_file)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    feature_key = f"{result.functional.lower()}_features"
    with lib.H5FileWrap(feature_file, "a") as fh5:
        # Remove the legacy grouped schema when updating an existing file.
        if "ifdft" in fh5:
            del fh5["ifdft"]
        for key in (feature_key, "e_classical", "nelec"):
            if key in fh5:
                del fh5[key]
        features = fh5.create_dataset(
            feature_key, data=np.asarray(result.features, dtype=np.float64)
        )
        features.attrs.create(
            "feature_labels",
            np.asarray(result.feature_labels, dtype=string_dtype),
            dtype=string_dtype,
        )
        features.attrs["spin_polarized"] = bool(spin_polarized)
        if result.density_xc is not None:
            features.attrs["density_xc"] = result.density_xc
        fh5.create_dataset("e_classical", data=np.float64(result.e_classical))
        fh5.create_dataset(
            "nelec", data=np.asarray(np.rint(result.nelec), dtype=np.int64)
        )


class _IFPostBase(lib.StreamObject):
    def __init__(
        self,
        mf_or_mol,
        xc="ML25",
        *,
        density_xc=None,
        dm=None,
        grids=None,
        nlcgrids=None,
        feature_file=None,
        **options,
    ):
        mol = mf_or_mol.mol if hasattr(mf_or_mol, "mol") else mf_or_mol
        self.results = None
        self.e_tot = None
        self.e_ifxc = None
        self.e_hf_exchange = None
        self.features = None
        self.density_mf = None
        self.if_model = get_functional(xc)
        self.if_xc = self.if_model.name
        self._numint = IntegralFeatureNumInt(self.if_model)
        self.density_xc = density_xc
        self.dm = dm
        self.feature_file = feature_file
        self._mf_or_mol = mf_or_mol
        self._options = options
        if grids is not None:
            self.grids = grids
        else:
            self.grids = dft.Grids(mol)
            self.grids.atom_grid = DEFAULT_ATOM_GRID
        if nlcgrids is not None:
            self.nlcgrids = nlcgrids
        elif getattr(mf_or_mol, "nlcgrids", None) is not None:
            self.nlcgrids = copy_grids(mf_or_mol.nlcgrids)
        elif bool(
            getattr(self._numint.ifxc, "requires_nlc_grid", False)
        ):
            self.nlcgrids = _default_nlc_grids(mol)
        else:
            self.nlcgrids = None
        for key, value in options.items():
            setattr(self, key, value)

        self._x2c_enabled = False

    def x2c(self):
        """Return a copy configured to use PySCF's spin-free X2C method."""
        obj = copy.copy(self)
        obj._x2c_enabled = True
        return obj

    def _resolve_mol(self):
        if hasattr(self._mf_or_mol, "mol"):
            return self._mf_or_mol.mol
        return self._mf_or_mol

    def make_rdm1(self, mo_coeff, mo_occ):
        mol = self._resolve_mol()
        if self.spin_polarized:
            if _mo_is_unrestricted(mo_coeff, mo_occ):
                density_mf = _density_method_for_mol(mol, True)
                if self._x2c_enabled:
                    density_mf = density_mf.x2c()
                return _format_dm_for_spin(density_mf.make_rdm1(mo_coeff, mo_occ), True)
            density_mf = _density_method_for_mol(mol, False)
            if self._x2c_enabled:
                density_mf = density_mf.x2c()
            dm = density_mf.make_rdm1(mo_coeff, mo_occ)
            return np.array((0.5 * dm, 0.5 * dm))
        if _mo_is_unrestricted(mo_coeff, mo_occ):
            density_mf = _density_method_for_mol(mol, True)
            if self._x2c_enabled:
                density_mf = density_mf.x2c()
            return _format_dm_for_spin(density_mf.make_rdm1(mo_coeff, mo_occ), False)
        density_mf = _density_method_for_mol(mol, False)
        if self._x2c_enabled:
            density_mf = density_mf.x2c()
        return density_mf.make_rdm1(mo_coeff, mo_occ)

    def _resolve_density_source(self, dm=None):
        explicit_dm = dm if dm is not None else self.dm
        if self.density_xc is None:
            source = self._mf_or_mol
            if self._x2c_enabled and hasattr(source, "make_rdm1"):
                source = _x2c_method(source)
            if explicit_dm is not None:
                if hasattr(source, "make_rdm1"):
                    return source, explicit_dm
                if hasattr(source, "atom"):
                    density_mf = _density_method_for_mol(source, self.spin_polarized)
                    if self._x2c_enabled:
                        density_mf = density_mf.x2c()
                    return density_mf, explicit_dm
                raise ValueError("A molecule alone does not define a density")
            if hasattr(source, "make_rdm1"):
                return source, _density_matrix_from_mf(source)
            raise ValueError("A molecule alone does not define a density")
        mol = self._resolve_mol()
        if hasattr(self._mf_or_mol, "mol"):
            source = self._mf_or_mol
            dm0 = explicit_dm if explicit_dm is not None else _density_matrix_from_mf(source)
        else:
            source = None
            dm0 = explicit_dm
        density_mf = _clone_density_method(
            source,
            mol=mol,
            xc=self.density_xc,
            spin_polarized=self.spin_polarized,
            grids=copy_grids(self.grids),
            x2c=self._x2c_enabled,
        )
        if dm0 is not None:
            density_mf.kernel(dm0=dm0)
        else:
            density_mf.kernel()
        return density_mf, density_mf.make_rdm1()

    def _evaluate(self, density_mf, dm):
        mol = density_mf.mol if hasattr(density_mf, "mol") else self._resolve_mol()
        spin_polarized = self.spin_polarized
        dm = _format_dm_for_spin(dm, spin_polarized)
        grids = self.grids if self.grids is not None else getattr(density_mf, "grids", None)
        if grids is None:
            grids = dft.Grids(mol)
        grids = IFGridManager(mol, grids).ensure_built(with_non0tab=True)
        nlcgrids = self.nlcgrids
        if bool(getattr(self._numint.ifxc, "requires_nlc_grid", False)):
            if nlcgrids is None:
                nlcgrids = grids
            nlcgrids = IFGridManager(
                mol, nlcgrids
            ).ensure_built(with_non0tab=True)
        ni = self._numint
        feat, nelec = ni.integrate_features(
            mol,
            grids,
            dm,
            spin=1 if spin_polarized else 0,
            max_memory=getattr(self, "max_memory", 2000),
            method=density_mf,
            nlcgrids=nlcgrids,
        )
        context = build_context(mol, spin_polarized=spin_polarized, grids=grids)
        e_ifxc = float(self.if_model.evaluate(feat, order=0, need_energy=True, context=context).energy)
        hf_coeff = float(getattr(self.if_model, "hf_coeff", 0.0))
        cached_energy = _cached_classical_energy_and_hf_exchange(
            density_mf, dm, hf_coeff
        )
        if cached_energy is None:
            e_classical, e_hf_exchange = _classical_energy_and_hf_exchange(
                mol, dm, spin_polarized, template=density_mf, hf_coeff=hf_coeff
            )
        else:
            e_classical, e_hf_exchange = cached_energy
        e_hf_exchange *= hf_coeff
        e_tot = e_classical + e_ifxc + e_hf_exchange
        result = IFDFTEnergyResult(
            functional=self.if_xc,
            density_xc=self.density_xc,
            e_tot=float(e_tot),
            e_classical=float(e_classical),
            e_ifxc=float(e_ifxc),
            e_hf_exchange=float(e_hf_exchange),
            features=np.asarray(feat, dtype=float).reshape(-1),
            feature_labels=tuple(ni.last_feature_labels or getattr(self.if_model, "feature_labels", ())),
            nelec=float(np.sum(np.asarray(nelec, dtype=float))),
            converged_density=getattr(density_mf, "converged", None),
            density_mf=density_mf,
        )
        if self.feature_file is not None:
            _write_feature_file(
                self.feature_file, result, spin_polarized=spin_polarized
            )
        self.results = result
        self.e_tot = result.e_tot
        self.e_ifxc = result.e_ifxc
        self.e_hf_exchange = result.e_hf_exchange
        self.features = result.features
        self.density_mf = density_mf
        return result

    def kernel(self, dm=None):
        density_mf, density_dm = self._resolve_density_source(dm=dm)
        return self._evaluate(density_mf, density_dm)


class IFPostRKS(_IFPostBase):
    spin_polarized = False


class IFPostUKS(_IFPostBase):
    spin_polarized = True
