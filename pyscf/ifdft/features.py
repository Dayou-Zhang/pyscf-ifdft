from dataclasses import dataclass

from pyscf import dft

from .grids import copy_grids
from .models import IFContext


@dataclass(frozen=True)
class FeatureInfo:
    index: int
    key: str


@dataclass(frozen=True)
class FeatureSetInfo:
    key: str
    name: str
    nfeatures: int
    max_deriv_order: int
    xctype: str
    labels: tuple[str, ...]


def build_context(mol, *, spin_polarized: bool, grids=None, nelec=None):
    if nelec is None:
        nelec = getattr(mol, "nelec", mol.nelectron)
    return IFContext(mol=mol, spin_polarized=spin_polarized, nelec=nelec, grids=grids)


def make_density_mf(mol, *, spin_polarized: bool, xc: str):
    return dft.UKS(mol, xc=xc) if spin_polarized else dft.RKS(mol, xc=xc)
