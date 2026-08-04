import importlib.util

import numpy as np
import pytest
from pyscf import dft, gto, ifdft

from pyscf.ifdft.registry import get_functional


PUBLIC_API = {
    "IFDFTEnergyResult",
    "FeatureInfo",
    "FeatureSetInfo",
    "IFXCFeatureSet",
    "IFPostRKS",
    "IFPostUKS",
    "available_functionals",
    "available_feature_sets",
    "describe_functional",
    "describe_feature_set",
}


def test_public_api_is_postscf_only():
    assert set(ifdft.__all__) == PUBLIC_API
    assert ifdft.available_functionals() == ("ML25", "ML26")
    for name in (
        "RIF",
        "UIF",
        "IFModel",
        "IFModelSpec",
        "get_functional",
        "register_functional",
        "mixers",
    ):
        assert not hasattr(ifdft, name)


@pytest.mark.parametrize("module", ("scf", "response", "grad"))
def test_self_consistent_workflow_modules_are_not_distributed(module):
    spec = importlib.util.find_spec(f"pyscf.ifdft.{module}")
    assert spec is None or spec.origin is None


def test_functional_metadata_disables_self_consistent_capabilities():
    for name in ifdft.available_functionals():
        spec = ifdft.describe_functional(name)
        model = get_functional(name)
        assert spec.supports_scf is False
        assert spec.supports_gradients is False
        assert spec.supports_response is False
        assert model.supports_scf is False
        assert model.supports_gradients is False
        assert model.supports_response is False


def test_unknown_if_functional_is_rejected():
    with pytest.raises(KeyError, match="Unknown IF functional"):
        ifdft.describe_functional("MN15")


@pytest.mark.parametrize(
    "post_cls, density_cls, atom, spin",
    (
        (ifdft.IFPostRKS, dft.RKS, "H 0 0 0; H 0 0 0.74", 0),
        (ifdft.IFPostUKS, dft.UKS, "Li 0 0 0", 1),
    ),
)
def test_postscf_x2c_uses_a_copy(post_cls, density_cls, atom, spin):
    mol = gto.M(atom=atom, basis="sto-3g", spin=spin, verbose=0)
    source = density_cls(mol, xc="PBE")
    post = post_cls(
        source,
        xc="ML25",
        dm=source.get_init_guess(),
        grids=source.grids,
    ).x2c()
    post.grids.level = 0

    result = post.kernel()

    assert result.density_mf is not source
    assert getattr(result.density_mf, "with_x2c", None) is not None
    assert not hasattr(source, "with_x2c")
    assert np.isfinite(result.e_tot)
