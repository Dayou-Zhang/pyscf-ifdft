from importlib import resources

import h5py
import numpy as np
import pytest
from pyscf import dft, gto, ifdft

from pyscf.ifdft.models import IFContext
from pyscf.ifdft.registry import get_functional


ML26_EXTRA_LABELS = (
    "r35_2_beta0p1_local_hybrid",
    "r35_2_beta0p1_b05",
    "r35_2_beta0p2_local_hybrid",
    "r35_2_beta0p2_b05",
    "r35_2_beta1p0_local_hybrid",
    "r35_2_beta1p0_b05",
    "r35_2_beta3p0_local_hybrid",
    "r35_2_beta3p0_b05",
    "cs1_same_spin_1",
    "cs1_same_spin_2",
    "cs1_opposite_spin",
    "vv10",
    "e_hf_exchange",
)


def _small_grids(mol):
    grids = dft.Grids(mol)
    grids.level = 0
    grids.prune = None
    return grids


def test_ml26_metadata_and_parameter_archive():
    model = get_functional("ML26")
    ml25 = get_functional("ML25")
    assert model.nfeatures == 79
    assert model.feature_labels[:66] == ml25.feature_labels
    assert model.feature_labels[66:] == ML26_EXTRA_LABELS
    assert model.supports_scf is False
    assert model.supports_gradients is False
    assert model.supports_response is False

    data = resources.files("pyscf.ifdft").joinpath("data/ml26_param.npz")
    with resources.as_file(data) as resolved, np.load(resolved) as params:
        assert params["input_weight"].shape == (5000, 79)
        assert params["output_weight"].shape == (1, 5000)

    normalized = np.linspace(-1.5, 0.75, 79)
    context = IFContext(
        mol=object(), spin_polarized=False, nelec=7, grids=None
    )
    energy = model.evaluate(
        normalized * 7.0,
        order=0,
        need_energy=True,
        context=context,
    ).energy
    np.testing.assert_allclose(energy, -80.60368404948929, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "post_cls, density_cls, atom, spin",
    (
        (ifdft.IFPostRKS, dft.RKS, "H 0 0 0; H 0 0 0.74", 0),
        (ifdft.IFPostUKS, dft.UKS, "Li 0 0 0", 1),
    ),
)
def test_ml26_fixed_density_features(post_cls, density_cls, atom, spin):
    mol = gto.M(atom=atom, basis="sto-3g", spin=spin, verbose=0)
    source = density_cls(mol)
    grids = _small_grids(mol)
    result = post_cls(
        source,
        xc="ML26",
        dm=source.get_init_guess(),
        grids=grids,
        nlcgrids=grids,
    ).kernel()

    assert result.features.shape == (79,)
    assert result.feature_labels[66:] == ML26_EXTRA_LABELS
    assert np.all(np.isfinite(result.features))
    np.testing.assert_allclose(result.e_hf_exchange, 0.0, rtol=0, atol=2e-14)
    np.testing.assert_allclose(
        result.e_tot, result.e_classical + result.e_ifxc, rtol=2e-14, atol=2e-14
    )


def test_ml26_hdf5_feature_metadata(tmp_path):
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0
    )
    source = dft.RKS(mol)
    grids = _small_grids(mol)
    feature_file = tmp_path / "ml26.h5"
    result = ifdft.IFPostRKS(
        source,
        xc="ML26",
        dm=source.get_init_guess(),
        grids=grids,
        nlcgrids=grids,
        feature_file=feature_file,
    ).kernel()

    with h5py.File(feature_file, "r") as fh5:
        features = fh5["ml26_features"]
        np.testing.assert_allclose(
            features[()], result.features, rtol=2e-14, atol=2e-14
        )
        assert features.shape == (79,)
        assert (
            features.attrs["feature_labels"].astype(str)[-1]
            == "e_hf_exchange"
        )
