import h5py
import numpy as np
import pytest
from pyscf import dft, gto, ifdft


def _small_grids(mol):
    grids = dft.Grids(mol)
    grids.level = 0
    grids.prune = None
    return grids


@pytest.mark.parametrize(
    "post_cls, density_cls, atom, spin",
    (
        (ifdft.IFPostRKS, dft.RKS, "H 0 0 0; H 0 0 0.74", 0),
        (ifdft.IFPostUKS, dft.UKS, "Li 0 0 0", 1),
    ),
)
def test_fixed_density_evaluation_uses_features_only(
    post_cls, density_cls, atom, spin
):
    mol = gto.M(atom=atom, basis="sto-3g", spin=spin, verbose=0)
    source = density_cls(mol)
    post = post_cls(
        source,
        xc="ML25",
        dm=source.get_init_guess(),
        grids=_small_grids(mol),
    )
    calls = 0
    integrate_features = post._numint.integrate_features

    def counted_features(*args, **kwargs):
        nonlocal calls
        calls += 1
        return integrate_features(*args, **kwargs)

    def forbidden_potential(*args, **kwargs):
        raise AssertionError("fixed-density evaluation built an IF potential")

    post._numint.integrate_features = counted_features
    post._numint.nr_rks = forbidden_potential
    post._numint.nr_uks = forbidden_potential
    result = post.kernel()

    assert calls == 1
    assert result.density_mf is source
    assert result.density_xc is None
    assert result.features.shape == (66,)
    assert np.isfinite(result.e_tot)


def test_supplied_converged_density_is_not_rerun(monkeypatch):
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0
    )
    source = dft.RKS(mol, xc="PBE")
    source.grids.level = 0
    source.run()

    def forbidden_kernel(*args, **kwargs):
        raise AssertionError("a supplied density was rerun")

    monkeypatch.setattr(source, "kernel", forbidden_kernel)
    result = ifdft.IFPostRKS(
        source, xc="ML25", grids=source.grids
    ).kernel()

    assert result.density_mf is source
    assert result.converged_density is True
    assert np.isfinite(result.e_tot)


def test_density_xc_generates_an_ordinary_density_without_mutating_source():
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0
    )
    source = dft.RKS(mol, xc="PBE")
    source.grids.level = 0
    source.run()
    source_xc = source.xc
    source_grid_level = source.grids.level

    post = ifdft.IFPostRKS(
        source,
        xc="ML25",
        density_xc="LDA",
        grids=_small_grids(mol),
    )
    result = post.kernel()

    assert result.density_mf is not source
    assert result.density_mf.xc == "LDA"
    assert result.density_xc == "LDA"
    assert source.xc == source_xc
    assert source.grids.level == source_grid_level
    assert np.isfinite(result.e_tot)


def test_feature_hdf5_output_replaces_only_ifdft_datasets(tmp_path):
    mol = gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1, verbose=0)
    source = dft.UKS(mol)
    feature_file = tmp_path / "features.h5"
    with h5py.File(feature_file, "w") as fh5:
        fh5["unrelated"] = 7

    result = ifdft.IFPostUKS(
        source,
        xc="ML25",
        dm=source.get_init_guess(),
        grids=_small_grids(mol),
        feature_file=feature_file,
    ).kernel()

    with h5py.File(feature_file, "r") as fh5:
        assert fh5["unrelated"][()] == 7
        np.testing.assert_allclose(
            fh5["ml25_features"][()], result.features, rtol=2e-14, atol=2e-14
        )
        np.testing.assert_array_equal(
            fh5["ml25_features"].attrs["feature_labels"].astype(str),
            result.feature_labels,
        )
        np.testing.assert_allclose(
            fh5["e_classical"][()], result.e_classical, rtol=2e-14, atol=2e-14
        )
        np.testing.assert_allclose(
            fh5["nelec"][()], np.rint(result.nelec), rtol=2e-14, atol=2e-14
        )
