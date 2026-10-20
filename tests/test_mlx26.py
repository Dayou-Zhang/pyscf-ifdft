import hashlib
from importlib import resources
import shutil

import numpy as np
import pytest
from pyscf import dft, gto, ifdft

from pyscf.ifdft import postscf
from pyscf.ifdft.grids import IFGridManager
from pyscf.ifdft.mixers.base import MixerResult
from pyscf.ifdft.mixers.mlx26 import MLX26Mixer
from pyscf.ifdft.models import IFContext, IFModel, build_feature_map
from pyscf.ifdft.numint import IntegralFeatureNumInt
from pyscf.ifdft.registry import available_functionals, get_functional


# Independently transcribed from the pinned libifxc labels and ML26 descriptor
# definitions; do not derive the expected order from the MLX26 configuration.
EXPECTED_LABELS = (
    ("lak_x", "lak_c")
    + tuple(f"lyp_t{i}" for i in range(1, 7))
    + tuple(
        f"mn15_cc{i}{j}{k}"
        for i, count in enumerate((6, 5, 4, 3))
        for j, width in enumerate(range(count, 2, -1))
        for k in range(width)
    )
    + tuple(f"mn15_a{i}" for i in range(9))
    + tuple(f"mn15_b{i}" for i in range(9))
    + tuple(
        f"r35_2_beta{tag}_{term}"
        for tag in ("0p1", "0p2", "1p0", "3p0")
        for term in ("local_hybrid", "b05")
    )
    + ("cs1_same_spin_1", "cs1_same_spin_2", "cs1_opposite_spin", "vv10")
)
PARAM_SHAPES = {
    "input_weight": (5000, 78),
    "input_bias": (5000,),
    "output_weight": (1, 5000),
    "output_bias": (1,),
    "hf_coeff": (1,),
}
GOLDEN_CASES = (
    (10, -4.5, -5.573230476537696, -10.073230476537695),
    (17, -8.25, -30.711617761115157, -38.96161776111516),
)


def _context(nelec):
    return IFContext(mol=object(), spin_polarized=False, nelec=nelec)


def _normalized_case(nelec):
    j = np.arange(1, 79, dtype=np.float64)
    if nelec == 10:
        return -0.25 / j + 0.01 * np.sin(j)
    return -0.2 / np.sqrt(j) + 0.02 * np.cos(0.5 * j)


@pytest.fixture(scope="module")
def params():
    resource = resources.files("pyscf.ifdft").joinpath("data/mlx26_param.npz")
    with resources.as_file(resource) as path, np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def test_mlx26_registration_and_parameter_archive(params, tmp_path):
    model = get_functional("MLX26")
    assert "MLX26" in available_functionals()
    assert get_functional("mgga_xc_mlx26") is model
    assert get_functional(" mlx26 ") is model
    assert model.nfeatures == model.mixer.nfeatures == 78
    assert model.feature_labels == EXPECTED_LABELS
    assert model.spec.feature_keys == EXPECTED_LABELS
    assert model.feature_set == "ml26"
    assert model.hf_coeff == model.mixer.hf_coeff == 1.0
    assert not any((model.supports_scf, model.supports_gradients, model.supports_response))
    assert set(params) == set(PARAM_SHAPES)
    for key, shape in PARAM_SHAPES.items():
        assert params[key].shape == shape
        assert params[key].dtype == np.dtype("float64")
        assert np.isfinite(params[key]).all()
    np.testing.assert_array_equal(params["hf_coeff"], [1.0])

    resource = resources.files("pyscf.ifdft").joinpath("data/mlx26_param.npz")
    assert resource.is_file()
    assert hashlib.sha256(resource.read_bytes()).hexdigest() == (
        "02105ab57f14ceb6537a3452822e48f25f3ebd543c9cfc590102c4993241963a"
    )
    custom_path = tmp_path / "custom.npz"
    with resources.as_file(resource) as path:
        shutil.copyfile(path, custom_path)
    custom = MLX26Mixer.from_npz(custom_path)
    default = MLX26Mixer.from_npz()
    features = 10 * _normalized_case(10)
    assert custom.energy(features, _context(10)) == default.energy(features, _context(10))


@pytest.mark.parametrize("key", PARAM_SHAPES)
@pytest.mark.parametrize("defect", ("shape", "nonfinite", "precision"))
def test_mlx26_rejects_invalid_parameters(params, key, defect):
    invalid = dict(params)
    value = params[key].copy()
    if defect == "shape":
        value = value[..., :-1]
    elif defect == "nonfinite":
        value.flat[0] = np.nan
    else:
        value = value.astype(np.float32)
    invalid[key] = value
    with pytest.raises((ValueError, TypeError)):
        MLX26Mixer(invalid)


@pytest.mark.parametrize("coefficient", (0.0, -1.0, 100.0, 1.0 + 1e-14))
def test_mlx26_rejects_nonunit_exchange(params, coefficient):
    with pytest.raises(ValueError):
        MLX26Mixer(dict(params, hf_coeff=np.array([coefficient])))


@pytest.mark.parametrize("defect", ("missing", "extra", "pickle"))
def test_mlx26_validates_custom_archive(params, tmp_path, defect):
    invalid = dict(params)
    if defect == "missing":
        invalid.pop("hf_coeff")
    elif defect == "extra":
        invalid["unknown"] = np.array([1.0])
    else:
        invalid["input_bias"] = invalid["input_bias"].astype(object)
    path = tmp_path / "invalid.npz"
    np.savez(path, **invalid)
    with pytest.raises((ValueError, TypeError, KeyError)):
        MLX26Mixer.from_npz(path)


@pytest.mark.parametrize("nelec", (0, -1, np.nan, np.inf, (0, 0), (np.nan, 1)))
def test_mlx26_rejects_invalid_electron_count(nelec):
    with pytest.raises(ValueError):
        MLX26Mixer.from_npz().energy(np.zeros(78), _context(nelec))


@pytest.mark.parametrize("shape", ((77,), (79,), (2, 78)))
def test_mlx26_rejects_invalid_feature_count(shape):
    with pytest.raises(ValueError):
        MLX26Mixer.from_npz().energy(np.zeros(shape), _context(10))


@pytest.mark.parametrize("order", (0, 1, 2))
@pytest.mark.parametrize("need_energy", (False, True))
def test_mlx26_evaluation_contract(order, need_energy):
    mixer = MLX26Mixer.from_npz()
    features = 10 * _normalized_case(10)
    out = MixerResult(123.0, np.ones(1), np.ones((1, 1)))
    result = mixer.evaluate(
        features.reshape(1, 78), order=order, need_energy=need_energy,
        context=_context((6, 4)), out=out,
    )
    assert result is out
    if need_energy:
        np.testing.assert_allclose(result.energy, GOLDEN_CASES[0][2], rtol=0, atol=1e-9)
    else:
        assert result.energy is None
    assert (result.gradient is None) == (order == 0)
    assert (result.hessian is None) == (order < 2)
    if order:
        assert result.gradient.shape == (78,)
    if order == 2:
        assert result.hessian.shape == (78, 78)
    for invalid_order in (-1, 3):
        with pytest.raises(ValueError):
            mixer.evaluate(features, order=invalid_order, need_energy=need_energy)


@pytest.mark.parametrize("nelec,hfx,expected_if,expected_total", GOLDEN_CASES)
def test_mlx26_checkpoint_golden_energies(nelec, hfx, expected_if, expected_total):
    energy = MLX26Mixer.from_npz().energy(nelec * _normalized_case(nelec), _context(nelec))
    np.testing.assert_allclose(energy, expected_if, rtol=0, atol=1e-9)
    np.testing.assert_allclose(energy + hfx, expected_total, rtol=0, atol=1e-9)


@pytest.mark.parametrize("nelec", (2, 17))
def test_mlx26_feature_derivatives_and_electron_scaling(nelec):
    mixer = MLX26Mixer.from_npz()
    z = _normalized_case(10)
    features = nelec * z
    context = _context(nelec)
    result = mixer.evaluate(features, order=2, need_energy=True, context=context)
    unit = mixer.evaluate(z, order=2, need_energy=True, context=_context(1))
    np.testing.assert_allclose(result.energy, nelec * unit.energy, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.gradient, unit.gradient, rtol=0, atol=1e-12)
    np.testing.assert_allclose(nelec * result.hessian, unit.hessian, rtol=0, atol=1e-11)
    np.testing.assert_allclose(result.hessian, result.hessian.T, rtol=0, atol=1e-11)
    step = nelec * 1e-5
    for index in (0, 7, 31, 65, 66, 73, 74, 77):
        plus, minus = features.copy(), features.copy()
        plus[index] += step
        minus[index] -= step
        finite_gradient = (mixer.energy(plus, context) - mixer.energy(minus, context)) / (2 * step)
        finite_hessian = (mixer.gradient(plus, context) - mixer.gradient(minus, context)) / (2 * step)
        np.testing.assert_allclose(result.gradient[index], finite_gradient, rtol=2e-7, atol=2e-7)
        np.testing.assert_allclose(result.hessian[:, index], finite_hessian, rtol=2e-7, atol=2e-7)


def test_mlx26_reorders_sentinel_backend_and_excludes_hfx(monkeypatch):
    spec = get_functional("MLX26").spec
    labels = ("e_hf_exchange",) + EXPECTED_LABELS[::-1]
    values = {label: (index + 1) / 100 for index, label in enumerate(EXPECTED_LABELS)}
    values["e_hf_exchange"] = -12345.0

    class SentinelBackend:
        feature_labels = labels

        def new_evaluator(self):
            return self

        def integrate_features(self, *args, **kwargs):
            return 10, np.array([values[label] for label in labels])

    backend = SentinelBackend()
    feature_map, inverse_map = build_feature_map(spec, backend)
    model = IFModel(spec, backend, feature_map, inverse_map)
    ni = IntegralFeatureNumInt(model)
    monkeypatch.setattr(IFGridManager, "ensure_built", lambda self, **kw: self.grids)
    selected, _ = ni.integrate_features(object(), object(), np.zeros((1, 1)), spin=0)
    np.testing.assert_array_equal(selected, np.arange(1, 79) / 100)
    assert ni.last_feature_labels == EXPECTED_LABELS
    result = model.evaluate(selected, order=2, need_energy=True, context=_context(10))
    assert model.to_backend_gradient(result.gradient)[0] == 0
    hessian = model.to_backend_hessian(result.hessian)
    assert not np.any(hessian[0]) and not np.any(hessian[:, 0])


@pytest.mark.parametrize("nelec,hfx,expected_if,expected_total", GOLDEN_CASES)
def test_mlx26_actual_postscf_exchange_assembly(
    monkeypatch, nelec, hfx, expected_if, expected_total
):
    mol = gto.M(atom="Ne" if nelec == 10 else "Cl", basis="sto-3g", spin=nelec % 2, verbose=0)
    source = (dft.RKS if nelec == 10 else dft.UKS)(mol)
    dm = np.zeros((mol.nao_nr(), mol.nao_nr()))
    if nelec % 2:
        dm = np.array((dm, dm))
    post = (ifdft.IFPostRKS if nelec == 10 else ifdft.IFPostUKS)(source, xc="MLX26", dm=dm)
    features = nelec * _normalized_case(nelec)
    state = {"hfx": hfx, "features": features}
    labels = post._numint.ifxc.feature_labels()
    monkeypatch.setattr(IFGridManager, "ensure_built", lambda self, **kw: self.grids)

    def integrate(*args, **kwargs):
        values = dict(zip(EXPECTED_LABELS, state["features"]))
        values["e_hf_exchange"] = state["hfx"]
        return nelec, np.array([values[label] for label in labels])

    def classical(*args, hf_coeff, **kwargs):
        assert hf_coeff == 1.0
        return 0.0, state["hfx"]

    monkeypatch.setattr(post._numint.ifxc, "integrate_features", integrate)
    monkeypatch.setattr(postscf, "_classical_energy_and_hf_exchange", classical)
    result = post.kernel()
    np.testing.assert_allclose(result.e_ifxc, expected_if, rtol=0, atol=1e-9)
    np.testing.assert_allclose(result.e_tot, expected_total, rtol=0, atol=1e-9)
    assert result.e_classical == 0 and result.e_hf_exchange == hfx
    exchange_step = 0.125

    def exchange_derivatives():
        state["hfx"] = hfx + exchange_step
        plus = post.kernel()
        state["hfx"] = hfx - exchange_step
        minus = post.kernel()
        state["hfx"] = hfx
        center = post.kernel()
        assert plus.e_ifxc == minus.e_ifxc == center.e_ifxc
        np.testing.assert_allclose(plus.e_tot - minus.e_tot, 2 * exchange_step, rtol=0, atol=1e-13)
        gradient = (plus.e_tot - minus.e_tot) / (2 * exchange_step)
        hessian = (plus.e_tot - 2 * center.e_tot + minus.e_tot) / exchange_step**2
        np.testing.assert_allclose(gradient, 1.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(hessian, 0.0, rtol=0, atol=1e-11)
        return gradient

    base_gradient = exchange_derivatives()
    for index in (0, 66, 77):
        state["features"] = features.copy()
        state["features"][index] += 1e-4
        mixed_hessian = (exchange_derivatives() - base_gradient) / 1e-4
        np.testing.assert_allclose(mixed_hessian, 0.0, rtol=0, atol=1e-8)


@pytest.mark.parametrize(
    "post_cls,density_cls,atom,spin",
    (
        (ifdft.IFPostRKS, dft.RKS, "H 0 0 0; H 0 0 0.74", 0),
        (ifdft.IFPostUKS, dft.UKS, "Li 0 0 0", 1),
    ),
)
def test_mlx26_converged_mn15_postscf_components(post_cls, density_cls, atom, spin):
    mol = gto.M(atom=atom, basis="sto-3g", spin=spin, verbose=0)
    source = density_cls(mol, xc="MN15")
    source.grids.level = 0
    source.grids.prune = None
    source.conv_tol = 1e-10
    source.kernel()
    assert source.converged
    dm = source.make_rdm1()
    result = post_cls(source, xc="MLX26", grids=source.grids, nlcgrids=source.grids).kernel()
    assert result.density_mf is source
    assert result.converged_density and source.xc == "MN15"
    np.testing.assert_array_equal(result.density_mf.make_rdm1(), dm)
    assert result.feature_labels == EXPECTED_LABELS
    assert result.features.shape == (78,) and np.isfinite(result.features).all()
    vj, vk = source.get_jk(mol, dm)
    if spin:
        total_dm = dm.sum(axis=0)
        coulomb = 0.5 * np.einsum("ij,ji", total_dm, vj.sum(axis=0))
        exchange = -0.5 * np.einsum("sij,sji", dm, vk)
    else:
        total_dm = dm
        coulomb = 0.5 * np.einsum("ij,ji", dm, vj)
        exchange = -0.25 * np.einsum("ij,ji", dm, vk)
    classical = mol.energy_nuc() + np.einsum("ij,ji", total_dm, source.get_hcore()) + coulomb
    neural = MLX26Mixer.from_npz().energy(result.features, _context(mol.nelectron))
    np.testing.assert_allclose(
        [result.e_classical, result.e_ifxc, result.e_hf_exchange, result.e_tot],
        [classical, neural, exchange, classical + neural + exchange],
        rtol=0, atol=1e-10,
    )
