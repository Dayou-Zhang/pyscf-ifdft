import numpy as np

from pyscf import dft, gto
from pyscf.dft import numint

from pyscf.ifdft.ifxc import IFXCEvaluator
from pyscf.ifdft.numint import IntegralFeatureNumInt
from pyscf.ifdft.registry import get_functional


def _small_grid(mol):
    grids = dft.Grids(mol)
    grids.level = 0
    grids.build(with_non0tab=True)
    return grids


def _manual_rks_features(ni, mol, grids, dm):
    evaluator = IFXCEvaluator()
    nao = mol.nao_nr()
    make_rho = ni._gen_rho_evaluator(mol, dm, 1, False, grids)[0]
    features = np.zeros(evaluator.nfeatures)
    nelec = 0.0
    for ao, mask, weight, _ in ni.block_loop(mol, grids, nao, 1, max_memory=2000):
        rho = make_rho(0, ao, mask, "MGGA")
        features += _reference_integral_features(evaluator, rho, weight, spin=0)
        nelec += np.dot(weight, rho[0])
    return features, nelec


def _manual_uks_features(ni, mol, grids, dm):
    evaluator = IFXCEvaluator()
    nao = mol.nao_nr()
    dma, dmb = numint._format_uks_dm(dm)
    make_rhoa = ni._gen_rho_evaluator(mol, dma, 1, False, grids)[0]
    make_rhob = ni._gen_rho_evaluator(mol, dmb, 1, False, grids)[0]
    features = np.zeros(evaluator.nfeatures)
    nelec = np.zeros(2)
    for ao, mask, weight, _ in ni.block_loop(mol, grids, nao, 1, max_memory=2000):
        rho_a = make_rhoa(0, ao, mask, "MGGA")
        rho_b = make_rhob(0, ao, mask, "MGGA")
        features += _reference_integral_features(evaluator, (rho_a, rho_b), weight, spin=1)
        nelec[0] += np.dot(weight, rho_a[0])
        nelec[1] += np.dot(weight, rho_b[0])
    return features, nelec


def _symmetrize(dm):
    arr = np.asarray(dm, dtype=float)
    return 0.5 * (arr + np.swapaxes(arr, -1, -2))


def _normalize_mgga_rho(rho):
    arr = np.asarray(rho, dtype=np.float64, order="C")
    if arr.shape[0] == 6:
        arr = np.asarray(arr[[0, 1, 2, 3, 5]], dtype=np.float64, order="C")
    return arr


def _rks_semilocal_input(rho):
    rho = _normalize_mgga_rho(rho)
    grad = rho[1:4]
    sigma = np.sum(grad * grad, axis=0).reshape(1, -1)
    return rho[0:1], sigma, rho[4:5], grad


def _uks_semilocal_input(rho):
    rho_a, rho_b = rho
    rho_a = _normalize_mgga_rho(rho_a)
    rho_b = _normalize_mgga_rho(rho_b)
    grad_a = rho_a[1:4]
    grad_b = rho_b[1:4]
    sigma = np.asarray(
        [
            np.sum(grad_a * grad_a, axis=0),
            np.sum(grad_a * grad_b, axis=0),
            np.sum(grad_b * grad_b, axis=0),
        ],
        dtype=np.float64,
    )
    return (
        np.asarray([rho_a[0], rho_b[0]], dtype=np.float64),
        sigma,
        np.asarray([rho_a[4], rho_b[4]], dtype=np.float64),
        grad_a,
        grad_b,
    )


def _reference_integral_features(evaluator, rho, weight, *, spin):
    if spin == 0:
        rho_arr, sigma_arr, tau_arr, _ = _rks_semilocal_input(rho)
    else:
        rho_arr, sigma_arr, tau_arr, _, _ = _uks_semilocal_input(rho)
    block = evaluator._backend.eval_block(
        rho_arr,
        sigma_arr,
        tau_arr,
        weight,
        spin=spin,
        want_integral_features=True,
    )
    return np.asarray(block.features, dtype=np.float64)


def _reference_weighted_first_derivatives(evaluator, rho, weight, coeff, *, spin):
    if spin == 0:
        rho_arr, sigma_arr, tau_arr, grad = _rks_semilocal_input(rho)
        block = evaluator._backend.eval_block(
            rho_arr,
            sigma_arr,
            tau_arr,
            weight,
            spin=0,
            want_first_derivatives=True,
        )
        d_rho = np.tensordot(block.d_rho, coeff, axes=([1], [0]))
        d_sigma = np.tensordot(block.d_sigma, coeff, axes=([1], [0]))
        d_tau = np.tensordot(block.d_tau, coeff, axes=([1], [0]))
        expected = np.zeros((5, rho_arr.shape[1]), dtype=np.float64)
        expected[0] = d_rho[:, 0]
        expected[1:4] = 2.0 * d_sigma[:, 0] * grad
        expected[4] = d_tau[:, 0]
        return expected

    rho_arr, sigma_arr, tau_arr, grad_a, grad_b = _uks_semilocal_input(rho)
    block = evaluator._backend.eval_block(
        rho_arr,
        sigma_arr,
        tau_arr,
        weight,
        spin=1,
        want_first_derivatives=True,
    )
    d_rho = np.tensordot(block.d_rho, coeff, axes=([1], [0]))
    d_sigma = np.tensordot(block.d_sigma, coeff, axes=([1], [0]))
    d_tau = np.tensordot(block.d_tau, coeff, axes=([1], [0]))
    expected = np.zeros((2, 5, rho_arr.shape[1]), dtype=np.float64)
    expected[0, 0] = d_rho[:, 0]
    expected[1, 0] = d_rho[:, 1]
    expected[0, 1:4] = 2.0 * d_sigma[:, 0] * grad_a + d_sigma[:, 1] * grad_b
    expected[1, 1:4] = 2.0 * d_sigma[:, 2] * grad_b + d_sigma[:, 1] * grad_a
    expected[0, 4] = d_tau[:, 0]
    expected[1, 4] = d_tau[:, 1]
    return expected


def test_integrate_ml25_features_rks_matches_block_backend():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, verbose=0)
    grids = _small_grid(mol)
    dm = dft.RKS(mol).get_init_guess()
    ni = IntegralFeatureNumInt(get_functional("ML25"))

    features, nelec = ni.integrate_features(mol, grids, dm, spin=0)
    expected_features, expected_nelec = _manual_rks_features(ni, mol, grids, dm)

    assert features.shape == (66,)
    assert np.all(np.isfinite(features))
    assert np.allclose(features, expected_features)
    assert np.allclose(nelec, expected_nelec)
    assert ni.last_feature_labels == get_functional("ML25").feature_labels


def test_integrate_ml25_features_uks_matches_block_backend():
    mol = gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1, verbose=0)
    grids = _small_grid(mol)
    dm = dft.UKS(mol).get_init_guess()
    ni = IntegralFeatureNumInt(get_functional("ML25"))

    features, nelec = ni.integrate_features(mol, grids, dm, spin=1)
    expected_features, expected_nelec = _manual_uks_features(ni, mol, grids, dm)

    assert features.shape == (66,)
    assert np.all(np.isfinite(features))
    assert np.allclose(features, expected_features)
    assert np.allclose(nelec, expected_nelec)


def test_weighted_first_derivatives_accept_5_and_6_component_mgga_inputs():
    rng = np.random.default_rng(33)
    npoints = 5
    rho5 = np.vstack(
        [
            rng.uniform(0.4, 1.0, size=npoints),
            rng.normal(scale=0.12, size=(3, npoints)),
            rng.uniform(0.2, 0.7, size=npoints),
        ]
    )
    lapl = rng.normal(scale=0.03, size=npoints)
    rho6 = np.vstack([rho5[0:4], lapl, rho5[4:5]])
    weights = rng.uniform(0.05, 0.4, size=npoints)
    coeff = rng.normal(size=66)
    evaluator = IFXCEvaluator()

    out5 = np.empty((5, npoints), dtype=np.float64)
    direct5 = evaluator.eval_weighted_first_derivatives(rho5, weights, coeff, spin=0, out=out5)
    direct6 = evaluator.eval_weighted_first_derivatives(rho6, weights, coeff, spin=0)
    features_out = np.empty(66, dtype=np.float64)
    features = evaluator.eval_integral_features(rho5, weights, spin=0, out=features_out)
    expected = _reference_weighted_first_derivatives(evaluator, rho5, weights, coeff, spin=0)
    evaluator.close()

    assert direct5 is out5
    assert features is features_out
    assert features.shape == (66,)
    assert direct5.shape == (5, npoints)
    assert np.allclose(direct5, expected, rtol=1e-10, atol=1e-10)
    assert np.allclose(direct6, expected, rtol=1e-10, atol=1e-10)


def test_uks_weighted_first_derivatives_accept_6_component_mgga_inputs():
    rng = np.random.default_rng(34)
    npoints = 4
    rhoa5 = np.vstack(
        [
            rng.uniform(0.4, 1.0, size=npoints),
            rng.normal(scale=0.1, size=(3, npoints)),
            rng.uniform(0.2, 0.7, size=npoints),
        ]
    )
    rhob5 = np.vstack(
        [
            rng.uniform(0.3, 0.9, size=npoints),
            rng.normal(scale=0.1, size=(3, npoints)),
            rng.uniform(0.2, 0.7, size=npoints),
        ]
    )
    rhoa6 = np.vstack([rhoa5[0:4], rng.normal(scale=0.02, size=npoints), rhoa5[4:5]])
    rhob6 = np.vstack([rhob5[0:4], rng.normal(scale=0.02, size=npoints), rhob5[4:5]])
    weights = rng.uniform(0.05, 0.4, size=npoints)
    coeff = rng.normal(size=66)
    evaluator = IFXCEvaluator()

    out = np.empty((2, 5, npoints), dtype=np.float64)
    direct = evaluator.eval_weighted_first_derivatives((rhoa6, rhob6), weights, coeff, spin=1, out=out)
    expected = _reference_weighted_first_derivatives(evaluator, (rhoa5, rhob5), weights, coeff, spin=1)
    evaluator.close()

    assert direct is out
    assert direct.shape == (2, 5, npoints)
    assert np.allclose(direct, expected, rtol=1e-10, atol=1e-10)


def test_rks_weighted_first_derivative_batch_matches_scalar_calls():
    rng = np.random.default_rng(35)
    npoints = 5
    rho = np.vstack(
        [
            rng.uniform(0.4, 1.0, size=npoints),
            rng.normal(scale=0.12, size=(3, npoints)),
            rng.uniform(0.2, 0.7, size=npoints),
        ]
    )
    weights = rng.uniform(0.05, 0.4, size=npoints)
    coeffs = rng.normal(size=(3, 66))
    evaluator = IFXCEvaluator()

    out = np.empty((3, 5, npoints), dtype=np.float64)
    actual = evaluator.eval_weighted_first_derivatives_batch(rho, weights, coeffs, spin=0, out=out)
    expected = np.stack(
        [evaluator.eval_weighted_first_derivatives(rho, weights, coeff, spin=0) for coeff in coeffs]
    )
    evaluator.close()

    assert actual is out
    assert actual.shape == (3, 5, npoints)
    assert np.allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_rks_feature_directional_derivative_matches_feature_finite_difference():
    rng = np.random.default_rng(41)
    npoints = 5
    rho0 = np.vstack(
        [
            rng.uniform(0.6, 1.2, size=npoints),
            rng.normal(scale=0.15, size=(3, npoints)),
            rng.uniform(0.2, 0.8, size=npoints),
        ]
    )
    rho1 = np.vstack(
        [
            rng.normal(scale=0.05, size=npoints),
            rng.normal(scale=0.03, size=(3, npoints)),
            rng.normal(scale=0.02, size=npoints),
        ]
    )
    weights = rng.uniform(0.05, 0.4, size=npoints)
    evaluator = IFXCEvaluator()

    actual = evaluator.eval_feature_directional_derivative(rho0, rho1, weights, spin=0)
    eps = 1e-6
    fp = evaluator.eval_integral_features(rho0 + eps * rho1, weights, spin=0)
    fm = evaluator.eval_integral_features(rho0 - eps * rho1, weights, spin=0)
    fd = (fp - fm) / (2.0 * eps)
    evaluator.close()

    assert actual.shape == (66,)
    assert np.allclose(actual, fd, rtol=2e-5, atol=2e-6)


def test_rks_weighted_response_vvars_matches_weighted_vvar_finite_difference():
    rng = np.random.default_rng(42)
    npoints = 4
    rho0 = np.vstack(
        [
            rng.uniform(0.5, 1.1, size=npoints),
            rng.normal(scale=0.12, size=(3, npoints)),
            rng.uniform(0.2, 0.7, size=npoints),
        ]
    )
    rho1 = np.vstack(
        [
            rng.normal(scale=0.04, size=npoints),
            rng.normal(scale=0.03, size=(3, npoints)),
            rng.normal(scale=0.02, size=npoints),
        ]
    )
    weights = rng.uniform(0.05, 0.4, size=npoints)
    coeff = rng.normal(size=66)
    evaluator = IFXCEvaluator()

    contracted = evaluator.eval_weighted_second_derivative_response(rho0, rho1, weights, coeff, spin=0)
    eps = 1e-6
    plus = evaluator.eval_weighted_first_derivatives(rho0 + eps * rho1, weights, coeff, spin=0)
    minus = evaluator.eval_weighted_first_derivatives(rho0 - eps * rho1, weights, coeff, spin=0)
    finite_difference = (plus - minus) / (2.0 * eps)
    evaluator.close()

    assert contracted.shape == (5, npoints)
    assert np.allclose(contracted, finite_difference, rtol=2e-5, atol=2e-6)


def test_uks_weighted_response_vvars_matches_weighted_vvar_finite_difference():
    rng = np.random.default_rng(43)
    npoints = 4
    rhoa0 = np.vstack(
        [
            rng.uniform(0.4, 1.0, size=npoints),
            rng.normal(scale=0.1, size=(3, npoints)),
            rng.uniform(0.2, 0.7, size=npoints),
        ]
    )
    rhob0 = np.vstack(
        [
            rng.uniform(0.3, 0.9, size=npoints),
            rng.normal(scale=0.1, size=(3, npoints)),
            rng.uniform(0.2, 0.7, size=npoints),
        ]
    )
    rhoa1 = np.vstack(
        [
            rng.normal(scale=0.04, size=npoints),
            rng.normal(scale=0.03, size=(3, npoints)),
            rng.normal(scale=0.02, size=npoints),
        ]
    )
    rhob1 = np.vstack(
        [
            rng.normal(scale=0.04, size=npoints),
            rng.normal(scale=0.03, size=(3, npoints)),
            rng.normal(scale=0.02, size=npoints),
        ]
    )
    weights = rng.uniform(0.05, 0.4, size=npoints)
    coeff = rng.normal(size=66)
    evaluator = IFXCEvaluator()

    contracted = evaluator.eval_weighted_second_derivative_response(
        (rhoa0, rhob0), (rhoa1, rhob1), weights, coeff, spin=1
    )
    eps = 1e-6
    plus = evaluator.eval_weighted_first_derivatives(
        (rhoa0 + eps * rhoa1, rhob0 + eps * rhob1), weights, coeff, spin=1
    )
    minus = evaluator.eval_weighted_first_derivatives(
        (rhoa0 - eps * rhoa1, rhob0 - eps * rhob1), weights, coeff, spin=1
    )
    finite_difference = (plus - minus) / (2.0 * eps)
    evaluator.close()

    assert contracted.shape == (2, 5, npoints)
    assert np.allclose(contracted, finite_difference, rtol=2e-5, atol=2e-6)


def test_uks_response_batch_matches_scalar_response_and_directional_calls():
    rng = np.random.default_rng(44)
    npoints = 4
    rhoa0 = np.vstack(
        [
            rng.uniform(0.4, 1.0, size=npoints),
            rng.normal(scale=0.1, size=(3, npoints)),
            rng.uniform(0.2, 0.7, size=npoints),
        ]
    )
    rhob0 = np.vstack(
        [
            rng.uniform(0.3, 0.9, size=npoints),
            rng.normal(scale=0.1, size=(3, npoints)),
            rng.uniform(0.2, 0.7, size=npoints),
        ]
    )
    rhoa1_batch = np.stack(
        [
            np.vstack(
                [
                    rng.normal(scale=0.04, size=npoints),
                    rng.normal(scale=0.03, size=(3, npoints)),
                    rng.normal(scale=0.02, size=npoints),
                ]
            )
            for _ in range(2)
        ]
    )
    rhob1_batch = np.stack(
        [
            np.vstack(
                [
                    rng.normal(scale=0.04, size=npoints),
                    rng.normal(scale=0.03, size=(3, npoints)),
                    rng.normal(scale=0.02, size=npoints),
                ]
            )
            for _ in range(2)
        ]
    )
    weights = rng.uniform(0.05, 0.4, size=npoints)
    coeff = rng.normal(size=66)
    evaluator = IFXCEvaluator()

    out_kernel = np.empty((2, 2, 5, npoints), dtype=np.float64)
    out_dfeatures = np.empty((2, 66), dtype=np.float64)
    actual_kernel, actual_dfeatures = evaluator.eval_response_batch(
        (rhoa0, rhob0),
        (rhoa1_batch, rhob1_batch),
        weights,
        coeff,
        spin=1,
        out_kernel=out_kernel,
        out_dfeatures=out_dfeatures,
    )
    expected_kernel = np.stack(
        [
            evaluator.eval_weighted_second_derivative_response(
                (rhoa0, rhob0), (rhoa1_batch[i], rhob1_batch[i]), weights, coeff, spin=1
            )
            for i in range(2)
        ]
    )
    expected_dfeatures = np.stack(
        [
            evaluator.eval_feature_directional_derivative(
                (rhoa0, rhob0), (rhoa1_batch[i], rhob1_batch[i]), weights, spin=1
            )
            for i in range(2)
        ]
    )
    evaluator.close()

    assert actual_kernel is out_kernel
    assert actual_dfeatures is out_dfeatures
    assert actual_kernel.shape == (2, 2, 5, npoints)
    assert actual_dfeatures.shape == (2, 66)
    assert np.allclose(actual_kernel, expected_kernel, rtol=1e-10, atol=1e-10)
    assert np.allclose(actual_dfeatures, expected_dfeatures, rtol=1e-10, atol=1e-10)
