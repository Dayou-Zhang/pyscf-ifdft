import os
from pathlib import Path

import numpy as np
import pytest

from pyscf import lib

from pyscf.ifdft import _libifxc
from pyscf.ifdft.backends import libifxc
from pyscf.ifdft.exceptions import IFDFTError


def test_libifxc_bridge_loads_with_pyscf_loader():
    assert lib.load_library("libifdft_ifxc") is not None


def test_libifxc_metadata():
    assert _libifxc._version() == (0, 4, 0)
    assert libifxc.available_feature_sets() == ("ml25", "ml26")

    info = libifxc.describe_feature_set("ml25")
    assert info.key == "ml25"
    assert info.name == "ML25 integral features"
    assert info.nfeatures == 66
    assert info.max_deriv_order >= 2
    assert info.xctype == "MGGA"
    assert info.labels[0] == "lak_x"
    assert info.labels[8:48] == tuple(
        f"mn15_cc{i}{j}{k}"
        for i, count in enumerate((6, 5, 4, 3))
        for j, width in enumerate(range(count, 2, -1))
        for k in range(width)
    )
    assert info.labels[48:57] == tuple(f"mn15_a{i}" for i in range(9))
    assert info.labels[57:66] == tuple(f"mn15_b{i}" for i in range(9))
    assert libifxc.describe_feature_set("mgga_xc_ml25").key == "ml25"

    ml26 = libifxc.describe_feature_set("ml26")
    assert ml26.name == "ML26 semilocal integral features"
    assert ml26.nfeatures == 69
    assert ml26.max_deriv_order == 2
    assert ml26.xctype == "MGGA"
    assert ml26.labels[:66] == info.labels
    assert ml26.labels[66:] == (
        "cs1_same_spin_1",
        "cs1_same_spin_2",
        "cs1_opposite_spin",
    )
    assert libifxc.describe_feature_set("MGGA_XC_ML26").key == "ml26"


def test_libifxc_ml26_block_through_second_order():
    rho = np.full((1, 3), 0.7)
    sigma = np.full((1, 3), 0.2)
    tau = np.full((1, 3), 0.4)
    weights = np.array([0.1, 0.2, 0.3])

    with libifxc.IFXCFeatureSet("ml26") as backend:
        block = backend.eval_block(
            rho,
            sigma,
            tau,
            weights,
            spin=0,
            want_integral_features=True,
            want_local_values=True,
            want_first_derivatives=True,
            want_second_derivatives=True,
        )

    assert block.features.shape == (69,)
    assert block.local_values.shape == (3, 69)
    np.testing.assert_allclose(
        block.features,
        np.dot(weights, block.local_values),
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.all(np.isfinite(block.features))
    assert np.all(np.isfinite(block.d_rho))
    assert np.all(np.isfinite(block.dd_rho_rho))


@pytest.mark.parametrize("spin", [0, 1])
def test_ml26_prefix_derivatives_equal_ml25_through_second_order(spin):
    rng = np.random.default_rng(2600 + spin)
    npoints = 3
    if spin == 0:
        rho = rng.uniform(0.3, 1.0, size=(1, npoints))
        sigma = rng.uniform(0.02, 0.2, size=(1, npoints))
        tau = rng.uniform(0.1, 0.8, size=(1, npoints))
    else:
        rho = rng.uniform(0.3, 1.0, size=(2, npoints))
        sigma = rng.uniform(0.02, 0.2, size=(3, npoints))
        sigma[1] = np.minimum(
            sigma[1], 0.5 * (sigma[0] + sigma[2])
        )
        tau = rng.uniform(0.1, 0.8, size=(2, npoints))
    weights = rng.uniform(0.1, 0.5, size=npoints)

    flags = dict(
        spin=spin,
        want_local_values=True,
        want_first_derivatives=True,
        want_second_derivatives=True,
    )
    with libifxc.IFXCFeatureSet("ml25") as ml25:
        prefix = ml25.eval_block(rho, sigma, tau, weights, **flags)
    with libifxc.IFXCFeatureSet("ml26") as ml26:
        extended = ml26.eval_block(rho, sigma, tau, weights, **flags)

    np.testing.assert_allclose(
        extended.local_values[:, :66],
        prefix.local_values,
        rtol=2e-14,
        atol=2e-14,
    )
    for name in (
        "d_rho",
        "d_sigma",
        "d_tau",
        "dd_rho_rho",
        "dd_rho_sigma",
        "dd_rho_tau",
        "dd_sigma_sigma",
        "dd_sigma_tau",
        "dd_tau_tau",
    ):
        np.testing.assert_array_equal(
            getattr(extended, name)[:, :66],
            getattr(prefix, name),
        )


@pytest.mark.parametrize("spin", [0, 1])
def test_ml26_contracted_first_derivatives_match_full_tensor(spin):
    rng = np.random.default_rng(2670 + spin)
    npoints = 4
    rho_mgga = _random_pyscf_mgga(rng, npoints, spin)
    weights = rng.uniform(0.05, 0.4, size=npoints)
    coefficients = rng.normal(size=69)

    from pyscf.ifdft.ifxc import IFXCEvaluator

    evaluator = IFXCEvaluator("ml26")
    try:
        if spin == 0:
            rho, sigma, tau = evaluator._semilocal_input(
                rho_mgga, spin=spin
            )
            gradients = rho_mgga[1:4]
        else:
            rho, sigma, tau = evaluator._semilocal_input(
                rho_mgga, spin=spin
            )
            gradients = rho_mgga[:, 1:4]
        with libifxc.IFXCFeatureSet("ml26") as backend:
            full = backend.eval_block(
                rho,
                sigma,
                tau,
                weights,
                spin=spin,
                want_first_derivatives=True,
            )
            actual = backend.eval_weighted_vvars_from_pyscf_mgga(
                rho_mgga, weights, coefficients, spin=spin
            )
    finally:
        evaluator.close()

    d_rho = np.tensordot(full.d_rho, coefficients, axes=([1], [0]))
    d_sigma = np.tensordot(full.d_sigma, coefficients, axes=([1], [0]))
    d_tau = np.tensordot(full.d_tau, coefficients, axes=([1], [0]))
    if spin == 0:
        expected = np.zeros((5, npoints))
        expected[0] = d_rho[:, 0]
        expected[1:4] = 2.0 * d_sigma[:, 0] * gradients
        expected[4] = d_tau[:, 0]
    else:
        expected = np.zeros((2, 5, npoints))
        expected[0, 0] = d_rho[:, 0]
        expected[1, 0] = d_rho[:, 1]
        expected[0, 1:4] = (
            2.0 * d_sigma[:, 0] * gradients[0]
            + d_sigma[:, 1] * gradients[1]
        )
        expected[1, 1:4] = (
            2.0 * d_sigma[:, 2] * gradients[1]
            + d_sigma[:, 1] * gradients[0]
        )
        expected[0, 4] = d_tau[:, 0]
        expected[1, 4] = d_tau[:, 1]
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def _unpack_symmetric(values, dimension):
    values = np.asarray(values)
    unpacked = np.empty(
        (values.shape[0], dimension, dimension), dtype=np.float64
    )
    component = 0
    for row in range(dimension):
        for column in range(row, dimension):
            unpacked[:, row, column] = values[:, component]
            unpacked[:, column, row] = values[:, component]
            component += 1
    return unpacked


@pytest.mark.parametrize("spin", [0, 1])
def test_ml26_first_and_second_derivatives_match_local_differences(spin):
    if spin == 0:
        dimensions = (1, 1, 1)
        inputs = [
            np.asarray([[0.73]]),
            np.asarray([[0.08]]),
            np.asarray([[0.41]]),
        ]
    else:
        dimensions = (2, 3, 2)
        inputs = [
            np.asarray([[0.61], [0.43]]),
            np.asarray([[0.06], [0.012], [0.045]]),
            np.asarray([[0.38], [0.27]]),
        ]
    weights = np.ones(1)

    with libifxc.IFXCFeatureSet("ml26") as backend:
        base = backend.eval_block(
            *inputs,
            weights,
            spin=spin,
            want_local_values=True,
            want_first_derivatives=True,
            want_second_derivatives=True,
        )
        first = np.concatenate(
            (base.d_rho[0], base.d_sigma[0], base.d_tau[0]), axis=1
        )
        nrho, nsigma, ntau = dimensions
        nvariables = sum(dimensions)
        hessian = np.zeros((69, nvariables, nvariables))
        rho_slice = slice(0, nrho)
        sigma_slice = slice(nrho, nrho + nsigma)
        tau_slice = slice(nrho + nsigma, nvariables)
        hessian[:, rho_slice, rho_slice] = _unpack_symmetric(
            base.dd_rho_rho[0], nrho
        )
        hessian[:, sigma_slice, sigma_slice] = _unpack_symmetric(
            base.dd_sigma_sigma[0], nsigma
        )
        hessian[:, tau_slice, tau_slice] = _unpack_symmetric(
            base.dd_tau_tau[0], ntau
        )
        mixed_blocks = (
            (
                rho_slice,
                sigma_slice,
                base.dd_rho_sigma[0].reshape(69, nrho, nsigma),
            ),
            (
                rho_slice,
                tau_slice,
                base.dd_rho_tau[0].reshape(69, nrho, ntau),
            ),
            (
                sigma_slice,
                tau_slice,
                base.dd_sigma_tau[0].reshape(69, nsigma, ntau),
            ),
        )
        for left, right, block in mixed_blocks:
            hessian[:, left, right] = block
            hessian[:, right, left] = block.transpose(0, 2, 1)

        np.testing.assert_array_equal(
            hessian, hessian.transpose(0, 2, 1)
        )
        step = 1e-6
        for variable in range(nvariables):
            plus = [np.array(values, copy=True) for values in inputs]
            minus = [np.array(values, copy=True) for values in inputs]
            component = variable
            for group, dimension in enumerate(dimensions):
                if component < dimension:
                    plus[group][component, 0] += step
                    minus[group][component, 0] -= step
                    break
                component -= dimension
            flags = dict(
                spin=spin,
                want_local_values=True,
                want_first_derivatives=True,
            )
            plus_block = backend.eval_block(
                *plus, weights, **flags
            )
            minus_block = backend.eval_block(
                *minus, weights, **flags
            )
            finite_first = (
                plus_block.local_values[0]
                - minus_block.local_values[0]
            ) / (2.0 * step)
            plus_first = np.concatenate(
                (
                    plus_block.d_rho[0],
                    plus_block.d_sigma[0],
                    plus_block.d_tau[0],
                ),
                axis=1,
            )
            minus_first = np.concatenate(
                (
                    minus_block.d_rho[0],
                    minus_block.d_sigma[0],
                    minus_block.d_tau[0],
                ),
                axis=1,
            )
            finite_second = (plus_first - minus_first) / (2.0 * step)
            np.testing.assert_allclose(
                first[:, variable],
                finite_first,
                rtol=2e-7,
                atol=5e-9,
            )
            np.testing.assert_allclose(
                hessian[:, :, variable],
                finite_second,
                rtol=3e-7,
                atol=5e-9,
            )


@pytest.mark.parametrize("spin", [0, 1])
def test_ml26_serial_and_openmp_contracted_paths_agree(spin):
    rng = np.random.default_rng(2690 + spin)
    npoints = 777
    rho = _random_pyscf_mgga(rng, npoints, spin)
    weights = rng.uniform(0.01, 0.2, size=npoints)
    coefficients = rng.normal(size=69)

    with libifxc.IFXCFeatureSet("ml26") as backend:
        with lib.with_omp_threads(1):
            serial_features = backend.eval_features_from_pyscf_mgga(
                rho, weights, spin=spin
            )
            serial_vvars = backend.eval_weighted_vvars_from_pyscf_mgga(
                rho, weights, coefficients, spin=spin
            )
        with lib.with_omp_threads(8):
            parallel_features = backend.eval_features_from_pyscf_mgga(
                rho, weights, spin=spin
            )
            parallel_vvars = backend.eval_weighted_vvars_from_pyscf_mgga(
                rho, weights, coefficients, spin=spin
            )

    np.testing.assert_allclose(
        parallel_features, serial_features, rtol=2e-13, atol=2e-13
    )
    np.testing.assert_allclose(
        parallel_vvars, serial_vvars, rtol=2e-13, atol=2e-13
    )


def test_unknown_feature_set_fails_clearly():
    with pytest.raises(IFDFTError, match="libifxc bridge call failed"):
        libifxc.describe_feature_set("does_not_exist")


def test_libifxc_block_smoke():
    rho = np.full((1, 3), 0.7)
    sigma = np.full((1, 3), 0.2)
    tau = np.full((1, 3), 0.4)
    weights = np.array([0.1, 0.2, 0.3])

    with libifxc.IFXCFeatureSet("ml25") as backend:
        block = backend.eval_block(
            rho,
            sigma,
            tau,
            weights,
            spin=0,
            want_integral_features=True,
            want_local_values=True,
            want_first_derivatives=True,
        )

    assert block.features.shape == (66,)
    assert block.local_values.shape == (3, 66)
    assert block.d_rho.shape == (3, 66, 1)
    assert block.d_sigma.shape == (3, 66, 1)
    assert block.d_tau.shape == (3, 66, 1)
    assert np.all(np.isfinite(block.features))
    assert np.all(np.isfinite(block.local_values))
    assert np.all(np.isfinite(block.d_rho))


def test_feature_set_handle_reuses_evaluator_for_many_blocks():
    rho = np.full((1, 2), 0.7)
    sigma = np.full((1, 2), 0.2)
    tau = np.full((1, 2), 0.4)
    weights = np.array([0.1, 0.2])

    with libifxc.IFXCFeatureSet("ml25") as backend:
        for _ in range(100):
            block = backend.eval_block(rho, sigma, tau, weights, spin=0, want_integral_features=True)
            assert block.features.shape == (66,)
            assert np.all(np.isfinite(block.features))


def test_weighted_vvars_rks_matches_full_derivative_contraction():
    rng = np.random.default_rng(31)
    npoints = 5
    rho_gradient = rng.normal(size=(3, npoints))
    rho = rng.uniform(0.3, 1.2, size=(1, npoints))
    sigma = np.einsum("xg,xg->g", rho_gradient, rho_gradient, optimize=True).reshape(1, npoints)
    tau = rng.uniform(0.1, 0.9, size=(1, npoints))
    weights = rng.uniform(0.05, 0.4, size=npoints)
    coeff = rng.normal(size=66)

    with libifxc.IFXCFeatureSet("ml25") as backend:
        block = backend.eval_block(rho, sigma, tau, weights, spin=0, want_first_derivatives=True)
        actual = backend.eval_weighted_vvars(rho, sigma, tau, rho_gradient, weights, coeff, spin=0)

    d_rho = np.tensordot(block.d_rho, coeff, axes=([1], [0]))
    d_sigma = np.tensordot(block.d_sigma, coeff, axes=([1], [0]))
    d_tau = np.tensordot(block.d_tau, coeff, axes=([1], [0]))
    expected = np.zeros((5, npoints))
    expected[0] = d_rho[:, 0]
    expected[1:4] = 2.0 * d_sigma[:, 0] * rho_gradient
    expected[4] = d_tau[:, 0]

    assert actual.shape == (5, npoints)
    assert np.allclose(actual, expected)


def test_weighted_vvars_uks_matches_full_derivative_contraction():
    rng = np.random.default_rng(32)
    npoints = 4
    grad_a = rng.normal(size=(3, npoints))
    grad_b = rng.normal(size=(3, npoints))
    rho = rng.uniform(0.2, 1.1, size=(2, npoints))
    sigma = np.asarray(
        [
            np.einsum("xg,xg->g", grad_a, grad_a, optimize=True),
            np.einsum("xg,xg->g", grad_a, grad_b, optimize=True),
            np.einsum("xg,xg->g", grad_b, grad_b, optimize=True),
        ]
    )
    tau = rng.uniform(0.1, 0.8, size=(2, npoints))
    rho_gradient = np.asarray([grad_a, grad_b])
    weights = rng.uniform(0.05, 0.4, size=npoints)
    coeff = rng.normal(size=66)

    with libifxc.IFXCFeatureSet("ml25") as backend:
        block = backend.eval_block(rho, sigma, tau, weights, spin=1, want_first_derivatives=True)
        actual = backend.eval_weighted_vvars(rho, sigma, tau, rho_gradient, weights, coeff, spin=1)

    d_rho = np.tensordot(block.d_rho, coeff, axes=([1], [0]))
    d_sigma = np.tensordot(block.d_sigma, coeff, axes=([1], [0]))
    d_tau = np.tensordot(block.d_tau, coeff, axes=([1], [0]))
    expected = np.zeros((2, 5, npoints))
    expected[0, 0] = d_rho[:, 0]
    expected[1, 0] = d_rho[:, 1]
    expected[0, 1:4] = 2.0 * d_sigma[:, 0] * grad_a + d_sigma[:, 1] * grad_b
    expected[1, 1:4] = 2.0 * d_sigma[:, 2] * grad_b + d_sigma[:, 1] * grad_a
    expected[0, 4] = d_tau[:, 0]
    expected[1, 4] = d_tau[:, 1]

    assert actual.shape == (2, 5, npoints)
    assert np.allclose(actual, expected)


def _random_pyscf_mgga(rng, npoints, spin):
    def one_spin():
        rho = np.empty((6, npoints))
        rho[0] = rng.uniform(0.2, 1.1, size=npoints)
        rho[1:4] = rng.normal(scale=0.3, size=(3, npoints))
        rho[4] = rng.normal(scale=0.1, size=npoints)
        rho[5] = rng.uniform(0.1, 0.8, size=npoints)
        return rho

    rho_a = one_spin()
    if spin == 0:
        return rho_a
    return np.asarray([rho_a, one_spin()])


@pytest.mark.parametrize("spin", [0, 1])
@pytest.mark.parametrize("npoints", [127, 512, 777])
def test_parallel_pyscf_mgga_features_and_weighted_vvars_match_serial(spin, npoints):
    rng = np.random.default_rng(4100 + 10 * spin + npoints)
    rho = _random_pyscf_mgga(rng, npoints, spin)
    weights = rng.uniform(0.01, 0.2, size=npoints)
    coeff = rng.normal(size=66)

    with libifxc.IFXCFeatureSet("ml25") as backend:
        with lib.with_omp_threads(1):
            serial_features = backend.eval_features_from_pyscf_mgga(rho, weights, spin=spin)
            serial_vvars = backend.eval_weighted_vvars_from_pyscf_mgga(
                rho, weights, coeff, spin=spin
            )
        with lib.with_omp_threads(32):
            parallel_features = backend.eval_features_from_pyscf_mgga(rho, weights, spin=spin)
            parallel_vvars = backend.eval_weighted_vvars_from_pyscf_mgga(
                rho, weights, coeff, spin=spin
            )
            repeated_features = backend.eval_features_from_pyscf_mgga(rho, weights, spin=spin)

    assert np.allclose(parallel_features, serial_features, rtol=2e-13, atol=2e-13)
    assert np.allclose(parallel_vvars, serial_vvars, rtol=2e-13, atol=2e-13)
    assert np.array_equal(repeated_features, parallel_features)


@pytest.mark.parametrize("spin", [0, 1])
@pytest.mark.parametrize("npoints", [127, 777])
@pytest.mark.parametrize("nbatch", [1, 3])
def test_parallel_batch_kernels_match_serial_and_reuse_outputs(spin, npoints, nbatch):
    rng = np.random.default_rng(6200 + 10 * spin + npoints)
    rho0 = _random_pyscf_mgga(rng, npoints, spin)
    rho1 = np.stack([_random_pyscf_mgga(rng, npoints, spin) for _ in range(nbatch)])
    weights = rng.uniform(0.01, 0.2, size=npoints)
    coefficients = rng.normal(size=(nbatch, 66))
    response_coeff = rng.normal(size=66)
    vvar_shape = (nbatch, 5, npoints) if spin == 0 else (nbatch, 2, 5, npoints)
    kernel_shape = (nbatch, 5, npoints) if spin == 0 else (nbatch, 2, 5, npoints)

    with libifxc.IFXCFeatureSet("ml25") as backend:
        with lib.with_omp_threads(1):
            serial_vvars = backend.eval_weighted_vvars_batch_from_pyscf_mgga(
                rho0, weights, coefficients, spin=spin
            )
            serial_kernel, serial_dfeatures = backend.eval_response_batch_from_pyscf_mgga(
                rho0, rho1, weights, response_coeff, spin=spin
            )

        vvar_out = np.empty(vvar_shape)
        kernel_out = np.empty(kernel_shape)
        dfeature_out = np.empty((nbatch, 66))
        with lib.with_omp_threads(4):
            parallel_vvars = backend.eval_weighted_vvars_batch_from_pyscf_mgga(
                rho0, weights, coefficients, spin=spin, out=vvar_out
            )
            parallel_kernel, parallel_dfeatures = backend.eval_response_batch_from_pyscf_mgga(
                rho0,
                rho1,
                weights,
                response_coeff,
                spin=spin,
                out_kernel=kernel_out,
                out_dfeatures=dfeature_out,
            )
            repeated_kernel, repeated_dfeatures = backend.eval_response_batch_from_pyscf_mgga(
                rho0, rho1, weights, response_coeff, spin=spin
            )

    assert parallel_vvars is vvar_out
    assert parallel_kernel is kernel_out
    assert parallel_dfeatures is dfeature_out
    assert np.allclose(parallel_vvars, serial_vvars, rtol=2e-13, atol=2e-13)
    assert np.allclose(parallel_kernel, serial_kernel, rtol=2e-13, atol=2e-13)
    assert np.allclose(parallel_dfeatures, serial_dfeatures, rtol=2e-13, atol=2e-13)
    assert np.array_equal(repeated_kernel, parallel_kernel)
    assert np.array_equal(repeated_dfeatures, parallel_dfeatures)


def test_many_streamed_blocks_keep_bounded_scratch_and_do_not_spill(tmp_path, monkeypatch):
    statm = Path("/proc/self/statm")
    if not statm.exists():
        pytest.skip("RSS regression requires Linux /proc")

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    rng = np.random.default_rng(5112)
    npoints = 1024
    rho = _random_pyscf_mgga(rng, npoints, spin=1)
    weights = rng.uniform(0.01, 0.2, size=npoints)
    coeff = rng.normal(size=66)
    feature_out = np.empty(66)
    vvar_out = np.empty((2, 5, npoints))

    def rss_bytes():
        resident_pages = int(statm.read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")

    with libifxc.IFXCFeatureSet("ml25") as backend, lib.with_omp_threads(8):
        # Warm every worker and allocate the maximum per-block scratch first.
        backend.eval_features_from_pyscf_mgga(rho, weights, spin=1, out=feature_out)
        backend.eval_weighted_vvars_from_pyscf_mgga(
            rho, weights, coeff, spin=1, out=vvar_out
        )
        rss_before = rss_bytes()
        for _ in range(50):
            backend.eval_features_from_pyscf_mgga(rho, weights, spin=1, out=feature_out)
            backend.eval_weighted_vvars_from_pyscf_mgga(
                rho, weights, coeff, spin=1, out=vvar_out
            )
        rss_after = rss_bytes()

    assert rss_after - rss_before < 16 * 1024**2
    assert not any(tmp_path.iterdir())
