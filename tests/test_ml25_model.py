from importlib import resources

import numpy as np

from pyscf.ifdft.models import IFContext
from pyscf.ifdft.registry import get_functional


def test_ml25_archive_and_derivative():
    data = resources.files("pyscf.ifdft").joinpath("data/ml25_param.npz")
    assert data.is_file()

    func = get_functional("ML25")
    assert func.spec.feature_set == "ml25"
    assert func.mixer.nfeatures == 66
    assert len(func.feature_labels) == 66
    assert func.feature_labels[0] == "lak_x"

    rng = np.random.default_rng(1)
    feat = rng.normal(size=66)
    ctx = IFContext(mol=object(), spin_polarized=True, nelec=10)
    result = func.evaluate(feat, order=2, need_energy=True, context=ctx)
    assert result.energy is not None
    assert result.gradient is not None
    assert result.hessian is not None
    grad = result.gradient
    assert grad.shape == (66,)

    for idx in (0, 17, 65):
        step = 1e-6
        probe = feat.copy()
        probe[idx] += step
        e_plus = func.evaluate(probe, order=0, need_energy=True, context=ctx).energy
        probe[idx] -= 2 * step
        e_minus = func.evaluate(probe, order=0, need_energy=True, context=ctx).energy
        fd = (e_plus - e_minus) / (2 * step)
        assert np.isclose(fd, grad[idx], rtol=1e-5, atol=1e-5)

    hess = result.hessian
    assert hess.shape == (66, 66)
    assert np.allclose(hess, hess.T, atol=1e-10)

    for idx in (0, 17, 65):
        step = 1e-5
        probe = feat.copy()
        probe[idx] += step
        grad_plus = func.evaluate(probe, order=1, need_energy=False, context=ctx).gradient
        probe[idx] -= 2 * step
        grad_minus = func.evaluate(probe, order=1, need_energy=False, context=ctx).gradient
        fd = (grad_plus - grad_minus) / (2 * step)
        assert np.allclose(fd, hess[:, idx], rtol=5e-4, atol=5e-4)


def test_ml25_evaluate_order_two_without_energy():
    rng = np.random.default_rng(2)
    feat = rng.normal(size=66)
    ctx = IFContext(mol=object(), spin_polarized=False, nelec=2)

    result = get_functional("ML25").evaluate(
        feat, order=2, need_energy=False, context=ctx
    )

    assert result.energy is None
    assert result.gradient.shape == (66,)
    assert result.hessian.shape == (66, 66)
