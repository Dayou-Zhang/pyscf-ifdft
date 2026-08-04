import numpy as np

from pyscf.ifdft import registry as registry_mod
from pyscf.ifdft.mixers import LinearMixer
from pyscf.ifdft.models import IFModel, IFModelSpec, build_feature_map
from pyscf.ifdft.numint import IntegralFeatureNumInt
from pyscf.ifdft.registry import (
    available_functionals,
    describe_functional,
    get_functional,
    register_functional,
)


class ToyBackend:
    nfeatures = 3

    def feature_labels(self):
        return ("a", "b", "c")


def test_linear_mixer_evaluate_contract():
    mixer = LinearMixer([1.0, -2.0, 0.5], intercept=0.25)
    features = np.array([3.0, 4.0, 5.0])

    result = mixer.evaluate(features, order=1, need_energy=True)
    assert np.isclose(result.energy, 0.25 + 3.0 - 8.0 + 2.5)
    assert np.allclose(result.gradient, [1.0, -2.0, 0.5])
    assert result.hessian is None

    result = mixer.evaluate(features, order=2, need_energy=False)
    assert result.energy is None
    assert np.allclose(result.gradient, [1.0, -2.0, 0.5])
    assert result.hessian.shape == (3, 3)
    assert np.count_nonzero(result.hessian) == 0


def test_model_feature_key_reordering_and_backend_embedding():
    mixer = LinearMixer([2.0, -1.0])
    spec = IFModelSpec(
        name="TOY",
        feature_set="toy",
        mixer=mixer,
        feature_keys=("c", "a"),
    )
    feature_map, inverse_map = build_feature_map(spec, ToyBackend())
    model = IFModel(spec=spec, backend=ToyBackend(), feature_map=feature_map, inverse_map=inverse_map)

    backend_features = np.array([10.0, 20.0, 30.0])
    model_features = model.select_features(backend_features)
    result = model.evaluate(model_features, order=2, need_energy=True)

    assert np.allclose(model_features, [30.0, 10.0])
    assert np.isclose(result.energy, 50.0)
    assert np.allclose(result.gradient, [2.0, -1.0])
    assert np.allclose(model.to_backend_gradient(result.gradient), [-1.0, 0.0, 2.0])
    assert np.allclose(model.to_backend_hessian(result.hessian), np.zeros((3, 3)))


def test_register_functional_requires_model_spec():
    mixer = LinearMixer([1.0])
    spec = IFModelSpec(
        name="TMP_LINEAR",
        feature_set="mgga_xc_ml25",
        mixer=mixer,
        feature_keys=("lak_x",),
    )

    try:
        register_functional(spec, aliases=("TMP_LINEAR_ALIAS",))
        model = get_functional("TMP_LINEAR_ALIAS")
        assert "TMP_LINEAR" in available_functionals()
        assert describe_functional("TMP_LINEAR_ALIAS") is spec
        assert model.backend.feature_set == "ml25"
        assert model.feature_labels == ("lak_x",)
    finally:
        registry_mod._SPECS.pop("TMP_LINEAR", None)
        registry_mod._MODELS.pop("TMP_LINEAR", None)
        registry_mod._ALIASES.pop("TMP_LINEAR", None)
        registry_mod._ALIASES.pop("TMP_LINEAR_ALIAS", None)


def test_numint_evaluator_factory_uses_functional_feature_set(monkeypatch):
    import pyscf.ifdft.ifxc as ifxc_mod

    calls = []

    class DummyEvaluator:
        def __init__(self, feature_set):
            calls.append(feature_set)
            self.feature_set = feature_set

    class DummyFunctional:
        feature_set = "future_feature_set"

    monkeypatch.setattr(ifxc_mod, "IFXCEvaluator", DummyEvaluator)

    evaluator = IntegralFeatureNumInt._make_ifxc_evaluator(DummyFunctional())

    assert calls == ["future_feature_set"]
    assert evaluator.feature_set == "future_feature_set"
