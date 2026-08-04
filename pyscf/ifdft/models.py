from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mixers.base import IFMixer, MixerResult


@dataclass(frozen=True)
class IFContext:
    mol: object
    spin_polarized: bool
    nelec: float | tuple[float, float]
    grids: object | None = None


@dataclass(frozen=True)
class IFModelSpec:
    name: str
    feature_set: str
    mixer: IFMixer
    hf_coeff: float = 0.0
    rsh_coeff: tuple[float, float, float] = (0.0, 0.0, 0.0)
    xctype: str = "MGGA"
    feature_keys: tuple[str, ...] | None = None
    spin_policy: str = "both"
    supports_scf: bool = False
    supports_gradients: bool = False
    supports_response: bool = False


@dataclass
class IFModel:
    spec: IFModelSpec
    backend: object
    feature_map: np.ndarray
    inverse_map: np.ndarray | None = None

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def feature_set(self) -> str:
        return self.spec.feature_set

    @property
    def mixer(self) -> IFMixer:
        return self.spec.mixer

    @property
    def hf_coeff(self) -> float:
        return self.spec.hf_coeff

    @property
    def xctype(self) -> str:
        return self.spec.xctype

    @property
    def rsh_coeff(self) -> tuple[float, float, float]:
        return self.spec.rsh_coeff

    @property
    def supports_scf(self) -> bool:
        return self.spec.supports_scf

    @property
    def supports_gradients(self) -> bool:
        return self.spec.supports_gradients

    @property
    def supports_response(self) -> bool:
        return self.spec.supports_response

    @property
    def feature_labels(self) -> tuple[str, ...]:
        labels = _backend_feature_labels(self.backend)
        return tuple(labels[int(idx)] for idx in self.feature_map)

    @property
    def nfeatures(self) -> int:
        return int(self.feature_map.size)

    @property
    def nbackend_features(self) -> int:
        return len(_backend_feature_labels(self.backend))

    def select_features(self, backend_features) -> np.ndarray:
        arr = np.asarray(backend_features, dtype=np.float64)
        if self.inverse_map is None:
            return arr
        return np.asarray(arr[..., self.feature_map], dtype=np.float64)

    def to_backend_gradient(self, gradient) -> np.ndarray:
        grad = np.asarray(gradient, dtype=np.float64).reshape(-1)
        if grad.size != self.nfeatures:
            raise ValueError(f"{self.name} expects {self.nfeatures} gradient coefficients, got {grad.size}")
        if self.inverse_map is None:
            return grad
        full = np.zeros(self.nbackend_features, dtype=np.float64)
        full[self.feature_map] = grad
        return full

    def to_backend_hessian(self, hessian) -> np.ndarray:
        hess = np.asarray(hessian, dtype=np.float64)
        if hess.shape != (self.nfeatures, self.nfeatures):
            raise ValueError(f"{self.name} expects Hessian shape {(self.nfeatures, self.nfeatures)}, got {hess.shape}")
        if self.inverse_map is None:
            return hess
        full = np.zeros((self.nbackend_features, self.nbackend_features), dtype=np.float64)
        idx = np.ix_(self.feature_map, self.feature_map)
        full[idx] = hess
        return full

    def evaluate(
        self,
        features,
        *,
        order: int,
        need_energy: bool,
        context: IFContext | None = None,
        out: MixerResult | None = None,
    ) -> MixerResult:
        return self.mixer.evaluate(features, order=order, need_energy=need_energy, out=out, context=context)

    def eval_energy(self, features, context: IFContext | None = None) -> float:
        return float(self.evaluate(features, order=0, need_energy=True, context=context).energy)

    def eval_feature_derivative(self, features, context: IFContext | None = None) -> np.ndarray:
        return np.asarray(self.evaluate(features, order=1, need_energy=False, context=context).gradient)

    def eval_feature_hessian(self, features, context: IFContext | None = None) -> np.ndarray:
        return np.asarray(self.evaluate(features, order=2, need_energy=False, context=context).hessian)


def _backend_feature_labels(backend) -> tuple[str, ...]:
    labels = getattr(backend, "feature_labels", None)
    if callable(labels):
        return tuple(labels())
    if labels is not None:
        return tuple(labels)
    nfeatures = int(getattr(backend, "nfeatures"))
    return tuple(str(idx) for idx in range(nfeatures))


def build_feature_map(spec: IFModelSpec, backend) -> tuple[np.ndarray, np.ndarray | None]:
    labels = _backend_feature_labels(backend)
    if spec.feature_keys is None:
        return np.arange(len(labels), dtype=np.int64), None

    by_key = {key: idx for idx, key in enumerate(labels)}
    missing = [key for key in spec.feature_keys if key not in by_key]
    if missing:
        missing_text = ", ".join(missing)
        raise KeyError(f"Feature set {spec.feature_set!r} does not define feature key(s): {missing_text}")
    feature_map = np.asarray([by_key[key] for key in spec.feature_keys], dtype=np.int64)
    inverse = np.full(len(labels), -1, dtype=np.int64)
    inverse[feature_map] = np.arange(feature_map.size, dtype=np.int64)
    return feature_map, inverse
