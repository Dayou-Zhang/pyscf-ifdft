from __future__ import annotations

from functools import lru_cache
from importlib import resources

import numpy as np
from scipy.linalg.blas import dgemm
from scipy.special import erf

from .base import IFMixerBase, MixerResult

_INV_SQRT2 = 0.7071067811865476
_INV_SQRT2PI = 0.3989422804014327


def _load_param_archive(path=None) -> dict[str, np.ndarray]:
    if path is None:
        data = resources.files("pyscf.ifdft").joinpath("data/ml25_param.npz")
        with resources.as_file(data) as resolved:
            with np.load(resolved, allow_pickle=False) as loaded:
                return {name: loaded[name] for name in loaded.files}
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


@lru_cache(1)
def _default_params() -> dict[str, np.ndarray]:
    return _load_param_archive()


class ML25Mixer(IFMixerBase):
    def __init__(self, params: dict[str, np.ndarray]):
        self._i0 = np.asarray(params["i0"], dtype=np.float64).reshape(-1)
        self._i1 = np.asarray(params["i1"], dtype=np.float64).reshape(-1)
        self._i2 = np.asarray(params["i2"], dtype=np.float64)
        self._i3 = np.asarray(params["i3"], dtype=np.float64).reshape(-1)
        self._i4 = np.asarray(params["i4"], dtype=np.float64)
        self._hf_coeff = float(self._i0[0])
        self._nfeatures = int(self._i2.shape[1])

    @classmethod
    def from_npz(cls, path=None):
        params = _default_params() if path is None else _load_param_archive(path)
        return cls(params)

    @property
    def hf_coeff(self) -> float:
        return self._hf_coeff

    @property
    def nfeatures(self) -> int:
        return self._nfeatures

    @staticmethod
    def _feature_scale(context) -> float:
        if context is None:
            return 1.0
        nelec = getattr(context, "nelec", 1.0)
        return float(nelec if np.isscalar(nelec) else sum(nelec))

    def _prepare_features(self, features) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape != (1, self._nfeatures):
            raise ValueError(f"ML25 expects one {self._nfeatures}-feature vector, got {arr.shape}")
        return arr

    def evaluate(
        self,
        features,
        *,
        order: int,
        need_energy: bool,
        out: MixerResult | None = None,
        context=None,
    ) -> MixerResult:
        if order < 0 or order > 2:
            raise ValueError(f"Unsupported derivative order {order}")
        result = out if out is not None else MixerResult()
        feat = self._prepare_features(features)
        scale = self._feature_scale(context)
        div = feat / scale
        linear = dgemm(alpha=1.0, a=div, b=self._i2, trans_a=False, trans_b=True) + self._i1
        erf_term = None
        exp_term = None

        if need_energy:
            erf_term = erf(linear * _INV_SQRT2)
            gelu = 0.5 * linear * (1.0 + erf_term)
            linear_1 = dgemm(alpha=1.0, a=gelu, b=self._i4, trans_a=False, trans_b=True) + self._i3
            result.energy = float(np.asarray(linear_1).reshape(-1)[0] * scale)
        else:
            result.energy = None

        if order >= 1:
            if erf_term is None:
                erf_term = erf(linear * _INV_SQRT2)
            exp_term = np.exp(-0.5 * linear * linear)
            gelu_grad = 0.5 * (1.0 + erf_term)
            gelu_grad = gelu_grad + linear * _INV_SQRT2PI * exp_term
            grad = dgemm(alpha=1.0, a=gelu_grad * self._i4, b=self._i2, trans_a=False, trans_b=False)
            result.gradient = np.asarray(grad, dtype=np.float64).reshape(-1)
        else:
            result.gradient = None

        if order >= 2:
            if exp_term is None:
                exp_term = np.exp(-0.5 * linear * linear)
            gelu_hess = (2.0 - linear * linear) * _INV_SQRT2PI * exp_term
            coeff = self._i4.reshape(-1) * gelu_hess.reshape(-1)
            weighted = coeff[:, None] * self._i2
            hess = dgemm(alpha=1.0, a=self._i2, b=weighted, trans_a=True, trans_b=False)
            result.hessian = np.asarray(hess / scale, dtype=np.float64)
        else:
            result.hessian = None

        return result
