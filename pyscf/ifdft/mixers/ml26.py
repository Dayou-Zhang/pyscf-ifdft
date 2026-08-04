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
        data = resources.files("pyscf.ifdft").joinpath("data/ml26_param.npz")
        with resources.as_file(data) as resolved:
            with np.load(resolved, allow_pickle=False) as loaded:
                return {name: loaded[name] for name in loaded.files}
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


@lru_cache(1)
def _default_params() -> dict[str, np.ndarray]:
    return _load_param_archive()


class ML26Mixer(IFMixerBase):
    """Energy and derivatives of the electron-normalized ML26 network."""

    def __init__(self, params: dict[str, np.ndarray]):
        self._input_weight = np.asarray(params["input_weight"], dtype=np.float64)
        self._input_bias = np.asarray(
            params["input_bias"], dtype=np.float64
        ).reshape(-1)
        self._output_weight = np.asarray(
            params["output_weight"], dtype=np.float64
        )
        self._output_bias = float(
            np.asarray(params["output_bias"], dtype=np.float64).reshape(-1)[0]
        )

        if self._input_weight.ndim != 2:
            raise ValueError("ML26 input weights must be a matrix")
        nhidden, nfeatures = self._input_weight.shape
        if self._input_bias.shape != (nhidden,):
            raise ValueError("ML26 input bias shape does not match its weights")
        if self._output_weight.shape != (1, nhidden):
            raise ValueError("ML26 output weight shape does not match its hidden layer")
        self._nfeatures = int(nfeatures)

    @classmethod
    def from_npz(cls, path=None):
        params = _default_params() if path is None else _load_param_archive(path)
        return cls(params)

    @property
    def nfeatures(self) -> int:
        return self._nfeatures

    @staticmethod
    def _feature_scale(context) -> float:
        if context is None:
            return 1.0
        nelec = getattr(context, "nelec", 1.0)
        scale = float(nelec if np.isscalar(nelec) else sum(nelec))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"ML26 requires a positive finite electron count, got {scale}"
            )
        return scale

    def _prepare_features(self, features) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape != (1, self._nfeatures):
            raise ValueError(
                f"ML26 expects one {self._nfeatures}-feature vector, got {arr.shape}"
            )
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
        normalized = feat / scale
        linear = dgemm(
            alpha=1.0,
            a=normalized,
            b=self._input_weight,
            trans_a=False,
            trans_b=True,
        )
        linear += self._input_bias
        erf_term = erf(linear * _INV_SQRT2)
        if need_energy:
            gelu = 0.5 * linear * (1.0 + erf_term)
            correction = dgemm(
                alpha=1.0,
                a=gelu,
                b=self._output_weight,
                trans_a=False,
                trans_b=True,
            )
            result.energy = float(
                (correction.reshape(-1)[0] + self._output_bias) * scale
            )
        else:
            result.energy = None
        if order >= 1:
            gelu_grad = 0.5 * (1.0 + erf_term)
            exp_term = np.exp(-0.5 * linear * linear)
            gelu_grad += linear * _INV_SQRT2PI * exp_term
            weighted = gelu_grad * self._output_weight
            gradient = dgemm(
                alpha=1.0,
                a=weighted,
                b=self._input_weight,
                trans_a=False,
                trans_b=False,
            )
            # The outer electron-count factor and the normalization of the
            # inputs cancel exactly in dE/dF.
            result.gradient = np.asarray(
                gradient, dtype=np.float64
            ).reshape(-1)
        else:
            result.gradient = None
        if order >= 2:
            gelu_hess = (
                (2.0 - linear * linear) * _INV_SQRT2PI * exp_term
            ).reshape(-1)
            coefficients = self._output_weight.reshape(-1) * gelu_hess
            weighted_input = coefficients[:, None] * self._input_weight
            hessian = dgemm(
                alpha=1.0,
                a=self._input_weight,
                b=weighted_input,
                trans_a=True,
                trans_b=False,
            )
            result.hessian = np.asarray(
                hessian / scale, dtype=np.float64
            )
        else:
            result.hessian = None
        return result


__all__ = ["ML26Mixer"]
