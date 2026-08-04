from __future__ import annotations

import numpy as np

from .base import IFMixerBase, MixerResult


class LinearMixer(IFMixerBase):
    def __init__(self, coefficients, intercept: float = 0.0):
        self.coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        self.intercept = float(intercept)
        self._zero_hessian = np.zeros((self.coefficients.size, self.coefficients.size), dtype=np.float64)

    def _prepare_features(self, features) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float64).reshape(-1)
        if arr.size != self.coefficients.size:
            raise ValueError(f"LinearMixer expects {self.coefficients.size} features, got {arr.size}")
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
        result.energy = float(self.intercept + np.dot(self.coefficients, feat)) if need_energy else None
        result.gradient = self.coefficients.copy() if order >= 1 else None
        result.hessian = self._zero_hessian if order >= 2 else None
        return result
