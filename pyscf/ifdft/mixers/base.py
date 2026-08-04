from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class MixerResult:
    energy: float | None = None
    gradient: np.ndarray | None = None
    hessian: np.ndarray | None = None


class IFMixer(Protocol):
    def evaluate(
        self,
        features: np.ndarray,
        *,
        order: int,
        need_energy: bool,
        out: MixerResult | None = None,
        context: object | None = None,
    ) -> MixerResult:
        ...


class IFMixerBase:
    def energy(self, features, context=None):
        return self.evaluate(features, order=0, need_energy=True, context=context).energy

    def energy_gradient(self, features, context=None):
        result = self.evaluate(features, order=1, need_energy=True, context=context)
        return result.energy, result.gradient

    def gradient(self, features, context=None):
        return self.evaluate(features, order=1, need_energy=False, context=context).gradient

    def gradient_hessian(self, features, context=None):
        result = self.evaluate(features, order=2, need_energy=False, context=context)
        return result.gradient, result.hessian

    def energy_gradient_hessian(self, features, context=None):
        result = self.evaluate(features, order=2, need_energy=True, context=context)
        return result.energy, result.gradient, result.hessian
