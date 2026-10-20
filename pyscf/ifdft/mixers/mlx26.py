from __future__ import annotations

from functools import lru_cache
from importlib import resources

import numpy as np

from .ml26 import ML26Mixer


_PARAMETER_SHAPES = {
    "input_weight": (5000, 78),
    "input_bias": (5000,),
    "output_weight": (1, 5000),
    "output_bias": (1,),
    "hf_coeff": (1,),
}


def _load_param_archive(path=None) -> dict[str, np.ndarray]:
    if path is None:
        data = resources.files("pyscf.ifdft").joinpath("data/mlx26_param.npz")
        with resources.as_file(data) as resolved:
            with np.load(resolved, allow_pickle=False) as loaded:
                return {name: loaded[name] for name in loaded.files}
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


@lru_cache(1)
def _default_params() -> dict[str, np.ndarray]:
    return _load_param_archive()


class MLX26Mixer(ML26Mixer):
    """Electron-normalized MLX26 network, excluding external full HFX."""

    def __init__(self, params: dict[str, np.ndarray]):
        if set(params) != set(_PARAMETER_SHAPES):
            raise ValueError(
                "MLX26 parameter keys must be "
                + ", ".join(_PARAMETER_SHAPES)
            )
        for name, shape in _PARAMETER_SHAPES.items():
            array = np.asarray(params[name])
            if array.shape != shape:
                raise ValueError(
                    f"MLX26 {name} shape must be {shape}, got {array.shape}"
                )
            if array.dtype != np.dtype(np.float64):
                raise ValueError(f"MLX26 {name} must have float64 precision")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"MLX26 {name} must contain only finite values")
        if params["hf_coeff"][0] != 1.0:
            raise ValueError("MLX26 requires fixed hf_coeff = 1.0")
        super().__init__(params)
        self._hf_coeff = 1.0

    @classmethod
    def from_npz(cls, path=None):
        params = _default_params() if path is None else _load_param_archive(path)
        return cls(params)

    @property
    def hf_coeff(self) -> float:
        return self._hf_coeff


__all__ = ["MLX26Mixer"]
