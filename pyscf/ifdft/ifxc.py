from __future__ import annotations

import numpy as np

from .backends.libifxc import IFXCFeatureSet


class IFXCEvaluator:
    def __init__(self, feature_set: str = "ml25"):
        self.feature_set = feature_set
        self._backend = IFXCFeatureSet(self.feature_set)

    def close(self):
        self._backend.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def new_evaluator(self):
        return type(self)(self.feature_set)

    @property
    def nfeatures(self) -> int:
        return self._backend.nfeatures

    def feature_labels(self) -> tuple[str, ...]:
        return self._backend.feature_labels()

    @staticmethod
    def _normalize_mgga_rho(rho) -> np.ndarray:
        arr = np.asarray(rho, dtype=np.float64, order="C")
        if arr.ndim != 2:
            raise ValueError(f"Expected a single MGGA rho array, got shape {arr.shape}")
        if arr.shape[0] == 6:
            arr = np.asarray(arr[[0, 1, 2, 3, 5], :], dtype=np.float64, order="C")
        if arr.shape[0] != 5:
            raise ValueError(f"Expected 5 or 6 MGGA rho components, got shape {arr.shape}")
        return arr

    @staticmethod
    def _split_uks_rho(rho):
        if isinstance(rho, tuple) and len(rho) == 2:
            return rho
        arr = np.asarray(rho, dtype=np.float64, order="C")
        if arr.ndim != 3 or arr.shape[0] != 2:
            raise ValueError(f"Expected UKS rho pair, got shape {arr.shape}")
        return arr[0], arr[1]

    @classmethod
    def _semilocal_input(cls, rho, *, spin: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if spin == 0:
            rho_arr = cls._normalize_mgga_rho(rho)
            grad = rho_arr[1:4]
            sigma = np.sum(grad * grad, axis=0).reshape(1, -1)
            return (
                np.asarray(rho_arr[0:1], dtype=np.float64, order="C"),
                np.asarray(sigma, dtype=np.float64, order="C"),
                np.asarray(rho_arr[4:5], dtype=np.float64, order="C"),
            )
        if spin == 1:
            rho_a, rho_b = cls._split_uks_rho(rho)
            rho_a = cls._normalize_mgga_rho(rho_a)
            rho_b = cls._normalize_mgga_rho(rho_b)
            if rho_a.shape[1] != rho_b.shape[1]:
                raise ValueError("Alpha and beta rho arrays must have the same grid size")
            grad_a = rho_a[1:4]
            grad_b = rho_b[1:4]
            sigma = np.empty((3, rho_a.shape[1]), dtype=np.float64)
            sigma[0] = np.sum(grad_a * grad_a, axis=0)
            sigma[1] = np.sum(grad_a * grad_b, axis=0)
            sigma[2] = np.sum(grad_b * grad_b, axis=0)
            return (
                np.asarray([rho_a[0], rho_b[0]], dtype=np.float64, order="C"),
                sigma,
                np.asarray([rho_a[4], rho_b[4]], dtype=np.float64, order="C"),
            )
        raise ValueError(f"Unsupported spin {spin}")

    def eval_integral_features(self, rho, weight, *, spin: int, out=None) -> np.ndarray:
        return np.asarray(
            self._backend.eval_features_from_pyscf_mgga(rho, weight, spin=spin, out=out),
            dtype=np.float64,
        )

    def eval_local_values(self, rho, *, spin: int) -> np.ndarray:
        rho_arr, sigma_arr, tau_arr = self._semilocal_input(rho, spin=spin)
        block = self._backend.eval_block(
            rho_arr,
            sigma_arr,
            tau_arr,
            spin=spin,
            want_local_values=True,
        )
        return np.asarray(block.local_values, dtype=np.float64)

    def _validated_feature_coefficients(self, coefficients) -> np.ndarray:
        coeff = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        if coeff.size != self.nfeatures:
            raise ValueError(f"Expected {self.nfeatures} feature coefficients, got {coeff.size}")
        return coeff

    def eval_weighted_first_derivatives(self, rho, weight, coefficients, *, spin: int, out=None) -> np.ndarray:
        coeff = self._validated_feature_coefficients(coefficients)
        return self._backend.eval_weighted_vvars_from_pyscf_mgga(rho, weight, coeff, spin=spin, out=out)

    def eval_weighted_first_derivatives_batch(self, rho, weight, coefficients, *, spin: int, out=None) -> np.ndarray:
        coeffs = np.asarray(coefficients, dtype=np.float64)
        if coeffs.ndim == 1:
            coeffs = coeffs.reshape(1, -1)
        if coeffs.ndim != 2 or coeffs.shape[1] != self.nfeatures:
            raise ValueError(f"Expected feature coefficient batch with trailing size {self.nfeatures}, got {coeffs.shape}")
        return self._backend.eval_weighted_vvars_batch_from_pyscf_mgga(rho, weight, coeffs, spin=spin, out=out)

    def eval_feature_directional_derivative(self, rho0, rho1, weight, *, spin: int, out=None) -> np.ndarray:
        return self._backend.eval_feature_directional_from_pyscf_mgga(rho0, rho1, weight, spin=spin, out=out)

    def eval_weighted_second_derivative_response(
        self, rho0, rho1, weight, coefficients, *, spin: int, out=None
    ) -> np.ndarray:
        coeff = self._validated_feature_coefficients(coefficients)
        return self._backend.eval_weighted_response_from_pyscf_mgga(
            rho0,
            rho1,
            weight,
            coeff,
            spin=spin,
            out=out,
        )

    def eval_response_batch(
        self,
        rho0,
        rho1_batch,
        weight,
        coefficients,
        *,
        spin: int,
        out_kernel=None,
        out_dfeatures=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        coeff = self._validated_feature_coefficients(coefficients)
        return self._backend.eval_response_batch_from_pyscf_mgga(
            rho0,
            rho1_batch,
            weight,
            coeff,
            spin=spin,
            out_kernel=out_kernel,
            out_dfeatures=out_dfeatures,
        )
