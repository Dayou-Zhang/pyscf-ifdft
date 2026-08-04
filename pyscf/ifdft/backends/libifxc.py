from functools import lru_cache

from .. import _libifxc
from ..features import FeatureInfo, FeatureSetInfo

_ALIASES = {
    "mgga_xc_ml25": "ml25",
    "ml25": "ml25",
    "mgga_xc_ml26": "ml26",
    "ml26": "ml26",
}

_NAMES = {
    "ml25": "ML25 integral features",
    "ml26": "ML26 semilocal integral features",
}


def canonical_feature_set_key(key: str) -> str:
    text = str(key).strip().lower()
    canonical = _ALIASES.get(text, text)
    if canonical not in _libifxc._feature_set_keys():
        # Let the backend raise its native error message for unknown keys.
        _libifxc._feature_count(canonical)
    return canonical


def available_feature_sets() -> tuple[str, ...]:
    return _libifxc._feature_set_keys()


@lru_cache(None)
def describe_feature_set(key: str) -> FeatureSetInfo:
    canonical = canonical_feature_set_key(key)
    labels = _libifxc._feature_keys(canonical)
    return FeatureSetInfo(
        key=canonical,
        name=_NAMES.get(canonical, canonical),
        nfeatures=_libifxc._feature_count(canonical),
        max_deriv_order=_libifxc._feature_set_max_deriv_order(canonical),
        xctype=_libifxc._feature_set_xctype(canonical),
        labels=labels,
    )


def describe_feature(key: str, index: int) -> FeatureInfo:
    info = describe_feature_set(key)
    return FeatureInfo(index=int(index), key=info.labels[int(index)])


class IFXCFeatureSet:
    def __init__(self, key: str):
        self.key = canonical_feature_set_key(key)
        self.info = describe_feature_set(self.key)
        self._handle = _libifxc._create_evaluator(self.key)
        self._closed = False

    def close(self):
        if not self._closed:
            _libifxc._destroy_evaluator(self._handle)
            self._handle = None
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("IFXCFeatureSet is closed")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def nfeatures(self) -> int:
        return self.info.nfeatures

    def feature_labels(self) -> tuple[str, ...]:
        return self.info.labels

    def eval_block(self, rho, sigma, tau, weights=None, *, spin: int, **flags):
        if self._closed:
            raise RuntimeError("IFXCFeatureSet is closed")
        return _libifxc._eval_evaluator_block(
            self._handle,
            self.info.nfeatures,
            rho,
            sigma,
            tau,
            weights,
            spin=spin,
            **flags,
        )

    def eval_weighted_vvars(self, rho, sigma, tau, rho_gradient, weights, feature_gradient, *, spin: int, out=None):
        if self._closed:
            raise RuntimeError("IFXCFeatureSet is closed")
        return _libifxc._eval_weighted_vvars(
            self._handle,
            self.info.nfeatures,
            rho,
            sigma,
            tau,
            rho_gradient,
            weights,
            feature_gradient,
            spin=spin,
            out=out,
        )

    def eval_weighted_vvars_from_pyscf_mgga(self, rho, weights, feature_gradient, *, spin: int, out=None):
        if self._closed:
            raise RuntimeError("IFXCFeatureSet is closed")
        return _libifxc._eval_weighted_vvars_from_pyscf_mgga(
            self._handle,
            self.info.nfeatures,
            rho,
            weights,
            feature_gradient,
            spin=spin,
            out=out,
        )

    def eval_weighted_vvars_batch_from_pyscf_mgga(
        self, rho, weights, feature_gradients, *, spin: int, out=None
    ):
        if self._closed:
            raise RuntimeError("IFXCFeatureSet is closed")
        return _libifxc._eval_weighted_vvars_batch_from_pyscf_mgga(
            self._handle,
            self.info.nfeatures,
            rho,
            weights,
            feature_gradients,
            spin=spin,
            out=out,
        )

    def eval_features_from_pyscf_mgga(self, rho, weights, *, spin: int, out=None):
        if self._closed:
            raise RuntimeError("IFXCFeatureSet is closed")
        return _libifxc._eval_features_from_pyscf_mgga(
            self._handle,
            self.info.nfeatures,
            rho,
            weights,
            spin=spin,
            out=out,
        )

    def eval_feature_directional_from_pyscf_mgga(self, rho0, rho1, weights, *, spin: int, out=None):
        if self._closed:
            raise RuntimeError("IFXCFeatureSet is closed")
        return _libifxc._eval_feature_directional_from_pyscf_mgga(
            self._handle,
            self.info.nfeatures,
            rho0,
            rho1,
            weights,
            spin=spin,
            out=out,
        )

    def eval_weighted_response_from_pyscf_mgga(self, rho0, rho1, weights, feature_gradient, *, spin: int, out=None):
        if self._closed:
            raise RuntimeError("IFXCFeatureSet is closed")
        return _libifxc._eval_weighted_response_from_pyscf_mgga(
            self._handle,
            self.info.nfeatures,
            rho0,
            rho1,
            weights,
            feature_gradient,
            spin=spin,
            out=out,
        )

    def eval_response_batch_from_pyscf_mgga(
        self,
        rho0,
        rho1_batch,
        weights,
        feature_gradient,
        *,
        spin: int,
        out_kernel=None,
        out_dfeatures=None,
    ):
        if self._closed:
            raise RuntimeError("IFXCFeatureSet is closed")
        return _libifxc._eval_response_batch_from_pyscf_mgga(
            self._handle,
            self.info.nfeatures,
            rho0,
            rho1_batch,
            weights,
            feature_gradient,
            spin=spin,
            out_kernel=out_kernel,
            out_dfeatures=out_dfeatures,
        )

    def eval_feature_directional_derivative(self, rho, sigma, tau, drho, dsigma, dtau, weights, *, spin: int, out=None):
        if self._closed:
            raise RuntimeError("IFXCFeatureSet is closed")
        return _libifxc._eval_feature_directional_derivative(
            self._handle,
            self.info.nfeatures,
            rho,
            sigma,
            tau,
            drho,
            dsigma,
            dtau,
            weights,
            spin=spin,
            out=out,
        )

    def eval_weighted_response_vvars(
        self,
        rho,
        sigma,
        tau,
        drho,
        dsigma,
        dtau,
        rho_gradient,
        drho_gradient,
        weights,
        feature_gradient,
        *,
        spin: int,
        out=None,
    ):
        if self._closed:
            raise RuntimeError("IFXCFeatureSet is closed")
        return _libifxc._eval_weighted_response_vvars(
            self._handle,
            self.info.nfeatures,
            rho,
            sigma,
            tau,
            drho,
            dsigma,
            dtau,
            rho_gradient,
            drho_gradient,
            weights,
            feature_gradient,
            spin=spin,
            out=out,
        )
