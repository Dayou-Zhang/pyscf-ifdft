import numpy as np
from pyscf.dft import numint

from .grids import IFGridManager
from .streaming import IFStreamingNumInt


def _if_xctype(if_model) -> str:
    return str(getattr(if_model, "xctype", "MGGA")).upper()


class IntegralFeatureNumInt(numint.NumInt):
    """Numerical integration required for fixed-density IF evaluation."""

    def __init__(self, if_model=None):
        super().__init__()
        self.if_model = if_model
        self.feature_labels = tuple(getattr(if_model, "feature_labels", ()))
        self.ifxc = self._make_ifxc_evaluator(if_model)
        self.last_feat_e = None
        self.last_feature_labels = None

    @staticmethod
    def _make_ifxc_evaluator(functional):
        if functional is None:
            return None
        backend = getattr(functional, "backend", None)
        evaluator_factory = getattr(backend, "new_evaluator", None)
        if callable(evaluator_factory):
            return evaluator_factory()
        from .ifxc import IFXCEvaluator

        return IFXCEvaluator(getattr(functional, "feature_set", ""))

    def _use_functional(self) -> bool:
        if self.if_model is None:
            return False
        if self.ifxc is None:
            raise NotImplementedError(
                "Unsupported integral-feature backend "
                f"{getattr(self.if_model, 'feature_set', None)!r}"
            )
        return True

    @property
    def feature_count(self) -> int:
        if self._use_functional():
            return len(tuple(getattr(self.if_model, "feature_labels", ())))
        return 0

    def integrate_features(
        self,
        mol,
        grids,
        dm,
        *,
        spin: int,
        max_memory=2000,
        method=None,
        nlcgrids=None,
    ):
        if not self._use_functional():
            raise ValueError("No integral-feature functional configured")
        xctype = _if_xctype(self.if_model)
        grids = IFGridManager(mol, grids).ensure_built(with_non0tab=True)
        integral_evaluator = getattr(self.ifxc, "integrate_features", None)
        if callable(integral_evaluator):
            evaluator_options = {}
            if bool(getattr(self.ifxc, "requires_nlc_grid", False)):
                if nlcgrids is None:
                    nlcgrids = grids
                nlcgrids = IFGridManager(
                    mol, nlcgrids
                ).ensure_built(with_non0tab=True)
                evaluator_options["nlcgrids"] = nlcgrids
            nelec, feat_e = integral_evaluator(
                self,
                mol,
                grids,
                dm,
                spin=spin,
                max_memory=max_memory,
                xctype=xctype,
                method=method,
                **evaluator_options,
            )
        else:
            stream = IFStreamingNumInt(
                self, mol, grids, spin, max_memory=max_memory
            )
            nelec, feat_e = stream.integrate_features(dm, xctype=xctype)
        feat_e = np.asarray(self.if_model.select_features(feat_e), dtype=float)
        self.last_feat_e = feat_e.reshape(1, -1)
        self.last_feature_labels = tuple(
            getattr(self.if_model, "feature_labels", ())
        )
        return feat_e, nelec
