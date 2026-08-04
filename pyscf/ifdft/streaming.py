from __future__ import annotations

import numpy as np
from pyscf.dft import numint


class IFStreamingNumInt:
    """Stream semilocal integral features for fixed-density evaluation."""

    def __init__(self, ni, mol, grids, spin: int, *, max_memory=2000):
        if spin not in (0, 1):
            raise ValueError(f"Unsupported spin {spin}")
        self.ni = ni
        self.mol = mol
        self.grids = grids
        self.spin = int(spin)
        self.max_memory = max_memory
        self.nao = mol.nao_nr()

    def integrate_features(self, dm, *, xctype: str = "MGGA"):
        if self.spin == 0:
            return self._integrate_rks_features(dm, xctype=xctype)
        return self._integrate_uks_features(dm, xctype=xctype)

    def _integrate_rks_features(self, dm, *, xctype: str):
        make_rho, _, _ = self.ni._gen_rho_evaluator(
            self.mol, dm, 1, False, self.grids
        )
        feat_e = np.zeros(self.ni.ifxc.nfeatures, dtype=np.float64)
        feature_tmp = np.empty(self.ni.ifxc.nfeatures, dtype=np.float64)
        nelec = 0.0
        for ao, mask, weight, _ in self.ni.block_loop(
            self.mol,
            self.grids,
            self.nao,
            1,
            max_memory=self.max_memory,
        ):
            rho = make_rho(0, ao, mask, xctype)
            nelec += np.dot(weight, rho[0])
            feat_e += self.ni.ifxc.eval_integral_features(
                rho, weight, spin=0, out=feature_tmp
            )
        return float(nelec), feat_e

    def _integrate_uks_features(self, dms, *, xctype: str):
        dma, dmb = numint._format_uks_dm(dms)
        make_rhoa, _, _ = self.ni._gen_rho_evaluator(
            self.mol, dma, 1, False, self.grids
        )
        make_rhob = self.ni._gen_rho_evaluator(
            self.mol, dmb, 1, False, self.grids
        )[0]
        feat_e = np.zeros(self.ni.ifxc.nfeatures, dtype=np.float64)
        feature_tmp = np.empty(self.ni.ifxc.nfeatures, dtype=np.float64)
        nelec = np.zeros(2, dtype=np.float64)
        for ao, mask, weight, _ in self.ni.block_loop(
            self.mol,
            self.grids,
            self.nao,
            1,
            max_memory=self.max_memory,
        ):
            rho_a = make_rhoa(0, ao, mask, xctype)
            rho_b = make_rhob(0, ao, mask, xctype)
            nelec[0] += np.dot(weight, rho_a[0])
            nelec[1] += np.dot(weight, rho_b[0])
            feat_e += self.ni.ifxc.eval_integral_features(
                (rho_a, rho_b), weight, spin=1, out=feature_tmp
            )
        return nelec, feat_e
