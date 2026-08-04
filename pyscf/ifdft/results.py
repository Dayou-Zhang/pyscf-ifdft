from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IFDFTEnergyResult:
    functional: str
    density_xc: str | None
    e_tot: float
    e_classical: float
    e_ifxc: float
    e_hf_exchange: float
    features: np.ndarray
    feature_labels: tuple[str, ...]
    nelec: float
    converged_density: bool | None
    density_mf: object | None
