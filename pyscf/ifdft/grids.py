from __future__ import annotations

import copy

import numpy as np


DEFAULT_ATOM_GRID = (99, 590)


class IFGridManager:
    def __init__(self, mol, grids):
        self.mol = mol
        self.grids = grids

    def is_built(self) -> bool:
        coords = getattr(self.grids, "coords", None)
        weights = getattr(self.grids, "weights", None)
        if coords is None or weights is None:
            return False
        coords = np.asarray(coords)
        weights = np.asarray(weights)
        return coords.ndim == 2 and coords.shape[1] == 3 and weights.ndim == 1 and coords.shape[0] == weights.size

    def ensure_built(self, *, with_non0tab=True):
        if getattr(self.grids, "mol", None) is not self.mol:
            self.grids.mol = self.mol
        if not self.is_built():
            self.grids.build(with_non0tab=with_non0tab)
        elif with_non0tab and getattr(self.grids, "non0tab", None) is None:
            self.grids.non0tab = self.grids.make_mask(self.mol, self.grids.coords)
            self.grids.screen_index = self.grids.non0tab
        self.freeze()
        return self.grids

    def freeze(self):
        if self.is_built():
            self.grids.built = True
            self.grids._ifdft_static = True
            self.grids._ifdft_signature = self.signature()
        return self.grids

    def signature(self):
        coords = getattr(self.grids, "coords", None)
        weights = getattr(self.grids, "weights", None)
        if coords is None or weights is None:
            return (id(self.grids), None, None, False)
        weights = np.asarray(weights)
        return (
            id(self.grids),
            int(weights.size),
            tuple(np.asarray(coords).shape),
            getattr(self.grids, "level", None),
            bool(getattr(self.grids, "non0tab", None) is not None),
        )


def grids_are_built(grids) -> bool:
    if grids is None:
        return False
    return IFGridManager(getattr(grids, "mol", None), grids).is_built()


def grids_are_equivalent(left, right) -> bool:
    """Return whether two built grids carry exactly the same quadrature."""
    if left is right:
        return True
    if not grids_are_built(left) or not grids_are_built(right):
        return False
    left_coords = np.asarray(left.coords)
    right_coords = np.asarray(right.coords)
    left_weights = np.asarray(left.weights)
    right_weights = np.asarray(right.weights)
    if (
        left_coords.shape != right_coords.shape
        or left_weights.shape != right_weights.shape
    ):
        return False
    if left_coords is right_coords and left_weights is right_weights:
        return True
    return bool(
        np.array_equal(left_coords, right_coords)
        and np.array_equal(left_weights, right_weights)
    )


def copy_grids(grids):
    if grids is None:
        return None
    new_grids = copy.copy(grids)
    for name in ("coords", "weights", "non0tab", "screen_index", "atm_idx", "quadrature_weights"):
        if hasattr(grids, name):
            setattr(new_grids, name, getattr(grids, name))
    if grids_are_built(new_grids):
        new_grids.built = True
    return new_grids
