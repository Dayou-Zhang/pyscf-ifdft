from pathlib import Path

import h5py
from pyscf import dft, gto, ifdft


mol = gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1, verbose=4)
density_mf = dft.UKS(mol, xc="PBE")
density_mf.grids.atom_grid = (99, 590)
density_mf.run()

feature_file = Path("ml25_postuks_features.h5")
result = ifdft.IFPostUKS(
    density_mf,
    xc="ML25",
    grids=density_mf.grids,
    feature_file=feature_file,
).kernel()

with h5py.File(feature_file, "r") as fh5:
    features = fh5["ml25_features"][()]
    labels = fh5["ml25_features"].attrs["feature_labels"].astype(str)

print(f"Saved {len(features)} integral features to {feature_file}")
print(f"First feature: {labels[0]} = {features[0]}")
print(f"Total IF energy: {result.e_tot}")
