from pyscf import dft, gto, ifdft


mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=4)
density_mf = dft.RKS(mol, xc="PBE")
density_mf.grids.atom_grid = (99, 590)
density_mf.run()

result = ifdft.IFPostRKS(
    density_mf, xc="ML26", grids=density_mf.grids
).kernel()
print(result.e_tot)
print(result.features[-1])  # Hartree–Fock exchange energy input feature.
