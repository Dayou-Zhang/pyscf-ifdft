from pyscf import dft, gto, ifdft


mol = gto.M(
    atom="O 0 0 0; H 0 0 0.958; H 0 0.926 -0.239",
    basis="sto-3g",
    verbose=4,
)
density_mf = dft.UKS(mol, xc="MN15")
density_mf.grids.atom_grid = (99, 590)
density_mf.run()

result = ifdft.IFPostUKS(
    density_mf, xc="ML25", grids=density_mf.grids
).kernel()
print(result.e_tot)
