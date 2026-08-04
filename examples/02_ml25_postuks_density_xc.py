from pyscf import dft, gto, ifdft


mol = gto.M(
    atom="O 0 0 0; H 0 0 0.958; H 0 0.926 -0.239",
    basis="sto-3g",
    verbose=4,
)
grids = dft.Grids(mol)
grids.atom_grid = (99, 590)

result = ifdft.IFPostUKS(
    mol, xc="ML25", density_xc="MN15", grids=grids
).kernel()
print(result.e_tot)
