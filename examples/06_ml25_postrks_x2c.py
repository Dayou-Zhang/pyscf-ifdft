from pyscf import dft, gto, ifdft


mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=4)
grids = dft.Grids(mol)
grids.atom_grid = (99, 590)

post = ifdft.IFPostRKS(
    mol, xc="ML25", density_xc="PBE", grids=grids
).x2c()
result = post.kernel()
print(result.e_tot)
