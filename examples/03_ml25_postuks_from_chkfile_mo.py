import sys

from pyscf import ifdft, lib


if len(sys.argv) < 2:
    raise SystemExit(f"Usage: {sys.argv[0]} /path/to/chkfile [more.chk ...]")

for chkfile in sys.argv[1:]:
    mol = lib.chkfile.load_mol(chkfile)
    mo_coeff = lib.chkfile.load(chkfile, "scf/mo_coeff")
    mo_occ = lib.chkfile.load(chkfile, "scf/mo_occ")
    use_x2c = bool(lib.chkfile.load(chkfile, "scf/x2c"))

    post = ifdft.IFPostUKS(mol, xc="ML25")
    if use_x2c:
        post = post.x2c()
    post.grids.atom_grid = (99, 590)
    dm = post.make_rdm1(mo_coeff, mo_occ)
    print(chkfile, post.kernel(dm=dm).e_tot)
