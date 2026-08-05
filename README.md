# pyscf-ifdft

`pyscf-ifdft` is a [PySCF](https://pyscf.org) extension for evaluating
integral-feature density functionals on PySCF densities. It is imported as
`pyscf.ifdft` and provides the ML25 and ML26 functionals through `IFPostRKS`
and `IFPostUKS`.

ML25 and ML26 are post-SCF methods evaluated on RKS or UKS reference densities
with `IFPostRKS` and `IFPostUKS`. Integral-feature DFT is not a Kohn–Sham
theory.

## Installation

```bash
python -m pip install pyscf-ifdft
```

Binary wheels include the compiled backend. Source builds obtain
[`libifxc`](https://github.com/Dayou-Zhang/libifxc) automatically.

## Example

The following evaluates ML25 on a converged MN15 unrestricted density:

```python
from pyscf import dft, gto, ifdft

mol = gto.M(
    atom="O 0 0 0; H 0 0 0.958; H 0 0.926 -0.239",
    basis="sto-3g",
)
mf = dft.UKS(mol, xc="MN15").run()

result = ifdft.IFPostUKS(mf, xc="ML25").kernel()
print(result.e_tot)
```

`IFPostRKS` provides the corresponding restricted-density interface.
`kernel()` returns an `IFDFTEnergyResult` containing the total energy,
integral features, and energy components.

See the [`examples`](examples) directory for density generation with
`density_xc`, checkpoint input, HDF5 feature export, ML26, and X2C calculations.

## References

1. Zhang, D.; Shu, Y.; Truhlar, D. G. Reinventing Density Functional Theory
   with Machine Learning on Integral Features. *J. Chem. Theory Comput.*
   **2026**, *22*, 6295–6303.
   [https://doi.org/10.1021/acs.jctc.6c00999](https://doi.org/10.1021/acs.jctc.6c00999).
2. Zhang, D.; Shu, Y.; Truhlar, D. G. Manuscript on ML26 in preparation.

## License

`pyscf-ifdft` is licensed under Apache-2.0. The separately linked `libifxc`
library is licensed under MPL-2.0. See [`LICENSE`](LICENSE), [`NOTICE`](NOTICE),
and [`LICENSES/MPL-2.0.txt`](LICENSES/MPL-2.0.txt).

## Recommended citation

D. Zhang, Y. Shu, and D. G. Truhlar, *pyscf-ifdft*, version 0.1.0,
University of Minnesota, Minneapolis, 2026.
[https://github.com/Dayou-Zhang/pyscf-ifdft](https://github.com/Dayou-Zhang/pyscf-ifdft).
