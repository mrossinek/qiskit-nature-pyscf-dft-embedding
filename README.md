# Qiskit Nature + PySCF DFT Embedding

This repository contains the latest prototype implementation of the Qiskit Nature + PySCF DFT Embedding.

With Qiskit Nature 0.6, this implementation has become almost trivial.
One could still consider further refactoring this class to remove the dependency on the `PySCFDriver` in favor of a more
plugin-like approach similar to how https://github.com/qiskit-community/qiskit-nature-pyscf works.

## Installation

You can simply install the contents of this repository after cloning it:
```
pip install .
```

## Usage

The file `demo.py` shows an example of how to use this embedding solver.
After installing, you can run it as:
```
python demo.py
```

## Testing

You can also run the unittests.
For this you need to ensure that you have `ddt` installed: `pip install ddt`.
Afterwards you are able to run the test suite as follows:
```
python -m unittest discover tests
```

## Citing

When using this software, please cite the corresponding paper:

> Max Rossmannek, Panagiotis Kl. Barkoutsos, Pauline J. Ollitrault, Ivano Tavernelli;
> Quantum HF/DFT-embedding algorithms for electronic structure calculations: Scaling up to complex molecular systems.
> J. Chem. Phys. 21 March 2021; 154 (11): 114105.
>
> https://doi.org/10.1063/5.0029536

You should also cite [Qiskit](https://github.com/Qiskit/qiskit-terra),
[Qiskit Nature](https://github.com/Qiskit/qiskit-nature) and [PySCF](https://pyscf.org/) as per the citation
instructions provided by each of these software packages.
