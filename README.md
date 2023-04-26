# Qiskit Nature + PySCF DFT Embedding

This repository contains the latest prototype implementation of the Qiskit Nature + PySCF DFT Embedding.

With Qiskit Nature 0.6, this implementation has become almost trivial.
I would still like to refactor this further by removing the dependency on the
`PySCFDriver` in favor of a more plugin-like approach similar to how
https://github.com/qiskit-community/qiskit-nature-pyscf works.
