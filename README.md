# Qiskit Nature + PySCF DFT Embedding

This repository contains the latest prototype implementation of the Qiskit Nature + PySCF DFT Embedding.

With Qiskit Nature 0.5, this implementation has become almost trivial. There are still some aspects to be improved:
- when I implement [this feature](https://github.com/Qiskit/qiskit-nature/issues/847), I will make the transformer more
  configurable which will allow us to remove
  [this monkey patching](https://github.ibm.com/oss-zurich/pyscf-dft-embedding/blob/73a8d20/dft_embedding_solver.py#L94)
- the current prototype still accesses a lot of private runtime attributes:
    - for the `ActiveSpaceTransformer`, this situation will also improve when fixing the previous bullet point
    - for the `PySCFDriver` I do not think of this as a problem, since this prototype relies on specifically that driver
      anyways

Speaking of driver-dependence, this is currently the case in two ways:
- the way of obtaining the range-separated problem instance
- the implementation of the `evaluate_energy` method

If one were to try to lift these limitations, the interface would become considerably more complex and more strain would
be placed on the user, as they need to understand to:
- run an initial reference calculation and *not* change that
- create a constant AO-to-MO `BasisTransformer` instance
- also generate the range-separated problem instance
- implement a custom `evaluate_energy` method
If a classical code (like e.g. CP2K) can take of the above on its side, that simplifies things considerably. I do not
think PySCF has intentions to do this, though.

In conclusion, once the `ActiveSpaceTransformer` missing feature is implemented, we should be able to publish this as a
standalone prototype. It should *not* go into Qiskit Nature directly.
I also do *not* want to make this part of the new Qiskit Nature + PySCF integration plugin, because that plugin should
completely avoid the usage of our drivers. Using the `QiskitSolver` available in that plugin is not possible, since it
assumes being used by PySCF's `mcscf` routines, which do not implement this DFT embedding approach.
We could envision creating the `ElectronicStructureProblem` creation routines which I am leveraging from the driver to
then change the interface here to take PySCF objects directly as input. Then one may revisit the integration of this
algorithm into that repository. But this is an idea for the future.
