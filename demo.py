import numpy as np
from qiskit_nature.second_q.algorithms import (GroundStateEigensolver,
                                               NumPyMinimumEigensolverFactory)
from qiskit_nature.second_q.drivers import MethodType, PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, QubitConverter
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

from dft_embedding_solver import DFTEmbeddingSolver


def _main():
    omega = 1.0

    # setup driver
    driver = PySCFDriver(
        atom="O 0.0 0.0 0.115; H 0.0 0.754 -0.459; H 0.0 -0.754 -0.459",
        basis="6-31g*",
        method=MethodType.RKS,
        xc_functional=f"ldaerf + lr_hf({omega})",
        xcf_library="xcfun",
    )

    # specify active space
    active_space = ActiveSpaceTransformer(4, 4)

    # setup solver
    converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
    solver = NumPyMinimumEigensolverFactory(use_default_filter_criterion=True)
    algo = GroundStateEigensolver(converter, solver)

    dft_solver = DFTEmbeddingSolver(active_space, algo)

    # NOTE: By default, no mixing will be applied to the active density.
    # Uncomment any of the following to apply the given mixing method.
    # (1) density mixing using the last `history_length` number of densities
    # history_length = 10
    # dft_solver.damp_density = lambda history: np.mean(history[-history_length:])
    # (2) density mixing using a constant damping parameter `_alpha`
    # alpha = 0.5
    # dft_solver.damp_density = (
    #     lambda history: alpha * history[-2] + (1.0 - alpha) * history[-1]
    #     if len(history) > 1
    #     else history[-1]
    # )

    result = dft_solver.solve(driver, omega)
    print(result)


if __name__ == "__main__":
    _main()
