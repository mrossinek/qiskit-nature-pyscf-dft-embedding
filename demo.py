# pylint: disable=invalid-file-header

"""Iterative Embedding based on Qiskit Nature's `properties` framework."""

import logging

from qiskit_nature.logging import logging as nature_logging
from qiskit_nature.settings import settings
from qiskit_nature.algorithms.ground_state_solvers import (
    NumPyMinimumEigensolverFactory,
)
from qiskit_nature.algorithms.ground_state_solvers.iterative_embedding import (
    IterativeEmbedding,
)
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.drivers.second_quantization import MethodType
from qiskit_nature.drivers.second_quantization.pyscfd.embedding_pyscfdriver import (
    EmbeddingPySCFDriver,
)
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit_nature.problems.second_quantization.electronic.embedding_problem import (
    EmbeddingProblem,
)
from qiskit_nature.transformers.second_quantization.electronic import ActiveSpaceTransformer

nature_logging.set_levels_for_names({"qiskit_nature": logging.INFO, "qiskit": logging.INFO})
settings.dict_aux_operators = True


driver = EmbeddingPySCFDriver(
    atom="O 0.0 0.0 0.115; H 0.0 0.754 -0.459; H 0.0 -0.754 -0.459",
    basis="6-31g*",
    method=MethodType.RKS,
    xc_functional="ldaerf + lr_hf",
    xcf_library="xcfun",
    omega=1.0,
)

transformer = ActiveSpaceTransformer(4, 4, indirect_computation=True)

problem = EmbeddingProblem(
    driver,
    [transformer],
    # NOTE: By default, no mixing will be applied to the active density.
    #       Uncomment any of the following to apply the given mixing method.
    # (1) density mixing using recent history averaging
    # density_mixing=lambda history: np.mean(history[-5:]),
    # (2) density mixing using a constant damping parameter `_alpha`
    # density_mixing=lambda history: _alpha * history[-2]
    # + (1.0 - _alpha) * history[-1]
    # if len(history) > 1
    # else history[-1],
)

mapper = ParityMapper()
converter = QubitConverter(mapper, two_qubit_reduction=True)
solver = NumPyMinimumEigensolverFactory(use_default_filter_criterion=True)

app = IterativeEmbedding(converter, solver)
app.max_iter = 10
app.threshold = 1e-4

result = app.solve(problem)
print(result)
