from copy import deepcopy
from functools import partial
import numpy as np
from qiskit_nature.second_q.algorithms import (
    GroundStateEigensolver,
    NumPyMinimumEigensolverFactory,
)
from qiskit_nature.second_q.drivers import PySCFDriver, MethodType
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import ParityMapper, QubitConverter
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.problems import ElectronicBasis
from qiskit_nature.second_q.properties import ElectronicDensity
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer, BasisTransformer


def compute_inactive_fock(
    active_space_transformer,
    hamiltonian,
    density_active,
    total_fock_ref,
    e_inactive_ref,
):
    if density_active is None:
        density_active = active_space_transformer._density_active

    active_fock = (
        hamiltonian.fock(density_active) - hamiltonian.electronic_integrals.one_body
    )

    inactive_fock = total_fock_ref - active_fock

    e_inactive = -1.0 * ElectronicIntegrals.einsum(
        {"ij,ji": ("+-", "+-", "")}, total_fock_ref, density_active
    )
    e_inactive += 0.5 * ElectronicIntegrals.einsum(
        {"ij,ji": ("+-", "+-", "")}, active_fock, density_active
    )
    e_inactive_sum = e_inactive_ref + sum(
        e_inactive[key].get("", 0.0) for key in e_inactive
    )

    new_hamil = ElectronicEnergy(
        active_space_transformer._transform_active.transform_electronic_integrals(
            inactive_fock + hamiltonian.electronic_integrals.two_body
        )
    )
    new_hamil.constants = deepcopy(hamiltonian.constants)
    new_hamil.constants[active_space_transformer.__class__.__name__] = e_inactive_sum

    return new_hamil


def active_density(total_density, active_space):
    subspace = active_space._transform_active.transform_electronic_integrals(
        total_density
    )
    subspace = active_space._transform_active.invert().transform_electronic_integrals(
        subspace
    )
    return subspace


def merge_densities(problem, active_space, res):
    # get total MO density
    total = problem.properties.electronic_density
    # find total MO density component which overlaps with active space
    subspace = active_density(total, active_space)
    # subtract that component from the total
    total -= subspace
    # get active space component
    active = res.electronic_density
    # expand active component to total system size
    superspace = deepcopy(active)
    superspace = active_space._transform_active.invert().transform_electronic_integrals(
        superspace
    )
    # add expanded active space component to the total
    total += superspace

    return total


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
    # run reference calculation
    driver.run_pyscf()

    # build AO-to-MO transformer based on fixed coeffs from reference calculation
    basis_trafo = BasisTransformer(
        ElectronicBasis.AO,
        ElectronicBasis.MO,
        ElectronicIntegrals.from_raw_integrals(driver._calc.mo_coeff),
    )

    # generate problem with range-separation enabled in 2-body terms
    with driver._mol.with_range_coulomb(omega=omega):
        problem = driver.to_problem(basis=ElectronicBasis.MO, include_dipole=False)
        problem.properties.electronic_density = (
            ElectronicDensity.from_orbital_occupation(
                problem.orbital_occupations,
                problem.orbital_occupations_b,
                include_rdm2=False,
            )
        )

    # some reference values
    total_fock_ref = basis_trafo.transform_electronic_integrals(
        ElectronicIntegrals.from_raw_integrals(driver._calc.get_fock())
    )
    e_nuc = problem.hamiltonian.nuclear_repulsion_energy
    e_tot = driver._calc.e_tot
    density_active = None

    # specify active space
    active_space = ActiveSpaceTransformer(4, 4)

    # setup solver
    converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
    solver = NumPyMinimumEigensolverFactory(use_default_filter_criterion=True)
    algo = GroundStateEigensolver(converter, solver)

    max_iter = 10
    threshold = 1e-4
    e_prev = float("NaN")
    e_next = float("NaN")
    converged = False
    n_iter = 0

    while n_iter < max_iter:
        n_iter += 1

        # overwrite inactive Fock computation for active space
        active_space._transform_electronic_energy = partial(
            compute_inactive_fock,
            active_space,
            density_active=density_active,
            total_fock_ref=total_fock_ref,
            e_inactive_ref=e_tot - e_nuc,
        )

        # reduce problem to active space
        as_problem = active_space.transform(problem)

        # solve active space problem
        res = algo.solve(as_problem)
        print(res)

        e_prev = e_next
        e_next = res.total_energies[0]

        converged = np.abs(e_prev - e_next) < threshold
        if converged:
            break

        # merge active space density into total MO density
        total = merge_densities(problem, active_space, res)

        # evaluate energy at new density
        ao_density = basis_trafo.invert().transform_electronic_integrals(total)
        rho = ao_density.trace_spin()["+-"]
        rho[np.abs(rho) < 1e-8] = 0.0
        e_tot = driver._calc.energy_tot(dm=rho)
        density_active = active_density(total, active_space)


if __name__ == "__main__":
    _main()
