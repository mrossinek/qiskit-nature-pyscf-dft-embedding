from __future__ import annotations

from copy import deepcopy
from functools import partial

import numpy as np
from qiskit_nature.second_q.algorithms import GroundStateSolver
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.problems import (ElectronicBasis,
                                             ElectronicStructureResult)
from qiskit_nature.second_q.properties import ElectronicDensity
from qiskit_nature.second_q.transformers import (ActiveSpaceTransformer,
                                                 BasisTransformer)


def _extract_from_driver(driver, property_name):
    attribute = getattr(driver._calc, property_name)
    if callable(attribute):
        attribute = attribute()

    if isinstance(attribute, tuple):
        return attribute[0], attribute[1]

    if isinstance(attribute, np.ndarray):
        shape = attribute.shape
        if len(shape) == 3:
            return attribute[0], attribute[1]

        return attribute, None

    return None, None


class DFTEmbeddingSolver:
    """TODO."""

    def __init__(
        self,
        active_space: ActiveSpaceTransformer,
        solver: GroundStateSolver,
        *,
        max_iter: int = 100,
        threshold: float = 1e-5,
    ) -> None:
        """TODO."""
        self.active_space = active_space
        self.solver = solver
        self.max_iter = max_iter
        self.threshold = threshold
        self.active_density_history = []

    def solve(self, driver: PySCFDriver, omega: float) -> ElectronicStructureResult:
        """TODO."""
        # 1. run the reference calculation
        driver.run_pyscf()

        # 2. build the AO-to-MO basis transformer to ensure constant MOs during entire procedure
        mo_coeff, mo_coeff_b = _extract_from_driver(driver, "mo_coeff")
        basis_trafo = BasisTransformer(
            ElectronicBasis.AO,
            ElectronicBasis.MO,
            ElectronicIntegrals.from_raw_integrals(mo_coeff, h1_b=mo_coeff_b),
        )

        # 3. generate the problem with range-separated 2-body terms
        with driver._mol.with_range_coulomb(omega=omega):
            problem = driver.to_problem(basis=ElectronicBasis.MO, include_dipole=False)
            density = ElectronicDensity.from_orbital_occupation(
                problem.orbital_occupations,
                problem.orbital_occupations_b,
                include_rdm2=False,
            )
            self.active_density_history.append(density)
            problem.properties.electronic_density = density

        # 4. initialize the iterative algorithm
        fock_a, fock_b = _extract_from_driver(driver, "get_fock")
        total_fock_ref = basis_trafo.transform_electronic_integrals(
            ElectronicIntegrals.from_raw_integrals(fock_a, h1_b=fock_b)
        )
        e_nuc = problem.hamiltonian.nuclear_repulsion_energy
        e_tot = driver._calc.e_tot
        density_active = None

        e_next = float("NaN")
        e_prev = float("NaN")
        converged = False
        n_iter = 0

        while n_iter < self.max_iter:
            n_iter += 1

            # overwrite inactive Fock computation for active space
            self.active_space._transform_electronic_energy = partial(
                compute_inactive_fock,
                self.active_space,
                density_active=density_active,
                total_fock_ref=total_fock_ref,
                e_inactive_ref=e_tot - e_nuc,
            )

            # reduce problem to active space
            as_problem = self.active_space.transform(problem)

            # solve active space problem
            result = self.solver.solve(as_problem)
            self.active_density_history.append(result.electronic_density)

            e_prev = e_next
            e_next = result.total_energies[0]

            converged = np.abs(e_prev - e_next) < self.threshold
            if converged:
                break

            # merge active space density into total MO density
            total = self.update_total_density(problem, result)

            # evaluate energy at new density
            e_tot, density_active = self.evaluate_energy(total, basis_trafo, driver)

        return result

    def evaluate_energy(self, total_density, basis_trafo, driver):
        """TODO."""
        # convert MO density to AO
        ao_density = basis_trafo.invert().transform_electronic_integrals(total_density)

        # construct density in PySCF required form
        if basis_trafo.coefficients.beta.is_empty():
            rho = ao_density.trace_spin()["+-"]
        else:
            rho = np.asarray([ao_density.alpha["+-"], ao_density.beta["+-"]])

        # chop density
        rho[np.abs(rho) < 1e-8] = 0.0

        # evaluate new energy
        e_tot = driver._calc.energy_tot(dm=rho)

        # reduce to active space density
        density_active = self.get_active_density_component(total_density)

        return e_tot, density_active

    def get_active_density_component(self, total_density):
        """TODO."""
        active_density = (
            self.active_space._transform_active.transform_electronic_integrals(
                total_density
            )
        )
        active_density = (
            self.active_space._transform_active.invert().transform_electronic_integrals(
                active_density
            )
        )
        return active_density

    def update_total_density(self, problem, result):
        """TODO."""
        # get total MO density
        total = problem.properties.electronic_density
        # find total MO density component which overlaps with active space
        subspace = self.get_active_density_component(total)
        # subtract that component from the total
        total -= subspace
        # get active space component
        active = self.damp_active_density(self.active_density_history)
        self.active_density_history[-1] = active
        # expand active component to total system size
        superspace = deepcopy(active)
        superspace = (
            self.active_space._transform_active.invert().transform_electronic_integrals(
                superspace
            )
        )
        # add expanded active space component to the total
        total += superspace

        return total

    @staticmethod
    def damp_active_density(density_history):
        return density_history[-1]


def compute_inactive_fock(
    active_space_transformer,
    hamiltonian,
    density_active,
    total_fock_ref,
    e_inactive_ref,
):
    """TODO."""
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
    e_inactive_sum = (
        e_inactive_ref
        + e_inactive.alpha.get("", 0.0)
        + e_inactive.beta.get("", 0.0)
        + e_inactive.beta_alpha.get("", 0.0)
    )

    new_hamil = ElectronicEnergy(
        active_space_transformer._transform_active.transform_electronic_integrals(
            inactive_fock + hamiltonian.electronic_integrals.two_body
        )
    )
    new_hamil.constants = deepcopy(hamiltonian.constants)
    new_hamil.constants[active_space_transformer.__class__.__name__] = e_inactive_sum

    return new_hamil
