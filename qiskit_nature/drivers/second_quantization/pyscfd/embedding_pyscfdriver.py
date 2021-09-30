# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Embedding-PySCFDriver."""

import logging
from typing import List, Optional, Union, cast

import numpy as np

from qiskit_nature.properties.second_quantization.electronic import (
    ElectronicDensity,
    ElectronicEnergy,
    ElectronicStructureDriverResult,
    ParticleNumber,
)
from qiskit_nature.properties.second_quantization.electronic.bases import (
    ElectronicBasis,
    ElectronicBasisTransform,
)
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)

from .pyscfdriver import InitialGuess, PySCFDriver
from ..electronic_structure_driver import MethodType
from ...units_type import UnitsType

logger = logging.getLogger(__name__)


class EmbeddingPySCFDriver(PySCFDriver):
    """TODO."""

    def __init__(
        self,
        atom: Union[str, List[str]] = "H 0.0 0.0 0.0; H 0.0 0.0 0.735",
        unit: UnitsType = UnitsType.ANGSTROM,
        charge: int = 0,
        spin: int = 0,
        basis: str = "sto3g",
        method: MethodType = MethodType.RHF,
        xc_functional: str = "lda,vwn",
        xcf_library: str = "libxc",
        conv_tol: float = 1e-9,
        max_cycle: int = 50,
        init_guess: InitialGuess = InitialGuess.MINAO,
        max_memory: Optional[int] = None,
        chkfile: Optional[str] = None,
        omega: float = 1.0,
    ) -> None:
        """TODO."""
        super().__init__(
            atom=atom,
            unit=unit,
            charge=charge,
            spin=spin,
            basis=basis,
            method=method,
            xc_functional=xc_functional + f"({omega})",
            xcf_library=xcf_library,
            conv_tol=conv_tol,
            max_cycle=max_cycle,
            init_guess=init_guess,
            max_memory=max_memory,
            chkfile=chkfile,
        )
        self._omega = omega

        self._e_tot: float
        self._e_ref: float
        self._density: ElectronicDensity = None

        self._cached_electronic_integrals: List[ElectronicIntegrals] = None

    @property
    def omega(self) -> float:
        """Return the omega."""
        return self._omega

    @omega.setter
    def omega(self, omega: float) -> None:
        """Set the omega."""
        self._omega = omega

    def run_pyscf(self) -> None:
        """TODO."""
        super().run_pyscf()

        self._e_tot = self._calc.e_tot
        self._e_ref = self._calc.e_tot

    def evaluate_energy(self, density):
        """TODO."""
        calc_mo_coeff, calc_mo_coeff_b = self._extract_mo_data("mo_coeff", array_dimension=3)
        trafo = ElectronicBasisTransform(
            ElectronicBasis.AO,
            ElectronicBasis.MO,
            calc_mo_coeff,
            calc_mo_coeff_b,
        ).invert()
        density.transform_basis(trafo)
        self._density = density

        rho_ao_alpha, rho_ao_beta = density.get_electronic_integral(ElectronicBasis.AO, 1)._matrices

        rho_ao = np.asarray([rho_ao_alpha, rho_ao_beta])

        if calc_mo_coeff_b is None:
            rho_ao = rho_ao_alpha + rho_ao_beta

        self._e_tot = self._calc.energy_tot(dm=rho_ao)

        return self._construct_driver_result()

    def _construct_driver_result(self) -> ElectronicStructureDriverResult:
        driver_result = super()._construct_driver_result()
        if self._density is None:
            particle_number = cast(ParticleNumber, driver_result.get_property(ParticleNumber))
            driver_result.add_property(ElectronicDensity.from_particle_number(particle_number))
        else:
            driver_result.add_property(self._density)
        return driver_result

    def _populate_driver_result_electronic_energy(
        self, driver_result: ElectronicStructureDriverResult
    ) -> None:
        basis_transform = driver_result.get_property(ElectronicBasisTransform)

        fock = self._calc.get_fock()
        if len(fock.shape) == 3:
            fock_ints = (fock[0], fock[1])
        else:
            fock_ints = (fock, None)
        fock_ao = OneBodyElectronicIntegrals(ElectronicBasis.AO, fock_ints)
        fock_mo = fock_ao.transform_basis(basis_transform)

        if self._cached_electronic_integrals is None:
            one_body_ao = OneBodyElectronicIntegrals(
                ElectronicBasis.AO,
                (self._calc.get_hcore(), None),
            )
            one_body_mo = one_body_ao.transform_basis(basis_transform)

            with self._mol.with_range_coulomb(omega=self._omega):
                two_body_ao = TwoBodyElectronicIntegrals(
                    ElectronicBasis.AO,
                    (self._mol.intor("int2e", aosym=1), None, None, None),
                )
                two_body_mo = two_body_ao.transform_basis(basis_transform)

            self._cached_electronic_integrals = [one_body_mo, two_body_mo]

        electronic_energy = ElectronicEnergy(
            self._cached_electronic_integrals,
            nuclear_repulsion_energy=0.0,
            reference_energy=self._e_tot,
        )

        electronic_energy.fock = fock_mo

        driver_result.add_property(electronic_energy)
