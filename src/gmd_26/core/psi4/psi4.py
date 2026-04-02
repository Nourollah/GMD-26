import os
import typing

from ase import md
from ase import units
from ase.calculators.psi4 import Psi4
import psi4
from rich.progress import track

from gmd_26.core import base


class LangevinWithPsi4(base.MolecularDynamicsAbstract):
    def __init__(
            self,
            molecule_smile: str,
            **kwargs
    ):
        super().__init__()
        self.set_molecule(molecule_smile)
        self.set_calculator(**kwargs)
        self.set_dynamics()

    @base.fluent
    @base.requires_molecule
    def set_calculator(
            self,
            calculator_method: typing.Any = None,
            method: str = "wB97M-V",
            basis: str = "def2-TZVPD",
            memory: str = "4 GB",
            num_threads: int = 4,
            directory: str = "psi4_files"
    ) -> "LangevinWithPsi4":
        os.makedirs(directory, exist_ok=True)
        psi4.set_output_file(os.path.join(directory, "psi4_output.dat"), False)
        if calculator_method is None:
            calculator_method = Psi4(
                method=method,
                basis=basis,
                memory=memory,
                num_threads=num_threads,
                directory=directory
            )
            psi4.set_options({
                'basis': basis,
                'scf_type': 'df',
                'reference': 'uhf',
                'scf_convergence': 8,
                'd_convergence': 8,
                'maxiter': 200
            })
        self._calculator = calculator_method
        self._molecule.calc = self._calculator
        print("Psi4 calculator has been set on the molecule successfully ...")
        return self

    @staticmethod
    def available_methods():
        print("Available method: wB97M-V with def2-TZVPD basis set")

    @base.fluent
    @base.requires_molecule
    def set_dynamics(
            self,
            dynamics: typing.Optional[object] = None,
            timestep: float = 0.5,
            temperature_K: float = 300.0,
            friction: float = 0.01,
            fixcm: bool = True
    ) -> "LangevinWithPsi4":
        if dynamics is None:
            if self._molecule is None or self._calculator is None:
                raise ValueError("Molecule and calculator must be set before default Langevin dynamics.")
            dynamics = md.langevin.Langevin(
                self._molecule,
                timestep=timestep * units.fs,
                temperature_K=temperature_K,
                friction=friction,
                fixcm=fixcm
            )
        if not isinstance(dynamics, md.langevin.Langevin):
            raise TypeError("dynamics must be a valid ASE Langevin dynamics object")
        self._dynamics = dynamics
        return self

    @base.requires_molecule
    @base.requires_dynamics
    @base.requires_calculator
    def run(self, step: int, *args, **kwargs):
        """Run the Langevin dynamics for the specified number of steps."""
        chunk_size = max(10, min(100, step // 10))
        for _ in track(range(0, step, chunk_size), description="Running Psi4 dynamics..."):
            remaining_steps = min(chunk_size, step - _)
            self._dynamics.run(remaining_steps)
