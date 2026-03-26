import typing

import ase
import torch
from ase import units
from flashmd import get_universal_model
from flashmd.ase.langevin import Langevin
from pet_mad.calculator import PETMADCalculator
from rich.progress import track

from gmd_26.core import base


class LangevinWithFlashMD(base.MolecularDynamicsAbstract):
    """
    Molecular dynamics simulation using Langevin dynamics with the ORCA calculator.
    This class is configured to use the ωB97M-V/def2-TZVPD level of theory,
    matching the high-accuracy calculations from the OMol25 paper.
    """

    def __init__(
            self,
            molecule_smile: str,
            **kwargs
    ):
        super().__init__()
        # Set dummy calculator first to allow molecule creation
        self.set_molecule(molecule_smile)
        self.set_calculator()
        self.set_dynamics(**kwargs)

    @base.fluent
    @base.requires_molecule
    def set_calculator(
            self,
            multiplicity: int = 1,
    ) -> "LangevinWithFlashMD":
        """
        Sets up the ORCA calculator using a specific command path.

        This method is more robust as it does not rely on the system's PATH
        variable. You provide the full, direct path to the ORCA executable.

        Args:
            orca_path (str): The full path to the ORCA executable.
            log_directory (str): Folder to store ORCA output files.
            charge (int): Total charge of the system.
            multiplicity (int): Spin multiplicity of the system.
            num_threads (int): Number of processor cores for ORCA to use.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        calculator_method = PETMADCalculator(device=device)

        self._calculator = calculator_method
        if self._molecule:
            self._molecule.calc = self._calculator
            try:
                forces = self._molecule.get_forces()
                print(f"Test forces: {forces}")
            except Exception as e:
                print(f"Error computing forces: {e}")
                raise
        print("ORCA calculator has been set on the molecule successfully ...")
        return self

    @staticmethod
    def available_methods() -> None:
        """Prints the fixed level of theory used by this class."""
        print("Level of Theory: wB97M-V with def2-TZVPD basis set (from OMol25 paper)")  # [cite: 7, 145]

    @base.fluent
    @base.requires_molecule
    def set_dynamics(
            self,
            dynamics: typing.Optional[object] = None,
            timestep: float = 16,  # 16 fs model; also available: 1, 4, 8, 32, 64 fs
            temperature_K: float = 300.0,
            friction: float = 0.01,
            fixcm: bool = True,
    ) -> "LangevinWithORCA":
        if dynamics is None:
            if self._molecule is None or self._calculator is None:
                raise ValueError("Molecule and calculator must be set before default Langevin dynamics.")
            model = get_universal_model(timestep)
            dynamics = Langevin(
                self._molecule,
                timestep=timestep * units.fs,
                time_constant=100 * ase.units.fs,
                model=model,
                device="cuda" if torch.cuda.is_available() else "cpu",
                temperature_K=temperature_K,
            )
        if not isinstance(dynamics, Langevin):
            raise TypeError("dynamics must be a valid ASE Langevin dynamics object")
        self._dynamics = dynamics
        return self

    @base.requires_molecule
    @base.requires_dynamics
    @base.requires_calculator
    def run(self, step: int, *args, **kwargs) -> None:
        """Run the Langevin dynamics for the specified number of steps."""
        chunk_size = max(10, min(50, step // 10))
        for _ in track(range(0, step, chunk_size), description="Running FlashMD..."):
            remaining_steps = min(chunk_size, step - _)
            self._dynamics.run(remaining_steps)