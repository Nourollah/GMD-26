# Updated import to include OrcaProfile
import os
import subprocess
import typing

from ase import md
from ase import units
from ase.calculators.orca import ORCA, OrcaProfile
from rich.progress import track

from gmd_26.core import base


class LangevinWithORCA(base.MolecularDynamicsAbstract):
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
        self.set_calculator(**kwargs)
        self.set_dynamics()

    @base.fluent
    @base.requires_molecule
    def set_calculator(
            self,
            orca_path: str,
            log_directory: str = "orca_files",
            orca_keywords: str = "EnGrad wB97M-V def2-TZVPD RIJCOSX TightSCF DEFGRID3",  # [cite: 7, 145, 154] ""
            charge: int = 0,
            multiplicity: int = 1,
            num_threads: int = 1
    ) -> "LangevinWithORCA":
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
        # Check for OpenMPI if num_threads > 1
        if num_threads > 1:
            try:
                result = subprocess.run(['mpirun', '--version'], capture_output=True, text=True, check=True)
                print(f"OpenMPI version: {result.stdout.splitlines()[0]}")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"OpenMPI check failed: {e}")
                raise RuntimeError(
                    "OpenMPI is not installed. Cannot use more than one thread (num_threads > 1). Set num_threads=1 or install OpenMPI.")
        os.makedirs(log_directory, exist_ok=True)
        # Explicitly create the OrcaProfile with the direct command path.
        # This is the robust fix you implemented.
        profile = OrcaProfile(command=orca_path)

        # Settings from the OMol25 paper [cite: 7, 145, 151, 154, 87, 88]
        # orca_keywords = "EnGrad def2-SVP"  # [cite: 7, 145, 154] ""

        # Instantiate ORCA, ensuring correct syntax with commas
        calculator_method = ORCA(
            # template=OrcaTemplate(),
            profile=profile,
            label=os.path.join(log_directory, "orca_calc"),
            charge=charge,
            mult=multiplicity,
            task="gradient",
            orcasimpleinput=orca_keywords,
            orcablocks=f"%scf Convergence verytight \n maxiter 300 end \n %pal nprocs {num_threads} end"
        )
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
            timestep: float = 0.5,
            temperature_K: float = 300.0,
            friction: float = 0.01,
            fixcm: bool = True
    ) -> "LangevinWithFlashMD":
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
    def run(self, step: int, *args, **kwargs) -> None:
        """Run the Langevin dynamics for the specified number of steps."""
        chunk_size = max(10, min(50, step // 10))
        for _ in track(range(0, step, chunk_size), description="Running ORCA AIMD..."):
            remaining_steps = min(chunk_size, step - _)
            self._dynamics.run(remaining_steps)
