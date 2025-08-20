import abc
import typing
import os
import pathlib
import subprocess
import glob

import numpy as np
import pandas as pd
import torch
import ase
from ase import Atoms, units
from ase.io import Trajectory, write, read
from ase.md import MDLogger
from ase.data.pubchem import pubchem_atoms_search
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.calculators.singlepoint import SinglePointCalculator
from rdkit import Chem
from rdkit.Chem import AllChem
import typer
from rich.progress import track

# engine: str = "xtb"
# if engine.lower() == "xtb":
from xtb.utils import _methods
from xtb.ase.calculator import XTB

# if engine.lower() == "psi4":
#     import psi4
# from ase.calculators.psi4 import Psi4

# if engine.lower() == "orca":
# Updated import to include OrcaProfile
# from ase.calculators.orca import ORCA, OrcaProfile, OrcaTemplate

# if engine.lower() == "flashmd":
from pet_mad.calculator import PETMADCalculator
from flashmd import get_universal_model
from flashmd.ase.langevin import Langevin
from pet_mad.calculator import PETMADCalculator

app = typer.Typer(help="Molecular Dynamics Simulation CLI")


def fluent(method: typing.Callable) -> typing.Callable:
    """Decorator to ensure a method returns self for fluent chaining."""

    def wrapper(self, *args, **kwargs):
        method(self, *args, **kwargs)
        return self

    return wrapper


def requires_molecule(method: typing.Callable) -> typing.Callable:
    """Decorator to ensure molecule is set before method execution."""

    def wrapper(self, *args, **kwargs):
        if self._molecule is None:
            raise ValueError("Molecule must be set before calling this method.")
        return method(self, *args, **kwargs)

    return wrapper


def requires_calculator(method: typing.Callable) -> typing.Callable:
    """Decorator to ensure calculator is set before method execution."""

    def wrapper(self, *args, **kwargs):
        if self._calculator is None:
            raise ValueError("A calculator must be set before calling this method.")
        return method(self, *args, **kwargs)

    return wrapper


def requires_dynamics(method: typing.Callable) -> typing.Callable:
    """Decorator to ensure dynamics is set before method execution."""

    def wrapper(self, *args, **kwargs):
        if self._dynamics is None:
            raise ValueError("Dynamics method must be set before calling this method.")
        return method(self, *args, **kwargs)

    return wrapper


def generate_3d_coordinates_from_smiles(smiles: str) -> typing.Tuple[typing.List[str], np.ndarray]:
    """
    Generates 3D coordinates for a molecule from its SMILES string.

    This function uses RDKit to:
    1. Create a molecule object from a SMILES string.
    2. Add implicit hydrogen atoms.
    3. Generate a 3D conformation using the ETKDG algorithm.
    4. Optimize the geometry using the MMFF94 force field.
    5. Extract atomic symbols and 3D coordinates.

    Args:
        smiles: The SMILES string representation of the molecule.

    Returns:
        A tuple containing:
        - A list of atomic symbols (e.g., ['C', 'O', 'H', ...]).
        - A NumPy array of atomic coordinates with shape (N, 3), where N
          is the number of atoms.

    Raises:
        ValueError: If a valid 3D conformation cannot be generated.
    """
    # 1. Create a molecule from the SMILES string
    mol = Chem.MolFromSmiles(smiles)

    # 2. Add hydrogens
    mol = Chem.AddHs(mol)

    # 3. Generate a 3D conformation. EmbedMolecule returns an ID for the
    #    conformation generated. -1 means it failed.
    #    AllChem.ETKDG() is a modern algorithm for conformer generation.
    conformer_id = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if conformer_id == -1:
        raise ValueError(f"Could not generate a 3D conformation for SMILES: {smiles}")

    # 4. Optimize the geometry using a force field (e.g., MMFF94)
    #    This step refines the initial 3D guess to a more stable,
    #    low-energy structure.
    AllChem.MMFFOptimizeMolecule(mol, confId=0)

    # 5. Extract the conformer and coordinates
    conformer = mol.GetConformer(0)
    positions = conformer.GetPositions()  # Returns a NumPy array

    # Extract atomic symbols
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    return symbols, positions


class MolecularDynamicsAbstract(abc.ABC):
    def __init__(self):
        self._calculator: typing.Optional[typing.Callable] = None
        self._dynamics: typing.Optional[object] = None
        self._molecule: typing.Optional[Atoms] = None
        self._trajectory_logger: typing.Optional[Trajectory] = None
        self._trajectory_path: typing.Optional[str] = None
        self._md_logger: typing.Optional[MDLogger] = None
        self._md_log_path: typing.Optional[str] = None
        self._logging_interval: int = 10

    @property
    def logging_interval(self) -> int:
        return self._logging_interval

    @fluent
    @logging_interval.setter
    def logging_interval(self, logging_interval: int) -> "MolecularDynamicsAbstract":
        self._logging_interval = logging_interval
        return self

    @fluent
    def set_logging_interval(self, logging_interval: int) -> "MolecularDynamicsAbstract":
        self._logging_interval = logging_interval
        return self

    @property
    @requires_dynamics
    def dynamics(self) -> object:
        if self._dynamics is None:
            raise ValueError("Dynamics has not been set.")
        return self._dynamics

    @fluent
    @abc.abstractmethod
    def set_dynamics(self, dynamics: object) -> "MolecularDynamicsAbstract":
        pass

    @property
    @requires_calculator
    def calculator(self) -> typing.Callable:
        if self._calculator is None:
            raise ValueError("Calculator has not been set.")
        return self._calculator

    @fluent
    @abc.abstractmethod
    def set_calculator(self) -> "MolecularDynamicsAbstract":
        pass

    @property
    def molecule(self) -> Atoms:
        if self._molecule is None:
            raise ValueError("Molecule has not been set.")
        return self._molecule

    @fluent
    def set_molecule(self, molecule_smile: str) -> "MolecularDynamicsAbstract":

        # molecule = pubchem_atoms_search(smiles=molecule_smile)
        # if molecule is not None and isinstance(molecule, Atoms):
            # typer.echo(f"Fetched molecule from PubChem for SMILE: {molecule_smile}")
        # else:
        typer.echo(f"Creating molecule from SMILE: {molecule_smile}")
        symbols, coordinates = generate_3d_coordinates_from_smiles(molecule_smile)
        if symbols is None or coordinates is None:
            raise ValueError(f"Could not generate a molecule from SMILE: {molecule_smile}")
        molecule = Atoms(symbols=symbols, positions=coordinates, charges=[0 for _ in range(len(symbols))])
        
        if not isinstance(molecule, Atoms):
            raise TypeError(f"The given smiles '{molecule_smile}' does not produce a valid ASE Atoms object.")
        self._molecule = molecule
        if self._calculator:
            self._molecule.calc = self._calculator
        return self

    @requires_molecule
    def save_state(self, filename: str) -> None:
        """
        Save the current state of the molecule to a file.
        This can be used to continue the simulation later.

        Args:
           filename: Path to save the molecule state
        """
        write(filename, self._molecule)
        print(f"Molecule state saved to {filename}")

    @fluent
    def load_state(self, filename: str) -> "MolecularDynamicsAbstract":
        """
        Load a molecule state from a file to continue a simulation.

        Args:
           filename: Path to the molecule state file

        Returns:
           Self for method chaining
        """
        # Load the molecule
        loaded_molecule = read(filename)

        # Make sure to preserve the calculator
        if self._calculator is not None:
            loaded_molecule.calc = self._calculator

        self._molecule = loaded_molecule

        # If dynamics already exists, we need to update its atoms reference
        if self._dynamics is not None:
            self._dynamics.atoms = self._molecule

        print(f"Molecule state loaded from {filename}")
        return self

    @fluent
    @requires_molecule
    def set_box(self, cell_scale: typing.Union[list, np.ndarray, float] = 10.0,
                pbc: bool = False) -> "MolecularDynamicsAbstract":
        cell_array = np.array(cell_scale)
        d = 0.776 / 1e24  # A placeholder density value
        mass_kg = self._molecule.get_masses().sum() * (units.u / units.kg)
        volume_m3 = mass_kg / d
        L_m = volume_m3 ** (1. / 3.)
        L_angstrom = L_m * 1e10

        self._molecule.set_cell((L_angstrom, L_angstrom, L_angstrom))
        self._molecule.set_pbc(pbc)
        self._molecule.center()
        return self

    @property
    def trajectory_file(self) -> typing.Optional[str]:
        return self._trajectory_path

    @fluent
    @requires_dynamics
    @requires_molecule
    def set_trajectory_file(self, trajectory_file: str) -> "MolecularDynamicsAbstract":
        self._trajectory_path = trajectory_file
        self._trajectory_logger = Trajectory(
            self._trajectory_path,
            "w",
            self._molecule
        )
        self._dynamics.attach(
            self._trajectory_logger.write,
            interval=self._logging_interval
        )
        return self

    @property
    def md_logger_file(self) -> typing.Optional[str]:
        return self._md_log_path

    @fluent
    @requires_dynamics
    @requires_molecule
    def set_md_logger(
            self,
            md_logger_file: str,
            header: bool = True,
            stress: bool = False,
            per_atom: bool = True,
            mode: str = "a"
    ) -> "MolecularDynamicsAbstract":
        self._md_log_path = md_logger_file

        self._md_logger = MDLogger(
            self._dynamics,
            self._molecule,
            self._md_log_path,
            header=header,
            stress=stress,
            peratom=per_atom,
            mode=mode,
        )
        self._dynamics.attach(
            self._md_logger,
            interval=self._logging_interval
        )
        return self

    @fluent
    def cleanup(self) -> "MolecularDynamicsAbstract":
        """Clean up resources used by the molecular dynamics simulation."""
        if self._dynamics is not None:
            if self._trajectory_logger is not None:
                if hasattr(self._dynamics, '_observers'):
                    write_func = self._trajectory_logger.write
                    self._dynamics._observers = [(interval, func) for interval, func in self._dynamics._observers
                                                 if func != write_func]
                self._trajectory_logger.close()
                self._trajectory_logger = None
                self._trajectory_path = None

            if self._md_logger is not None:
                if hasattr(self._dynamics, '_observers'):
                    self._dynamics._observers = [(interval, func) for interval, func in self._dynamics._observers
                                                 if func != self._md_logger]
                if hasattr(self._md_logger, 'close'):
                    self._md_logger.close()
                self._md_logger = None
                self._md_log_path = None
        return self

    @requires_dynamics
    @requires_calculator
    @requires_molecule
    def __call__(self, step: int = 1, *args, **kwargs):
        return self.run(step, *args, **kwargs)

    @abc.abstractmethod
    def run(self, step: int, *args, **kwargs):
        """Abstract method to run the dynamics."""
        pass


class LangevinWithXTB(MolecularDynamicsAbstract):
    def __init__(
            self,
            molecule_smile: str,
            **kwargs
    ):
        super().__init__()
        self.set_molecule(molecule_smile)
        self.set_calculator(**kwargs)
        self.set_dynamics()

    @fluent
    @requires_molecule
    def set_calculator(
            self,
            calculator_method: typing.Any = None,
            method: str = "GFN2-xTB",
            directory: str = "xtb_files",
    ) -> "LangevinWithXTB":
        os.makedirs(directory, exist_ok=True)
        if calculator_method is None:
            calculator_method = XTB(
                method=method,
                directory=directory,
            )

        self._calculator = calculator_method
        self._molecule.calc = self._calculator

        print("Calculator has been set on the molecule successfully ...")
        return self

    @staticmethod
    def available_methods():
        print(_methods)

    @fluent
    @requires_molecule
    def set_dynamics(
            self,
            dynamics: typing.Optional[object] = None,
            timestep: float = 0.1,
            temperature_K: float = 300.0,
            friction: float = 0.01,
            fixcm: bool = True
    ) -> "LangevinWithXTB":
        if dynamics is None:
            if self._molecule is None or self._calculator is None:
                raise ValueError("Molecule and calculator must be set before default Langevin dynamics.")
            dynamics = Langevin(
                self._molecule,
                timestep=timestep * units.fs,
                temperature_K=temperature_K,
                friction=friction,
                fixcm=fixcm
            )
        if not isinstance(dynamics, Langevin):
            raise TypeError("dynamics must be a valid ASE Langevin dynamics object")
        self._dynamics = dynamics
        return self

    @requires_molecule
    @requires_dynamics
    @requires_calculator
    def run(self, step: int, *args, **kwargs):
        """Run the Langevin dynamics for the specified number of steps."""
        chunk_size = max(100, min(1000, step // 10))
        for _ in track(range(0, step, chunk_size), description="Running XTB dynamics..."):
            remaining_steps = min(chunk_size, step - _)
            self._dynamics.run(remaining_steps)


class LangevinWithPsi4(MolecularDynamicsAbstract):
    def __init__(
            self,
            molecule_smile: str,
            **kwargs
    ):
        super().__init__()
        self.set_molecule(molecule_smile)
        self.set_calculator(**kwargs)
        self.set_dynamics()

    @fluent
    @requires_molecule
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

    @fluent
    @requires_molecule
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
            dynamics = Langevin(
                self._molecule,
                timestep=timestep * units.fs,
                temperature_K=temperature_K,
                friction=friction,
                fixcm=fixcm
            )
        if not isinstance(dynamics, Langevin):
            raise TypeError("dynamics must be a valid ASE Langevin dynamics object")
        self._dynamics = dynamics
        return self

    @requires_molecule
    @requires_dynamics
    @requires_calculator
    def run(self, step: int, *args, **kwargs):
        """Run the Langevin dynamics for the specified number of steps."""
        chunk_size = max(10, min(100, step // 10))
        for _ in track(range(0, step, chunk_size), description="Running Psi4 dynamics..."):
            remaining_steps = min(chunk_size, step - _)
            self._dynamics.run(remaining_steps)


class LangevinWithORCA(MolecularDynamicsAbstract):
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

    @fluent
    @requires_molecule
    def set_calculator(
            self,
            orca_path: str,
            log_directory: str = "orca_files",
            orca_keywords: str = "EnGrad wB97M-V def2-TZVPD RIJCOSX TightSCF DEFGRID3",  # [cite: 7, 145, 154] ""
            charge: int = 0,
            multiplicity: int = 1,
            num_threads: int = 1
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

    @fluent
    @requires_molecule
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
            dynamics = Langevin(
                self._molecule,
                timestep=timestep * units.fs,
                temperature_K=temperature_K,
                friction=friction,
                fixcm=fixcm
            )
        if not isinstance(dynamics, Langevin):
            raise TypeError("dynamics must be a valid ASE Langevin dynamics object")
        self._dynamics = dynamics
        return self

    @requires_molecule
    @requires_dynamics
    @requires_calculator
    def run(self, step: int, *args, **kwargs) -> None:
        """Run the Langevin dynamics for the specified number of steps."""
        chunk_size = max(10, min(50, step // 10))
        for _ in track(range(0, step, chunk_size), description="Running ORCA AIMD..."):
            remaining_steps = min(chunk_size, step - _)
            self._dynamics.run(remaining_steps)


class LangevinWithFlashMD(MolecularDynamicsAbstract):
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

    @fluent
    @requires_molecule
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

    @fluent
    @requires_molecule
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

    @requires_molecule
    @requires_dynamics
    @requires_calculator
    def run(self, step: int, *args, **kwargs) -> None:
        """Run the Langevin dynamics for the specified number of steps."""
        chunk_size = max(10, min(50, step // 10))
        for _ in track(range(0, step, chunk_size), description="Running FlashMD..."):
            remaining_steps = min(chunk_size, step - _)
            self._dynamics.run(remaining_steps)


@app.command()
def run(
        smiles_file: pathlib.Path = typer.Option(
            "smiles.txt",
            "--smiles-file",
            "-sf",
            help="Path to file containing SMILES strings",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True
        ),
        run_steps: int = typer.Option(
            1000,
            "--run-steps",
            "-s",
            help="Number of steps to run in each simulation segment"
        ),
        from_checkpoint: bool = typer.Option(
            False,
            "--from-checkpoint",
            "-c",
            help="Whether to load from the last checkpoint if available"
        ),
        parent_dir: str = typer.Option(
            "RandomMolecules",
            "--parent-dir",
            "-d",
            help="Parent directory for storing simulation results"
        ),
        segments: int = typer.Option(
            5,
            "--segments",
            "-n",
            help="Number of simulation segments to run"
        ),
        final_steps: int = typer.Option(
            1000,
            "--final-steps",
            "-fs",
            help="Number of steps to run in the final simulation"
        ),
        logging_interval: int = typer.Option(
            10,
            "--logging-interval",
            "-li",
            help="Interval for logging during regular segments"
        ),
        final_logging_interval: int = typer.Option(
            2,
            "--final-logging-interval",
            "-fli",
            help="Interval for logging during the final simulation"
        ),
        temperature: float = typer.Option(
            300,
            "--temperature",
            "-t",
            help="Temperature in Kelvin for Maxwell-Boltzmann distribution"
        ),
        engine: str = typer.Option(
            "orca",
            "--engine",
            "-e",
            help="Simulation engine to use: 'xtb', 'psi4', or 'orca'"
        ),
        orca_path: str = typer.Option(
            "",
            "--orca-path",
            "-op",
            help="Full path to the ORCA executable (only used with --engine orca)"
        ),
        num_threads: int = typer.Option(
            1,
            "--num-threads",
            "-nt",
            help="Number of threads for parallel calculators like Psi4 or ORCA"
        ),
        skip_frames: int = typer.Option(
            16,
            "--skip-frames",
            "-sf",
            help="Number of frames to skip for FlashMD simulations (default: 16 fs model, can be 1, 4, 8, 32, or 64 fs)"
        )
):
    """
    Run molecular dynamics simulations for a list of SMILES strings.
    """
    smile_list: typing.List[str] = pd.read_csv(smiles_file, header=None)[0].tolist()
    os.makedirs(parent_dir, exist_ok=True)
    original_dir = os.getcwd()

    engine_map = {
        "orca": LangevinWithFlashMD,
        "xtb": LangevinWithXTB,
        "psi4": LangevinWithPsi4,
        "flashmd": LangevinWithFlashMD
    }

    if engine.lower() not in engine_map:
        typer.echo(f"Error: Invalid engine '{engine}'. Choose from {list(engine_map.keys())}")
        raise typer.Exit(code=1)

    EngineClass = engine_map[engine.lower()]

    typer.echo(f"Using simulation engine: {engine.upper()}")
    typer.echo(f"Running {len(smile_list)} simulations...")

    for smile in smile_list:
        typer.echo(f"--- Processing SMILE: {smile} ---")
        dir_name = f"{smile.lower().count('c')}_{smile.replace('/', '_')}"
        smile_dir = os.path.join(parent_dir, dir_name)
        os.makedirs(smile_dir, exist_ok=True)
        os.chdir(smile_dir)

        try:

            match engine.lower():
                case "orca":
                    md = EngineClass(molecule_smile=smile, orca_path=orca_path, num_threads=num_threads)
                case "xtb":
                    md = EngineClass(molecule_smile=smile, num_threads=num_threads)
                case "psi4":
                    md = EngineClass(molecule_smile=smile, num_threads=num_threads)
                case "flashmd":
                    md = EngineClass(molecule_smile=smile, timestep=skip_frames)
            # if isinstance(md, LangevinWithORCA):
            # 	md.set_calculator(orca_path=orca_path)
            # elif isinstance(md, LangevinWithXTB):
            # 	md.set_calculator(num_threads=num_threads)
            # elif isinstance(md, LangevinWithPsi4):
            # 	md.set_calculator(num_threads=num_threads)

            checkpoint_file = "last_step.traj"
            if os.path.isfile(checkpoint_file) and from_checkpoint:
                md.load_state(checkpoint_file)
                typer.echo(f"Loaded state from {checkpoint_file}")
            else:
                MaxwellBoltzmannDistribution(md.molecule, temperature_K=temperature)
                typer.echo("Initialized new simulation with Maxwell-Boltzmann velocities.")

            md.set_logging_interval(logging_interval)
            md.set_trajectory_file(f"{smile}_total.traj").set_md_logger(f"{smile}_total.log")

        except Exception as e:
            os.chdir(original_dir)
            typer.echo(f"Error during initialization for SMILE '{smile}': {e}")
            continue

        for i in range(segments):
            typer.echo(f"Running segment {i + 1}/{segments} with {run_steps} steps...")
            md(run_steps)
            md.save_state(checkpoint_file)

        typer.echo(f"Running final simulation with {final_steps} steps...")
        md.set_logging_interval(final_logging_interval).set_trajectory_file(
            f"{smile}_final.traj").set_md_logger(f"{smile}_final.log", mode="w")
        md(final_steps)
        md.save_state(checkpoint_file)
        md.cleanup()

        os.chdir(original_dir)


@app.command()
def compute(
        trajectory_dir: pathlib.Path = typer.Option(
            "RandomMolecules",
            "--trajectory-dir",
            "-td",
            help="Directory containing trajectory files to process",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True
        ),
):
    """
    Compute the energy and forces for a list of trajectory files.
    """
    # Loading trajectories
    traj_list = glob.glob(os.path.join(trajectory_dir, "*.traj"))
    if len(traj_list) == 0:
        typer.echo("No trajectories found in the specified directory.")
        raise typer.Exit(code=1)
    traj_list.sort()
    typer.echo(f"Found {len(traj_list)} trajectories.")

    # Setup the calculator
    calculator_method = XTB(
        method="GFN2-xTB",
        directory=f"{trajectory_dir}/xtb",
    )
    new_molecules = []
    for traj in traj_list:
        typer.echo(f"Processing trajectory: {traj}")
        try:
            molecules: list = read(traj, index=":")
            for mol in track(molecules):
                mol.calc = calculator_method
                energy = mol.get_potential_energy()
                forces = mol.get_forces()
                mol.calc = SinglePointCalculator(
                    mol,
                    energy=energy,
                    forces=forces
                )  # This keeps the energy and forces until the position of atoms gets changed

                new_molecules.append(mol)

            write(f"{traj.split('.')[0]}_xtb.traj", new_molecules)
        except Exception as e:
            typer.echo(f"Error processing trajectory '{traj}': {e}")
            continue


if __name__ == "__main__":
    app()
