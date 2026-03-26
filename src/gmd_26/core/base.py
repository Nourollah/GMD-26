import abc
import typing

import numpy as np
import typer
from ase import Atoms, units
from ase import io as atoms_io
from ase import md as md_logger
from rdkit import Chem
from rdkit.Chem import AllChem


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
		self._trajectory_logger: typing.Optional[md_logger.Trajectory] = None
		self._trajectory_path: typing.Optional[str] = None
		self._md_logger: typing.Optional[md_logger.MDLogger] = None
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
		atoms_io.write(filename, self._molecule)
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
		loaded_molecule = atoms_io.read(filename)

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
		self._trajectory_logger = md_logger.Trajectory(
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

		self._md_logger = md_logger.MDLogger(
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


	@requires_dynamics
	@requires_calculator
	@requires_molecule
	def __call__(self, step: int = 1, *args, **kwargs):
		return self.run(step, *args, **kwargs)

	@abc.abstractmethod
	def run(self, step: int, *args, **kwargs):
		"""Abstract method to run the dynamics."""
		pass
