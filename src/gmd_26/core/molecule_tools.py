import typing
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import ase


def generate_3d_coordinates_from_smiles(smiles: str) -> typing.Tuple[typing.List[str], np.ndarray]:
	"""
	Generates 3D coordinates for a molecule from its SMILES string.

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
	# Create a molecule from the SMILES string
	mol = Chem.MolFromSmiles(smiles)

	# Add hydrogens
	mol = Chem.AddHs(mol)

	# Generate a 3D conformation. EmbedMolecule returns an ID for the
	# conformation generated. -1 means it failed.
	# AllChem.ETKDG() is a modern algorithm for conformer generation.
	conformer_id = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
	if conformer_id == -1:
		raise ValueError(f"Could not generate a 3D conformation for SMILES: {smiles}")

	# Optimize the geometry using a force field (e.g., MMFF94)
	# This step refines the initial 3D guess to a more stable,
	# low-energy structure.
	AllChem.MMFFOptimizeMolecule(mol, confId=0)

	# Extract the conformer and coordinates
	conformer = mol.GetConformer(0)
	positions = conformer.GetPositions()  # Returns a NumPy array

	# Extract atomic symbols
	symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

	return symbols, positions


def box_molecule(
		molecule: ase.Atoms,
		box_scale: float = 10.0,
		pbc: bool = True,
		**kwargs,
) -> None:
	"""
	Sets a cubic cell around the molecule based on its diameter.

	Args:
	    molecule: An ASE Atoms object representing the molecule.
	    pbc: Whether to apply periodic boundary conditions.
	"""
	# Compute the diameter of the molecule
	positions = molecule.get_positions()
	diameter: float = np.max(np.ptp(positions, axis=0))  # Maximum span in any direction

	# Set a box 10 times bigger than the molecule
	box_size: float = box_scale * diameter
	molecule.set_cell(
		[box_size, box_size, box_size],
		**kwargs
	)

	# Put the molecule in the center of the box
	molecule.center(**kwargs)
	# Set periodic boundary conditions
	molecule.set_pbc(pbc)
