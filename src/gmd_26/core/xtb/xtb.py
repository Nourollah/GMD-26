import typing
import os

from ase import units
from ase import md
from xtb.utils import _methods
from xtb.ase.calculator import XTB
from rich.progress import track

from gmd_26.core import base


class LangevinWithXTB(base.MolecularDynamicsAbstract):
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

	@base.fluent
	@base.requires_molecule
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
		chunk_size = max(100, min(1000, step // 10))
		for _ in track(range(0, step, chunk_size), description="Running XTB dynamics..."):
			remaining_steps = min(chunk_size, step - _)
			self._dynamics.run(remaining_steps)
