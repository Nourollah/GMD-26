import shutil
import typing
import os
import pathlib
import ase.calculators.calculator
import torch
from ase.md.langevin import Langevin
from flashmd.ase.langevin import Langevin as fLangevin
from metatomic.torch import AtomisticModel
from ase import Atoms, units
import abc


class DynamicsFactory:
	"""
	Factory class to register builder and create calculators. Singleton pattern.
	"""
	_builders: typing.Dict[str, "DynamicsFactory"] = {}

	@classmethod
	def register(cls, key: str, builder: "DynamicsFactory") -> None:
		cls._builders[key] = builder

	@classmethod
	def create(cls, key: str, *args: typing.Any, **kwargs: typing.Any):
		builder: typing.Optional["DynamicsFactory"] = cls._builders.get(key)

		if not builder:
			available_keys: str = ", ".join(cls._builders.keys())
			raise ValueError(f"Dynamic builder '{key}' is not registered. Available dynamics: {available_keys}")

		return builder.build(*args, **kwargs)


class DynamicsBuilder(abc.ABC):
	@abc.abstractmethod
	def build(self, *args: typing.Any, **kwargs: typing.Any):
		raise NotImplementedError("Subclasses must implement the build method.")


def register_calculator(key: str) -> typing.Callable[[typing.Type[DynamicsBuilder]], typing.Type[DynamicsBuilder]]:
	def decorator(builder_cls: typing.Type[DynamicsBuilder]) -> typing.Type[DynamicsBuilder]:
		instance = builder_cls()
		DynamicsBuilder.register(key, instance)
		return builder_cls

	return decorator


@register_calculator("langevin_ase")
class LangevinASEBuilder(DynamicsBuilder):
	def build(
			self,
			molecule: Atoms,
			timestep: float = 1.0 * units.fs,
			temperature_K: float = 300.0,
			friction: float = 0.5,
			fixcm: bool = True,
			**kwargs
	) -> Langevin:
		dynamic = Langevin(
			molecule,
			timestep=timestep,
			temperature_K=temperature_K,
			friction=friction,
			fixcm=fixcm,
			**kwargs
		)
		return dynamic


@register_calculator("langevin_flashmd")
class LangevinASEBuilder(DynamicsBuilder):
	def build(
			self,
			molecule: Atoms,
			model: AtomisticModel,
			timestep: float = 1.0 * units.fs,
			temperature_K: float = 300.0,
			friction: float = 0.5,
			fixcm: bool = True,
			**kwargs
	) -> Langevin:
		dynamic = fLangevin(
			molecule,
			model=model,
			timestep=timestep,
			temperature_K=temperature_K,
			device="cuda" if torch.cuda.is_available() else "cpu",
			**kwargs
		)

		return dynamic


if __name__ == "__main__":
	# Example usage
	cp2k_calculator = DynamicsFactory.create(
		"cp2k",
		cp2k_path="cp2k.ssmp",
		input_content_path="/home/masoud/Projects/SBMD/oneconf.inp"
	)
	print(cp2k_calculator)
