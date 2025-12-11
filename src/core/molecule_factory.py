import typing
import abc



from ase import io as atoms_io
import ase

from molecule_tools import generate_3d_coordinates_from_smiles


class MoleculeFactory:
	"""
	Factory class to register builder and create molecules. It can have single or multiple molecules.
	"""

	@classmethod
	def register(cls, key: str, builder: "MoleculeBuilder") -> None:
		cls._builders[key] = builder

	@classmethod
	def create(cls, key: str, *args: typing.Any, **kwargs: typing.Any) -> ase.Atoms:
		builder: typing.Optional["MoleculeBuilder"] = cls._builders.get(key)

		if not builder:
			available_keys: str = ", ".join(cls._builders.keys())
			raise ValueError(f"Molecule builder '{key}' is not registered. Available builders: {available_keys}")

		return builder.build(*args, **kwargs)


class MoleculeBuilder(abc.ABC):
	@abc.abstractmethod
	def build(self, *args: typing.Any, **kwargs: typing.Any) -> ase.Atoms | typing.List[ase.Atoms]:
		raise NotImplementedError("Subclasses must implement the build method.")


def register_molecule_set(key: str) -> typing.Callable[[typing.Type[MoleculeBuilder]], typing.Type[MoleculeBuilder]]:
	def decorator(builder_cls: typing.Type[MoleculeBuilder]) -> typing.Type[MoleculeBuilder]:
		instance = builder_cls()
		MoleculeFactory.register(key, instance)
		return builder_cls

	return decorator


@register_molecule_set("from_smiles")
class FromSmilesBuilder(MoleculeBuilder):
	def build(self, smiles: str) -> ase.Atoms:
		symbols, positions = generate_3d_coordinates_from_smiles(smiles)
		return ase.Atoms(
			symbols,
			positions=positions
		)

@register_molecule_set("from_file")
class FromFileBuilder(MoleculeBuilder):
	def build(
			self,
			file_path: str,
			index: str = ":",
			**kwargs
	) -> typing.List[ase.Atoms]:
		return atoms_io.read(
			file_path,
			index=index,
			**kwargs
		)

@register_molecule_set("from_db")
class FromDBBuilder(MoleculeBuilder):
	def build(self, db_path: str, **kwargs) -> typing.List[ase.Atoms]:
		raise NotImplementedError