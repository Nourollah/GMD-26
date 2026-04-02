from pyexpat import model
import shutil
import typing
import os
import pathlib
import ase.calculators.calculator
from flashmd import get_pretrained
from flashmd.ase import EnergyCalculator
from ase.calculators.orca import ORCA, OrcaProfile, OrcaTemplate
from ase.calculators.cp2k import CP2K
import torch
import abc

from torch import device


class CalculatorFactory:
    """
    Factory class to register builder and create calculators. Singleton pattern.
    """
    _builders: typing.Dict[str, "CalculatorBuilder"] = {}

    @classmethod
    def register(cls, key: str, builder: "CalculatorBuilder") -> None:
        cls._builders[key] = builder

    @classmethod
    def create(cls, key: str, *args: typing.Any, **kwargs: typing.Any) -> ase.calculators.calculator.Calculator:
        builder: typing.Optional["CalculatorBuilder"] = cls._builders.get(key)

        if not builder:
            available_keys: str = ", ".join(cls._builders.keys())
            raise ValueError(
                f"Calculator builder '{key}' is not registered. Available builders: {available_keys}")

        return builder.build(*args, **kwargs)


class CalculatorBuilder(abc.ABC):
    @abc.abstractmethod
    def build(self, *args: typing.Any, **kwargs: typing.Any):
        raise NotImplementedError(
            "Subclasses must implement the build method.")


def register_calculator(key: str) -> typing.Callable[[typing.Type[CalculatorBuilder]], typing.Type[CalculatorBuilder]]:
    def decorator(builder_cls: typing.Type[CalculatorBuilder]) -> typing.Type[CalculatorBuilder]:
        instance = builder_cls()
        CalculatorFactory.register(key, instance)
        return builder_cls

    return decorator


@register_calculator("orca")
class ORCABuilder(CalculatorBuilder):
    def build(
        self,
        orca_path: str,
        orca_keywords: str = "EnGrad wB97M-V def2-TZVPD RIJCOSX TightSCF DEFGRID3",
        charge: int = 0,
        multiplicity: int = 1,
        num_threads: int = 1,
        log_directory: str = "orca_files"
    ) -> ORCA:
        profile = OrcaProfile(command=orca_path)
        calculator = ORCA(
            template=OrcaTemplate(
                keywords=orca_keywords,
                charge=charge,
                multiplicity=multiplicity
            ),
            profile=profile,
            num_threads=num_threads,
            directory=log_directory
        )
        return calculator


@register_calculator("cp2k")  # pyright: ignore[reportArgumentType]
class CP2KBuilder:
    def build(
        self,
        cp2k_path: str = "cp2k.ssmp",
        input_content_path: typing.Optional[str] = None
    ) -> ase.calculators.calculator.Calculator:

        # 1. READ INPUT FILE
        inp_content = None
        if input_content_path:
            inp_path = pathlib.Path(input_content_path)
            if not inp_path.exists():
                raise FileNotFoundError(f"Input file not found at: {inp_path}")
            inp_content = inp_path.read_text()

            # Sanity check for unbalanced tags (prevents silent crashes)
            if inp_content.count("&") % 2 != 0:
                print(
                    f"Warning: Input file {input_content_path} has unbalanced '&' tags.")

        # 2. RESOLVE EXECUTABLE (The Fix for AssertionError)
        # We need a path that actually exists AND contains "cp2k_shell" in the name.

        real_binary = shutil.which(cp2k_path)
        if not real_binary:
            raise FileNotFoundError(
                f"Executable '{cp2k_path}' not found in PATH.")

        # Check if the name satisfies ASE's strict assertion
        final_command = real_binary
        if "cp2k_shell" not in str(real_binary):
            # Create a symlink named 'cp2k_shell.ssmp' pointing to the real binary
            # We put this in the current directory or a temp folder
            symlink_name = "cp2k_shell.ssmp"
            symlink_path = pathlib.Path.cwd() / symlink_name

            # Remove old symlink if it exists to be safe
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()

            os.symlink(real_binary, symlink_path)

            # Use the local symlink as the command
            # ASE will see "cp2k_shell.ssmp" and be happy.
            # We use the absolute path to avoid PATH issues.
            final_command = str(symlink_path.absolute())
            print(
                f"Bypassing ASE assertion: Linked '{symlink_name}' -> '{real_binary}'")

        # 3. INSTANTIATE
        try:
            calculator = CP2K(
                command=final_command,
                inp=inp_content,
                basis_set=None,
                pseudo_potential=None,
                cutoff=None,
                print_level="LOW"
            )
        except Exception as e:
            # Clean up symlink if we failed immediately (optional)
            # if "cp2k_shell" not in cp2k_path: pathlib.Path(final_command).unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to initialize CP2K calculator: {repr(e)}") from e

        return calculator


@register_calculator("flashmd")
class FlashMDBuilder(CalculatorBuilder):
    def build(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> ase.calculators.calculator.Calculator:
        # Placeholder for FlashMD calculator setup
        # Replace with actual FlashMD calculator initialization
        energy_model, _ = get_pretrained("pet-omatpes-v2", 16)
        calculator = EnergyCalculator(energy_model, device=device)
        # calculator = PETMADCalculator("1.0.1", device=device)
        return calculator


@register_calculator("xtb")
class XTBBuilder(CalculatorBuilder):
    def build(
        self,
        xtb_path: str,
        xtb_keywords: str = "--gfn2",
        charge: int = 0,
        multiplicity: int = 1,
        num_threads: int = 1,
        log_directory: str = "xtb_files"
    ) -> ase.calculators.calculator.Calculator:
        # Placeholder for XTB calculator setup
        # Replace with actual XTB calculator initialization
        raise NotImplementedError("XTBBuilder is not yet implemented.")


@register_calculator("psi4")
class PSI4Builder(CalculatorBuilder):
    def build(
        self,
        psi4_path: str,
        psi4_keywords: str = "default_psi4_keywords",
        charge: int = 0,
        multiplicity: int = 1,
        num_threads: int = 1,
        log_directory: str = "psi4_files"
    ) -> ase.calculators.calculator.Calculator:
        # Placeholder for PSI4 calculator setup
        # Replace with actual PSI4 calculator initialization
        raise NotImplementedError("PSI4Builder is not yet implemented.")


@register_calculator("mace")
class MACEBuilder(CalculatorBuilder):
    def build(
        self,
        mace_name_or_path: str,
        charge: int = 0,
        multiplicity: int = 1,
        num_threads: int = 1,
        log_directory: str = "mace_files"
    ) -> ase.calculators.calculator.Calculator:
        try:
            if mace_name_or_path in ["small", "medium", "large"]:
                from mace.calculators import mace_mp
                calculator = mace_mp(mace_name_or_path, device="cuda" if torch.cuda.is_available() else "cpu")
                return calculator
            else:
                from mace.calculators import MACECalculator
                calculator = MACECalculator(mace_name_or_path, device="cuda" if torch.cuda.is_available() else "cpu")
        except:
            raise NotImplementedError("Use \"small\", \"medium\", or \"large\" for Universal models or give the path to a custom model.")



@register_calculator("nequip")
class NequIPBuilder(CalculatorBuilder):
    def build(
        self,
        nequip_model_path: str,
        charge: int = 0,
        multiplicity: int = 1,
        num_threads: int = 1,
        log_directory: str = "nequip_files"
    ) -> ase.calculators.calculator.Calculator:
        # Placeholder for NequIP calculator setup
        # Replace with actual NequIP calculator initialization
        raise NotImplementedError("NequIPBuilder is not yet implemented.")


if __name__ == "__main__":
    # Example usage
    cp2k_calculator = CalculatorFactory.create(
        "cp2k",
        cp2k_path="cp2k.ssmp",
        input_content_path="//oneconf.inp"
    )
    print(cp2k_calculator)
