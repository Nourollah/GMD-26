from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import MDLogger
from ase import io as atoms_io
import ase
import inspect
import typing
from ase.calculators.singlepoint import SinglePointCalculator
from xtb.libxtb import new_molecule


def run_dynamics(
    dynamics: object,
    number_of_steps: int,
    each_step_size: int,
    number_of_turns: int,
    restart_thermostat_per_turn: bool = False,
    temperature: float = 300,
    *kwargs,
):
    total_steps = number_of_turns * number_of_steps

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            "[cyan]Running dynamics...",
            total=total_steps
        )

        for turn in range(number_of_turns):
            steps_completed = 0
            while steps_completed < number_of_steps:
                steps_to_run = min(
                    each_step_size, number_of_steps - steps_completed)
                dynamics.run(steps_to_run)
                steps_completed += steps_to_run
                progress.update(task, advance=steps_to_run)

            if restart_thermostat_per_turn:
                MaxwellBoltzmannDistribution(
                    atoms=dynamics.atoms,
                    temperature_K=temperature
                )


def md_setup_logger(
    dynamics: object,
    molecule: ase.Atoms,
    md_logger_file: str | None = None,
    md_logger_interval: int = 10,
    md_logger_header: bool = True,
    md_logger_stress: bool = True,
    md_logger_per_atom: bool = True,
    md_logger_mode: str = "a",
    trajectory_path_file: str | None = None,
    trajectory_interval: int = 10,
    trajectory_logger_mode: str = "w",
    **kwargs
) -> None:
    if trajectory_path_file:  # It won't accept empty strings!
        trajectory_params = set(inspect.signature(
            atoms_io.Trajectory).parameters.keys())
        trajectory_kwargs = {k: v for k,
                             v in kwargs.items() if k in trajectory_params}

        trajectory_logger = atoms_io.Trajectory(
            filename=trajectory_path_file,
            mode=trajectory_logger_mode,
            atoms=molecule
        )
        dynamics.attach(
            trajectory_logger.write,
            interval=trajectory_interval
        )

    if md_logger_file:
        md_logger_params = set(inspect.signature(MDLogger).parameters.keys())
        md_logger_kwargs = {k: v for k,
                            v in kwargs.items() if k in md_logger_params}

        md_logger = MDLogger(
            dynamics,
            dynamics.atoms,
            logfile=md_logger_file,
            header=md_logger_header,
            stress=md_logger_stress,
            peratom=md_logger_per_atom,
            mode=md_logger_mode,
            **md_logger_kwargs
        )
        print("MD Logger set up successfully.")
        dynamics.attach(
            md_logger,
            interval=md_logger_interval
        )


def get_energy_and_forces(
    molecule: ase.Atoms | typing.List[ase.Atoms],
    calculator: ase.calculators.calculator.Calculator,
) -> typing.List[ase.Atoms] | ase.Atoms:
    # Check if the molecule is a list or just a single molecule
    if isinstance(molecule, list):
        # Make a progress bar with rich to track the progress of the calculation
        new_molecule_list: typing.List[ase.Atoms] = []
        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Calculating energy and forces...", total=len(molecule))
            for i, mol in enumerate(molecule):
                mol.calc = calculator
                energy = mol.get_potential_energy()
                forces = mol.get_forces()
                stress = mol.get_stress()
                mol.calc = SinglePointCalculator(
                    mol,
                    energy=energy,
                    forces=forces,
                    stress=stress
                )
                new_molecule_list.append(mol)
                progress.update(task, advance=1)
        return new_molecule_list
    elif isinstance(molecule, ase.Atoms):
        molecule.calc = calculator
        energy = molecule.get_potential_energy()
        forces = molecule.get_forces()
        stress = molecule.get_stress()
        molecule.calc = SinglePointCalculator(
            molecule,
            energy=energy,
            forces=forces,
            stress=stress
        )
        return molecule
    else:
        raise ValueError("Molecule must be a list or a single molecule.")


def write_output_file(
    molecule: typing.List[ase.Atoms],
    output_file: str,
    format: str = "traj"
) -> None:
    atoms_io.write(molecule, output_file, format=format)
