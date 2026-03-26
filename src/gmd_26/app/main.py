import typing

import ase

import typer

from gmd_26.app.functions import run_dynamics, md_setup_logger
from gmd_26.app.functions import get_energy_and_forces, write_output_file
from gmd_26.core.molecule_tools import box_molecule
from gmd_26.core.molecule_factory import MoleculeFactory
from gmd_26.core.calculator_factory import CalculatorFactory
from gmd_26.core.md_factory import DynamicsFactory

app = typer.Typer(
    help="Molecular Dynamics Simulation CLI",
    no_args_is_help=True,
)


def validate_calculator(value: str) -> str:
    valid_choices = ["xtb", "orca", "cp2k",
                     "flashmd", "mace", "nequip", "psi4"]
    if value.lower() not in valid_choices:
        raise typer.BadParameter(
            f"Invalid calculator. Choose from: {', '.join(valid_choices)}")
    return value.lower()


def sanity_check_calculator(calculator: ase.calculators.calculator.Calculator) -> None:
    """
    Sanity check for calculator by creating a tiny H2 molecule and computing energy and forces.
    Raises an exception if the calculator fails to compute.
    """
    # Create a tiny H2 molecule
    test_molecule = ase.Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    test_molecule.calc = calculator

    try:
        # Attempt to get energy and forces
        energy = test_molecule.get_potential_energy()
        forces = test_molecule.get_forces()
        typer.echo(f"Calculator sanity check passed: E={energy:.4f} eV")
    except Exception as e:
        raise RuntimeError(f"Calculator sanity check failed: {str(e)}")


@app.command()
def single_molecule_md(
    calculator: str = typer.Option(
        "cp2k",
        "--calculator",
        "-c",
        callback=validate_calculator,
        help="The calculator to use for the simulation. Choices: xtb, orca, cp2k, flashmd, mace, nequip, psi4",
    ),  # @TODO: adding optional arguments for each calculator
    molecule_generation_method: str = typer.Option(
        "from_smiles",
        "--molecule-generation-method",
        "-mgm",
        help="The method to use for generating the molecule. Choices: from_smiles, from_file",
    ),
    molecule_smiles: str | None = typer.Option(
        "CC(=O)O",
        "--molecule-smiles",
        "-smi",
        help="The SMILES string of the molecule to simulate."
    ),
        molecule_path: str | None = typer.Option(
        "Ge8.xyz",
        "--molecule-path",
        "-mp",
        help="The path to the molecule file to simulate."
    ),
    periodic_boundary_conditions: bool = typer.Option(
        True,
        "--periodic-boundary-conditions",
        "-pbc",
        help="Whether to use periodic boundary conditions."
    ),
    dynamical_model: str = typer.Option(
        "langevin",
        "--dynamical-model",
        "-dyn",
        help="The dynamical model to use for the simulation.",
    ),
    number_of_steps: int = typer.Option(
        1000,
        "--number-of-steps",
        "-ns",
        help="How many step to run the dynamics model for in each turn."
    ),
    number_of_turns: int = typer.Option(
        1,
        "--number-of-turns",
        "-nt",
        help="How many set to run dynamics model for."
    ),
    each_step_size: int = typer.Option(
        10,
        "--each-step-size",
        "-es",
        help="Integrity of running dynamics model for number of steps."
    ),
    temperature: float = typer.Option(
        300,
        "--temperature",
        "-t",
        help="The temperature of the simulation in Kelvin."
    ),
    restart_thermostat_per_turn: bool = typer.Option(
        False,
        "--restart-thermostat-per-turn",
        "-rst",
        help="Whether to restart the thermostat per turn."
    ),
    md_logger_file: str = typer.Option(
        "md_log.txt",
        "--md-logger-file",
        "-lf",
        help="The file to write the MD log to."
    ),
    md_logger_interval: int = typer.Option(
        10,
        "--md-logger-interval",
        "-li",
        help="Interval for logging during regular segments"
    ),
    md_trajectory_file: str = typer.Option(
        "trajectory.traj",
        "--md-trajectory-file",
        "-tf",
        help="The file to write the MD trajectory to."
    ),
    md_trajectory_interval: int = typer.Option(
        10,
        "--md-trajectory-interval",
        "-ti",
        help="Interval for writing trajectory during regular segments"
    ),
):
    molecule = MoleculeFactory.create(
        molecule_generation_method,
        smiles=molecule_smiles if molecule_generation_method == "from_smiles" else molecule_path
    )
    if molecule_generation_method == "from_smiles":
        box_molecule(
            molecule,
            pbc=periodic_boundary_conditions
        )
    molecule.calc = CalculatorFactory.create(
        calculator
    )
    dynamics = DynamicsFactory.create(
        dynamical_model,
        molecule=molecule,
    )
    # Set logger for dynamics
    md_setup_logger(
        dynamics=dynamics,
        molecule=molecule,
        md_logger_file=md_logger_file,
        md_logger_interval=md_logger_interval,
        trajectory_path_file=md_trajectory_file,
        trajectory_interval=md_trajectory_interval,
    )
    run_dynamics(
        dynamics=dynamics,
        number_of_steps=number_of_steps,
        each_step_size=each_step_size,
        number_of_turns=number_of_turns,
        restart_thermostat_per_turn=restart_thermostat_per_turn,
        temperature=temperature,
    )


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cast_energy_and_forces(
    ctx: typer.Context,
    input_path: str = typer.Option(
        "",
        "--input-path",
        "-ip",
        help="Path to file containing trajectory/molecules to process"
    ),
    output_path: str = typer.Option(
        "",
        "--output-path",
        "-op",
        help="Path to directory to write output files to"
    ),
    calculator: str = typer.Option(
        "cp2k",
        "--calculator",
        "-c",
        callback=validate_calculator,
        help="The calculator to use for the simulation. Choices: xtb, orca, cp2k, flashmd, mace, nequip, psi4",
    ),
):
    # Parse extra arguments from context
    extra_args = {}
    if ctx.args:
        for i in range(0, len(ctx.args), 2):
            key = ctx.args[i].lstrip('--').replace('-', '_')
            value = ctx.args[i + 1] if i + 1 < len(ctx.args) else None
            extra_args[key] = value

    molecules: typing.List[ase.Atoms] = MoleculeFactory.create(
        "from_file",
        file_path=input_path,
    )
    new_calculator = CalculatorFactory.create(
        calculator,
        **extra_args
    )

    # Sanity check the calculator
    sanity_check_calculator(new_calculator)

    casted_molecules: typing.List[ase.Atoms] = get_energy_and_forces(
        molecules,
        new_calculator
    )
    write_output_file(
        casted_molecules,
        output_file=output_path
    )


if __name__ == "__main__":
    app()
