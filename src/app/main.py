import glob
import os
import pathlib
import typing

import pandas as pd
import ase

import typer
from ase.calculators.singlepoint import SinglePointCalculator

from functions import run_dynamics, md_setup_logger
from src.app.functions import get_energy_and_forces, write_output_file
from src.core.molecule_tools import box_molecule
from src.core.molecule_factory import MoleculeFactory
from src.core.calculator_factory import CalculatorFactory
from src.core.md_factory import DynamicsFactory

app = typer.Typer(
	help="Molecular Dynamics Simulation CLI",
	no_args_is_help=True,
)


@app.command()
def single_molecule_md(
		calculator: str = typer.Option(
			"cp2k",
			"--calculator",
			"-c",
			case_sensitive=False,
			metavar="CALCULATOR",
			help="The calculator to use for the simulation.",
			click_type=typer.Choice([
				"xtb",
				"orca",
				"cp2k",
				"flashmd",
				"mace",
				"nequip",
				"psi4"
			])
		),  # @TODO: adding optional arguments for each calculator
		molecule_smiles: str | None = typer.Option(
			"CC(=O)O",
			"--molecule-smiles",
			"-smi",
			help="The SMILES string of the molecule to simulate."
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
		"from_smiles",
		smiles=molecule_smiles
	)
	box_molecule(
		molecule,
		periodic_boundary_conditions=periodic_boundary_conditions
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
		md_logger_file=md_logger_file,
		md_logger_interval=md_logger_interval,
		md_trajectory_file=md_trajectory_file,
		md_trajectory_interval=md_trajectory_interval,
	)
	run_dynamics(
		dynamics=dynamics,
		number_of_steps=number_of_steps,
		each_step_size=each_step_size,
		number_of_turns=number_of_turns,
		restart_thermostat_per_turn=restart_thermostat_per_turn,
		temperature=temperature,
	)


@app.command()
def cast_energy_and_forces(
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
			case_sensitive=False,
			metavar="CALCULATOR",
			help="The calculator to use for the simulation.",
			click_type=typer.Choice([
				"xtb",
				"orca",
				"cp2k",
				"flashmd",
				"mace",
				"nequip",
				"psi4"
			])
		)
):
	molecules: typing.List[ase.Atoms] = MoleculeFactory(
		"from_file",
		file_path=input_path,
	)
	new_calculator = CalculatorFactory(
		calculator,
	)
	casted_molecules: typing.List[ase.Atoms] = get_energy_and_forces(
		molecules,
		new_calculator
	)
	write_output_file(
		casted_molecules,
		output_file=output_path
	)
