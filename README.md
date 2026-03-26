# GMD-26

Molecular dynamics toolkit built around ASE, with a practical CLI and optional API layer for computing energies and forces using multiple calculator backends (CP2K, ORCA, FlashMD, and others in progress).

The repository is designed for research workflows where you need to:

- generate molecules from SMILES or input files,
- run MD with configurable dynamics,
- cast/reference energies and forces onto trajectories,
- switch quantum/ML calculators through a factory interface,
- optionally expose energy/force comparison through a FastAPI service.

## What This Project Does

At a high level, `gmd_26` provides a modular pipeline:

1. Build molecules from SMILES or files.
2. Build a calculator backend (for example CP2K or FlashMD).
3. Build a dynamics model (for example Langevin variants).
4. Run MD in chunks with progress reporting.
5. Save logs and trajectories for post-processing.

The implementation is split into app-level commands and core factories:

- `src/gmd_26/app/main.py`: Typer CLI with commands to run MD and cast energies/forces.
- `src/gmd_26/app/functions.py`: runtime helpers (`run_dynamics`, logging setup, IO helpers).
- `src/gmd_26/app/api.py`: FastAPI endpoints for energy/force compute + error metrics.
- `src/gmd_26/core/calculator_factory.py`: pluggable calculator registry/builders.
- `src/gmd_26/core/md_factory.py`: pluggable dynamics registry/builders.
- `src/gmd_26/core/molecule_factory.py`: molecule construction from SMILES or file input.
- `src/gmd_26/core/molecule_tools.py`: geometry generation and boxing helpers.

## Current Calculator Support

### Implemented

- `cp2k`: ASE CP2K calculator wrapper with executable resolution and compatibility workaround for `cp2k_shell` naming.
- `orca`: ASE ORCA calculator builder with profile/template setup.
- `flashmd`: FlashMD pretrained model integration via ASE-compatible calculator.

### Declared But Not Implemented Yet

- `xtb` (factory builder placeholder raises `NotImplementedError`)
- `psi4` (factory builder placeholder raises `NotImplementedError`)
- `mace` (factory builder placeholder raises `NotImplementedError`)
- `nequip` (factory builder placeholder raises `NotImplementedError`)

Note: There are additional class-based prototypes under `src/gmd_26/core/xtb`, `src/gmd_26/core/psi4`, `src/gmd_26/core/orca`, and `src/gmd_26/core/flashmd`, but the CLI currently uses the factory registration path above.

## Requirements

- Linux (recommended in current setup)
- Python 3.13+
- Conda-forge ecosystem (for CP2K, xtb, RDKit, etc.)
- Optional external executables depending on calculator choice:
	- CP2K (`cp2k_shell*` available in PATH or provided binary)
	- ORCA binary path when using ORCA backend

## Installation and Running With Pixi

This is the recommended path for reproducible environment setup.

### 1. Install Pixi

If Pixi is not installed yet:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Then restart your shell (or source your shell profile) so `pixi` is available.

### 2. Create environment

From repository root:

```bash
pixi install
```

This resolves dependencies from `pyproject.toml` (`[tool.pixi.*]`) and installs this project in editable mode.

### 3. Run the project

Use configured Pixi tasks:

```bash
# Run a single-molecule MD example (FlashMD backend)
pixi run do_md

# Run MD from a molecule file
pixi run do_md_from_file

# Start app module task (defined in pyproject)
pixi run start_app
```

Or run commands directly inside the Pixi environment:

```bash
pixi run python -m gmd_26.app.main --help
```

## CLI Usage

Primary CLI module:

```bash
python -m gmd_26.app.main --help
```

### Command: `single-molecule-md`

Runs molecular dynamics for one molecule.

Example (SMILES input):

```bash
python -m gmd_26.app.main single-molecule-md \
	-c cp2k \
	-mgm from_smiles \
	-smi "CC(=O)O" \
	-dyn langevin_ase \
	-ns 1000 \
	-nt 1 \
	-es 10 \
	-t 300 \
	-lf md_log.txt \
	-tf trajectory.traj
```

Example (molecule file input):

```bash
python -m gmd_26.app.main single-molecule-md \
	-c flashmd \
	-mgm from_file \
	-mp Ge8.xyz \
	-dyn langevin_flashmd \
	-ns 1000
```

Common outputs:

- MD log file (for example `md_log.txt`)
- trajectory file (for example `trajectory.traj`)

### Command: `cast-energy-and-forces`

Reads molecules/frames, applies selected calculator, computes energy/forces/stress, and writes output.

Example:

```bash
python -m gmd_26.app.main cast-energy-and-forces \
	-ip input.traj \
	-op casted_output.traj \
	-c cp2k \
	--input-content-path oneconf.inp
```

Extra unknown options are accepted and forwarded to the selected calculator builder (key-value style).

## API Usage (FastAPI)

Start API server:

```bash
uvicorn gmd_26.app.api:app --host 0.0.0.0 --port 8000
```

### `POST /compute_e_f`

Computes energy and forces for an XYZ string and compares them to reference values.

Request body fields:

- `xyz_string`: full XYZ content as string
- `energy`: reference energy
- `forces`: reference force matrix (`N x 3`)
- `calculator_type`: backend name (default `xtb` in API model)
- `calculator_kwargs`: optional calculator options

Response includes:

- `computed_energy`
- `computed_forces`
- `energy_error` (absolute error)
- `forces_error` (RMSE)
- `max_force_error`

### `POST /scc`

Simple health/sanity endpoint.

## CP2K Input Notes

Example CP2K input template is provided in `oneconf.inp`.

When using CP2K from the CLI, pass the path as an extra forwarded argument:

```bash
python -m gmd_26.app.main cast-energy-and-forces \
	-ip input.traj \
	-op output.traj \
	-c cp2k \
	--input-content-path oneconf.inp
```

The CP2K builder performs:

- input file loading,
- binary resolution with `shutil.which`,
- automatic symlink workaround when executable name does not contain `cp2k_shell`.

## Development

### Project layout

```text
src/
	gmd_26/
		app/
			api.py
			functions.py
			main.py
		core/
			calculator_factory.py
			md_factory.py
			molecule_factory.py
			molecule_tools.py
```

### Packaging

- Build metadata lives in `pyproject.toml`.
- Package is installed editable in Pixi via `[tool.pixi.pypi-dependencies]`.

## Known Caveats

- Some declared calculators are placeholders in factory mode and will raise `NotImplementedError`.
- CP2K and ORCA execution depends on local binaries and runtime environment.
- The root `main.py` is currently not the main entrypoint; use `python -m gmd_26.app.main`.

## License

See `LICENSE`.
