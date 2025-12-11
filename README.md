# GMD-25

A Python-based molecular dynamics calculation tool for computational chemistry simulations.

## Overview

GMD-25 is designed for calculating molecular dynamics simulations, providing computational chemistry researchers with tools to analyze molecular behavior and properties. The project supports both FlashMD (default) and GFN2-xTB calculation methods.

## Features

- Molecular dynamics calculations using Python
- Support for SMILES notation input
- FlashMD integration (default method)
- GFN2-xTB recalculation capabilities via `xib_compute`
- Configurable task variables through pixi.toml

## Prerequisites

- [Pixi](https://pixi.sh/) - Modern package management solution

## Installation

This project uses Pixi for dependency management. To install:

```bash
pixi install
```

## Usage

### 1. Prepare Your Molecules

Update the `smiles.txt` file with the molecules you want to analyze using SMILES notation.

### 2. Configure Task Variables

Edit the task variables in `pixi.toml` to point to your input file:

```toml
# Update the file path in pixi.toml to match your smiles.txt location
```

### 3. Run Calculations

#### Using FlashMD (Default)
The project runs with FlashMD by default after following steps 1 and 2.

#### Using GFN2-xTB Method
If you prefer to use GFN2-xTB from the beginning, configure this in your settings.

Alternatively, if you already have MD files from FlashMD:
1. Place your MD files in a directory
2. Use `xib_compute` to recalculate them using GFN2-xTB:

```bash
# Run xib_compute on your MD files directory
```

## Project Structure

```
GMD-25/
├── smiles.txt          # Input file for molecule SMILES notation
├── pixi.toml          # Pixi configuration and task variables
├── pixi.lock          # Pixi lock file
└── ...                # Additional project files
```

## Methods

- **FlashMD**: Default molecular dynamics calculation method
- **GFN2-xTB**: Alternative semi-empirical quantum chemistry method available through `xib_compute`

## Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.


*For questions or support, please open an issue in this repository.*
