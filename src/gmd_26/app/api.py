from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import ase
from ase.calculators.singlepoint import SinglePointCalculator
from ase import io as atoms_io
from io import StringIO
from gmd_26.core.calculator_factory import CalculatorFactory
from gmd_26.app.functions import get_energy_and_forces

app = FastAPI()


def parse_xyz_string(xyz_string: str) -> ase.Atoms:
    """
    Parse an XYZ format string into an ASE Atoms object.
    
    XYZ format:
        <number_of_atoms>
        <comment line>
        <symbol> <x> <y> <z>
        <symbol> <x> <y> <z>
        ...
    
    Args:
        xyz_string: XYZ format as a string
        
    Returns:
        ASE Atoms object
    """
    try:
        # Use ASE's read function with StringIO to parse the XYZ string
        atoms = atoms_io.read(StringIO(xyz_string), format='xyz')
        return atoms
    except Exception as e:
        raise ValueError(f"Failed to parse XYZ string: {str(e)}")


class ComputeRequest(BaseModel):
    """Request model for energy and forces computation"""
    xyz_string: str  # XYZ format as a string
    energy: float  # Reference energy from client
    forces: list[list[float]]  # Reference forces (N, 3)
    calculator_type: str = "xtb"  # Calculator to use (xtb, orca, cp2k, etc.)
    calculator_kwargs: dict = {}  # Additional kwargs for calculator


class ComputeResponse(BaseModel):
    """Response model with computed values and errors"""
    computed_energy: float
    computed_forces: list[list[float]]
    energy_error: float  # MAE (Mean Absolute Error) in eV
    forces_error: float  # RMSE (Root Mean Square Error) in eV/Å
    max_force_error: float  # Maximum absolute error in any force component


@app.post("/compute_e_f",
          response_model=ComputeResponse,
          summary="Compute energy and forces for a given molecule")
def compute_energy_and_forces(request: ComputeRequest):
    """
    Compute energy and forces using a specified calculator and compare with reference values.
    
    Args:
        request: ComputeRequest with molecular structure (as XYZ string) and reference values
        
    Returns:
        ComputeResponse with computed values and error metrics
    """
    try:
        # 1. Parse XYZ string to create ASE Atoms object
        try:
            atoms = parse_xyz_string(request.xyz_string)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid XYZ format: {str(e)}")
        
        # 2. Create and set calculator
        try:
            calculator = CalculatorFactory.create(
                request.calculator_type,
                **request.calculator_kwargs
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid calculator: {str(e)}")
        
        atoms.calc = calculator
        
        # 3. Get computed energy and forces
        computed_energy = atoms.get_potential_energy()
        computed_forces = atoms.get_forces()
        
        # 4. Convert reference values to numpy arrays for calculation
        ref_energy = np.array(request.energy)
        ref_forces = np.array(request.forces)
        computed_forces_array = np.array(computed_forces)
        
        # 5. Calculate error metrics
        energy_error = float(np.abs(computed_energy - ref_energy))
        
        forces_diff = computed_forces_array - ref_forces
        forces_error = float(np.sqrt(np.mean(forces_diff**2)))  # RMSE
        max_force_error = float(np.max(np.abs(forces_diff)))
        
        # 6. Prepare response
        return ComputeResponse(
            computed_energy=float(computed_energy),
            computed_forces=computed_forces.tolist(),
            energy_error=energy_error,
            forces_error=forces_error,
            max_force_error=max_force_error
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error computing energy and forces: {str(e)}"
        )
    
@app.post("/scc",
          summary="Check if the API is reachable")
def sanity_check_connection():
    return {"message": "API is reachable"}