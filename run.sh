#!/bin/bash --login
#BATCH --exclusive              # exclusive node access
#SBATCH -p compute                   # selected queue (This is the standard compute queue, dev can also be used to test with a short walltime e.g. 5 minutes)
# standard error file name (%J gives the job number)
#SBATCH --error=error.%J
#SBATCH --ntasks-per-node=40     # tasks to run per node (40 for hawk)
#SBATCH --ntasks=120             # number of parallel processes (tasks) (80 would call 2 nodes, 120 3x nodes)
#SBATCH --time=2-10:00:00          # time limit
# Job name
#SBATCH -J #[job name]# 
#SBATCH --mail-user=nourollaha@cardiff.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -A scw2243    # project code definition

## Usage
#      sbatch runVASPhawk.sh        # submit job
#      squeue -u $USER               # view job status
#      scancel [job number]		# Cancel job using job number

# If a job hits the walltime then the partially complete job can be found in /scratch/[username]/[job number]/


#SBATCH --job-name=molecular_data_generation_%J
 
#SBATCH --output=SBMD/output%J.txt
#SBATCH --error=SBMDerror%J.txt
 
#SBATCH --mem=128G

# remove all default modules
#	module purge
#	module load python/3.10.4


# directory to run the job using /scratch/$USER
	WDPATH=/scratch/$USER/$SLURM_JOBID
	rm -rf ${WDPATH} ; mkdir -p ${WDPATH}

# Running task
	cd /scratch/c.c23125717/SBMD
	source /scratch/c.c23125717/miniconda3/bin/activate
	conda activate SBMD

	python3 MolecularDynamics.py \
		--smiles-file smiles.txt \
		--parent-dir "Alkanes" \
		--logging-interval 1 \
		--run-steps 100000 \
		--segments 2 \
		--engine "FlashMD" \
		--skip-frames 16
		
