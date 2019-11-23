#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=nonlinear_kappa
#SBATCH --account=coen
#SBATCH --output=nonlinear_kappa-%j-%N.out
#SBATCH --error=nonlinear_kappa-%j-%N.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mail-type=END
#SBATCH --mail-user=strenge@control.tu-berlin.de

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

module load julia/1.1.0
module load hpc/2015
julia sim_nonlinear_kappa.jl $SLURM_NTASKS
