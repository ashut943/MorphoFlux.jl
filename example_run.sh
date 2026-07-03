#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -t 360
#SBATCH -G 1
#SBATCH -n 16

# Add Juliaup to PATH
export PATH=/home/at943/.juliaup/bin${PATH:+:${PATH}}

# Set number of Julia threads to match SLURM allocation
export JULIA_NUM_THREADS=16

# Print Julia version
julia --version

# Run the specific Julia file with the current environment
echo "Running nca_trial.jl..."
julia --project=. example_scripts/nca_trial.jl

echo "Script completed!"