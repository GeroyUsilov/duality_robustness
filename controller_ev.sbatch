#!/bin/bash
#SBATCH --job-name=controller_ev
#SBATCH --output=filter_sbatch.out
#SBATCH --error=filter_sbatch.err
#SBATCH --account=pi-amurugan
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10000

module load python

./controller_ev.sh $1