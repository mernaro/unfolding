#!/bin/sh

## -- Name and log files
#SBATCH --job-name=Train-unfolding
#SBATCH --output=ML-%j/-Exp.out
#SBATCH --error=ML-%j/-Exp.err

## -- Param 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=RTX8000Nodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

## -- All path
export ENVIRONMENT_PATH=/projects/memaro/environement/bin/python
export SINGULARITY_PATH=/apps/containerCollections/CUDA12/pytorch2-NGC-24-02.sif
export PROJECT_PATH=/projects/memaro
export CONFIG_PATH=$PROJECT_PATH/rpujol/unfolding/config.json

## -- Config choice


## -- Launch the program
srun singularity exec $SINGULARITY_PATH $ENVIRONMENT_PATH $PROJECT_PATH/rpujol/unfolding/launcher.py \
    -c $CONFIG_PATH -a train