#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8

#SBATCH --mem=60G
#SBATCH --partition=amdlong
#SBATCH --output=logs/ego_preannotation_%j.out
#SBATCH --error=logs/ego_preannotation_%j.out


ml torchsparse

cd $HOME/

python -u timespace/scripts/run_ego_annotation.py
