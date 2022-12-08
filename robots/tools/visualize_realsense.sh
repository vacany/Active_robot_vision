#!/bin/bash
#SBATCH --time=4:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4

#SBATCH --mem=30G
#SBATCH --partition=amdfast
#SBATCH --error=logs/%j.out
#SBATCH --output=logs/%j.out

ml OpenCV/4.5.3-foss-2021a-contrib
cd $HOME

python -u data_utils/rgbd.py

