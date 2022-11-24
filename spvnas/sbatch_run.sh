#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1     # processes per node
#SBATCH --cpus-per-task=12      # number of CPU cores per process
#SBATCH --partition=gpulong        # put the job into the gpu partition/queue
#SBATCH --gres=gpu:1
#SBATCH --output=log.out     # file name for stdout/stderr
#SBATCH --mem=50G              # how much CPU memory can be allocated for the job (hardware limit: 384 GB per node)
#SBATCH --time=3-00:00:00         # maximum wall time allocated for the job (max 24h for the gpu partition)
#SBATCH --job-name=spvnas        # job name (default is the name of this file)

cd $HOME/lidar/models/spvnas
#  poustej pres for loop s ampersandem a logovanim
module purge

ml torchsparse/1.4.0-foss-2021a-CUDA-11.3.1
ml Shapely

python -u train.py --config configs/semantic_kitti/spvcnn/default.yaml


