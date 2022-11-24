#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=5

#SBATCH --mem=60G
#SBATCH --partition=amdfast
#SBATCH --output=/home/vacekpa2/tmp/datagen_rad_%j.out
#SBATCH --error=/home/vacekpa2/tmp/datagen_rad_%j.out

ml scikit-image/0.18.3-foss-2021a
ml PyTorch/1.10.0-foss-2021a-CUDA-11.3.1
ml torchvision/0.11.3-foss-2021a-CUDA-11.3.1

cd $HOME/RandLA/

python -u data_prepare_semantickitti.py --src_path /mnt/personal/vacekpa2/data/semantic_kitti/dataset/sequences --dst_path /mnt/personal/vacekpa2/data/semantic_kitti/dataset/randla
