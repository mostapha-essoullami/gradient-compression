#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH -p gpu
#SBATCH -A OPT_FOR_ML-TIFCYQKTZTK-DEFAULT-GPU
#SBATCH --job-name test_alloc
#SBATCH --output /home/mostapha.essoullami/logs/unet%j.log
#SBATCH --error /home/mostapha.essoullami/logs/er_unet%j.log
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=328G
#SBATCH --export=ALL

source ~/.bashrc
conda activate ddl
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# python dawn.py
sleep 1000000
    