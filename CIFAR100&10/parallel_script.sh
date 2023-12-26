#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH -p gpu
#SBATCH -A OPT_FOR_ML-TIFCYQKTZTK-DEFAULT-GPU
#SBATCH --job-name mem_lancher
#SBATCH --output /home/mostapha.essoullami/logs/mem_lancher%j.log
#SBATCH --error /home/mostapha.essoullami/logs/error_mem_lancher%j.log
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=86G

source ~/.bashrc
conda activate ddl

python adaptative_lancher.py
    