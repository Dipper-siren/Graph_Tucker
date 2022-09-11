#!/bin/bash
#SBATCH -N 2
#SBATCH --partition=batch
#SBATCH -J MyJob
#SBATCH -o MyJob.%J.out
#SBATCH -e MyJob.%J.err
#SBATCH --mail-user=424233015@qq.com
#SBATCH --mail-type=ALL
#SBATCH --time=15:30:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --constraint=[gpu]

#run the application: 
srun run_fb15k.sh
