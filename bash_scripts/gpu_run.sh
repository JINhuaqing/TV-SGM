#!/bin/bash
#SBATCH --mem=200gb                          # Job memory request
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --partition=gpu                    # Run on partition "dgx" (e.g. not the default partition called "long")
#SBATCH --gres=gpu:1   # Allocate 1 GPU resource for this job.
#SBATCH --output=sgmnet-%x.out
#SBATCH -J large


echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"


python -u /data/rajlab1/user_data/jin/MyResearch/TV-SGM/python_scripts/SGM_approx.py
