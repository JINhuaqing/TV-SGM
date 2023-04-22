#!/bin/bash
#SBATCH --mem=200gb                          # Job memory request
#SBATCH --partition=long                    # Run on partition "dgx" (e.g. not the default partition called "long")
#SBATCH --output=R20-%x.out
#SBATCH -J data
#SBATCH --ntasks=30


echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"

python -u /data/rajlab1/user_data/jin/MyResearch/TV-SGM/python_scripts/SGM_approx.py

