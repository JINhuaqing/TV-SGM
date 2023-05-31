#!/bin/bash
#SBATCH --mem=200gb                          # Job memory request
#SBATCH --partition=long                    # Run on partition "dgx" (e.g. not the default partition called "long")
#SBATCH --output=logs/LSTM-%x.out
#SBATCH -J wmse_geialpha_stat
#SBATCH --ntasks=30


echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"

#python -u /data/rajlab1/user_data/jin/MyResearch/TV-SGM/python_scripts/real_data_demo.py

singularity exec ~/jin/singularity_containers/tvsgm_latest.sif python -u /data/rajlab1/user_data/jin/MyResearch/TV-SGM/python_scripts/real_data_demo.py

