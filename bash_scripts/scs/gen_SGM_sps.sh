#!/bin/bash
#SBATCH --mem=200gb                          # Job memory request
#SBATCH --partition=long                    # Run on partition "dgx" (e.g. not the default partition called "long")
#SBATCH --output=logs/EYE-%x.out
#SBATCH -J gen_SGM
#SBATCH --ntasks=30


echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"


singularity exec ~/jin/singularity_containers/tvsgm_latest.sif python -u /data/rajlab1/user_data/jin/MyResearch/TV-SGM/python_scripts/gen_SGM_sps_eye_close.py

