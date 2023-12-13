#!/bin/bash
#SBATCH --mem=200gb                          # Job memory request
#SBATCH --partition=long                    # Run on partition "dgx" (e.g. not the default partition called "long")
#SBATCH --output=scs/logs/Logi-ctrl-%x-%j.out
#SBATCH -J wmse_1111110
#SBATCH --cpus-per-task=30
#SBATCH --time=48:00:00
#SBATCH --chdir=/home/hujin/jin/MyResearch/TV-SGM/bash_scripts/
####SBATCH --ntasks=30

echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"

singularity exec ~/jin/singularity_containers/tvsgm_latest.sif python -u /data/rajlab1/user_data/jin/MyResearch/TV-SGM/python_scripts/real_data_demo_ctrl.py --loss wmse
