#!/bin/bash
#### sbatch scs_sub.sh to submit the job

#### Job memory request
#SBATCH --mem=200gb                  
#SBATCH --nodes=1
##SBATCH --nodelist=concord
#### Num of cores required, I think I should use --cpus-per-task other than --ntasks
#SBATCH --cpus-per-task=25
#### Run on partition "dgx" (e.g. not the default partition called "long")
### long for CPU, gpu/dgx for CPU, dgx is slow
#SBATCH --partition=gpu
#### Allocate 1 GPU resource for this job. 
##SBATCH --gres=gpu:teslav100:1   
#SBATCH --output=logs/gen-sgm-pairs-%x-%j.out
#SBATCH -J run
#SBATCH --chdir=/home/hujin/jin/MyResearch/TV-SGM_dev/bash_scripts/scs/

#### You job
echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"

source /netopt/rhel7/versions/python/Anaconda3-edge/etc/profile.d/conda.sh
module load SCS/anaconda/anaconda3
conda activate TVSGM

python -u ../../python_scripts/gen_SGM_pairs.py
