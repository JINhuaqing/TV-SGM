#!/bin/bash
#### The job script, run it as qsub xxx.sh 

#### the shell language when run via the job scheduler [IMPORTANT]
#$ -S /bin/bash
#### job should run in the current working directory
#$ -cwd
#### Specify job name
#$ -N wmse_geitaui_stat
#### Output file
#$ -o logs/LSTM-$JOB_NAME_$JOB_ID.out
#### Error file
#$ -e logs/LSTM-$JOB_NAME_$JOB_ID.err
#### memory per core
#$ -l mem_free=8G
#### number of cores 
#$ -pe smp 30
#### Maximum run time 
#$ -l h_rt=20:00:00
#### job requires up to 2 GB local space
#$ -l scratch=2G
#### Specify queue
###  gpu.q for using gpu
###  if not gpu.q, do not need to specify it
### #$ -q gpu.q 
#### The GPU memory required, in MiB
### #$ -l gpu_mem=12000M

singularity exec ~/MyResearch/tvsgm_latest.sif python ../python_scripts/real_data_demo.py

[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
