#!/bin/bash
#SBATCH -J newjob       # job name
#SBATCH -o newjob.o%j       # output and error file name (%j expands to jobID)
#SBATCH -n 1 -N 1           
#SBATCH -p gpu
#SBATCH -t 24:00:00         # run time (hh:mm:ss)
#SBATCH --mem-per-cpu=3000

module load tensorflow
module load cuda-toolkit/8.0
python  TrainingCode.py