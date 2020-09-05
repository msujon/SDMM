#!/bin/bash 

#SBATCH -N 1
#SBATCH -p azad 
#SBATCH -t 30:30:00
#SBATCH -J sdmmtime
#SBATCH -o sdmmtime.o%j 

module load gcc

export OMP_PLACES=cores
export OMP_PROC_BIND=close


srun -p azad -N 1 -n 1 -c 1 bash runAll.sh 
