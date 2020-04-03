#!/bin/bash

rdir=/home/msujon/git/IU/msujon/SDMM
mdir=datasets/SuiteSparse
dset=N50k-100k 

dsets="N25k-N50k N50k-100k N100k-N200k"

#FILES=${rdir}/${mdir}/${dset}/*.mtx 
#FILES=${mdir}/${dset}/*.mtx 


resdir=results

nrep=4 
m=256
d=128 



for dset in $dsets
do
   FILES=${mdir}/${dset}/*.mtx 
   res=${resdir}/${dset}.csv
   
   echo "FILENAME, NNZ, M, N, D, t0, t1, speedup" > $res
   
   for file in $FILES 
   do 
      #echo "$file"
      ./bin/CompAlgo -input $file -M $m -D $d -nrep 4 | tee -a $res 
   done
done

