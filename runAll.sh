#!/bin/bash

rdir=/home/msujon/git/IU/msujon/SDMM
mdir=datasets/SuiteSparse
dset=N50k-100k 

dsets="N25k-N50k N50k-100k N100k-N200k"

#FILES=${rdir}/${mdir}/${dset}/*.mtx 
#FILES=${mdir}/${dset}/*.mtx 


resdir=results
mkdir $resdir


nrep=4 
m=256
d=128 


mkdir -p $resdir

#
#  enable parallel time, set pttime=0 if want to run sequential 
#

pttime=1

if [ $pttime -eq 1 ]
then
   par="_pt"
else
   par=""
fi

for dset in $dsets
do
   FILES=${mdir}/${dset}/*.mtx 
   res=${resdir}/${dset}${par}.csv
   
   echo "FILENAME,NNZ,M,N,D,Trusted_inspect_time,trusted_exe_time,Test_inspect_time,Test_exe_time,Speedup_exe_time,Speedup_total_time,Critical_point" > $res
   
   for file in $FILES 
   do 
      #echo "$file"
      ./bin/CompAlgo${par} -input $file -M $m -D $d -nrep 4 -skHd 1 | tee -a $res 
   done
done

