#!/bin/bash

rdir=/home/msujon/git/IU/msujon/SDMM
mdir=datasets/SuiteSparse
dset=N50k-100k 

dsets="N25k-N50k N50k-100k N100k-N200k"

resdir=results

#defualt values 
isMAll=0
m=256
d=128 
isPTtime=1
nrep=10
ialpha=2
ibeta=2

#commandline argument 

usage="Usage: $0 [OPTION] ...
Options: 
-d [val]    value of D (default 128)
-m [1/0]    is value of M all rows of A ? (default 0) 
-M [val]    value of M (default 256)
-p [1/0]    time parallel version (default parallel varsion)
-r [val]    number of repeatation (default 10) 
-a [0,1,2]  value of alpha: 0,0, 1.0, X (default X) 
-b [0,1,2]  value of beta: 0,0, 1.0, X (default X) 
--help      display help and exit
"


while getopts "d:m:M:p:r:" opt
do
   case $opt in 
      d) 
         d=$OPTARG
         ;;
      m)
         isMAll=$OPTARG
         ;;
      M)
         m=$OPTARG
         ;;
      p)
         isPTtime=$OPTARG
         ;;
      \?)
         echo "$usage"
         exit 1
         ;;
   esac
done


mkdir -p $resdir

#
#  enable parallel time, set pttime=0 if want to run sequential 
#
if [ $isPTtime -eq 1 ]
then
   par="_pt"
else
   par=""
fi
#
# select M
#
if [ $isMAll -eq 1 ]
then
   Mval=""  #by default all 
   Mstr="_allM"
else
   Mval="-M $m"  #by default all 
   Mstr="_M${m}"
fi

for dset in $dsets
do
   FILES=${mdir}/${dset}/*.mtx 
   res=${resdir}/${dset}${Mstr}_a${ialpha}b${ibeta}${par}.csv
   
   echo "FILENAME,NNZ,M,N,D,Trusted_inspect_time,trusted_exe_time,Test_inspect_time,Test_exe_time,Speedup_exe_time,Speedup_total_time,Critical_point" > $res
   
   for file in $FILES 
   do 
      #echo "$file"
      ./bin/CompAlgo${par} -input $file $Mval -D $d -nrep $nrep -skHd 1 \
         -ialpha $ialpha -ibeta $ibeta | tee -a $res 
   done
done

