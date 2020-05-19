#!/bin/bash

#
# change the directory address accordingly
#     rdir = project directory 
#     mdir = dataset directory 
#
rdir=./
#mdir=../dataset/SuiteSparse/formated
mdir=../../dataset/SuiteSparse/formated
resdir=results

#rdir=/home/msujon/git/IU/msujon/SDMM
#rdir=/home/msujon/git/IU/msujon/Timing/SDMM 
#mdir=/home/msujon/git/IU/msujon/dataset/SuiteSparse/formated

#dsets="N1500k-2M"
#dsets="N1M-1500k"
#dsets="N200k-500k"
#dsets="N100k-N200k"
#dsets="N25k-N50k N50k-100k N100k-N200k N200k-500k N500k-1M N1M-1500k"

#dsets="N100k-N200k-FM N200k-500k-FM N500k-1M-FM N1M-1500k-FM N1500k-2M-FM N2M-3M-FM N3M-4M-FM N4M-5M-FM N5M-N6M N7M-10M"

dsets="N100k-200k-FM N200k-500k-FM N500k-1M-FM N1M-1500k-FM N1500k-2M-FM N2M-3M-FM N3M-4M-FM N4M-5M-FM N5M-6M-FM N7M-10M-FM N10M-15M-FM N15M-20M-FM N20M-30M-FM"

#dsets="N30M-40M-FM"
#dsets="N40M-50M-FM"
#dsets="N50M-60M-FM"
#dsets="N60M-70M-FM"


#defualt values 
isMAll=0
m=256
d=128 
isPTtime=1
nrep=10
nblk=10
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
-k [val]    number of random blocks when M < rows (default 10) 
-a [0,1,2]  value of alpha: 0,0, 1.0, X (default X) 
-b [0,1,2]  value of beta: 0,0, 1.0, X (default X) 
--help      display help and exit
"


while getopts "d:m:M:p:r:a:b:k:" opt
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
      r)
         nrep=$OPTARG
         ;;
      a)
         ialpha=$OPTARG
         ;;
      b)
         ibeta=$OPTARG
         ;;
      k)
         nblk=$OPTARG
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
   Mval="-M 0"  #by default all 
   Mstr="_allM"
else
   Mval="-M $m"  #by default all 
   Mstr="_M${m}"
fi

#
#  run xsdmmtime for all dataset
#
for dset in $dsets
do
   FILES=${mdir}/${dset}/*.mtx 
   res=${resdir}/${dset}${Mstr}_a${ialpha}b${ibeta}_nr${nrep}${par}.csv
   
   echo "FILENAME,NNZ,M,N,D,Trusted_inspect_time,trusted_exe_time,Test_inspect_time,Test_exe_time,Speedup_exe_time,Speedup_total_time,Critical_point" > $res
   
   for file in $FILES 
   do 
      #echo "$file"
      ./bin/xsdmmtime${par} -input $file $Mval -D $d -nrep $nrep -skHd 1 \
         -ialpha $ialpha -ibeta $ibeta -nrblk $nblk | tee -a $res 
   done
done

