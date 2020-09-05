#!/bin/bash


#
#  use bigger repeatation number for smaller matrices 
#

#nrep=100
nrep=20
#nrep=0  # means all block divisible by block size 
nblk=10

nthreads=48
#nthreads=24
#nthreads=12

# alpha=1.0 beta=1.0 -- use different kernel, but not interesting 
#./runtime.sh -r $nrep -p 0 -M 256 -a 1 -b 1  
#./runtime.sh -r $nrep 0 -m 1 -a 1 -b 1
#./runtime.sh -r $nrep -p 1 -M 256 -a 1 -b 1  
#./runtime.sh -r $nrep -p 1 -m 1 -a 1 -b 1

#alpha=X beta=X
#./runtime.sh -r $nrep -p 0 -M 256 -k $nblk -a 2 -b 2  
#./runtime.sh -r $nrep -p 0 -m 1 -a 2 -b 2
#./runtime.sh -r $nrep -p 1 -M 256 -k $nblk -a 2 -b 2  
#./runtime.sh -r $nrep -p 1 -m 1 -a 2 -b 2

#./runtime.sh -r $nrep -p 0 -M 256 -a 2 -b 2  
#./runtime.sh -r $nrep -p 0 -m 1 -a 2 -b 2  

#LDB=static
./runtime.sh -r $nrep -p 1 -M 256 -k $nblk -a 2 -b 2 -t $nthreads -l s
./runtime.sh -r $nrep -p 1 -M 512 -k $nblk -a 2 -b 2  -t $nthreads -l s
./runtime.sh -r $nrep -p 1 -M 1024 -k $nblk -a 2 -b 2  -t $nthreads -l s 
./runtime.sh -r $nrep -p 1 -m 1 -a 2 -b 2 -t $nthreads -l s 

#LDB=load_balance 
./runtime.sh -r $nrep -p 1 -M 256 -k $nblk -a 2 -b 2 -t $nthreads -l l
./runtime.sh -r $nrep -p 1 -M 512 -k $nblk -a 2 -b 2  -t $nthreads -l l
./runtime.sh -r $nrep -p 1 -M 1024 -k $nblk -a 2 -b 2  -t $nthreads -l l 
./runtime.sh -r $nrep -p 1 -m 1 -a 2 -b 2 -t $nthreads -l l 

#LDB=dynamic
./runtime.sh -r $nrep -p 1 -M 256 -k $nblk -a 2 -b 2 -t $nthreads -l d
./runtime.sh -r $nrep -p 1 -M 512 -k $nblk -a 2 -b 2  -t $nthreads -l d
./runtime.sh -r $nrep -p 1 -M 1024 -k $nblk -a 2 -b 2  -t $nthreads -l d 
./runtime.sh -r $nrep -p 1 -m 1 -a 2 -b 2 -t $nthreads -l d

