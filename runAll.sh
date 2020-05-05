#!/bin/bash


#
#  use bigger repeatation number for smaller matrices 
#

nrep=10
nblk=10

# alpha=1.0 beta=1.0 -- use different kernel, but not interesting 
#./runtime.sh -r $nrep -p 0 -M 256 -a 1 -b 1  
#./runtime.sh -r $nrep 0 -m 1 -a 1 -b 1
#./runtime.sh -r $nrep -p 1 -M 256 -a 1 -b 1  
#./runtime.sh -r $nrep -p 1 -m 1 -a 1 -b 1

#alpha=X beta=X
#./runtime.sh -r $nrep -p 0 -M 256 -k $nblk -a 2 -b 2  
#./runtime.sh -r $nrep -p 0 -m 1 -a 2 -b 2
./runtime.sh -r $nrep -p 1 -M 256 -k $nblk -a 2 -b 2  
#./runtime.sh -r $nrep -p 1 -m 1 -a 2 -b 2
