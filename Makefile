SAMPLE = ./sample
BIN = ./bin
KERNEL = ./kernels

#
#  to compiler C kernels 
#
KCC = gcc 
#KCCFLAGS = -g -fopenmp -O3 -mavx  
KCCFLAGS = -fopenmp -O3 -march=native 

#
# Application compiler 
#

CC = g++
#FLAGS = -g -fopenmp -O3 -mavx -std=c++11 
FLAGS = -fopenmp -O3 -march=native -std=c++11
#FLAGS = -g -fopenmp -O3 -march=native -std=c++11

#
# My parallel flags 
#

ldb=l
NTHDS=48
LDB=LOAD_BALANCE 

#MYPT_FLAG = -DPTTIME 
#MYPT_FLAG = -DPTTIME -DNTHREADS=48  
#MYPT_FLAG = -DPTTIME -DLOAD_BALANCE -DNTHREADS=18
#MYPT_FLAG = -DPTTIME -DLOAD_BALANCE -DNTHREADS=32
#MYPT_FLAG = -DPTTIME -DLOAD_BALANCE -DNTHREADS=24
#MYPT_FLAG = -DPTTIME -DLOAD_BALANCE -DNTHREADS=48
#MYPT_FLAG = -DPTTIME -DDYNAMIC -DNTHREADS=48  

MYPT_FLAG = -DPTTIME -DNTHREADS=$(NTHDS) -D$(LDB)  


#
# flags needed to run mkl
#
MKLROOT = /opt/intel/mkl

#
#parallel version of MKL 
#
PT_CC_MKL_FLAG = -DMKL_ILP64 -m64 -I${MKLROOT}/include
PT_LD_MKL_FLAG =  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
	       ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a \
	       ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp \
	       -lpthread -lm -ldl  

#serial version of MKL 
CC_MKL_FLAG =  -DMKL_ILP64 -m64 -I${MKLROOT}/include
LD_MKL_FLAG =  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
	       ${MKLROOT}/lib/intel64/libmkl_sequential.a \
	       ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread \
	       -lm -ldl 


TOCOMPILE= 
LIBS=

#INCLUDE = simd.h kernels.h dkernels_avxz.h 

INCLUDE = $(KERNEL)/ 


all: $(BIN)/xsdmmtime $(BIN)/xsdmmtime_pt $(BIN)/xsdmmtime_$(NTHDS)$(ldb)pt  

#$(BIN)/%: $(SAMPLE)/%.cpp
#	mkdir -p $(BIN)
#	$(CC) $(FLAGS) $(INCLUDE) $(CC_MKL_FLAG) -o $@ $^ -DCPP \
#	   -DHW_EXE $(TOCOMPILE) $(LIBS) $(LD_MKL_FLAG)

$(BIN)/dkernel.o: $(KERNEL)/dkernels.c $(KERNEL)/kernels.h \
   $(KERNEL)/dkernels_D128.h 
	$(KCC) $(KCCFLAGS) -o $@ -c $(KERNEL)/dkernels.c  


$(BIN)/sdmmtime.o: sdmmtime.cpp $(KERNEL)/kernels.h  
	mkdir -p $(BIN)
	$(CC) $(FLAGS) -I$(INCLUDE) $(CC_MKL_FLAG) -DCPP -c sdmmtime.cpp -o $@   

$(BIN)/xsdmmtime: $(BIN)/sdmmtime.o $(BIN)/dkernel.o  
	mkdir -p $(BIN)
	$(CC) $(FLAGS) -o $@ $^ $(LIBS) $(LD_MKL_FLAG)


#
## parallel version 
#

$(BIN)/dkernel_pt.o: $(KERNEL)/dkernels.c $(KERNEL)/kernels.h $(KERNEL)/dkernels_D128.h 
	$(KCC) $(KCCFLAGS) $(MYPT_FLAG) -o $@ -c $(KERNEL)/dkernels.c  

$(BIN)/sdmmtime_pt.o: sdmmtime.cpp $(KERNEL)/kernels.h  
	mkdir -p $(BIN)
	#$(CC) $(FLAGS) -I$(INCLUDE) $(PT_CC_MKL_FLAG) -DCPP -c sdmmtime.cpp -o $@   
	$(CC) $(FLAGS) -I$(INCLUDE) $(PT_CC_MKL_FLAG) $(MYPT_FLAG) -DCPP -c sdmmtime.cpp -o $@   

$(BIN)/xsdmmtime_pt: $(BIN)/sdmmtime_pt.o $(BIN)/dkernel_pt.o  
	mkdir -p $(BIN)
	$(CC) $(FLAGS) -o $@ $^ $(LIBS) $(PT_LD_MKL_FLAG)

#
## specific parallel version 
#
#
sLDB=-DSTATIC
dLDB=-DDYNAMIC
lLDB=-DLOAD_BALANCE 

PFLAG="-DPTTIME $($(ldb)LDB) -DNTHREADS=$(NTHDS)"

$(BIN)/dkernel_$(ldb)$(NTHDS)pt.o: $(KERNEL)/dkernels.c $(KERNEL)/kernels.h $(KERNEL)/dkernels_D128.h 
	$(KCC) $(KCCFLAGS) $(PFLAG) -o $@ -c $(KERNEL)/dkernels.c  

$(BIN)/sdmmtime_$(ldb)$(NTHDS)pt.o: sdmmtime.cpp $(KERNEL)/kernels.h  
	mkdir -p $(BIN)
	$(CC) $(FLAGS) -I$(INCLUDE) $(PT_CC_MKL_FLAG) $(PFLAG) -DCPP -c sdmmtime.cpp -o $@   

$(BIN)/xsdmmtime_$(NTHDS)$(ldb)pt: $(BIN)/sdmmtime_$(ldb)$(NTHDS)pt.o $(BIN)/dkernel_$(ldb)$(NTHDS)pt.o  
	mkdir -p $(BIN)
	$(CC) $(FLAGS) -o $@ $^ $(LIBS) $(PT_LD_MKL_FLAG)

clean:
	rm -rf ./bin/*

gen-er:
	./scripts/gen_er.sh

gen-rmat:
	./scripts/gen_rmat.sh

download:
	./scripts/download.sh
