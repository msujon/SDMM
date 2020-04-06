SAMPLE = ./sample
BIN = ./bin

CC = g++

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
MYPT_FLAG = -DPTTIME 


#serial version of MKL 
CC_MKL_FLAG =  -DMKL_ILP64 -m64 -I${MKLROOT}/include
LD_MKL_FLAG =  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
	       ${MKLROOT}/lib/intel64/libmkl_sequential.a \
	       ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread \
	       -lm -ldl 

# valgrind doesn't support avx512, to check memory error call with avx 
#FLAGS = -g -fopenmp -O3 -mavx -std=c++11 

#FLAGS = -g -fopenmp -O3 -march=native -std=c++11
FLAGS = -fopenmp -O3 -march=native -std=c++11

TOCOMPILE=
LIBS=

#INCLUDE = simd.h kernels.h dkernels_avxz.h 
INCLUDE =  

#
##
#
all: $(BIN)/CompAlgo $(BIN)/CompAlgo_pt 

$(BIN)/%: $(SAMPLE)/%.cpp
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(INCLUDE) $(CC_MKL_FLAG) -o $@ $^ -DCPP \
	   -DHW_EXE $(TOCOMPILE) $(LIBS) $(LD_MKL_FLAG)

$(BIN)/CompAlgo: CompAlgo.cpp $(INCLUDE) 
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(INCLUDE) $(CC_MKL_FLAG) -o $@ $^ -DCPP \
	   $(TOCOMPILE) $(LIBS) $(LD_MKL_FLAG)

#
## parallel version 
#

$(BIN)/CompAlgo_pt: CompAlgo.cpp $(INCLUDE)
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(INCLUDE) $(MYPT_FLAG) $(PT_CC_MKL_FLAG) -o $@ $^ -DCPP \
	   $(TOCOMPILE) $(LIBS) $(PT_LD_MKL_FLAG)
clean:
	rm -rf ./bin/*

gen-er:
	./scripts/gen_er.sh

gen-rmat:
	./scripts/gen_rmat.sh

download:
	./scripts/download.sh
