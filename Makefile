SAMPLE = ./sample
BIN = ./bin

CC = g++

# valgrind doesn't support avxz, to check memory error call with avx 
#FLAGS = -g -fopenmp -O3 -mavx 

MKLROOT = /opt/intel/mkl
CC_MKL_FLAG = /DMKL_ILP64 -m64 -I${MKLROOT}/include
LD_MKL_FLAG =  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
	       ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a \
	       ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp \
	       -lpthread -lm -ldl


FLAGS = -g -fopenmp -O3 -march=native
TOCOMPILE=
LIBS=

$(BIN)/%: $(SAMPLE)/%.cpp
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(INCLUDE) $(CC_MKL_FLAG) -o $@ $^ -DCPP -DHW_EXE \
	   ${TOCOMPILE} ${LIBS}


$(BIN)/CompAlgo: CompAlgo.cpp 
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(INCLUDE) -o $@ $^ -DCPP ${TOCOMPILE} ${LIBS} \
	   $(LD_MKL_FLAG)

clean:
	rm -rf ./bin/*

gen-er:
	./scripts/gen_er.sh

gen-rmat:
	./scripts/gen_rmat.sh

download:
	./scripts/download.sh
