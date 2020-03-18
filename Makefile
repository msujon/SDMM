SAMPLE = ./sample
BIN = ./bin

CC = g++
FLAGS = -g -fopenmp -O3 -march=native

TOCOMPILE=
LIBS=

$(BIN)/%: $(SAMPLE)/%.cpp
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(INCLUDE) -o $@ $^ -DCPP -DHW_EXE ${TOCOMPILE} ${LIBS}


$(BIN)/CompAlgo: CompAlgo.cpp 
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(INCLUDE) -o $@ $^ -DCPP ${TOCOMPILE} ${LIBS}

clean:
	rm -rf ./bin/*

gen-er:
	./scripts/gen_er.sh

gen-rmat:
	./scripts/gen_rmat.sh

download:
	./scripts/download.sh
