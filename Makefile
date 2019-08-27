.PHONY: all clean target_clean
HOST ?= tx2fk
EXE = launch-overhead
REMOTE_TARGET = nvidia@$(HOST)
REMOTE_WORKING_DIR = ~/$(EXE)
CFLAGS := -Wall -Werror -O3
NVCCFLAGS := -Xptxas -O3 --ptxas-options=-v --compiler-options="$(CFLAGS)" \
	--generate-code arch=compute_62,code=[compute_62,sm_62] \
	--default-stream per-thread -Xcompiler -fopenmp \

#Export CUDA paths
export LIBRARY_PATH:=/usr/local/cuda/lib64:$(LIBRARY_PATH)
export LD_LIBRARY_PATH:=/usr/local/cuda/lib64:$(LD_LIBRARY_PATH)
export PATH:=/usr/local/cuda/bin:$(PATH)

ODIR=obj
DEPS = util.cuh
_OBJ = $(EXE).o util.o 
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

all: $(EXE)

obj:
	mkdir $@

$(ODIR)/%.o: %.cu $(DEPS) | obj
		nvcc -dc -o $@ $< $(NVCCFLAGS)

$(EXE): $(OBJ)
		nvcc -o $@ $^ $(NVCCFLAGS)

clean:
	rm -f $(EXE)
	rm -rf obj

target_build: deploy
	ssh -t $(REMOTE_TARGET) "cd $(REMOTE_WORKING_DIR) && make clean && make all"

target_run: target_build
	ssh -t $(REMOTE_TARGET) "cd $(REMOTE_WORKING_DIR) && ./$(EXE)"

deploy:
	rsync -avz --exclude '.*' --exclude 'README.md' --exclude 'tags' . ${REMOTE_TARGET}:${REMOTE_WORKING_DIR}

target_clean:
	ssh $(REMOTE_TARGET) "cd $(REMOTE_WORKING_DIR) && make clean"
