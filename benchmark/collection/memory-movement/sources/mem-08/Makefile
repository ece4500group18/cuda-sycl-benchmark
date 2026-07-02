# Makefile for histogramSharedPrivatization (CUDA build)
#
# Usage:
#   make            # build ./app with nvcc
#   make run        # build and run, writing output/cuda_output.txt
#   make clean

CC      = nvcc
CFLAGS  = -O2 -std=c++17
ARCH    = -gencode arch=compute_70,code=sm_70 \
          -gencode arch=compute_80,code=sm_80 \
          -gencode arch=compute_90,code=sm_90

SRC     = main.cu
APP     = app
OUTDIR  = output
OUT     = $(OUTDIR)/cuda_output.txt

.PHONY: all run clean

all: $(APP)

$(APP): $(SRC) reference.h
	$(CC) $(CFLAGS) $(ARCH) $(SRC) -o $(APP)

run: $(APP)
	@mkdir -p $(OUTDIR)
	./$(APP) $(OUT)

clean:
	rm -rf $(APP) $(OUTDIR)
