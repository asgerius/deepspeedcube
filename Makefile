CC      = gcc
CFLAGS  = -g0 -O3 -funroll-loops -fopenmp
CSHARED = -shared -fPIC

CPP     = g++
CXX	    = nvcc

OPT     = --compiler-options "-g0 -O3 -funroll-loops"
XOPTS   = -Xptxas=-v
ARCH    = -arch=sm_70
OMP     = -fopenmp

CUDA_PATH = /appl/cuda/11.5.1

all:
	make clean
	mkdir -p lib
	make lib/libdsc.so
	@if command -v nvcc ; then\
		make lib/cube_cuda.so;\
	fi

lib/libdsc.so:
	$(CC) -o $@ \
		deepspeedcube/c/astar.c deepspeedcube/c/heap.c deepspeedcube/c/unique.c \
		deepspeedcube/c/envs/envs.c deepspeedcube/c/envs/cube.c \
		deepspeedcube/c/hashmap_plus.c deepspeedcube/c/hashmap.c/*.c \
		$(CFLAGS) $(CSHARED)

lib/cube_cuda.so:
	$(CXX) $(OPT) $(ARCH) $(XOPTS) -Xcompiler "-fPIC" -dc \
		deepspeedcube/cuda/envs/cube.cu \
		-o lib/cube_cuda.o
	$(CXX) $(ARCH) $(XOPTS) -Xcompiler "-fPIC" -dlink \
		lib/cube_cuda.o \
		-o lib/link.o
	$(CPP) -shared -L$(CUDA_PATH)/lib64 -lcudart \
		lib/cube_cuda.o lib/link.o \
		-o lib/cube_cuda.so

clean:
	$(RM) -r lib/*
