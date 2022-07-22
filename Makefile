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
	mkdir -p lib/envs
	make lib/libdsc.so
	@if command -v nvcc ; then\
		make lib/cube_cuda.so;\
	fi

lib/libdsc.so:
	make lib/astar.so
	make lib/heap.so
	make lib/unique.so
	make lib/envs/envs.so
	make lib/envs/cube.so
	$(CC) -o $@ \
		lib/astar.so lib/heap.so lib/unique.so \
		lib/envs/envs.so lib/envs/cube.so \
		$(CFLAGS) $(CSHARED)

lib/astar.so:
	$(CC) -o $@ deepspeedcube/c/astar.c deepspeedcube/c/hashmap_plus.c $(CFLAGS) $(CSHARED)

lib/heap.so:
	$(CC) -o $@ deepspeedcube/c/heap.c $(CFLAGS) $(CSHARED)

lib/unique.so:
	$(CC) -o $@ deepspeedcube/c/unique.c deepspeedcube/c/hashmap_plus.c deepspeedcube/c/hashmap.c/*.c $(CFLAGS) $(CSHARED)

lib/envs/envs.so:
	$(CC) -o $@ deepspeedcube/c/envs/envs.c $(CFLAGS) $(CSHARED)

lib/envs/cube.so:
	$(CC) -o $@ deepspeedcube/c/envs/cube.c $(CFLAGS) $(CSHARED)

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
