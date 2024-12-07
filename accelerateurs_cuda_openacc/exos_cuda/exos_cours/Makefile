# in:
# all: library.cpp main.cpp
#    $@ evaluates to all
#    $< evaluates to library.cpp
#    $^ evaluates to library.cpp main.cpp


CC=gc

CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/11.8
CUDASDK=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/examples/OpenACC/SDK/include/
CUDANVIDIA=/usr/lib/nvidia
NVCC=/usr/bin/nvcc

NVCFLAGS= -I $(CUDA_HOME)/include -I$(CUDASDK)

CFLAGS= -L $(CUDA_HOME)/lib64 -L $(CUDANVIDIA)

hello: hello.o
	$(NVCC) $(CFLAGS) -o $@ $^

add: add.o
	$(NVCC) $(CFLAGS) -o $@ $^

add_block: add_block.o
	$(NVCC) $(CFLAGS) -o $@ $^

clean:
	rm -f *.o *.linkinfo hello add add_block

%_debug: %.cu
	$(NVCC) $(NVCFLAGS) -c -o -ggdb $@ $<

%.o: %.cu
	$(NVCC) $(NVCFLAGS) -c -o $@ $<
