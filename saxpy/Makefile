NVCC = nvcc 
NVCC_FLAGS = -O0 -g -G
NVCC_INC = -I/usr/local/cuda/include 
LD_LIBS += -lcuda -lcudart

BINARIES = saxpy_p saxpy_m

all: $(BINARIES)

saxpy_p: saxpy.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

saxpy_m: saxpy.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(BINARIES)
