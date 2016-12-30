NVCC = nvcc 
NVCC_FLAGS =
NVCC_INC = -I/usr/local/cuda/include 
LD_LIBS += -lcuda -lcudart

BINARIES = saxpy_p saxpy_k

all: $(BINARIES)

saxpy_p: saxpy.cu
	$(NVCC) $< -o $@

saxpy_k: saxpy.cu
	$(NVCC) $< -o $@

clean:
	rm -f $(BINARIES)
