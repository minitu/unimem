NVCC = nvcc 
NVCC_FLAGS = -O0 -g -G
NVCC_INC = -I/usr/local/cuda/include 
LD_LIBS += -lcuda -lcudart

BINARIES = add_p add_m

all: $(BINARIES)

add_p: add.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

add_m: add.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(BINARIES)
