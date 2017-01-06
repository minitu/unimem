NVCC = nvcc 
NVCC_FLAGS = -O0 -g -G
NVCC_INC = -I/usr/local/cuda/include 
LD_LIBS += -lcuda -lcudart

BINARIES = devinfo

all: $(BINARIES)

devinfo: devinfo.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(BINARIES)
