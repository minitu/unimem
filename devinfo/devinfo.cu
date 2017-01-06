#include <stdio.h>

#define cudaCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"cudaAssert: %s at %s:%d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

const char *boolStrings[2] = {"NO", "YES"};

int main(void)
{
  // print GPU info
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device Name: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Number of SMs: %d\n", prop.multiProcessorCount);
    printf("  Core Clock Rate (KHz): %d\n",
        prop.clockRate);
    printf("  Memory Clock Rate (KHz): %d\n",
        prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
        prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
        2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  Device Overlap Supported: %s\n", boolStrings[prop.deviceOverlap]);
    printf("  Concurrent Kernels Supported: %s\n", boolStrings[prop.concurrentKernels]);
    printf("  Managed Memory Supported: %s\n", boolStrings[prop.managedMemory]);
    printf("  Concurrent Managed Memory Access Supported: %s\n\n",
        boolStrings[prop.concurrentManagedAccess]);
  }

  return 0;
}
