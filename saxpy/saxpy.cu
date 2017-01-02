#include <stdio.h>
#include <cuda_profiler_api.h>

#define cudaCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"cudaAssert: %s at %s:%d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

__global__ void saxpy(int n, float a, float *x, float *y)
{
  // setup
  int total_thread_num = gridDim.x * blockDim.x;
  int num_per_thread = n / total_thread_num;
  int leftover = n % total_thread_num;
  int thread_index = blockIdx.x*blockDim.x + threadIdx.x;
  int start_index = num_per_thread * thread_index;
  if (thread_index < leftover) {
    start_index += thread_index;
    num_per_thread++;
  }
  else {
    start_index += leftover;
  }

  // saxpy
  for (int i = start_index; i < start_index + num_per_thread; i++) {
    y[i] = a*x[i] + y[i];
  }
}

int main(void)
{
  int nDevices;

  // print GPU info
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
        prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
        prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
        2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }

  int N = 1<<10;
  printf("N: %d\n", N);
  float *x, *y;

  // start profiling
  cudaCheck(cudaProfilerStart());

  // allocate memory with UM
  printf("Allocating %d bytes each...\n", (size_t)N*sizeof(float));
  cudaCheck(cudaMallocManaged(&x, N*sizeof(float)));
  cudaCheck(cudaMallocManaged(&y, N*sizeof(float)));

  // initialization on host -> page fault (CPU)
  printf("Initializing...\n");
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // kernel launch -> page fault (GPU)
  printf("Launching kernel...\n");
  saxpy<<<min(1024, (N+255)/256), 256>>>(N, 2.0f, x, y); // watch out for grid size limit

  // check for kernel launch error
  cudaCheck(cudaPeekAtLastError());

  // wait for kernel to finish
  printf("Synchronizing...\n");
  cudaCheck(cudaDeviceSynchronize());

  // check error -> page fault (CPU)
  printf("Checking error...\n");
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  // free memory
  printf("Freeing memory...\n");
  cudaCheck(cudaFree(x));
  cudaCheck(cudaFree(y));

  // end profiling
  printf("All done!\n");
  cudaCheck(cudaProfilerStop());
}
