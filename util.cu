#include "util.cuh"
#include <cuda.h>
#include <stdio.h>

#define SPIN_DURATION (1000000000)
// Prints a message and returns zero if the given value is not cudaSuccess
#define CheckCUDAError(val) (InternalCheckCUDAError((val), #val, __FILE__, __LINE__))

// Called internally by CheckCUDAError
static int InternalCheckCUDAError(cudaError_t result, const char *fn,
		const char *file, int line) {
	if (result == cudaSuccess) return 0;
	printf("CUDA error %d in %s, line %d (%s): %s\n", (int) result, file, line,
			fn, cudaGetErrorString(result));
	return -1;
}

static __global__ void getTimeKernel(uint64_t *time) {
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(*time));
}

__global__ void spinKernel(uint64_t spin_duration) {
	uint64_t start_time = UTIL_getTimeGPU();
	while ((UTIL_getTimeGPU() - start_time) < spin_duration) {
		continue;
	}
}

double UTIL_getCpuTime(void){
	struct timespec start;
	if(clock_gettime(CLOCK_MONOTONIC, &start)){
		printf("Error getting CPU time.\n");
		exit(0);
	}
	return (double)start.tv_sec *1e9 + (double)start.tv_nsec;
}


int UTIL_getHostDeviceTimeOffset(int deviceId, uint64_t *device_ns, double *host_ns){
	uint64_t *time_d;

	if (CheckCUDAError(cudaSetDevice(deviceId))) return -1;
	if (CheckCUDAError(cudaMalloc(&time_d, sizeof(*time_d)))) return -1;

	// Warm-up
	getTimeKernel<<<1,1>>>(time_d); 
	if (CheckCUDAError(cudaDeviceSynchronize())) return -1;

	// Do Measurement
	getTimeKernel<<<1, 1>>>(time_d);
	*host_ns = UTIL_getCpuTime();
	if (CheckCUDAError(cudaMemcpy(device_ns, time_d, sizeof(*device_ns), cudaMemcpyDeviceToHost))) {
		cudaFree(time_d);
		return -1;
	}
	cudaFree(time_d);
	return 0;
}

int UTIL_getGpuTimeScale(int deviceId, double* scale){
	double cpuStart, cpuStop;
	if (CheckCUDAError(cudaSetDevice(deviceId))) return -1;

	// Warm-up
	spinKernel<<<1,1>>>(1000);
	if (CheckCUDAError(cudaDeviceSynchronize())) return -1;

	cpuStart = UTIL_getCpuTime();
	spinKernel<<<1, 1>>>(SPIN_DURATION);
	if (CheckCUDAError(cudaDeviceSynchronize())) return -1;
	cpuStop = UTIL_getCpuTime();
	*scale = (double)(cpuStop-cpuStart)/(double)SPIN_DURATION;
	return 0;
}
