#include <errno.h>
#include <error.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <omp.h>
#include "util.cuh"

#define DEVICE_NUMBER (0)
#define NOF_STREAMS (6)
#define NOF_REP (3)

#if 1
#define SPIN_DURATION (30000) // 30us
#else
#define SPIN_DURATION (30000000) // 30ms
#endif

#if 1
#define STREAM_SYNC() CheckCUDAError(cudaStreamSynchronize(cudaStreamPerThread));
#else
#define STREAM_SYNC()
#endif

typedef struct gpu_data{
	double scale;
	double host_ns;
	uint64_t device_ns;
	uint64_t *startTimesGpu_d;
	uint64_t *stopTimesGpu_d;
	uint64_t *startTimesGpu_h;
	uint64_t *stopTimesGpu_h;
	double *startTimesCpu;
} gpu_data_t;


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

__global__ void spinKernel(uint64_t spin_duration, uint64_t *startTime, uint64_t *stopTime) {
	uint64_t start_time = UTIL_getTimeGPU();
	if(threadIdx.x == 0){
		*startTime = start_time;
	}
	while ((UTIL_getTimeGPU() - start_time) < spin_duration) {
		continue;
	}
	if(threadIdx.x == 0){
		*stopTime = UTIL_getTimeGPU();
	}
}

static int initializeTest(gpu_data_t *gpuData){

	// Allocate data
	if (CheckCUDAError(cudaMalloc(&gpuData->startTimesGpu_d, NOF_STREAMS * NOF_REP * sizeof(*gpuData->startTimesGpu_d)))) {
		printf("Failed to allocate startTimesGpu_d\n");
		return -1;
	}
	if (CheckCUDAError(cudaMalloc(&gpuData->stopTimesGpu_d, NOF_STREAMS * NOF_REP * sizeof(*gpuData->stopTimesGpu_d)))) {
		printf("Failed to allocate stopTimesGpu_d\n");
		return -1;
	}

	gpuData->startTimesGpu_h = NULL;
	gpuData->startTimesGpu_h = (uint64_t*)malloc(NOF_STREAMS * NOF_REP * sizeof(*gpuData->startTimesGpu_h));
	if(gpuData->startTimesGpu_h == NULL){
		printf("Failed to allocate startTimesGpu_h\n");
		return -1;
	}
	memset(gpuData->startTimesGpu_h, 0,NOF_STREAMS * NOF_REP * sizeof(*gpuData->startTimesGpu_h));

	gpuData->stopTimesGpu_h = NULL;
	gpuData->stopTimesGpu_h = (uint64_t*)malloc(NOF_STREAMS * NOF_REP * sizeof(*gpuData->stopTimesGpu_h));
	if(gpuData->stopTimesGpu_h == NULL){
		printf("Failed to allocate stopTimesGpu_h\n");
		return -1;
	}

	gpuData->startTimesCpu = NULL;
	gpuData->startTimesCpu = (double*)malloc(NOF_STREAMS * NOF_REP * sizeof(*gpuData->startTimesCpu));
	if(gpuData->startTimesCpu == NULL){
		printf("Failed to allocate startTimesCpu\n");
		return -1;
	}

	// Get time parameters
	UTIL_getGpuTimeScale(0,&gpuData->scale);
	UTIL_getHostDeviceTimeOffset(0, &gpuData->device_ns, &gpuData->host_ns);
	printf("Scale: %f, CpuStart: %f, GpuStart: %f\n",gpuData->scale, gpuData->host_ns, (double)gpuData->device_ns);
	return 0;
}

static int runTest(gpu_data_t *gpuData){
#pragma omp parallel for schedule(static)
	for(int i = 0 ; i<NOF_STREAMS; i++){
		for(int j=0;j<NOF_REP;j++){
			gpuData->startTimesCpu[i*NOF_REP+j] = UTIL_getCpuTime();
			spinKernel<<<1,256>>>(SPIN_DURATION, &gpuData->startTimesGpu_d[i*NOF_REP+j], &gpuData->stopTimesGpu_d[i*NOF_REP+j]);
            STREAM_SYNC();
		}
	}
	if (CheckCUDAError(cudaDeviceSynchronize())) return -1;

	// Copyback results
	if (CheckCUDAError(cudaMemcpy(gpuData->startTimesGpu_h,
					gpuData->startTimesGpu_d,
					NOF_STREAMS * NOF_REP * sizeof(*gpuData->startTimesGpu_d),
					cudaMemcpyDeviceToHost))) return -1;
	if (CheckCUDAError(cudaMemcpy(gpuData->stopTimesGpu_h,
					gpuData->stopTimesGpu_d,
					NOF_STREAMS * NOF_REP * sizeof(*gpuData->stopTimesGpu_d),
					cudaMemcpyDeviceToHost))) return -1;
	return 0;
}

static double convertGpuToCpu(uint64_t gpuStartTime, uint64_t gpuTime, double scale){
	return (double)(gpuTime-gpuStartTime) * scale;
}

static int writeResults(gpu_data_t *gpuData){
	double cpuStart, gpuStart, launchOh, gpuEnd;
	printf("|%-6.6s|%-4.4s|%-15.15s|%-15.15s|%-15.15s|%-15.15s|%-13.13s|\n","Stream","Rep.","CPU start [us]", "GPU start [us]", "Launch OH [us]", "GPU end [us]", "GPU dur. [us]");
	for(int stream = 0; stream < NOF_STREAMS; stream++){
		for(int rep = 0; rep < NOF_REP; rep++){
			cpuStart = gpuData->startTimesCpu[stream*NOF_REP + rep] - gpuData->host_ns;
			gpuStart = convertGpuToCpu(gpuData->device_ns, gpuData->startTimesGpu_h[stream*NOF_REP + rep], gpuData->scale);
			launchOh = gpuStart-cpuStart;
			gpuEnd = convertGpuToCpu(gpuData->device_ns, gpuData->stopTimesGpu_h[stream*NOF_REP + rep], gpuData->scale);
			// scale to ms
			cpuStart = cpuStart/1e3;
			gpuStart = gpuStart/1e3;
			launchOh = launchOh/1e3;
			gpuEnd = gpuEnd/1e3;
			printf("|%-6d|%-4d|%15.6f|%15.6f|%15.6f|%15.6f|%17.6f|\n",stream,rep, cpuStart, gpuStart, launchOh, gpuEnd, gpuEnd-gpuStart);
		}
	}
return 0;
}

static int cleanUp(gpu_data_t *gpuData){
	// Free data
	CheckCUDAError(cudaFree(gpuData->startTimesGpu_d));
	CheckCUDAError(cudaFree(gpuData->stopTimesGpu_d));
	free(gpuData->startTimesGpu_h);
	free(gpuData->stopTimesGpu_h);
	free(gpuData->startTimesCpu);
	return 0;
}

int main(int argc, char **argv) {
	gpu_data_t data;

	// Set CUDA device
	if (CheckCUDAError(cudaSetDevice(DEVICE_NUMBER))){
		printf("Failed do set CUDA device\n");
		return EXIT_FAILURE;
	}

	// Initialize parameters
	if (initializeTest(&data) < 0){
		printf("Failed to initialize test\n");
		return EXIT_FAILURE;
	}

	// Run test
	if (runTest(&data) < 0){
		printf("Failed to run test\n");
		return EXIT_FAILURE;
	}

	// Write results
	if (writeResults(&data) < 0) {
		printf("Failed to write results\n");
		return EXIT_FAILURE;
	}

	// Clean up
	if (cleanUp(&data) < 0) {
		printf("Failed to clean up\n");
		return EXIT_FAILURE;
	}

	printf("Finished testrun\n");
	cudaDeviceReset();
	return 0;
}
