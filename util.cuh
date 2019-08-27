#ifndef UTIL_H
#define UTIL_H
#include <stdint.h>
#include <cuda.h>

__forceinline__ __device__ uint64_t UTIL_getTimeGPU(void) {
	uint64_t time = 0;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));
	return time;
}

double UTIL_getCpuTime(void);
int UTIL_getHostDeviceTimeOffset(int deviceId, uint64_t *device_ns, double *host_ns);
int UTIL_getGpuTimeScale(int deviceId, double* scale);
#endif
