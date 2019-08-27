#Launch overhead measurements
This is a small experiment to measure the launch overhead of CUDA kernels on
the Jetson-TX2. Mulitple spinning kernels are launched from different streams.
To measure the overhead the time before the launch is recorded on the CPU.
After the kernels starts executing the timestamp of the globaltimer is taken
and converted to CPU time. The difference of these start times represent the
launch overhead.

## Configuration

Set device number, used number of streams (realized with cudaStreamPerThread) and number of kernel repetitions per stream.

```
#define DEVICE_NUMBER (0)
#define NOF_STREAMS (6)
#define NOF_REP (3)
```

Set spin duration of a single kernel
```
#if 1
#define SPIN_DURATION (30000) // 30us
#else
#define SPIN_DURATION (30000000) // 30ms
#endif
```

Synchronize with the stream after each kernel launch.
```
#if 1
#define STREAM_SYNC() CheckCUDAError(cudaStreamSynchronize(cudaStreamPerThread));
#else
#define STREAM_SYNC()
#endif
```

## Building
If you clone directly to the device call:
```
make all
```
and then run 
```
./launch-overhead
```

If you have ssh access to a Jetson-TX2 with an existing `nvidia` account call:
```
make target_run HOST=tx2fk
```

This will copy the project to the device via ssh and compile and run it on the target.
