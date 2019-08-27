#Launch overhead measurements
This is a small experiment to measure the launch overhead of CUDA kernels on
the Jetson-TX2. Mulitple spinning kernels are launched from different streams.
To measure the overhead the time before the launch is recorded on the CPU.
After the kernels starts executing the timestamp of the globaltimer is taken
and converted to CPU time. The difference of these start times represent the
launch overhead.
