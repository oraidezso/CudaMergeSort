nvcc -O3 gpgpu_merge.cpp gpuMerge.cu cpuMerge.cpp
#nvcc -g gpgpu_merge.cpp gpuMerge.cu cpuMerge.cpp
./a.out