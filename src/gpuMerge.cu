/*
 * gpuMerge.cu
 *
 *  Created on: Dec 16, 2018
 *      Author: Orai Dezso Gergely
 */

#include "gpuMerge.cuh"
#include <iostream>
static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define MIN_RUNTIME_VERSION 1000
#define MIN_COMPUTE_VERSION 0x10
int MaxThread = 512;

/**
 * CUDA kernel that sorts a float array
 */
__global__ void gpuMergeSortKernel(float *data, unsigned vectorSize) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < vectorSize)
		data[idx] = 1.0/data[idx];
}

/**
 * Host function that copies the data and launches the work on GPU
 */
void gpuMergeSort(float *data, unsigned size)
{
	float *gpuData;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuData, data, sizeof(float)*size, cudaMemcpyHostToDevice));

	static const int BLOCK_SIZE = MaxThread;
	const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;
	gpuMergeSortKernel<<<blockCount, BLOCK_SIZE>>> (gpuData, size);

	CUDA_CHECK_RETURN(cudaMemcpy(data, gpuData, sizeof(float)*size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(gpuData));
}


/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}



bool findCudaDevice(){
	int deviceCount, bestDev=-1;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&deviceCount));
    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        int runtimeVersion = 0;
        cudaRuntimeGetVersion(&runtimeVersion);
        if (runtimeVersion >= MIN_RUNTIME_VERSION && ((deviceProp.major<<4) + deviceProp.minor) >= MIN_COMPUTE_VERSION)
        {
            if (bestDev == -1)
            {
                bestDev = dev;
                MaxThread = deviceProp.maxThreadsPerBlock;
            }
        }
    }
    return bestDev != -1;
}


