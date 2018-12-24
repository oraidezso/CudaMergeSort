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
int BlockNum=2,CoreInBlock=128;

/**
 * CUDA kernel that sorts a float array
 */


__global__ void merge1(float *akt, float *next, float *end, int length){
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	float *l=akt+idx*length*2;
	float *r=akt+idx*length*2+length;
	float *to=next+idx*length*2;
	float *lend=(r<end) ? r : end;
	float *rend=(r+length <end) ? r+length : end;

	while(true){
		if(l==lend){
			while(r<rend){
				*to++=*r++;
			}
			break;
		}
		if(r>=rend){
			while(l<lend){
				*to++=*l++;
			}
			break;
		}
		*to++ = (*l < *r) ? *l++ : *r++;
	}
}




/**
 * Host function that copies the data and launches the work on GPU
 */
void gpuMergeSort(float *data, unsigned size)
{
	float *gpuData;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuData, data, sizeof(float)*size, cudaMemcpyHostToDevice));
	float *tmp;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&tmp, sizeof(float)*size));
	float *akt=gpuData;
	float *next=tmp;

	static const int BLOCK_SIZE = MaxThread;

	for (unsigned length = 1; length < size; length *= 2){
		float *end=akt+size;
		int blockCount=((size+BLOCK_SIZE-1)/(BLOCK_SIZE*2*length))+1;
		for(unsigned col = 0; col< size; col+=2*length){
			merge1<<<blockCount,BLOCK_SIZE>>>(akt, next, end, length);
		}
		float *c = akt;
		akt=next;
		next=c;
	}

	CUDA_CHECK_RETURN(cudaMemcpy(data, akt, sizeof(float)*size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(gpuData));
	CUDA_CHECK_RETURN(cudaFree(tmp));
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

/**
 * Get core/sm for optimalization purposes
 */
int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    switch (devProp.major){
     case 2: // Fermi
      if (devProp.minor == 1) cores = 48;
      else cores = 32;
      break;
     case 3: // Kepler
      cores = 192;
      break;
     case 5: // Maxwell
      cores = 128;
      break;
     case 6: // Pascal
      if (devProp.minor == 1) cores = 128;
      else if (devProp.minor == 0) cores = 64;
      else printf("Unknown device type\n");
      break;
     case 7: // Volta
      if (devProp.minor == 0) cores = 64;
      else printf("Unknown device type\n");
      break;
     default:
      printf("Unknown device type\n");
      break;
      }
    return cores;
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
                BlockNum=deviceProp.multiProcessorCount;
                CoreInBlock=getSPcores(deviceProp);
                if(CoreInBlock==0)return false;
            }
        }
    }
    if(bestDev != -1)cudaSetDevice(bestDev);
    return bestDev != -1;
}



