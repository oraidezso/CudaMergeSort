/*
 * gpuMerge.cu
 *
 *  Created on: Dec 16, 2018
 *      Author: Orai Dezso Gergely
 */

#include "gpuMerge.cuh"
#include <iostream>
#include <stdio.h>
static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define MIN_RUNTIME_VERSION 1000
#define MIN_COMPUTE_VERSION 0x10
int MaxThread = 512;
int BlockNum=2,CoreInBlock=128;



void cmerge(float *l, float *r, float *to, float *end, int length){
	//int length = r - l;
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

void cpuMergeSort(float *data, unsigned int size, int length=1)
{
	float *tmp = new float[size];
	float *akt = data;
	float *next = tmp;

	for (; length < size; length *= 2){
		float *end=akt+size;
		for(unsigned col = 0; col< size; col+=2*length){
			cmerge(akt + col, akt + col + length, next + col, end, length);
		}
		float *c = akt;
		akt=next;
		next=c;
	}
	if(akt!=data)for(unsigned i=0;i<size;++i)data[i]=akt[i];

	delete[] tmp;
}




/**
 * CUDA kernel what merges two float arrays
 */

__device__ void kernelMerge(float *l, float *r, float *to, float *end, int length){
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
 * CUDA kernel that sorts a float array
 */

 __global__ void gpuKernelMergeSort(float *data, float *tmpIn, unsigned int fullSize, unsigned int size, unsigned int length=1)
{
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	float *tmp = tmpIn + (idx * size);
	float *akt = data + (idx * size);
	// The size of the last section is diferent so we have to check it
	if(data+fullSize > akt) size = (data + fullSize) - akt;
	float *next = tmp;

	for (; length < size; length *= 2){
		float *end=akt+size;
		for(unsigned col = 0; col< size; col+=2*length){
			kernelMerge(akt + col, akt + col + length, next + col, end, length);
		}
		float *c = akt;
		akt=next;
		next=c;
	}
	if(akt != data+(idx*size))for(unsigned i=0;i<size;++i)data[i]=akt[i];

}






/**
 * Host function that copies the data and launches the work on GPU
 */
void gpuMergeSort(float *data, unsigned size)
{
	if(size < CoreInBlock*BlockNum*4){
		cpuMergeSort(data,size);
		return;
	}
	float *gpuData;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuData, data, sizeof(float)*size, cudaMemcpyHostToDevice));
	float *tmp;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&tmp, sizeof(float)*size));

	int arraySizeInBlock = CoreInBlock*BlockNum;

	gpuKernelMergeSort<<<BlockNum,CoreInBlock>>>(gpuData, tmp, size, arraySizeInBlock);
	gpuKernelMergeSort<<<1,1>>>(gpuData, tmp, size, size, arraySizeInBlock);

	CUDA_CHECK_RETURN(cudaMemcpy(data, gpuData, sizeof(float)*size, cudaMemcpyDeviceToHost));
	
	//cpuMergeSort(data,size,arraySizeInBlock);

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



