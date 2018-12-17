/*
 ============================================================================
 Name        : gpgpu_merge.cpp
 Author      : Orai Dezso Gergely
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <algorithm>
#include "gpuMerge.cuh"
#include "cpuMerge.hpp"


void initialize(float *data, unsigned size, const bool ordered)
{
	if(ordered){
		for (unsigned i = 0; i < size; ++i)
			data[i]= 1000.0 - static_cast <float> (i*1000) / static_cast <float> (size);
	}
	else{
		for (unsigned i = 0; i < size; ++i)
			data[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1000));
	}
}
bool valid(float *std, float *cpu, float *gpu, unsigned size)
{
	std::cout<<'\n';
	//for (unsigned i = 0; i < size; ++i)std::cout<<std[i]<<' '<<cpu[i]<<'\n';
	for (unsigned i = 0; i < size; ++i)
		if (
				std[i] != cpu[i]
				//|| std[i] != gpu[i]
			) return false;
	return true;

}

bool test(const int WORK_SIZE,const bool ordered, double &cpuTime,double &gpuTime,double &stdTime,double &stdSTime){
	/* Generate input */
	float *cpuSort = new float[WORK_SIZE];
	initialize(cpuSort, WORK_SIZE, ordered);
	float *gpuSort = new float[WORK_SIZE];
	//std::copy(cpuSort,cpuSort + WORK_SIZE, gpuSort);
	float *stdSort = new float[WORK_SIZE];
	std::copy(cpuSort,cpuSort + WORK_SIZE, stdSort);
	float *stdSSort = new float[WORK_SIZE];
	std::copy(cpuSort,cpuSort + WORK_SIZE, stdSSort);

	time_t t1;
	time_t t2;

	/* Measure cpu time*/
	t1 = clock();
	cpuMergeSort(cpuSort, WORK_SIZE);
	t2 = clock();
	cpuTime = difftime(t2,t1);

	/* Measure gpu time */
	//t1 = clock();
	//gpuMergeSort(gpuSort, WORK_SIZE);
	//t2 = clock();
	gpuTime = difftime(t2,t1);

	/* Measure std time */
	t1 = clock();
	std::sort(stdSort, stdSort+WORK_SIZE);
	t2 = clock();
	stdTime = difftime(t2,t1);

	t1 = clock();
	std::stable_sort(stdSSort, stdSSort+WORK_SIZE);
	t2 = clock();
	stdSTime = difftime(t2,t1);

	/*for (unsigned i = 0; i < WORK_SIZE; ++i)
		std::cout
			<<cpuSort[i]
			//<<' '<<gpuSort[i]
			<<' '<<stdSort[i]<<'\n';*/

	/* Verify the results */
	bool val= valid(stdSort,cpuSort,gpuSort,WORK_SIZE);

	/* Free memory */
	delete[] stdSort;
	delete[] cpuSort;
	delete[] gpuSort;

	return val;
}

int main()
{
	if(!findCudaDevice()) exit(EXIT_FAILURE);

	double cpuTime,gpuTime,stdSTime,stdTime;

	//test(15,cpuTime,gpuTime,stdTime);
	//std::cout<<"cpu: "<<cpuTime<<", gpu: "<<gpuTime<<" std: "<<stdTime<<'\n';
	double sumc=0,sums=0;
	for(unsigned i=1000;i<10000;i+=1000){
		std::cout<<i<<'\n';
		std::cout<<test(i,true,cpuTime,gpuTime,stdTime,stdSTime)<<'\n';
		std::cout<<"cpu: "<<cpuTime<<", gpu: "<<gpuTime<<" std: "<<stdTime<<" stableStd: "<<stdSTime<<'\n';
		sumc+=cpuTime;
		sums+=stdTime;
	}
	std::cout<<(sumc/sums-1)*100<<'\n';

	return 0;
}

