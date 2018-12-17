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


void initialize(float *data, unsigned size)
{
	for (unsigned i = 0; i < size; ++i)
		data[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1000));
}
bool valid(float *std, float *cpu, float *gpu, unsigned size)
{
	for (unsigned i = 0; i < size; ++i)
		if (std[i] != cpu[i] || std[i] != gpu[i]) return false;
	return true;
}

bool test(const int WORK_SIZE,double &cpuTime,double &gpuTime,double &stdTime){
	/* Generate input */
	float *cpuSort = new float[WORK_SIZE];
	initialize(cpuSort, WORK_SIZE);
	float *gpuSort = new float[WORK_SIZE];
	std::copy(cpuSort,cpuSort + WORK_SIZE, gpuSort);
	float *stdSort = new float[WORK_SIZE];
	std::copy(cpuSort,cpuSort + WORK_SIZE, stdSort);

	time_t t1;
	time_t t2;

	/* Measure cpu time*/
	t1 = clock();
	cpuMergeSort(cpuSort, WORK_SIZE);
	t2 = clock();
	cpuTime = difftime(t2,t1);

	/* Measure gpu time */
	t1 = clock();
	gpuMergeSort(gpuSort, WORK_SIZE);
	t2 = clock();
	gpuTime = difftime(t2,t1);

	/* Measure std time */
	t1 = clock();
	std::sort(stdSort, stdSort+WORK_SIZE);
	t2 = clock();
	stdTime = difftime(t2,t1);

	//for (unsigned i = 0; i < WORK_SIZE; ++i)std::cout<<data[i]<<' '<<cpuMSort[i]<<' '<<gpuMSort[i]<<'\n';

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
	double cpuTime,gpuTime,stdTime;

	test(65535,cpuTime,gpuTime,stdTime);
	std::cout<<"cpu: "<<cpuTime<<", gpu: "<<gpuTime<<" std: "<<stdTime<<'\n';

	test(65535,cpuTime,gpuTime,stdTime);
	std::cout<<"cpu: "<<cpuTime<<", gpu: "<<gpuTime<<" std: "<<stdTime<<'\n';


	return 0;
}

