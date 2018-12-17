/*
 * cpuMerge.cpp
 *
 *  Created on: Dec 16, 2018
 *      Author: Orai Dezso Gergely
 */

#include "cpuMerge.hpp"

void cpuMergeSort(float *data, unsigned size)
{
	for (unsigned cnt = 0; cnt < size; ++cnt) data[cnt] = 1.0/data[cnt];
}

