/*
 * gpuMerge.cuh
 *
 *  Created on: Dec 16, 2018
 *      Author: Orai Dezso Gergely
 */

#ifndef GPUMERGE_H_
#define GPUMERGE_H_

void gpuMergeSort(float *data, unsigned size);
bool findCudaDevice();

#endif /* GPUMERGE_H_ */
