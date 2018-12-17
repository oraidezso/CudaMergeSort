/*
 * cpuMerge.cpp
 *
 *  Created on: Dec 16, 2018
 *      Author: Orai Dezso Gergely
 */

#include "cpuMerge.hpp"


void merge(float *l, float *r, float *to, float *end, int length){
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

void cpuMergeSort(float *data, unsigned int size)
{
	float *tmp = new float[size];
	float *akt = data;
	float *next = tmp;

	for (unsigned length = 1; length < size; length *= 2){
		float *end=akt+size;
		for(unsigned col = 0; col< size; col+=2*length){
			merge(akt + col, akt + col + length, next + col, end, length);
		}
		float *c = akt;
		akt=next;
		next=c;
	}
	if(akt!=data)for(unsigned i=0;i<size;++i)data[i]=akt[i];

	delete[] tmp;
}

