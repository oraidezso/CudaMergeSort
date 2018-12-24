# General-purpose computing on graphics processing units assignment: Merge sort in cuda.

## Table of Contents
- [General-purpose computing on graphics processing units assignment: Merge sort in cuda.](#general-purpose-computing-on-graphics-processing-units-assignment--merge-sort-in-cuda)
	- [Table of Contents](#table-of-contents)
	- [Basic Merge Sort for comparison](#basic-merge-sort-for-comparison)

## Basic Merge Sort for comparison

---

```c++
void merge(float *l, float *r, float *to, float *end, int length){
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

void MergeSort(float *data, unsigned int size)
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
```