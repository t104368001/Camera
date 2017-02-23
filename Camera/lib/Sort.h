//Template Sort class definition
#ifndef _SORT_H_
#define _SORT_H_

#include <iostream>

using namespace std;

//Merge Sort template declaration
template <typename DATATYPE>
class MergeSort{
private:
	static void sort(DATATYPE *, DATATYPE *, int, int);
	static void merge(DATATYPE *, DATATYPE *, int, int, int, int);
public:
	static void print(DATATYPE *, int);
	static void sort(DATATYPE *, int);
};

//Merge sort:public print function
template <typename DATATYPE>
void MergeSort<DATATYPE>::print(DATATYPE *ptrArray, int length){
	for (int i = 0; i < length; i++){
		cout << ptrArray[i] << " ";
	}
	cout << endl;
}

//Merge sort:public sort function
template <typename DATATYPE>
void MergeSort<DATATYPE>::sort(DATATYPE *ptrArray, int length){
	DATATYPE *array = new DATATYPE[length];
	sort(ptrArray, array, 0, length);
}

//Merge sort:private sort function [recursive]
template <typename DATATYPE>
void MergeSort<DATATYPE>::sort(DATATYPE *ptrArray, DATATYPE *tmpArray, int start, int end){
	if (end < 2)
		return;
	int middle = end / 2;
	sort(ptrArray, tmpArray, start, middle);	//left array
	sort(ptrArray, tmpArray, start + middle, end - middle);	//right array
	merge(ptrArray, tmpArray, start, middle, start + middle, end - middle);
}

//Merge sort:private sort function [recursive]
template <typename DATATYPE>
void MergeSort<DATATYPE>::merge(DATATYPE *array, DATATYPE *tmpArray, int leftStart, int leftEnd, int rightStart, int rightEnd){
	int left = leftStart, right = rightStart, leftBound = leftStart + leftEnd, rightBound = rightStart + rightEnd, index = leftStart;
	while (left < leftBound || right < rightBound){
		if (left < leftBound && right < rightBound){
			if (array[left] < array[right])
				tmpArray[index] = array[left++];
			else
				tmpArray[index] = array[right++];
		}
		else if (left < leftBound){
			tmpArray[index] = array[left++];
		}else{
			tmpArray[index] = array[right++];
		}
		index++;
	}
	for (int i = leftStart; i < index; i++)
		array[i] = tmpArray[i];
}

#endif /* _SORT_H_*/