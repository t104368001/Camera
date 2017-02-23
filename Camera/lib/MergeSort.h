#include "TopDown.h"
 
void TopDown::Sort(int* array, int length)
{
    int* workArray = new int[length];
    Sort(array, workArray, length, 0, length);
}
 
void TopDown::Sort(int* array, int* workArray, int length, int start, int count)
{
    if (count < 2)
        return;
 
    TopDown::Sort(array, workArray, length, start, count / 2);
    TopDown::Sort(array, workArray, length, start + count / 2, count - count / 2);
    TopDown::Merge(array, workArray, length, start, count / 2, start + count / 2, count - count / 2);
}
 
void TopDown::Merge(int* array, int* workArray, int length, int leftStart, int leftCount, int rightStart, int rightCount)
{
    int i = leftStart, j = rightStart, leftBound = leftStart + leftCount, rightBound = rightStart + rightCount, index = leftStart;
    while (i < leftBound || j < rightBound)
    {
        if (i < leftBound && j < rightBound)
        {
            if (array[j] < array[i])
                workArray[index] = array[j++];
            else
                workArray[index] = array[i++];
        }
        else if (i < leftBound)
            workArray[index] = array[i++];
        else
            workArray[index] = array[j++];
        ++index;
    }
    for (i = leftStart; i < index; ++i)
        array[i] = workArray[i];
}