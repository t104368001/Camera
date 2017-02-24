#include <iostream>
#include <iomanip>
#include "lib/Tree.h"	//Tree class definition
#include "lib/TreeND.h"	//Tree class definition
#include "lib/Sort.h"	//Sort class definition

using namespace std;

int main() {
	//const int size = 11;
	//int ar[] = {7, 4, 1, 5, 16, 8, 11, 12, 15, 9, 2};

//	MergeSort <int>::sort(ar, size);
//	MergeSort <int>::print(ar, size);

	TreeND<float> floatTree2D;
	TreeND<double> intTree2D;

	double ar2[][3] = { { 1040.51, 10 }, { 550, 10.0 }, { 100.01 } };

	//intTree2D.insertNode(ar2[0]);
	cout << "sizeof ar2 = " << sizeof ar2 << ", sizeof ar2[0] = " << sizeof ar2[0] << ", sizeof / sizeof ar2[0] = " << sizeof ar2 / sizeof ar2[0] << endl;
	//TreeND<double> intTree2D(sizeof ar2 / sizeof ar2[0]);

	for (int i = 0; i < sizeof ar2 / sizeof ar2[0]; i++){
		intTree2D.insertNode(ar2[i], sizeof ar2[i] / sizeof ar2[i][0]);
	}

	cout << "\nPreorder traversal\n";
	intTree2D.preOrderTraversal();

	cout << endl;

	system("pause");
	return 0;
}