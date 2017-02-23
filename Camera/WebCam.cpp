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

	//TreeND<float> intTree2D;
	//Tree<int> intTree;

	double ar2[][2] = { { 40.2, 20 }, { 50, 1 } };

	//intTree2D.insertNode(ar2[0]);
	TreeND<double> intTree2D(sizeof ar2 / sizeof ar2[0]);


	for (int i = 0; i < sizeof ar2 / sizeof ar2[0]; i++){
		intTree2D.insertNode(ar2[i]);
	}

	cout << "\nPreorder traversal\n";
	intTree2D.preOrderTraversal();

	cout << endl;

	system("pause");
	return 0;
}