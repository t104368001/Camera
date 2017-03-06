#include <iostream>
#include "lib/Tree.h"	//Tree class definition
#include "lib/Sort.h"	//Sort class definition
#include "vector"

using namespace std;

int main() {
//	int ar[] = { 8, 3, 10, 1, 6, 14, 4, 7, 13 };
//	MergeSort <int>::sort(ar, sizeof ar / sizeof ar[0]);
//	MergeSort <int>::print(ar, size);
	
//	TreeND<double> intTree2D;
	TreeND<double> intTree2D;

	

	vector< vector<double> > ar2 = { { 8, 23 }, { 3.01, 17.33 }, { 10, -0.1 }, { 1 }, { 6 }, { 14 }, { 4 }, { 7 }, { 13, 20 } };
//	vector< vector<int> > ar2 = { { 0, 5 }, { 0, 2 }, { 1, 3 }, { 2, 1 }, { 3, 4 }, { 4, 2 }, { 5, 5 }, { 6, 3 }, { 7, 1 }, { 6, 0 } };
//	KDTree<int> intTree2D;
	
	intTree2D.buildTree(ar2);

	cout << "\nPreorder traversal\n";
	//intTree2D.preOrderTraversal(false);
	intTree2D.preOrderTraversal(false);
	cout << endl;

	system("pause");
	return 0;
}