//Template Tree class definition
#ifndef _TREE_H_
#define _TREE_H_

#include <iostream>
#include <algorithm>
#include <string>
#include <stack>
#include "vector"
#include "TreeNode.h"

using namespace std;

/*************************************/
/* Tree class-template declaration    /
/*************************************/
template <typename NODETYPE> 
class Tree{
private:
	TreeNode<NODETYPE> *rootPtr;
protected:
	//utility functions
	void insertNodeHelper(TreeNode <NODETYPE> **, const NODETYPE &);
	void preOrderHelper(TreeNode <NODETYPE> *) const;
	void inOrderHelper(TreeNode <NODETYPE> *) const;
	void postOrderHelper(TreeNode <NODETYPE> *) const;
public:
	/*
	Tree constructor
	*/
	Tree();
	void insertNode(const NODETYPE &);
	void preOrderTraversal() const;
	void inOrderTraversal() const;
	void postOrderTraversal() const;
};

//Tree constructor
template<typename NODETYPE>
Tree<NODETYPE>::Tree(){
	rootPtr = 0;	//initally empty
}

//insert node in Tree
template<typename NODETYPE>
void Tree<NODETYPE>::insertNode(const NODETYPE &value){
	insertNodeHelper(&rootPtr, value);
}

//utility function called by insertNode; receives a pointer
//to a pointer so that the function can modify pointer's value
template <typename NODETYPE>
void Tree<NODETYPE> ::insertNodeHelper(TreeNode<NODETYPE> **ptr, const NODETYPE &value){
	//subtree is empty; create new TreeNode containing value
	if (*ptr == 0)
		*ptr = new TreeNode<NODETYPE>(value);
	else {	//subtree is not empty
		if (value < (*ptr)->data)
			//data to insert is less than data in current node 
			insertNodeHelper(&((*ptr)->leftPtr), value);
		else{
			//data to insert is more than data in current node 
			if (value >(*ptr)->data)
				insertNodeHelper(&((*ptr)->rightPtr), value);
			else //duplicate data value ignored
				cout << value << " : duplicate" << endl;
		} //end value than the size 
	}//end checking subtree
}//end function insertNodeHelper

//begin preorder traversal of Tree
template <typename NODETYPE>
void Tree<NODETYPE>::preOrderTraversal() const{
	preOrderHelper(rootPtr);
}

//utility function to perform preorder traversal of Tree
template <typename NODETYPE>
void Tree<NODETYPE>::preOrderHelper(TreeNode <NODETYPE> *ptr) const{
	if (ptr != 0){
		cout << ptr->data << ' ';	//process node
		preOrderHelper(ptr->leftPtr);	//traversal left subtree
		preOrderHelper(ptr->rightPtr);	//traversal right subtree
	}//end if 
}//end function preOrderHelper

//begin inorder traversal of Tree
template <typename NODETYPE>
void Tree<NODETYPE>::inOrderTraversal() const{
	inOrderHelper(rootPtr);
}

//utility function to perform inorder traversal of Tree
template <typename NODETYPE>
void Tree<NODETYPE>::inOrderHelper(TreeNode <NODETYPE> *ptr) const{
	if (ptr != 0){
		inOrderHelper(ptr->leftPtr);	//traversal left subtree
		cout << ptr->data << ' ';	//process node
		inOrderHelper(ptr->rightPtr);	//traversal right subtree
	}//end if 
}//end function inOrderHelper

//begin postorder traversal of Tree
template <typename NODETYPE>
void Tree<NODETYPE>::postOrderTraversal() const{
	postOrderHelper(rootPtr);
}

//utility function to perform postorder traversal of Tree
template <typename NODETYPE>
void Tree<NODETYPE>::postOrderHelper(TreeNode <NODETYPE> *ptr) const{
	if (ptr != 0){
		postOrderHelper(ptr->leftPtr);	//traversal left subtree
		postOrderHelper(ptr->rightPtr);	//traversal right subtree
		cout << ptr->data << ' ';	//process node
	}//end if 
}//end function postOrderHelper

/****************************************************/
/* Template Tree 2D class-template declaration       /
/****************************************************/
template <typename NODETYPE>
class TreeND : public Tree<NODETYPE>{
private:
protected:
	TreeNodeND<NODETYPE> *rootPtr;		//tree root
	int count;		//tree stratum
	vector< vector<NODETYPE> > pointVector;	//store data array

	//utility functions
	void insertNodeHelper(TreeNodeND <NODETYPE> **, NODETYPE *, int, int);
	void preOrderHelper(TreeNodeND <NODETYPE> *) const;
	void preOrderHelper(TreeNodeND <NODETYPE> *, bool) const;
	void inOrderHelper(TreeNodeND <NODETYPE> *) const;
	void postOrderHelper(TreeNodeND <NODETYPE> *) const;
	void printfData(TreeNodeND <NODETYPE> *) const;
	string deleteZero(NODETYPE) const;
public:
	TreeND() : rootPtr(0){};
	void buildTree(vector< vector<NODETYPE> >);
	void insertNode(NODETYPE *, int);
	void preOrderTraversal() const;
	void preOrderTraversal(bool) const;
	void inOrderTraversal() const;
	void postOrderTraversal() const;
	void printAllElement();
	string printElement(NODETYPE *, int);
};

//get Array to build tree
template<typename NODETYPE>
void TreeND<NODETYPE>::buildTree(vector< vector<NODETYPE> > arrayVector){
	pointVector = arrayVector;	//stored pointVector from arrayVector
	for (int i = 0; i < pointVector.size(); i++){
		NODETYPE *pointArray = &pointVector[i][0];	//Type conversion : vector to array
		insertNode(pointArray, pointVector[i].size());	//insert tree node
	}//end for
}//end buildTree

//print all item in array
template<typename NODETYPE>
void TreeND<NODETYPE>::printAllElement(){
	for (int i = 0; i < pointVector.size(); i++){
		vector <int> ::iterator iter = pointVector[i].begin();
		for (int j = 0; iter != pointVector[i].end(); ++iter, ++j)
			cout << *iter << " ";
		cout << endl;
	}//end for
}//end printAllElement

//print one item
template<typename NODETYPE>
string TreeND<NODETYPE>::printElement(NODETYPE *value, int length){
	string str = "(";
	for (int i = 0; i < length; i++){
		str += deleteZero(value[i]);	//process node
		str += ",";
	}
	str.erase(str.length() - 1);	//erase last ','
	str += ")";
	return str;	//return "(X,X,...)"
}//end printElement

//insert node in Tree
template<typename NODETYPE>
void TreeND<NODETYPE>::insertNode(NODETYPE *value, int length){
	insertNodeHelper(&rootPtr, value, -1, length);
}

//utility function called by insertNode; receives a pointer
//to a pointer so that the function can modify pointer's value
template <typename NODETYPE>
void TreeND<NODETYPE> ::insertNodeHelper(TreeNodeND<NODETYPE> **ptr, NODETYPE *value, int count, int length){
	count++;	//count tree deep
	//subtree is empty; create new TreeNode containing value
	if (*ptr == 0)
		*ptr = new TreeNodeND<NODETYPE>(value, count, length, false);
	else {	//subtree is not empty
		if (*value < (*ptr)->data[0])
			//data to insert is less than data in current node 
			insertNodeHelper(&((*ptr)->leftPtr), value, count, length);
		else{
			//data to insert is more than data in current node 
			if (*value > (*ptr)->data[0])
				insertNodeHelper(&((*ptr)->rightPtr), value, count, length);
			else //duplicate data value ignored
				cout << printElement(value, length) << " : duplicate" << endl;
		} //end value than the size
	}//end checking subtree
}//end function insertNodeHelper

//begin preorder traversal of Tree
template <typename NODETYPE>
void TreeND<NODETYPE>::preOrderTraversal() const{
	preOrderHelper(rootPtr);
}

//utility function to perform preorder traversal of Tree
template <typename NODETYPE>
void TreeND<NODETYPE>::preOrderHelper(TreeNodeND <NODETYPE> *ptr) const{
	if (ptr != 0){
		printfData(ptr);				//process node
		preOrderHelper(ptr->leftPtr);	//traversal left subtree
		preOrderHelper(ptr->rightPtr);	//traversal right subtree
	}//end if 
}//end function preOrderHelper

//begin preorder traversal of Tree
template <typename NODETYPE>
void TreeND<NODETYPE>::preOrderTraversal(bool printBranch) const{
	preOrderHelper(rootPtr, printBranch);
}

//utility function to perform preorder traversal of Tree (print branch or not version)
template <typename NODETYPE>
void TreeND<NODETYPE>::preOrderHelper(TreeNodeND <NODETYPE> *ptr, bool printBranch) const{
	if (ptr != 0){
		//print branch or not
		if (printBranch || !(ptr->branch))
			printfData(ptr);				//process node
		preOrderHelper(ptr->leftPtr, printBranch);	//traversal left subtree
		preOrderHelper(ptr->rightPtr, printBranch);	//traversal right subtree
	}//end if 
}//end function preOrderHelper

//begin inorder traversal of Tree
template <typename NODETYPE>
void TreeND<NODETYPE>::inOrderTraversal() const{
	inOrderHelper(rootPtr);
}

//utility function to perform inorder traversal of Tree
template <typename NODETYPE>
void TreeND<NODETYPE>::inOrderHelper(TreeNodeND <NODETYPE> *ptr) const{
	if (ptr != 0){
		inOrderHelper(ptr->leftPtr);	//traversal left subtree
		printfData(ptr);				//process node
		inOrderHelper(ptr->rightPtr);	//traversal right subtree
	}//end if 
}//end function inOrderHelper

//begin postorder traversal of Tree
template <typename NODETYPE>
void TreeND<NODETYPE>::postOrderTraversal() const{
	postOrderHelper(rootPtr);
}

//utility function to perform postorder traversal of Tree
template <typename NODETYPE>
void TreeND<NODETYPE>::postOrderHelper(TreeNodeND <NODETYPE> *ptr) const{
	if (ptr != 0){
		postOrderHelper(ptr->leftPtr);	//traversal left subtree
		postOrderHelper(ptr->rightPtr);	//traversal right subtree
		printfData(ptr);				//process node
	}//end if 
}//end function postOrderHelper

//utility function to print Data of Tree
template <typename NODETYPE>
void TreeND<NODETYPE>::printfData(TreeNodeND <NODETYPE> *ptr) const{
	if (ptr->data != 0){	//data is exist
		string str = "(";
		cout << ptr->stratum << " ";	//output data in tree stratum
		for (int i = 0; i < ptr->length; i++){
			str += deleteZero(ptr->data[i]);	//process node
			str += ",";
		}
		str.erase(str.length() - 1);			//erase last ','
		str += ")";
		cout << str << endl;
	}//end if 
}//end function printfData

//utility function to adjustment flaot point of Data
template <typename NODETYPE>
string TreeND<NODETYPE>::deleteZero(NODETYPE data) const{
	string str = to_string(data);			//NODETYPE transform string
	if (typeid(data) != typeid(int)){		//check out data type
		char tmp[1024];						//tmporary space
		strncpy_s(tmp, str.c_str(), sizeof(tmp));	//copy data from String to char array
		int i = str.size() - 1;				//get string length, except '\0' 
		for (; i >= 0; i--){				//check flaot point 0
			if (tmp[i] != '0'){
				if (tmp[i] == '.')			//if tmp is integer no float point  
					i--;
				break;
			}//end if
		}//end for 
		str = str.substr(0, i + 1);			//string substring
	}//end if
	return str;
}//end function adjustment point

/****************************************************/
/* Template KD-Tree class-template declaration		 /
/****************************************************/
//order x axis
struct orderX
{
	bool operator() (vector<int> a, vector<int> b) const {
		return a[0] < b[0];	//check two data size
	}
};//end struct

//order y axis
struct orderY
{
	bool operator() (vector<int> a, vector<int> b) const {
		return a[1] < b[1]; //check two data size
	}
};//end struct

//print vector< vector<int> > array
void print(vector< vector<int> > array){
	for (int i = 0; i < (int)array.size(); i++){
		vector <int> ::iterator iter = array[i].begin();
		for (int j = 0; iter != array[i].end(); ++iter, ++j)
			cout << *iter << " ";	//output => data1 " " data2 ...
			//end for
		cout << "\t";
	}//end for
	cout << endl;
}//end print function

//Kd-Tree 2D class-template declaration
template <typename NODETYPE>
class KDTree : public TreeND<NODETYPE>{
private:
	stack<vector<NODETYPE> > pointStack;	//store point in stack
	NODETYPE *pointNode;					//point to point data
protected:
	//utility functions
	//overload insertNodeHelper
	void insertNodeHelper(bool, vector< vector<NODETYPE> >, TreeNodeND <NODETYPE> **, int, int, int);
public:
	//build tree
	void buildTree(vector< vector<NODETYPE> >);
};

//get Array to build tree
template<typename NODETYPE>
void KDTree<NODETYPE>::buildTree(vector< vector<NODETYPE> > arrayVector){
	insertNodeHelper(true, arrayVector, &rootPtr, -1, 0, arrayVector.size());
}//end buildTree

//utility function called by insertNode; receives a pointer
//to a pointer so that the function can modify pointer's value
template <typename NODETYPE>
void KDTree<NODETYPE> ::insertNodeHelper(bool bXY, vector< vector<NODETYPE> > arrayVector, TreeNodeND<NODETYPE> **ptr, 
	int count, int start, int end){
	count++;	//
	if (end - start < 2){	//only one data node
		pointStack.push(arrayVector[start]);	//push data in stack
		pointNode = &pointStack.top()[0];		//get data in stack 
		*ptr = new TreeNodeND<NODETYPE>(pointNode, count, arrayVector[start].size(), false);	//build tree leaf
		return;
	}//end if
	if (bXY){
		sort(arrayVector.begin() + start, arrayVector.begin() + end, orderX());
	}//end if
	else{
		sort(arrayVector.begin() + start, arrayVector.begin() + end, orderY());
	}//end else if
	int middle = (end - start - 1) / 2 + start;	//calculation array middle
	pointStack.push(arrayVector[middle]);		//push data in stack
	pointNode = &pointStack.top()[0];			//get data in stack 
	*ptr = new TreeNodeND<NODETYPE>(pointNode, count, arrayVector[middle].size());		//build tree branch
	insertNodeHelper(!bXY, arrayVector, &((*ptr)->leftPtr), count, start, middle + 1);	//insert left subtree
	insertNodeHelper(!bXY, arrayVector, &((*ptr)->rightPtr), count, middle + 1, end);	//insert right subtree
}//end function insertNodeHelper

#endif /* _TREE_H_ */