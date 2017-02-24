//Template Tree ND class definition
#ifndef _TREEND_H_
#define _TREEND_H_

#include <iostream>
#include <string>
#include <iomanip>
#include "Tree.h"

using namespace std;

//Tree 2D class-template declaration
template <typename NODETYPE>
class TreeND : public Tree<NODETYPE>{
private:
	TreeNodeND<NODETYPE> *rootPtr;
	int length;
protected:
	//utility functions
	void insertNodeHelper(TreeNodeND <NODETYPE> **, NODETYPE *);
	void preOrderHelper(TreeNodeND <NODETYPE> *) const;
	void inOrderHelper(TreeNodeND <NODETYPE> *) const;
	void postOrderHelper(TreeNodeND <NODETYPE> *) const;
	void printfData(TreeNodeND <NODETYPE> *) const;
	string deleteZero(NODETYPE) const;
public:
	TreeND() : rootPtr(0){};
	TreeND(int);
	void insertNode(NODETYPE *, int);
	void preOrderTraversal() const;
	void inOrderTraversal() const;
	void postOrderTraversal() const;
};

//insert node in Tree
template<typename NODETYPE>
void TreeND<NODETYPE>::insertNode(NODETYPE *value, int length){
	this->length = length;
	insertNodeHelper(&rootPtr, value);
}

//utility function called by insertNode; receives a pointer
//to a pointer so that the function can modify pointer's value
template <typename NODETYPE>
void TreeND<NODETYPE> ::insertNodeHelper(TreeNodeND<NODETYPE> **ptr, NODETYPE *value){
	//subtree is empty; create new TreeNode containing value
	if (*ptr == 0)
		*ptr = new TreeNodeND<NODETYPE>(value);
	else {	//subtree is not empty
		if (*value < (*ptr)->data[0])
			//data to insert is less than data in current node 
			insertNodeHelper(&((*ptr)->leftPtr), value);
		else{
			//data to insert is more than data in current node 
			if (*value > (*ptr)->data[0])
				insertNodeHelper(&((*ptr)->rightPtr), value);
			else //duplicate data value ignored
				cout << value << "duplicate" << endl;
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
	if (ptr->data != 0){
		string str = "(";
		for (int i = 0; i < this->length; i++){
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
	string str = to_string(data);
	if (typeid(data) != typeid(int)){		//check out data type
		char tmp[1024];
		strncpy_s(tmp, str.c_str(), sizeof(tmp));	//copy data from String to char array
		int i = str.size() - 1;				//get string length, except '\0' 
		for (; i >= 0; i--){				//check flaot point 0
			if (tmp[i] != '0'){
				if (tmp[i] == '.')			//if tmp is integer no float point  
					i--;
				break;
			}
		}
		str = str.substr(0, i + 1);			//string substring
	}
	return str;
}//end function adjustment point

#endif /*_TREEND_H_*/