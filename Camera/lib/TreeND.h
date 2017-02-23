//Template Tree ND class definition
#ifndef _TREEND_H_
#define _TREEND_H_

#include <iostream>
#include <string>
#include "Tree.h"

using namespace std;

//Tree 2D class-template declaration
template <typename NODETYPE>
class TreeND : public Tree<NODETYPE>{
private:
	TreeNodeND<NODETYPE> *rootPtr;
	int dataCount;
protected:
	//utility functions
	void insertNodeHelper(TreeNodeND <NODETYPE> **, NODETYPE *);
	void preOrderHelper(TreeNodeND <NODETYPE> *) const;
	void inOrderHelper(TreeNodeND <NODETYPE> *) const;
	void postOrderHelper(TreeNodeND <NODETYPE> *) const;
public:
	TreeND();
	TreeND(int);
	void insertNode(NODETYPE *);
	void preOrderTraversal() const;
	void inOrderTraversal() const;
	void postOrderTraversal() const;
};

//Tree ND constructor
template<typename NODETYPE>
TreeND<NODETYPE>::TreeND(){
	dataCount = 2;	//Default data array count 
	rootPtr = 0;	//initally empty
}

//Tree ND constructor
template<typename NODETYPE>
TreeND<NODETYPE>::TreeND(int count){
	dataCount = count;	//Default data array count 
	rootPtr = 0;	//initally empty
}

//insert node in Tree
template<typename NODETYPE>
void TreeND<NODETYPE>::insertNode(NODETYPE *value){
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
		string str = "(";
		inOrderHelper(ptr->leftPtr);	//traversal left subtree
		for (int i = 0; i < this->dataCount; i++){
			NODETYPE tmp = ptr->data[i];	//process node
			str += to_string(tmp);
			str += ",";
		}
		str.erase(str.length() - 1);
		str += ")";
		cout << str << endl;
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
		string str = "(";
		inOrderHelper(ptr->leftPtr);	//traversal left subtree
		for (int i = 0; i < this->dataCount; i++){
			NODETYPE tmp = ptr->data[i];	//process node
			str += to_string(tmp);
			str += ",";
		}
		str.erase(str.length() - 1);
		str += ")";
		cout << str << endl;
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
		string str = "(";
		inOrderHelper(ptr->leftPtr);	//traversal left subtree
		for (int i = 0; i < this->*dataCount; i++){
			NODETYPE tmp = ptr->data[i];	//process node
			str += to_string(tmp);
			str += ",";
		}
		str.erase(str.length() - 1);
		str += ")";
		cout << str << endl;
	}//end if 
}//end function postOrderHelper
#endif /*_TREEND_H_*/