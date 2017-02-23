//Template Tree class definition
#ifndef _TREE_H_
#define _TREE_H_

#include <iostream>
#include "TreeNode.h"

using namespace std;

//Tree class-template declaration
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
				cout << value << "duplicate" << endl;
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


#endif /* _TREE_H_ */