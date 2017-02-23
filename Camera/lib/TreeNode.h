//Template TreeNode class definition
#ifndef _TREENODE_H_
#define _TREENODE_H_

//Forward declaration of class Tree
template <typename NODETYPE> class Tree;

//TreeNode class-template definition
template <typename NODETYPE>
class TreeNode{
	friend class Tree<NODETYPE>;
private:
	TreeNode <NODETYPE> *leftPtr;	//pointer to left subtree
	NODETYPE data;					//tree node data
	TreeNode <NODETYPE> *rightPtr;	//pointer to right subtree
public:
	/*
	TreeNode constructor
	leftPtr is pointer to left subtree
	data is tree node data
	rightPtr is pointer to right subtree
	*/
	TreeNode(const NODETYPE &d) : leftPtr(0), data(d), rightPtr(0){
		//empty constructor
	}
	/*
	Get node's data
	*/
	NODETYPE getData() const {
		return data;
	}
};

//Forward declaration of class KD Tree
template <typename NODETYPE> class TreeND;

//TreeNode class-template definition
template <typename NODETYPE>
class TreeNodeND{
	friend class TreeND<NODETYPE>;
private:
	TreeNodeND <NODETYPE> *leftPtr;	//pointer to left subtree
	NODETYPE *data;					//tree ND node data
	TreeNodeND <NODETYPE> *rightPtr;	//pointer to right subtree
public:
	/*
	TreeNode constructor
	leftPtr is pointer to left subtree
	data is tree node data
	rightPtr is pointer to right subtree
	*/
	TreeNodeND(NODETYPE *d) : leftPtr(0), data(d), rightPtr(0){
		//empty constructor
	}
	/*
	Get node's data
	*/
	NODETYPE getData() const {
		return data;
	}
};
#endif /* _TREENODE_H_ */