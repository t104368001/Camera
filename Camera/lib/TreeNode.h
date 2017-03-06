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
template <typename NODETYPE> class KDTree;

//TreeNode class-template definition
template <typename NODETYPE>
class TreeNodeND{
	friend class TreeND<NODETYPE>;
	friend class KDTree<NODETYPE>;
private:
	int stratum;	//data stratum
	int length;		//data length
	bool branch;	//store branch or leaf
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
	TreeNodeND(NODETYPE *d) : leftPtr(0), data(d), rightPtr(0), stratum(0){
		//empty constructor
	}
	//Default data is branch
	TreeNodeND(NODETYPE *d, int number, int length) : leftPtr(0), data(d), rightPtr(0), stratum(number), length(length), branch(true){
		//empty constructor
	}
	TreeNodeND(NODETYPE *d, int number, int length, bool branch) : leftPtr(0), data(d), rightPtr(0), stratum(number), length(length), branch(branch){
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