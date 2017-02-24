# Library 

## TreeNode
程式碼：[TreeNode.h](TreeNode.h)

建置樹所需要的樹節點

TreeNode
-	一般樹的節點

來源至：C++程式設計藝術 - 第20章 資料結構 圖20.20

TreeNodeND
-	動態樹的節點

## Tree
程式碼：[Tree.h](Tree.h)

建置一般樹

Tree  
-	公開函數 - 提供插入，搜尋(前置，中置，後置)
-	保護函數 - 實線 公開函數的方法

Node
-	using [TreeNode.h](TreeNode.h) 的 TreeNode

來源至：C++程式設計藝術 - 第20章 資料結構 圖20.21

## TreeND
建置動態節點樹

程式碼：[TreeND.h](TreeND.h)

TreeND   繼承自[Tree.h](Tree.h)
-	複寫插入，搜尋 (用TreeNodeND，所以大致與Tree的方法一樣)
-	新增Data 的印列與小數點的無效0的修正

Node
-	using [TreeNodeND.h](TreeNodeND.h) 的 TreeNodeND

## Sort
排序演算法

程式碼：[Sort.h](Sort.h)

MergeSort
-	將數列對分成兩個子數列，並遞回對分

參考至：[[演算法] 合併排序法(Merge Sort)](http://notepad.yehyeh.net/Content/Algorithm/Sort/Merge/Merge.php)
	