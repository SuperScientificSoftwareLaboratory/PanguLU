#ifndef SEARCHTREE_H
#define SEARCHTREE_H

#include "mynd_functionset.h"

//  Binary Search Tree Version 1.0
void mynd_binary_search_tree_Init(binary_search_tree_t *tree)
{
    tree->nownodes = 0;
	tree->treenode = NULL;
}

binary_search_tree_t *mynd_binary_search_tree_Create()
{
    binary_search_tree_t *tree;

    tree = (binary_search_tree_t *)mynd_check_malloc(sizeof(binary_search_tree_t), "binary_search_tree_Create: tree");
    mynd_binary_search_tree_Init(tree);

    return tree;
}

void mynd_Free_Treenode(treenode_t *node)
{
	if(node != NULL)
	{
		mynd_Free_Treenode(node->left);
		mynd_Free_Treenode(node->right);
		mynd_check_free(node, sizeof(treenode_t), "Free_Treenode: node");
	}
}

void mynd_binary_search_tree_Free(binary_search_tree_t *tree)
{
	if (tree == NULL) return;
	mynd_Free_Treenode(tree->treenode);
	// mynd_check_free(tree->locator);
	tree->nownodes = 0;
}

void mynd_binary_search_tree_Destroy(binary_search_tree_t *tree)
{
	if (tree == NULL) return;
	mynd_binary_search_tree_Free(tree);
	mynd_check_free(tree, sizeof(binary_search_tree_t), "binary_search_tree_Destroy: tree");
}

reordering_int_t mynd_binary_search_tree_Length(binary_search_tree_t *tree)
{
	return tree->nownodes;
}

treenode_t *mynd_Create_TreeNode(reordering_int_t val, reordering_int_t key)
{
	treenode_t *newnode = (treenode_t *)mynd_check_malloc(sizeof(treenode_t), "Create_TreeNode: newnode");
    
	newnode->val = val;
	newnode->key = key;
    newnode->left = newnode->right = NULL;
    
	return newnode;
}

treenode_t *mynd_Insert_TreeNode(treenode_t *node, reordering_int_t val, reordering_int_t key, reordering_int_t *nownodes)
{
	// if empty
    if (node == NULL) 
	{
		node = mynd_Create_TreeNode(val, key);
		(*nownodes)++;
		return node;
	}

    // if less than
    if (val < node->val)
        node->left = mynd_Insert_TreeNode(node->left, val, key, nownodes);
    // if greater than
    else if (val > node->val)
        node->right = mynd_Insert_TreeNode(node->right, val, key, nownodes);
	
	//	if equal
	else
		node->key += key;

    return node;
}

void mynd_binary_search_tree_Insert(binary_search_tree_t *tree, reordering_int_t val, reordering_int_t key)
{
	treenode_t *root = tree->treenode;
	
	root = mynd_Insert_TreeNode(root, val, key, &tree->nownodes);
	tree->treenode = root;

	return ;
}

reordering_int_t mynd_InorderTraversal_TreeNode(treenode_t *root, reordering_int_t *dst1, reordering_int_t *dst2, reordering_int_t *ptr) 
{
    if (root != NULL) 
	{
        *ptr = mynd_InorderTraversal_TreeNode(root->left,dst1,dst2,ptr);
		//	do operation
        // printf("root->val=%"PRIDX" root->key=%"PRIDX" ", root->val,root->key);
		// if(dst != NULL) 
		// {
			dst1[*ptr] = root->val;
			dst2[*ptr] = root->key;
			// printf("root->val=%"PRIDX" dst[ptr]=%"PRIDX" ptr=%"PRIDX"\n",root->val, dst[*ptr], *ptr);
			(*ptr) ++;
		// }

        *ptr = mynd_InorderTraversal_TreeNode(root->right,dst1,dst2,ptr);
    }

	return *ptr;
}

void mynd_binary_search_tree_Traversal(binary_search_tree_t *tree, reordering_int_t *dst1, reordering_int_t *dst2)
{
	treenode_t *root = tree->treenode;
	reordering_int_t ptr = 0;

	mynd_InorderTraversal_TreeNode(root, dst1, dst2, &ptr);
}



//  Binary Search Tree Version 2.0
void mynd_binary_search_tree_Init2(binary_search_tree2_t *tree, reordering_int_t size)
{
    tree->nownodes = 0;
	tree->maxnodes = size;
	tree->treenode = (treenode2_t *)mynd_check_malloc(sizeof(treenode2_t) * size, "binary_search_tree2_Init: tree->treenode");

	for(reordering_int_t i = 0;i < size;i++)
	{
		tree->treenode[i].val = -1;
		tree->treenode[i].key = 0;
	}
}

binary_search_tree2_t *mynd_binary_search_tree_Create2(reordering_int_t size)
{
    binary_search_tree2_t *tree;

    tree = (binary_search_tree2_t *)mynd_check_malloc(sizeof(binary_search_tree2_t), "binary_search_tree2_Create: tree");
    
	mynd_binary_search_tree_Init2(tree, size);

    return tree;
}

void mynd_exam_binary_search_tree2(binary_search_tree2_t *tree)
{
	printf("val:");
	for(reordering_int_t i = 0;i < tree->maxnodes;i++)
	{
		printf("%"PRIDX" ",tree->treenode[i].val);
	}
	printf("\n");
	printf("key:");
	for(reordering_int_t i = 0;i < tree->maxnodes;i++)
	{
		printf("%"PRIDX" ",tree->treenode[i].key);
	}
	printf("\n");
}

void mynd_exam_binary_search_tree2_flag(binary_search_tree2_t *tree)
{
	reordering_int_t flag = 0;
	printf("val:");
	for(reordering_int_t i = 0;i < tree->maxnodes;i++)
	{
		printf("%"PRIDX" ",tree->treenode[i].val);
		if(tree->treenode[i].val != -1)
			flag = 1;
	}
	printf("\n");
	if(flag == 1)
		printf("flag=1\n");
	printf("key:");
	for(reordering_int_t i = 0;i < tree->maxnodes;i++)
	{
		printf("%"PRIDX" ",tree->treenode[i].key);
		if(tree->treenode[i].key != 0)
			flag = 2;
	}
	printf("\n");
	if(flag == 2)
		printf("flag=2\n");
}

void mynd_binary_search_tree_Free2(binary_search_tree2_t *tree)
{
	if (tree == NULL) return;
	mynd_check_free(tree->treenode, sizeof(treenode2_t) * tree->maxnodes, "binary_search_tree_Free2: tree->treenode");
	tree->nownodes = 0;
	tree->maxnodes = 0;
}

void mynd_binary_search_tree_Destroy2(binary_search_tree2_t *tree)
{
	if (tree == NULL) return;
	mynd_binary_search_tree_Free2(tree);
	mynd_check_free(tree, sizeof(binary_search_tree2_t), "binary_search_tree_Destroy2: tree");
}

reordering_int_t mynd_binary_search_tree_Length2(binary_search_tree2_t *tree)
{
	return tree->nownodes;
}

void mynd_Insert_TreeNode2(binary_search_tree2_t *tree, reordering_int_t val, reordering_int_t key)
{
	reordering_int_t ptr = 0;
	treenode2_t *treenode = tree->treenode;

	while (treenode[ptr].val != -1) 
	{
		if(ptr >= tree->maxnodes)
		{
			printf("mynd_check_realloc\n");
			treenode = tree->treenode = (treenode2_t *)mynd_check_realloc(treenode, sizeof(treenode2_t) * tree->maxnodes * 2, sizeof(reordering_int_t) * tree->maxnodes, "Insert_TreeNode2: treenode");
			for(reordering_int_t i = tree->maxnodes;i < tree->maxnodes * 2;i++)
			{
				treenode[i].val = -1;
				treenode[i].key = 0;
			}
			tree->maxnodes *= 2;
		}

        if (treenode[ptr].val < val) 
            ptr = 2 * ptr + 2;
        else if (treenode[ptr].val > val) 
            ptr = 2 * ptr + 1; 
		else if(treenode[ptr].val == val) 
		{
			treenode[ptr].key += key;
			printf("Update: val=%"PRIDX" key=%"PRIDX" ptr=%"PRIDX"\n",val,treenode[ptr].key,ptr);
            return ;
        }
    }

	printf("Insert: val=%"PRIDX" key=%"PRIDX" ptr=%"PRIDX"\n",val,key,ptr);
    treenode[ptr].val = val;
    treenode[ptr].key = key;
    tree->nownodes++;

    return ;
}

void mynd_binary_search_tree_Insert2(binary_search_tree2_t *tree, reordering_int_t val, reordering_int_t key)
{
	mynd_Insert_TreeNode2(tree, val, key);
	printf("\n");
	return ;
}

void mynd_InorderTraversal_TreeNode2(binary_search_tree2_t *tree, treenode2_t *treenode, reordering_int_t maxnodes, reordering_int_t *dst1, reordering_int_t *dst2, reordering_int_t located, reordering_int_t *ptr) 
{
	printf("InorderTraversal_TreeNode2 1 located=%"PRIDX"\n",located);
	mynd_exam_binary_search_tree2(tree);
	if(treenode[located].val == -1)
		return;

	if (2 * located + 1 < maxnodes)
		mynd_InorderTraversal_TreeNode2(tree, treenode, maxnodes, dst1, dst2, 2 * located + 1, ptr);
	printf("located=%"PRIDX" ptr=%"PRIDX" val=%"PRIDX" key=%"PRIDX"\n",located,*ptr,treenode[located].val,treenode[located].key);
	mynd_exam_binary_search_tree2(tree);
	dst1[*ptr] = treenode[located].val;
	printf("1\n");
	mynd_exam_binary_search_tree2(tree);
	dst2[*ptr] = treenode[located].key;
	printf("2\n");
	mynd_exam_binary_search_tree2(tree);
	(*ptr)++;
	printf("3\n");
	mynd_exam_binary_search_tree2(tree);
	if(2 * located + 2 < maxnodes)
		mynd_InorderTraversal_TreeNode2(tree, treenode, maxnodes, dst1, dst2, 2 * located + 2, ptr);
}

void mynd_binary_search_tree_Traversal2(binary_search_tree2_t *tree, reordering_int_t *dst1, reordering_int_t *dst2)
{
	treenode2_t *treenode = tree->treenode;

	reordering_int_t ptr = 0;

	mynd_InorderTraversal_TreeNode2(tree, treenode, tree->maxnodes, dst1, dst2, 0, &ptr);
}

void mynd_Reset_TreeNode2(treenode2_t *treenode, reordering_int_t maxnodes, reordering_int_t located) 
{
	if(treenode[located].val == -1)
		return;
	else
	{
		// if (2 * located + 1 < maxnodes)
		// 	Reset_TreeNode2(treenode, maxnodes, located * 2 + 1);
		// treenode[located].val = -1;
		// treenode[located].key = 0;
		// if (2 * located + 2 < maxnodes)
		// 	Reset_TreeNode2(treenode, maxnodes, located * 2 + 2);
		
        if (2 * located + 1 < maxnodes)
            mynd_Reset_TreeNode2(treenode, maxnodes, 2 * located + 1);
        treenode[located].val = -1;
        treenode[located].key = 0;
        if (2 * located + 2 < maxnodes)
            mynd_Reset_TreeNode2(treenode, maxnodes, 2 * located + 2);
	}
}

void mynd_binary_search_tree_Reset2(binary_search_tree2_t *tree)
{
	treenode2_t *treenode = tree->treenode;

	mynd_Reset_TreeNode2(treenode, tree->maxnodes, 0);
}

#endif