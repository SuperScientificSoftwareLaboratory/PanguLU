#ifndef QUEUE_H
#define QUEUE_H

#include "mynd_functionset.h"

//	common queue
reordering_int_t mynd_init_queue(reordering_int_t ptr, reordering_int_t *bndptr, reordering_int_t nvtxs)
{
	mynd_set_value_int(nvtxs, -1, bndptr);
	return 0;
}

/*************************************************************************/
/*! Execution process:	(n -> empty)
		nbnd:	 5
		bndind:	 6  4  5  9  2  n  n  n  n  n
		bndptr:	-1 -1  4 -1  1  2  0 -1 -1  3
	aftering insert 8
		nbnd:	 6
		bndind:	 6  4  5  9  2  8  n  n  n  n
		bndptr:	-1 -1  4 -1  1  2  0 -1  5  3
 */
/**************************************************************************/
																//  /\		//
reordering_int_t mynd_insert_queue(reordering_int_t nbnd, reordering_int_t *bndptr, reordering_int_t *bndind, reordering_int_t vertex)// /  \		//
{													//				||
	bndind[nbnd]   = vertex;						//	bndind[5] = 8
	bndptr[vertex] = nbnd;							//	bndptr[8] = 5
	nbnd ++;										//	nbnd      = 6

	return nbnd;
}

/*************************************************************************/
/*! Execution process:	(n -> empty)
		nbnd:	 6
		bndind:	 6  4  5  9  2  8  n  n  n  n
		bndptr:	-1 -1  4 -1  1  2  0 -1  5  3
	aftering delete 4
		nbnd:	 5
		bndind:	 6  8  5  9  2  n  n  n  n  n
		bndptr:	-1 -1  4 -1 -1  2  0 -1  1  3
 */
/**************************************************************************/
																//  /\		//
reordering_int_t mynd_delete_queue(reordering_int_t nbnd, reordering_int_t *bndptr, reordering_int_t *bndind, reordering_int_t vertex)// /  \		//
{													//				||
	bndind[bndptr[vertex]]   = bndind[nbnd - 1];	//	bndind[1] = 8
	bndptr[bndind[nbnd - 1]] = bndptr[vertex];		//	bndptr[8] = 1
	bndptr[vertex] = -1;							//	bndptr[4] = -1
	nbnd --;										//	nbnd      = 5

	return nbnd;
}

#endif