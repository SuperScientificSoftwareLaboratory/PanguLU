#ifndef PRIORITYQUEUE_H
#define PRIORITYQUEUE_H

#include "mynd_functionset.h"

//	priority queue
/*************************************************************************/
/*! This function initializes the data structures of the priority queue */
/**************************************************************************/
void mynd_priority_queue_Init(priority_queue_t *queue, reordering_int_t maxnodes)
{
	queue->nownodes = 0;
	queue->maxnodes = maxnodes;
	queue->heap     = (node_t *)mynd_check_malloc(sizeof(node_t) * maxnodes, "priority_queue_Init: heap");
	queue->locator  = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * maxnodes, "priority_queue_Init: locator");
	for(reordering_int_t i = 0;i < maxnodes;i++)
		queue->locator[i] = -1;
}

/*************************************************************************/
/*! This function creates and initializes a priority queue */
/**************************************************************************/
priority_queue_t *mynd_priority_queue_Create(reordering_int_t maxnodes)
{
	priority_queue_t *queue; 

	queue = (priority_queue_t *)mynd_check_malloc(sizeof(priority_queue_t), "mynd_priority_queue_Create: queue");
	mynd_priority_queue_Init(queue, maxnodes);

	return queue;
}

/*************************************************************************/
/*! This function resets the priority queue */
/**************************************************************************/
void mynd_priority_queue_Reset(priority_queue_t *queue)
{
	reordering_int_t i;
	reordering_int_t *locator = queue->locator;
	node_t *heap    = queue->heap;

	for (i = queue->nownodes - 1; i >= 0; i--)
		locator[heap[i].val] = -1;
	queue->nownodes = 0;
}

/*************************************************************************/
/*! This function frees the internal datastructures of the priority queue */
/**************************************************************************/
void mynd_priority_queue_Free(priority_queue_t *queue)
{
	if (queue == NULL) return;
	mynd_check_free(queue->locator, sizeof(reordering_int_t) * queue->maxnodes, "priority_queue_Free: queue->locator");
	mynd_check_free(queue->heap, sizeof(node_t) * queue->maxnodes, "priority_queue_Free: queue->heap");
	queue->maxnodes = 0;
}

/*************************************************************************/
/*! This function frees the internal datastructures of the priority queue 
    and the queue itself */
/**************************************************************************/
void mynd_priority_queue_Destroy(priority_queue_t *queue)
{
	if (queue == NULL) return;
	mynd_priority_queue_Free(queue);
	mynd_check_free(queue, sizeof(priority_queue_t), "mynd_priority_queue_Destroy: queue");
}

/*************************************************************************/
/*! This function returns the length of the queue */
/**************************************************************************/
reordering_int_t mynd_priority_queue_Length(priority_queue_t *queue)
{
	return queue->nownodes;
}

/*************************************************************************/
/*! This function adds an item in the priority queue */
/**************************************************************************/
reordering_int_t mynd_priority_queue_Insert(priority_queue_t *queue, reordering_int_t node, reordering_int_t key)
{
	reordering_int_t i, j;
	reordering_int_t *locator=queue->locator;
	node_t *heap = queue->heap;

	i = queue->nownodes++;
	while (i > 0) 
	{
		j = (i - 1) >> 1;
		if (m_gt_n(key, heap[j].key)) 
		{
			heap[i] = heap[j];
			locator[heap[i].val] = i;
			i = j;
		}
		else
			break;
	}
  
	heap[i].key   = key;
	heap[i].val   = node;
	locator[node] = i;

	return 0;
}

/*************************************************************************/
/*! This function deletes an item from the priority queue */
/**************************************************************************/
reordering_int_t mynd_priority_queue_Delete(priority_queue_t *queue, reordering_int_t node)
{
	reordering_int_t i, j, nownodes;
	reordering_int_t newkey, oldkey;
	reordering_int_t *locator = queue->locator;
	node_t *heap = queue->heap;

	i = locator[node];
	locator[node] = -1;

	if (--queue->nownodes > 0 && heap[queue->nownodes].val != node) 
	{
		node   = heap[queue->nownodes].val;
		newkey = heap[queue->nownodes].key;
		oldkey = heap[i].key;

		if (m_gt_n(newkey, oldkey)) 
		{ /* Filter-up */
			while (i > 0) 
			{
				j = (i - 1) >> 1;
				if (m_gt_n(newkey, heap[j].key)) 
				{
					heap[i] = heap[j];
					locator[heap[i].val] = i;
					i = j;
				}
				else
					break;
			}
		}
		else 
		{ /* Filter down */
			nownodes = queue->nownodes;
			while ((j = (i << 1) + 1) < nownodes) 
			{
				if (m_gt_n(heap[j].key, newkey)) 
				{
					if (j + 1 < nownodes && m_gt_n(heap[j + 1].key, heap[j].key))
						j++;
					heap[i] = heap[j];
					locator[heap[i].val] = i;
					i = j;
				}
				else if (j + 1 < nownodes && m_gt_n(heap[j + 1].key, newkey)) 
				{
					j++;
					heap[i] = heap[j];
					locator[heap[i].val] = i;
					i = j;
				}
				else
					break;
			}
		}

		heap[i].key   = newkey;
		heap[i].val   = node;
		locator[node] = i;
	}

	return 0;
}

/*************************************************************************/
/*! This function updates the key values associated for a particular item */ 
/**************************************************************************/
void mynd_priority_queue_Update(priority_queue_t *queue, reordering_int_t node, reordering_int_t newkey)
{
	reordering_int_t i, j, nownodes;
	reordering_int_t oldkey;
	reordering_int_t *locator = queue->locator;
	node_t *heap = queue->heap;

	oldkey = heap[locator[node]].key;

	i = locator[node];

	if (m_gt_n(newkey, oldkey)) 
	{ /* Filter-up */
		while (i > 0) 
		{
			j = (i - 1) >> 1;
			if (m_gt_n(newkey, heap[j].key)) 
			{
				heap[i] = heap[j];
				locator[heap[i].val] = i;
				i = j;
			}
			else
				break;
		}
	}
	else 
	{ /* Filter down */
		nownodes = queue->nownodes;
		while ((j = (i << 1) + 1) < nownodes) 
		{
			if (m_gt_n(heap[j].key, newkey)) 
			{
				if (j + 1 < nownodes && m_gt_n(heap[j + 1].key, heap[j].key))
					j++;
				heap[i] = heap[j];
				locator[heap[i].val] = i;
				i = j;
			}
			else if (j + 1 < nownodes && m_gt_n(heap[j + 1].key, newkey)) 
			{
				j++;
				heap[i] = heap[j];
				locator[heap[i].val] = i;
				i = j;
			}
			else
				break;
		}
	}

	heap[i].key   = newkey;
	heap[i].val   = node;
	locator[node] = i;

	return;
}

/*************************************************************************/
/*! This function returns the item at the top of the queue and removes
    it from the priority queue */
/**************************************************************************/
reordering_int_t mynd_priority_queue_GetTop(priority_queue_t *queue)
{
	reordering_int_t i, j;
	reordering_int_t *locator;
	node_t *heap;
	reordering_int_t vtx, node;
	reordering_int_t key;

	if (queue->nownodes == 0)
		return -1;

	queue->nownodes--;

	heap    = queue->heap;
	locator = queue->locator;

	vtx = heap[0].val;
	locator[vtx] = -1;

	if ((i = queue->nownodes) > 0) 
	{
		key  = heap[i].key;
		node = heap[i].val;
		i = 0;
		while ((j = 2 * i + 1) < queue->nownodes) 
		{
			if (m_gt_n(heap[j].key, key)) 
			{
				if (j + 1 < queue->nownodes && m_gt_n(heap[j + 1].key, heap[j].key))
					j = j+1;
				heap[i] = heap[j];
				locator[heap[i].val] = i;
				i = j;
			}
			else if (j + 1 < queue->nownodes && m_gt_n(heap[j + 1].key, key)) 
			{
				j = j + 1;
				heap[i] = heap[j];
				locator[heap[i].val] = i;
				i = j;
			}
			else
				break;
		}

		heap[i].key   = key;
		heap[i].val   = node;
		locator[node] = i;
	}

	return vtx;
}

/*************************************************************************/
/*! This function returns the item at the top of the queue. The item is not
    deleted from the queue. */
/**************************************************************************/
reordering_int_t mynd_priority_queue_SeeTopVal(priority_queue_t *queue)
{
  return (queue->nownodes == 0 ? -1 : queue->heap[0].val);
}

/*************************************************************************/
/*! This function returns the item at the top of the queue. The item is not
    deleted from the queue. */
/**************************************************************************/
reordering_int_t mynd_priority_queue_SeeTopKey(priority_queue_t *queue)
{
  return (queue->nownodes == 0 ? -1 : queue->heap[0].key);
}

void mynd_exam_priority_queue(priority_queue_t *queue)
{
	printf("nownodes=%"PRIDX" maxnodes=%"PRIDX"\n",queue->nownodes,queue->maxnodes);
	printf("key:");
	for(reordering_int_t i = 0;i < queue->nownodes;i++)
		printf("%"PRIDX" ",queue->heap[i].key);
	printf("\n");
	printf("val:");
	for(reordering_int_t i = 0;i < queue->nownodes;i++)
		printf("%"PRIDX" ",queue->heap[i].val);
	printf("\n");
}

#endif