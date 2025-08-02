#ifndef MATCH_H
#define MATCH_H

#include "mynd_functionset.h"

/*************************************************************************/
/*! This function matches the unmatched vertices whose degree is less than
    maxdegree using a 2-hop matching that involves vertices that are two 
    hops away from each other. 
    The requirement of the 2-hop matching is a simple non-empty overlap
    between the adjancency lists of the vertices. */
/**************************************************************************/
reordering_int_t Match_2HopAny(graph_t *graph, reordering_int_t *perm, reordering_int_t *match, reordering_int_t cnvtxs, reordering_int_t *r_nunmatched, reordering_int_t maxdegree)
{
	reordering_int_t i, pi, j, jj, nvtxs;
	reordering_int_t *xadj, *adjncy, *colptr, *rowind;
	reordering_int_t *cmap;
	reordering_int_t nunmatched;

	nvtxs  = graph->nvtxs;
	xadj   = graph->xadj;
	adjncy = graph->adjncy;
	cmap   = graph->cmap;

	nunmatched = *r_nunmatched;

	/* create the inverted index */
	colptr = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * (nvtxs + 1), "Match_2HopAny: colptr");
	mynd_set_value_int(nvtxs + 1, 0, colptr);

	for (i = 0; i < nvtxs; i++) 
	{
		if (match[i] == -1 && xadj[i + 1] - xadj[i] < maxdegree) 
		{
			for (j = xadj[i]; j < xadj[i + 1]; j++)
				colptr[adjncy[j]]++;
		}
	}
	MAKECSR(i, nvtxs, colptr);

	reordering_int_t rowind_size = colptr[nvtxs];
	rowind = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * colptr[nvtxs], "Match_2HopAny: rowind");
	for (pi = 0; pi < nvtxs; pi++) 
	{
		i = perm[pi];
		if (match[i] == -1 && xadj[i + 1] - xadj[i] < maxdegree) 
		{
			for (j = xadj[i]; j < xadj[i + 1]; j++)
				rowind[colptr[adjncy[j]]++] = i;
		}
	}
	SHIFTCSR(i, nvtxs, colptr);

	// printf("Match_2HopAny 0\n");

	/* compute matchings by going down the inverted index */
	for (pi = 0; pi < nvtxs; pi++) 
	{
		i = perm[pi];
		if (colptr[i + 1] - colptr[i] < 2)
			continue;

		for (jj = colptr[i + 1], j = colptr[i]; j < jj; j++) 
		{
			if (match[rowind[j]] == -1) 
			{
				for (jj--; jj > j; jj--) 
				{
					if (match[rowind[jj]] == -1) 
					{
						cmap[rowind[j]] = cmap[rowind[jj]] = cnvtxs++;
						match[rowind[j]]  = rowind[jj];
						match[rowind[jj]] = rowind[j];
						nunmatched -= 2;
						break;
					}
				}
			}
		}
	}

	// printf("Match_2HopAny 1\n");
	mynd_check_free(rowind, sizeof(reordering_int_t) * rowind_size, "Match_2HopAny: rowind");
	mynd_check_free(colptr, sizeof(reordering_int_t) * (nvtxs + 1), "Match_2HopAny: colptr");

	*r_nunmatched = nunmatched;
	return cnvtxs;
}

/*************************************************************************/
/*! This function matches the unmatched vertices whose degree is less than
    maxdegree using a 2-hop matching that involves vertices that are two 
    hops away from each other. 
    The requirement of the 2-hop matching is that of identical adjacency
    lists.
 */
/**************************************************************************/
reordering_int_t Match_2HopAll(graph_t *graph, reordering_int_t *perm, reordering_int_t *match, reordering_int_t cnvtxs, reordering_int_t *r_nunmatched, reordering_int_t maxdegree)
{
	reordering_int_t i, pi, pk, j, jj, k, nvtxs, mask, idegree;
	reordering_int_t *xadj, *adjncy;
	reordering_int_t *cmap, *mark;
	ikv_t *keys;
	reordering_int_t nunmatched, ncand;

	nvtxs  = graph->nvtxs;
	xadj   = graph->xadj;
	adjncy = graph->adjncy;
	cmap   = graph->cmap;

	nunmatched = *r_nunmatched;
	mask = IDX_MAX / maxdegree;

	/* collapse vertices with identical adjancency lists */
	keys = (ikv_t *)mynd_check_malloc(sizeof(ikv_t) * nunmatched, "Match_2HopAll: keys");
	for (ncand = 0, pi = 0; pi < nvtxs; pi++) 
	{
		i = perm[pi];
		idegree = xadj[i + 1] - xadj[i];
		if (match[i] == -1 && idegree > 1 && idegree < maxdegree) 
		{
			for (k = 0, j = xadj[i]; j < xadj[i + 1]; j++) 
				k += adjncy[j] % mask;
			keys[ncand].val = i;
			keys[ncand].key = (k % mask) * maxdegree + idegree;
			ncand++;
		}
	}
	mynd_ikvsorti(ncand, keys);

	mark = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "Match_2HopAll: mark");
	mynd_set_value_int(nvtxs, 0, mark);
	for (pi = 0; pi < ncand; pi++) 
	{
		i = keys[pi].val;
		if (match[i] != -1)
			continue;
		
		for (j = xadj[i]; j < xadj[i + 1]; j++)
			mark[adjncy[j]] = i;

		for (pk = pi + 1; pk < ncand; pk++) 
		{
			k = keys[pk].val;
			if (match[k] != -1)
				continue;

			if (keys[pi].key != keys[pk].key)
				break;
			if (xadj[i + 1] - xadj[i] != xadj[k + 1] - xadj[k])
				break;
			
			for (jj = xadj[k]; jj < xadj[k + 1]; jj++) 
			{
				if (mark[adjncy[jj]] != i)
					break;
			}
			if (jj == xadj[k+1]) 
			{
				cmap[i] = cmap[k] = cnvtxs++;
				match[i] = k;
				match[k] = i;
				nunmatched -= 2;
				break;
			}
		}
	}

	mynd_check_free(mark, sizeof(reordering_int_t) * nvtxs, "Match_2HopAll: mark");
	mynd_check_free(keys, sizeof(ikv_t) * (*r_nunmatched), "Match_2HopAll: keys");

	*r_nunmatched = nunmatched;
	return cnvtxs;
}

/*************************************************************************/
/*! This function matches the unmatched vertices using a 2-hop matching 
    that involves vertices that are two hops away from each other. */
/**************************************************************************/
reordering_int_t Match_2Hop(graph_t *graph, reordering_int_t *perm, reordering_int_t *match, reordering_int_t cnvtxs, reordering_int_t nunmatched)
{
	// printf("Match_2Hop 0\n");
	cnvtxs = Match_2HopAny(graph, perm, match, cnvtxs, &nunmatched, 2);
	// printf("Match_2HopAny match\n");
	// exam_num(match,graph->nvtxs);
	// printf("Match_2Hop 1\n");
	cnvtxs = Match_2HopAll(graph, perm, match, cnvtxs, &nunmatched, 64);
	// printf("Match_2HopAll match\n");
	// exam_num(match,graph->nvtxs);
	// printf("Match_2Hop 2\n");
	if (nunmatched > 1.5 * 0.10 * graph->nvtxs) 
	{
		// printf("Match_2Hop 3\n");
		cnvtxs = Match_2HopAny(graph, perm, match, cnvtxs, &nunmatched, 3);
		// printf("Match_2Hop 4\n");
	}
	if (nunmatched > 2.0 * 0.10 * graph->nvtxs) 
	{
		// printf("Match_2Hop 5\n");
		cnvtxs = Match_2HopAny(graph, perm, match, cnvtxs, &nunmatched, graph->nvtxs);
		// printf("Match_2Hop 6\n");
	}

	return cnvtxs;
}

/*************************************************************************/
/*! This function finds a matching by randomly selecting one of the 
    unmatched adjacent vertices. 
 */
/**************************************************************************/
reordering_int_t mynd_Match_RM(graph_t *graph, reordering_int_t maxvwgt)
{
	reordering_int_t i, pi, j,  k, nvtxs, cnvtxs, maxidx, last_unmatched;
	reordering_int_t *xadj, *vwgt, *adjncy;
	reordering_int_t *match, *cmap, *perm;
	reordering_int_t nunmatched = 0;

	nvtxs  = graph->nvtxs;
	// ncon   = graph->ncon;
	xadj   = graph->xadj;
	vwgt   = graph->vwgt;
	adjncy = graph->adjncy;
	cmap   = graph->cmap;

	// graph->match = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "mynd_Match_RM: graph->match");
	match = graph->match;
	mynd_set_value_int(nvtxs, -1, match);

	perm  = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "mynd_Match_RM: perm");

  	mynd_irandArrayPermute(nvtxs, perm, nvtxs/8, 1);
	// printf("perm\n");
	// exam_num(perm,nvtxs);

	// printf("mynd_Match_RM 0\n");

	for (cnvtxs = 0, last_unmatched = 0, pi = 0; pi < nvtxs; pi++) 
	{
		i = perm[pi];

		if (match[i] == -1) 
		{  
			/* Unmatched */
			maxidx = i;

			if(vwgt[i] < maxvwgt)
			{
				/* Deal with island vertices. Find a non-island and match it with. 
					The matching ignores ctrl->maxvwgt requirements */
				if (xadj[i] == xadj[i + 1]) 
				{
					last_unmatched = lyj_max(pi, last_unmatched) + 1;
					for (; last_unmatched<nvtxs; last_unmatched++) 
					{
						j = perm[last_unmatched];
						if (match[j] == -1) 
						{
							maxidx = j;
							break;
						}
					}
				}
				else
				{
					/* single constraint version */
					for (j = xadj[i]; j < xadj[i + 1]; j++) 
					{
						k = adjncy[j];
						if (match[k] == -1 && vwgt[i] + vwgt[k] <= maxvwgt) 
						{
							maxidx = k;
							break;
						}
					}

					/* If it did not match, record for a 2-hop matching. */
					if (maxidx == i && 3 * vwgt[i] < maxvwgt) 
					{
						nunmatched++;
						maxidx = -1;
					}
				}
			}

			if (maxidx != -1) 
			{
				cmap[i]  = cmap[maxidx] = cnvtxs++;
				match[i] = maxidx;
				match[maxidx] = i;
			}
		}
	}

	// printf("mynd_Match_RM 1\n");
	// exam_num(match,nvtxs);

  //printf("nunmatched: %zu\n", nunmatched);

	/* see if a 2-hop matching is required/allowed */
	if (nunmatched > 0.1 * nvtxs) 
		cnvtxs = Match_2Hop(graph, perm, match, cnvtxs, nunmatched);

	// printf("mynd_Match_RM 2\n");
	// exam_num(match,nvtxs);

	/* match the final unmatched vertices with themselves and reorder the vertices 
		of the coarse graph for memory-friendly contraction */
	for (cnvtxs = 0, i = 0; i < nvtxs; i++) 
	{
		if (match[i] == -1) 
		{
			match[i] = i;
			cmap[i]  = cnvtxs++;
		}
		else 
		{
			if (i <= match[i]) 
				cmap[i] = cmap[match[i]] = cnvtxs++;
		}
	}

	// printf("mynd_Match_RM 3\n");
	mynd_check_free(perm, sizeof(reordering_int_t) * nvtxs, "mynd_Match_RM: perm");
	// exam_num(match,nvtxs);

	return cnvtxs;
}

/*************************************************************************
* This function uses simple counting sort to return a permutation array
* corresponding to the sorted order. The keys are arsumed to start from
* 0 and they are positive.  This sorting is used during matching.
**************************************************************************/
void BucketSortKeysInc(reordering_int_t n, reordering_int_t max, reordering_int_t *keys, reordering_int_t *tperm, reordering_int_t *perm)
{
	reordering_int_t i, ii;
	reordering_int_t *counts;

	counts = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * (max + 2), "BucketSortKeysInc: counts");
	mynd_set_value_int(max + 2, 0, counts);

	for (i = 0; i < n; i++)
		counts[keys[i]]++;
	MAKECSR(i, max + 1, counts);

	for (ii = 0; ii < n; ii++) 
	{
		i = tperm[ii];
		perm[counts[keys[i]]++] = i;
	}

	mynd_check_free(counts, sizeof(reordering_int_t) * (max + 2), "BucketSortKeysInc: counts");
}

/**************************************************************************/
/*! This function finds a matching using the HEM heuristic. The vertices 
    are visited based on increasing degree to ensure that all vertices are 
    given a chance to match with something. 
 */
/**************************************************************************/
reordering_int_t mynd_Match_SHEM(graph_t *graph, reordering_int_t maxvwgt)
{
	reordering_int_t i, pi, j, k, nvtxs, cnvtxs, maxidx, maxwgt, 
			last_unmatched, avgdegree;
	reordering_int_t *xadj, *vwgt, *adjncy, *adjwgt;
	reordering_int_t *match, *cmap, *degrees, *perm, *tperm;
	reordering_int_t nunmatched=0;

	nvtxs  = graph->nvtxs;
	// ncon   = graph->ncon;
	xadj   = graph->xadj;
	vwgt   = graph->vwgt;
	adjncy = graph->adjncy;
	adjwgt = graph->adjwgt;
	cmap   = graph->cmap;

	match = graph->match;
	mynd_set_value_int(nvtxs, -1, match);
	perm = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "mynd_Match_SHEM: perm");
	tperm = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "mynd_Match_SHEM: tperm");
	degrees = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "mynd_Match_SHEM: degrees");

	mynd_irandArrayPermute(nvtxs, tperm, nvtxs/8, 1);

	avgdegree = 0.7 * (xadj[nvtxs] / nvtxs);
	for (i = 0; i < nvtxs; i++) 
		degrees[i] = (xadj[i + 1] - xadj[i] > avgdegree ? avgdegree : xadj[i + 1] - xadj[i]);
	BucketSortKeysInc(nvtxs, avgdegree, degrees, tperm, perm);

	// printf("mynd_Match_SHEM 0\n");
	for (cnvtxs = 0, last_unmatched = 0, pi = 0; pi < nvtxs; pi++) 
	{
		i = perm[pi];

		if (match[i] == -1) 
		{  /* Unmatched */
			maxidx = i;
			maxwgt = -1;

			if (vwgt[i] < maxvwgt) 
			{
				/* Deal with island vertices. Find a non-island and match it with. 
					The matching ignores ctrl->maxvwgt requirements */
				if (xadj[i] == xadj[i + 1]) 
				{ 
					last_unmatched = lyj_max(pi, last_unmatched) + 1;
					for (; last_unmatched < nvtxs; last_unmatched++) 
					{
						j = perm[last_unmatched];
						if (match[j] == -1) 
						{
							maxidx = j;
							break;
						}
					}
				}
				else 
				{
					/* Find a heavy-edge matching, subject to maxvwgt constraints */
					/* single constraint version */
					for (j = xadj[i]; j < xadj[i + 1]; j++) 
					{
						k = adjncy[j];
						if (match[k] == -1 && maxwgt < adjwgt[j] && vwgt[i] + vwgt[k] <= maxvwgt) 
						{
							maxidx = k;
							maxwgt = adjwgt[j];
						}
					}

					/* If it did not match, record for a 2-hop matching. */
					if (maxidx == i && 3 * vwgt[i] < maxvwgt) 
					{
						nunmatched++;
						maxidx = -1;
					}
				}
			}

			if (maxidx != -1) 
			{
				cmap[i]  = cmap[maxidx] = cnvtxs++;
				match[i] = maxidx;
				match[maxidx] = i;
			}
    	}
  	}
	// printf("mynd_Match_SHEM 1\n");
	// exam_num(match,nvtxs);
	//printf("nunmatched: %zu\n", nunmatched);

	/* see if a 2-hop matching is required/allowed */
	if (nunmatched > 0.1 * nvtxs) 
		cnvtxs = Match_2Hop(graph, perm, match, cnvtxs, nunmatched);
	// printf("mynd_Match_SHEM 2\n");
	// exam_num(match,nvtxs);

	/* match the final unmatched vertices with themselves and reorder the vertices 
		of the coarse graph for memory-friendly contraction */
	for (cnvtxs=0, i=0; i<nvtxs; i++) 
	{
		if (match[i] == -1) 
		{
			match[i] = i;
			cmap[i] = cnvtxs++;
		}
		else 
		{
			if (i <= match[i]) 
				cmap[i] = cmap[match[i]] = cnvtxs++;
		}
	}
	// printf("mynd_Match_SHEM 3\n");
	mynd_check_free(degrees, sizeof(reordering_int_t) * nvtxs, "mynd_Match_SHEM: degrees");
	mynd_check_free(tperm, sizeof(reordering_int_t) * nvtxs, "mynd_Match_SHEM: tperm");
	mynd_check_free(perm, sizeof(reordering_int_t) * nvtxs, "mynd_Match_SHEM: perm");
	// exam_num(match,nvtxs);
	return cnvtxs;
}


#endif