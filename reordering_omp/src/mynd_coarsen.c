#ifndef COARSEN_H
#define COARSEN_H

#include "mynd_functionset.h"

graph_t *mynd_CoarsenGraph(graph_t *graph, reordering_int_t Coarsen_Threshold)
{
	reordering_int_t i, eqewgts, level, maxvwgt, cnvtxs;

	/* determine if the weights on the edges are all the same */
	for (eqewgts = 1, i = 1; i < graph->nedges; i++) 
	{
		if (graph->adjwgt[0] != graph->adjwgt[i]) 
		{
			eqewgts = 0;
			break;
		}
	}

	/* set the maximum allowed coarsest vertex weight */
	for (i = 0; i < 1; i++)
		maxvwgt = 1.5 * graph->tvwgt[i] / Coarsen_Threshold;

	level = 0;

	do 
	{
		// printf("level=%"PRIDX" graph->nvtxs=%"PRIDX"\n",level,graph->nvtxs);
		/* allocate memory for cmap, if it has not already been done due to
			multiple cuts */
		if (graph->match == NULL)
			graph->match = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * graph->nvtxs, "CoarsenGraph: graph->match");
		if (graph->cmap == NULL)
			graph->cmap = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * graph->nvtxs, "CoarsenGraph: graph->cmap");
		if (graph->where == NULL)
			graph->where  = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * graph->nvtxs, "CoarsenGraph: graph->where");
		/* determine which matching scheme you will use */

		// printf("CoarsenGraph 0\n");
		CONTROL_COMMAND(control, MATCH_Time, mynd_gettimebegin(&start_match, &end_match, &time_match));
		if (eqewgts || graph->nedges == 0)
			cnvtxs = mynd_Match_RM(graph, maxvwgt);
		else
			cnvtxs = mynd_Match_SHEM(graph, maxvwgt);
		CONTROL_COMMAND(control, MATCH_Time, mynd_gettimeend(&start_match, &end_match, &time_match));
		// printf("CoarsenGraph 1\n");
		// printf("cnvtxs=%"PRIDX"\n",cnvtxs);
		// exam_num(graph->match,graph->nvtxs);

		CONTROL_COMMAND(control, CREATCOARSENGRAPH_Time, mynd_gettimebegin(&start_createcoarsengraph, &end_createcoarsengraph, &time_createcoarsengraph));
		// CreateCoarseGraph(graph, cnvtxs);
		// CreateCoarseGraph_S(graph, cnvtxs);
		// CreateCoarseGraph_BST(graph, cnvtxs);
		// CreateCoarseGraph_BST_2(graph, cnvtxs);
		// mynd_CreateCoarseGraph_HT(graph, cnvtxs);
		mynd_CreateCoarseGraph_HT_2(graph, cnvtxs);
		CONTROL_COMMAND(control, CREATCOARSENGRAPH_Time, mynd_gettimeend(&start_createcoarsengraph, &end_createcoarsengraph, &time_createcoarsengraph));
		// printf("CoarsenGraph 2\n");
		// printf("CreateCoarseGraph=%10.3"PRREAL"\n",time_createcoarsengraph);

		graph = graph->coarser;
		eqewgts = 0;
		level++;

		//  sort for adjncy adjwgt
		// for(reordering_int_t z = 0;z < graph->nvtxs;z++)
		// {
		// 	for(reordering_int_t y = graph->xadj[z];y < graph->xadj[z + 1];y++)
		// 	{
		// 		reordering_int_t t = y;
		// 		for(reordering_int_t x = y + 1;x < graph->xadj[z + 1];x++)
		// 			if(graph->adjncy[x] < graph->adjncy[t]) t = x;
		// 		reordering_int_t temp;
		// 		temp = graph->adjncy[t],graph->adjncy[t] = graph->adjncy[y], graph->adjncy[y] = temp;
		// 		temp = graph->adjwgt[t],graph->adjwgt[t] = graph->adjwgt[y], graph->adjwgt[y] = temp;
		// 	}
		// }

		// exam_nvtxs_nedges(graph);
        // exam_xadj(graph);
        // exam_vwgt(graph);
        // exam_adjncy_adjwgt(graph);

	} while (graph->nvtxs > Coarsen_Threshold && 
			graph->nvtxs < 0.85 * graph->finer->nvtxs && 
			graph->nedges > graph->nvtxs / 2);

	return graph;
}

/*************************************************************************/
/*! This function takes a graph and creates a sequence of nlevels coarser 
    graphs, where nlevels is an input parameter.
 */
/*************************************************************************/
graph_t *mynd_CoarsenGraphNlevels_metis(graph_t *graph, reordering_int_t Coarsen_Threshold, reordering_int_t nlevels)
{
	reordering_int_t i, eqewgts, level, maxvwgt, cnvtxs;

	/* determine if the weights on the edges are all the same */
	for (eqewgts = 1, i = 1; i < graph->nedges; i++) 
	{
		if (graph->adjwgt[0] != graph->adjwgt[i]) 
		{
			eqewgts = 0;
			break;
		}
	}

	/* set the maximum allowed coarsest vertex weight */
	for (i = 0; i < 1; i++)
		maxvwgt = 1.5 * graph->tvwgt[i] / Coarsen_Threshold;

	for (level = 0; level < nlevels; level++) 
	{
		// printf("level=%"PRIDX" graph->nvtxs=%"PRIDX"\n",level,graph->nvtxs);
		/* allocate memory for cmap, if it has not already been done due to
			multiple cuts */
		if (graph->match == NULL)
			graph->match = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * graph->nvtxs, "CoarsenGraphNlevels_metis: graph->match");
		if (graph->cmap == NULL)
			graph->cmap = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * graph->nvtxs, "CoarsenGraphNlevels_metis: graph->cmap");
		if (graph->where == NULL)
			graph->where  = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * graph->nvtxs, "CoarsenGraphNlevels_metis: graph->where");
		
		// printf("CoarsenGraphNlevels 0\n");
    	/* determine which matching scheme you will use */
		CONTROL_COMMAND(control, MATCH_Time, mynd_gettimebegin(&start_match, &end_match, &time_match));
        if (eqewgts || graph->nedges == 0)
			cnvtxs = mynd_Match_RM(graph, maxvwgt);
        else
			cnvtxs = mynd_Match_SHEM(graph, maxvwgt);
		CONTROL_COMMAND(control, MATCH_Time, mynd_gettimeend(&start_match, &end_match, &time_match));
		// printf("CoarsenGraphNlevels 1\n");
		// printf("cnvtxs=%"PRIDX"\n",cnvtxs);

		CONTROL_COMMAND(control, CREATCOARSENGRAPH_Time, mynd_gettimebegin(&start_createcoarsengraph, &end_createcoarsengraph, &time_createcoarsengraph));
		// CreateCoarseGraph(graph, cnvtxs);
		// CreateCoarseGraph_S(graph, cnvtxs);
		// CreateCoarseGraph_BST(graph, cnvtxs);
		// CreateCoarseGraph_BST_2(graph, cnvtxs);
		// mynd_CreateCoarseGraph_HT(graph, cnvtxs);
		mynd_CreateCoarseGraph_HT_2(graph, cnvtxs);
		// mynd_CreateCoarseGraph_HT_2_time(graph, cnvtxs);
		CONTROL_COMMAND(control, CREATCOARSENGRAPH_Time, mynd_gettimeend(&start_createcoarsengraph, &end_createcoarsengraph, &time_createcoarsengraph));

		// printf("CoarsenGraphNlevels 2\n");
		// printf("CreateCoarseGraph=%10.3"PRREAL"\n",time_createcoarsengraph);

		graph = graph->coarser;
		eqewgts = 0;
		// if(level == 0)
		// 	printf("level=0\n");

		//  sort for adjncy adjwgt
		// for(reordering_int_t z = 0;z < graph->nvtxs;z++)
		// {
		// 	for(reordering_int_t y = graph->xadj[z];y < graph->xadj[z + 1];y++)
		// 	{
		// 		reordering_int_t t = y;
		// 		for(reordering_int_t x = y + 1;x < graph->xadj[z + 1];x++)
		// 			if(graph->adjncy[x] < graph->adjncy[t]) t = x;
		// 		reordering_int_t temp;
		// 		temp = graph->adjncy[t],graph->adjncy[t] = graph->adjncy[y], graph->adjncy[y] = temp;
		// 		temp = graph->adjwgt[t],graph->adjwgt[t] = graph->adjwgt[y], graph->adjwgt[y] = temp;
		// 	}
		// }

		if (graph->nvtxs < Coarsen_Threshold || 
			graph->nvtxs > 0.85 * graph->finer->nvtxs || 
			graph->nedges < graph->nvtxs / 2)
		break; 
	}

	return graph;
}

#endif