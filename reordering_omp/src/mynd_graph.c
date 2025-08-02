#ifndef GRAPH_H
#define GRAPH_H

#include "mynd_functionset.h"

/*************************************************************************/
/*! This function initializes a graph_t data structure */
/*************************************************************************/
void InitGraph(graph_t *graph) 
{
	memset((void *)graph, 0, sizeof(graph_t));

	/* graph size constants */
	graph->nvtxs     = -1;
	graph->nedges    = -1;
	// graph->ncon      = -1;
	graph->mincut    = -1;
	graph->minvol    = -1;
	graph->nbnd      = -1;

	/* memory for the graph structure */
	graph->xadj      = NULL;
	graph->vwgt      = NULL;
	// graph->vsize     = NULL;
	graph->adjncy    = NULL;
	graph->adjwgt    = NULL;
	graph->label     = NULL;
	graph->cmap      = NULL;
	graph->match     = NULL;
	graph->tvwgt     = NULL;
	graph->invtvwgt  = NULL;

	/* memory for the partition/refinement structure */
	graph->where     = NULL;
	graph->pwgts     = NULL;
	graph->id        = NULL;
	graph->ed        = NULL;
	graph->bndptr    = NULL;
	graph->bndind    = NULL;
	graph->nrinfo    = NULL;
	// graph->ckrinfo   = NULL;
	// graph->vkrinfo   = NULL;

	/* linked-list structure */
	graph->coarser   = NULL;
	graph->finer     = NULL;
}


/*************************************************************************/
/*! This function creates and initializes a graph_t data structure */
/*************************************************************************/
graph_t *mynd_CreateGraph(void)
{
	graph_t *graph;

	graph = (graph_t *)mynd_check_malloc(sizeof(graph_t), "CreateGraph: graph");

	InitGraph(graph);

	return graph;
}

/*************************************************************************/
/*! Set's up the tvwgt/invtvwgt info */
/*************************************************************************/
void mynd_SetupGraph_tvwgt(graph_t *graph)
{
	if (graph->tvwgt == NULL) 
		graph->tvwgt = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * 1, "SetupGraph_tvwgt: tvwgt");
	if (graph->invtvwgt == NULL) 
		graph->invtvwgt = (double *)mynd_check_malloc(sizeof(double) * 1, "SetupGraph_tvwgt: invtvwgt");

	for (reordering_int_t i = 0; i < 1; i++) 
	{
		graph->tvwgt[i]    = mynd_sum_int(graph->nvtxs, graph->vwgt + i, 1);
		graph->invtvwgt[i] = 1.0 / (graph->tvwgt[i] > 0 ? graph->tvwgt[i] : 1);
	}
}


/*************************************************************************/
/*! Set's up the label info */
/*************************************************************************/
void mynd_SetupGraph_label(graph_t *graph)
{
	if (graph->label == NULL)
		graph->label = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * graph->nvtxs, "SetupGraph_label: label");

	for (reordering_int_t i = 0; i < graph->nvtxs; i++)
		graph->label[i] = i;
}

/*************************************************************************/
/*! This function sets up the graph from the user input */
/*************************************************************************/
graph_t *mynd_SetupGraph(reordering_int_t nvtxs, reordering_int_t *xadj, reordering_int_t *adjncy, reordering_int_t *vwgt, reordering_int_t *adjwgt) 
{
	/* allocate the graph and fill in the fields */
	graph_t *graph = mynd_CreateGraph();

	graph->nvtxs  = nvtxs;
	graph->nedges = xadj[nvtxs];

	graph->xadj      = xadj;
	graph->adjncy    = adjncy;

	/* setup the vertex weights */
	if (vwgt) 
		graph->vwgt      = vwgt;
	else 
	{
		vwgt = graph->vwgt = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "SetupGraph: vwgt");
		mynd_set_value_int(nvtxs, 1, vwgt);
	}

	graph->tvwgt = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t), "SetupGraph: tvwgts");
	graph->invtvwgt = (double *)mynd_check_malloc(sizeof(double), "SetupGraph: invtvwgts");
	for (reordering_int_t i = 0; i < 1; i++) 
	{
		graph->tvwgt[i]    = mynd_sum_int(nvtxs, vwgt + i, 1);
		graph->invtvwgt[i] = 1.0 / (graph->tvwgt[i] > 0 ? graph->tvwgt[i] : 1);
	}

	// if (ctrl->objtype == OBJTYPE_VOL) 
	// { 
	// 	/* Setup the vsize */
	// 	if (vsize) 
	// 	{
	// 		graph->vsize      = vsize;
	// 		graph->free_vsize = 0;
	// 	}
	// 	else 
	// 	{
	// 		vsize = graph->vsize = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "SetupGraph: vsize");
	// 		mynd_set_value_int(nvtxs, 1, vsize);
	// 	}

	// 	/* Allocate memory for edge weights and initialize them to the sum of the vsize */
	// 	adjwgt = graph->adjwgt = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * graph->nedges, "SetupGraph: adjwgt");
	// 	for (i = 0; i < nvtxs; i++) 
	// 	{
	// 		for (j = xadj[i]; j < xadj[i + 1]; j++)
	// 			adjwgt[j] = 1 + vsize[i] + vsize[adjncy[j]];
	// 	}
	// }
	// else 
	// { /* For edgecut minimization */
		/* setup the edge weights */
		if (adjwgt) 
			graph->adjwgt = adjwgt;
		else 
		{
			adjwgt = graph->adjwgt = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * graph->nedges, "SetupGraph: adjwgt");
			mynd_set_value_int(graph->nedges, 1, adjwgt);
		}
	// }


	/* setup various derived info */
	mynd_SetupGraph_tvwgt(graph);

	// if (ctrl->optype == OP_PMETIS || ctrl->optype == OP_OMETIS) 
		mynd_SetupGraph_label(graph);

	return graph;
}

/*************************************************************************/
/*! Setup the various arrays for the coarse graph 
 */
/*************************************************************************/
graph_t *mynd_SetupCoarseGraph(graph_t *graph, reordering_int_t cnvtxs)
{
	graph_t *cgraph = mynd_CreateGraph();

	cgraph->nvtxs = cnvtxs;

	cgraph->finer  = graph;
	graph->coarser = cgraph;

	/* Allocate memory for the coarser graph */
	cgraph->xadj     = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * (cnvtxs + 1), "SetupCoarseGraph: xadj");
	// cgraph->adjncy   = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * graph->nedges, "SetupCoarseGraph: adjncy");
	// cgraph->adjwgt   = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * graph->nedges, "SetupCoarseGraph: adjwgt");
	cgraph->vwgt     = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * cnvtxs, "SetupCoarseGraph: vwgt");
	cgraph->tvwgt    = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t), "SetupCoarseGraph: tvwgt");
	cgraph->invtvwgt = (double *)mynd_check_malloc(sizeof(double), "SetupCoarseGraph: invtvwgt");

	return cgraph;
}

/*************************************************************************/
/*! Setup the various arrays for the splitted graph */
/*************************************************************************/
graph_t *mynd_SetupSplitGraph(graph_t *graph, reordering_int_t subnvtxs, reordering_int_t subnedges)
{
	graph_t *subgraph = mynd_CreateGraph();

	subgraph->nvtxs  = subnvtxs;
	subgraph->nedges = subnedges;

	/* Allocate memory for the splitted graph */
	subgraph->xadj     = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * (subnvtxs + 1), "SetupSplitGraph: xadj");
	subgraph->vwgt     = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * subnvtxs, "SetupSplitGraph: vwgt");
	subgraph->adjncy   = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * subnedges,  "SetupSplitGraph: adjncy");
	subgraph->adjwgt   = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * subnedges,  "SetupSplitGraph: adjwgt");
	subgraph->label	   = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * subnvtxs,   "SetupSplitGraph: label");
	subgraph->tvwgt    = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t), "SetupSplitGraph: tvwgt");
	subgraph->invtvwgt = (double *)mynd_check_malloc(sizeof(double), "SetupSplitGraph: invtvwgt");

	return subgraph;
}

/*************************************************************************/
/*! This function frees the refinement/partition memory stored in a graph */
/*************************************************************************/
void mynd_FreeRefineData(graph_t *graph) 
{
	/* The following is for the -minconn and -contig to work properly in
		the vol-refinement routines */
	// if ((void *)graph->ckrinfo == (void *)graph->vkrinfo)
	// 	graph->ckrinfo = NULL;

	/* free partition/refinement structure */
	if(graph->nrinfo != NULL) 
		mynd_check_free(graph->nrinfo, sizeof(nrinfo_t) * graph->nvtxs, "FreeRefineData: graph->nrinfo");
	if(graph->ed != NULL) 
		mynd_check_free(graph->ed, sizeof(reordering_int_t) * graph->nvtxs, "FreeRefineData: graph->ed");
	if(graph->id != NULL) 
		mynd_check_free(graph->id, sizeof(reordering_int_t) * graph->nvtxs, "FreeRefineData: graph->id");
	if(graph->bndind != NULL) 
		mynd_check_free(graph->bndind, sizeof(reordering_int_t) * graph->nvtxs, "FreeRefineData: graph->bndind");
	if(graph->bndptr != NULL) 
		mynd_check_free(graph->bndptr, sizeof(reordering_int_t) * graph->nvtxs, "FreeRefineData: graph->bndptr");
	if(graph->pwgts != NULL) 
		mynd_check_free(graph->pwgts, sizeof(reordering_int_t) * 3, "FreeRefineData: graph->pwgts");
	// mynd_check_free(graph->ckrinfo);
	// mynd_check_free(graph->vkrinfo);
}

/*************************************************************************/
/*! This function deallocates any memory stored in a graph */
/*************************************************************************/
void mynd_FreeGraph(graph_t **r_graph) 
{
	graph_t *graph = *r_graph;

	/* free partition/refinement structure */
	mynd_FreeRefineData(graph);
	
	/* free graph structure */
	if(graph->where != NULL)
		mynd_check_free(graph->where, sizeof(reordering_int_t) * graph->nvtxs, "FreeGraph: graph->where");
	if(graph->cmap != NULL) 
		mynd_check_free(graph->cmap, sizeof(reordering_int_t) * graph->nvtxs, "FreeGraph: graph->cmap");
	if(graph->match != NULL) 
		mynd_check_free(graph->match, sizeof(reordering_int_t) * graph->nvtxs, "FreeGraph:graph->match");
	if(graph->adjwgt != NULL) 
		mynd_check_free(graph->adjwgt, sizeof(reordering_int_t) * graph->nedges, "FreeGraph: graph->adjwgt");
	if(graph->adjncy != NULL) 
		mynd_check_free(graph->adjncy, sizeof(reordering_int_t) * graph->nedges, "FreeGraph: graph->adjncy");
	if(graph->invtvwgt != NULL) 
		mynd_check_free(graph->invtvwgt, sizeof(double), "FreeGraph: graph->invtvwgt");
	if(graph->tvwgt != NULL) 
		mynd_check_free(graph->tvwgt, sizeof(reordering_int_t), "FreeGraph: graph->tvwgt");
	if(graph->vwgt != NULL) 
		mynd_check_free(graph->vwgt, sizeof(reordering_int_t) * graph->nvtxs, "FreeGraph: graph->vwgt");
	if(graph->xadj != NULL) 
		mynd_check_free(graph->xadj, sizeof(reordering_int_t) * (graph->nvtxs + 1), "FreeGraph: graph->xadj");
	
	if(graph->label != NULL) 
		mynd_check_free(graph->label, sizeof(reordering_int_t) * graph->nvtxs, "FreeGraph: graph->label");

	mynd_check_free(graph, sizeof(graph_t), "FreeGraph: graph");
	
	*r_graph = NULL;
}

/*************************************************************************/
/*! This function changes the numbering to start from 0 instead of 1 */
/*************************************************************************/
void mynd_Change2CNumbering(reordering_int_t nvtxs, reordering_int_t *xadj, reordering_int_t *adjncy)
{
	for (reordering_int_t i = 0; i <= nvtxs; i++)
		xadj[i]--;
	for (reordering_int_t i = 0; i < xadj[nvtxs]; i++)
		adjncy[i]--;
}

/*************************************************************************/
/*! This function changes the numbering to start from 1 instead of 0 */
/*************************************************************************/
void mynd_Change2FNumbering(reordering_int_t nvtxs, reordering_int_t *xadj, reordering_int_t *adjncy, reordering_int_t *vector)
{
	for (reordering_int_t i = 0; i < nvtxs; i++)
		vector[i]++;

	for (reordering_int_t i = 0; i < xadj[nvtxs]; i++)
		adjncy[i]++;

	for (reordering_int_t i = 0; i <= nvtxs; i++)
		xadj[i]++;
}

void mynd_exam_nvtxs_nedges(graph_t *graph)
{
    printf("nvtxs:%"PRIDX" nedges:%"PRIDX"\n",graph->nvtxs,graph->nedges);
}

void mynd_exam_xadj(graph_t *graph)
{
    printf("xadj:\n");
    for(reordering_int_t i = 0;i <= graph->nvtxs;i++)
		printf("%"PRIDX" ",graph->xadj[i]);
	printf("\n");
}

void mynd_exam_vwgt(graph_t *graph)
{
    printf("vwgt:\n");
    for(reordering_int_t i = 0;i < graph->nvtxs;i++)
		printf("%"PRIDX" ",graph->vwgt[i]);
	printf("\n");
}

void mynd_exam_adjncy_adjwgt(graph_t *graph)
{
    printf("adjncy adjwgt:\n");
    for(reordering_int_t i = 0;i < graph->nvtxs;i++)
	{
		printf("ncy:");
		for(reordering_int_t j = graph->xadj[i];j < graph->xadj[i + 1];j++)
			printf("%"PRIDX" ",graph->adjncy[j]);
		printf("\nwgt:");
		for(reordering_int_t j = graph->xadj[i];j < graph->xadj[i + 1];j++)
			printf("%"PRIDX" ",graph->adjwgt[j]);
		printf("\n");
	}
}

void mynd_exam_label(graph_t *graph)
{
    printf("label:\n");
    for(reordering_int_t i = 0;i < graph->nvtxs;i++)
		printf("%"PRIDX" ",graph->label[i]);
	printf("\n");
}

void mynd_exam_where(graph_t *graph)
{
    printf("where:\n");
    for(reordering_int_t i = 0;i < graph->nvtxs;i++)
		printf("%"PRIDX" ",graph->where[i]);
	printf("\n");
}

void mynd_exam_pwgts(graph_t *graph)
{
    printf("pwgts:");
    for(reordering_int_t i = 0;i < 3;i++)
		printf("%"PRIDX" ",graph->pwgts[i]);
	printf("\n");
}

void mynd_exam_edid(graph_t *graph)
{
    printf("edid:\n");
	printf("ed:");
    for(reordering_int_t i = 0;i < graph->nvtxs;i++)
		printf("%"PRIDX" ",graph->ed[i]);
	printf("\n");
	printf("id:");
    for(reordering_int_t i = 0;i < graph->nvtxs;i++)
		printf("%"PRIDX" ",graph->id[i]);
	printf("\n");
}

void mynd_exam_bnd(graph_t *graph)
{
    printf("bnd:\n");
	printf("nbnd=%"PRIDX"\n",graph->nbnd);
	printf("bndind:\n");
    for(reordering_int_t i = 0;i < graph->nbnd;i++)
		printf("%"PRIDX" ",graph->bndind[i]);
	printf("\n");
	printf("bndptr:\n");
    for(reordering_int_t i = 0;i < graph->nvtxs;i++)
		printf("%"PRIDX" ",graph->bndptr[i]);
	printf("\n");
}

void mynd_exam_nrinfo(graph_t *graph)
{
	printf("nrinfo.edegrees[0]:\n");
    for(reordering_int_t i = 0;i < graph->nbnd;i++)
		printf("%"PRIDX" ",graph->nrinfo[i].edegrees[0]);
	printf("\n");
	printf("nrinfo.edegrees[1]:\n");
    for(reordering_int_t i = 0;i < graph->nbnd;i++)
		printf("%"PRIDX" ",graph->nrinfo[i].edegrees[1]);
	printf("\n");
}

void mynd_exam_num(reordering_int_t *num, reordering_int_t n)
{
    printf("num:\n");
    for(reordering_int_t i = 0;i < n;i++)
		printf("%"PRIDX" ",num[i]);
	printf("\n");
}

#endif