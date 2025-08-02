#ifndef COMPRESSGRAPH_H
#define COMPRESSGRAPH_H

#include "mynd_functionset.h"

graph_t *mynd_Compress_Graph(reordering_int_t nvtxs, reordering_int_t *xadj, reordering_int_t *adjncy, reordering_int_t *vwgt, reordering_int_t *cptr, reordering_int_t *cind)
{
    reordering_int_t i, ii, iii, j, jj, k, l, cnvtxs, cnedges;
    reordering_int_t *cxadj, *cvwgt, *cadjncy, *cadjwgt, *mark, *map;
    ikv_t *keys;
    graph_t *graph = NULL;

    // printf("Compress_Graph 0\n");

    mark = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "Compress_Graph: mark");
    map  = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "Compress_Graph: map");
    keys = (ikv_t *)mynd_check_malloc(sizeof(ikv_t) * nvtxs, "Compress_Graph: keys");
    mynd_set_value_int(nvtxs, -1, mark);
    mynd_set_value_int(nvtxs, -1, map);

    /* Compute a key for each adjacency list */
    for (i = 0; i < nvtxs; i++) 
    {
        k = 0;
        for (j = xadj[i]; j < xadj[i + 1]; j++)
            k += adjncy[j];
        keys[i].key = k + i; /* Add the diagonal entry as well */
        keys[i].val = i;
    }

    mynd_ikvsorti(nvtxs, keys);

    // printf("Compress_Graph 1\n");

    l = cptr[0] = 0;
    for (cnvtxs = i = 0; i < nvtxs; i++) 
    {
        ii = keys[i].val;
        if (map[ii] == -1) 
        {
            mark[ii] = i;  /* Add the diagonal entry */
            for (j = xadj[ii]; j < xadj[ii + 1]; j++) 
                mark[adjncy[j]] = i;

            map[ii]   = cnvtxs;
            cind[l++] = ii;

            for (j = i + 1; j < nvtxs; j++) 
            {
                iii = keys[j].val;

                if (keys[i].key != keys[j].key || xadj[ii + 1] - xadj[ii] != xadj[iii + 1] - xadj[iii])
                    break; /* Break if keys or degrees are different */

                if (map[iii] == -1) 
                { /* Do a comparison if iii has not been mapped */ 
                    for (jj = xadj[iii]; jj < xadj[iii + 1]; jj++) 
                    {
                        if (mark[adjncy[jj]] != i)
                            break;
                    }

                    if (jj == xadj[iii + 1]) 
                    { /* Identical adjacency structure */
                        map[iii]  = cnvtxs;
                        cind[l++] = iii;
                    }
                }
            }

            cptr[++cnvtxs] = l;
        }
    }

    // printf("Compress_Graph 2\n");

    if (cnvtxs < 0.85 * nvtxs) 
    {
        // printf("compress !!!\n");
        graph = mynd_CreateGraph();

        cnedges = 0;
        for (i = 0; i < cnvtxs; i++) 
        {
            ii = cind[cptr[i]];
            cnedges += xadj[ii + 1] - xadj[ii];
        }

        // printf("compress 0 !!!\n");

        // cxadj   = graph->xadj   = imalloc(cnvtxs+1, "CompressGraph: xadj");
        // cvwgt   = graph->vwgt   = ismalloc(cnvtxs, 0, "CompressGraph: vwgt");
        // cadjncy = graph->adjncy = imalloc(cnedges, "CompressGraph: adjncy");
        //         graph->adjwgt = ismalloc(cnedges, 1, "CompressGraph: adjwgt");
        cxadj   = graph->xadj   = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * (cnvtxs + 1), "Compress_Graph: graph->xadj");
        cvwgt   = graph->vwgt   = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * cnvtxs, "Compress_Graph: graph->vwgt");
        cadjncy = graph->adjncy = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * cnedges, "Compress_Graph: graph->adjncy");
        cadjwgt = graph->adjwgt = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * cnedges, "Compress_Graph: graph->adjwgt");
        mynd_set_value_int(cnvtxs, 0, cvwgt);
        mynd_set_value_int(cnedges, 1, cadjwgt);

        // printf("compress 1 !!!\n");

        mynd_set_value_int(nvtxs, -1, mark);
        l = cxadj[0] = 0;
        for (i = 0; i < cnvtxs; i++) 
        {
            mark[i] = i;
            for (j = cptr[i]; j < cptr[i + 1]; j++) 
            {
                ii = cind[j];

                cvwgt[i] += (vwgt == NULL ? 1 : vwgt[ii]);

                for (jj = xadj[ii]; jj < xadj[ii + 1]; jj++) 
                {
                    k = map[adjncy[jj]];
                    if (mark[k] != i) 
                    {
                        mark[k] = i;
                        cadjncy[l++] = k;
                    }
                }
            }
            cxadj[i + 1] = l;
        }

        // printf("compress 2 !!!\n");

        graph->nvtxs  = cnvtxs;
        graph->nedges = l;
        // graph->ncon   = 1;
        // printf("sizeof(reordering_int_t) * cnedges=%"PRIDX" sizeof(reordering_int_t) * graph->nedges=%"PRIDX"\n",sizeof(reordering_int_t) * cnedges,sizeof(reordering_int_t) * graph->nedges);
        cadjncy = graph->adjncy = (reordering_int_t *)mynd_check_realloc(cadjncy, sizeof(reordering_int_t) * graph->nedges, sizeof(reordering_int_t) * cnedges, "Compress_Graph: adjncy");
	    cadjwgt = graph->adjwgt = (reordering_int_t *)mynd_check_realloc(cadjwgt, sizeof(reordering_int_t) * graph->nedges, sizeof(reordering_int_t) * cnedges, "Compress_Graph: adjwgt");
    

        mynd_SetupGraph_tvwgt(graph);
        mynd_SetupGraph_label(graph);
    }

    // printf("Compress_Graph 3\n");

    // printf("%"PRIDX"\n",mark[0]);
    // for(reordering_int_t i = 0;i < nvtxs;i++)
    //     printf("%"PRIDX" ",mark[i]);
    
    mynd_check_free(keys, sizeof(ikv_t) * nvtxs, "Compress_Graph: keys");
    // printf("Compress_Graph 3.1\n");
    mynd_check_free(map, sizeof(reordering_int_t) * nvtxs, "Compress_Graph: map");
    // printf("Compress_Graph 3.2\n");
    // printf("mark=%p \n",mark);
    mynd_check_free(mark, sizeof(reordering_int_t) * nvtxs, "Compress_Graph: mark");

    // printf("Compress_Graph 4\n");

    return graph;
}

#endif