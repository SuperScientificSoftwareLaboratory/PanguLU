#ifndef SPLITGRAPH_H
#define SPLITGRAPH_H

#include "mynd_functionset.h"

void mynd_SplitGraphReorder(graph_t *graph, graph_t **sub_lgraph, graph_t **sub_rgraph, reordering_int_t level)
{
    reordering_int_t nvtxs;
    reordering_int_t *xadj, *vwgt, *adjncy, *adjwgt, *label, *where;
    reordering_int_t subnvtxs[3], subnedges[3];
    reordering_int_t *subxadj[2], *subvwgt[2], *subadjncy[2], *subadjwgt[2], *sublabel[2];
    reordering_int_t *rename;
    graph_t *lgraph, *rgraph;

    nvtxs  = graph->nvtxs;
    xadj   = graph->xadj;
    vwgt   = graph->vwgt;
    adjncy = graph->adjncy;
    adjwgt = graph->adjwgt;
    label  = graph->label;
    where  = graph->where;

    subnvtxs[0]  = subnvtxs[1]  = subnvtxs[2]  = 0;
    subnedges[0] = subnedges[1] = subnedges[2] = 0;

    //cmap --> rename
    rename = graph->cmap;
    // rename = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "SplitGraphoRerder: rename");

    // printf("SplitGraphoRerder 0\n");
    for(reordering_int_t i = 0;i < nvtxs;i++)
    {
        reordering_int_t partition = where[i];
        if(partition != 2)
        {
            rename[i] = subnvtxs[partition];
            subnvtxs[partition] ++;
            for(reordering_int_t j = xadj[i];j < xadj[i + 1];j++)
            {
                reordering_int_t k = adjncy[j];
                if(partition == where[k])
                    subnedges[partition]++;
            }
        }
    }
    // exam_num(rename, nvtxs);
    // printf("subnvtxs[0]=%d subnvtxs[1]=%d subnedges[0]=%d subnedges[1]=%d\n",subnvtxs[0],subnvtxs[1],subnedges[0],subnedges[1]);

    lgraph = mynd_SetupSplitGraph(lgraph, subnvtxs[0], subnedges[0]);
    subxadj[0]   = lgraph->xadj;
    subvwgt[0]   = lgraph->vwgt;
    subadjncy[0] = lgraph->adjncy;
    subadjwgt[0] = lgraph->adjwgt;
    sublabel[0]  = lgraph->label;

    rgraph = mynd_SetupSplitGraph(rgraph, subnvtxs[1], subnedges[1]);
    subxadj[1]   = rgraph->xadj;
    subvwgt[1]   = rgraph->vwgt;
    subadjncy[1] = rgraph->adjncy;
    subadjwgt[1] = rgraph->adjwgt;
    sublabel[1]  = rgraph->label;

    // printf("SplitGraphoRerder 1\n");
    subnvtxs[0]  = subnvtxs[1]  = subnvtxs[2]  = 0;
    subnedges[0] = subnedges[1] = subnedges[2] = 0;
    subxadj[0][0] = subxadj[1][0] = 0;
    for(reordering_int_t i = 0;i < nvtxs;i++)
    {
        reordering_int_t partition = where[i];

        if(partition == 2)
            continue;
        
        reordering_int_t numedge = 0;
        reordering_int_t map = subnvtxs[partition];
        reordering_int_t ptr = subnedges[partition];
        subxadj[partition][map + 1] = subxadj[partition][map];
        // printf("subxadj[partition][map + 1]=%d\n", subxadj[partition][map + 1]);
        for(reordering_int_t j = xadj[i];j < xadj[i + 1];j++)
        {
            reordering_int_t k = adjncy[j];
            if(where[k] == partition)
            {
                subadjncy[partition][ptr] = rename[k];
                subadjwgt[partition][ptr] = adjwgt[j];
                // printf("k=%d rename[k]=%d subadjncy[partition][ptr]=%d subadjwgt[partition][ptr]=%d ptr=%d numedge=%d map=%d\n",
                //     k,rename[k],subadjncy[partition][ptr],subadjwgt[partition][ptr],ptr, numedge,map);
                ptr++;
                numedge++;
            }
        }
        // printf("numedge=%d map=%d\n",numedge,map);
        subxadj[partition][map + 1] += numedge;
        subvwgt[partition][map] = vwgt[i];
        sublabel[partition][map] = label[i];
        subnvtxs[partition]++;
        subnedges[partition] = ptr;
        // printf("subxadj[partition][map + 1]=%d subvwgt[partition][map]=%d sublabel[partition][map]=%d subnvtxs[partition]=%d subnedges[partition]=%d\n",
        //     subxadj[partition][map + 1],subvwgt[partition][map],sublabel[partition][map],subnvtxs[partition],subnedges[partition]);
    }

    // printf("SplitGraphoRerder 2\n");
    // printf("subnvtxs[0]=%d subnvtxs[1]=%d subnedges[0]=%d subnedges[1]=%d\n",subnvtxs[0],subnvtxs[1],subnedges[0],subnedges[1]);
    lgraph->nvtxs  = subnvtxs[0];
    lgraph->nedges = subnedges[0];
    rgraph->nvtxs  = subnvtxs[1];
    rgraph->nedges = subnedges[1];

    mynd_SetupGraph_tvwgt(lgraph);
    mynd_SetupGraph_tvwgt(rgraph);

    *sub_lgraph = lgraph;
    *sub_rgraph = rgraph;

    // mynd_check_free(rename,"SplitGraphReorder: rename");

    // printf("lgraph:\n");
    // exam_nvtxs_nedges(lgraph);
    // exam_xadj(lgraph);
    // exam_vwgt(lgraph);
    // exam_adjncy_adjwgt(lgraph);
    // exam_label(lgraph);

    // printf("rgraph:\n");
    // exam_nvtxs_nedges(rgraph);
    // exam_xadj(rgraph);
    // exam_vwgt(rgraph);
    // exam_adjncy_adjwgt(rgraph);
    // exam_label(rgraph);
}

#endif