#ifndef INITIALPARTITION_H
#define INITIALPARTITION_H

#include "mynd_functionset.h"

//  InitSeparator + GrowBisectionNode
void mynd_ReorderBisection(graph_t *graph, reordering_int_t niparts)
{
    reordering_real_t ntpwgts[2] = {0.5, 0.5};

    /* this is required for the cut-based part of the refinement */
    // Setup2WayBalMultipliers(graph, ntpwgts);

    // GrowBisectionNode(graph, ntpwgts, niparts);
    reordering_int_t i, j, k, nvtxs, drain, nleft, first, last, pwgts[2], oneminpwgt, 
        onemaxpwgt, bestcut=0, inbfs;
    reordering_int_t *xadj, *vwgt, *adjncy, *where, *bndind;
    reordering_int_t *queue, *touched, *bestwhere;

    nvtxs  = graph->nvtxs;
    xadj   = graph->xadj;
    vwgt   = graph->vwgt;
    adjncy = graph->adjncy;

    onemaxpwgt = 1.2000499 * graph->tvwgt[0] * 0.5;
    oneminpwgt = (1.0 / 1.2000499) * graph->tvwgt[0] * 0.5;

    /* Allocate refinement memory. Allocate sufficient memory for both edge and node */
    graph->where  = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "ReorderBisection: graph->where");
    graph->pwgts  = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * 3, "ReorderBisection: graph->pwgts");
    graph->bndptr = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "ReorderBisection: graph->bndptr");
    graph->bndind = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "ReorderBisection: graph->bndind");
    graph->id     = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "ReorderBisection: graph->id");
    graph->ed     = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "ReorderBisection: graph->ed");
    graph->nrinfo = (nrinfo_t *)mynd_check_malloc(sizeof(nrinfo_t) * nvtxs, "ReorderBisection: graph->nrinfo");

    bestwhere = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "ReorderBisection: bestwhere");
    queue     = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "ReorderBisection: queue");
    touched   = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "ReorderBisection: touched");
  
    where  = graph->where;
    bndind = graph->bndind;
    // printf("ReorderBisection 0\n");
    for (inbfs = 0; inbfs < niparts; inbfs++) 
    {
        mynd_set_value_int(nvtxs, 1, where);
        mynd_set_value_int(nvtxs, 0, touched);

        pwgts[1] = graph->tvwgt[0];
        pwgts[0] = 0;

        queue[0] = mynd_irandInRange(nvtxs);
        touched[queue[0]] = 1;
        first = 0; last = 1;
        nleft = nvtxs-1;
        drain = 0;

        /* Start the BFS from queue to get a partition */
        for (;;) 
        {
            if (first == last) 
            { 
                /* Empty. Disconnected graph! */
                if (nleft == 0 || drain)
                    break;
  
                k = mynd_irandInRange(nleft);
                for (i = 0; i < nvtxs; i++) 
                { 
                    /* select the kth untouched vertex */
                    if (touched[i] == 0) 
                    {
                        if (k == 0)
                            break;
                        else
                            k--;
                    }
                }

                queue[0]   = i;
                touched[i] = 1;
                first      = 0; 
                last       = 1;
                nleft--;
            }

            i = queue[first++];
            if (pwgts[1] - vwgt[i] < oneminpwgt) 
            {
                drain = 1;
                continue;
            }

            where[i] = 0;
            pwgts[0] += vwgt[i];
            pwgts[1] -= vwgt[i];
            if (pwgts[1] <= onemaxpwgt)
                break;

            drain = 0;
            for (j = xadj[i]; j < xadj[i + 1]; j++) 
            {
                k = adjncy[j];
                if (touched[k] == 0) 
                {
                queue[last++] = k;
                touched[k] = 1;
                nleft--;
                }
            }
        }
        // printf("ReorderBisection 1\n");
        // printf("ReorderBisection 1 end ccnt=%"PRIDX"\n",rand_count());
        // exam_where(graph);

        /*************************************************************
        * Do some partition refinement 
        **************************************************************/
        CONTROL_COMMAND(control, PARTITIOBINF2WAY, mynd_gettimebegin(&start_partitioninf2way, &end_partitioninf2way, &time_partitioninf2way));
		mynd_Compute_Partition_Informetion_2way(graph);
		CONTROL_COMMAND(control, PARTITIOBINF2WAY, mynd_gettimeend(&start_partitioninf2way, &end_partitioninf2way, &time_partitioninf2way));
        // printf("ReorderBisection 2\n");
        // exam_where(graph);
        CONTROL_COMMAND(control, FM2WAYCUTBALANCE_Time, mynd_gettimebegin(&start_fm2waycutbalance, &end_fm2waycutbalance, &time_fm2waycutbalance));
		mynd_Balance2Way(graph, ntpwgts);
		CONTROL_COMMAND(control, FM2WAYCUTBALANCE_Time, mynd_gettimeend(&start_fm2waycutbalance, &end_fm2waycutbalance, &time_fm2waycutbalance));
        // printf("ReorderBisection 3\n");
        // printf("ReorderBisection 3 end ccnt=%"PRIDX"\n",rand_count());
        // exam_where(graph);
        CONTROL_COMMAND(control, FM2WAYCUTREFINE_Time, mynd_gettimebegin(&start_fm2waycutrefine, &end_fm2waycutrefine, &time_fm2waycutrefine));
		mynd_FM_2WayCutRefine(graph, ntpwgts, 4);
		CONTROL_COMMAND(control, FM2WAYCUTREFINE_Time, mynd_gettimeend(&start_fm2waycutrefine, &end_fm2waycutrefine, &time_fm2waycutrefine));
        // printf("ReorderBisection 4\n");
        // printf("ReorderBisection 4 end ccnt=%"PRIDX"\n",rand_count());
        // exam_where(graph);

        /* Construct and refine the vertex separator */
        for (i = 0; i < graph->nbnd; i++) 
        {
            j = bndind[i];
            if (xadj[j + 1] - xadj[j] > 0) /* ignore islands */
                where[j] = 2;
        }

        // printf("ReorderBisection 5\n");
        // exam_where(graph);

        CONTROL_COMMAND(control, REORDERINF2WAY_Time, mynd_gettimebegin(&start_reorderinf2way, &end_reorderinf2way, &time_reorderinf2way));
		mynd_Compute_Reorder_Informetion_2way(graph); 
		CONTROL_COMMAND(control, REORDERINF2WAY_Time, mynd_gettimeend(&start_reorderinf2way, &end_reorderinf2way, &time_reorderinf2way));
        // printf("ReorderBisection 6\n");
        // exam_where(graph);
        CONTROL_COMMAND(control, FM2SIDENODEREFINE_Time, mynd_gettimebegin(&start_fm2sidenoderefine, &end_fm2sidenoderefine, &time_fm2sidenoderefine));
		mynd_FM_2WayNodeRefine2Sided(graph, 1);
		CONTROL_COMMAND(control, FM2SIDENODEREFINE_Time, mynd_gettimeend(&start_fm2sidenoderefine, &end_fm2sidenoderefine, &time_fm2sidenoderefine));
        // printf("ReorderBisection 7\n");
        // printf("ReorderBisection 7 end ccnt=%"PRIDX"\n",rand_count());
        // exam_where(graph);
        CONTROL_COMMAND(control, FM1SIDENODEREFINE_Time, mynd_gettimebegin(&start_fm1sidenoderefine, &end_fm1sidenoderefine, &time_fm1sidenoderefine));
		mynd_FM_2WayNodeRefine1Sided(graph, 4);
		CONTROL_COMMAND(control, FM1SIDENODEREFINE_Time, mynd_gettimeend(&start_fm1sidenoderefine, &end_fm1sidenoderefine, &time_fm1sidenoderefine));
        // printf("ReorderBisection 8\n");
        // printf("ReorderBisection 8 end ccnt=%"PRIDX"\n",rand_count());
        // exam_where(graph);

        /*
        printf("ISep: [%"PRIDX" %"PRIDX" %"PRIDX" %"PRIDX"] %"PRIDX"\n", 
            inbfs, graph->pwgts[0], graph->pwgts[1], graph->pwgts[2], bestcut); 
        */
        // printf("inbfs=%"PRIDX"\n",inbfs);
        if (inbfs == 0 || bestcut > graph->mincut) 
        {
            bestcut = graph->mincut;
            mynd_copy_int(nvtxs, where, bestwhere);
        }
    }

    graph->mincut = bestcut;
    mynd_copy_int(nvtxs, bestwhere, where);
    // printf("ReorderBisection 9\n");
    mynd_check_free(touched, sizeof(reordering_int_t) * nvtxs, "ReorderBisection: touched");
    mynd_check_free(queue, sizeof(reordering_int_t) * nvtxs, "ReorderBisection: queue");
    mynd_check_free(bestwhere, sizeof(reordering_int_t) * nvtxs, "ReorderBisection: bestwhere");
    //     exam_where(graph);
}

#endif