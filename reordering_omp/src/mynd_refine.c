#ifndef REFINE_H
#define REFINE_H

#include "mynd_functionset.h"

void mynd_Compute_Partition_Informetion_2way(graph_t *graph)
{
    reordering_int_t nvtxs, nbnd, mincut;
    reordering_int_t *xadj, *vwgt, *adjncy, *adjwgt, *where, *bndptr, *bndind, *ed, *id, *pwgts;

    nvtxs = graph->nvtxs;
    nbnd  = graph->nbnd;

    xadj   = graph->xadj;
    vwgt   = graph->vwgt;
    adjncy = graph->adjncy;
    adjwgt = graph->adjwgt;
    where  = graph->where;
    bndptr = graph->bndptr;
    bndind = graph->bndind;
    ed     = graph->ed;
    id     = graph->id;
    pwgts  = graph->pwgts;

    mincut = 0;

    //  init nbnd, bndptr and bndind
    nbnd = mynd_init_queue(nbnd, bndptr, nvtxs);

    //  init pwgts
    mynd_set_value_int(3,0,pwgts);

    //  compute array nbnd, bndptr, bndind, ed, id
    for(reordering_int_t i = 0;i < nvtxs;i++)
    {
        reordering_int_t partition = where[i];
        reordering_int_t ted = 0;
        reordering_int_t tid = 0;
        for(reordering_int_t j = xadj[i];j < xadj[i + 1];j++)
        {
            reordering_int_t k = adjncy[j];
            if(partition != where[k])
                ted += adjwgt[j];
            else 
                tid += adjwgt[j];
        }

        // printf("i=%d flag_boundary=%d\n",i,flag_boundary);

        if(ted > 0 || xadj[i] == xadj[i + 1])
        {
            // printf("i=%d flag_boundary=%d\n",i,flag_boundary);
            nbnd = mynd_insert_queue(nbnd, bndptr, bndind, i);
            mincut += ted;
        }

        ed[i] = ted;
        id[i] = tid;
        pwgts[partition] += vwgt[i];
    }

    // exam_where(graph);
    // exam_edid(graph);
    // exam_pwgts(graph);

    graph->nbnd   = nbnd;
    graph->mincut = mincut / 2;

    // exam_bnd(graph);
}

void mynd_Compute_Reorder_Informetion_2way(graph_t *graph)
{
    reordering_int_t nvtxs, nbnd;
    reordering_int_t *xadj, *vwgt, *adjncy, *where, *bndptr, *bndind, *pwgts;
    nrinfo_t *nrinfo;

    nvtxs = graph->nvtxs;
    nbnd  = graph->nbnd;

    xadj   = graph->xadj;
    vwgt   = graph->vwgt;
    adjncy = graph->adjncy;
    
    where  = graph->where;
    bndptr = graph->bndptr;
    bndind = graph->bndind;
    nrinfo = graph->nrinfo;
    pwgts  = graph->pwgts;

    //  init nbnd, bndptr and bndind
    nbnd = mynd_init_queue(nbnd, bndptr, nvtxs);

    //  init pwgts
    mynd_set_value_int(3,0,pwgts);

    //  compute array nbnd, bndptr, bndind, ed, id
    for(reordering_int_t i = 0;i < nvtxs;i++)
    {
        reordering_int_t partition = where[i];

        pwgts[partition] += vwgt[i];

        if(partition == 2)
        {
            nbnd = mynd_insert_queue(nbnd, bndptr, bndind, i);
            nrinfo[i].edegrees[0] = nrinfo[i].edegrees[1] = 0;

            for(reordering_int_t j = xadj[i];j < xadj[i + 1];j++)
            {
                reordering_int_t k = adjncy[j];
                reordering_int_t other = where[k];
                if(other != 2)
                    nrinfo[i].edegrees[other] += vwgt[k];
            }
        }
    }

    graph->nbnd   = nbnd;
    graph->mincut = pwgts[2];

    // exam_nrinfo(graph);
    // exam_pwgts(graph);
    // exam_bnd(graph);
}

void mynd_project_Reorder(graph_t *graph)
{
    reordering_int_t nvtxs, *cmap, *where, *cwhere;
    graph_t *cgraph = graph->coarser;

    nvtxs  = graph->nvtxs;
    cmap   = graph->cmap;
    cwhere = cgraph->where;

    // printf("project_Reorder 0\n");
    where  = graph->where;
    for(reordering_int_t i = 0;i < nvtxs;i++)
        where[i] = cwhere[cmap[i]];
    
    // printf("project_Reorder 1\n");
    
    mynd_FreeGraph(&graph->coarser);
    graph->coarser = NULL;

    graph->pwgts  = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * 3, "project_Reorder: pwgts");
    graph->bndptr = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "project_Reorder: bndptr");
    graph->bndind = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "project_Reorder: bndind");
    graph->nrinfo = (nrinfo_t *)mynd_check_malloc(sizeof(nrinfo_t) * nvtxs, "project_Reorder: nrinfo");


    // printf("project_Reorder 2\n");
    mynd_Compute_Reorder_Informetion_2way(graph);
    // printf("project_Reorder 3\n");
}

/*************************************************************************/
/*! This function performs a cut-focused FM refinement */
/*************************************************************************/
void mynd_FM_2WayCutRefine(graph_t *graph, reordering_real_t *ntpwgts, reordering_int_t niter)
{
    reordering_int_t i, ii, j, k, kwgt, nvtxs, nbnd, nswaps, from, to, pass, limit, tmp;
    reordering_int_t *xadj, *vwgt, *adjncy, *adjwgt, *where, *id, *ed, *bndptr, *bndind, *pwgts;
    reordering_int_t *moved, *swaps, *perm;
    priority_queue_t *queues[2];
    reordering_int_t higain, mincut, mindiff, origdiff, initcut, newcut, mincutorder, avgvwgt;
    reordering_int_t tpwgts[2];

    nvtxs  = graph->nvtxs;
    xadj   = graph->xadj;
    vwgt   = graph->vwgt;
    adjncy = graph->adjncy;
    adjwgt = graph->adjwgt;
    where  = graph->where;
    id     = graph->id;
    ed     = graph->ed;
    pwgts  = graph->pwgts;
    bndptr = graph->bndptr;
    bndind = graph->bndind;

    moved = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "FM_2WayCutRefine: moved");
    swaps = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "FM_2WayCutRefine: swaps");
    perm  = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "FM_2WayCutRefine: perm");

    tpwgts[0] = graph->tvwgt[0] * ntpwgts[0];
    tpwgts[1] = graph->tvwgt[0] - tpwgts[0];
    
    limit   = lyj_min(lyj_max(0.01 * nvtxs, 15), 100);
    avgvwgt = lyj_min((pwgts[0] + pwgts[1]) / 20, 2 * (pwgts[0] + pwgts[1]) / nvtxs);

    queues[0] = mynd_priority_queue_Create(nvtxs);
    queues[1] = mynd_priority_queue_Create(nvtxs);

    origdiff = lyj_abs(tpwgts[0] - pwgts[0]);
    mynd_set_value_int(nvtxs, -1, moved);
    // exam_pwgts(graph);
    for (pass = 0; pass < niter; pass++) 
    { 
        // printf("pass=%"PRIDX"\n",pass);
        // printf("rollback\n");
        // exam_where(graph);
        // exam_bnd(graph);
        // exam_priority_queue(queues[0]);
        // exam_priority_queue(queues[1]);
        /* Do a number of passes */
        mynd_priority_queue_Reset(queues[0]);
        mynd_priority_queue_Reset(queues[1]);

        mincutorder = -1;
        newcut = mincut = initcut = graph->mincut;
        mindiff = lyj_abs(tpwgts[0] - pwgts[0]);

        /* Insert boundary nodes in the priority queues */
        nbnd = graph->nbnd;
        mynd_irandArrayPermute(nbnd, perm, nbnd, 1);
        for (ii = 0; ii < nbnd; ii++) 
        {
            i = perm[ii];
            mynd_priority_queue_Insert(queues[where[bndind[i]]], bndind[i], ed[bndind[i]] - id[bndind[i]]);
        }

        // printf("reset\n");
        // exam_priority_queue(queues[0]);
        // exam_priority_queue(queues[1]);

        for (nswaps = 0; nswaps < nvtxs; nswaps++) 
        {
            from = (tpwgts[0] - pwgts[0] < tpwgts[1] - pwgts[1] ? 0 : 1);
            to = (from + 1) % 2;

            if ((higain = mynd_priority_queue_GetTop(queues[from])) == -1)
                break;
            
            // printf("higain=%"PRIDX" from=%"PRIDX" to=%"PRIDX" \n",higain,from,to);
            // exam_pwgts(graph);

            newcut -= (ed[higain] - id[higain]);
            pwgts[to] += vwgt[higain];
            pwgts[from] -= vwgt[higain];

            if ((newcut < mincut && lyj_abs(tpwgts[0] - pwgts[0]) <= origdiff + avgvwgt) || 
                (newcut == mincut && lyj_abs(tpwgts[0] - pwgts[0]) < mindiff)) 
            {
                mincut  = newcut;
                mindiff = lyj_abs(tpwgts[0] - pwgts[0]);
                mincutorder = nswaps;
                // printf("tpwgts[0] - pwgts[0]=%"PRIDX" mindiff=%"PRIDX"\n",tpwgts[0] - pwgts[0], mindiff);
                // printf("nswaps=%"PRIDX" mincutorder=%"PRIDX"\n",nswaps,mincutorder);
            }
            else if (nswaps - mincutorder > limit) 
            { 
                /* We hit the limit, undo last move */
                newcut += (ed[higain] - id[higain]);
                pwgts[from] += vwgt[higain];
                pwgts[to] -= vwgt[higain];
                // printf("nswaps-mincutorder > limit\n");
                // printf("nswaps=%"PRIDX" mincutorder=%"PRIDX" limit=%"PRIDX"\n",nswaps,mincutorder,limit);
                break;
            }

            where[higain] = to;
            moved[higain] = nswaps;
            swaps[nswaps] = higain;

            // printf("Moved %6"PRIDX" from %"PRIDX". [%3"PRIDX" %3"PRIDX"] %5"PRIDX" [%4"PRIDX" %4"PRIDX"]\n", higain, from, ed[higain]-id[higain], vwgt[higain], newcut, pwgts[0], pwgts[1]);

            /**************************************************************
             * Update the id[i]/ed[i] values of the affected nodes
             ***************************************************************/
            lyj_swap(id[higain], ed[higain], tmp);
            if (ed[higain] == 0 && xadj[higain] < xadj[higain + 1]) 
                nbnd = mynd_delete_queue(nbnd, bndptr, bndind, higain);

            for (j = xadj[higain]; j < xadj[higain + 1]; j++) 
            {
                k = adjncy[j];

                kwgt = (to == where[k] ? adjwgt[j] : -adjwgt[j]);
                id[k] += kwgt;
                ed[k] -= kwgt;

                /* Update its boundary information and queue position */
                if (bndptr[k] != -1) 
                { 
                    /* If k was a boundary vertex */
                    if (ed[k] == 0) 
                    { 
                        /* Not a boundary vertex any more */
                        nbnd = mynd_delete_queue(nbnd, bndptr, bndind, k);
                        if (moved[k] == -1)  /* Remove it if in the queues */
                        {
                            mynd_priority_queue_Delete(queues[where[k]], k);
                        }
                    }
                    else 
                    { 
                        /* If it has not been moved, update its position in the queue */
                        if (moved[k] == -1) 
                        {
                            mynd_priority_queue_Update(queues[where[k]], k, ed[k]-id[k]);
                        }
                    }
                }
                else 
                {
                    if (ed[k] > 0) 
                    {  
                        /* It will now become a boundary vertex */
                        nbnd = mynd_insert_queue(nbnd, bndptr, bndind, k);
                        if (moved[k] == -1) 
                        {
                            mynd_priority_queue_Insert(queues[where[k]], k, ed[k] - id[k]);
                        }
                    }
                }
            }
        }
        // printf("moved\n");
        // exam_where(graph);
        // graph->nbnd = nbnd;
        // exam_bnd(graph);
        // exam_priority_queue(queues[0]);
        // exam_priority_queue(queues[1]);

        /****************************************************************
        * Roll back computations
        *****************************************************************/
        for (i = 0; i < nswaps; i++)
            moved[swaps[i]] = -1;  /* reset moved array */
        for (nswaps--; nswaps > mincutorder; nswaps--) 
        {
            higain = swaps[nswaps];

            to = where[higain] = (where[higain] + 1) % 2;
            lyj_swap(id[higain], ed[higain], tmp);
            if (ed[higain] == 0 && bndptr[higain] != -1 && xadj[higain] < xadj[higain + 1])
                nbnd = mynd_delete_queue(nbnd, bndptr, bndind, higain);
            else if (ed[higain] > 0 && bndptr[higain] == -1)
                nbnd = mynd_insert_queue(nbnd, bndptr, bndind, higain);

            pwgts[to] += vwgt[higain];
            pwgts[(to + 1) % 2] -= vwgt[higain];
            // exam_pwgts(graph);
            for (j = xadj[higain]; j < xadj[higain + 1]; j++) 
            {
                k = adjncy[j];

                kwgt = (to == where[k] ? adjwgt[j] : -adjwgt[j]);
                id[k] += kwgt;
                ed[k] -= kwgt;

                if (bndptr[k] != -1 && ed[k] == 0)
                    nbnd = mynd_delete_queue(nbnd, bndptr, bndind, k);
                if (bndptr[k] == -1 && ed[k] > 0)
                    nbnd = mynd_insert_queue(nbnd, bndptr, bndind, k);
            }
        }

        graph->mincut = mincut;
        graph->nbnd   = nbnd;

        if (mincutorder <= 0 || mincut == initcut)
            break;
    }

    mynd_priority_queue_Destroy(queues[1]);
    mynd_priority_queue_Destroy(queues[0]);

    mynd_check_free(perm, sizeof(reordering_int_t) * nvtxs, "FM_2WayCutRefine: perm");
    mynd_check_free(swaps, sizeof(reordering_int_t) * nvtxs, "FM_2WayCutRefine: swaps");
    mynd_check_free(moved, sizeof(reordering_int_t) * nvtxs, "FM_2WayCutRefine: moved");
}

/*************************************************************************/
/*! This function performs a node-based FM refinement */
/**************************************************************************/
void mynd_FM_2WayNodeRefine2Sided(graph_t *graph, reordering_int_t niter)
{
    reordering_int_t i, ii, j, k, jj, kk, nvtxs, nbnd, nswaps, nmind;
    reordering_int_t *xadj, *vwgt, *adjncy, *where, *pwgts, *edegrees, *bndind, *bndptr;
    reordering_int_t *mptr, *mind, *moved, *swaps;
    priority_queue_t *queues[2]; 
    nrinfo_t *rinfo;
    reordering_int_t higain, oldgain, mincut, initcut, mincutorder;	
    reordering_int_t pass, to, other, limit;
    reordering_int_t badmaxpwgt, mindiff, newdiff;
    reordering_int_t u[2], g[2];
    double mult;   

    nvtxs  = graph->nvtxs;
    xadj   = graph->xadj;
    adjncy = graph->adjncy;
    vwgt   = graph->vwgt;

    bndind = graph->bndind;
    bndptr = graph->bndptr;
    where  = graph->where;
    pwgts  = graph->pwgts;
    rinfo  = graph->nrinfo;

    queues[0] = mynd_priority_queue_Create(nvtxs);
    queues[1] = mynd_priority_queue_Create(nvtxs);

    moved = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "FM_2WayNodeRefine2Sided: moved");
    swaps = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "FM_2WayNodeRefine2Sided: swaps");
    mptr = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * (nvtxs + 1), "FM_2WayNodeRefine2Sided: mptr");
    mind = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs * 2, "FM_2WayNodeRefine2Sided: mind");

    mult = 0.5 * 1.2000499;
    badmaxpwgt = (reordering_int_t)(mult * (pwgts[0] + pwgts[1] + pwgts[2]));

    // printf("Partitions-N2: [%6"PRIDX" %6"PRIDX"] Nv-Nb[%6"PRIDX" %6"PRIDX"]. ISep: %6"PRIDX"\n", pwgts[0], pwgts[1], graph->nvtxs, graph->nbnd, graph->mincut);

    for (pass = 0; pass < niter; pass++) 
    {
        // printf("pass=%"PRIDX" \n",pass);
        mynd_set_value_int(nvtxs, -1, moved);
        mynd_priority_queue_Reset(queues[0]);
        mynd_priority_queue_Reset(queues[1]);

        mincutorder = -1;
        initcut = mincut = graph->mincut;
        nbnd = graph->nbnd;

        /* use the swaps array in place of the traditional perm array to save memory */
        mynd_irandArrayPermute(nbnd, swaps, nbnd, 1);
        for (ii = 0; ii < nbnd; ii++) 
        {
            i = bndind[swaps[ii]];
            mynd_priority_queue_Insert(queues[0], i, vwgt[i]-rinfo[i].edegrees[1]);
            mynd_priority_queue_Insert(queues[1], i, vwgt[i]-rinfo[i].edegrees[0]);
        }

        limit = (0 ? lyj_min(5*nbnd, 400) : lyj_min(2*nbnd, 300));

        /******************************************************
        * Get into the FM loop
        *******************************************************/
        mptr[0] = nmind = 0;
        mindiff = lyj_abs(pwgts[0] - pwgts[1]);
        to = (pwgts[0] < pwgts[1] ? 0 : 1);
        for (nswaps = 0; nswaps < nvtxs; nswaps++) 
        {
            u[0] = mynd_priority_queue_SeeTopVal(queues[0]);  
            u[1] = mynd_priority_queue_SeeTopVal(queues[1]);
            // printf("u[0]=%"PRIDX" u[1]=%"PRIDX"\n",u[0],u[1]);
            if (u[0] != -1 && u[1] != -1) 
            {
                g[0] = vwgt[u[0]] - rinfo[u[0]].edegrees[1];
                g[1] = vwgt[u[1]] - rinfo[u[1]].edegrees[0];

                to = (g[0] > g[1] ? 0 : (g[0] < g[1] ? 1 : pass % 2)); 
                
                if (pwgts[to] + vwgt[u[to]] > badmaxpwgt) 
                    to = (to + 1) % 2;
            }
            else if (u[0] == -1 && u[1] == -1) 
                break;
            else if (u[0] != -1 && pwgts[0] + vwgt[u[0]] <= badmaxpwgt)
                to = 0;
            else if (u[1] != -1 && pwgts[1] + vwgt[u[1]] <= badmaxpwgt)
                to = 1;
            else
                break;

            other = (to+1)%2;

            higain = mynd_priority_queue_GetTop(queues[to]);
            if (moved[higain] == -1) /* Delete if it was in the separator originally */
                mynd_priority_queue_Delete(queues[other], higain);

            /* The following check is to ensure we break out if there is a posibility
                of over-running the mind array.  */
            if (nmind + xadj[higain + 1] - xadj[higain] >= 2 * nvtxs - 1) 
                break;

            pwgts[2] -= (vwgt[higain] - rinfo[higain].edegrees[other]);

            newdiff = lyj_abs(pwgts[to] + vwgt[higain] - (pwgts[other] - rinfo[higain].edegrees[other]));
            if (pwgts[2] < mincut || (pwgts[2] == mincut && newdiff < mindiff)) 
            {
                mincut = pwgts[2];
                mincutorder = nswaps;
                mindiff = newdiff;
                // printf("nswaps=%"PRIDX" mincutorder=%"PRIDX" pwgts[2]=%"PRIDX" mincut=%"PRIDX" mindiff=%"PRIDX"\n",nswaps,mincutorder,pwgts[2],mincut,mindiff);
                
            }
            else 
            {
                if (nswaps - mincutorder > 2 * limit || 
                    (nswaps - mincutorder > limit && pwgts[2] > 1.10 * mincut)) 
                {
                    pwgts[2] += (vwgt[higain] - rinfo[higain].edegrees[other]);
                    // printf("nswaps-mincutorder > 2 * limit=%d || nswaps - mincutorder > limit && pwgts[2] > 1.10 * mincut=%d\n",
                    //     nswaps-mincutorder > 2 * limit,nswaps - mincutorder > limit && pwgts[2] > 1.10 * mincut);
                    // printf("nswaps=%"PRIDX" mincutorder=%"PRIDX" limit=%"PRIDX" pwgts[2]=%"PRIDX" mincut=%"PRIDX"\n",nswaps,mincutorder,limit,pwgts[2],mincut);
                
                    break; /* No further improvement, break out */
                }
            }

            nbnd = mynd_delete_queue(nbnd,bndptr,bndind,higain);
            pwgts[to] += vwgt[higain];
            where[higain] = to;
            moved[higain] = nswaps;
            swaps[nswaps] = higain;  

            /**********************************************************
             * Update the degrees of the affected nodes
             ***********************************************************/
            for (j = xadj[higain]; j < xadj[higain + 1]; j++) 
            {
                k = adjncy[j];
                if (where[k] == 2) 
                { 
                    /* For the in-separator vertices modify their edegree[to] */
                    oldgain = vwgt[k] - rinfo[k].edegrees[to];
                    rinfo[k].edegrees[to] += vwgt[higain];
                    if (moved[k] == -1 || moved[k] == -(2 + other))
                        mynd_priority_queue_Update(queues[other], k, oldgain - vwgt[higain]);
                }
                else if (where[k] == other) 
                { 
                    /* This vertex is pulled into the separator */
                    nbnd = mynd_insert_queue(nbnd,bndptr,bndind,k);

                    mind[nmind++] = k;  /* Keep track for rollback */
                    where[k] = 2;
                    pwgts[other] -= vwgt[k];

                    edegrees = rinfo[k].edegrees;
                    edegrees[0] = edegrees[1] = 0;
                    for (jj = xadj[k]; jj < xadj[k + 1]; jj++) 
                    {
                        kk = adjncy[jj];
                        if (where[kk] != 2) 
                            edegrees[where[kk]] += vwgt[kk];
                        else 
                        {
                            oldgain = vwgt[kk] - rinfo[kk].edegrees[other];
                            rinfo[kk].edegrees[other] -= vwgt[k];
                            if (moved[kk] == -1 || moved[kk] == -(2 + to))
                                mynd_priority_queue_Update(queues[to], kk, oldgain + vwgt[k]);
                        }
                    }

                    /* Insert the new vertex into the priority queue. Only one side! */
                    if (moved[k] == -1) 
                    {
                        mynd_priority_queue_Insert(queues[to], k, vwgt[k] - edegrees[other]);
                        moved[k] = -(2+to);
                    }
                }
            }
            mptr[nswaps + 1] = nmind;

            // printf("Moved %6"PRIDX" to %3"PRIDX", Gain: %5"PRIDX" [%5"PRIDX"] [%4"PRIDX" %4"PRIDX"] \t[%5"PRIDX" %5"PRIDX" %5"PRIDX"]\n", higain, to, g[to], g[other], vwgt[u[to]], vwgt[u[other]], pwgts[0], pwgts[1], pwgts[2]);
        }

        // exam_where(graph);
        // exam_pwgts(graph);
        /****************************************************************
        * Roll back computation 
        *****************************************************************/
        for (nswaps--; nswaps > mincutorder; nswaps--) 
        {
            higain = swaps[nswaps];

            to = where[higain];
            other = (to + 1) % 2;
            pwgts[2] += vwgt[higain];
            pwgts[to] -= vwgt[higain];
            where[higain] = 2;
            nbnd = mynd_insert_queue(nbnd,bndptr,bndind,higain);

            edegrees = rinfo[higain].edegrees;
            edegrees[0] = edegrees[1] = 0;
            for (j = xadj[higain]; j < xadj[higain + 1]; j++) 
            {
                k = adjncy[j];
                if (where[k] == 2) 
                    rinfo[k].edegrees[to] -= vwgt[higain];
                else
                    edegrees[where[k]] += vwgt[k];
            }

            /* Push nodes out of the separator */
            for (j = mptr[nswaps]; j < mptr[nswaps + 1]; j++) 
            {
                k = mind[j];
                where[k] = other;
                pwgts[other] += vwgt[k];
                pwgts[2] -= vwgt[k];
                nbnd = mynd_delete_queue(nbnd,bndptr,bndind,k);
                for (jj = xadj[k]; jj < xadj[k + 1]; jj++) 
                {
                    kk = adjncy[jj];
                    if (where[kk] == 2) 
                        rinfo[kk].edegrees[other] += vwgt[k];
                }
            }
        }

        // exam_where(graph);
        // exam_pwgts(graph);
        // printf("\tMinimum sep: %6"PRIDX" at %5"PRIDX", PWGTS: [%6"PRIDX" %6"PRIDX"], NBND: %6"PRIDX"\n", mincut, mincutorder, pwgts[0], pwgts[1], nbnd);

        graph->mincut = mincut;
        graph->nbnd = nbnd;

        if (mincutorder == -1 || mincut >= initcut)
            break;
    }

    mynd_check_free(mind, sizeof(reordering_int_t) * nvtxs * 2, "FM_2WayNodeRefine2Sided: mind");
    mynd_check_free(mptr, sizeof(reordering_int_t) * (nvtxs + 1), "FM_2WayNodeRefine2Sided: mptr");
    mynd_check_free(swaps, sizeof(reordering_int_t) * nvtxs, "FM_2WayNodeRefine2Sided: swaps");
    mynd_check_free(moved, sizeof(reordering_int_t) * nvtxs, "FM_2WayNodeRefine2Sided: moved");
    mynd_priority_queue_Destroy(queues[1]);
    mynd_priority_queue_Destroy(queues[0]);
}

/*************************************************************************/
/*! This function performs a node-based FM refinement. 
    Each refinement iteration is split into two sub-iterations. 
    In each sub-iteration only moves to one of the left/right partitions 
    is allowed; hence, it is one-sided. 
*/
/**************************************************************************/
void mynd_FM_2WayNodeRefine1Sided(graph_t *graph, reordering_int_t niter)
{
    reordering_int_t i, ii, j, k, jj, kk, nvtxs, nbnd, nswaps, nmind, iend;
    reordering_int_t *xadj, *vwgt, *adjncy, *where, *pwgts, *edegrees, *bndind, *bndptr;
    reordering_int_t *mptr, *mind, *swaps;
    priority_queue_t *queue; 
    nrinfo_t *rinfo;
    reordering_int_t higain, mincut, initcut, mincutorder;	
    reordering_int_t pass, to, other, limit;
    reordering_int_t badmaxpwgt, mindiff, newdiff;
    double mult;

    nvtxs  = graph->nvtxs;
    xadj   = graph->xadj;
    adjncy = graph->adjncy;
    vwgt   = graph->vwgt;

    bndind = graph->bndind;
    bndptr = graph->bndptr;
    where  = graph->where;
    pwgts  = graph->pwgts;
    rinfo  = graph->nrinfo;

    queue = mynd_priority_queue_Create(nvtxs);

    swaps = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "mynd_FM_2WayNodeRefine1Sided: swaps");
    mptr   = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * (nvtxs + 1), "mynd_FM_2WayNodeRefine1Sided: mptr");
    mind   = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs * 2, "mynd_FM_2WayNodeRefine1Sided: mind");
    
    mult = 0.5 * 1.2000499;
    badmaxpwgt = (reordering_int_t)(mult * (pwgts[0] + pwgts[1] + pwgts[2]));

    // printf("Partitions-N1: [%6"PRIDX" %6"PRIDX"] Nv-Nb[%6"PRIDX" %6"PRIDX"]. ISep: %6"PRIDX"\n", pwgts[0], pwgts[1], graph->nvtxs, graph->nbnd, graph->mincut);

    to = (pwgts[0] < pwgts[1] ? 1 : 0);
    for (pass = 0; pass< 2 * niter; pass++) 
    {  
        /* the 2*niter is for the two sides */
        other = to; 
        to    = (to + 1) % 2;

        mynd_priority_queue_Reset(queue);

        mincutorder = -1;
        initcut = mincut = graph->mincut;
        nbnd = graph->nbnd;

        /* use the swaps array in place of the traditional perm array to save memory */
        mynd_irandArrayPermute(nbnd, swaps, nbnd, 1);
        for (ii=0; ii<nbnd; ii++) 
        {
            i = bndind[swaps[ii]];
            mynd_priority_queue_Insert(queue, i, vwgt[i]-rinfo[i].edegrees[other]);
        }

        limit = (0 ? lyj_min(5 * nbnd, 500) : lyj_min(3 * nbnd, 300));

        /******************************************************
        * Get into the FM loop
        *******************************************************/
        mptr[0] = nmind = 0;
        mindiff = lyj_abs(pwgts[0] - pwgts[1]);
        for (nswaps = 0; nswaps < nvtxs; nswaps++) 
        {
            if ((higain = mynd_priority_queue_GetTop(queue)) == -1)
                break;

            /* The following check is to ensure we break out if there is a posibility
                of over-running the mind array.  */
            if (nmind + xadj[higain + 1] - xadj[higain] >= 2 * nvtxs - 1) 
                break;

            if (pwgts[to] + vwgt[higain] > badmaxpwgt) 
                break;  /* No point going any further. Balance will be bad */

            pwgts[2] -= (vwgt[higain] - rinfo[higain].edegrees[other]);

            newdiff = lyj_abs(pwgts[to] + vwgt[higain] - (pwgts[other] - rinfo[higain].edegrees[other]));
            if (pwgts[2] < mincut || (pwgts[2] == mincut && newdiff < mindiff)) 
            {
                mincut      = pwgts[2];
                mincutorder = nswaps;
                mindiff     = newdiff;
            }
            else 
            {
                if (nswaps - mincutorder > 3 * limit || 
                    (nswaps - mincutorder > limit && pwgts[2] > 1.10 * mincut)) 
                {
                    pwgts[2] += (vwgt[higain]-rinfo[higain].edegrees[other]);
                    break; /* No further improvement, break out */
                }
            }

            nbnd = mynd_delete_queue(nbnd,bndptr,bndind,higain);
            pwgts[to]     += vwgt[higain];
            where[higain]  = to;
            swaps[nswaps]  = higain;  


            /**********************************************************
             * Update the degrees of the affected nodes
             ***********************************************************/
            for (j = xadj[higain]; j < xadj[higain + 1]; j++) 
            {
                k = adjncy[j];

                if (where[k] == 2) 
                { 
                    /* For the in-separator vertices modify their edegree[to] */
                    rinfo[k].edegrees[to] += vwgt[higain];
                }
                else if (where[k] == other) 
                { 
                    /* This vertex is pulled into the separator */
                    nbnd = mynd_insert_queue(nbnd,bndptr,bndind,k);
        
                    mind[nmind++] = k;  /* Keep track for rollback */
                    where[k] = 2;
                    pwgts[other] -= vwgt[k];

                    edegrees = rinfo[k].edegrees;
                    edegrees[0] = edegrees[1] = 0;
                    for (jj = xadj[k], iend = xadj[k + 1]; jj < iend; jj++) 
                    {
                        kk = adjncy[jj];
                        if (where[kk] != 2) 
                            edegrees[where[kk]] += vwgt[kk];
                        else 
                        {
                            rinfo[kk].edegrees[other] -= vwgt[k];

                            /* Since the moves are one-sided this vertex has not been moved yet */
                            mynd_priority_queue_Update(queue, kk, vwgt[kk] - rinfo[kk].edegrees[other]); 
                        }
                    }

                    /* Insert the new vertex into the priority queue. Safe due to one-sided moves */
                    mynd_priority_queue_Insert(queue, k, vwgt[k]-edegrees[other]);
                }
            }
            mptr[nswaps+1] = nmind;

            // printf("Moved %6"PRIDX" to %3"PRIDX", Gain: %5"PRIDX" [%5"PRIDX"] \t[%5"PRIDX" %5"PRIDX" %5"PRIDX"] [%3"PRIDX" %2"PRIDX"]\n", 
            //     higain, to, (vwgt[higain]-rinfo[higain].edegrees[other]), vwgt[higain], pwgts[0], pwgts[1], pwgts[2], nswaps, limit);
        }


        /****************************************************************
        * Roll back computation 
        *****************************************************************/
        for (nswaps--; nswaps > mincutorder; nswaps--) 
        {
            higain = swaps[nswaps];

            pwgts[2] += vwgt[higain];
            pwgts[to] -= vwgt[higain];
            where[higain] = 2;
            nbnd = mynd_insert_queue(nbnd,bndptr,bndind,higain);
        
            edegrees = rinfo[higain].edegrees;
            edegrees[0] = edegrees[1] = 0;
            for (j = xadj[higain]; j < xadj[higain + 1]; j++) 
            {
                k = adjncy[j];
                if (where[k] == 2) 
                    rinfo[k].edegrees[to] -= vwgt[higain];
                else
                    edegrees[where[k]] += vwgt[k];
            }

            /* Push nodes out of the separator */
            for (j = mptr[nswaps]; j < mptr[nswaps + 1]; j++) 
            {
                k = mind[j];
                where[k] = other;
                pwgts[other] += vwgt[k];
                pwgts[2] -= vwgt[k];
                nbnd = mynd_delete_queue(nbnd,bndptr,bndind,k);
                for (jj = xadj[k], iend = xadj[k+1]; jj < iend; jj++) 
                {
                    kk = adjncy[jj];
                    if (where[kk] == 2) 
                        rinfo[kk].edegrees[other] += vwgt[k];
                }
            }
        }

        // printf("\tMinimum sep: %6"PRIDX" at %5"PRIDX", PWGTS: [%6"PRIDX" %6"PRIDX"], NBND: %6"PRIDX"\n", mincut, mincutorder, pwgts[0], pwgts[1], nbnd);

        graph->mincut = mincut;
        graph->nbnd   = nbnd;

        if (pass % 2 == 1 && (mincutorder == -1 || mincut >= initcut))
            break;
    }

    mynd_check_free(mind, sizeof(reordering_int_t) * nvtxs * 2, "mynd_FM_2WayNodeRefine1Sided: mind");
    mynd_check_free(mptr, sizeof(reordering_int_t) * (nvtxs + 1), "mynd_FM_2WayNodeRefine1Sided: mptr");
    mynd_check_free(swaps, sizeof(reordering_int_t) * nvtxs, "mynd_FM_2WayNodeRefine1Sided: swaps");

    mynd_priority_queue_Destroy(queue);
}

void mynd_Refine2WayNode(graph_t *graph, graph_t *origraph)
{
    if (graph == origraph) 
        mynd_Compute_Reorder_Informetion_2way(graph);
    else 
    {
        do 
        {
            graph = graph->finer;
            // printf("mynd_Refine2WayNode 0\n");
            mynd_project_Reorder(graph);
            // printf("mynd_Refine2WayNode 1\n");
            // exam_where(graph);
            CONTROL_COMMAND(control, FMNODEBALANCE_Time, mynd_gettimebegin(&start_fmnodebalance, &end_fmnodebalance, &time_fmnodebalance));
            mynd_FM_2WayNodeBalance(graph);
            CONTROL_COMMAND(control, FMNODEBALANCE_Time, mynd_gettimeend(&start_fmnodebalance, &end_fmnodebalance, &time_fmnodebalance));

            // printf("mynd_Refine2WayNode 2\n");
            // exam_where(graph);
            CONTROL_COMMAND(control, FM1SIDENODEREFINE_Time, mynd_gettimebegin(&start_fm1sidenoderefine, &end_fm1sidenoderefine, &time_fm1sidenoderefine));
            mynd_FM_2WayNodeRefine1Sided(graph, 10);
            CONTROL_COMMAND(control, FM1SIDENODEREFINE_Time, mynd_gettimeend(&start_fm1sidenoderefine, &end_fm1sidenoderefine, &time_fm1sidenoderefine));
 
            // printf("mynd_Refine2WayNode 3\n");
            // exam_where(graph);

        } while (graph != origraph);
    }
}

#endif