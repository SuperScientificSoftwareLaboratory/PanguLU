#ifndef BALANCE_H
#define BALANCE_H

#include "mynd_functionset.h"

/*************************************************************************/
/*! Computes the maximum load imbalance difference of a partitioning 
    solution over all the constraints. 
    The difference is defined with respect to the allowed maximum 
    unbalance for the respective constraint. 
 */
/**************************************************************************/ 
reordering_real_t mynd_ComputeLoadImbalanceDiff(graph_t *graph, reordering_int_t nparts, reordering_real_t ubvec)
{
    reordering_int_t j, *pwgts;
    reordering_real_t max, cur;

    pwgts = graph->pwgts;

    max = -1.0;
    for (j = 0; j < nparts; j++) 
    {
        cur = pwgts[j] * (reordering_real_t)((reordering_real_t)graph->invtvwgt[0] / (reordering_real_t)0.5) - ubvec;
        if (cur > max)
            max = cur;
    }
    //  need exam
    return max;
}

/*************************************************************************
* This function balances two partitions by moving boundary nodes
* from the domain that is overweight to the one that is underweight.
**************************************************************************/
void mynd_Bnd2WayBalance(graph_t *graph, reordering_real_t *ntpwgts)
{
    reordering_int_t i, ii, j, k, kwgt, nvtxs, nbnd, nswaps, from, to, tmp;
    reordering_int_t *xadj, *vwgt, *adjncy, *adjwgt, *where, *id, *ed, *bndptr, *bndind, *pwgts;
    reordering_int_t *moved, *perm;
    priority_queue_t *queue;
    reordering_int_t higain, mincut, mindiff;
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

    moved = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "mynd_Bnd2WayBalance: moved");
    perm  = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "mynd_Bnd2WayBalance: perm");

    /* Determine from which domain you will be moving data */
    tpwgts[0] = graph->tvwgt[0] * ntpwgts[0];
    tpwgts[1] = graph->tvwgt[0] - tpwgts[0];
    mindiff   = lyj_abs(tpwgts[0] - pwgts[0]);
    from      = (pwgts[0] < tpwgts[0] ? 1 : 0);
    to        = (from + 1) % 2;

    // printf("Partitions: [%6"PRIDX" %6"PRIDX"] T[%6"PRIDX" %6"PRIDX"], Nv-Nb[%6"PRIDX" %6"PRIDX"]. ICut: %6"PRIDX" [B]\n", 
    //     pwgts[0], pwgts[1], tpwgts[0], tpwgts[1], graph->nvtxs, graph->nbnd, graph->mincut);

    queue = mynd_priority_queue_Create(nvtxs);

    mynd_set_value_int(nvtxs, -1, moved);

    /* Insert the boundary nodes of the proper partition whose size is OK in the priority queue */
    nbnd = graph->nbnd;
    mynd_irandArrayPermute(nbnd, perm, nbnd/5, 1);
    for (ii = 0; ii < nbnd; ii++)
    {
        i = perm[ii];
        if (where[bndind[i]] == from && vwgt[bndind[i]] <= mindiff)
            mynd_priority_queue_Insert(queue, bndind[i], ed[bndind[i]] - id[bndind[i]]);
    }

    mincut = graph->mincut;
    for (nswaps = 0; nswaps < nvtxs; nswaps++) 
    {
        if ((higain = mynd_priority_queue_GetTop(queue)) == -1)
            break;

        if (pwgts[to] + vwgt[higain] > tpwgts[to])
            break;

        mincut -= (ed[higain] - id[higain]);
        pwgts[to] += vwgt[higain];
        pwgts[from] -= vwgt[higain];

        where[higain] = to;
        moved[higain] = nswaps;

        // printf("Moved %6"PRIDX" from %"PRIDX". [%3"PRIDX" %3"PRIDX"] %5"PRIDX" [%4"PRIDX" %4"PRIDX"]\n", higain, from, ed[higain]-id[higain], vwgt[higain], mincut, pwgts[0], pwgts[1]);

        /**************************************************************
        * Update the id[i]/ed[i] values of the affected nodes
        ***************************************************************/
        lyj_swap(id[higain], ed[higain], tmp);
        if (ed[higain] == 0 && xadj[higain] < xadj[higain + 1]) 
            nbnd = mynd_delete_queue(nbnd, bndptr,  bndind, higain);

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
                    nbnd = mynd_delete_queue(nbnd, bndptr,  bndind, k);
                    if (moved[k] == -1 && where[k] == from && vwgt[k] <= mindiff)  /* Remove it if in the queues */
                        mynd_priority_queue_Delete(queue, k);
                }
                else 
                { 
                    /* If it has not been moved, update its position in the queue */
                    if (moved[k] == -1 && where[k] == from && vwgt[k] <= mindiff)
                        mynd_priority_queue_Update(queue, k, ed[k] - id[k]);
                }
            }
            else 
            {
                if (ed[k] > 0) 
                {  
                    /* It will now become a boundary vertex */
                    nbnd = mynd_insert_queue(nbnd, bndptr,  bndind, k);
                    if (moved[k] == -1 && where[k] == from && vwgt[k] <= mindiff) 
                        mynd_priority_queue_Insert(queue, k, ed[k] - id[k]);
                }
            }
        }
    }

    // printf("\tMinimum cut: %6"PRIDX", PWGTS: [%6"PRIDX" %6"PRIDX"], NBND: %6"PRIDX"\n", mincut, pwgts[0], pwgts[1], nbnd);
    // printf("mynd_Bnd2WayBalance\n");
    graph->mincut = mincut;
    graph->nbnd   = nbnd;

    mynd_priority_queue_Destroy(queue);

    mynd_check_free(moved, sizeof(reordering_int_t) * nvtxs, "mynd_Bnd2WayBalance: moved");
    mynd_check_free(perm, sizeof(reordering_int_t) * nvtxs, "mynd_Bnd2WayBalance: perm");
}

/*************************************************************************
* This function balances two partitions by moving the highest gain 
* (including negative gain) vertices to the other domain.
* It is used only when tha unbalance is due to non contigous
* subdomains. That is, the are no boundary vertices.
* It moves vertices from the domain that is overweight to the one that 
* is underweight.
**************************************************************************/
void mynd_General2WayBalance(graph_t *graph, reordering_real_t *ntpwgts)
{
    reordering_int_t i, ii, j, k, kwgt, nvtxs, nbnd, nswaps, from, to, tmp;
    reordering_int_t *xadj, *vwgt, *adjncy, *adjwgt, *where, *id, *ed, *bndptr, *bndind, *pwgts;
    reordering_int_t *moved, *perm;
    priority_queue_t *queue;
    reordering_int_t higain, mincut, mindiff;
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

    moved = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "mynd_Bnd2WayBalance: moved");
    perm  = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "mynd_Bnd2WayBalance: perm");

    /* Determine from which domain you will be moving data */
    tpwgts[0] = graph->tvwgt[0] * ntpwgts[0];
    tpwgts[1] = graph->tvwgt[0] - tpwgts[0];
    mindiff   = lyj_abs(tpwgts[0] - pwgts[0]);
    from      = (pwgts[0] < tpwgts[0] ? 1 : 0);
    to        = (from + 1) % 2;

    // printf("Partitions: [%6"PRIDX" %6"PRIDX"] T[%6"PRIDX" %6"PRIDX"], Nv-Nb[%6"PRIDX" %6"PRIDX"]. ICut: %6"PRIDX" [B]\n", 
    //     pwgts[0], pwgts[1], tpwgts[0], tpwgts[1], graph->nvtxs, graph->nbnd, graph->mincut);

    queue = mynd_priority_queue_Create(nvtxs);

    mynd_set_value_int(nvtxs, -1, moved);

    /* Insert the nodes of the proper partition whose size is OK in the priority queue */
    mynd_irandArrayPermute(nvtxs, perm, nvtxs / 5, 1);
    for (ii = 0; ii < nvtxs; ii++) 
    {
        i = perm[ii];
        if (where[i] == from && vwgt[i] <= mindiff)
            mynd_priority_queue_Insert(queue, i, ed[i] - id[i]);
    }

    mincut = graph->mincut;
    nbnd = graph->nbnd;
    for (nswaps = 0; nswaps < nvtxs; nswaps++) 
    {
        if ((higain = mynd_priority_queue_GetTop(queue)) == -1)
            break;

        if (pwgts[to] + vwgt[higain] > tpwgts[to])
            break;

        mincut -= (ed[higain] - id[higain]);
        pwgts[to] += vwgt[higain];
        pwgts[from] -= vwgt[higain];

        where[higain] = to;
        moved[higain] = nswaps;

        // printf("Moved %6"PRIDX" from %"PRIDX". [%3"PRIDX" %3"PRIDX"] %5"PRIDX" [%4"PRIDX" %4"PRIDX"]\n", higain, from, ed[higain]-id[higain], vwgt[higain], mincut, pwgts[0], pwgts[1]);

        /**************************************************************
        * Update the id[i]/ed[i] values of the affected nodes
        ***************************************************************/
        lyj_swap(id[higain], ed[higain], tmp);
        if (ed[higain] == 0 && bndptr[higain] != -1 && xadj[higain] < xadj[higain + 1]) 
            nbnd = mynd_delete_queue(nbnd, bndptr,  bndind, higain);
        if (ed[higain] > 0 && bndptr[higain] == -1)
            nbnd = mynd_insert_queue(nbnd, bndptr,  bndind, higain);

        for (j = xadj[higain]; j < xadj[higain + 1]; j++) 
        {
            k = adjncy[j];

            kwgt = (to == where[k] ? adjwgt[j] : -adjwgt[j]);
            id[k] += kwgt;
            ed[k] -= kwgt;

            /* Update the queue position */
            if (moved[k] == -1 && where[k] == from && vwgt[k] <= mindiff)
                mynd_priority_queue_Update(queue, k, ed[k] - id[k]);

            /* Update its boundary information */
            if (ed[k] == 0 && bndptr[k] != -1) 
                nbnd = mynd_delete_queue(nbnd, bndptr, bndind, k);
            else if (ed[k] > 0 && bndptr[k] == -1)  
                nbnd = mynd_insert_queue(nbnd, bndptr,  bndind, k);
        }
    }

    // printf("\tMinimum cut: %6"PRIDX", PWGTS: [%6"PRIDX" %6"PRIDX"], NBND: %6"PRIDX"\n", mincut, pwgts[0], pwgts[1], nbnd);

    graph->mincut = mincut;
    graph->nbnd   = nbnd;

    mynd_priority_queue_Destroy(queue);

    mynd_check_free(moved, sizeof(reordering_int_t) * nvtxs, "mynd_General2WayBalance: moved");
    mynd_check_free(perm, sizeof(reordering_int_t) * nvtxs, "mynd_General2WayBalance: perm");
}

void mynd_Balance2Way(graph_t *graph, reordering_real_t *ntpwgts)
{
    // printf("Balance2Way begin %lf\n",mynd_ComputeLoadImbalanceDiff(graph, 2, 1.200050));
    // printf("ubvec=%lf\n",ubvec);
    if (mynd_ComputeLoadImbalanceDiff(graph, 2, 1.200050) <= 0) 
        return;

    /* return right away if the balance is OK */
    if (lyj_abs(ntpwgts[0] * graph->tvwgt[0] - graph->pwgts[0]) < 3 * graph->tvwgt[0] / graph->nvtxs)
      return;

    if (graph->nbnd > 0)
    {
        // printf("mynd_Bnd2WayBalance begin\n");
        mynd_Bnd2WayBalance(graph, ntpwgts);
        // printf("mynd_Bnd2WayBalance end\n");
    }
    else
        mynd_General2WayBalance(graph, ntpwgts);
}

/*************************************************************************/
/*! This function balances the left/right partitions of a separator 
    tri-section */
/*************************************************************************/
void mynd_FM_2WayNodeBalance(graph_t *graph)
{
    reordering_int_t i, ii, j, k, jj, kk, nvtxs, nbnd, nswaps, gain;
    reordering_int_t badmaxpwgt, higain, oldgain, to, other;
    reordering_int_t *xadj, *vwgt, *adjncy, *where, *pwgts, *edegrees, *bndind, *bndptr;
    reordering_int_t *perm, *moved;
    priority_queue_t *queue; 
    nrinfo_t *rinfo;
    reordering_real_t mult;

    nvtxs  = graph->nvtxs;
    xadj   = graph->xadj;
    adjncy = graph->adjncy;
    vwgt   = graph->vwgt;

    bndind = graph->bndind;
    bndptr = graph->bndptr;
    where  = graph->where;
    pwgts  = graph->pwgts;
    rinfo  = graph->nrinfo;

    mult = 0.5 * 1.2000499;

    badmaxpwgt = (reordering_int_t)(mult * (pwgts[0] + pwgts[1]));
    if (lyj_max(pwgts[0], pwgts[1]) < badmaxpwgt)
        return;
    if (lyj_abs(pwgts[0] - pwgts[1]) < 3 * graph->tvwgt[0] / nvtxs)
        return;

    to    = (pwgts[0] < pwgts[1] ? 0 : 1); 
    other = (to + 1) % 2;

    queue = mynd_priority_queue_Create(nvtxs);

    perm   = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "FM_2WayNodeBalance: perm");
    moved  = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "FM_2WayNodeBalance: moved");
    mynd_set_value_int(nvtxs,-1,moved);


    // printf("Partitions: [%6"PRIDX" %6"PRIDX"] Nv-Nb[%6"PRIDX" %6"PRIDX"]. ISep: %6"PRIDX" [B]\n", pwgts[0], pwgts[1], graph->nvtxs, graph->nbnd, graph->mincut);

    nbnd = graph->nbnd;
    // printf("FM_2WayNodeBalance ccnt=%"PRIDX"\n",rand_count());
    mynd_irandArrayPermute(nbnd, perm, nbnd, 1);
    // printf("FM_2WayNodeBalance ccnt=%"PRIDX"\n",rand_count());
    for (ii = 0; ii < nbnd; ii++) 
    {
        i = bndind[perm[ii]];
        mynd_priority_queue_Insert(queue, i, vwgt[i]-rinfo[i].edegrees[other]);
    }

    /******************************************************
     * Get into the FM loop
     *******************************************************/
    for (nswaps = 0; nswaps < nvtxs; nswaps++) 
    {
        if ((higain = mynd_priority_queue_GetTop(queue)) == -1)
            break;

        moved[higain] = 1;

        gain = vwgt[higain] - rinfo[higain].edegrees[other];
        badmaxpwgt = (reordering_int_t)(mult * (pwgts[0] + pwgts[1]));

        /* break if other is now underwight */
        if (pwgts[to] > pwgts[other])
            break;

        /* break if balance is achieved and no +ve or zero gain */
        if (gain < 0 && pwgts[other] < badmaxpwgt) 
            break;

        /* skip this vertex if it will violate balance on the other side */
        if (pwgts[to] + vwgt[higain] > badmaxpwgt) 
            continue;

        pwgts[2] -= gain;

        nbnd = mynd_delete_queue(nbnd,bndptr,bndind,higain);        
        pwgts[to] += vwgt[higain];
        where[higain] = to;

        // printf("Moved %6"PRIDX" to %3"PRIDX", Gain: %3"PRIDX", \t[%5"PRIDX" %5"PRIDX" %5"PRIDX"]\n", higain, to, vwgt[higain]-rinfo[higain].edegrees[other], pwgts[0], pwgts[1], pwgts[2]);

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
            { /* This vertex is pulled into the separator */
                nbnd = mynd_insert_queue(nbnd, bndptr, bndind, k);

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

                        if (moved[kk] == -1)
                            mynd_priority_queue_Update(queue, kk, oldgain + vwgt[k]);
                    }
                }

                /* Insert the new vertex into the priority queue */
                mynd_priority_queue_Insert(queue, k, vwgt[k]-edegrees[other]);
            }
        }
    }

    // printf("\tBalanced sep: %6"PRIDX" at %4"PRIDX", PWGTS: [%6"PRIDX" %6"PRIDX"], NBND: %6"PRIDX"\n", pwgts[2], nswaps, pwgts[0], pwgts[1], nbnd);
    // printf("FM_2WayNodeBalance\n");
    graph->mincut = pwgts[2];
    graph->nbnd   = nbnd;

    mynd_priority_queue_Destroy(queue);

    mynd_check_free(moved, sizeof(reordering_int_t) * nvtxs, "FM_2WayNodeBalance: moved");
    mynd_check_free(perm, sizeof(reordering_int_t) * nvtxs, "FM_2WayNodeBalance: perm");
}

#endif