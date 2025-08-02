#ifndef REORDERGRAPH_H
#define REORDERGRAPH_H

#include "mynd_functionset.h"

reordering_int_t num = 0;

void mynd_Reorderpartition(graph_t *graph, reordering_int_t niparts, reordering_int_t level)
{
	graph_t *cgraph;

	reordering_int_t Coarsen_Threshold = graph->nvtxs / 8;
	if (Coarsen_Threshold > 100)
		Coarsen_Threshold = 100;
	else if (Coarsen_Threshold < 40)
		Coarsen_Threshold = 40;

    // printf("Reorderpartition 0\n");
    CONTROL_COMMAND(control, COARSEN_Time, mynd_gettimebegin(&start_coarsen, &end_coarsen, &time_coarsen));
	cgraph = mynd_CoarsenGraph(graph, Coarsen_Threshold);
	CONTROL_COMMAND(control, COARSEN_Time, mynd_gettimeend(&start_coarsen, &end_coarsen, &time_coarsen));    
    // printf("Reorderpartition 1\n");
    // printf("CoarsenGraph end ccnt=%"PRIDX"\n",rand_count());
    // printf("cgraph:\n");
    // exam_nvtxs_nedges(cgraph);
	// exam_xadj(cgraph);
	// exam_vwgt(cgraph);
	// exam_adjncy_adjwgt(cgraph);

	niparts = lyj_max(1, (cgraph->nvtxs <= Coarsen_Threshold ? niparts / 2: niparts));
    CONTROL_COMMAND(control, REORDERBISECTION_Time, mynd_gettimebegin(&start_reorderbisection, &end_reorderbisection, &time_reorderbisection));
	mynd_ReorderBisection(cgraph, niparts);
	CONTROL_COMMAND(control, REORDERBISECTION_Time, mynd_gettimeend(&start_reorderbisection, &end_reorderbisection, &time_reorderbisection));    

    // printf("Reorderpartition 2\n");
    // printf("ReorderBisection end ccnt=%"PRIDX"\n",rand_count());
    // printf("InitSeparator\n");
    // exam_where(cgraph);

    CONTROL_COMMAND(control, REFINE2WAYNODE_Time, mynd_gettimebegin(&start_refine2waynode, &end_refine2waynode, &time_refine2waynode));
	mynd_Refine2WayNode(cgraph, graph);
	CONTROL_COMMAND(control, REFINE2WAYNODE_Time, mynd_gettimeend(&start_refine2waynode, &end_refine2waynode, &time_refine2waynode));    

    // printf("Reorderpartition 3\n");
    // printf("L1 mynd_Refine2WayNode end ccnt=%"PRIDX"\n",rand_count());
    // printf("L1\n");
    // exam_where(graph);

    /*// choose the best one later
    reordering_int_t Coarsen_Threshold = lyj_max(graph->nvtxs / 8, 128);
    printf("Reorderpartition 0\n");
    printf("Coarsen_Threshold=%d\n",Coarsen_Threshold);
    graph_t *cgraph = graph;
    cgraph = mynd_CoarsenGraph(graph,Coarsen_Threshold);
    printf("Reorderpartition 1\n");
    mynd_ReorderBisection(cgraph);
    printf("Reorderpartition 2\n");
    ReorderRefinement(cgraph, graph);*/
}

/*************************************************************************/
/*! This version of the main idea of the Bisection function
		*1 the coarsening is divided into two times
		*2 the graph is small in second coarsening, 
            so can execute a few times to select the best one
		*3 the vertex weight constraint in coarsening is removed
*/
/*************************************************************************/
void mynd_Bisection(graph_t *graph, reordering_int_t niparts, reordering_int_t nthreads, reordering_int_t level)
{
	reordering_int_t i, mincut, nruns = 5;
	graph_t *cgraph; 
	reordering_int_t *bestwhere;

	/* if the graph is small, just find a single vertex separator */
	if (graph->nvtxs < 5000) 
	{
        // printf("Bisection 0\n");
		mynd_Reorderpartition(graph, niparts, level);
        // printf("Bisection 1\n");
		return;
	}

	reordering_int_t Coarsen_Threshold = lyj_max(100, graph->nvtxs / 30);
	
    // printf("Bisection 00\n");
    CONTROL_COMMAND(control, COARSEN_Time, mynd_gettimebegin(&start_coarsen, &end_coarsen, &time_coarsen));
    cgraph = mynd_CoarsenGraphNlevels_metis(graph, Coarsen_Threshold, 4);
	CONTROL_COMMAND(control, COARSEN_Time, mynd_gettimeend(&start_coarsen, &end_coarsen, &time_coarsen));    

    // printf("Bisection 11\n");
    // printf("cgraph:\n");
    // exam_nvtxs_nedges(cgraph);
	// exam_xadj(cgraph);
	// exam_vwgt(cgraph);
	// exam_adjncy_adjwgt(cgraph);

	bestwhere = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * cgraph->nvtxs, "Bisection: bestwhere");
    
	mincut = graph->tvwgt[0];
	for (i = 0; i < nruns; i++) 
	{
        // printf("Bisection 2\n");
		mynd_Reorderpartition(cgraph, 0.7 * niparts, level);
        // printf("Bisection 3\n");

		if (i == 0 || cgraph->mincut < mincut) 
		{
			mincut = cgraph->mincut;
			if (i < nruns - 1)
				mynd_copy_int(cgraph->nvtxs, cgraph->where, bestwhere);
		}

		if (mincut == 0)
			break;

		if (i < nruns - 1) 
			mynd_FreeRefineData(cgraph);
        // printf("Bisection 4\n");
	}

	if (mincut != cgraph->mincut) 
		mynd_copy_int(cgraph->nvtxs, bestwhere, cgraph->where);
    mynd_check_free(bestwhere, sizeof(reordering_int_t) * cgraph->nvtxs, "Bisection: bestwhere");

    // printf("Bisection 5\n");
    CONTROL_COMMAND(control, REFINE2WAYNODE_Time, mynd_gettimebegin(&start_refine2waynode, &end_refine2waynode, &time_refine2waynode));
	mynd_Refine2WayNode(cgraph, graph);
	CONTROL_COMMAND(control, REFINE2WAYNODE_Time, mynd_gettimeend(&start_refine2waynode, &end_refine2waynode, &time_refine2waynode));    

    // printf("Bisection 6\n");
    // printf("L2 mynd_Refine2WayNode end ccnt=%"PRIDX"\n",rand_count());
    // exam_where(graph);
}

void mynd_BisectionBest(graph_t *graph, reordering_int_t nthreads, reordering_int_t level)
{
	/* if the graph is small, just find a single vertex separator */
	if (1 || graph->nvtxs < (0 ? 1000 : 2000)) 
	{
        // printf("BisectionBest 0\n");
		mynd_Bisection(graph, 7, nthreads, level);
        // printf("BisectionBest 1\n");
		return;
	}
}

void mynd_NestedBisection(graph_t *graph, reordering_int_t *reflect, reordering_int_t *reordernumend, reordering_int_t nthreads, reordering_int_t level)
{
    int tid = omp_get_thread_num();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(tid%sysconf(_SC_NPROCESSORS_ONLN), &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0)
    {
        perror("pthread_setaffinity_np error");
    }

    // printf("begin level=%"PRIDX" reordernum=%"PRIDX"\n",level, reordernumend[level]);
    reordering_int_t *treordernum = (reordering_int_t *)malloc(sizeof(reordering_int_t));
    treordernum[0] = reordernumend[level];
	// struct timeval nb_t1, nb_t2;
    // gettimeofday(&nb_t1, NULL);

    reordering_int_t *bndind, *label;
    graph_t *lgraph, *rgraph;

    // printf("NestedBisection 0\n");
    // printf("BisectionBest begin ccnt=%"PRIDX"\n",rand_count());
    // reordering_int_t nthread_id = omp_get_thread_num();
    // printf("nthread_id=%"PRIDX" graph->nvtxs=%"PRIDX"\n",nthread_id,graph->nvtxs);
    //  Bisection
    CONTROL_COMMAND(control, BISECTIONBEST_Time, mynd_gettimebegin(&start_bisectionbest, &end_bisectionbest, &time_bisectionbest));
	mynd_BisectionBest(graph, nthreads, level);
	CONTROL_COMMAND(control, BISECTIONBEST_Time, mynd_gettimeend(&start_bisectionbest, &end_bisectionbest, &time_bisectionbest));

    // printf("BisectionBest end ccnt=%"PRIDX"\n",rand_count());

    //  check function SplitGraphoRerder
    // graph->where = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * graph->nvtxs, "NestedBisection: where");
    // for(reordering_int_t i = 0;i < graph->nvtxs / 3;i++)
    //    graph->where[i] = 0;
    // for(reordering_int_t i = graph->nvtxs / 3;i < graph->nvtxs / 3 * 2;i++)
    //    graph->where[i] = 2;
    // for(reordering_int_t i = graph->nvtxs / 3 * 2;i < graph->nvtxs;i++)
    //    graph->where[i] = 1;
    // exam_where(graph);
    // printf("NestedBisection 1\n");
    //  set up reflect
    bndind = graph->bndind;
    label  = graph->label;
    for(reordering_int_t i = 0;i < graph->nbnd;i++)
    {
        // if( reflect[label[bndind[i]]] != -1)
        // {
        //     printf("task_id=%"PRIDX" reordernum=%"PRIDX" label[bndind[i]]=%"PRIDX" \n",task_id, reordernum,label[bndind[i]]);
        // }
        // jilu_xiabiao[label[bndind[i]]]++;
        reflect[label[bndind[i]]] = --treordernum[0];
        // if( reflect[label[bndind[i]]] < 0 || reflect[label[bndind[i]]] != reordernum)
        //     printf("task_id=%"PRIDX" reordernum=%"PRIDX" graph->label[i]=%"PRIDX" \n",task_id, reordernum,label[bndind[i]]);
    }

    // exam_num(reflect,graph->nvtxs);
    // printf("NestedBisection 2\n");
    //  set up subgraph
    CONTROL_COMMAND(control, SPLITGRAPHREORDER_Time, mynd_gettimebegin(&start_splitgraphreorder, &end_splitgraphreorder, &time_splitgraphreorder));
	mynd_SplitGraphReorder(graph, &lgraph, &rgraph, level);
	CONTROL_COMMAND(control, SPLITGRAPHREORDER_Time, mynd_gettimeend(&start_splitgraphreorder, &end_splitgraphreorder, &time_splitgraphreorder));    
    
    // printf("ordered level=%"PRIDX" graph->nbnd=%"PRIDX" reordernum=%"PRIDX" lgraph->nvtxs=%"PRIDX" rgraph->nvtxs=%"PRIDX"\n",level, graph->nbnd, treordernum[0],lgraph->nvtxs, rgraph->nvtxs);
    
    reordernumend[level * 2 + 1] = treordernum[0] - rgraph->nvtxs;
    reordernumend[level * 2 + 2] = treordernum[0];
    // printf(" lreordernumend=%"PRIDX" rreordernumend=%"PRIDX"\n",reordernumend[level * 2 + 1], reordernumend[level * 2 + 2]);
    // printf("level=%"PRIDX"\n",level++);
    // printf("NestedBisection 3\n");
    mynd_FreeGraph(&graph);

    // nthread_id = omp_get_thread_num();
    // printf("nthread_id=%"PRIDX" lgraph->nvtxs=%"PRIDX" rgraph->nvtxs=%"PRIDX"\n",nthread_id,lgraph->nvtxs, rgraph->nvtxs);

    // if(level == 4)
    //     return ;

    // gettimeofday(&nb_t2, NULL);
    // double time_nb = (nb_t2.tv_sec - nb_t1.tv_sec) * 1000.0 + (nb_t2.tv_usec - nb_t1.tv_usec) / 1000.0;
    // if(level < 4)
    //     printf("level=%"PRIDX" lgraph->nvtxs=%"PRIDX" rgraph->nvtxs=%"PRIDX" NestedBisection used %4.2f ms\n", level, lgraph->nvtxs, rgraph->nvtxs, time_nb);
	// printf("level=%"PRIDX" NestedBisection used %4.2f ms\n", level, time_nb);
    
    // reordering_int_t *treordernum = (reordering_int_t *)malloc(sizeof(reordering_int_t));
    // treordernum[0] = reordernum;
    //  Nest
    if(lgraph->nvtxs > 120 && lgraph->nedges > 0)
    {
        // printf("lgraph 0\n");
        // treordernum[0] = reordernum[0] - rgraph->nvtxs;
        // printf("lgraph level=%"PRIDX" treordernum=%"PRIDX"\n",level * 2 + 1, reordernumend[level * 2 + 1]);
		if(lgraph->nvtxs > 800)
		{
			#pragma omp task
        	// NestedBisection(lgraph, reflect, reordernum - rgraph->nvtxs, nthreads, level * 2 + 1, temp, jilu_xiabiao, jilu_taskid);
            mynd_NestedBisection(lgraph, reflect, reordernumend, nthreads, level * 2 + 1);
		}
		else
	        // NestedBisection(lgraph, reflect, reordernum - rgraph->nvtxs, nthreads, level * 2 + 1, temp, jilu_xiabiao, jilu_taskid);
            mynd_NestedBisection(lgraph, reflect, reordernumend, nthreads, level * 2 + 1);
        // printf("lgraph 1\n");
        // printf("lgraph level=%"PRIDX" treordernum=%"PRIDX"\n",level * 2 + 1, reordernumend[level * 2 + 1]);
    }
    else
    {
        // treordernum[0] = reordernum[0] - rgraph->nvtxs;
        // printf("MMD_Order lgraph 0\n");
        // MMD_Order_line(lgraph, reflect, reordernum - rgraph->nvtxs, -1);
        // MMD_Order_line(lgraph, reflect, reordernumend, level * 2 + 1);
        mynd_MMD_Order(lgraph, reflect, reordernumend, level * 2 + 1);
        // printf("lmmd level=%"PRIDX" treordernum=%"PRIDX"\n",level * 2 + 1, reordernumend[level * 2 + 1]);
        // printf("MMD_Order lgraph 1\n");
        mynd_FreeGraph(&lgraph);
        // printf("MMD_Order lgraph 2\n");
    }
    if(rgraph->nvtxs > 120 && rgraph->nedges > 0)
    {
        // treordernum[0] = reordernum[0];
        // printf("lgraph 0\n");
        // printf("rgraph level=%"PRIDX" treordernum=%"PRIDX"\n",level * 2 + 2, reordernumend[level * 2 + 2]);
		if(rgraph->nvtxs > 800)
		{
			#pragma omp task
        	mynd_NestedBisection(rgraph, reflect, reordernumend, nthreads, level * 2 + 2);
		}
		else
	        mynd_NestedBisection(rgraph, reflect, reordernumend, nthreads, level * 2 + 2);
        // printf("lgraph 1\n");
        // printf("rgraph level=%"PRIDX" treordernum=%"PRIDX"\n",level * 2 + 2, reordernumend[level * 2 + 2]);
    }
    else
    {
        // treordernum[0] = reordernum[0];
        // printf("MMD_Order rgraph 0\n");
        // MMD_Order_line(rgraph, reflect, reordernumend, level * 2 + 2);
        mynd_MMD_Order(rgraph, reflect, reordernumend, level * 2 + 2);
        // printf("rmmd level=%"PRIDX" treordernum=%"PRIDX"\n",level * 2 + 2, reordernumend[level * 2 + 2]);
        // printf("MMD_Order rgraph 1\n");
        mynd_FreeGraph(&rgraph);
        // printf("MMD_Order rgraph 2\n");
    }
}

void mynd_NestedBisection_omp(graph_t *graph, reordering_int_t *reflect, reordering_int_t *reordernum, reordering_int_t nthreads, reordering_int_t level)
{
    reordering_int_t nvtxs = graph->nvtxs;

    reordering_int_t *reordernumend = (reordering_int_t * )mynd_check_malloc(sizeof(reordering_int_t) * nvtxs, "NestedBisection_omp: reordernumend");
    mynd_set_value_int(nvtxs,-1,reordernumend);
    reordernumend[0] = reordernum[0];

    #pragma omp parallel //proc_bind(close)
    #pragma omp single //nowait
    mynd_NestedBisection(graph, reflect, reordernumend, nthreads, 0);

    mynd_check_free(reordernumend, sizeof(reordering_int_t) * nvtxs, "NestedBisection_omp: reordernumend");

}

void mynd_ReorderGraph(reordering_int_t *nvtxs, reordering_int_t *nedges, reordering_int_t *xadj, reordering_int_t *vwgt, reordering_int_t *adjncy, reordering_int_t *adjwgt, 
    reordering_int_t *treflect, reordering_int_t *reflect, reordering_int_t *compress, reordering_int_t *tcontrol, reordering_int_t *is_memery_manage_before, reordering_int_t nthreads)
{
    mynd_Timer_Init();
    control = tcontrol[0];

    CONTROL_COMMAND(control, ALL_Time, mynd_gettimebegin(&start_all, &end_all, &time_all));

    //  init memery manage
    if(is_memery_manage_before[0] == 0)
        if(mynd_init_memery_manage(NULL) == 0)
        {
            char *error_message = (char *)mynd_check_malloc(sizeof(char) * 1024, "check_fopen: error_message");
            sprintf(error_message, "init memery manage failed.");
            mynd_error_exit(error_message);
            return ;
        }
    
    //  init
    mynd_InitRandom(-1);
    graph_t *graph = NULL;
    reordering_int_t nnvtxs = 0;
    reordering_int_t *cptr, *cind;

    // printf("mynd_irandInRange(nvtxs)=%"PRIDX"\n",mynd_irandInRange(nvtxs[0]));

    // printf("ReorderGraph 1\n");

    if(compress[0] == 1)
    {
        // exit(0);
        cptr = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * (nvtxs[0] + 1), "ReorderGraph: cptr");
        cind = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs[0], "ReorderGraph: cind");

        graph = mynd_Compress_Graph(nvtxs[0], xadj, adjncy, vwgt, cptr, cind);

        // printf("Compress_Graph end\n");

        if (graph == NULL) 
        {
            mynd_check_free(cptr, sizeof(reordering_int_t) * (nvtxs[0] + 1), "ReorderGraph: cptr");
            mynd_check_free(cind, sizeof(reordering_int_t) * nvtxs[0], "ReorderGraph: cind");
            compress[0] = 0; 
        }
        else 
        {
            // mynd_check_free(xadj, sizeof(reordering_int_t) * (nvtxs[0] + 1), "main: xadj");
            // mynd_check_free(vwgt, sizeof(reordering_int_t) * nvtxs[0], "main: vwgt");
            // mynd_check_free(adjncy, sizeof(reordering_int_t) * nedges[0], "main: adjncy");
            // mynd_check_free(adjwgt, sizeof(reordering_int_t) * nedges[0], "main: adjwgt");
            nnvtxs = graph->nvtxs;
            // printf("compress nnvtxs=%"PRIDX"\n",nnvtxs);
            // cfactor = 1.0 * (nvtxs[0]) / nnvtxs;
            // if (cfactor > 1.5 && nseps == 1)
            //     nseps = 2;
            //ctrl->nseps = (reordering_int_t)(ctrl->cfactor*ctrl->nseps);
        }
    }

    //  set up graph
    if(compress[0] == 0)
        graph = mynd_SetupGraph(nvtxs[0], xadj, adjncy, vwgt, adjwgt); 
    // printf("graph->tvwgt[0]=%"PRIDX" cgraph->invtvwgt[0]=%lf\n", graph->tvwgt[0],graph->invtvwgt[0]);
    reordering_int_t reordernum = graph->nvtxs;

    // reordering_int_t *reflect = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * nvtxs[0], "ReorderGraph: reflect");
    mynd_set_value_int(nvtxs[0],-1,reflect);
    mynd_set_value_int(nvtxs[0],-1,treflect);
    // printf("ReorderGraph 2\n");

    CONTROL_COMMAND(control, NESTEDBISECTION_Time, mynd_gettimebegin(&start_nestedbisection, &end_nestedbisection, &time_nestedbisection));

    // struct timeval t1, t2;
    omp_set_num_threads(nthreads);
    // gettimeofday(&t1, NULL);
    mynd_NestedBisection_omp(graph, reflect, &reordernum, nthreads, 0);
    // gettimeofday(&t2, NULL);
    // reordering_real_t time_NestedBisection_omp = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    // printf("[%2"PRIDX"-t] NestedBisection used %4.2"PRREAL" ms\n", nthreads, time_NestedBisection_omp);

	// NestedBisection(graph, reflect, reordernum, 0);
	CONTROL_COMMAND(control, NESTEDBISECTION_Time, mynd_gettimeend(&start_nestedbisection, &end_nestedbisection, &time_nestedbisection));

    // printf("ReorderGraph 3\n");
    // printf("NestedBisection end ccnt=%"PRIDX"\n",rand_count());
    // exam_num(reflect, nvtxs[0]);

    if(compress[0] == 0)
        nnvtxs = nvtxs[0];
    // printf("compress[0]=%"PRIDX" nnvtxs=%"PRIDX"\n",compress[0],nnvtxs);
    // reordering_int_t cnt = 0, cnt1 = 0, cnt2 = 0, cnt3 = 0;
    // for(reordering_int_t i = 0;i < nnvtxs;i++)
    // {
    //     if(reflect[i] >= nvtxs[0] || reflect[i] < 0)
    //     {
    //         cnt++;
    //         if(reflect[i] == -1)
    //         {
    //             cnt1++;
    //             // printf("reflect[%"PRIDX"]=%"PRIDX"\n",i,reflect[i]);
    //         }
    //         else 
    //             cnt2++;
    //     }
    //     else
    //     {
    //         if(treflect[reflect[i]] != -1)
    //             cnt3++;
    //         treflect[reflect[i]] = i;
    //     }
    // }
    // printf("cnt=%"PRIDX" cnt1=%"PRIDX" cnt2=%"PRIDX" cnt3=%"PRIDX"\n",cnt, cnt1, cnt2, cnt3);

    if(compress[0])
    {
        for (reordering_int_t i = 0; i < nnvtxs; i++)
            treflect[reflect[i]] = i; 
        for (reordering_int_t l = 0, i = 0, ii = 0; ii < nnvtxs; ii++) 
        {
            i = treflect[ii];
            for (reordering_int_t j = cptr[i];j < cptr[i + 1]; j++)
                reflect[cind[j]] = l++;
        }

        mynd_check_free(cptr, sizeof(reordering_int_t) * (nvtxs[0] + 1), "ReorderGraph: cptr");
        mynd_check_free(cind, sizeof(reordering_int_t) * nvtxs[0], "ReorderGraph: cind");

        mynd_check_free(xadj, sizeof(reordering_int_t) * (nvtxs[0] + 1), "ReorderGraph: xadj");
		mynd_check_free(vwgt, sizeof(reordering_int_t) * nvtxs[0], "ReorderGraph: vwgt");
		mynd_check_free(adjncy, sizeof(reordering_int_t) * nedges[0], "ReorderGraph: adjncy");
		mynd_check_free(adjwgt, sizeof(reordering_int_t) * nedges[0], "ReorderGraph: adjwgt");
    }

    // for (reordering_int_t i = 0; i < nvtxs[0]; i++)
    // {
    //     if(reflect[i] >= nvtxs[0] || reflect[i] < 0)
    //         printf("reflect[%"PRIDX"]=%"PRIDX"\n",i,reflect[i]);
    //     treflect[reflect[i]] = i;
    // }

    // exam_num(ans, nvtxs[0]);

    //  free memery manage
    if(is_memery_manage_before[0] == 0)
        mynd_free_memory_block();

    CONTROL_COMMAND(control, ALL_Time, mynd_gettimeend(&start_all, &end_all, &time_all));
}

#endif