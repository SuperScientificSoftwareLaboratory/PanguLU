#ifndef CREATECOARSENGRAPH_H
#define CREATECOARSENGRAPH_H

#include "mynd_functionset.h"

void CreateCoarseGraph(graph_t *graph, reordering_int_t cnvtxs)
{
    reordering_int_t nvtxs, cnedges;
	reordering_int_t *xadj, *vwgt, *adjncy, *adjwgt, *match, *cmap;
    reordering_int_t *cxadj, *cvwgt, *cadjncy, *cadjwgt;
    graph_t *cgraph;

	nvtxs  = graph->nvtxs;

	xadj   = graph->xadj;
	vwgt   = graph->vwgt;
	adjncy = graph->adjncy;
	adjwgt = graph->adjwgt;
	cmap   = graph->cmap;
	match  = graph->match;

    //  Set up Coarse Graph
    cgraph   = mynd_SetupCoarseGraph(graph, cnvtxs);
    cxadj    = cgraph->xadj;
    cvwgt    = cgraph->vwgt;
    // cadjncy  = cgraph->adjncy;
    // cadjwgt  = cgraph->adjwgt;

    // printf("CreateCoarseGraph 0\n");
    //  compute cnedges and set up cvwgt cxadj
    cnedges = 0;
    cxadj[0] = 0;
    for(reordering_int_t i = 0;i < nvtxs;i++)
    {
        reordering_int_t u = match[i];

        //  for verticex with small label 
        if(u < i)
            continue;
        
        //  set up cvwgt
        cvwgt[cmap[i]] = vwgt[i];
        if(u != i)
            cvwgt[cmap[i]] += vwgt[u];

        //  !!!ke geng huan er cha cha zhao shu
        reordering_int_t begin, end, length;
        begin = xadj[i];
        end   = xadj[i + 1];
        length = end - begin;
        if(u != i)
            length += xadj[u + 1] - xadj[u];
        if(length == 0)
        {
            cxadj[cmap[i] + 1] = cnedges;
            continue;
        }
        reordering_int_t *temp = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * length, "CreateCoarseGraph: temp");

        // set up temp
        reordering_int_t ptr = 0;
        for(reordering_int_t j = begin;j < end;j++)
        {
            temp[ptr] = cmap[adjncy[j]];
            ptr++;
        }
        if(u != i)
        {
            begin = xadj[u];
            end   = xadj[u + 1];
            for(reordering_int_t j = begin;j < end;j++)
            {
                temp[ptr] = cmap[adjncy[j]];
                ptr++;
            }
        }

        // exam_num(temp,length);
        
        mynd_select_sort(temp,length);

        // exam_num(temp,length);
        
        //  Calculate the required length
        reordering_int_t cnt = 0;
        for(reordering_int_t j = 0;j < length;j++)
        {
            if(cmap[i] == temp[j])
                continue;
            if(j == 0)
                cnt++;
            else if(temp[j] != temp[j - 1])
                cnt++;
            // printf("i=%d temp[j]=%d cnt=%d\n",i,temp[j], cnt);
        }

        cnedges += cnt;
        cxadj[cmap[i] + 1] = cnedges;
        mynd_check_free(temp, sizeof(reordering_int_t) * length, "CreateCoarseGraph: temp");

        // printf("i=%d u=%d cmap[i]=%d cnedges=%d cxadj[i + 1]=%d\n",i, u, cmap[i], cnedges, cxadj[i + 1]);
    }

    cgraph->nvtxs = cnvtxs;
    cgraph->nedges = cnedges;
    cadjncy = cgraph->adjncy = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * cnedges, "CreateCoarseGraph: adjncy");
	cadjwgt = cgraph->adjwgt = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * cnedges, "CreateCoarseGraph: adjwgt");

    // exam_xadj(cgraph);
    // printf("CreateCoarseGraph 1\n");
    //  set up cadjncy cadjwgt
    for(reordering_int_t i = 0;i < nvtxs;i++)
    {
        reordering_int_t u = match[i];

        //  for verticex with small label 
        if(u < i)
            continue;

        //  !!!ke geng huan er cha cha zhao shu
        reordering_int_t begin, end, length;
        begin = xadj[i];
        end   = xadj[i + 1];
        length = end - begin;
        if(u != i)
            length += xadj[u + 1] - xadj[u];
        if(length == 0)
        {
            // cxadj[i + 1] = cnedges;
            continue;
        }
        reordering_int_t *temp = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * length * 2, "CreateCoarseGraph: temp");

        // set up temp
        reordering_int_t ptr = 0;
        for(reordering_int_t j = begin;j < end;j++)
        {
            temp[ptr] = cmap[adjncy[j]];
            temp[ptr + length] = adjwgt[j];
            ptr++;
        }
        if(u != i)
        {
            begin = xadj[u];
            end   = xadj[u + 1];
            for(reordering_int_t j = begin;j < end;j++)
            {
                temp[ptr] = cmap[adjncy[j]];
                temp[ptr + length] = adjwgt[j];
                ptr++;
            }
        }

        // exam_num(temp,length * 2);

        mynd_select_sort_val(temp,length);

        // exam_num(temp,length * 2);

        //  set up cadjncy cadjwgt
        begin = cxadj[cmap[i]];
        end   = cxadj[cmap[i] + 1];
        ptr = cxadj[cmap[i]];
        for(reordering_int_t j = 0;j < length;j++)
        {
            //  remove self-loop
            if(temp[j] == cmap[i])
                continue;
            
            if(j == 0)
            {
                cadjncy[ptr] = temp[j];
                cadjwgt[ptr] = temp[j + length];
                ptr++;
                continue;
            }
            else if(temp[j] != temp[j - 1]) 
            {
                cadjncy[ptr] = temp[j];
                cadjwgt[ptr] = temp[j + length];
                ptr++;
            }
            else
                cadjwgt[ptr - 1] += temp[j + length];
        }

        //  Calculate the required length
        reordering_int_t cnt = 0;
        for(reordering_int_t j = 0;j < length;j++)
        {
            if(cmap[i] == temp[j])
                continue;
            if(j == 0)
                cnt++;
            else if(temp[j] != temp[j - 1])
                cnt++;
            // printf("i=%d temp[j]=%d cnt=%d\n",i,temp[j], cnt);
        }

        mynd_check_free(temp, sizeof(reordering_int_t) * length * 2, "CreateCoarseGraph: temp");

        // printf("ncy:");
        // for(reordering_int_t j = cxadj[cmap[i]];j < cxadj[cmap[i] + 1];j++)
        //     printf("%d ",cadjncy[j]);
        // printf("\n");
        // printf("wgt:");
        // for(reordering_int_t j = cxadj[cmap[i]];j < cxadj[cmap[i] + 1];j++)
        //     printf("%d ",cadjwgt[j]);
        // printf("\n");
    }

    // printf("CreateCoarseGraph 2\n");
    mynd_check_free(graph->match, sizeof(reordering_int_t) * nvtxs, "CreateCoarseGraph: graph->match");
    cgraph->tvwgt[0] = graph->tvwgt[0];
    cgraph->invtvwgt[0] = graph->invtvwgt[0];
    // printf("graph->tvwgt[0]=%"PRIDX" cgraph->invtvwgt[0]=%lf cgraph->tvwgt[0]=%"PRIDX" cgraph->invtvwgt[0]=%lf\n",
    //     graph->tvwgt[0],graph->invtvwgt[0],cgraph->tvwgt[0],cgraph->invtvwgt[0]);
}

void CreateCoarseGraph_S(graph_t *graph, reordering_int_t cnvtxs)
{
    reordering_int_t nvtxs, cnedges, clabel;
	reordering_int_t *xadj, *vwgt, *adjncy, *adjwgt, *match, *cmap;
    reordering_int_t *cxadj, *cvwgt, *cadjncy, *cadjwgt;
    graph_t *cgraph;

	nvtxs  = graph->nvtxs;

	xadj   = graph->xadj;
	vwgt   = graph->vwgt;
	adjncy = graph->adjncy;
	adjwgt = graph->adjwgt;
	cmap   = graph->cmap;
	match  = graph->match;

    //  Set up Coarse Graph
    cgraph   = mynd_SetupCoarseGraph(graph, cnvtxs);
    cxadj    = cgraph->xadj;
    cvwgt    = cgraph->vwgt;
    // cadjncy  = cgraph->adjncy;
    // cadjwgt  = cgraph->adjwgt;

    // printf("CreateCoarseGraph 0\n");
    //  compute cnedges and set up cvwgt cxadj
    cnedges = 0;
    cxadj[0] = 0;
    for(reordering_int_t i = 0;i < nvtxs;i++)
    {
        reordering_int_t u = match[i];

        //  for verticex with small label 
        if(u < i)
            continue;
        
        //  set up cvwgt
        clabel = cmap[i];
        cvwgt[clabel] = vwgt[i];
        if(u != i)
            cvwgt[clabel] += vwgt[u];

        //  !!!ke geng huan er cha cha zhao shu
        reordering_int_t begin, end, length;
        begin = xadj[i];
        end   = xadj[i + 1];
        length = end - begin;
        if(u != i)
            length += xadj[u + 1] - xadj[u];
        if(length == 0)
        {
            cxadj[clabel + 1] = cnedges;
            continue;
        }
        reordering_int_t *temp = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * length, "CreateCoarseGraph: temp");

        // set up temp
        reordering_int_t ptr = 0;
        for(reordering_int_t j = begin;j < end;j++)
        {
            temp[ptr] = cmap[adjncy[j]];
            ptr++;
        }
        if(u != i)
        {
            begin = xadj[u];
            end   = xadj[u + 1];
            for(reordering_int_t j = begin;j < end;j++)
            {
                temp[ptr] = cmap[adjncy[j]];
                ptr++;
            }
        }

        // exam_num(temp,length);
        
        mynd_select_sort(temp,length);

        // exam_num(temp,length);
        
        //  Calculate the required length
        reordering_int_t cnt = 0;
        for(reordering_int_t j = 0;j < length;j++)
        {
            if(clabel == temp[j])
                continue;
            if(j == 0)
                cnt++;
            else if(temp[j] != temp[j - 1])
                cnt++;
            // printf("i=%d temp[j]=%d cnt=%d\n",i,temp[j], cnt);
        }

        cnedges += cnt;
        cxadj[clabel + 1] = cnedges;
        mynd_check_free(temp, sizeof(reordering_int_t) * length, "CreateCoarseGraph_S: temp");

        // printf("i=%d u=%d cmap[i]=%d cnedges=%d cxadj[i + 1]=%d\n",i, u, cmap[i], cnedges, cxadj[i + 1]);
    }

    cgraph->nvtxs = cnvtxs;
    cgraph->nedges = cnedges;
    cadjncy = cgraph->adjncy = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * cnedges, "CreateCoarseGraph: adjncy");
	cadjwgt = cgraph->adjwgt = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * cnedges, "CreateCoarseGraph: adjwgt");

    // exam_xadj(cgraph);
    // printf("CreateCoarseGraph 1\n");
    //  set up cadjncy cadjwgt
    for(reordering_int_t i = 0;i < nvtxs;i++)
    {
        reordering_int_t u = match[i];

        //  for verticex with small label 
        if(u < i)
            continue;

        //  !!!ke geng huan er cha cha zhao shu
        reordering_int_t begin, end, length;
        begin = xadj[i];
        end   = xadj[i + 1];
        length = end - begin;
        if(u != i)
            length += xadj[u + 1] - xadj[u];
        if(length == 0)
        {
            // cxadj[i + 1] = cnedges;
            continue;
        }
        reordering_int_t *temp = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * length * 2, "CreateCoarseGraph: temp");

        // set up temp
        reordering_int_t ptr = 0;
        for(reordering_int_t j = begin;j < end;j++)
        {
            temp[ptr] = cmap[adjncy[j]];
            temp[ptr + length] = adjwgt[j];
            ptr++;
        }
        if(u != i)
        {
            begin = xadj[u];
            end   = xadj[u + 1];
            for(reordering_int_t j = begin;j < end;j++)
            {
                temp[ptr] = cmap[adjncy[j]];
                temp[ptr + length] = adjwgt[j];
                ptr++;
            }
        }

        // exam_num(temp,length * 2);

        mynd_select_sort_val(temp,length);

        // exam_num(temp,length * 2);

        //  set up cadjncy cadjwgt
        clabel = cmap[i];
        begin = cxadj[clabel];
        end   = cxadj[clabel + 1];
        ptr = cxadj[clabel];
        for(reordering_int_t j = 0;j < length;j++)
        {
            //  remove self-loop
            if(temp[j] == clabel)
                continue;
            
            if(j == 0)
            {
                cadjncy[ptr] = temp[j];
                cadjwgt[ptr] = temp[j + length];
                ptr++;
                continue;
            }
            else if(temp[j] != temp[j - 1]) 
            {
                cadjncy[ptr] = temp[j];
                cadjwgt[ptr] = temp[j + length];
                ptr++;
            }
            else
                cadjwgt[ptr - 1] += temp[j + length];
        }

        mynd_check_free(temp, sizeof(reordering_int_t) * length * 2, "CreateCoarseGraph_S: temp");

        // printf("ncy:      ");
        // for(reordering_int_t j = cxadj[clabel];j < cxadj[clabel + 1];j++)
        //     printf("%"PRIDX" ",cadjncy[j]);
        // printf("\n");
        // printf("wgt:      ");
        // for(reordering_int_t j = cxadj[cmap[i]];j < cxadj[cmap[i] + 1];j++)
        //     printf("%"PRIDX" ",cadjwgt[j]);
        // printf("\n");
    }

    // printf("CreateCoarseGraph 2\n");
    mynd_check_free(graph->match, sizeof(reordering_int_t) * nvtxs, "CreateCoarseGraph_S: graph->match");
    cgraph->tvwgt[0] = graph->tvwgt[0];
    cgraph->invtvwgt[0] = graph->invtvwgt[0];
    // printf("graph->tvwgt[0]=%"PRIDX" cgraph->invtvwgt[0]=%lf cgraph->tvwgt[0]=%"PRIDX" cgraph->invtvwgt[0]=%lf\n",
    //     graph->tvwgt[0],graph->invtvwgt[0],cgraph->tvwgt[0],cgraph->invtvwgt[0]);
}

void mynd_CreateCoarseGraph_BST(graph_t *graph, reordering_int_t cnvtxs)
{
    reordering_int_t nvtxs, cnedges, clabel;
	reordering_int_t *xadj, *vwgt, *adjncy, *adjwgt, *match, *cmap;
    reordering_int_t *cxadj, *cvwgt, *cadjncy, *cadjwgt;
    void **addr;
    graph_t *cgraph;

	nvtxs  = graph->nvtxs;

	xadj   = graph->xadj;
	vwgt   = graph->vwgt;
	adjncy = graph->adjncy;
	adjwgt = graph->adjwgt;
	cmap   = graph->cmap;
	match  = graph->match;

    //  Set up Coarse Graph
    cgraph   = mynd_SetupCoarseGraph(graph, cnvtxs);
    cxadj    = cgraph->xadj;
    cvwgt    = cgraph->vwgt;
    // cadjncy  = cgraph->adjncy;
    // cadjwgt  = cgraph->adjwgt;
    //  tree addr
    addr = graph->addr = (void **)mynd_check_malloc(sizeof(void*) * nvtxs, "CreateCoarseGraph: graph->addr");

    // printf("CreateCoarseGraph 0\n");
    //  compute cnedges and set up cvwgt cxadj
    cnedges = 0;
    cxadj[0] = 0;
    for(reordering_int_t i = 0;i < nvtxs;i++)
    {
        reordering_int_t u = match[i];

        //  for verticex with small label 
        if(u < i)
            continue;
        
        //  set up cvwgt
        clabel = cmap[i];
        cvwgt[clabel] = vwgt[i];
        if(u != i)
            cvwgt[clabel] += vwgt[u];

        //  !!!ke geng huan er cha cha zhao shu
        reordering_int_t begin, end;

        //  Binary Search Tree
        binary_search_tree_t *tree;
        tree = mynd_binary_search_tree_Create();

        //  insert
        begin = xadj[i];
        end   = xadj[i + 1];
        for(reordering_int_t j = begin;j < end;j++)
        {
            reordering_int_t t = cmap[adjncy[j]];
            if(t != clabel) 
                mynd_binary_search_tree_Insert(tree, t, adjwgt[j]);
        }
        if(u != i)
        {
            begin = xadj[u];
            end   = xadj[u + 1];
            for(reordering_int_t j = begin;j < end;j++)
            {
                reordering_int_t t = cmap[adjncy[j]];
                if(t != clabel) 
                    mynd_binary_search_tree_Insert(tree, t, adjwgt[j]);
            }
        }

        addr[i] = tree;
        // tree = (binary_search_tree_t *)addr[i];

        //  print
        // binary_search_tree_Traversal(tree, NULL);
        // printf("\n");
        
        //  Calculate the required length
        reordering_int_t cnt = mynd_binary_search_tree_Length(tree);

        cnedges += cnt;
        cxadj[clabel + 1] = cnedges;

        // printf("i=%d u=%d cmap[i]=%d cnedges=%d cxadj[i + 1]=%d\n",i, u, cmap[i], cnedges, cxadj[i + 1]);
    }

    cgraph->nvtxs = cnvtxs;
    cgraph->nedges = cnedges;
    cadjncy = cgraph->adjncy = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * cnedges, "CreateCoarseGraph: adjncy");
	cadjwgt = cgraph->adjwgt = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * cnedges, "CreateCoarseGraph: adjwgt");

    // exam_xadj(cgraph);
    // printf("CreateCoarseGraph 1\n");
    //  set up cadjncy cadjwgt
    for(reordering_int_t i = 0;i < nvtxs;i++)
    {
        reordering_int_t u = match[i];

        //  for verticex with small label 
        if(u < i)
            continue;
        
        //  set up cadjncy cadjwgt
        clabel = cmap[i];

        binary_search_tree_t *tree;
        tree = (binary_search_tree_t *)addr[i];

        //  print
        // printf("traversal:");
        mynd_binary_search_tree_Traversal(tree, &cadjncy[cxadj[clabel]], &cadjwgt[cxadj[clabel]]);
        mynd_binary_search_tree_Destroy(tree);
        // printf("\n");
        // printf("traversal:");
        // for(reordering_int_t j = cxadj[clabel];j < cxadj[clabel + 1];j++)
        //     printf("%"PRIDX" ",cadjncy[j]);
        // printf("\n");
        // printf("wgt:      ");
        // for(reordering_int_t j = cxadj[clabel];j < cxadj[clabel + 1];j++)
        //     printf("%"PRIDX" ",cadjwgt[j]);
        // printf("\n");
        // printf("\n");
    }

    mynd_check_free(graph->addr, sizeof(reordering_int_t) * nvtxs, "CreateCoarseGraph_BST: graph->addr");

    // printf("CreateCoarseGraph 2\n");
    mynd_check_free(graph->match, sizeof(reordering_int_t) * nvtxs, "CreateCoarseGraph_BST: graph->match");
    cgraph->tvwgt[0] = graph->tvwgt[0];
    cgraph->invtvwgt[0] = graph->invtvwgt[0];
    // printf("graph->tvwgt[0]=%"PRIDX" cgraph->invtvwgt[0]=%lf cgraph->tvwgt[0]=%"PRIDX" cgraph->invtvwgt[0]=%lf\n",
    //     graph->tvwgt[0],graph->invtvwgt[0],cgraph->tvwgt[0],cgraph->invtvwgt[0]);
}

void mynd_CreateCoarseGraph_BST_2(graph_t *graph, reordering_int_t cnvtxs)
{
    reordering_int_t nvtxs, cnedges, clabel;
	reordering_int_t *xadj, *vwgt, *adjncy, *adjwgt, *match, *cmap;
    reordering_int_t *cxadj, *cvwgt, *cadjncy, *cadjwgt;
    binary_search_tree2_t *tree;
    graph_t *cgraph;

	nvtxs  = graph->nvtxs;

	xadj   = graph->xadj;
	vwgt   = graph->vwgt;
	adjncy = graph->adjncy;
	adjwgt = graph->adjwgt;
	cmap   = graph->cmap;
	match  = graph->match;

    //  Set up Coarse Graph
    cgraph   = mynd_SetupCoarseGraph(graph, cnvtxs);
    cxadj    = cgraph->xadj;
    cvwgt    = cgraph->vwgt;
    cadjncy = cgraph->adjncy = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * graph->nedges, "CreateCoarseGraph: adjncy");
	cadjwgt = cgraph->adjwgt = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * graph->nedges, "CreateCoarseGraph: adjwgt");

    //  tree addr
    tree = graph->tree = mynd_binary_search_tree_Create2(nvtxs);

    // printf("CreateCoarseGraph 0\n");
    //  compute cnedges and set up cvwgt cxadj
    cnedges = 0;
    cxadj[0] = 0;

    mynd_binary_search_tree_Reset2(tree);

    mynd_exam_binary_search_tree2_flag(tree);

    for(reordering_int_t i = 0;i < nvtxs;i++)
    {
        reordering_int_t u = match[i];

        //  for verticex with small label 
        if(u < i)
            continue;
        
        //  set up cvwgt
        clabel = cmap[i];
        cvwgt[clabel] = vwgt[i];
        if(u != i)
            cvwgt[clabel] += vwgt[u];

        //  !!!ke geng huan er cha cha zhao shu
        reordering_int_t begin, end;

        //  for vertex i
        printf("i begin\n");
        begin = xadj[i];
        end   = xadj[i + 1];
        for(reordering_int_t j = begin;j < end;j++)
        {
            reordering_int_t t = cmap[adjncy[j]];
            if(t != clabel) 
            {
                printf("1 t=%"PRIDX" adjwgt[j]=%"PRIDX"\n",t,adjwgt[j]);
                mynd_binary_search_tree_Insert2(tree, t, adjwgt[j]);
                printf("2\n");
                // exam_binary_search_tree2(tree);
            }
        }
        printf("i end\n");
        if(u != i)
        {
            begin = xadj[u];
            end   = xadj[u + 1];
            for(reordering_int_t j = begin;j < end;j++)
            {
                reordering_int_t t = cmap[adjncy[j]];
                if(t != clabel) 
                {
                    printf("1 t=%"PRIDX" adjwgt[j]=%"PRIDX"\n",t,adjwgt[j]);
                    mynd_binary_search_tree_Insert2(tree, t, adjwgt[j]);
                    printf("2\n");
                    // exam_binary_search_tree2(tree);
                    printf("3\n");
                }
            }
        }

        printf("u end\n");

        mynd_exam_binary_search_tree2(tree);
        printf("exam_binary_search_tree2 end\n");

        //  print
        // binary_search_tree_Traversal(tree, NULL);
        // printf("\n");

        mynd_binary_search_tree_Traversal2(tree, &cadjncy[cxadj[clabel]], &cadjwgt[cxadj[clabel]]);

        printf("Traversal end\n");
        mynd_exam_binary_search_tree2(tree);
        
        //  Calculate the required length
        reordering_int_t cnt = mynd_binary_search_tree_Length2(tree);

        mynd_binary_search_tree_Reset2(tree);
        printf("Reset end\n");
        mynd_exam_binary_search_tree2_flag(tree);

        cnedges += cnt;
        cxadj[clabel + 1] = cnedges;

        // printf("i=%d u=%d cmap[i]=%d cnedges=%d cxadj[i + 1]=%d\n",i, u, cmap[i], cnedges, cxadj[i + 1]);
    }

    cgraph->nvtxs = cnvtxs;
    cgraph->nedges = cnedges;
    cadjncy = cgraph->adjncy = (reordering_int_t *)mynd_check_realloc(cadjncy, sizeof(reordering_int_t) * cnedges, sizeof(reordering_int_t) * graph->nedges, "CreateCoarseGraph: adjncy");
	cadjwgt = cgraph->adjwgt = (reordering_int_t *)mynd_check_realloc(cadjwgt, sizeof(reordering_int_t) * cnedges, sizeof(reordering_int_t) * graph->nedges, "CreateCoarseGraph: adjwgt");

    // exam_xadj(cgraph);
    // printf("CreateCoarseGraph 1\n");

    mynd_binary_search_tree_Destroy2(tree);

    // printf("CreateCoarseGraph 2\n");
    mynd_check_free(graph->match, sizeof(reordering_int_t) * nvtxs, "CreateCoarseGraph_BST_2: graph->match");
    cgraph->tvwgt[0] = graph->tvwgt[0];
    cgraph->invtvwgt[0] = graph->invtvwgt[0];
    // printf("graph->tvwgt[0]=%"PRIDX" cgraph->invtvwgt[0]=%lf cgraph->tvwgt[0]=%"PRIDX" cgraph->invtvwgt[0]=%lf\n",
    //     graph->tvwgt[0],graph->invtvwgt[0],cgraph->tvwgt[0],cgraph->invtvwgt[0]);
}

void mynd_CreateCoarseGraph_HT(graph_t *graph, reordering_int_t cnvtxs)
{
    reordering_int_t nvtxs, cnedges, clabel;
	reordering_int_t *xadj, *vwgt, *adjncy, *adjwgt, *match, *cmap;
    reordering_int_t *cxadj, *cvwgt, *cadjncy, *cadjwgt;
    void **addr;
    graph_t *cgraph;

	nvtxs  = graph->nvtxs;

	xadj   = graph->xadj;
	vwgt   = graph->vwgt;
	adjncy = graph->adjncy;
	adjwgt = graph->adjwgt;
	cmap   = graph->cmap;
	match  = graph->match;

    //  Set up Coarse Graph
    cgraph   = mynd_SetupCoarseGraph(graph, cnvtxs);
    cxadj    = cgraph->xadj;
    cvwgt    = cgraph->vwgt;
    // cadjncy  = cgraph->adjncy;
    // cadjwgt  = cgraph->adjwgt;
    //  tree addr
    addr = graph->addr = (void **)mynd_check_malloc(sizeof(void*) * nvtxs, "CreateCoarseGraph: graph->addr");

    // printf("CreateCoarseGraph 0\n");
    //  compute cnedges and set up cvwgt cxadj
    cnedges = 0;
    cxadj[0] = 0;
    for(reordering_int_t i = 0;i < nvtxs;i++)
    {
        reordering_int_t u = match[i];

        //  for verticex with small label 
        if(u < i)
            continue;
        
        //  set up cvwgt
        clabel = cmap[i];
        cvwgt[clabel] = vwgt[i];
        if(u != i)
            cvwgt[clabel] += vwgt[u];

        //  !!!ke geng huan er cha cha zhao shu
        reordering_int_t begin, end, length;
        begin = xadj[i];
        end   = xadj[i + 1];
        length = end - begin;
        if(u != i)
        {
            begin = xadj[u];
            end   = xadj[u + 1];
            length += end - begin;
        }
        // printf("length=%"PRIDX"\n",length);
        //  Hash Table
        hash_table_t *hash;
        hash = mynd_hash_table_Create(length);

        //  insert
        begin = xadj[i];
        end   = xadj[i + 1];
        for(reordering_int_t j = begin;j < end;j++)
        {
            reordering_int_t t = cmap[adjncy[j]];
            if(t != clabel) 
                mynd_hash_table_Insert(hash, t, adjwgt[j]);
        }
        if(u != i)
        {
            begin = xadj[u];
            end   = xadj[u + 1];
            for(reordering_int_t j = begin;j < end;j++)
            {
                reordering_int_t t = cmap[adjncy[j]];
                if(t != clabel) 
                    mynd_hash_table_Insert(hash, t, adjwgt[j]);
            }
        }

        addr[i] = hash;
        // tree = (binary_search_tree_t *)addr[i];

        //  print
        // binary_search_tree_Traversal(tree, NULL);
        // printf("\n");
        
        //  Calculate the required length
        reordering_int_t cnt = mynd_hash_table_Length(hash);

        cnedges += cnt;
        cxadj[clabel + 1] = cnedges;

        // printf("i=%d u=%d cmap[i]=%d cnedges=%d cxadj[i + 1]=%d\n",i, u, cmap[i], cnedges, cxadj[i + 1]);
    }

    cgraph->nvtxs = cnvtxs;
    cgraph->nedges = cnedges;
    cadjncy = cgraph->adjncy = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * cnedges, "CreateCoarseGraph: adjncy");
	cadjwgt = cgraph->adjwgt = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * cnedges, "CreateCoarseGraph: adjwgt");

    // exam_xadj(cgraph);
    // printf("CreateCoarseGraph 1\n");
    //  set up cadjncy cadjwgt
    for(reordering_int_t i = 0;i < nvtxs;i++)
    {
        reordering_int_t u = match[i];

        //  for verticex with small label 
        if(u < i)
            continue;
        
        //  set up cadjncy cadjwgt
        clabel = cmap[i];

        hash_table_t *hash;
        hash = (hash_table_t *)addr[i];

        //  print
        // printf("traversal:");
        mynd_hash_table_Traversal(hash, &cadjncy[cxadj[clabel]], &cadjwgt[cxadj[clabel]]);
        mynd_hash_table_Destroy(hash);
        // printf("\n");
        // printf("traversal:");
        // for(reordering_int_t j = cxadj[clabel];j < cxadj[clabel + 1];j++)
        //     printf("%"PRIDX" ",cadjncy[j]);
        // printf("\n");
        // printf("wgt:      ");
        // for(reordering_int_t j = cxadj[clabel];j < cxadj[clabel + 1];j++)
        //     printf("%"PRIDX" ",cadjwgt[j]);
        // printf("\n");
        // printf("\n");
    }

    mynd_check_free(graph->addr, sizeof(reordering_int_t) * nvtxs, "CreateCoarseGraph_HT: graph->addr");

    // printf("CreateCoarseGraph 2\n");
    // mynd_check_free(graph->match, sizeof(reordering_int_t) * nvtxs, "CreateCoarseGraph_HT: graph->match");
    cgraph->tvwgt[0] = graph->tvwgt[0];
    cgraph->invtvwgt[0] = graph->invtvwgt[0];
    // printf("graph->tvwgt[0]=%"PRIDX" cgraph->invtvwgt[0]=%lf cgraph->tvwgt[0]=%"PRIDX" cgraph->invtvwgt[0]=%lf\n",
    //     graph->tvwgt[0],graph->invtvwgt[0],cgraph->tvwgt[0],cgraph->invtvwgt[0]);
}

void mynd_CreateCoarseGraph_HT_2(graph_t *graph, reordering_int_t cnvtxs)
{
    reordering_int_t nvtxs, cnedges, clabel;
	reordering_int_t *xadj, *vwgt, *adjncy, *adjwgt, *match, *cmap;
    reordering_int_t *cxadj, *cvwgt, *cadjncy, *cadjwgt;
    hash_table2_t *hash;
    graph_t *cgraph;

	nvtxs  = graph->nvtxs;

	xadj   = graph->xadj;
	vwgt   = graph->vwgt;
	adjncy = graph->adjncy;
	adjwgt = graph->adjwgt;
	cmap   = graph->cmap;
	match  = graph->match;

    //  Set up Coarse Graph
    cgraph   = mynd_SetupCoarseGraph(graph, cnvtxs);
    cxadj    = cgraph->xadj;
    cvwgt    = cgraph->vwgt;
    cadjncy = cgraph->adjncy = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * graph->nedges, "CreateCoarseGraph: adjncy");
	cadjwgt = cgraph->adjwgt = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * graph->nedges, "CreateCoarseGraph: adjwgt");

    // cadjncy  = cgraph->adjncy;
    // cadjwgt  = cgraph->adjwgt;
    //  tree addr
    hash = graph->hash = mynd_hash_table_Create2(nvtxs);

    // printf("CreateCoarseGraph 0\n");
    cxadj[0] = 0;
    cnedges  = 0;
    // exam_num(hash->hashelement,nvtxs);
    // mynd_hash_table_Reset2(hash,&cadjncy[cxadj[clabel]]);
    // exam_num(hash->hashelement,nvtxs);

    for(reordering_int_t i = 0;i < nvtxs;i++)
    {
        reordering_int_t u = match[i];

        //  for verticex with small label 
        if(u < i)
            continue;

        //  set up cvwgt
        clabel = cmap[i];
        cvwgt[clabel] = vwgt[i];
        if(u != i)
            cvwgt[clabel] += vwgt[u];

        reordering_int_t begin, end;
        
        //  for vertex i
        begin = xadj[i];
        end   = xadj[i + 1];
        for(reordering_int_t j = begin;j < end;j++)
        {
            reordering_int_t k = cmap[adjncy[j]];
            // printf("i=%"PRIDX" adjncy[j]=%"PRIDX" k=%"PRIDX" \n",i,adjncy[j],k);
            //  remove self-loop
            if(k == clabel)
                continue;
            reordering_int_t t = mynd_hash_table_Insert2(hash, k, cnedges);
            if(t) 
            {
                cadjncy[cnedges] = k;
                cadjwgt[cnedges] = adjwgt[j];
                cnedges++;
                // printf("k=%"PRIDX" t=%"PRIDX" cnedges=%"PRIDX" cadjncy[cnedges]=%"PRIDX" cadjwgt[cnedges]=%"PRIDX" \n",
                //     k, t, cnedges-1, cadjncy[cnedges-1], cadjwgt[cnedges-1]);
            }
            else
            {
                cadjwgt[mynd_hash_table_Find2(hash, k)] += adjwgt[j];
                // printf("k=%"PRIDX" t=%"PRIDX" find=%"PRIDX" cadjncy[k]=%"PRIDX" cadjwgt[k]=%"PRIDX" \n",
                //     k, t, hash_table_Find2(hash, k), cadjncy[hash_table_Find2(hash, k)], cadjwgt[hash_table_Find2(hash, k)]);
            }
        }
        //  for vertex u
        if(i != u)
        {
            begin = xadj[u];
            end   = xadj[u + 1];
            for(reordering_int_t j = begin;j < end;j++)
            {
                reordering_int_t k = cmap[adjncy[j]];
                // printf("u=%"PRIDX" adjncy[j]=%"PRIDX" k=%"PRIDX" \n",u,adjncy[j],k);

                //  remove self-loop
                if(k == clabel)
                    continue;
                reordering_int_t t = mynd_hash_table_Insert2(hash, k, cnedges);
                if(t) 
                {
                    cadjncy[cnedges] = k;
                    cadjwgt[cnedges] = adjwgt[j];
                    cnedges++;
                    // printf("k=%"PRIDX" t=%"PRIDX" cnedges=%"PRIDX" cadjncy[cnedges]=%"PRIDX" cadjwgt[cnedges]=%"PRIDX" \n",
                    //     k, t, cnedges-1, cadjncy[cnedges-1], cadjwgt[cnedges-1]);
                }
                else
                {
                    cadjwgt[mynd_hash_table_Find2(hash, k)] += adjwgt[j];
                    // printf("k=%"PRIDX" t=%"PRIDX" find=%"PRIDX" cadjncy[k]=%"PRIDX" cadjwgt[k]=%"PRIDX" \n",
                    //     k, t, hash_table_Find2(hash, k), cadjncy[hash_table_Find2(hash, k)], cadjwgt[hash_table_Find2(hash, k)]);
                }
            }
        }

        cxadj[clabel + 1] = cxadj[clabel] + mynd_hash_table_Length2(hash);
        
        //  reset hash table
        // exam_num(hash->hashelement,nvtxs);
        mynd_hash_table_Reset2(hash,&cadjncy[cxadj[clabel]]);
        // exam_num(hash->hashelement,nvtxs);

        // printf("\n");
        // printf("traversal:");
        // for(reordering_int_t j = cxadj[clabel];j < cxadj[clabel + 1];j++)
        //     printf("%"PRIDX" ",cadjncy[j]);
        // printf("\n");
        // printf("wgt:      ");
        // for(reordering_int_t j = cxadj[clabel];j < cxadj[clabel + 1];j++)
        //     printf("%"PRIDX" ",cadjwgt[j]);
        // printf("\n");
        // printf("\n");
    }

    mynd_hash_table_Destroy2(hash);

    cgraph->nvtxs  = cnvtxs;
    cgraph->nedges = cnedges;
    cadjncy = cgraph->adjncy = (reordering_int_t *)mynd_check_realloc(cadjncy, sizeof(reordering_int_t) * cnedges, sizeof(reordering_int_t) * graph->nedges, "CreateCoarseGraph: adjncy");
	cadjwgt = cgraph->adjwgt = (reordering_int_t *)mynd_check_realloc(cadjwgt, sizeof(reordering_int_t) * cnedges, sizeof(reordering_int_t) * graph->nedges, "CreateCoarseGraph: adjwgt");
    
    // exam_xadj(cgraph);
    // printf("CreateCoarseGraph 1\n");

    // printf("CreateCoarseGraph 2\n");
    // mynd_check_free(graph->match, sizeof(reordering_int_t) * nvtxs, "mynd_CreateCoarseGraph_HT_2:graph->match");
    cgraph->tvwgt[0] = graph->tvwgt[0];
    cgraph->invtvwgt[0] = graph->invtvwgt[0];
    // printf("graph->tvwgt[0]=%"PRIDX" cgraph->invtvwgt[0]=%lf cgraph->tvwgt[0]=%"PRIDX" cgraph->invtvwgt[0]=%lf\n",
    //     graph->tvwgt[0],graph->invtvwgt[0],cgraph->tvwgt[0],cgraph->invtvwgt[0]);
}

#endif