#ifndef TIMER_H
#define TIMER_H

#include "mynd_functionset.h"

/*reordering_real_t time_all = 0;
struct timeval start_all;
struct timeval end_all;

reordering_real_t time_nestedbisection = 0;
struct timeval start_nestedbisection;
struct timeval end_nestedbisection;

reordering_real_t time_bisectionbest = 0;
struct timeval start_bisectionbest;
struct timeval end_bisectionbest;

reordering_real_t time_coarsen = 0;
struct timeval start_coarsen;
struct timeval end_coarsen;

reordering_real_t time_reorderbisection = 0;
struct timeval start_reorderbisection;
struct timeval end_reorderbisection;

reordering_real_t time_refine2waynode = 0;
struct timeval start_refine2waynode;
struct timeval end_refine2waynode;

reordering_real_t time_splitgraphreorder = 0;
struct timeval start_splitgraphreorder;
struct timeval end_splitgraphreorder;

reordering_real_t time_match = 0;
struct timeval start_match;
struct timeval end_match;

reordering_real_t time_createcoarsengraph = 0;
struct timeval start_createcoarsengraph;
struct timeval end_createcoarsengraph;

reordering_real_t time_partitioninf2way = 0;
struct timeval start_partitioninf2way;
struct timeval end_partitioninf2way;

reordering_real_t time_fm2waycutbalance = 0;
struct timeval start_fm2waycutbalance;
struct timeval end_fm2waycutbalance;

reordering_real_t time_fm2waycutrefine = 0;
struct timeval start_fm2waycutrefine;
struct timeval end_fm2waycutrefine;

reordering_real_t time_reorderinf2way = 0;
struct timeval start_reorderinf2way;
struct timeval end_reorderinf2way;

reordering_real_t time_fmnodebalance = 0;
struct timeval start_fmnodebalance;
struct timeval end_fmnodebalance;

reordering_real_t time_fm1sidenoderefine = 0;
struct timeval start_fm1sidenoderefine;
struct timeval end_fm1sidenoderefine;

reordering_real_t time_fm2sidenoderefine = 0;
struct timeval start_fm2sidenoderefine;
struct timeval end_fm2sidenoderefine;

reordering_real_t time_malloc = 0;
struct timeval start_malloc;
struct timeval end_malloc;

reordering_real_t time_free = 0;
struct timeval start_free;
struct timeval end_free;*/

void mynd_Timer_Init()
{
	time_all = 0;
	time_nestedbisection = 0;
	time_bisectionbest = 0;
	time_coarsen = 0;
	time_reorderbisection = 0;
	time_refine2waynode = 0;
	time_splitgraphreorder = 0;
	time_match = 0;
	time_createcoarsengraph = 0;
	time_partitioninf2way = 0;
	time_fm2waycutbalance = 0;
	time_fm2waycutrefine = 0;
	time_reorderinf2way = 0;
	time_fmnodebalance = 0;
	time_fm1sidenoderefine = 0;
	time_fm2sidenoderefine = 0;
	time_malloc = 0;
	time_free = 0;
}

void mynd_gettimebegin(struct timeval *start, struct timeval *end, reordering_real_t *time)
{
	gettimeofday(start,NULL);
}

void mynd_gettimeend(struct timeval *start, struct timeval *end, reordering_real_t *time)
{
	gettimeofday(end,NULL);
	time[0] += (end[0].tv_sec - start[0].tv_sec) * 1000 + (end[0].tv_usec - start[0].tv_usec) / 1000.0;
}

void mynd_PrintTimeGeneral()
{
	printf("\nTiming Information -------------------------------------------------\n");
	printf("All:                 %10.3"PRREAL" ms\n", time_all);
	printf("-------------------------------------------------------------------\n");
}

void mynd_PrintTimePhases()
{
	printf("\nTiming Information -------------------------------------------------\n");
	printf("All:                 %10.3"PRREAL" ms\n", time_all);
	printf("    Nested Bisection:      %10.3"PRREAL" ms\n", time_nestedbisection);
	printf("        Bisection-Best:          %10.3"PRREAL" ms\n", time_bisectionbest);
	printf("            Coarsen:                   %10.3"PRREAL" ms\n", time_coarsen);
	printf("            Reorder Bisection:         %10.3"PRREAL" ms\n", time_reorderbisection);
	printf("            Refine 2way-Node:          %10.3"PRREAL" ms\n", time_refine2waynode);
	printf("        Reorder Split Graph:     %10.3"PRREAL" ms\n", time_splitgraphreorder);
	printf("-------------------------------------------------------------------\n");

}

void mynd_PrintTimeSteps()
{
	printf("\nTiming Information -------------------------------------------------\n");
	printf("All:                 %10.3"PRREAL" ms\n", time_all);
	printf("    Nested Bisection:      %10.3"PRREAL" ms\n", time_nestedbisection);
	printf("        Bisection-Best:          %10.3"PRREAL" ms\n", time_bisectionbest);
	printf("            Coarsen:                   %10.3"PRREAL" ms\n", time_coarsen);
	printf("                Matching:                    %10.3"PRREAL" ms\n", time_match);
	printf("                Create Coarsen Graph:        %10.3"PRREAL" ms\n", time_createcoarsengraph);
	printf("            Reorder Bisection:         %10.3"PRREAL" ms\n", time_reorderbisection);
	printf("                Compute Partition Inf 2way:  %10.3"PRREAL" ms\n", time_partitioninf2way);
	printf("                FM 2way-Cut Balance:         %10.3"PRREAL" ms\n", time_fm2waycutbalance);
	printf("                FM 2way-Cut Refine:          %10.3"PRREAL" ms\n", time_fm2waycutrefine);
	printf("            Refine 2way-Node:          %10.3"PRREAL" ms\n", time_refine2waynode);
	printf("                Compute Reorder Inf 2way:    %10.3"PRREAL" ms\n", time_reorderinf2way);
	printf("                FM 2way-Node Balance:        %10.3"PRREAL" ms\n", time_fmnodebalance);
	printf("                FM 1Side-Node Refine:        %10.3"PRREAL" ms\n", time_fm1sidenoderefine);
	printf("                FM 2Side-Node Refine:        %10.3"PRREAL" ms\n", time_fm2sidenoderefine);
	printf("        Reorder Split Graph:     %10.3"PRREAL" ms\n", time_splitgraphreorder);
	printf("-------------------------------------------------------------------\n");

}

void mynd_PrintTime(reordering_int_t control)
{
	//	001
	if(control & PRINTTIMESTEPS) 
		mynd_PrintTimeSteps();
	//	010
	else if(control & PRINTTIMEPHASES) 
		mynd_PrintTimePhases();
	//	100
	else if(control & PRINTTIMEGENERAL) 
		mynd_PrintTimeGeneral();
}

#endif