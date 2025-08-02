#ifndef VARIABLES_H
#define VARIABLES_H

#include "mynd_functionset.h"

reordering_int_t control;

reordering_real_t time_all;
struct timeval start_all;
struct timeval end_all;

reordering_real_t time_nestedbisection;
struct timeval start_nestedbisection;
struct timeval end_nestedbisection;

reordering_real_t time_bisectionbest;
struct timeval start_bisectionbest;
struct timeval end_bisectionbest;

reordering_real_t time_coarsen;
struct timeval start_coarsen;
struct timeval end_coarsen;

reordering_real_t time_reorderbisection;
struct timeval start_reorderbisection;
struct timeval end_reorderbisection;

reordering_real_t time_refine2waynode;
struct timeval start_refine2waynode;
struct timeval end_refine2waynode;

reordering_real_t time_splitgraphreorder;
struct timeval start_splitgraphreorder;
struct timeval end_splitgraphreorder;

reordering_real_t time_match;
struct timeval start_match;
struct timeval end_match;

reordering_real_t time_createcoarsengraph;
struct timeval start_createcoarsengraph;
struct timeval end_createcoarsengraph;

reordering_real_t time_partitioninf2way;
struct timeval start_partitioninf2way;
struct timeval end_partitioninf2way;

reordering_real_t time_fm2waycutbalance;
struct timeval start_fm2waycutbalance;
struct timeval end_fm2waycutbalance;

reordering_real_t time_fm2waycutrefine;
struct timeval start_fm2waycutrefine;
struct timeval end_fm2waycutrefine;

reordering_real_t time_reorderinf2way;
struct timeval start_reorderinf2way;
struct timeval end_reorderinf2way;

reordering_real_t time_fmnodebalance;
struct timeval start_fmnodebalance;
struct timeval end_fmnodebalance;

reordering_real_t time_fm1sidenoderefine;
struct timeval start_fm1sidenoderefine;
struct timeval end_fm1sidenoderefine;

reordering_real_t time_fm2sidenoderefine;
struct timeval start_fm2sidenoderefine;
struct timeval end_fm2sidenoderefine;

reordering_real_t time_malloc;
struct timeval start_malloc;
struct timeval end_malloc;

reordering_real_t time_free;
struct timeval start_free;
struct timeval end_free;

#endif