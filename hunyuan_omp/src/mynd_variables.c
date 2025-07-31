#ifndef VARIABLES_H
#define VARIABLES_H

#include "mynd_functionset.h"

Hunyuan_int_t control;

Hunyuan_real_t time_all;
struct timeval start_all;
struct timeval end_all;

Hunyuan_real_t time_nestedbisection;
struct timeval start_nestedbisection;
struct timeval end_nestedbisection;

Hunyuan_real_t time_bisectionbest;
struct timeval start_bisectionbest;
struct timeval end_bisectionbest;

Hunyuan_real_t time_coarsen;
struct timeval start_coarsen;
struct timeval end_coarsen;

Hunyuan_real_t time_reorderbisection;
struct timeval start_reorderbisection;
struct timeval end_reorderbisection;

Hunyuan_real_t time_refine2waynode;
struct timeval start_refine2waynode;
struct timeval end_refine2waynode;

Hunyuan_real_t time_splitgraphreorder;
struct timeval start_splitgraphreorder;
struct timeval end_splitgraphreorder;

Hunyuan_real_t time_match;
struct timeval start_match;
struct timeval end_match;

Hunyuan_real_t time_createcoarsengraph;
struct timeval start_createcoarsengraph;
struct timeval end_createcoarsengraph;

Hunyuan_real_t time_partitioninf2way;
struct timeval start_partitioninf2way;
struct timeval end_partitioninf2way;

Hunyuan_real_t time_fm2waycutbalance;
struct timeval start_fm2waycutbalance;
struct timeval end_fm2waycutbalance;

Hunyuan_real_t time_fm2waycutrefine;
struct timeval start_fm2waycutrefine;
struct timeval end_fm2waycutrefine;

Hunyuan_real_t time_reorderinf2way;
struct timeval start_reorderinf2way;
struct timeval end_reorderinf2way;

Hunyuan_real_t time_fmnodebalance;
struct timeval start_fmnodebalance;
struct timeval end_fmnodebalance;

Hunyuan_real_t time_fm1sidenoderefine;
struct timeval start_fm1sidenoderefine;
struct timeval end_fm1sidenoderefine;

Hunyuan_real_t time_fm2sidenoderefine;
struct timeval start_fm2sidenoderefine;
struct timeval end_fm2sidenoderefine;

Hunyuan_real_t time_malloc;
struct timeval start_malloc;
struct timeval end_malloc;

Hunyuan_real_t time_free;
struct timeval start_free;
struct timeval end_free;

#endif