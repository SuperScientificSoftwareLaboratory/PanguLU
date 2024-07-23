#ifndef PANGULU_TIME_H
#define PANGULU_TIME_H

#include "pangulu_common.h"

void pangulu_time_check_begin(struct timeval *GET_TIME_START)
{
    gettimeofday((GET_TIME_START), NULL);
}

double pangulu_time_check_end(struct timeval *GET_TIME_START)
{
    struct timeval GET_TIME_END;
    gettimeofday((&GET_TIME_END), NULL);
    return (((GET_TIME_END.tv_sec - GET_TIME_START->tv_sec) * 1000.0 + (GET_TIME_END.tv_usec - GET_TIME_START->tv_usec) / 1000.0))/1000.0;
}

void pangulu_time_init()
{
    TIME_transport = 0.0;
    TIME_isend = 0.0;
    TIME_receive = 0.0;
    TIME_getrf = 0.0;
    TIME_tstrf = 0.0;
    TIME_gessm = 0.0;
    TIME_gessm_sparse = 0.0;
    TIME_gessm_dense = 0.0;
    TIME_ssssm = 0.0;
    TIME_cuda_memcpy = 0.0;
    TIME_wait = 0.0;
    return;
}

void pangulu_time_simple_output(int_t rank)
{
    printf("%ld\t%.5lf\t%.5lf\t%.5lf\t%.5lf\t%.5lf\t%.5lf\t%.5lf\n",
           rank,
           calculate_TIME_wait,
           TIME_getrf,
           TIME_tstrf,
           TIME_gessm,
           TIME_ssssm, TIME_gessm+TIME_getrf + TIME_tstrf + TIME_ssssm, TIME_cuda_memcpy);
}


#endif