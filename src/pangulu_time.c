#include "pangulu_common.h"

#ifdef CHECK_TIME
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
    time_transpose = 0.0;
    time_isend = 0.0;
    time_receive = 0.0;
    time_getrf = 0.0;
    time_tstrf = 0.0;
    time_gessm = 0.0;
    time_gessm_sparse = 0.0;
    time_gessm_dense = 0.0;
    time_ssssm = 0.0;
    time_cuda_memcpy = 0.0;
    time_wait = 0.0;
    return;
}

void pangulu_time_simple_output(pangulu_int64_t rank)
{
    printf( FMT_PANGULU_INT64_T "\t" "%.5lf\t%.5lf\t%.5lf\t%.5lf\t%.5lf\t%.5lf\t%.5lf\n",
           rank,
           calculate_time_wait,
           time_getrf,
           time_tstrf,
           time_gessm,
           time_ssssm, time_gessm + time_getrf + time_tstrf + time_ssssm, time_cuda_memcpy);
}
#endif // CHECK_TIME