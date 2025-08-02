#include <stdio.h>
#include <stdlib.h>
typedef struct pangulu_init_options
{
    int nthread;
    int nb;
    int gpu_kernel_warp_per_block;
    int gpu_data_move_warp_per_block;
    int reordering_nthread;
    int sizeof_value;
    int is_complex_matrix;
    float mpi_recv_buffer_level;
}pangulu_init_options;

typedef struct pangulu_gstrf_options
{
}pangulu_gstrf_options;

typedef struct pangulu_gstrs_options
{
}pangulu_gstrs_options;
