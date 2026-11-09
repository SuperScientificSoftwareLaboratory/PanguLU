#ifndef PANGULU_INTERFACE_COMMON_H
#define PANGULU_INTERFACE_COMMON_H
#include <stdio.h>
#include <stdlib.h>
typedef struct pangulu_init_options
{
    int nthread;
    int nb;
    int gpu_kernel_warp_per_block;
    int gpu_data_move_warp_per_block;
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

/* Numerical and argument errors are recoverable through the checked API.
 * Allocation, MPI transport and CUDA runtime failures retain their existing
 * fatal behavior. All checked operations are collective on MPI_COMM_WORLD. */
typedef enum pangulu_status_t {
    PANGULU_SUCCESS = 0,
    PANGULU_INVALID_ARGUMENT = 1,
    PANGULU_INVALID_STATE = 2,
    PANGULU_ZERO_PIVOT = 3,
    PANGULU_NUMERICAL_FAILURE = 4
} pangulu_status_t;
#endif
