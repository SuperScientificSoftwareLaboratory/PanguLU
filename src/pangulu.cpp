#include "pangulu_common.h"
#include "pangulu_test.h"

int_t CPU_MEMORY = 0;
int_t GPU_MEMORY = 0;
int_t HEAP_SELECT;
calculate_type *TEMP_A_value = NULL;
idx_int *CUDA_B_idx_COL = NULL;
calculate_type *CUDA_TEMP_value = NULL;
int_t *ssssm_col_ops_u = NULL;
idx_int *ssssm_ops_pointer = NULL;
idx_int *getrf_diagIndex_csc = NULL;
idx_int *getrf_diagIndex_csr = NULL;

int_t STREAM_DENSE_INDEX = 0;
int_t INDEX_NUM = 0;
idx_int PANGU_OMP_NUM_THREADS = 4;

int_t FLOP = 0;
double TIME_transport = 0.0;
double TIME_isend = 0.0;
double TIME_receive = 0.0;
double TIME_getrf = 0.0;
double TIME_tstrf = 0.0;
double TIME_gessm = 0.0;
double TIME_gessm_dense = 0.0;
double TIME_gessm_sparse = 0.0;
double TIME_ssssm = 0.0;
double TIME_cuda_memcpy = 0.0;
double TIME_wait = 0.0;
double calculate_TIME_wait = 0.0;

int_32t RANK;
int_32t LEVEL;
int_32t OMP_THREAD;

void pangulu(int ARGC, char **ARGV)
{
    int_32t rank;
    int_32t size;
#ifndef OVERLAP
    MPI_Init(&ARGC, &ARGV);
#else
    int provided;
    MPI_Init_thread(&ARGC, &ARGV, MPI_THREAD_MULTIPLE, &provided);
#endif

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    pangulu_common *common = (pangulu_common *)pangulu_malloc(sizeof(pangulu_common));
    common->rank = rank;
    common->size = size;

    RANK = rank;
    pangulu_get_common(common, ARGC, ARGV, size);
    pangulu_Smatrix *origin_Smatrix = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
    pangulu_init_pangulu_Smatrix(origin_Smatrix);

    if (rank == 0)
    {
        pangulu_read_pangulu_Smatrix(origin_Smatrix, common->file_name);
        if (origin_Smatrix->row == 0)
        {
            printf("matrix read error row is 0\n");
            return;
        }
        // pangulu_display_pangulu_Smatrix(origin_Smatrix);
    }

    pangulu_test(common, origin_Smatrix);

    if (rank == 0)
    {
        pangulu_destroy_part_pangulu_Smatrix(origin_Smatrix);
    }
    else
    {
        free(origin_Smatrix);
        origin_Smatrix = NULL;
    }

    if (rank != 0)
    {
        pangulu_destroy_pangulu_common(common);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return;
}
