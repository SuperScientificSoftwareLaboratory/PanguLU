#ifndef PANGULU_COMMON_H
#define PANGULU_COMMON_H

//#define ADAPTIVE_KERNEL_SELECTION
#define SYNCHRONIZE_FREE
#define PANGULU_MC64
#define METIS
#define SYMBOLIC
#define PANGULU_SPTRSV

#ifdef SYMBOLIC
#define symmetric
#endif

#define CPU_OPTION
#define OVERLAP
//#define CHECK_LU
#define GPU_OPEN
// #define CHECK_TIME

#ifdef GPU_OPEN
#define GPU_TSTRF
#define GPU_GESSM
#define ADD_GPU_MEMORY
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#ifdef GPU_OPEN
#include <cuda_runtime.h>
#endif

#include <omp.h>

typedef int64_t int_t;
typedef int32_t int_32t;
typedef int idx_int;

#define MPI_INT_TYPE MPI_LONG

//#define calculate_type float
#define calculate_type double
// #if calculate_type==double
#define MPI_VAL_TYPE MPI_DOUBLE
// #elif calculate_type==float
//#define MPI_VAL_TYPE MPI_FLOAT
// #endif

#define ERROR 1e-8
#define CHECK_ERROR 1e-8
#define BIN_LENGTH 12
#define pangulu_exchange_PQ(row, P) \
    (row + P - 1) % P

#define calculate_offset(offset_init, now_level, PQ_length) \
    ((offset_init - now_level % PQ_length) < 0) ? (offset_init - now_level % PQ_length + PQ_length) : (offset_init - now_level % PQ_length)

#define pangulu_Calculate_Block(nrow, block) \
    nrow / block + ((nrow % block) ? 1 : 0)

#define calculate_diagonal_rank(level, P, Q) \
    level % P *Q + level % Q

extern int_t CPU_MEMORY;
extern int_t GPU_MEMORY;

extern int_t FLOP;
extern double TIME_transport;
extern double TIME_isend;
extern double TIME_receive;
extern double TIME_getrf;
extern double TIME_tstrf;
extern double TIME_gessm;
extern double TIME_gessm_dense;
extern double TIME_gessm_sparse;
extern double TIME_ssssm;
extern double TIME_cuda_memcpy;
extern double TIME_wait;
extern double calculate_TIME_wait;

extern idx_int PANGU_OMP_NUM_THREADS;

extern calculate_type *TEMP_A_value;
extern calculate_type *CUDA_TEMP_value;
extern idx_int *CUDA_B_idx_COL;
extern int_t *ssssm_col_ops_u;
extern idx_int *ssssm_ops_pointer;
extern idx_int *getrf_diagIndex_csr;
extern idx_int *getrf_diagIndex_csc;
extern int_t calculate_time;

extern int_32t RANK;
extern int_32t LEVEL;
extern int32_t OMP_THREAD;
typedef struct compare_struct
{
    int_t row;
    int_t col;
    int_t kernel_id;
    int_t task_level;
    int_t compare_flag;
} compare_struct;

#ifdef OVERLAP

typedef struct bsem
{
    pthread_mutex_t *mutex;
    pthread_cond_t *cond;
    int_t v;
} bsem;

#endif

typedef struct pangulu_heap
{
    int_t length;
    int_t *heap_queue;
    compare_struct *comapre_queue;
    int_t max_length;
    int_t nnz_flag;

#ifdef OVERLAP
    bsem *heap_bsem;
#endif

} pangulu_heap;

typedef struct pangulu_Smatrix
{

    calculate_type *value;
    calculate_type *value_CSC;
    int_t *CSR_to_CSC_index;
    int_t *CSC_to_CSR_index;
    int_t *rowpointer;
    idx_int *columnindex;
    int_t *columnpointer;
    idx_int *rowindex;
    int_t column;
    int_t row;
    int_t nnz;
    int_32t *nnzU;
    int_t *bin_rowpointer;
    int_t *bin_rowindex;

#ifdef GPU_OPEN
    int_t *CUDA_rowpointer;
    idx_int *CUDA_columnindex;
    calculate_type *CUDA_value;
    int_32t *CUDA_nnzU;
    int_t *CUDA_bin_rowpointer;
    int_t *CUDA_bin_rowindex;

    int *graphInDegree;
    int *d_graphInDegree;
    int *d_id_extractor;
    calculate_type *d_left_sum;
#else
    int_t num_lev;
    int_t *level_idx;
    int_t *level_size;
#endif

} pangulu_Smatrix;

typedef struct pangulu_vector
{
    calculate_type *value;
    int_t row;
} pangulu_vector;

typedef struct pangulu_common
{
    int_32t P;
    int_32t Q;
    int_32t rank;
    int_t N;
    int_32t NB;
    int_32t size;
    char *file_name;
    int_32t sum_rank_size;
    int_32t omp_thread;
    struct timeval start_time;
    struct timeval stop_time;

} pangulu_common;

typedef struct pangulu_block_Smatrix
{

    // reorder array
    int_t *row_perm;
    int_t *col_perm;
    int_t *metis_perm;
    calculate_type *row_scale;
    calculate_type *col_scale;

    // symbolic
    int_t *symbolic_rowpointer;
    int_32t *symbolic_columnindex;

    int_t *block_Smatrix_nnzA_num;
    int_t *block_Smatrix_non_zero_vector_L;
    int_t *block_Smatrix_non_zero_vector_U;
    int_t *mapper_Big_pangulu_Smatrix;
    int_t *Big_pangulu_Smatrix_rowpointer;
    int_t *Big_pangulu_Smatrix_columnindex;
    pangulu_Smatrix **Big_pangulu_Smatrix_value;
    pangulu_Smatrix **Big_pangulu_Smatrix_copy_value;
    int_t *L_pangulu_Smatrix_columnpointer;
    int_t *L_pangulu_Smatrix_rowindex;
    int_t L_Smatrix_nzz;
    pangulu_Smatrix **L_pangulu_Smatrix_value;
    int_t *U_pangulu_Smatrix_rowpointer;
    int_t *U_pangulu_Smatrix_columnindex;
    int_t U_Smatrix_nzz;
    pangulu_Smatrix **U_pangulu_Smatrix_value;
    int_t *mapper_diagonal;
    pangulu_Smatrix **diagonal_Smatrix_L;
    pangulu_Smatrix **diagonal_Smatrix_U;
    pangulu_Smatrix *calculate_L;
    pangulu_Smatrix *calculate_U;
    pangulu_Smatrix *calculate_X;

    int_t task_level_length;
    int_t *task_level_num;
    int_t *mapper_LU;
    int_t *task_flag_id;
    pangulu_heap *heap;
    int_t *now_level_L_length;
    int_t *now_level_U_length;
    int_t *save_now_level_L;
    int_t *save_now_level_U;
    int_t *send_flag;
    int_t *send_diagonal_flag_L;
    int_t *send_diagonal_flag_U;
    int_t *grid_process_id;
    int_t *save_send_rank_flag;
    int_t *level_task_rank_id;
    int_t *real_matrix_flag;
    int_t *sum_flag_block_num;
    int_t *receive_level_num;
    char *save_tmp;

    int_t *level_index;
    int_t *level_index_reverse;
    int_t *mapper_mpi;
    int_t *mapper_mpi_reverse;
    int_t *mpi_level_num;

    char *flag_save_L;
    char *flag_save_U;
    char *flag_dignon_L;
    char *flag_dignon_U;

#ifdef OVERLAP
    bsem *run_bsem1;
    bsem *run_bsem2;

#endif

    // sptrsv  malloc
    pangulu_vector **Big_row_vector;
    pangulu_vector **Big_col_vector;
    int_t *diagonal_flag;
    int_t *L_row_task_nnz;
    int_t *L_col_task_nnz;
    int_t *U_row_task_nnz;
    int_t *U_col_task_nnz;
    pangulu_heap *sptrsv_heap;
    pangulu_vector *save_vector;
    int_t *L_send_flag;
    int_t *U_send_flag;
    int_t *L_sptrsv_task_columnpointer;
    int_t *L_sptrsv_task_rowindex;
    int_t *U_sptrsv_task_columnpointer;
    int_t *U_sptrsv_task_rowindex;

} pangulu_block_Smatrix;

typedef struct pangulu_block_common
{
    int_t N;
    int_32t rank;
    int_32t P;
    int_32t Q;
    int_32t NB;
    int_32t block_length;
    int_32t rank_row_length;
    int_32t rank_col_length;
    int_32t sum_rank_size;
    int_32t max_PQ;
    int_32t every_level_length;

} pangulu_block_common;

#ifdef OVERLAP

typedef struct thread_param
{
    pangulu_block_common *common;
    pangulu_block_Smatrix *Smatrix;
} thread_param;

#endif

#endif
