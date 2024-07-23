#ifndef PANGULU_COMMON_H
#define PANGULU_COMMON_H

//#define ADAPTIVE_KERNEL_SELECTION
#define SYNCHRONIZE_FREE
#define METIS
#define SYMBOLIC
#define PANGULU_SPTRSV

#if !defined(CALCULATE_TYPE_CR64)
#define PANGULU_MC64
#endif

#ifdef SYMBOLIC
#define symmetric
#endif

#define CPU_OPTION
#define OVERLAP
#define CHECK_LU
#define CHECK_TIME

#ifdef GPU_OPEN
#define GPU_TSTRF
#define GPU_GESSM
#define ADD_GPU_MEMORY
#endif

#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
typedef int64_t int_t;
typedef int32_t int_32t;
typedef int idx_int;
typedef unsigned long long pangulu_exblock_ptr;
typedef unsigned int pangulu_exblock_idx;
typedef unsigned int pangulu_inblock_ptr; // 块内rowptr和colptr类型
typedef unsigned int pangulu_inblock_idx; // 块内colidx和rowidx类型
typedef long double pangulu_refinement_hp;

#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include "../include/pangulu_interface_common.h"
#ifdef GPU_OPEN
#include <cuda_runtime.h>
#endif

#include <omp.h>

// PanguLU Strings // W:有错误还能运行 // E:错误不能运行 // I:普通信息
#define PANGULUSTR_W_NO_BLOCK_ON_RANK "[PanguLU Warning] No block on rank %d.\n", RANK
#define PANGULUSTR_W_OPTION_IS_NULLPTR "[PanguLU Warning] Option struct pointer is NULL. Using default options.\n"
#define PANGULU_W_RANK_HEAP_DONT_NULL "[PanguLU Warning] rank %ld heap don't nuLL error have %ld task\n", rank, heap->length
#define PANGULU_W_ERR_RANK "[PanguLU Warning] error rank is %d reiceive message\n", rank
#define PANGULUSTR_E_NB_IS_ZERO "[PanguLU Error] NB is zero.\n"
#define PANGULUSTR_E_INIT_FAILED "[PanguLU Error] pangulu_init failed.\n"
#define PANGULU_E_DONT_SELEC_ERR "[PanguLU Error] don't select error\n"
#define PANGULU_E_RANK_ERR_DO_BIG_LEVEL "[PanguLU Error] rank %d error do the big_level in %ld\n", RANK, heap->nnz_flag
#define PANGULU_E_HEAP_AMPTY "[PanguLU Error] error heap is empty \n"
#define PANGULU_E_CPU_MEM "[PanguLU Error] error ------------ don't have cpu memory, allocating %ld bytes. %s:%ld\n", size, file, line
#define PANGULU_E_ISEND_CSR "[PanguLU Error] pangulu_isend_whole_pangulu_Smatrix_CSR error\n"
#define PANGULU_E_ISEND_CSC "[PanguLU Error] pangulu_isend_whole_pangulu_Smatrix_CSC error\n"
#define PANGULU_E_ROW_IS_NULL "[PanguLU Error] error this matrix exist row is nuLL\n"
#define PANGULU_E_ROW_DONT_HAVE_DIA "[PanguLU Error] ERROR the row %ld don't have diagonal\n", i
#define PANGULU_E_ERR_IN_RRCL "[PanguLU Error] error in rank %d row is %ld col is %ld level is %ld\n", RANK, row, col, level
#define PANGULU_E_K_ID "[PanguLU Error] error don't have this kernel id %ld\n", kernel_id
#define PANGULU_E_ASYM "[PanguLU Error] MPI_Barrier_asym error. Exit(2).\n"
#define PANGULU_E_ADD_DIA "[PanguLU Error] pangulu_add_diagonal_element error\n"
#define PANGULU_E_CUDA_MALLOC "[PanguLU Error] cuda malloc error\n"
#define PANGULU_E_ROW_IS_ZERO "[PanguLU Error] matrix read error row is 0\n"
#define PANGULU_E_MAX_NULL "[PanguLU Error] now max is NULL\n"
#define PANGULU_E_WORK_ERR "[PanguLU Error] work error \n"

#define PANGULUSTR_I_READING_MATRIX "Reading matrix %s\n", mtx_name
#define PANGULUSTR_I_READ_MTX_DONE "Read mtx done.\n"
#define PANGULU_I_VECT2NORM_ERR "the vec2norm |Ax - B | / AX -------error is %12.4e \n", error
#define PANGULU_CHECK_PASS "Check ------------------------------------- pass\n"
#define PANGULU_CHECK_ERROR "Check ------------------------------------ error\n"
#define PANGULU_I_DEV_IS "Device %s\n", prop.name
#define PANGULU_I_TASK_INFO "insert task row %ld col %ld level %ld kernel %ld\n", row, col, task_level, kernel_id
#define PANGULU_I_HEAP_LEN "now length is %ld heap_length is %ld\n", heap->length, heap->max_length
#define PANFULU_I_FOLLOW_Q "the follow queue is :\n"
#define PANGULU_I_A_A_OLD "pangulu_add_A_to_A_old called.\n"
#define PANGULU_I_ADAPTIVE_KERNEL_SELECTION_ON "ADAPTIVE_KERNEL_SELECTION -------------ON\n"
#define PANGULU_I_ADAPTIVE_KERNEL_SELECTION_OFF "ADAPTIVE_KERNEL_SELECTION -------------OFF\n"
#define PANGULU_I_SYNCHRONIZE_FREE_ON "SYNCHRONIZE_FREE -------------ON\n"
#define PANGULU_I_SYNCHRONIZE_FREE_OFF "SYNCHRONIZE_FREE -------------OFF\n"
#define PANGULU_I_BASIC_INFO "N=%ld, NNZ=%ld, NB=%d, PANGU_OMP_NUM_THREADS=%d, OPENBLAS_NUM_THREADS=%d, OMP_NUM_THREADS=%d\n", N, origin_Smatrix->rowpointer[N], NB, PANGU_OMP_NUM_THREADS, openblas_nthreads, omp_nthreads
#define PANGULU_I_TIME_REORDER "PanguLU the reorder time is %lf s\n", pangulu_get_spend_time(common)
#define PANGULU_I_TIME_SYMBOLIC "PanguLU the symbolic time is %lf s\n", pangulu_get_spend_time(common)
#define PANGULU_I_TIME_PRE "PanguLU the preprocess time is %lf s\n", pangulu_get_spend_time(common)
#define PANGULU_I_TIME_NUMERICAL "PanguLU the numerical time is %lf s %lf GFLOPs\n", pangulu_get_spend_time(common), FLOP / pangulu_get_spend_time(common) / 1000000000.0
#define PANGULU_I_TIME_SPTRSV "PanguLU the sptrsv time is %lf s\n", pangulu_get_spend_time(common)

#define MM_MAX_LINE_LENGTH 1025
#define MatrixMarketBanner "%%MatrixMarket"
#define MM_MAX_TOKEN_LENGTH 64

#define setbit(x, y) x |= (1 << y)    // set the yth bit of x is 1
#define getbit(x, y) ((x) >> (y) & 1) // get the yth bit of x

#define PANGULU_MC64_FLAG -5

#define PANGULU_MAX(A, B) \
    (((A) > (B)) ? (A) : (B))

#define PANGULU_MIN(A, B) \
    (((A) < (B)) ? (A) : (B))

#define PANGULU_ABS(A) \
    ((A > 0) ? (A) : (-A))

#define SPTRSV_ERROR 1e-8

#define clrbit(x, y) x &= ~(1 << y) // set the yth bit of x is 0

#define Bound 5000
#define binbd1 64
#define binbd2 4096

#define ZERO_ELEMENT 1e-12

#define ERROR 1e-8

#define MPI_INT_TYPE MPI_LONG

#if defined(CALCULATE_TYPE_CR64)

#define calculate_type double _Complex
#define calculate_real_type double
#define MPI_VAL_TYPE MPI_DOUBLE_COMPLEX
#define PANGULU_COMPLEX

#elif defined(CALCULATE_TYPE_R64)

#define calculate_type double
#define calculate_real_type double
#define MPI_VAL_TYPE MPI_DOUBLE

#else
#error [PanguLU Compile Error] Unknown value type. Set -DCALCULATE_TYPE_CR64 or -DCALCULATE_TYPE_R64 in compile command line.
#endif

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

#if defined(GPU_OPEN) && defined(PANGULU_COMPLEX)
#error Complex on GPU is comming soon.
#endif

extern int_t CPU_MEMORY;
extern int_t CPU_PEAK_MEMORY;
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
extern pangulu_inblock_idx *CUDA_B_idx_COL;
extern int_t *ssssm_col_ops_u;
extern idx_int *ssssm_ops_pointer;
extern idx_int *getrf_diagIndex_csr;
extern idx_int *getrf_diagIndex_csc;
extern int_t calculate_time;

extern idx_int *SSSSM_hash_LU;
extern char *SSSSM_flag_LU;
extern char *SSSSM_flag_L_row;
extern idx_int *SSSSM_hash_L_row;
extern idx_int zip_max_id;
extern idx_int zip_cur_id;
extern calculate_type *SSSSM_L_value;
extern calculate_type *SSSSM_U_value;
extern idx_int *zip_rows;
extern idx_int *zip_cols;
extern idx_int *SSSSM_hash_U_col;

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

typedef struct pangulu_origin_Smatrix
{

    calculate_type *value;
    calculate_type *value_CSC;
    int_t *CSR_to_CSC_index;
    int_t *CSC_to_CSR_index;
    int_t *rowpointer;
    idx_int *columnindex;
    int_t *columnpointer;
    idx_int *rowindex;
    int column;
    int row;
    char zip_flag;
    int zip_id;
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

} pangulu_origin_Smatrix;

typedef struct pangulu_Smatrix
{

    calculate_type *value;
    calculate_type *value_CSC;
    int_t *CSR_to_CSC_index;
    int_t *CSC_to_CSR_index;
    pangulu_inblock_ptr *rowpointer;
    pangulu_inblock_idx *columnindex;
    pangulu_inblock_ptr *columnpointer;
    pangulu_inblock_idx *rowindex;
    int column;
    int row;
    char zip_flag;
    int zip_id;
    int_t nnz;
    int_32t *nnzU;
    pangulu_inblock_ptr *bin_rowpointer;
    pangulu_inblock_idx *bin_rowindex;

#ifdef GPU_OPEN
    pangulu_inblock_ptr *CUDA_rowpointer;
    pangulu_inblock_idx *CUDA_columnindex;
    calculate_type *CUDA_value;
    int_32t *CUDA_nnzU;
    pangulu_inblock_ptr *CUDA_bin_rowpointer;
    pangulu_inblock_idx *CUDA_bin_rowindex;

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
    char *rhs_name;
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
    int_t symbolic_nnz;
    int_t *symbolic_rowpointer;
    int_32t *symbolic_columnindex;
    int_t *symbolic_full_rowpointer;
    int_32t *symbolic_full_columnindex;
    calculate_type* symbolic_full_value;

    int_t *block_Smatrix_nnzA_num;
    int_t *block_Smatrix_non_zero_vector_L;
    int_t *block_Smatrix_non_zero_vector_U;
    int_t *mapper_Big_pangulu_Smatrix;
    char* blocks_current_rank;
    pangulu_Smatrix *Big_pangulu_Smatrix_value;
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

struct pangulu_handle_t
{
    pangulu_block_common *block_common;
    pangulu_block_Smatrix *block_Smatrix;
    pangulu_common *commmon;
};

#endif
