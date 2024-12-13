#ifndef PANGULU_COMMON_H
#define PANGULU_COMMON_H

// #define ADAPTIVE_KERNEL_SELECTION
#define SYNCHRONIZE_FREE
#define SYMBOLIC
#define PANGULU_SPTRSV

#ifdef SYMBOLIC
#define symmetric
#endif

#define CPU_OPTION
#define OVERLAP
#define CHECK_LU
#define CHECK_TIME

#if defined(CALCULATE_TYPE_CR64)
#define calculate_type double _Complex
#define calculate_real_type double
#define MPI_VAL_TYPE MPI_DOUBLE_COMPLEX
#define PANGULU_COMPLEX
#elif defined(CALCULATE_TYPE_R64)
#define calculate_type double
#define calculate_real_type double
#define MPI_VAL_TYPE MPI_DOUBLE
#elif defined(CALCULATE_TYPE_CR32)
#define calculate_type float _Complex
#define calculate_real_type float
#define MPI_VAL_TYPE MPI_C_FLOAT_COMPLEX
#define PANGULU_COMPLEX
#elif defined(CALCULATE_TYPE_R32)
#define calculate_type float
#define calculate_real_type float
#define MPI_VAL_TYPE MPI_FLOAT
#else
#error[PanguLU Compile Error] Unknown value type. Set -DCALCULATE_TYPE_CR64 or -DCALCULATE_TYPE_R64 or -DCALCULATE_TYPE_CR32 or -DCALCULATE_TYPE_R32 in compile command line.
#endif

#if defined(PANGULU_COMPLEX) && defined(PANGULU_MC64)
#undef PANGULU_MC64
#endif

#if defined(GPU_OPEN) && defined(PANGULU_COMPLEX)
#warning Complex value on GPU is not supported now. Fallback to CPU.
#undef GPU_OPEN
#define PANGULU_GPU_COMPLEX_FALLBACK_FLAG
#endif

#ifdef GPU_OPEN
#define GPU_TSTRF
#define GPU_GESSM
#define ADD_GPU_MEMORY
#endif

typedef long long int pangulu_int64_t;
#define MPI_PANGULU_INT64_T MPI_LONG_LONG_INT
#define FMT_PANGULU_INT64_T "%lld"
typedef unsigned long long int pangulu_uint64_t;
#define MPI_PANGULU_UINT64_T MPI_UNSIGNED_LONG_LONG
#define FMT_PANGULU_UINT64_T "%llu"
typedef int pangulu_int32_t;
#define MPI_PANGULU_INT32_T MPI_INT
#define FMT_PANGULU_INT32_T "%d"
typedef unsigned int pangulu_uint32_t;
#define MPI_PANGULU_UINT32_T MPI_UNSIGNED
#define FMT_PANGULU_UINT32_T "%u"
typedef short int pangulu_int16_t;
#define MPI_PANGULU_INT16_T MPI_SHORT
#define FMT_PANGULU_INT16_T "%hd"
typedef unsigned short int pangulu_uint16_t;
#define MPI_PANGULU_UINT16_T MPI_UNSIGNED_SHORT
#define FMT_PANGULU_UINT16_T "%hu"

typedef pangulu_uint64_t pangulu_exblock_ptr;
#define MPI_PANGULU_EXBLOCK_PTR MPI_PANGULU_UINT64_T
#define FMT_PANGULU_EXBLOCK_PTR FMT_PANGULU_UINT64_T
typedef pangulu_uint32_t pangulu_exblock_idx;
#define MPI_PANGULU_EXBLOCK_IDX MPI_PANGULU_UINT32_T
#define FMT_PANGULU_EXBLOCK_IDX FMT_PANGULU_UINT32_T
typedef pangulu_uint32_t pangulu_inblock_ptr;
#define MPI_PANGULU_INBLOCK_PTR MPI_PANGULU_UINT32_T
#define FMT_PANGULU_INBLOCK_PTR FMT_PANGULU_UINT32_T
#ifdef GPU_OPEN
typedef pangulu_uint32_t pangulu_inblock_idx;
#define MPI_PANGULU_INBLOCK_IDX MPI_PANGULU_UINT32_T
#define FMT_PANGULU_INBLOCK_IDX FMT_PANGULU_UINT32_T
#else
typedef pangulu_uint16_t pangulu_inblock_idx;
#define MPI_PANGULU_INBLOCK_IDX MPI_PANGULU_UINT16_T
#define FMT_PANGULU_INBLOCK_IDX FMT_PANGULU_UINT16_T
#endif

typedef pangulu_exblock_ptr sparse_pointer_t;
typedef pangulu_exblock_idx sparse_index_t;
typedef calculate_type sparse_value_t;
typedef calculate_real_type sparse_value_real_t;

extern pangulu_int64_t cpu_memory;
extern pangulu_int64_t cpu_peak_memory;
extern pangulu_int64_t gpu_memory;

extern pangulu_int64_t flop;
extern double time_transpose;
extern double time_isend;
extern double time_receive;
extern double time_getrf;
extern double time_tstrf;
extern double time_gessm;
extern double time_gessm_dense;
extern double time_gessm_sparse;
extern double time_ssssm;
extern double time_cuda_memcpy;
extern double time_wait;
extern double calculate_time_wait;

extern pangulu_int32_t pangu_omp_num_threads;

extern calculate_type *temp_a_value;
extern calculate_type *cuda_temp_value;
extern pangulu_int32_t *cuda_b_idx_col;
extern pangulu_int64_t *ssssm_col_ops_u;
extern pangulu_int32_t *ssssm_ops_pointer;
extern pangulu_int32_t *getrf_diagIndex_csr;
extern pangulu_int32_t *getrf_diagIndex_csc;
extern pangulu_int64_t calculate_time;

extern pangulu_int32_t *ssssm_hash_lu;
extern pangulu_int32_t *ssssm_hash_l_row;
extern pangulu_int32_t zip_cur_id;
extern calculate_type *ssssm_l_value;
extern calculate_type *ssssm_u_value;
extern pangulu_int32_t *ssssm_hash_u_col;

extern pangulu_int64_t TEMP_calculate_type_len;
extern calculate_type *TEMP_calculate_type;
extern pangulu_int64_t TEMP_pangulu_inblock_ptr_len;
extern pangulu_inblock_ptr *TEMP_pangulu_inblock_ptr;

extern pangulu_int32_t rank;
extern pangulu_int32_t global_level;
extern pangulu_int32_t omp_thread;

extern pangulu_int64_t heap_select;

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <mpi.h>
#define __USE_GNU
#include <sched.h>
#include <pthread.h>
#include <cblas.h>
#include <getopt.h>
#include <omp.h>
#include "../include/pangulu.h"

#ifndef PANGULU_PLATFORM_ENV
#ifdef METIS
#include <metis.h>
#else
typedef int idx_t;
#endif
#endif

#ifdef GPU_OPEN
#include <cuda_runtime.h>
#include "./platforms/02_GPU/01_CUDA/000_CUDA/pangulu_cuda.h"
#endif

// #if !defined(PANGULU_EN)
#define PANGULU_EN
// #endif

#if defined(PANGULU_LOG_INFO) && !defined(PANGULU_LOG_WARNING)
#define PANGULU_LOG_WARNING
#endif
#if defined(PANGULU_LOG_WARNING) && !defined(PANGULU_LOG_ERROR)
#define PANGULU_LOG_ERROR
#endif

#include "./languages/pangulu_en.h"

#if !defined(PANGULU_LOG_ERROR)
#define PANGULU_E_NB_IS_ZERO ""
#define PANGULU_E_INVALID_HEAP_SELECT ""
#define PANGULU_E_HEAP_FULL ""
#define PANGULU_E_HEAP_EMPTY ""
#define PANGULU_E_CPU_MEM ""
#define PANGULU_E_ISEND_CSR ""
#define PANGULU_E_ISEND_CSC ""
#define PANGULU_E_ROW_IS_NULL ""
#define PANGULU_E_ROW_DONT_HAVE_DIA ""
#define PANGULU_E_ERR_IN_RRCL ""
#define PANGULU_E_K_ID ""
#define PANGULU_E_ASYM ""
#define PANGULU_E_ADD_DIA ""
#define PANGULU_E_CUDA_MALLOC ""
#define PANGULU_E_ROW_IS_ZERO ""
#define PANGULU_E_MAX_NULL ""
#define PANGULU_E_WORK_ERR ""
#define PANGULU_E_BIP_PTR_INVALID ""
#define PANGULU_E_BIP_INVALID ""
#define PANGULU_E_BIP_NOT_EMPTY ""
#define PANGULU_E_BIP_OUT_OF_RANGE ""
#define PANGULU_E_OPTION_IS_NULLPTR ""
#define PANGULU_E_GSTRF_OPTION_IS_NULLPTR ""
#define PANGULU_E_GSTRS_OPTION_IS_NULLPTR ""
#endif // !defined(PANGULU_LOG_ERROR)

#if !defined(PANGULU_LOG_WARNING)
#define PANGULU_W_RANK_HEAP_DONT_NULL ""
#define PANGULU_W_ERR_RANK ""
#define PANGULU_W_BIP_INCREASE_SPEED_TOO_SMALL ""
#define PANGULU_W_GPU_BIG_BLOCK ""
#define PANGULU_W_COMPLEX_FALLBACK ""
#endif // !defined(PANGULU_LOG_WARNING)

#if !defined(PANGULU_LOG_INFO)
#define PANGULU_I_VECT2NORM_ERR ""
#define PANGULU_I_CHECK_PASS ""
#define PANGULU_I_CHECK_ERROR ""
#define PANGULU_I_DEV_IS ""
#define PANGULU_I_TASK_INFO ""
#define PANGULU_I_HEAP_LEN ""
#define PANGULU_I_ADAPTIVE_KERNEL_SELECTION_ON ""
#define PANGULU_I_ADAPTIVE_KERNEL_SELECTION_OFF ""
#define PANGULU_I_SYNCHRONIZE_FREE_ON ""
#define PANGULU_I_SYNCHRONIZE_FREE_OFF ""
#define PANGULU_I_BASIC_INFO ""
#define PANGULU_I_TIME_REORDER ""
#define PANGULU_I_TIME_SYMBOLIC ""
#define PANGULU_I_TIME_PRE ""
#define PANGULU_I_TIME_NUMERICAL ""
#define PANGULU_I_TIME_SPTRSV ""
#define PANGULU_I_SYMBOLIC_NONZERO ""
#endif // PANGULU_LOG_INFO

#define MM_MAX_LINE_LENGTH 1025
#define MatrixMarketBanner "%%MatrixMarket"
#define MM_MAX_TOKEN_LENGTH 64

#define PANGULU_BIP_INITIAL_LEN 1000
#define PANGULU_BIP_INCREASE_SPEED 1.2
#define PANGULU_BIP_SIBLING_LEN 8
#define PANGULU_BIP_MAP_LENGTH(index_upper_bound) ((index_upper_bound + PANGULU_BIP_SIBLING_LEN - 1) / PANGULU_BIP_SIBLING_LEN)

#define PANGULU_ICEIL(a, b) (((a) + (b) - 1) / (b))
#define setbit(x, y) x |= (1 << y)    // set the yth bit of x is 1
#define getbit(x, y) ((x) >> (y) & 1) // get the yth bit of x

#define PANGULU_MC64_FLAG -5

#define PANGULU_MAX(a, b) \
    (((a) > (b)) ? (a) : (b))

#define PANGULU_MIN(a, b) \
    (((a) < (b)) ? (a) : (b))

#define PANGULU_ABS(a) \
    ((a > 0) ? (a) : (-a))

#define SPTRSV_ERROR 1e-8

#define clrbit(x, y) x &= ~(1 << y) // set the yth bit of x is 0

#define Bound 5000
#define binbd1 64
#define binbd2 4096

#define ZERO_ELEMENT 1e-12

#define ERROR 1e-8

#define ERROR 1e-8
#define CHECK_ERROR 1e-8
#define BIN_LENGTH 12
#define pangulu_exchange_PQ(row, p) \
    (row + p - 1) % p

#define calculate_offset(offset_init, now_level, PQ_length) \
    ((offset_init - now_level % PQ_length) < 0) ? (offset_init - now_level % PQ_length + PQ_length) : (offset_init - now_level % PQ_length)

#define pangulu_Calculate_Block(nrow, block) \
    nrow / block + ((nrow % block) ? 1 : 0)

#define calculate_diagonal_rank(level, p, q) \
    level % p *q + level % q

#ifndef CPU_ZERO
#define CPU_ZERO(cpusetp)                                                     \
    do                                                                        \
    {                                                                         \
        unsigned int __i;                                                     \
        unsigned char *__bits = (unsigned char *)(cpusetp);                   \
        for (__i = 0; __i < sizeof(cpu_set_t) / sizeof(unsigned char); ++__i) \
            __bits[__i] = (unsigned char)0;                                   \
    } while (0)
#endif

#ifndef CPU_SET
#define CPU_SET(cpu, cpusetp) (((unsigned char *)(cpusetp))[(cpu) / 8] |= (1U << ((cpu) % 8)))
#endif

#define pangulu_exit(ret) exit(ret)

typedef struct compare_struct
{
    pangulu_exblock_idx row;
    pangulu_exblock_idx col;
    pangulu_int16_t kernel_id;
    pangulu_exblock_idx task_level;
    pangulu_int64_t compare_flag;
} compare_struct;

#ifdef OVERLAP

typedef struct bsem
{
    pthread_mutex_t *mutex;
    pthread_cond_t *cond;
    pangulu_int32_t v;
} bsem;

#endif

typedef struct pangulu_heap
{
    pangulu_int64_t length;
    pangulu_int64_t *heap_queue;
    compare_struct *comapre_queue;
    pangulu_int64_t max_length;
    pangulu_int64_t nnz_flag;

#ifdef OVERLAP
    bsem *heap_bsem;
#endif

} pangulu_heap;

typedef struct pangulu_origin_smatrix
{

    calculate_type *value;
    calculate_type *value_csc;
    pangulu_exblock_ptr *csr_to_csc_index;
    pangulu_exblock_ptr *csc_to_csr_index;
    pangulu_exblock_ptr *rowpointer;
    pangulu_exblock_idx *columnindex;
    pangulu_exblock_ptr *columnpointer;
    pangulu_exblock_idx *rowindex;
    pangulu_exblock_idx column;
    pangulu_exblock_idx row;
    char zip_flag;
    int zip_id;
    pangulu_exblock_ptr nnz;
    pangulu_int32_t *nnzu;
    pangulu_exblock_ptr *bin_rowpointer;
    pangulu_exblock_idx *bin_rowindex;

#ifndef GPU_OPEN
    pangulu_int64_t num_lev;
    pangulu_int64_t *level_idx;
    pangulu_int64_t *level_size;
#endif

} pangulu_origin_smatrix;

typedef struct pangulu_smatrix
{

    calculate_type *value;
    calculate_type *value_csc;
    pangulu_inblock_ptr *csr_to_csc_index;
    pangulu_inblock_ptr *csc_to_csr_index;
    pangulu_inblock_ptr *rowpointer;
    pangulu_inblock_idx *columnindex;
    pangulu_inblock_ptr *columnpointer;
    pangulu_inblock_idx *rowindex;
    pangulu_inblock_idx column;
    pangulu_inblock_idx row;
    char zip_flag;
    int zip_id;
    pangulu_inblock_ptr nnz;
    pangulu_int32_t *nnzu;
    pangulu_inblock_ptr *bin_rowpointer;
    pangulu_inblock_idx *bin_rowindex;

#ifdef GPU_OPEN
    pangulu_inblock_ptr *cuda_rowpointer;
    pangulu_inblock_idx *cuda_columnindex;
    calculate_type *cuda_value;
    pangulu_int32_t *cuda_nnzu;
    pangulu_inblock_ptr *cuda_bin_rowpointer;
    pangulu_inblock_idx *cuda_bin_rowindex;

    pangulu_int32_t *graphindegree;
    pangulu_int32_t *d_graphindegree;
    pangulu_int32_t *d_id_extractor;
    calculate_type *d_left_sum;
#else
    pangulu_int64_t num_lev;
    pangulu_int64_t *level_idx;
    pangulu_int64_t *level_size;
#endif

} pangulu_smatrix;

typedef struct pangulu_vector
{
    calculate_type *value;
    pangulu_int64_t row;
} pangulu_vector;

typedef struct pangulu_common
{
    pangulu_int32_t p;
    pangulu_int32_t q;
    pangulu_int32_t rank;
    pangulu_exblock_idx n;
    pangulu_inblock_idx nb;
    pangulu_int32_t size;
    char *file_name;
    char *rhs_name;
    pangulu_int32_t sum_rank_size;
    pangulu_int32_t omp_thread;
    struct timeval start_time;
    struct timeval stop_time;

} pangulu_common;

typedef struct pangulu_block_info
{
    pangulu_int64_t block_smatrix_nnza_num; // default : 0
    pangulu_int64_t sum_flag_block_num;     // default : 0
    pangulu_int64_t mapper_a;               // default : -1
    pangulu_int64_t tmp_save_block_num;     // default : -1
    pangulu_int64_t task_flag_id;           // default : 0
    pangulu_int64_t mapper_mpi;             // default : -1
    pangulu_int64_t mapper_lu;              // default : -1
} pangulu_block_info;

typedef struct pangulu_block_info_pool
{
    pangulu_block_info *data;
    pangulu_int64_t *block_map; // default : -1
    pangulu_int64_t capacity;
    pangulu_int64_t length;
    pangulu_int64_t index_upper_bound;
} pangulu_block_info_pool;

typedef struct pangulu_block_smatrix
{
    pangulu_exblock_ptr current_rank_block_count;

    // reorder array
    pangulu_exblock_idx *row_perm;
    pangulu_exblock_idx *col_perm;
    pangulu_exblock_idx *metis_perm;
    calculate_type *row_scale;
    calculate_type *col_scale;

    // symbolic
    pangulu_exblock_ptr symbolic_nnz;
    pangulu_exblock_ptr *symbolic_rowpointer;
    pangulu_exblock_idx *symbolic_columnindex;
    pangulu_exblock_ptr *symbolic_full_rowpointer;
    pangulu_exblock_idx *symbolic_full_columnindex;

    pangulu_block_info_pool *BIP;
    pangulu_inblock_ptr *block_smatrix_nnza_num;
    pangulu_inblock_ptr *block_smatrix_non_zero_vector_l;
    pangulu_inblock_ptr *block_smatrix_non_zero_vector_u;
    pangulu_smatrix *big_pangulu_smatrix_value;
    pangulu_inblock_ptr *l_pangulu_smatrix_columnpointer;
    pangulu_inblock_idx *l_pangulu_smatrix_rowindex;
    pangulu_exblock_ptr l_smatrix_nzz;
    pangulu_smatrix **l_pangulu_smatrix_value;
    pangulu_exblock_ptr *u_pangulu_smatrix_rowpointer;
    pangulu_exblock_idx *u_pangulu_smatrix_columnindex;
    pangulu_exblock_ptr u_smatrix_nzz;
    pangulu_smatrix **u_pangulu_smatrix_value;
    pangulu_int64_t *mapper_diagonal;
    pangulu_smatrix **diagonal_smatrix_l;
    pangulu_smatrix **diagonal_smatrix_u;
    pangulu_smatrix *calculate_l;
    pangulu_smatrix *calculate_u;
    pangulu_smatrix *calculate_x;

    pangulu_int64_t task_level_length;
    pangulu_int64_t *task_level_num;
    pangulu_heap *heap;
    pangulu_int64_t *now_level_l_length;
    pangulu_int64_t *now_level_u_length;
    pangulu_int64_t *save_now_level_l;
    pangulu_int64_t *save_now_level_u;
    pangulu_int64_t *send_flag;
    pangulu_int64_t *send_diagonal_flag_l;
    pangulu_int64_t *send_diagonal_flag_u;
    pangulu_int32_t *grid_process_id;
    pangulu_int64_t *save_send_rank_flag;
    pangulu_int64_t *receive_level_num;
    char *save_tmp;

    pangulu_int64_t *level_index;
    pangulu_int64_t *level_index_reverse;
    pangulu_int64_t *mapper_mpi;
    pangulu_int64_t *mapper_mpi_reverse;
    pangulu_int64_t *mpi_level_num;

    char *flag_save_l;
    char *flag_save_u;
    char *flag_dignon_l;
    char *flag_dignon_u;

#ifdef OVERLAP
    bsem *run_bsem1;
    bsem *run_bsem2;

#endif

    // sptrsv
    pangulu_vector **big_row_vector;
    pangulu_vector **big_col_vector;
    char *diagonal_flag;
    pangulu_exblock_ptr *l_row_task_nnz;
    pangulu_exblock_ptr *l_col_task_nnz;
    pangulu_exblock_ptr *u_row_task_nnz;
    pangulu_exblock_ptr *u_col_task_nnz;
    pangulu_heap *sptrsv_heap;
    pangulu_vector *save_vector;
    char *l_send_flag;
    char *u_send_flag;
    pangulu_exblock_ptr *l_sptrsv_task_columnpointer;
    pangulu_exblock_idx *l_sptrsv_task_rowindex;
    pangulu_exblock_ptr *u_sptrsv_task_columnpointer;
    pangulu_exblock_idx *u_sptrsv_task_rowindex;

} pangulu_block_smatrix;

typedef struct pangulu_block_common
{
    pangulu_exblock_idx n;
    pangulu_int32_t rank;
    pangulu_int32_t p;
    pangulu_int32_t q;
    pangulu_inblock_idx nb;
    pangulu_exblock_idx block_length;
    pangulu_int32_t rank_row_length;
    pangulu_int32_t rank_col_length;
    pangulu_int32_t sum_rank_size;
    pangulu_int32_t max_pq;
    pangulu_int32_t every_level_length;

} pangulu_block_common;

#ifdef OVERLAP

typedef struct thread_param
{
    pangulu_block_common *common;
    pangulu_block_smatrix *smatrix;
} thread_param;

#endif

typedef struct pangulu_handle_t
{
    pangulu_block_common *block_common;
    pangulu_block_smatrix *block_smatrix;
    pangulu_common *commmon;
} pangulu_handle_t;

typedef struct node
{
    pangulu_int64_t value;
    struct node *next;
} node;

#ifndef PANGULU_PLATFORM_ENV
void pangulu_multiply_upper_upper_u(pangulu_block_common *block_common,
                                    pangulu_block_smatrix *block_smatrix,
                                    pangulu_vector *x, pangulu_vector *b);

void pangulu_multiply_triggle_l(pangulu_block_common *block_common,
                                pangulu_block_smatrix *block_smatrix,
                                pangulu_vector *x, pangulu_vector *b);

void pangulu_gather_pangulu_vector_to_rank_0(pangulu_int64_t rank,
                                             pangulu_vector *gather_v,
                                             pangulu_int64_t vector_length,
                                             pangulu_int64_t sum_rank_size);

void pangulu_check_answer(pangulu_vector *X1, pangulu_vector *X2, pangulu_int64_t n);

calculate_type vec2norm(const calculate_type *x, pangulu_int64_t n);

calculate_type sub_vec2norm(const calculate_type *x1, const calculate_type *x2, pangulu_int64_t n);

void pangulu_check_answer_vec2norm(pangulu_vector *X1, pangulu_vector *X2, pangulu_int64_t n);

void pangulu_check(pangulu_block_common *block_common,
                   pangulu_block_smatrix *block_smatrix,
                   pangulu_origin_smatrix *origin_smatrix);

long double max_check_ld(long double *x, int n);

void spmv_ld(int n, const pangulu_int64_t *row_ptr, const pangulu_int32_t *col_idx, const long double *val, const long double *x, long double *y);

void check_correctness_ld(int n, pangulu_int64_t *row_ptr, pangulu_int32_t *col_idx, long double *val, long double *x, long double *b);

void pangulu_add_pangulu_smatrix_cuda(pangulu_smatrix *a,
                                      pangulu_smatrix *b);

void pangulu_add_pangulu_smatrix_cpu(pangulu_smatrix *a,
                                     pangulu_smatrix *b);

void pangulu_add_pangulu_smatrix_csr_to_csc(pangulu_smatrix *a);

#ifdef GPU_OPEN
void pangulu_cuda_device_init(pangulu_int32_t rank);

void pangulu_cuda_device_init_thread(pangulu_int32_t rank);

void pangulu_cuda_free_interface(void *cuda_address);

void pangulu_smatrix_add_cuda_memory(pangulu_smatrix *s);

void pangulu_smatrix_cuda_memory_init(pangulu_smatrix *s, pangulu_int64_t nb, pangulu_int64_t nnz);

void pangulu_smatrix_add_cuda_memory_u(pangulu_smatrix *u);

void pangulu_smatrix_cuda_memcpy_a(pangulu_smatrix *s);

void pangulu_smatrix_cuda_memcpy_struct_csr(pangulu_smatrix *calculate_S, pangulu_smatrix *s);

void pangulu_smatrix_cuda_memcpy_struct_csc(pangulu_smatrix *calculate_S, pangulu_smatrix *s);

void pangulu_smatrix_cuda_memcpy_complete_csr(pangulu_smatrix *calculate_S, pangulu_smatrix *s);

void pangulu_smatrix_cuda_memcpy_nnzu(pangulu_smatrix *calculate_U, pangulu_smatrix *u);

void pangulu_smatrix_cuda_memcpy_value_csr(pangulu_smatrix *s, pangulu_smatrix *calculate_S);

void pangulu_smatrix_cuda_memcpy_value_csr_async(pangulu_smatrix *s, pangulu_smatrix *calculate_S, cudaStream_t *stream);

void pangulu_smatrix_cuda_memcpy_value_csc(pangulu_smatrix *s, pangulu_smatrix *calculate_S);

void pangulu_smatrix_cuda_memcpy_value_csc_async(pangulu_smatrix *s, pangulu_smatrix *calculate_S, cudaStream_t *stream);

void pangulu_smatrix_cuda_memcpy_value_csc_cal_length(pangulu_smatrix *s, pangulu_smatrix *calculate_S);

void pangulu_smatrix_cuda_memcpy_to_device_value_csc_async(pangulu_smatrix *calculate_S, pangulu_smatrix *s, cudaStream_t *stream);

void pangulu_smatrix_cuda_memcpy_to_device_value_csc(pangulu_smatrix *calculate_S, pangulu_smatrix *s);

void pangulu_smatrix_cuda_memcpy_complete_csr_async(pangulu_smatrix *calculate_S, pangulu_smatrix *s, cudaStream_t *stream);

void pangulu_smatrix_cuda_memcpy_complete_csc_async(pangulu_smatrix *calculate_S, pangulu_smatrix *s, cudaStream_t *stream);

void pangulu_smatrix_cuda_memcpy_complete_csc(pangulu_smatrix *calculate_S, pangulu_smatrix *s);

void pangulu_gessm_fp64_cuda_v9(pangulu_smatrix *a,
                                pangulu_smatrix *l,
                                pangulu_smatrix *x);

void pangulu_gessm_fp64_cuda_v11(pangulu_smatrix *a,
                                 pangulu_smatrix *l,
                                 pangulu_smatrix *x);

void pangulu_gessm_fp64_cuda_v7(pangulu_smatrix *a,
                                pangulu_smatrix *l,
                                pangulu_smatrix *x);

void pangulu_gessm_fp64_cuda_v8(pangulu_smatrix *a,
                                pangulu_smatrix *l,
                                pangulu_smatrix *x);

void pangulu_gessm_fp64_cuda_v10(pangulu_smatrix *a,
                                 pangulu_smatrix *l,
                                 pangulu_smatrix *x);

void pangulu_gessm_interface_g_v1(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *x);

void pangulu_gessm_interface_G_V2(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *x);

void pangulu_gessm_interface_G_V3(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *x);

void pangulu_getrf_fp64_cuda(pangulu_smatrix *a,
                             pangulu_smatrix *l,
                             pangulu_smatrix *u);

void pangulu_getrf_interface_G_V1(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *u);

void pangulu_getrf_interface_G_V2(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *u);

void pangulu_tstrf_fp64_cuda_v8(pangulu_smatrix *a,
                                pangulu_smatrix *x,
                                pangulu_smatrix *u);

void pangulu_tstrf_fp64_cuda_v9(pangulu_smatrix *a,
                                pangulu_smatrix *x,
                                pangulu_smatrix *u);

void pangulu_tstrf_fp64_cuda_v7(pangulu_smatrix *a,
                                pangulu_smatrix *x,
                                pangulu_smatrix *u);

void pangulu_tstrf_fp64_cuda_v10(pangulu_smatrix *a,
                                 pangulu_smatrix *x,
                                 pangulu_smatrix *u);

void pangulu_tstrf_fp64_cuda_v11(pangulu_smatrix *a,
                                 pangulu_smatrix *x,
                                 pangulu_smatrix *u);

void pangulu_tstrf_interface_G_V1(pangulu_smatrix *a,
                                  pangulu_smatrix *x,
                                  pangulu_smatrix *u);

void pangulu_tstrf_interface_G_V2(pangulu_smatrix *a,
                                  pangulu_smatrix *x,
                                  pangulu_smatrix *u);

void pangulu_tstrf_interface_G_V3(pangulu_smatrix *a,
                                  pangulu_smatrix *x,
                                  pangulu_smatrix *u);

void pangulu_ssssm_fp64_cuda(pangulu_smatrix *a,
                             pangulu_smatrix *l,
                             pangulu_smatrix *u);

void pangulu_ssssm_interface_G_V1(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *u);

void pangulu_ssssm_interface_G_V2(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *u);

#endif // GPU_OPEN

#ifdef CHECK_TIME

void pangulu_time_check_begin(struct timeval *GET_TIME_START);

double pangulu_time_check_end(struct timeval *GET_TIME_START);

void pangulu_time_init();

void pangulu_time_simple_output(pangulu_int64_t rank);

#endif // CHECK_TIME

pangulu_vector *pangulu_destroy_pangulu_vector(pangulu_vector *V);

pangulu_smatrix *pangulu_destroy_part_pangulu_origin_smatrix(pangulu_origin_smatrix *s);

pangulu_smatrix *pangulu_destroy_pangulu_smatrix(pangulu_smatrix *s);

pangulu_smatrix *pangulu_destroy_copy_pangulu_smatrix(pangulu_smatrix *s);

pangulu_smatrix *pangulu_destroy_big_pangulu_smatrix(pangulu_smatrix *s);

pangulu_smatrix *pangulu_destroy_calculate_pangulu_smatrix_X(pangulu_smatrix *s);

pangulu_common *pangulu_destroy_pangulu_common(pangulu_common *common);

#ifdef GPU_OPEN

void pangulu_destroy_cuda_memory_pangulu_smatrix(pangulu_smatrix *s);

#else // GPU_OPEN

void pangulu_destroy_smatrix_level(pangulu_smatrix *a);

#endif // GPU_OPEN

void pangulu_destroy(pangulu_block_common *block_common,
                     pangulu_block_smatrix *block_smatrix);

int findlevel(const pangulu_inblock_ptr *cscColPtr,
              const pangulu_inblock_idx *cscRowIdx,
              const pangulu_inblock_ptr *csrRowPtr,
              const pangulu_int64_t m,
              int *nlevel,
              int *levelPtr,
              int *levelItem);

void pangulu_gessm_fp64_cpu_1(pangulu_smatrix *a,
                              pangulu_smatrix *l,
                              pangulu_smatrix *x);

void pangulu_gessm_fp64_cpu_2(pangulu_smatrix *a,
                              pangulu_smatrix *l,
                              pangulu_smatrix *x);

void pangulu_gessm_fp64_cpu_3(pangulu_smatrix *a,
                              pangulu_smatrix *l,
                              pangulu_smatrix *x);

void pangulu_gessm_fp64_cpu_4(pangulu_smatrix *a,
                              pangulu_smatrix *l,
                              pangulu_smatrix *x);

void pangulu_gessm_fp64_cpu_5(pangulu_smatrix *a,
                              pangulu_smatrix *l,
                              pangulu_smatrix *x);

void pangulu_gessm_fp64_cpu_6(pangulu_smatrix *a,
                              pangulu_smatrix *l,
                              pangulu_smatrix *x);

void pangulu_gessm_interface_cpu_csc(pangulu_smatrix *a,
                                     pangulu_smatrix *l,
                                     pangulu_smatrix *x);

void pangulu_gessm_interface_cpu_csr(pangulu_smatrix *a,
                                     pangulu_smatrix *l,
                                     pangulu_smatrix *x);

void pangulu_gessm_interface_c_v1(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *x);

void pangulu_gessm_interface_c_v2(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *x);

pangulu_int64_t binary_search(pangulu_int64_t begin, pangulu_int64_t end, pangulu_int64_t aim, pangulu_inblock_idx *array);

void pangulu_sflu_fp64(pangulu_smatrix *a,
                       pangulu_smatrix *l,
                       pangulu_smatrix *u);

void pangulu_sflu_fp64_dense_col(pangulu_smatrix *a,
                                 pangulu_smatrix *l,
                                 pangulu_smatrix *u);

double wtc_get_time(struct timeval time_start, struct timeval time_end);

void pangulu_sflu_fp64_dense_row_purge(pangulu_smatrix *a,
                                       pangulu_smatrix *l,
                                       pangulu_smatrix *u);

void pangulu_sflu_fp64_2(pangulu_smatrix *a,
                         pangulu_smatrix *l,
                         pangulu_smatrix *u);

#ifndef GPU_OPEN
void pangulu_sflu_omp_fp64(pangulu_smatrix *a,
                           pangulu_smatrix *l,
                           pangulu_smatrix *u);
#endif

void pangulu_sflu_fp64_dense(pangulu_smatrix *a,
                             pangulu_smatrix *l,
                             pangulu_smatrix *u);

void pangulu_getrf_fp64(pangulu_smatrix *a,
                        pangulu_smatrix *l,
                        pangulu_smatrix *u);

void pangulu_getrf_interface_c_v1(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *u);

void pangulu_init_heap_select(pangulu_int64_t select);

void pangulu_init_pangulu_heap(pangulu_heap *heap, pangulu_int64_t max_length);

pangulu_heap *pangulu_destory_pangulu_heap(pangulu_heap *heap);

void pangulu_zero_pangulu_heap(pangulu_heap *heap);

pangulu_int64_t pangulu_compare(compare_struct *compare_queue, pangulu_int64_t a, pangulu_int64_t b);

void pangulu_swap(pangulu_int64_t *heap_queue, pangulu_int64_t a, pangulu_int64_t b);

void pangulu_heap_insert(pangulu_heap *heap, pangulu_int64_t row, pangulu_int64_t col, pangulu_int64_t task_level, pangulu_int64_t kernel_id, pangulu_int64_t compare_flag);

pangulu_int64_t heap_empty(pangulu_heap *heap);

void pangulu_heap_adjust(pangulu_heap *heap, pangulu_int64_t top, pangulu_int64_t n);

pangulu_int64_t pangulu_heap_delete(pangulu_heap *heap);

void pangulu_display_heap(pangulu_heap *heap);

void pangulu_getrf_interface(pangulu_smatrix *a, pangulu_smatrix *l, pangulu_smatrix *u,
                             pangulu_smatrix *calculate_L, pangulu_smatrix *calculate_U);

void pangulu_tstrf_interface(pangulu_smatrix *a, pangulu_smatrix *save_X, pangulu_smatrix *u,
                             pangulu_smatrix *calculate_X, pangulu_smatrix *calculate_U);

void pangulu_gessm_interface(pangulu_smatrix *a, pangulu_smatrix *save_X, pangulu_smatrix *l,
                             pangulu_smatrix *calculate_X, pangulu_smatrix *calculate_L);

void pangulu_ssssm_interface(pangulu_smatrix *a, pangulu_smatrix *l, pangulu_smatrix *u,
                             pangulu_smatrix *calculate_L, pangulu_smatrix *calculate_U);

#ifdef GPU_OPEN
void pangulu_addmatrix_interface(pangulu_smatrix *a,
                                 pangulu_smatrix *b);
#endif

void pangulu_addmatrix_interface_cpu(pangulu_smatrix *a,
                                     pangulu_smatrix *b);

void pangulu_spmv(pangulu_smatrix *s, pangulu_vector *z, pangulu_vector *answer, int vector_number);

void pangulu_sptrsv(pangulu_smatrix *s, pangulu_vector *answer, pangulu_vector *z, int vector_number, int32_t tag);

void pangulu_vector_add(pangulu_vector *answer, pangulu_vector *z);

void pangulu_vector_sub(pangulu_vector *answer, pangulu_vector *z);

void pangulu_vector_copy(pangulu_vector *answer, pangulu_vector *z);

void pangulu_spmv_cpu_choumi(pangulu_smatrix *s, pangulu_vector *x, pangulu_vector *b);

void pangulu_spmv_cpu_xishu(pangulu_smatrix *s, pangulu_vector *x, pangulu_vector *b, pangulu_int64_t vector_number);

void pangulu_spmv_cpu_xishu_csc(pangulu_smatrix *s, pangulu_vector *x, pangulu_vector *b, pangulu_int64_t vector_number);

void pangulu_vector_add_cpu(pangulu_vector *b, pangulu_vector *x);

void pangulu_vector_sub_cpu(pangulu_vector *b, pangulu_vector *x);

void pangulu_vector_copy_cpu(pangulu_vector *b, pangulu_vector *x);

void pangulu_sptrsv_cpu_choumi(pangulu_smatrix *s, pangulu_vector *x, pangulu_vector *b);

void pangulu_sptrsv_cpu_xishu(pangulu_smatrix *s, pangulu_vector *x, pangulu_vector *b, pangulu_int64_t vector_number);

void pangulu_sptrsv_cpu_xishu_csc(pangulu_smatrix *s, pangulu_vector *x, pangulu_vector *b, pangulu_int64_t vector_number, pangulu_int64_t tag);

int cmp_int32t_asc(const void *a, const void *b);

void pangulu_preprocessing(pangulu_block_common *block_common,
                           pangulu_block_smatrix *block_smatrix,
                           pangulu_origin_smatrix *reorder_matrix,
                           pangulu_int64_t nthread);

void *pangulu_malloc(const char *file, pangulu_int64_t line, pangulu_int64_t size);

void *pangulu_realloc(const char *file, pangulu_int64_t line, void *oldptr, pangulu_int64_t size);

void pangulu_free(const char *file, pangulu_int64_t line, void *ptr);

pangulu_int64_t pangulu_bin_map(pangulu_int64_t nnz);

void pangulu_get_pangulu_smatrix_to_u(pangulu_smatrix *s,
                                      pangulu_smatrix *u,
                                      pangulu_int64_t nb);

void pangulu_get_pangulu_smatrix_to_l(pangulu_smatrix *s,
                                      pangulu_smatrix *l,
                                      pangulu_int64_t nb);

void pangulu_smatrix_add_more_memory(pangulu_smatrix *a);

void pangulu_smatrix_add_more_memory_csr(pangulu_smatrix *a);

void pangulu_smatrix_add_csc(pangulu_smatrix *a);

void pangulu_origin_smatrix_add_csc(pangulu_origin_smatrix *a);

void pangulu_malloc_pangulu_smatrix_csc(pangulu_smatrix *s,
                                        pangulu_int64_t nb, pangulu_int64_t *save_columnpointer);

void pangulu_malloc_pangulu_smatrix_csc_value(pangulu_smatrix *s,
                                              pangulu_smatrix *b);

void pangulu_malloc_pangulu_smatrix_csr_value(pangulu_smatrix *s,
                                              pangulu_smatrix *b);

void pangulu_malloc_pangulu_smatrix_nnz_csc(pangulu_smatrix *s,
                                            pangulu_int64_t nb, pangulu_int64_t nnz);

void pangulu_malloc_pangulu_smatrix_nnz_csr(pangulu_smatrix *s,
                                            pangulu_int64_t nb, pangulu_int64_t nnz);

void pangulu_malloc_pangulu_smatrix_value_csc(pangulu_smatrix *s, pangulu_int64_t nnz);

void pangulu_malloc_pangulu_smatrix_value_csr(pangulu_smatrix *s, pangulu_int64_t nnz);

void pangulu_smatrix_add_memory_u(pangulu_smatrix *u);

#ifndef GPU_OPEN

void pangulu_malloc_smatrix_level(pangulu_smatrix *a);

#endif

#ifdef PANGULU_MC64
void pangulu_mc64dd(pangulu_int64_t col, pangulu_int64_t n, pangulu_int64_t *queue, calculate_type *row_scale_value, pangulu_int64_t *save_tmp);

void pangulu_mc64ed(pangulu_int64_t *queue_length, pangulu_int64_t n, pangulu_int64_t *queue, calculate_type *row_scale_value, pangulu_int64_t *save_tmp);

void pangulu_mc64fd(pangulu_int64_t loc_origin, pangulu_int64_t *queue_length, pangulu_int64_t n, pangulu_int64_t *queue, calculate_type *row_scale_value, pangulu_int64_t *save_tmp);

void pangulu_mc64(pangulu_origin_smatrix *s, pangulu_exblock_idx **perm, pangulu_exblock_idx **iperm,
                  calculate_type **row_scale, calculate_type **col_scale);
#endif

#ifdef METIS

void pangulu_get_graph_struct(pangulu_origin_smatrix *s, idx_t **xadj_adress, idx_t **adjincy_adress);

void pangulu_metis(pangulu_origin_smatrix *a, idx_t **metis_perm);

#endif

void pangulu_reorder_vector_x_tran(pangulu_block_smatrix *block_smatrix,
                                   pangulu_vector *X_origin,
                                   pangulu_vector *X_trans);

void pangulu_reorder_vector_b_tran(pangulu_block_smatrix *block_smatrix,
                                   pangulu_vector *B_origin,
                                   pangulu_vector *B_trans);

void pangulu_reorder(pangulu_block_smatrix *block_smatrix,
                     pangulu_origin_smatrix *origin_matrix,
                     pangulu_origin_smatrix *reorder_matrix);

void pangulu_probe_message(MPI_Status *status);

pangulu_int64_t pangulu_bcast_n(pangulu_int64_t n, pangulu_int64_t send_rank);

void pangulu_bcast_vector(pangulu_inblock_ptr *vector, pangulu_int32_t length, pangulu_int64_t send_rank);

void pangulu_bcast_vector_int64(pangulu_int64_t *vector, pangulu_int32_t length, pangulu_int64_t send_rank);

void pangulu_mpi_waitall(MPI_Request *Request, int num);

void pangulu_isend_vector_char_wait(char *a, pangulu_int64_t n, pangulu_int64_t send_id, int signal, MPI_Request *req);

void pangulu_send_vector_int(pangulu_int64_t *a, pangulu_int64_t n, pangulu_int64_t send_id, int signal);

void pangulu_recv_vector_int(pangulu_int64_t *a, pangulu_int64_t n, pangulu_int64_t receive_id, int signal);

void pangulu_send_vector_char(char *a, pangulu_int64_t n, pangulu_int64_t send_id, int signal);

void pangulu_recv_vector_char(char *a, pangulu_int64_t n, pangulu_int64_t receive_id, int signal);

void pangulu_send_vector_value(calculate_type *a, pangulu_int64_t n, pangulu_int64_t send_id, int signal);

void pangulu_recv_vector_value(calculate_type *a, pangulu_int64_t n, pangulu_int64_t receive_id, int signal);

void pangulu_send_pangulu_smatrix_value_csr(pangulu_smatrix *s,
                                            pangulu_int64_t send_id, int signal, pangulu_int64_t nb);

void pangulu_send_pangulu_smatrix_struct_csr(pangulu_smatrix *s,
                                             pangulu_int64_t send_id, int signal, pangulu_int64_t nb);

void pangulu_send_pangulu_smatrix_complete_csr(pangulu_smatrix *s,
                                               pangulu_int64_t send_id, int signal, pangulu_int64_t nb);

void pangulu_recv_pangulu_smatrix_struct_csr(pangulu_smatrix *s,
                                             pangulu_int64_t receive_id, int signal, pangulu_int64_t nb);

void pangulu_recv_pangulu_smatrix_value_csr(pangulu_smatrix *s,
                                            pangulu_int64_t receive_id, int signal, pangulu_int64_t nb);

void pangulu_recv_pangulu_smatrix_value_csr_in_signal(pangulu_smatrix *s,
                                                      pangulu_int64_t receive_id, int signal, pangulu_int64_t nb);

void pangulu_recv_pangulu_smatrix_complete_csr(pangulu_smatrix *s,
                                               pangulu_int64_t receive_id, int signal, pangulu_int64_t nb);

void pangulu_recv_whole_pangulu_smatrix_csr(pangulu_smatrix *s,
                                            pangulu_int64_t receive_id, int signal, pangulu_int64_t nnz, pangulu_int64_t nb);

void pangulu_send_pangulu_smatrix_value_csc(pangulu_smatrix *s,
                                            pangulu_int64_t send_id, int signal, pangulu_int64_t nb);

void pangulu_send_pangulu_smatrix_struct_csc(pangulu_smatrix *s,
                                             pangulu_int64_t send_id, int signal, pangulu_int64_t nb);

void pangulu_send_pangulu_smatrix_complete_csc(pangulu_smatrix *s,
                                               pangulu_int64_t send_id, int signal, pangulu_int64_t nb);

void pangulu_recv_pangulu_smatrix_struct_csc(pangulu_smatrix *s,
                                             pangulu_int64_t receive_id, int signal, pangulu_int64_t nb);

void pangulu_recv_pangulu_smatrix_value_csc(pangulu_smatrix *s,
                                            pangulu_int64_t receive_id, int signal, pangulu_int64_t nb);

void pangulu_recv_pangulu_smatrix_value_csc_in_signal(pangulu_smatrix *s,
                                                      pangulu_int64_t receive_id, int signal, pangulu_int64_t nb);

void pangulu_recv_pangulu_smatrix_complete_csc(pangulu_smatrix *s,
                                               pangulu_int64_t receive_id, int signal, pangulu_int64_t nb);

void pangulu_recv_whole_pangulu_smatrix_csc(pangulu_smatrix *s,
                                            pangulu_int64_t receive_id, int signal, pangulu_int64_t nnz, pangulu_int64_t nb);

int pangulu_iprobe_message(MPI_Status *status);

void pangulu_isend_pangulu_smatrix_value_csr(pangulu_smatrix *s,
                                             pangulu_int64_t send_id, int signal, pangulu_int64_t nb);

void pangulu_isend_pangulu_smatrix_struct_csr(pangulu_smatrix *s,
                                              pangulu_int64_t send_id, int signal, pangulu_int64_t nb);

void pangulu_isend_pangulu_smatrix_complete_csr(pangulu_smatrix *s,
                                                pangulu_int64_t send_id, int signal, pangulu_int64_t nb);

void pangulu_isend_whole_pangulu_smatrix_csr(pangulu_smatrix *s,
                                             pangulu_int64_t receive_id, int signal, pangulu_int64_t nb);

void pangulu_isend_pangulu_smatrix_value_csc(pangulu_smatrix *s,
                                             pangulu_int64_t send_id, int signal, pangulu_int64_t nb);

void pangulu_isend_pangulu_smatrix_value_csc_in_signal(pangulu_smatrix *s,
                                                       pangulu_int64_t send_id, int signal, pangulu_int64_t nb);

void pangulu_isend_pangulu_smatrix_struct_csc(pangulu_smatrix *s,
                                              pangulu_int64_t send_id, int signal, pangulu_int64_t nb);

void pangulu_isend_pangulu_smatrix_complete_csc(pangulu_smatrix *s,
                                                pangulu_int64_t send_id, int signal, pangulu_int64_t nb);

void pangulu_isend_whole_pangulu_smatrix_csc(pangulu_smatrix *s,
                                             pangulu_int64_t receive_id, int signal, pangulu_int64_t nb);

void pangulu_solve_a_to_lu(pangulu_int64_t level, pangulu_int64_t row, pangulu_int64_t col,
                           pangulu_block_common *block_common,
                           pangulu_block_smatrix *block_smatrix,
                           pangulu_smatrix *calculate_L,
                           pangulu_smatrix *calculate_U);

void pangulu_solve_xu_a(pangulu_int64_t level, pangulu_int64_t now_level, pangulu_int64_t row, pangulu_int64_t col,
                        pangulu_block_common *block_common,
                        pangulu_block_smatrix *block_smatrix,
                        pangulu_smatrix *calculate_U,
                        pangulu_smatrix *calculate_X);

void pangulu_solve_lx_a(pangulu_int64_t level, pangulu_int64_t now_level, pangulu_int64_t row, pangulu_int64_t col,
                        pangulu_block_common *block_common,
                        pangulu_block_smatrix *block_smatrix,
                        pangulu_smatrix *calculate_L,
                        pangulu_smatrix *calculate_X);

void pangulu_solve_a_lu(pangulu_int64_t level, pangulu_int64_t row, pangulu_int64_t col,
                        pangulu_block_common *block_common,
                        pangulu_block_smatrix *block_smatrix,
                        pangulu_smatrix *calculate_L,
                        pangulu_smatrix *calculate_U);

void pangulu_add_A_to_A_old(pangulu_int64_t level, pangulu_int64_t row, pangulu_int64_t col,
                            pangulu_block_common *block_common,
                            pangulu_block_smatrix *block_smatrix,
                            pangulu_smatrix *calculate_X);

void pangulu_numerical_work(compare_struct *flag,
                            pangulu_block_common *block_common,
                            pangulu_block_smatrix *block_smatrix,
                            pangulu_smatrix *calculate_L,
                            pangulu_smatrix *calculate_U,
                            pangulu_smatrix *calculate_X,
                            pangulu_int64_t now_level);

void pangulu_numerical_receive_message(MPI_Status status,
                                       pangulu_int64_t now_level,
                                       pangulu_block_common *block_common,
                                       pangulu_block_smatrix *block_smatrix);

#ifdef OVERLAP
void *thread_GPU_work(void *param);
void pangulu_create_pthread(pangulu_block_common *block_common,
                            pangulu_block_smatrix *block_smatrix);
#endif

void pangulu_numeric(pangulu_block_common *block_common,
                     pangulu_block_smatrix *block_smatrix);

void pangulu_sptrsv_preprocessing(pangulu_block_common *block_common,
                                  pangulu_block_smatrix *block_smatrix,
                                  pangulu_vector *vector);

void L_pangulu_sptrsv_receive_message(MPI_Status status,
                                      pangulu_block_common *block_common,
                                      pangulu_block_smatrix *block_smatrix);

void L_pangulu_sptrsv_work(compare_struct *flag,
                           pangulu_block_common *block_common,
                           pangulu_block_smatrix *block_smatrix,
                           pangulu_vector *save_vector);

void pangulu_sptrsv_L(pangulu_block_common *block_common,
                      pangulu_block_smatrix *block_smatrix);

void u_pangulu_sptrsv_receive_message(MPI_Status status,
                                      pangulu_block_common *block_common,
                                      pangulu_block_smatrix *block_smatrix);

void u_pangulu_sptrsv_work(compare_struct *flag,
                           pangulu_block_common *block_common,
                           pangulu_block_smatrix *block_smatrix,
                           pangulu_vector *save_vector);

void pangulu_sptrsv_U(pangulu_block_common *block_common,
                      pangulu_block_smatrix *block_smatrix);

void pangulu_sptrsv_vector_gather(pangulu_block_common *block_common,
                                  pangulu_block_smatrix *block_smatrix,
                                  pangulu_vector *vector);

pangulu_int64_t partition_key_val_pair(pangulu_int64_t *key, calculate_type *val, pangulu_int64_t length, pangulu_int64_t pivot_index);

void insert_sort_key_val_pair(pangulu_int64_t *key, calculate_type *val, pangulu_int64_t length);

void quick_sort_key_val_pair1(pangulu_int64_t *key, calculate_type *val, pangulu_int64_t length);

void segmented_sum(calculate_type *input, pangulu_int64_t *bit_flag, pangulu_int64_t length);

void thmkl_dcsrmultcsr(pangulu_inblock_idx *m, pangulu_inblock_idx *n, pangulu_inblock_idx *k,
                       calculate_type *a, pangulu_inblock_idx *ja, pangulu_inblock_ptr *ia,
                       calculate_type *b, pangulu_inblock_idx *jb, pangulu_inblock_ptr *ib,
                       calculate_type *c, pangulu_inblock_idx *jc, pangulu_inblock_ptr *ic);

const pangulu_inblock_idx *lower_bound(const pangulu_inblock_idx *begins, const pangulu_inblock_idx *ends, pangulu_inblock_idx key);

void spmm(
    pangulu_int64_t m, pangulu_int64_t k, pangulu_int64_t n,
    const pangulu_int64_t *A_csrOffsets,
    const pangulu_inblock_idx *A_columns,
    const calculate_type *A_values,
    const pangulu_int64_t *B_csrOffsets,
    const pangulu_inblock_idx *B_columns,
    const calculate_type *B_values,
    const pangulu_int64_t *C_csrOffsets,
    const pangulu_inblock_idx *C_columns,
    calculate_type *C_values);

int binary_search_right_boundary(const pangulu_int64_t *data,
                                 const pangulu_int64_t key_input,
                                 const int begin,
                                 const int end);

void cscmultcsc_dense(pangulu_smatrix *a,
                      pangulu_smatrix *l,
                      pangulu_smatrix *u);

void openblas_dgemm_reprecess(pangulu_smatrix *a,
                              pangulu_smatrix *l,
                              pangulu_smatrix *u);

void pangulu_ssssm_fp64(pangulu_smatrix *a,
                        pangulu_smatrix *l,
                        pangulu_smatrix *u);

void pangulu_ssssm_interface_c_v1(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *u);

void pangulu_ssssm_interface_c_v2(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *u);

void pangulu_tstrf_fp64_cpu_1(pangulu_smatrix *a,
                              pangulu_smatrix *x,
                              pangulu_smatrix *u);

void pangulu_tstrf_fp64_cpu_2(pangulu_smatrix *a,
                              pangulu_smatrix *x,
                              pangulu_smatrix *u);

void pangulu_tstrf_fp64_cpu_3(pangulu_smatrix *a,
                              pangulu_smatrix *x,
                              pangulu_smatrix *u);

void pangulu_tstrf_fp64_cpu_4(pangulu_smatrix *a,
                              pangulu_smatrix *x,
                              pangulu_smatrix *u);

void pangulu_tstrf_fp64_cpu_5(pangulu_smatrix *a,
                              pangulu_smatrix *x,
                              pangulu_smatrix *u);

void pangulu_tstrf_fp64_cpu_6(pangulu_smatrix *a,
                              pangulu_smatrix *x,
                              pangulu_smatrix *u);

void pangulu_tstrf_interface_cpu_csr(pangulu_smatrix *a,
                                     pangulu_smatrix *x,
                                     pangulu_smatrix *u);
void pangulu_tstrf_interface_cpu_csc(pangulu_smatrix *a,
                                     pangulu_smatrix *x,
                                     pangulu_smatrix *u);

void pangulu_tstrf_interface_c_v1(pangulu_smatrix *a,
                                  pangulu_smatrix *x,
                                  pangulu_smatrix *u);

void pangulu_tstrf_interface_c_v2(pangulu_smatrix *a,
                                  pangulu_smatrix *x,
                                  pangulu_smatrix *u);

void at_plus_a_dist(
    const pangulu_exblock_idx n,    /* number of columns in reorder_matrix a. */
    const pangulu_exblock_ptr nz,   /* number of nonzeros in reorder_matrix a */
    pangulu_exblock_ptr *colptr,    /* column pointer of size n+1 for reorder_matrix a. */
    pangulu_exblock_idx *rowind,    /* row indices of size nz for reorder_matrix a. */
    pangulu_exblock_ptr *bnz,       /* out - on exit, returns the actual number of nonzeros in reorder_matrix a'+a. */
    pangulu_exblock_ptr **b_colptr, /* out - size n+1 */
    pangulu_exblock_idx **b_rowind  /* out - size *bnz */
);

void add_prune(node *prune, node *prune_next, pangulu_int64_t num, pangulu_int64_t num_value, pangulu_int64_t p);

void fill_in_sym_prune(
    pangulu_exblock_idx n, pangulu_exblock_ptr nnz, pangulu_exblock_idx *ai, pangulu_exblock_ptr *ap,
    pangulu_exblock_ptr **symbolic_rowpointer, pangulu_exblock_idx **symbolic_columnindex,
    pangulu_inblock_idx nb, pangulu_exblock_idx block_length,
    pangulu_inblock_ptr *block_smatrix_non_zero_vector_L,
    pangulu_inblock_ptr *block_smatrix_non_zero_vector_U,
    pangulu_inblock_ptr *block_smatrix_nnzA_num,
    pangulu_exblock_ptr *symbolic_nnz
    // pangulu_block_info_pool* BIP
);

pangulu_int32_t pruneL(
    pangulu_exblock_idx jcol,
    pangulu_exblock_idx *U_r_idx,
    pangulu_exblock_ptr *U_c_ptr,
    pangulu_exblock_idx *L_r_idx,
    pangulu_exblock_ptr *L_c_ptr,
    pangulu_int64_t *work_space,
    pangulu_int64_t *prune_space);

void fill_in_2_no_sort_pruneL(
    pangulu_exblock_idx n,
    pangulu_exblock_ptr nnz,
    pangulu_exblock_idx *ai,
    pangulu_exblock_ptr *ap,
    pangulu_exblock_ptr **L_rowpointer,
    pangulu_exblock_idx **L_columnindex,
    pangulu_exblock_ptr **U_rowpointer,
    pangulu_exblock_idx **U_columnindex);

void pangulu_symbolic(pangulu_block_common *block_common,
                      pangulu_block_smatrix *block_smatrix,
                      pangulu_origin_smatrix *reorder_matrix);

void pangulu_mutex_init(pthread_mutex_t *mutex);

void pangulu_bsem_init(bsem *bsem_p, pangulu_int64_t value);

bsem *pangulu_bsem_destory(bsem *bsem_p);

void pangulu_bsem_post(pangulu_heap *heap);

pangulu_int64_t pangulu_bsem_wait(pangulu_heap *heap);

void pangulu_bsem_stop(pangulu_heap *heap);

void pangulu_bsem_synchronize(bsem *bsem_p);

void bind_to_core(int core);

void mpi_barrier_asym(MPI_Comm comm, int wake_rank, unsigned long long awake_interval_us);

double pangulu_fabs(double _Complex x);

double _Complex pangulu_log(double _Complex x);

double _Complex pangulu_sqrt(double _Complex x);

void exclusive_scan_1(pangulu_int64_t *input, int length);

void exclusive_scan_2(pangulu_int32_t *input, int length);

void exclusive_scan_3(unsigned int *input, int length);

void swap_key(pangulu_int64_t *a, pangulu_int64_t *b);

void swap_val(calculate_type *a, calculate_type *b);

int binarylowerbound(const pangulu_int64_t *arr, int len, pangulu_int64_t value);

pangulu_int64_t binarysearch(const int *arr, pangulu_int64_t left, pangulu_int64_t right, pangulu_int64_t target);

pangulu_int64_t binarysearch_inblock_idx(pangulu_int64_t begin, pangulu_int64_t end, pangulu_int64_t aim, pangulu_inblock_idx *array);

void pangulu_get_common(pangulu_common *common,
                        pangulu_init_options *init_options, pangulu_int32_t size);

void pangulu_init_pangulu_block_smatrix(pangulu_block_smatrix *block_smatrix);

void pangulu_init_pangulu_smatrix(pangulu_smatrix *s);

void pangulu_init_pangulu_origin_smatrix(pangulu_origin_smatrix *s);

void pangulu_read_pangulu_origin_smatrix(pangulu_origin_smatrix *s, int wcs_n, long long wcs_nnz, pangulu_exblock_ptr *csr_rowptr, pangulu_exblock_idx *csr_colidx, calculate_type *csr_value);

// void pangulu_time_start(pangulu_common *common);

// void pangulu_time_stop(pangulu_common *common);

void pangulu_time_start(struct timeval* start);

double pangulu_time_stop(struct timeval* start);

void pangulu_memcpy_zero_pangulu_smatrix_csc_value(pangulu_smatrix *s);

void pangulu_memcpy_zero_pangulu_smatrix_csr_value(pangulu_smatrix *s);

void pangulu_display_pangulu_smatrix_csc(pangulu_smatrix *s);

double pangulu_get_spend_time(pangulu_common *common);

void pangulu_transpose_pangulu_smatrix_csc_to_csr(pangulu_smatrix *s);

void pangulu_transpose_pangulu_smatrix_csr_to_csc(pangulu_smatrix *s);

void pangulu_pangulu_smatrix_memcpy_rowpointer_csr(pangulu_smatrix *s, pangulu_smatrix *copy_S);

void pangulu_pangulu_smatrix_memcpy_value_csr(pangulu_smatrix *s, pangulu_smatrix *copy_S);

void pangulu_pangulu_smatrix_memcpy_value_csr_copy_length(pangulu_smatrix *s, pangulu_smatrix *copy_S);

void pangulu_pangulu_smatrix_memcpy_value_csc_copy_length(pangulu_smatrix *s, pangulu_smatrix *copy_S);

void pangulu_pangulu_smatrix_memcpy_struct_csc(pangulu_smatrix *s, pangulu_smatrix *copy_S);

void pangulu_pangulu_smatrix_memcpy_columnpointer_csc(pangulu_smatrix *s, pangulu_smatrix *copy_S);

void pangulu_pangulu_smatrix_memcpy_value_csc(pangulu_smatrix *s, pangulu_smatrix *copy_S);

void pangulu_pangulu_smatrix_memcpy_complete_csc(pangulu_smatrix *s, pangulu_smatrix *copy_S);

void pangulu_pangulu_smatrix_multiple_pangulu_vector_csr(pangulu_smatrix *a,
                                                         pangulu_vector *x,
                                                         pangulu_vector *b);

void pangulu_origin_smatrix_multiple_pangulu_vector_csr(pangulu_origin_smatrix *a,
                                                        pangulu_vector *x,
                                                        pangulu_vector *b);

void pangulu_pangulu_smatrix_multiple_pangulu_vector(pangulu_smatrix *a,
                                                     pangulu_vector *x,
                                                     pangulu_vector *b);

void pangulu_pangulu_smatrix_multiply_block_pangulu_vector_csr(pangulu_smatrix *a,
                                                               calculate_type *x,
                                                               calculate_type *b);

void pangulu_pangulu_smatrix_multiple_pangulu_vector_csc(pangulu_smatrix *a,
                                                         pangulu_vector *x,
                                                         pangulu_vector *b);

void pangulu_pangulu_smatrix_multiply_block_pangulu_vector_csc(pangulu_smatrix *a,
                                                               calculate_type *x,
                                                               calculate_type *b);

void pangulu_get_init_value_pangulu_vector(pangulu_vector *x, pangulu_int64_t n);

void pangulu_init_pangulu_vector(pangulu_vector *b, pangulu_int64_t n);

void pangulu_zero_pangulu_vector(pangulu_vector *v);

void pangulu_add_diagonal_element(pangulu_origin_smatrix *s);

void pangulu_send_pangulu_vector_value(pangulu_vector *s,
                                       pangulu_int64_t send_id, pangulu_int64_t signal, pangulu_int64_t vector_length);

void pangulu_isend_pangulu_vector_value(pangulu_vector *s,
                                        int send_id, int signal, int vector_length);

void pangulu_recv_pangulu_vector_value(pangulu_vector *s, pangulu_int64_t receive_id, pangulu_int64_t signal, pangulu_int64_t vector_length);

void pangulu_init_vector_int(pangulu_int64_t *vector, pangulu_int64_t length);

pangulu_int64_t pangulu_choose_pivot(pangulu_int64_t i, pangulu_int64_t j);

void pangulu_swap_int(pangulu_int64_t *a, pangulu_int64_t *b);

void pangulu_quicksort_keyval(pangulu_int64_t *key, pangulu_int64_t *val, pangulu_int64_t start, pangulu_int64_t end);

double pangulu_standard_deviation(pangulu_int64_t *p, pangulu_int64_t num);

#ifndef GPU_OPEN
void pangulu_init_level_array(pangulu_smatrix *a, pangulu_int64_t *work_space);
#endif

pangulu_int64_t choose_pivot(pangulu_int64_t i, pangulu_int64_t j);

void swap_value(calculate_type *a, calculate_type *b);

void swap_index_1(pangulu_exblock_idx *a, pangulu_exblock_idx *b);

void swap_index_2(pangulu_inblock_idx *a, pangulu_inblock_idx *b);

void swap_index_3(int32_t *a, int32_t *b);

void pangulu_sort(pangulu_exblock_idx *key, calculate_type *val, pangulu_int64_t start, pangulu_int64_t end);

void pangulu_sort_struct_1(pangulu_exblock_idx *key, pangulu_exblock_ptr start, pangulu_exblock_ptr end);

void pangulu_sort_struct_2(pangulu_inblock_idx *key, pangulu_int64_t start, pangulu_int64_t end);

void pangulu_sort_pangulu_matrix(pangulu_int64_t n, pangulu_exblock_ptr *rowpointer, pangulu_exblock_idx *columnindex);

void pangulu_sort_pangulu_origin_smatrix(pangulu_origin_smatrix *s);

#ifdef GPU_OPEN

void triangle_pre_cpu(pangulu_inblock_idx *L_rowindex,
                      const pangulu_int64_t n,
                      const pangulu_int64_t nnzL,
                      int *d_graphindegree);

void pangulu_gessm_preprocess(pangulu_smatrix *l);

void pangulu_tstrf_preprocess(pangulu_smatrix *u);

#endif

int tstrf_csc_csc(
    pangulu_inblock_idx n,
    pangulu_inblock_ptr* U_colptr,
    pangulu_inblock_idx* U_rowidx,
    calculate_type* u_value,
    pangulu_inblock_ptr* A_colptr,
    pangulu_inblock_idx* A_rowidx,
    calculate_type* a_value
);

void pangulu_bip_init(pangulu_block_info_pool **BIP, pangulu_int64_t map_index_not_included);

void pangulu_bip_destroy(pangulu_block_info_pool **BIP);

const pangulu_block_info *pangulu_bip_get(pangulu_int64_t index, pangulu_block_info_pool *BIP);

pangulu_block_info *pangulu_bip_set(pangulu_int64_t index, pangulu_block_info_pool *BIP);

void pangulu_convert_csr_to_csc(
    int free_csrmatrix,
    pangulu_exblock_idx n,
    pangulu_exblock_ptr** csr_pointer,
    pangulu_exblock_idx** csr_index,
    calculate_type** csr_value,
    pangulu_exblock_ptr** csc_pointer,
    pangulu_exblock_idx** csc_index,
    calculate_type** csc_value
);

void pangulu_cm_distribute_csc_to_distcsc(
    pangulu_int32_t root_rank,
    int rootproc_free_originmatrix,
    pangulu_exblock_idx* n,
    pangulu_inblock_idx rowchunk_align,
    pangulu_int32_t* distcsc_nproc,
    pangulu_exblock_idx* n_loc,
    
    pangulu_exblock_ptr** distcsc_proc_nnzptr,
    pangulu_exblock_ptr** distcsc_pointer,
    pangulu_exblock_idx** distcsc_index,
    calculate_type** distcsc_value
);

void pangulu_cm_distribute_distcsc_to_distbcsc(
    int rootproc_free_originmatrix,
    int malloc_distbcsc_value,
    pangulu_exblock_idx n_glo,
    pangulu_exblock_idx n_loc,
    pangulu_inblock_idx block_order,
    
    pangulu_exblock_ptr* distcsc_proc_nnzptr,
    pangulu_exblock_ptr* distcsc_pointer,
    pangulu_exblock_idx* distcsc_index,
    calculate_type* distcsc_value,

    pangulu_exblock_ptr** bcsc_struct_pointer,
    pangulu_exblock_idx** bcsc_struct_index,
    pangulu_exblock_ptr** bcsc_struct_nnzptr,
    pangulu_inblock_ptr*** bcsc_inblock_pointers,
    pangulu_inblock_idx*** bcsc_inblock_indeces,
    calculate_type*** bcsc_values
);

void pangulu_convert_ordered_halfsymcsc_to_csc_struct(
    int free_halfmatrix,
    int if_colsort,
    pangulu_exblock_idx n,
    pangulu_exblock_ptr** half_csc_pointer,
    pangulu_exblock_idx** half_csc_index,
    pangulu_exblock_ptr** csc_pointer,
    pangulu_exblock_idx** csc_index
);

void pangulu_convert_bcsc_fill_value_to_struct(
    int free_valuebcsc,
    pangulu_exblock_idx n,
    pangulu_inblock_idx nb,

    pangulu_exblock_ptr* value_bcsc_struct_pointer,
    pangulu_exblock_idx* value_bcsc_struct_index,
    pangulu_exblock_ptr* value_bcsc_struct_nnzptr,
    pangulu_inblock_ptr** value_bcsc_inblock_pointers,
    pangulu_inblock_idx** value_bcsc_inblock_indeces,
    calculate_type** value_bcsc_values,

    pangulu_exblock_ptr* struct_bcsc_struct_pointer,
    pangulu_exblock_idx* struct_bcsc_struct_index,
    pangulu_exblock_ptr* struct_bcsc_struct_nnzptr,
    pangulu_inblock_ptr** struct_bcsc_inblock_pointers,
    pangulu_inblock_idx** struct_bcsc_inblock_indeces,
    calculate_type** struct_bcsc_values
);

void pangulu_cm_rank(pangulu_int32_t* rank);

void pangulu_cm_size(pangulu_int32_t* size);

void pangulu_cm_sync();

void pangulu_cm_bcast(void *buffer, int count, MPI_Datatype datatype, int root);

void pangulu_cm_isend(char* buf, pangulu_int64_t count, pangulu_int32_t remote_rank, pangulu_int32_t tag, pangulu_int32_t tag_ub);

void pangulu_cm_send(char* buf, pangulu_int64_t count, pangulu_int32_t remote_rank, pangulu_int32_t tag, pangulu_int32_t tag_ub);

void pangulu_cm_recv(char* buf, pangulu_int64_t count, pangulu_int32_t fetch_rank, pangulu_int32_t tag, pangulu_int32_t tag_ub);

void pangulu_sort_exblock_struct(
    pangulu_exblock_idx n,
    pangulu_exblock_ptr* pointer,
    pangulu_exblock_idx* index,
    pangulu_int32_t nthread
);

int cmp_exidx_asc(const void* a, const void* b);

void pangulu_kvsort(pangulu_exblock_idx *key, calculate_type *val, pangulu_int64_t start, pangulu_int64_t end);


void pangulu_cm_sync_asym(int wake_rank);

#endif // PANGULU_PLATFORM_ENV
#endif // PANGULU_COMMON_H