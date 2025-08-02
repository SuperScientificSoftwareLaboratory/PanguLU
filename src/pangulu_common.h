#ifndef PANGULU_COMMON_H
#define PANGULU_COMMON_H

#ifdef GPU_OPEN
#define PANGULU_DEFAULT_PLATFORM PANGULU_PLATFORM_GPU_CUDA
#define PANGULU_NONSHAREDMEM
#else
#define PANGULU_DEFAULT_PLATFORM PANGULU_PLATFORM_CPU_NAIVE
#endif

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
#define calculate_type double
#define calculate_real_type double
#define MPI_VAL_TYPE MPI_DOUBLE
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
typedef pangulu_uint16_t pangulu_inblock_idx;
#define MPI_PANGULU_INBLOCK_IDX MPI_PANGULU_UINT16_T
#define FMT_PANGULU_INBLOCK_IDX FMT_PANGULU_UINT16_T

typedef pangulu_exblock_ptr sparse_pointer_t;
typedef pangulu_exblock_idx sparse_index_t;
typedef calculate_type sparse_value_t;
typedef calculate_real_type sparse_value_real_t;

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#ifndef PANGULU_PLATFORM_ENV
#include <mpi.h>
#endif
#define __USE_GNU
#include <sched.h>
#include <pthread.h>
#ifndef PANGULU_PLATFORM_ENV
#include <cblas.h>
#endif
#include <getopt.h>
#include <omp.h>
#include <sys/resource.h>
#include <errno.h>
#include "pangulu_strings.h"

#ifndef PANGULU_PLATFORM_ENV
#ifdef METIS
#include <metis.h>
typedef idx_t reordering_int_t;
#else
#include <mynd_omp.h>
#endif
#endif

#define PANGULU_AGGR_QUEUE_MIN_CAP 4
#define PANGULU_AGGR_QUEUE_INCREASE_SPEED 2
#define PANGULU_ICEIL(a, b) (((a) + (b) - 1) / (b))
#define PANGULU_MIN(a, b) ((a) < (b) ? (a) : (b))
#define PANGULU_MAX(a, b) ((a) > (b) ? (a) : (b))
#define PANGULU_SETBIT(x, y) x |= (1 << y)    // set the yth bit of x is 1
#define PANGULU_GETBIT(x, y) ((x) >> (y) & 1) // get the yth bit of x
#define PANGULU_DIGINFO_OFFSET_STOREIDX (0)
#define PANGULU_DIGINFO_OFFSET_NNZ (39)
#define PANGULU_DIGINFO_OFFSET_BINID (61)
#define PANGULU_DIGINFO_MASK_STOREIDX (0x7FFFFFFFFF)
#define PANGULU_DIGINFO_MASK_NNZ (0x3FFFFF)
#define PANGULU_DIGINFO_MASK_BINID (0x7)
#define PANGULU_DIGINFO_SET_SLOT_IDX(x) (((pangulu_uint64_t)(x) & PANGULU_DIGINFO_MASK_STOREIDX) << PANGULU_DIGINFO_OFFSET_STOREIDX)
#define PANGULU_DIGINFO_SET_NNZ(x) (((pangulu_uint64_t)(x) & PANGULU_DIGINFO_MASK_NNZ) << PANGULU_DIGINFO_OFFSET_NNZ)
#define PANGULU_DIGINFO_SET_BINID(x) (((pangulu_uint64_t)(x) & PANGULU_DIGINFO_MASK_BINID) << PANGULU_DIGINFO_OFFSET_BINID)
#define PANGULU_DIGINFO_GET_SLOT_IDX(x) ((((pangulu_uint64_t)(x)) >> PANGULU_DIGINFO_OFFSET_STOREIDX) & PANGULU_DIGINFO_MASK_STOREIDX)
#define PANGULU_DIGINFO_GET_NNZ(x) ((((pangulu_uint64_t)(x)) >> PANGULU_DIGINFO_OFFSET_NNZ) & PANGULU_DIGINFO_MASK_NNZ)
#define PANGULU_DIGINFO_GET_BINID(x) ((((pangulu_uint64_t)(x)) >> PANGULU_DIGINFO_OFFSET_BINID) & PANGULU_DIGINFO_MASK_BINID)
#define PANGULU_TASK_GETRF 1
#define PANGULU_TASK_TSTRF 2
#define PANGULU_TASK_GESSM 3
#define PANGULU_TASK_SSSSM 4
#define PANGULU_LOWER 0
#define PANGULU_UPPER 1
#define PANGULU_DATA_INVALID 0
#define PANGULU_DATA_PREPARING 1
#define PANGULU_DATA_READY 2
#define PANGULU_TOL 1e-16
#define PANGULU_SPTRSV_TOL 1e-16
#define PANGULU_CALC_RANK(row, col, p, q) (((row) % (p)) * (q) + ((col) % (q)))
#define PANGULU_CALC_OFFSET(offset_init, now_level, PQ_length) \
    (((offset_init) - (now_level) % (PQ_length)) < 0) ? ((offset_init) - (now_level) % (PQ_length) + (PQ_length)) : ((offset_init) - (now_level) % (PQ_length))

typedef struct pangulu_stat_t
{
    double time_getrf;
    double time_tstrf;
    double time_gessm;
    double time_ssssm;

    double time_outer_kernel;
    double time_inner_kernel;
    double time_send;
    double time_recv;
    double time_wait;

    long long recv_cnt;
    long long kernel_cnt;
    long long flop;

    double time_sptodns;
    double time_dnstosp;

    FILE *trsm_log_file;
} pangulu_stat_t;
#ifdef PANGULU_PERF
extern pangulu_stat_t global_stat;
#endif

typedef struct pangulu_common
{
    pangulu_int32_t rank;
    pangulu_int32_t size;
    pangulu_exblock_idx n;
    pangulu_inblock_idx nb;
    pangulu_int32_t sum_rank_size;
    pangulu_int32_t omp_thread;
    pangulu_int32_t reordering_nthread;
    pangulu_int32_t p;
    pangulu_int32_t q;
    float basic_param;
} pangulu_common;

typedef struct pangulu_origin_smatrix
{
    pangulu_exblock_idx column;
    pangulu_exblock_idx row;
    pangulu_exblock_ptr nnz;
    pangulu_exblock_ptr *columnpointer;
    pangulu_exblock_idx *rowindex;
    calculate_type *value_csc;
    pangulu_exblock_ptr *rowpointer;
    pangulu_exblock_idx *columnindex;
    calculate_type *value;
    pangulu_exblock_ptr *csc_to_csr_index;
} pangulu_origin_smatrix;

typedef struct pangulu_bsem_t
{
    pthread_mutex_t *mutex;
    pthread_cond_t *cond;
    pangulu_int32_t v;
} pangulu_bsem_t;

typedef struct pangulu_aggregate_queue_t
{
    unsigned long long capacity;
    unsigned long long length;
    void *task_descriptors;
} pangulu_aggregate_queue_t;

typedef struct pangulu_storage_slot_t
{
    pangulu_exblock_idx brow_pos;
    pangulu_exblock_idx bcol_pos;
    pangulu_inblock_ptr *columnpointer;
    pangulu_inblock_idx *rowindex;
    calculate_type *value;
    pangulu_inblock_ptr *rowpointer;
    pangulu_inblock_idx *columnindex;
    pangulu_inblock_ptr *idx_of_csc_value_for_csr;
    volatile char data_status;
    struct pangulu_storage_slot_t *related_block;
    pangulu_int32_t is_upper;
    pangulu_int32_t bin_id;
    pangulu_int32_t slot_idx;
    pangulu_aggregate_queue_t *task_queue;
#ifdef PANGULU_NONSHAREDMEM
    pangulu_inblock_ptr *d_columnpointer;
    pangulu_inblock_idx *d_rowindex;
    calculate_type *d_value;
    pangulu_inblock_ptr *d_rowpointer;
    pangulu_inblock_idx *d_columnindex;
    pangulu_inblock_ptr *d_idx_of_csc_value_for_csr;
#endif
} pangulu_storage_slot_t;

typedef struct pangulu_task_t
{
    pangulu_exblock_idx row;
    pangulu_exblock_idx col;
    pangulu_int16_t kernel_id;
    pangulu_exblock_idx task_level;
    pangulu_int64_t compare_flag;
    pangulu_storage_slot_t *opdst;
    pangulu_storage_slot_t *op1;
    pangulu_storage_slot_t *op2;
} pangulu_task_t;

typedef struct pangulu_task_queue_t
{
    pangulu_int64_t length;
    pangulu_int64_t capacity;
    pangulu_int64_t *task_index_heap;
    pangulu_int64_t *task_storage_avail_queue;
    pangulu_int64_t task_storage_avail_queue_head;
    pangulu_int64_t task_storage_avail_queue_tail;
    pangulu_task_t *task_storage;
    pangulu_int32_t cmp_strategy;
    pangulu_bsem_t *heap_bsem;
} pangulu_task_queue_t;

typedef struct pangulu_storage_bin_t
{
    pangulu_storage_slot_t *slots;
    pangulu_int64_t slot_capacity;
    pangulu_int64_t slot_count;
    pangulu_int64_t nondiag_slot_cnt;
    pangulu_int32_t *avail_slot_queue;
    pangulu_int32_t queue_head;
    pangulu_int32_t queue_tail;
} pangulu_storage_bin_t;

typedef struct pangulu_storage_t
{
    pangulu_int32_t n_bin;
    pangulu_storage_bin_t *bins;
    pthread_mutex_t *mutex;
} pangulu_storage_t;

typedef struct pangulu_vector
{
    calculate_type *value;
    pangulu_int64_t row;
} pangulu_vector;

typedef struct pangulu_block_smatrix
{
    pangulu_exblock_idx *row_perm;
    pangulu_exblock_idx *col_perm;
    pangulu_exblock_idx *metis_perm;
    calculate_type *row_scale;
    calculate_type *col_scale;
    pangulu_exblock_ptr *symbolic_rowpointer;
    pangulu_exblock_idx *symbolic_columnindex;
    pangulu_bsem_t *run_bsem1;
    pangulu_task_queue_t *heap;
    pangulu_exblock_ptr symbolic_nnz;
    pangulu_storage_t *storage;
    pthread_mutex_t *info_mutex;
    char *sent_rank_flag;

    pangulu_int64_t aggregate_batch_tileid_capacity;
    pangulu_storage_slot_t **aggregate_batch_tileid;

    calculate_type *A_rowsum_reordered;

    pangulu_int64_t rank_remain_task_count;
    pangulu_int64_t rank_remain_recv_block_count;
    pangulu_int32_t *nondiag_remain_task_count;
    pangulu_int32_t *diag_remain_task_count;

    pangulu_uint64_t *diag_uniaddr;
    pangulu_inblock_ptr **diag_upper_rowptr;
    pangulu_inblock_idx **diag_upper_colidx;
    calculate_type **diag_upper_csrvalue;
    pangulu_inblock_ptr **diag_lower_colptr;
    pangulu_inblock_idx **diag_lower_rowidx;
    calculate_type **diag_lower_cscvalue;

    calculate_type **nondiag_cscvalue;
    pangulu_inblock_ptr **nondiag_colptr;
    pangulu_inblock_idx **nondiag_rowidx;
    pangulu_inblock_ptr **nondiag_csr_to_csc;
    pangulu_inblock_ptr **nondiag_rowptr;
    pangulu_inblock_idx **nondiag_colidx;

    pangulu_exblock_ptr *nondiag_block_colptr;
    pangulu_exblock_idx *nondiag_block_rowidx;
    pangulu_exblock_ptr *nondiag_block_rowptr;
    pangulu_exblock_idx *nondiag_block_colidx;
    pangulu_exblock_ptr *nondiag_block_csr_to_csc;

    pangulu_exblock_ptr *related_nondiag_block_colptr;
    pangulu_exblock_idx *related_nondiag_block_rowidx;
    pangulu_uint64_t *related_nondiag_uniaddr;
    pangulu_exblock_ptr *related_nondiag_block_rowptr;
    pangulu_exblock_idx *related_nondiag_block_colidx;
    pangulu_exblock_ptr *related_nondiag_block_csr_to_csc;
    pangulu_exblock_ptr *related_nondiag_fstblk_idx_after_diag;
    pangulu_exblock_ptr *related_nondiag_fstblk_idx_after_diag_csr;

    volatile pangulu_int32_t *rhs_remain_task_count;
    volatile pangulu_int32_t *rhs_remain_recv_count;
    calculate_type *rhs;
    calculate_type *spmv_acc;
    calculate_type *recv_buf;
} pangulu_block_smatrix;

typedef struct pangulu_block_common
{
    pangulu_int32_t rank;
    pangulu_int32_t p;
    pangulu_int32_t q;
    pangulu_inblock_idx nb;
    pangulu_exblock_idx n;
    pangulu_exblock_idx block_length;
    pangulu_exblock_idx first_dense_level;
    pangulu_int32_t sum_rank_size;

    pangulu_int32_t rank_row_length;
    pangulu_int32_t rank_col_length;
} pangulu_block_common;

typedef struct pangulu_numeric_thread_param
{
    pangulu_common *pangulu_common;
    pangulu_block_common *block_common;
    pangulu_block_smatrix *block_smatrix;
} pangulu_numeric_thread_param;

typedef struct pangulu_digest_coo_t
{
    pangulu_exblock_idx row;
    pangulu_exblock_idx col;
    pangulu_inblock_ptr nnz;
} pangulu_digest_coo_t;

typedef struct pangulu_handle_t
{
    pangulu_block_common *block_common;
    pangulu_block_smatrix *block_smatrix;
    pangulu_common *commmon;
} pangulu_handle_t;

typedef struct pangulu_symbolic_node_t
{
    pangulu_int64_t value;
    struct pangulu_symbolic_node_t *next;
} pangulu_symbolic_node_t;

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef PANGULU_PLATFORM_ENV
    // pangulu_communication.c
    void pangulu_cm_rank(pangulu_int32_t *rank);
    void pangulu_cm_size(pangulu_int32_t *size);
    void pangulu_cm_sync();
    void pangulu_cm_bcast(void *buffer, int count, MPI_Datatype datatype, int root);
    void pangulu_cm_isend(char *buf, pangulu_int64_t count, pangulu_int32_t remote_rank, pangulu_int32_t tag, pangulu_int32_t tag_ub);
    void pangulu_cm_send(char *buf, pangulu_int64_t count, pangulu_int32_t remote_rank, pangulu_int32_t tag, pangulu_int32_t tag_ub);
    void pangulu_cm_recv(char *buf, pangulu_int64_t count, pangulu_int32_t fetch_rank, pangulu_int32_t tag, pangulu_int32_t tag_ub);
    void pangulu_cm_sync_asym(int wake_rank);
    void pangulu_cm_probe(MPI_Status *status);
    void pangulu_cm_distribute_csc_to_distcsc(
        pangulu_int32_t root_rank,
        int rootproc_free_originmatrix,
        pangulu_exblock_idx *n,
        pangulu_inblock_idx rowchunk_align,
        pangulu_int32_t *distcsc_nproc,
        pangulu_exblock_idx *n_loc,
        pangulu_exblock_ptr **distcsc_proc_nnzptr,
        pangulu_exblock_ptr **distcsc_pointer,
        pangulu_exblock_idx **distcsc_index,
        calculate_type **distcsc_value);
    void pangulu_cm_distribute_csc_to_distbcsc(
        pangulu_exblock_idx n,
        pangulu_inblock_idx nb,
        pangulu_exblock_ptr *csc_pointer,
        pangulu_exblock_idx *csc_index,
        calculate_type *csc_value,
        pangulu_exblock_ptr **bcsc_struct_pointer,
        pangulu_exblock_idx **bcsc_struct_index,
        pangulu_exblock_ptr **bcsc_struct_nnzptr,
        pangulu_inblock_ptr ***bcsc_inblock_pointers,
        pangulu_inblock_idx ***bcsc_inblock_indeces,
        calculate_type ***bcsc_values,
        pangulu_exblock_ptr **bcsc_global_pointer,
        pangulu_exblock_idx **bcsc_global_index);
    void pangulu_set_omp_threads_per_process(int total_threads, int total_processes, int process_rank);
    void pangulu_cm_distribute_csc_to_distbcsc_symb(
        pangulu_exblock_idx n,
        pangulu_inblock_idx nb,
        pangulu_exblock_ptr *csc_pointer,
        pangulu_exblock_idx *csc_index,
        pangulu_inblock_ptr ***out_diag_upper_rowptr,
        pangulu_inblock_idx ***out_diag_upper_colidx,
        calculate_type ***out_diag_upper_csrvalue,
        pangulu_inblock_ptr ***out_diag_lower_colptr,
        pangulu_inblock_idx ***out_diag_lower_rowidx,
        calculate_type ***out_diag_lower_cscvalue,
        calculate_type ***out_nondiag_cscvalue,
        pangulu_inblock_ptr ***out_nondiag_colptr,
        pangulu_inblock_idx ***out_nondiag_rowidx,
        pangulu_inblock_ptr ***out_nondiag_csr_to_csc,
        pangulu_inblock_ptr ***out_nondiag_rowptr,
        pangulu_inblock_idx ***out_nondiag_colidx,
        pangulu_exblock_ptr **out_nondiag_block_colptr,
        pangulu_exblock_idx **out_nondiag_block_rowidx,
        pangulu_exblock_ptr **out_nondiag_block_rowptr,
        pangulu_exblock_idx **out_nondiag_block_colidx,
        pangulu_exblock_ptr **out_nondiag_block_csr_to_csc,
        pangulu_exblock_ptr **out_related_nondiag_block_colptr,
        pangulu_exblock_idx **out_related_nondiag_block_rowidx,
        pangulu_exblock_ptr **out_related_nondiag_block_rowptr,
        pangulu_exblock_idx **out_related_nondiag_block_colidx,
        pangulu_exblock_ptr **out_related_nondiag_block_csr_to_csc,
        pangulu_uint64_t **out_diag_uniaddr);
    void pangulu_cm_recv_block(
        MPI_Status *msg_stat,
        pangulu_storage_t *storage,
        pangulu_uint64_t slot_addr,
        pangulu_exblock_idx block_length,
        pangulu_inblock_idx nb,
        pangulu_exblock_idx *bcol_pos,
        pangulu_exblock_idx *brow_pos,
        pangulu_exblock_ptr *related_nondiag_block_colptr,
        pangulu_exblock_idx *related_nondiag_block_rowidx,
        pangulu_uint64_t *related_nondiag_uniaddr,
        pangulu_uint64_t *diag_uniaddr);
    void pangulu_cm_isend_block(
        pangulu_storage_slot_t *slot,
        pangulu_inblock_idx nb,
        pangulu_exblock_idx brow_pos,
        pangulu_exblock_idx bcol_pos,
        pangulu_int32_t target_rank);
    // pangulu_communication.c end
#endif

    // pangulu_conversion.c
    void pangulu_convert_csr_to_csc_block(
        int free_csrmatrix,
        pangulu_inblock_idx n,
        pangulu_inblock_ptr **csr_pointer,
        pangulu_inblock_idx **csr_index,
        calculate_type **csr_value,
        pangulu_inblock_ptr **csc_pointer,
        pangulu_inblock_idx **csc_index,
        calculate_type **csc_value);
    void pangulu_convert_csr_to_csc_block_with_index(
        pangulu_exblock_ptr n,
        pangulu_exblock_ptr *in_pointer,
        pangulu_exblock_idx *in_index,
        pangulu_exblock_ptr *out_pointer,
        pangulu_exblock_idx *out_index,
        pangulu_exblock_ptr *out_csr_to_csc);
    void pangulu_convert_csr_to_csc(
        int free_csrmatrix,
        pangulu_exblock_idx n,
        pangulu_exblock_ptr **csr_pointer,
        pangulu_exblock_idx **csr_index,
        calculate_type **csr_value,
        pangulu_exblock_ptr **csc_pointer,
        pangulu_exblock_idx **csc_index,
        calculate_type **csc_value);
    void pangulu_convert_halfsymcsc_to_csc_struct(
        int free_halfmatrix,
        int if_colsort,
        pangulu_exblock_idx n,
        pangulu_exblock_ptr **half_csc_pointer,
        pangulu_exblock_idx **half_csc_index,
        pangulu_exblock_ptr **csc_pointer,
        pangulu_exblock_idx **csc_index);
    void pangulu_convert_block_fill_value_to_struct(
        pangulu_int32_t p,
        pangulu_int32_t q,
        pangulu_int32_t rank,
        pangulu_exblock_idx n,
        pangulu_inblock_idx nb,
        pangulu_exblock_ptr *value_block_pointer,
        pangulu_exblock_idx *value_block_index,
        pangulu_inblock_ptr *value_block_nnzptr,
        pangulu_inblock_ptr **value_bcsc_inblock_pointers,
        pangulu_inblock_idx **value_bcsc_inblock_indeces,
        calculate_type **value_bcsc_values,
        pangulu_exblock_ptr *nondiag_block_colpointer,
        pangulu_exblock_idx *nondiag_block_rowindex,
        pangulu_inblock_ptr **nondiag_colpointers,
        pangulu_inblock_idx **nondiag_rowindeces,
        calculate_type **nondiag_values,
        pangulu_uint64_t *diag_uniaddr,
        pangulu_inblock_ptr **diag_upper_rowpointers,
        pangulu_inblock_idx **diag_upper_colindeces,
        calculate_type **diag_upper_values,
        pangulu_inblock_ptr **diag_lower_colpointers,
        pangulu_inblock_idx **diag_lower_rowindeces,
        calculate_type **diag_lower_values);
    void pangulu_convert_bcsc_to_digestcoo(
        pangulu_exblock_idx block_length,
        const pangulu_exblock_ptr *bcsc_struct_pointer,
        const pangulu_exblock_idx *bcsc_struct_index,
        const pangulu_exblock_ptr *bcsc_struct_nnzptr,
        pangulu_digest_coo_t *digest_info);
    // pangulu_conversion.c end

    // pangulu_kernel_interface.c
#ifdef PANGULU_PERF
    void pangulu_getrf_flop(
        pangulu_inblock_idx nb,
        pangulu_storage_slot_t *opdst,
        int tid);
    void pangulu_tstrf_flop(
        pangulu_inblock_idx nb,
        pangulu_storage_slot_t *opdst,
        pangulu_storage_slot_t *opdiag,
        int tid);
    void pangulu_gessm_flop(
        pangulu_inblock_idx nb,
        pangulu_storage_slot_t *opdst,
        pangulu_storage_slot_t *opdiag,
        int tid);
    void pangulu_ssssm_flop(
        pangulu_inblock_idx nb,
        pangulu_storage_slot_t *opdst,
        pangulu_storage_slot_t *op1,
        pangulu_storage_slot_t *op2,
        int tid);
#endif
    void pangulu_getrf_interface(
        pangulu_inblock_idx nb,
        pangulu_storage_slot_t *opdst,
        int tid);
    void pangulu_tstrf_interface(
        pangulu_inblock_idx nb,
        pangulu_storage_slot_t *opdst,
        pangulu_storage_slot_t *opdiag,
        int tid);
    void pangulu_gessm_interface(
        pangulu_inblock_idx nb,
        pangulu_storage_slot_t *opdst,
        pangulu_storage_slot_t *opdiag,
        int tid);
    void pangulu_ssssm_interface(
        pangulu_inblock_idx nb,
        pangulu_storage_slot_t *opdst,
        pangulu_storage_slot_t *op1,
        pangulu_storage_slot_t *op2,
        int tid);
    void pangulu_hybrid_batched_interface(
        pangulu_inblock_idx nb,
        pangulu_uint64_t ntask,
        pangulu_task_t *tasks);
    void pangulu_ssssm_batched_interface(
        pangulu_inblock_idx nb,
        pangulu_uint64_t ntask,
        pangulu_task_t *tasks);
    void pangulu_spmv_interface(
        pangulu_inblock_idx nb,
        pangulu_storage_slot_t *a,
        calculate_type *x,
        calculate_type *y);
    void pangulu_vecadd_interface(
        pangulu_int64_t length,
        calculate_type *bval,
        calculate_type *xval);
    void pangulu_sptrsv_interface(
        pangulu_inblock_idx nb,
        pangulu_storage_slot_t *s,
        calculate_type *xval,
        pangulu_int64_t uplo);
    // pangulu_kernel_interface.c end

    // pangulu_memory.c
    void *pangulu_malloc(const char *file, pangulu_int64_t line, pangulu_int64_t size);
    void *pangulu_realloc(const char *file, pangulu_int64_t line, void *oldptr, pangulu_int64_t size);
    void pangulu_free(const char *file, pangulu_int64_t line, void *ptr);
    void pangulu_origin_smatrix_add_csc(pangulu_origin_smatrix *a);
    // pangulu_memory.c end

#ifndef PANGULU_PLATFORM_ENV
    // pangulu_numeric.c
    void pangulu_numeric_receive_message(
        MPI_Status status,
        pangulu_block_common *block_common,
        pangulu_block_smatrix *block_smatrix);
    int pangulu_execute_aggregated_ssssm(
        unsigned long long ntask,
        void *_task_descriptors,
        void *_extra_params);
    void pangulu_numeric_work_batched(
        pangulu_int64_t ntask,
        pangulu_task_t *tasks,
        pangulu_common *common,
        pangulu_block_common *block_common,
        pangulu_block_smatrix *block_smatrix);
    void pangulu_numeric_work(
        pangulu_task_t *task,
        pangulu_common *common,
        pangulu_block_common *block_common,
        pangulu_block_smatrix *block_smatrix);
    void *pangulu_numeric_compute_thread(void *param);
    void pangulu_numeric(
        pangulu_common *common,
        pangulu_block_common *block_common,
        pangulu_block_smatrix *block_smatrix);
    void pangulu_numeric_check(
        pangulu_common *common,
        pangulu_block_common *block_common,
        pangulu_block_smatrix *block_smatrix);
    // pangulu_numeric.c end
#endif

    // pangulu_preprocessing.c
    void pangulu_preprocessing(
        pangulu_common *common,
        pangulu_block_common *bcommon,
        pangulu_block_smatrix *bsmatrix,
        pangulu_origin_smatrix *reorder_matrix,
        pangulu_int32_t nthread);
    // pangulu_preprocessing.c end

    // pangulu_reordering.c
#ifndef PANGULU_PLATFORM_ENV
#ifdef PANGULU_MC64
    void pangulu_mc64dd(
        pangulu_int64_t col,
        pangulu_int64_t n,
        pangulu_int64_t *queue,
        const calculate_type *row_scale_value,
        pangulu_int64_t *save_tmp);
    void pangulu_mc64ed(
        pangulu_int64_t *queue_length,
        pangulu_int64_t n,
        pangulu_int64_t *queue,
        const calculate_type *row_scale_value,
        pangulu_int64_t *save_tmp);
    void pangulu_mc64fd(
        pangulu_int64_t loc_origin,
        pangulu_int64_t *queue_length,
        pangulu_int64_t n,
        pangulu_int64_t *queue,
        const calculate_type *row_scale_value,
        pangulu_int64_t *save_tmp);
    void pangulu_mc64(
        pangulu_origin_smatrix *s,
        pangulu_exblock_idx **perm,
        pangulu_exblock_idx **iperm,
        calculate_type **row_scale,
        calculate_type **col_scale);
#endif
    void pangulu_reorder_vector_b_tran(
        pangulu_exblock_idx *row_perm,
        pangulu_exblock_idx *metis_perm,
        calculate_type *row_scale,
        pangulu_vector *B_origin,
        pangulu_vector *B_trans);
    void pangulu_reorder_vector_x_tran(
        pangulu_block_smatrix *block_smatrix,
        pangulu_vector *X_origin,
        pangulu_vector *X_trans);
    void pangulu_add_diagonal_element_csc(pangulu_origin_smatrix *s);
    void pangulu_origin_smatrix_add_csr(pangulu_origin_smatrix *a);
    void pangulu_get_graph_struct(pangulu_origin_smatrix *s, reordering_int_t **xadj_address, reordering_int_t **adjncy_address);
    void pangulu_get_graph_struct_csc(pangulu_origin_smatrix *s, reordering_int_t **xadj_address, reordering_int_t **adjncy_address);
#ifdef METIS
    void pangulu_metis(pangulu_origin_smatrix *a, reordering_int_t **metis_perm);
#else
    void pangulu_reordering_mt(pangulu_origin_smatrix *a, reordering_int_t **metis_perm, reordering_int_t nthreads);
#endif
    void pangulu_origin_smatrix_transport_transport_iperm(
        pangulu_origin_smatrix *s,
        pangulu_origin_smatrix *new_S,
        const pangulu_exblock_idx *metis_perm);
    void pangulu_reordering(
        pangulu_block_smatrix *block_smatrix,
        pangulu_origin_smatrix *origin_matrix,
        pangulu_origin_smatrix *reorder_matrix,
        pangulu_int32_t reordering_nthread);
#endif
    // pangulu_reordering.c end

    // pangulu_sptrsv.c
    void pangulu_sptrsv_preprocessing(
        pangulu_block_common *bcommon,
        pangulu_block_smatrix *bsmatrix,
        pangulu_vector *reordered_rhs);
    void pangulu_sptrsv_uplo(
        pangulu_block_common *bcommon,
        pangulu_block_smatrix *bsmatrix,
        pangulu_int32_t uplo);
    void pangulu_solve(
        pangulu_block_common *block_common,
        pangulu_block_smatrix *block_smatrix,
        pangulu_vector *result);
    // pangulu_sptrsv.c end

    // pangulu_storage.c
    pangulu_int32_t pangulu_storage_bin_navail(pangulu_storage_bin_t *bin);
    pangulu_int32_t pangulu_storage_slot_queue_alloc(pangulu_storage_bin_t *bin);
    void pangulu_storage_slot_queue_recycle(
        pangulu_storage_t *storage,
        pangulu_uint64_t *slot_addr);
    void pangulu_storage_slot_queue_recycle_by_ptr(
        pangulu_storage_t *storage,
        pangulu_storage_slot_t *slot);
    pangulu_uint64_t pangulu_storage_allocate_slot(
        pangulu_storage_t *storage,
        pangulu_int64_t size);
    pangulu_storage_slot_t *pangulu_storage_get_slot(
        pangulu_storage_t *storage,
        pangulu_uint64_t slot_addr);
    pangulu_storage_slot_t *pangulu_storage_get_diag(
        pangulu_exblock_idx block_length,
        pangulu_storage_t *storage,
        pangulu_uint64_t diag_addr);
    void pangulu_storage_bin_init(
        pangulu_storage_bin_t *bin,
        pangulu_int32_t bin_id,
        pangulu_int64_t slot_capacity,
        pangulu_int32_t slot_count);
    void pangulu_storage_init(
        pangulu_storage_t *storage,
        pangulu_int64_t *slot_capacity,
        pangulu_int32_t *slot_count,
        pangulu_exblock_idx block_length,
        pangulu_exblock_ptr *bcsc_pointer,
        pangulu_exblock_idx *bcsc_index,
        pangulu_inblock_ptr **bcsc_inblk_pointers,
        pangulu_inblock_idx **bcsc_inblk_indeces,
        calculate_type **bcsc_inblk_values,
        pangulu_inblock_ptr **bcsr_inblk_pointers,
        pangulu_inblock_idx **bcsr_inblk_indeces,
        pangulu_inblock_ptr **bcsr_inblk_valueindices,
        pangulu_uint64_t *diag_uniaddr,
        pangulu_inblock_ptr **diag_upper_rowptr,
        pangulu_inblock_idx **diag_upper_colidx,
        calculate_type **diag_upper_csrvalue,
        pangulu_inblock_ptr **diag_lower_colptr,
        pangulu_inblock_idx **diag_lower_rowidx,
        calculate_type **diag_lower_cscvalue,
        pangulu_inblock_idx nb);
    // pangulu_storage.c end

    // pangulu_symbolic.c
    void pangulu_a_plus_at(
        const pangulu_exblock_idx n,
        const pangulu_exblock_ptr nz,
        pangulu_exblock_ptr *colptr,
        pangulu_exblock_idx *rowind,
        pangulu_exblock_ptr *bnz,
        pangulu_exblock_ptr **b_colptr,
        pangulu_exblock_idx **b_rowind);
    void pangulu_symbolic_add_prune(
        pangulu_symbolic_node_t *prune,
        pangulu_symbolic_node_t *prune_next,
        pangulu_int64_t num,
        pangulu_int64_t num_value,
        pangulu_int64_t p);
    void pangulu_symbolic_symmetric(
        pangulu_exblock_idx n,
        pangulu_exblock_ptr nnz,
        pangulu_exblock_idx *ai,
        pangulu_exblock_ptr *ap,
        pangulu_exblock_ptr **symbolic_rowpointer,
        pangulu_exblock_idx **symbolic_columnindex,
        pangulu_inblock_idx nb,
        pangulu_exblock_idx block_length,
        pangulu_exblock_ptr *symbolic_nnz);
    void pangulu_symbolic(
        pangulu_block_common *block_common,
        pangulu_block_smatrix *block_smatrix,
        pangulu_origin_smatrix *reorder_matrix);
    // pangulu_symbolic.c end

    // pangulu_task.c
    int pangulu_aggregate_init();
    int pangulu_aggregate_task_store(pangulu_storage_slot_t *opdst, pangulu_task_t *task_descriptor);
    int pangulu_aggregate_task_compute(
        pangulu_storage_slot_t *opdst,
        int (*compute_callback)(unsigned long long, void *, void *),
        void *extra_params);
    int pangulu_aggregate_task_compute_multi_tile(
        unsigned long long ntile, 
        pangulu_storage_slot_t **opdst,
        int (*compute_callback)(unsigned long long, void *, void *), 
        void *extra_params);
    int pangulu_aggregate_idle_batch(
        pangulu_common *common,
        pangulu_block_common *block_common,
        pangulu_block_smatrix *block_smatrix,
        int chunk_ntask,
        int (*compute_callback)(unsigned long long, void *, void *),
        void *extra_params);
    pangulu_int64_t pangulu_task_queue_alloc(pangulu_task_queue_t *tq);
    void pangulu_task_queue_recycle(
        pangulu_task_queue_t *tq,
        pangulu_int64_t store_idx);
    void pangulu_task_queue_recycle(
        pangulu_task_queue_t *tq,
        pangulu_int64_t store_idx);
    void pangulu_task_queue_cmp_strategy(
        pangulu_task_queue_t *tq,
        pangulu_int32_t cmp_strategy);
    void pangulu_task_queue_init(
        pangulu_task_queue_t *heap,
        pangulu_int64_t capacity);
    pangulu_task_queue_t *pangulu_task_queue_destory(pangulu_task_queue_t *heap);
    void pangulu_task_queue_clear(pangulu_task_queue_t *heap);
    void pangulu_task_swap(pangulu_int64_t *task_index_heap, pangulu_int64_t a, pangulu_int64_t b);
    void pangulu_task_queue_push(
        pangulu_task_queue_t *heap,
        pangulu_int64_t row,
        pangulu_int64_t col,
        pangulu_int64_t task_level,
        pangulu_int64_t kernel_id,
        pangulu_int64_t compare_flag,
        pangulu_storage_slot_t *opdst,
        pangulu_storage_slot_t *op1,
        pangulu_storage_slot_t *op2,
        pangulu_int64_t block_length,
        const char *file,
        int line);
    char pangulu_task_queue_empty(pangulu_task_queue_t *heap);
    pangulu_task_t pangulu_task_queue_delete(pangulu_task_queue_t *heap);
    pangulu_task_t pangulu_task_queue_pop(pangulu_task_queue_t *heap);
    // pangulu_task.c end

    // pangulu_thread.c
    void pangulu_bind_to_core(pangulu_int32_t core);
    void pangulu_mutex_init(pthread_mutex_t *mutex);
    void pangulu_bsem_init(pangulu_bsem_t *bsem_p, pangulu_int64_t value);
    pangulu_bsem_t *pangulu_bsem_destory(pangulu_bsem_t *bsem_p);
    void pangulu_bsem_post(pangulu_task_queue_t *heap);
    void pangulu_bsem_stop(pangulu_task_queue_t *heap);
    void pangulu_bsem_synchronize(pangulu_bsem_t *bsem_p);
    // pangulu_thread.c end

    // pangulu_utils.c
    void pangulu_init_pangulu_origin_smatrix(pangulu_origin_smatrix *s);
    void pangulu_init_pangulu_block_smatrix(pangulu_block_smatrix *bs);
    void pangulu_time_start(struct timeval *start);
    double pangulu_time_stop(struct timeval *start);
    void pangulu_add_diagonal_element(pangulu_origin_smatrix *s);
    int pangulu_cmp_exidx_asc(const void *a, const void *b);
    void pangulu_kvsort(pangulu_exblock_idx *key, calculate_type *val, pangulu_int64_t start, pangulu_int64_t end);
    void pangulu_kvsort2(pangulu_exblock_idx *key, pangulu_uint64_t *val, pangulu_int64_t start, pangulu_int64_t end);
    void pangulu_sort_pangulu_origin_smatrix(pangulu_origin_smatrix *s);
    void pangulu_sort_pangulu_origin_smatrix_csc(pangulu_origin_smatrix *s);
    void pangulu_sort_exblock_struct(
        pangulu_exblock_idx n,
        pangulu_exblock_ptr *pointer,
        pangulu_exblock_idx *index,
        pangulu_int32_t nthread);
    void pangulu_sort_exblock_struct(
        pangulu_exblock_idx n,
        pangulu_exblock_ptr *pointer,
        pangulu_exblock_idx *index,
        pangulu_int32_t nthread);
    void pangulu_swap_index_1(pangulu_exblock_idx *a, pangulu_exblock_idx *b);
    void pangulu_swap_value(calculate_type *a, calculate_type *b);
    void pangulu_swap_value2(pangulu_uint64_t *a, pangulu_uint64_t *b);
    int pangulu_binarylowerbound(const pangulu_int64_t *arr, int len, pangulu_int64_t value);
    void pangulu_exclusive_scan_1(pangulu_int64_t *input, int length);
    void pangulu_exclusive_scan_3(unsigned int *input, int length);
    pangulu_int64_t pangulu_binarysearch(const pangulu_exblock_idx *arr, pangulu_int64_t left, pangulu_int64_t right, pangulu_int64_t target);
    pangulu_int32_t pangulu_binarysearch_inblk(const pangulu_inblock_idx *arr, pangulu_int32_t left, pangulu_int32_t right, pangulu_int32_t target);
    pangulu_int64_t pangulu_binarysearch_first_ge(const pangulu_exblock_idx *arr, pangulu_int64_t left, pangulu_int64_t right, pangulu_int64_t target);
    pangulu_int64_t pangulu_binarysearch_last_le(const pangulu_exblock_idx *arr, pangulu_int64_t left, pangulu_int64_t right, pangulu_int64_t target);
    int pangulu_binarysearch_in_column(pangulu_exblock_ptr *block_col_pt_full, pangulu_exblock_idx *block_row_idx_full, pangulu_int64_t block_length, pangulu_exblock_idx brow, pangulu_exblock_idx bcol);
    void pangulu_log_memory_usage();
    void pangulu_zero_pangulu_vector(pangulu_vector *v);
    void pangulu_init_pangulu_vector(pangulu_vector *b, pangulu_int64_t n);
    pangulu_vector *pangulu_destroy_pangulu_vector(pangulu_vector *v);
    void pangulu_transpose_struct_with_valueidx_inblock(
        const pangulu_inblock_idx nb,
        const pangulu_inblock_ptr *in_ptr,
        const pangulu_inblock_idx *in_idx,
        pangulu_inblock_ptr *out_ptr,
        pangulu_inblock_idx *out_idx,
        pangulu_inblock_ptr *out_valueidx);
    void pangulu_transpose_struct_with_valueidx_exblock(
        const pangulu_exblock_idx block_length,
        const pangulu_exblock_ptr *in_ptr,
        const pangulu_exblock_idx *in_idx,
        pangulu_exblock_ptr *out_ptr,
        pangulu_exblock_idx *out_idx,
        pangulu_exblock_ptr *out_valueidx,
        pangulu_exblock_ptr *aid_ptr);
    void pangulu_diag_block_trans(
        pangulu_inblock_idx nb,
        pangulu_inblock_ptr *diag_rowptr,
        pangulu_inblock_idx *diag_colidx,
        pangulu_inblock_ptr *nondiag_colptr,
        pangulu_inblock_idx *nondiag_rowidx);
    void pangulu_destroy(
        pangulu_block_common *block_common,
        pangulu_block_smatrix *block_smatrix);
    // pangulu_utils.c end
#include "../include/pangulu.h"

#ifdef __cplusplus
} // extern "C"
#endif

#include "./platforms/pangulu_platform_common.h"
#endif
