#include "pangulu_common.h"

pangulu_int64_t cpu_memory = 0;
pangulu_int64_t cpu_peak_memory = 0;
pangulu_int64_t gpu_memory = 0;
pangulu_int64_t heap_select;
calculate_type *temp_a_value = NULL;
pangulu_int32_t *cuda_b_idx_col = NULL;
calculate_type *cuda_temp_value = NULL;
pangulu_int64_t *ssssm_col_ops_u = NULL;
pangulu_int32_t *ssssm_ops_pointer = NULL;
pangulu_int32_t *getrf_diagIndex_csc = NULL;
pangulu_int32_t *getrf_diagIndex_csr = NULL;

pangulu_int64_t STREAM_DENSE_INDEX = 0;
pangulu_int64_t INDEX_NUM = 0;
pangulu_int32_t pangu_omp_num_threads = 1;

pangulu_int64_t flop = 0;
double time_transpose = 0.0;
double time_isend = 0.0;
double time_receive = 0.0;
double time_getrf = 0.0;
double time_tstrf = 0.0;
double time_gessm = 0.0;
double time_gessm_dense = 0.0;
double time_gessm_sparse = 0.0;
double time_ssssm = 0.0;
double time_cuda_memcpy = 0.0;
double time_wait = 0.0;
double calculate_time_wait = 0.0;
pangulu_int64_t calculate_time = 0;

pangulu_int32_t *ssssm_hash_lu = NULL;
pangulu_int32_t *ssssm_hash_l_row = NULL;
pangulu_int32_t zip_cur_id = 0;
calculate_type *ssssm_l_value = NULL;
calculate_type *ssssm_u_value = NULL;
pangulu_int32_t *ssssm_hash_u_col = NULL;

pangulu_int32_t rank;
pangulu_int32_t global_level;
pangulu_int32_t omp_thread;

void pangulu_init(pangulu_exblock_idx pangulu_n, pangulu_exblock_ptr pangulu_nnz, pangulu_exblock_ptr *csr_rowptr, pangulu_exblock_idx *csr_colidx, calculate_type *csr_value, pangulu_init_options *init_options, void **pangulu_handle)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    struct timeval time_start;
    double elapsed_time;

    pangulu_int32_t size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    pangulu_common *common = (pangulu_common *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_common));
    common->rank = rank;
    common->size = size;
    common->n = pangulu_n;
#ifdef GPU_OPEN
    if (init_options->nb > 256 && sizeof(pangulu_inblock_idx) == 2)
    {
        init_options->nb = 256;
        if (rank == 0)
        {
            printf(PANGULU_W_GPU_BIG_BLOCK);
        }
    }
#endif

    if (rank == 0)
    {
        if (init_options == NULL)
        {
            printf(PANGULU_E_OPTION_IS_NULLPTR);
            pangulu_exit(1);
        }
        if (init_options->nb == 0)
        {
            printf(PANGULU_E_NB_IS_ZERO);
            pangulu_exit(1);
        }
    }

    common->nb = init_options->nb;
    common->sum_rank_size = size;
    common->omp_thread = init_options->nthread;
    MPI_Bcast(&common->n, 1, MPI_PANGULU_EXBLOCK_IDX, 0, MPI_COMM_WORLD);
    MPI_Bcast(&common->nb, 1, MPI_PANGULU_INBLOCK_IDX, 0, MPI_COMM_WORLD);

    pangulu_int64_t tmp_p = sqrt(common->sum_rank_size);
    while (((common->sum_rank_size) % tmp_p) != 0)
    {
        tmp_p--;
    }

    common->p = tmp_p;
    common->q = common->sum_rank_size / tmp_p;
    pangulu_origin_smatrix *origin_smatrix = (pangulu_origin_smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_origin_smatrix));
    pangulu_init_pangulu_origin_smatrix(origin_smatrix);

    if (rank == 0)
    {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        pangulu_read_pangulu_origin_smatrix(origin_smatrix, pangulu_n, pangulu_nnz, csr_rowptr, csr_colidx, csr_value);
        gettimeofday(&end, NULL);
        if (origin_smatrix->row == 0)
        {
            printf(PANGULU_E_ROW_IS_ZERO);
            pangulu_exit(1);
        }
    }

    pangulu_int32_t p = common->p;
    pangulu_int32_t q = common->q;
    pangulu_int32_t nb = common->nb;
    MPI_Barrier(MPI_COMM_WORLD);
    common->n = pangulu_bcast_n(origin_smatrix->row, 0);
    pangulu_int64_t n = common->n;
    omp_set_num_threads(init_options->nthread);
#if defined(OPENBLAS_CONFIG_H) || defined(OPENBLAS_VERSION)
    openblas_set_num_threads(1);
#endif
    if (rank == 0)
    {
// #ifdef ADAPTIVE_KERNEL_SELECTION
//         printf(PANGULU_I_ADAPTIVE_KERNEL_SELECTION_ON);
// #else
//         printf(PANGULU_I_ADAPTIVE_KERNEL_SELECTION_OFF);
// #endif
// #ifdef SYNCHRONIZE_FREE
//         printf(PANGULU_I_SYNCHRONIZE_FREE_ON);
// #else
//         printf(PANGULU_I_SYNCHRONIZE_FREE_OFF);
// #endif
#ifdef PANGULU_GPU_COMPLEX_FALLBACK_FLAG
        printf(PANGULU_W_COMPLEX_FALLBACK);
#endif
        omp_thread = pangu_omp_num_threads;
        printf(PANGULU_I_BASIC_INFO);
    }

#ifdef GPU_OPEN
    pangulu_cuda_device_init(rank);
#endif

    pangulu_block_smatrix *block_smatrix = (pangulu_block_smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_block_smatrix));
    pangulu_init_pangulu_block_smatrix(block_smatrix);
    pangulu_block_common *block_common = (pangulu_block_common *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_block_common));
    block_common->rank = rank;
    block_common->p = p;
    block_common->q = q;
    block_common->nb = nb;
    block_common->n = n;
    block_common->block_length = pangulu_Calculate_Block(n, nb);
    block_common->sum_rank_size = common->sum_rank_size;
    block_common->max_pq = PANGULU_MAX(p, q);
    block_common->every_level_length = block_common->block_length;
    pangulu_bip_init(&(block_smatrix->BIP), block_common->block_length * (block_common->block_length + 1));

#ifdef SYNCHRONIZE_FREE
    block_common->every_level_length = 10;
#else
    block_common->every_level_length = 1;
#endif

    pangulu_origin_smatrix *reorder_matrix = (pangulu_origin_smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_origin_smatrix));
    pangulu_init_pangulu_origin_smatrix(reorder_matrix);

    block_common->rank_row_length = (block_common->block_length / p + (((block_common->block_length % p) > (rank / q)) ? 1 : 0));
    block_common->rank_col_length = (block_common->block_length / q + (((block_common->block_length % q) > (rank % q)) ? 1 : 0));
    block_common->every_level_length = PANGULU_MIN(block_common->every_level_length, block_common->block_length);
    MPI_Barrier(MPI_COMM_WORLD);
    pangulu_time_start(&time_start);

    pangulu_reorder(block_smatrix,
                    origin_smatrix,
                    reorder_matrix);

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = pangulu_time_stop(&time_start);
    if (rank == 0)
    {
        printf(PANGULU_I_TIME_REORDER);
    }

    calculate_time = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    pangulu_time_start(&time_start);
    if (rank == 0)
    {
        pangulu_symbolic(block_common,
                         block_smatrix,
                         reorder_matrix);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = pangulu_time_stop(&time_start);
    if (rank == 0)
    {
        printf(PANGULU_I_TIME_SYMBOLIC);
    }

    pangulu_init_heap_select(0);

    MPI_Barrier(MPI_COMM_WORLD);
    pangulu_time_start(&time_start);
    pangulu_preprocessing(
        block_common,
        block_smatrix,
        reorder_matrix,
        init_options->nthread);

    MPI_Barrier(MPI_COMM_WORLD);

    elapsed_time = pangulu_time_stop(&time_start);
    if (rank == 0)
    {
        printf(PANGULU_I_TIME_PRE);
    }

    // pangulu_free(__FILE__, __LINE__, block_smatrix->symbolic_rowpointer);
    // block_smatrix->symbolic_rowpointer = NULL;

    // pangulu_free(__FILE__, __LINE__, block_smatrix->symbolic_columnindex);
    // block_smatrix->symbolic_columnindex = NULL;

    pangulu_free(__FILE__, __LINE__, origin_smatrix);
    origin_smatrix = NULL;

    pangulu_free(__FILE__, __LINE__, reorder_matrix->rowpointer);
    pangulu_free(__FILE__, __LINE__, reorder_matrix->columnindex);
    pangulu_free(__FILE__, __LINE__, reorder_matrix->value);
    pangulu_free(__FILE__, __LINE__, reorder_matrix);
    reorder_matrix = NULL;

    (*pangulu_handle) = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_handle_t));
    (*(pangulu_handle_t **)pangulu_handle)->block_common = block_common;
    (*(pangulu_handle_t **)pangulu_handle)->block_smatrix = block_smatrix;
    (*(pangulu_handle_t **)pangulu_handle)->commmon = common;
}

void pangulu_gstrf(pangulu_gstrf_options *gstrf_options, void **pangulu_handle)
{
    pangulu_block_common *block_common = (*(pangulu_handle_t **)pangulu_handle)->block_common;
    pangulu_block_smatrix *block_smatrix = (*(pangulu_handle_t **)pangulu_handle)->block_smatrix;
    pangulu_common *common = (*(pangulu_handle_t **)pangulu_handle)->commmon;

    struct timeval time_start;
    double elapsed_time;

    if (rank == 0)
    {
        if (gstrf_options == NULL)
        {
            printf(PANGULU_E_GSTRF_OPTION_IS_NULLPTR);
            pangulu_exit(1);
        }
    }

#ifdef CHECK_TIME
    pangulu_time_init();
#endif
    MPI_Barrier(MPI_COMM_WORLD);

#ifdef OVERLAP
    pangulu_create_pthread(block_common,
                           block_smatrix);
#endif

    pangulu_time_init();
    MPI_Barrier(MPI_COMM_WORLD);
    pangulu_time_start(&time_start);

    pangulu_numeric(block_common,
                    block_smatrix);

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = pangulu_time_stop(&time_start);

    if (rank == 0)
    {

        pangulu_int64_t another_calculate_time = 0;
        for (pangulu_int64_t i = 1; i < block_common->sum_rank_size; i++)
        {
            pangulu_recv_vector_int(&another_calculate_time, 1, i, 0);
            calculate_time += another_calculate_time;
        }
        flop = calculate_time * 2;
    }
    else
    {
        pangulu_send_vector_int(&calculate_time, 1, 0, 0);
    }

    if (rank == 0)
    {
        printf(PANGULU_I_TIME_NUMERICAL);
    }
}

void pangulu_gstrs(calculate_type *rhs, pangulu_gstrs_options *gstrs_options, void **pangulu_handle)
{
    pangulu_block_common *block_common = (*(pangulu_handle_t **)pangulu_handle)->block_common;
    pangulu_block_smatrix *block_smatrix = (*(pangulu_handle_t **)pangulu_handle)->block_smatrix;
    pangulu_common *common = (*(pangulu_handle_t **)pangulu_handle)->commmon;

    struct timeval time_start;
    double elapsed_time;

    if (rank == 0)
    {
        if (gstrs_options == NULL)
        {
            printf(PANGULU_E_GSTRS_OPTION_IS_NULLPTR);
            pangulu_exit(1);
        }
    }

    pangulu_int64_t vector_length = common->n;
    pangulu_vector *x_vector = NULL;
    pangulu_vector *b_vector = NULL;
    pangulu_vector *answer_vector = NULL;

    if (rank == 0)
    {
        x_vector = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));
        b_vector = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));
        answer_vector = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));
        b_vector->row = common->n;
        b_vector->value = rhs;
        pangulu_init_pangulu_vector(x_vector, vector_length);
        pangulu_init_pangulu_vector(answer_vector, vector_length);
        pangulu_reorder_vector_b_tran(block_smatrix, b_vector, answer_vector);
    }

    pangulu_sptrsv_preprocessing(
        block_common,
        block_smatrix,
        answer_vector);

#ifdef PANGULU_SPTRSV

    MPI_Barrier(MPI_COMM_WORLD);
    pangulu_time_start(&time_start);

    pangulu_sptrsv_L(block_common, block_smatrix);
    pangulu_init_heap_select(4);
    pangulu_sptrsv_U(block_common, block_smatrix);

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = pangulu_time_stop(&time_start);

    if (rank == 0)
    {
        printf(PANGULU_I_TIME_SPTRSV);
    }

#endif

    // check sptrsv answer
    pangulu_sptrsv_vector_gather(block_common, block_smatrix, answer_vector);

    int n = common->n;

    if (rank == 0)
    {
        pangulu_reorder_vector_x_tran(block_smatrix, answer_vector, x_vector);

        for (int i = 0; i < n; i++)
        {
            rhs[i] = x_vector->value[i];
        }

        pangulu_destroy_pangulu_vector(x_vector);
        pangulu_destroy_pangulu_vector(answer_vector);
        pangulu_free(__FILE__, __LINE__, b_vector);
    }
}

void pangulu_gssv(calculate_type *rhs, pangulu_gstrf_options *gstrf_options, pangulu_gstrs_options *gstrs_options, void **pangulu_handle)
{
    pangulu_gstrf(gstrf_options, pangulu_handle);
    pangulu_gstrs(rhs, gstrs_options, pangulu_handle);
}

void pangulu_finalize(void **pangulu_handle)
{
    pangulu_block_common *block_common = (*(pangulu_handle_t **)pangulu_handle)->block_common;
    pangulu_block_smatrix *block_smatrix = (*(pangulu_handle_t **)pangulu_handle)->block_smatrix;
    pangulu_common *common = (*(pangulu_handle_t **)pangulu_handle)->commmon;

    pangulu_destroy(block_common, block_smatrix);

    pangulu_free(__FILE__, __LINE__, block_common);
    pangulu_free(__FILE__, __LINE__, block_smatrix);
    pangulu_free(__FILE__, __LINE__, common);
    pangulu_free(__FILE__, __LINE__, *(pangulu_handle_t **)pangulu_handle);
}