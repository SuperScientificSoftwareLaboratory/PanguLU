#include "pangulu_common.h"

#ifdef PANGULU_PERF
pangulu_stat_t global_stat;
#endif

int pangulu_gpu_kernel_warp_per_block;
int pangulu_gpu_data_move_warp_per_block;
int pangulu_gpu_shared_mem_size;

void pangulu_init(
    pangulu_exblock_idx pangulu_n, 
    pangulu_exblock_ptr pangulu_nnz, 
    pangulu_exblock_ptr *csc_colptr, 
    pangulu_exblock_idx *csc_rowidx, 
    calculate_type *csc_value, 
    pangulu_init_options *init_options, 
    void **pangulu_handle)
{
    pangulu_int32_t size, rank;
    pangulu_cm_rank(&rank);
    pangulu_cm_size(&size);
    pangulu_common *common = (pangulu_common *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_common));
    common->rank = rank;
    common->size = size;
    common->n = pangulu_n;

    if((!init_options->is_complex_matrix) != (sizeof(calculate_type) == sizeof(calculate_real_type))){
        if(rank == 0)printf(PANGULU_E_COMPLEX_MISMATCH);
        exit(1);
    }

    if(init_options->sizeof_value != sizeof(calculate_type)){
        if(rank == 0)printf(PANGULU_E_ELEM_SIZE_MISMATCH);
        exit(1);
    }

    if (rank == 0)
    {   
        if (init_options == NULL)
        {
            printf(PANGULU_E_OPTION_IS_NULLPTR);
            exit(1);
        }
        if (init_options->nb == 0)
        {
            printf(PANGULU_E_NB_IS_ZERO);
            exit(1);
        }
    }

    if(init_options->nb <= 0){
        common->nb = 256;
    }else{
        common->nb = init_options->nb;
    }

    common->sum_rank_size = size;
    
    if(init_options->nthread == 0){
        common->omp_thread = 1;
    }else{
        common->omp_thread = init_options->nthread;
    }

    common->basic_param = init_options->mpi_recv_buffer_level;

    if(init_options->hunyuan_nthread == 0){
        common->hunyuan_nthread = 4;
    }else{
        common->hunyuan_nthread = init_options->hunyuan_nthread;
    }

    if(init_options->gpu_data_move_warp_per_block == 0){
        pangulu_gpu_data_move_warp_per_block = 4;
    }else{
        pangulu_gpu_data_move_warp_per_block = init_options->gpu_data_move_warp_per_block;
    }

    if(init_options->gpu_kernel_warp_per_block == 0){
        pangulu_gpu_kernel_warp_per_block = 4;
    }else{
        pangulu_gpu_kernel_warp_per_block = init_options->gpu_kernel_warp_per_block;
    }
    
    pangulu_cm_bcast(&common->n, 1, MPI_PANGULU_EXBLOCK_IDX, 0);
    pangulu_cm_bcast(&common->nb, 1, MPI_PANGULU_INBLOCK_IDX, 0);

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
        origin_smatrix->row = pangulu_n;
        origin_smatrix->column = pangulu_n;
        origin_smatrix->columnpointer = csc_colptr;
        origin_smatrix->rowindex = csc_rowidx;
        origin_smatrix->nnz = pangulu_nnz;
        origin_smatrix->value_csc = csc_value;
        if (origin_smatrix->row == 0)
        {
            printf(PANGULU_E_ROW_IS_ZERO);
            exit(1);
        }
    }

    pangulu_int32_t p = common->p;
    pangulu_int32_t q = common->q;
    pangulu_int32_t nb = common->nb;
    pangulu_cm_sync();
    pangulu_cm_bcast(&origin_smatrix->row, 1, MPI_PANGULU_INT64_T, 0);
    common->n = origin_smatrix->row;
    pangulu_int64_t n = common->n;
    omp_set_num_threads(init_options->nthread);
#if defined(OPENBLAS_CONFIG_H) || defined(OPENBLAS_VERSION)
    openblas_set_num_threads(1);
#endif

    if (rank == 0)
    {
        printf(PANGULU_I_BASIC_INFO);
    }

#ifdef GPU_OPEN
    int device_num;
    pangulu_platform_get_device_num(&device_num, PANGULU_DEFAULT_PLATFORM);
    pangulu_platform_set_default_device(rank%device_num, PANGULU_DEFAULT_PLATFORM);
#endif

    pangulu_block_smatrix *block_smatrix = (pangulu_block_smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_block_smatrix));
    pangulu_init_pangulu_block_smatrix(block_smatrix);
    pangulu_block_common *block_common = (pangulu_block_common *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_block_common));
    block_common->rank = rank;
    block_common->p = p;
    block_common->q = q;
    block_common->nb = nb;
    block_common->n = n;
    block_common->block_length = PANGULU_ICEIL(n, nb);
    block_common->sum_rank_size = common->sum_rank_size;
    pangulu_aggregate_init();

    pangulu_origin_smatrix *reorder_matrix = (pangulu_origin_smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_origin_smatrix));
    pangulu_init_pangulu_origin_smatrix(reorder_matrix);

    struct timeval time_start;
    double elapsed_time;
    
    pangulu_cm_sync();
    pangulu_time_start(&time_start);
    pangulu_reordering(
        block_smatrix,
        origin_smatrix,
        reorder_matrix,
        common->hunyuan_nthread
    );
    pangulu_cm_sync();
    elapsed_time = pangulu_time_stop(&time_start);
    if (rank == 0){printf(PANGULU_I_TIME_REORDER);}

#ifdef PANGULU_PERF
    if(rank == 0){
        block_smatrix->A_rowsum_reordered = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);
        memset(block_smatrix->A_rowsum_reordered, 0, sizeof(calculate_type) * n);
        for(pangulu_exblock_idx col = 0; col < n; col++){
            for(pangulu_exblock_ptr idx = reorder_matrix->columnpointer[col]; idx < reorder_matrix->columnpointer[col+1]; idx++){
                block_smatrix->A_rowsum_reordered[reorder_matrix->rowindex[idx]]+=reorder_matrix->value_csc[idx];
            }
        }
    }
    pangulu_cm_sync();
#endif

    pangulu_time_start(&time_start);
    if (rank == 0)
    {
        pangulu_symbolic(block_common,
                         block_smatrix,
                         reorder_matrix);
    }
    pangulu_cm_sync();
    elapsed_time = pangulu_time_stop(&time_start);
    if (rank == 0){printf(PANGULU_I_TIME_SYMBOLIC);}

    pangulu_cm_sync();
    pangulu_time_start(&time_start);
    pangulu_preprocessing(
        common,
        block_common,
        block_smatrix,
        reorder_matrix,
        init_options->nthread);
    pangulu_cm_sync();
    elapsed_time = pangulu_time_stop(&time_start);
    if (rank == 0){printf(PANGULU_I_TIME_PRE);}

    pangulu_free(__FILE__, __LINE__, origin_smatrix);
    origin_smatrix = NULL;
    pangulu_free(__FILE__, __LINE__, reorder_matrix);
    reorder_matrix = NULL;

    pangulu_cm_sync();

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
    pangulu_int32_t size, rank;
    pangulu_cm_rank(&rank);
    pangulu_cm_size(&size);
    if (rank == 0)
    {
#ifdef PANGULU_PERF
        printf(PANGULU_W_PERF_MODE_ON);
#endif
        if (gstrf_options == NULL)
        {
            printf(PANGULU_E_GSTRF_OPTION_IS_NULLPTR);
            exit(1);
        }
    }

    struct timeval time_start;
    double elapsed_time;

    pangulu_cm_sync();
    pangulu_time_start(&time_start);
    pangulu_numeric(common,
                    block_common,
                    block_smatrix);
    pangulu_cm_sync();
    elapsed_time = pangulu_time_stop(&time_start);
#ifdef PANGULU_PERF
    long long flop_recvbuf;
    MPI_Reduce(&global_stat.flop, &flop_recvbuf, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    global_stat.flop = flop_recvbuf;
#endif
    if (rank == 0){printf(PANGULU_I_TIME_NUMERICAL);}

#ifdef PANGULU_PERF
    if(rank == 0){
        printf(PANGULU_I_PERF_TABLE_HEAD);
        fflush(stdout);
    }
    pangulu_cm_sync();
    for(int i = 0; i < size; i++){
        if(rank == i){
            printf(PANGULU_I_PERF_TABLE_ROW);
            fflush(stdout);
        }
        pangulu_cm_sync();
        usleep(10);
    }
#endif
    pangulu_cm_sync();
    pangulu_log_memory_usage();
#ifdef PANGULU_PERF
    pangulu_numeric_check(common, block_common, block_smatrix);
    pangulu_cm_sync();
#endif
}

void pangulu_gstrs(calculate_type *rhs, pangulu_gstrs_options *gstrs_options, void **pangulu_handle)
{
    pangulu_block_common *block_common = (*(pangulu_handle_t **)pangulu_handle)->block_common;
    pangulu_block_smatrix *block_smatrix = (*(pangulu_handle_t **)pangulu_handle)->block_smatrix;
    pangulu_common *common = (*(pangulu_handle_t **)pangulu_handle)->commmon;
    pangulu_int32_t size, rank;
    pangulu_cm_rank(&rank);
    pangulu_cm_size(&size);
    if (rank == 0)
    {
        if (gstrs_options == NULL)
        {
            printf(PANGULU_E_GSTRS_OPTION_IS_NULLPTR);
            exit(1);
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
        pangulu_reorder_vector_b_tran(block_smatrix->row_perm, block_smatrix->metis_perm, block_smatrix->row_scale, b_vector, answer_vector);
    }
    pangulu_sptrsv_preprocessing(
        block_common,
        block_smatrix,
        answer_vector);
    struct timeval time_start;
    double elapsed_time;
    pangulu_cm_sync();
    pangulu_time_start(&time_start);
    pangulu_solve(block_common, block_smatrix, answer_vector);
    pangulu_cm_sync();
    elapsed_time = pangulu_time_stop(&time_start);
    if (rank == 0)
    {
        printf(PANGULU_I_TIME_SPTRSV);
        pangulu_reorder_vector_x_tran(block_smatrix, answer_vector, x_vector);
        for (int i = 0; i < common->n; i++)
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