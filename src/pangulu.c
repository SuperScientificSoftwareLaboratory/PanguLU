#include "pangulu_common.h"

#ifdef PANGULU_PERF
pangulu_stat_t global_stat;
#endif

int pangulu_gpu_kernel_warp_per_block;
int pangulu_gpu_data_move_warp_per_block;
int pangulu_gpu_shared_mem_size;

static void pangulu_init_impl(
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
    memset(common, 0, sizeof(*common));
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
    omp_set_num_threads(common->omp_thread);
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
        common->reordering_nthread
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
        common->omp_thread);
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
    (*(pangulu_handle_t **)pangulu_handle)->status = PANGULU_SUCCESS;
    (*(pangulu_handle_t **)pangulu_handle)->factorized = 0;
}

static void pangulu_gstrf_impl(pangulu_gstrf_options *gstrf_options, void **pangulu_handle)
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

static void pangulu_gstrs_impl(calculate_type *rhs, pangulu_gstrs_options *gstrs_options, void **pangulu_handle)
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
    if (!pangulu_handle || !*pangulu_handle) return;
    pangulu_block_common *block_common = (*(pangulu_handle_t **)pangulu_handle)->block_common;
    pangulu_block_smatrix *block_smatrix = (*(pangulu_handle_t **)pangulu_handle)->block_smatrix;
    pangulu_common *common = (*(pangulu_handle_t **)pangulu_handle)->commmon;

    pangulu_cpu_workspace_destroy(&block_smatrix->cpu_workspace);
    pangulu_aggregate_finalize();
#ifdef GPU_OPEN
    pangulu_platform_0201000_finalize();
#endif
    pangulu_destroy(block_common, block_smatrix);

    pangulu_free(__FILE__, __LINE__, block_common);
    pangulu_free(__FILE__, __LINE__, block_smatrix);
    pangulu_free(__FILE__, __LINE__, common);
    pangulu_free(__FILE__, __LINE__, *(pangulu_handle_t **)pangulu_handle);
    *pangulu_handle = NULL;
}
static pangulu_status_t pangulu_collect_status(pangulu_status_t local)
{
    int initialized = 0, finalized = 0, value = local, global;
    MPI_Initialized(&initialized);
    if (!initialized) return PANGULU_INVALID_STATE;
    MPI_Finalized(&finalized);
    if (finalized) return PANGULU_INVALID_STATE;
    MPI_Allreduce(&value, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return (pangulu_status_t)global;
}

const char *pangulu_status_string(pangulu_status_t status)
{
    switch (status) {
    case PANGULU_SUCCESS: return "success";
    case PANGULU_INVALID_ARGUMENT: return "invalid argument";
    case PANGULU_INVALID_STATE: return "invalid solver state";
    case PANGULU_ZERO_PIVOT: return "zero pivot: matching or pivoting may be required";
    case PANGULU_NUMERICAL_FAILURE: return "nonfinite value or numerically unsafe pivot";
    default: return "unknown status";
    }
}

pangulu_status_t pangulu_get_status(void *handle)
{
    return handle ? ((pangulu_handle_t *)handle)->status : PANGULU_INVALID_STATE;
}

static pangulu_status_t pangulu_check_factors(pangulu_handle_t *handle)
{
    pangulu_block_smatrix *matrix = handle->block_smatrix;
    pangulu_block_common *common = handle->block_common;
    pangulu_status_t status = PANGULU_SUCCESS;
    for (pangulu_exblock_idx block = 0; block < common->block_length; ++block) {
        /* Received diagonal copies may already have been recycled. Only the
         * owning rank retains both factors; reduce its result collectively. */
        if ((block % common->p) * common->q + block % common->q != handle->commmon->rank)
            continue;
        if (!matrix->diag_uniaddr[block]) continue;
        pangulu_storage_slot_t *upper = pangulu_storage_get_diag(common->block_length,
            matrix->storage, matrix->diag_uniaddr[block]);
        if (!upper->is_upper) upper = upper->related_block;
        if (upper->numerical_status > status) status = upper->numerical_status;
        for (pangulu_inblock_idx row = 0; row < common->nb; ++row) {
            if ((pangulu_uint64_t)block * common->nb + row >= common->n) break;
            pangulu_inblock_ptr begin = upper->columnpointer[row], end = upper->columnpointer[row + 1];
            pangulu_status_t row_status = PANGULU_ZERO_PIVOT;
            if (begin != end && upper->rowindex[begin] == row)
                row_status = pangulu_pivot_status(upper->value[begin], upper->value, begin, end);
            if (row_status > status) status = row_status;
        }
    }
    return pangulu_collect_status(status);
}

pangulu_status_t pangulu_init_checked(sparse_index_t n, sparse_pointer_t nnz,
    sparse_pointer_t *colptr, sparse_index_t *rowidx, sparse_value_t *values,
    pangulu_init_options *options, void **handle)
{
    pangulu_status_t status = PANGULU_SUCCESS;
    int initialized = 0, finalized = 0, rank = 0;
    MPI_Initialized(&initialized);
    if (!initialized) return PANGULU_INVALID_STATE;
    MPI_Finalized(&finalized);
    if (finalized) return PANGULU_INVALID_STATE;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (!handle || !options) status = PANGULU_INVALID_ARGUMENT;
    else if (*handle) status = PANGULU_INVALID_STATE;
    else if (options->sizeof_value != sizeof(calculate_type) ||
             options->is_complex_matrix != (sizeof(calculate_type) != sizeof(calculate_real_type)) ||
             options->nthread < 0 || options->nb == 0 || options->nb > UINT16_MAX ||
             options->gpu_kernel_warp_per_block < 0 || options->gpu_data_move_warp_per_block < 0 ||
             !isfinite(options->mpi_recv_buffer_level) || options->mpi_recv_buffer_level < 0)
        status = PANGULU_INVALID_ARGUMENT;
    if (rank == 0 && status == PANGULU_SUCCESS) {
        unsigned block_size = options->nb < 0 ? 256 : (unsigned)options->nb;
        if (!n || !colptr || (nnz && (!rowidx || !values)) ||
            (pangulu_uint64_t)n + block_size - 1 > UINT32_MAX || colptr[0] || colptr[n] != nnz)
            status = PANGULU_INVALID_ARGUMENT;
        else {
          sparse_index_t *seen = pangulu_malloc(__FILE__, __LINE__, sizeof(*seen) * (size_t)n);
          memset(seen, 0, sizeof(*seen) * (size_t)n);
          for (sparse_index_t col = 0; col < n && status == PANGULU_SUCCESS; ++col) {
            int diagonal = 0;
            if (colptr[col] > colptr[col + 1] || colptr[col + 1] > nnz) {
                status = PANGULU_INVALID_ARGUMENT; break;
            }
            for (sparse_pointer_t k = colptr[col]; k < colptr[col + 1]; ++k) {
                if (rowidx[k] >= n || !pangulu_value_finite(values[k])) {
                    status = PANGULU_INVALID_ARGUMENT; break;
                }
                if (seen[rowidx[k]] == col + 1) {status = PANGULU_INVALID_ARGUMENT; break;}
                seen[rowidx[k]] = col + 1;
                if (rowidx[k] == col && pangulu_value_abs(values[k]) != 0) diagonal = 1;
            }
#ifndef PANGULU_MC64
            /* This build has no matching step or dynamic row pivoting. A zero
             * input diagonal is a breakdown, not a proof of singularity. */
            if (!diagonal && status == PANGULU_SUCCESS) status = PANGULU_ZERO_PIVOT;
#endif
          }
          pangulu_free(__FILE__, __LINE__, seen);
        }
    }
    status = pangulu_collect_status(status);
    if (status != PANGULU_SUCCESS) return status;
    /* Every rank must select the same precision-independent block/thread
     * options; root controls the matrix size through the existing broadcast. */
    int configuration[4] = {options->nb, options->nthread,
        options->gpu_kernel_warp_per_block, options->gpu_data_move_warp_per_block};
    int minimum[4], maximum[4];
    MPI_Allreduce(configuration, minimum, 4, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(configuration, maximum, 4, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    for (int k = 0; k < 4; ++k)
        if (minimum[k] != maximum[k]) return PANGULU_INVALID_ARGUMENT;
    pangulu_init_impl(n, nnz, colptr, rowidx, values, options, handle);
    return PANGULU_SUCCESS;
}

pangulu_status_t pangulu_gstrf_checked(pangulu_gstrf_options *options, void **handle)
{
    pangulu_status_t status = !options ? PANGULU_INVALID_ARGUMENT :
        (!handle || !*handle ? PANGULU_INVALID_STATE : PANGULU_SUCCESS);
    if (status == PANGULU_SUCCESS && ((pangulu_handle_t *)*handle)->factorized != 0)
        status = PANGULU_INVALID_STATE;
    status = pangulu_collect_status(status);
    if (status != PANGULU_SUCCESS) return status;
    omp_set_num_threads(((pangulu_handle_t *)*handle)->commmon->omp_thread);
    pangulu_cpu_workspace_select(&((pangulu_handle_t *)*handle)->block_smatrix->cpu_workspace);
#ifdef GPU_OPEN
    /* Temporary CUDA workspace sizes depend on nb. Factors are owned by the
     * handle's storage, so retiring this cache cannot invalidate another handle. */
    pangulu_platform_0201000_finalize();
#endif
    pangulu_gstrf_impl(options, handle);
    pangulu_handle_t *state = *handle;
    state->status = pangulu_check_factors(state);
    state->factorized = state->status == PANGULU_SUCCESS ? 1 : -1;
    return state->status;
}

pangulu_status_t pangulu_gstrs_checked(sparse_value_t *rhs, pangulu_gstrs_options *options, void **handle)
{
    pangulu_handle_t *state = handle ? *handle : NULL;
    pangulu_status_t status = !options ? PANGULU_INVALID_ARGUMENT :
        (!state ? PANGULU_INVALID_STATE : PANGULU_SUCCESS);
    if (state && status == PANGULU_SUCCESS) {
        if (state->factorized < 0) status = state->status;
        else if (!state->factorized) status = PANGULU_INVALID_STATE;
        else if (state->commmon->rank == 0) {
            if (!rhs) status = PANGULU_INVALID_ARGUMENT;
            else for (sparse_index_t i = 0; i < state->commmon->n; ++i)
                if (!pangulu_value_finite(rhs[i])) {status = PANGULU_INVALID_ARGUMENT; break;}
        }
    }
    status = pangulu_collect_status(status);
    if (status != PANGULU_SUCCESS) return status;
    calculate_type *saved_rhs = NULL;
    if (state->commmon->rank == 0) {
        saved_rhs = pangulu_malloc(__FILE__, __LINE__, sizeof(*saved_rhs) * state->commmon->n);
        memcpy(saved_rhs, rhs, sizeof(*saved_rhs) * state->commmon->n);
    }
    pangulu_gstrs_impl(rhs, options, handle);
    status = pangulu_check_factors(state);
    if (state->commmon->rank == 0)
        for (sparse_index_t i = 0; i < state->commmon->n; ++i)
            if (!pangulu_value_finite(rhs[i])) status = PANGULU_NUMERICAL_FAILURE;
    state->status = pangulu_collect_status(status);
    if (saved_rhs && state->status != PANGULU_SUCCESS)
        memcpy(rhs, saved_rhs, sizeof(*saved_rhs) * state->commmon->n);
    pangulu_free(__FILE__, __LINE__, saved_rhs);
    return state->status;
}

pangulu_status_t pangulu_gssv_checked(sparse_value_t *rhs, pangulu_gstrf_options *factor_options,
    pangulu_gstrs_options *solve_options, void **handle)
{
    pangulu_status_t status = pangulu_gstrf_checked(factor_options, handle);
    return status == PANGULU_SUCCESS ? pangulu_gstrs_checked(rhs, solve_options, handle) : status;
}

static void pangulu_report_status(pangulu_status_t status)
{
    if (status != PANGULU_SUCCESS)
        fprintf(stderr, "[PanguLU Error] %s\n", pangulu_status_string(status));
}

void pangulu_init(sparse_index_t n, sparse_pointer_t nnz, sparse_pointer_t *colptr,
    sparse_index_t *rowidx, sparse_value_t *values, pangulu_init_options *options, void **handle)
{
    if (handle) *handle = NULL; /* Legacy callers need not initialize the slot. */
    pangulu_report_status(pangulu_init_checked(n, nnz, colptr, rowidx, values, options, handle));
}

void pangulu_gstrf(pangulu_gstrf_options *options, void **handle)
{
    pangulu_report_status(pangulu_gstrf_checked(options, handle));
}

void pangulu_gstrs(sparse_value_t *rhs, pangulu_gstrs_options *options, void **handle)
{
    pangulu_report_status(pangulu_gstrs_checked(rhs, options, handle));
}
