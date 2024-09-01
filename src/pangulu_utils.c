#include "pangulu_common.h"

void pangulu_bip_init(pangulu_block_info_pool **BIP, pangulu_int64_t map_index_not_included)
{ // BIP : block info pool
    if(!BIP){
        printf(PANGULU_E_BIP_PTR_INVALID);
        pangulu_exit(1);
    }
    if(!(*BIP)){
        *BIP = (pangulu_block_info_pool*)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_block_info_pool));
    }

    pangulu_block_info_pool* pool = *BIP;
    pool->capacity = PANGULU_BIP_INITIAL_LEN;
    pool->length = 0;
    pool->index_upper_bound = map_index_not_included;
    pool->block_map = (pangulu_int64_t*)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * PANGULU_BIP_MAP_LENGTH(map_index_not_included));
    for(pangulu_int64_t i=0; i<PANGULU_BIP_MAP_LENGTH(map_index_not_included);i++){
        pool->block_map[i] = -1;
    }
    pool->data = NULL;
    pool->data = (pangulu_block_info*)pangulu_realloc(__FILE__, __LINE__, pool->data, sizeof(pangulu_block_info)*pool->capacity);
}

void pangulu_bip_destroy(pangulu_block_info_pool **BIP)
{
    if(!BIP){
        return;
    }
    if(!(*BIP)){
        return;
    }
    pangulu_free(__FILE__, __LINE__, (*BIP)->block_map);
    pangulu_free(__FILE__, __LINE__, (*BIP)->data);
    pangulu_free(__FILE__, __LINE__, *BIP);
    *BIP = NULL;
}

const pangulu_block_info *pangulu_bip_get(pangulu_int64_t index, pangulu_block_info_pool *BIP)
{
    if(!BIP){
        printf(PANGULU_E_BIP_INVALID);
        pangulu_exit(1);
    }
    if(index >= BIP->index_upper_bound){
        printf(PANGULU_E_BIP_OUT_OF_RANGE);
        pangulu_exit(1);
    }
    if(BIP->block_map[index/PANGULU_BIP_SIBLING_LEN]==-1){
        static const pangulu_block_info defaults = {
            .block_smatrix_nnza_num = 0,
            .sum_flag_block_num = 0,
            .mapper_a = -1,
            .tmp_save_block_num = -1,
            .task_flag_id = 0,
            .mapper_mpi = -1,
            // .index = -1,
            .mapper_lu = -1};
        return &defaults;
    }else{
        return &(BIP->data[BIP->block_map[index/PANGULU_BIP_SIBLING_LEN]*PANGULU_BIP_SIBLING_LEN+(index%PANGULU_BIP_SIBLING_LEN)]);
    }
}

// pthread_mutex_t pangulu_BIP_append_mutex = PTHREAD_MUTEX_INITIALIZER;
pangulu_block_info *pangulu_bip_set(pangulu_int64_t index, pangulu_block_info_pool *BIP)
{
    if(!BIP){
        printf(PANGULU_E_BIP_INVALID);
        pangulu_exit(1);
    }
    if(index >= BIP->index_upper_bound){
        printf(PANGULU_E_BIP_OUT_OF_RANGE);
        pangulu_exit(1);
    }
    if(BIP->block_map[index/PANGULU_BIP_SIBLING_LEN]!=-1){
        return &(BIP->data[BIP->block_map[index/PANGULU_BIP_SIBLING_LEN]*PANGULU_BIP_SIBLING_LEN+(index%PANGULU_BIP_SIBLING_LEN)]);
    }

    static const pangulu_block_info defaults = {
        .block_smatrix_nnza_num = 0,
        .sum_flag_block_num = 0,
        .mapper_a = -1,
        .tmp_save_block_num = -1,
        .task_flag_id = 0,
        .mapper_mpi = -1,
        // .index = -1,
        .mapper_lu = -1};

    // pthread_mutex_lock(&pangulu_BIP_append_mutex);
    if(BIP->length + PANGULU_BIP_SIBLING_LEN > BIP->capacity){
        float increase_speed = PANGULU_BIP_INCREASE_SPEED;
        while(BIP->capacity * increase_speed <= BIP->capacity){
            increase_speed += 1.0;
            printf(PANGULU_W_BIP_INCREASE_SPEED_TOO_SMALL);
        }
        BIP->capacity = (((BIP->capacity * increase_speed) + PANGULU_BIP_SIBLING_LEN - 1) / PANGULU_BIP_SIBLING_LEN) * PANGULU_BIP_SIBLING_LEN;
        BIP->data = (pangulu_block_info*)pangulu_realloc(__FILE__, __LINE__, BIP->data, sizeof(pangulu_block_info)*BIP->capacity);
    }
    BIP->block_map[index/PANGULU_BIP_SIBLING_LEN] = BIP->length/PANGULU_BIP_SIBLING_LEN;
    BIP->length+=PANGULU_BIP_SIBLING_LEN;
    pangulu_block_info* new_info = &(BIP->data[BIP->block_map[index/PANGULU_BIP_SIBLING_LEN]*PANGULU_BIP_SIBLING_LEN+(index%PANGULU_BIP_SIBLING_LEN)]);
    pangulu_block_info* new_chunk_head = &(BIP->data[BIP->block_map[index/PANGULU_BIP_SIBLING_LEN]*PANGULU_BIP_SIBLING_LEN]);
    for(pangulu_int32_t i=0;i<PANGULU_BIP_SIBLING_LEN;i++){
        memcpy(new_chunk_head + i, &defaults, sizeof(pangulu_block_info));
        // new_chunk_head[i].index = (index/PANGULU_BIP_SIBLING_LEN) + i;
    }
    return new_info;
}

void bind_to_core(int core)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);

    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0)
    {
        perror("pthread_setaffinity_np error");
    }
}

// bug fixed.
void mpi_barrier_asym(MPI_Comm comm, int wake_rank, unsigned long long awake_interval_us)
{
    int sum_rank_size = 0;
    MPI_Comm_size(comm, &sum_rank_size);
    if (rank == wake_rank)
    {
        for (int i = 0; i < sum_rank_size; i++)
        {
            if (i != wake_rank)
            {
                MPI_Send(&sum_rank_size, 1, MPI_INT, i, 0xCAFE, comm);
            }
        }
    }
    else
    {
        int mpi_buf_int;
        int mpi_flag;
        MPI_Status mpi_stat;
        while (1)
        {
            mpi_flag = 0;
            MPI_Iprobe(wake_rank, 0xCAFE, comm, &mpi_flag, &mpi_stat);
            if (mpi_flag != 0 && mpi_stat.MPI_TAG == 0xCAFE)
            {
                MPI_Recv(&mpi_buf_int, 1, MPI_INT, wake_rank, 0xCAFE, comm, &mpi_stat);
                if (mpi_buf_int == sum_rank_size)
                {
                    break;
                }
                else
                {
                    printf(PANGULU_E_ASYM);
                    pangulu_exit(1);
                }
            }
            usleep(awake_interval_us);
        }
    }
}

double pangulu_fabs(double _Complex x){
    return sqrt(__real__(x)*__real__(x) + __imag__(x)*__imag__(x));
}

double _Complex pangulu_log(double _Complex x){
    double _Complex y;
    __real__(y) = log(__real__(x)*__real__(x) + __imag__(x)*__imag__(x))/2;
    __imag__(y) = atan(__imag__(x)/__real__(x));
    return y;
}

double _Complex pangulu_sqrt(double _Complex x){
    double _Complex y;
    __real__(y) = sqrt(pangulu_fabs(x) + __real__(x))/sqrt(2);
    __imag__(y) = (sqrt(pangulu_fabs(x) - __real__(x))/sqrt(2))*(__imag__(x)>0?1:__imag__(x)==0?0:-1);
    return y;
}

void exclusive_scan_1(pangulu_int64_t *input, int length)
{
    if (length == 0 || length == 1)
        return;

    pangulu_int64_t old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i - 1];
        old_val = new_val;
    }
}

void exclusive_scan_2(pangulu_int32_t *input, int length)
{
    if (length == 0 || length == 1)
        return;

    pangulu_int32_t old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i - 1];
        old_val = new_val;
    }
}

void exclusive_scan_3(unsigned int *input, int length)
{
    if (length == 0 || length == 1)
        return;

    unsigned int old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i - 1];
        old_val = new_val;
    }
}

void swap_key(pangulu_int64_t *a, pangulu_int64_t *b)
{
    // if(a==NULL || b==NULL){
    //     return;
    // }
    pangulu_int64_t tmp = *a;
    *a = *b;
    *b = tmp;
}

void swap_val(calculate_type *a, calculate_type *b)
{
    // if(a==NULL || b==NULL){
    //     return;
    // }
    calculate_type tmp = *a;
    *a = *b;
    *b = tmp;
}

int binarylowerbound(const pangulu_int64_t *arr, int len, pangulu_int64_t value)
{
    int left = 0;
    int right = len;
    int mid;
    while (left < right)
    {
        mid = (left + right) >> 1;
        // value <= arr[mid] ? (right = mid) : (left = mid + 1);
        value < arr[mid] ? (right = mid) : (left = mid + 1);
    }
    return left;
}

pangulu_int64_t binarysearch(const int *arr, pangulu_int64_t left, pangulu_int64_t right, pangulu_int64_t target)
{
    pangulu_int64_t low = left;
    pangulu_int64_t high = right;
    pangulu_int64_t mid;
    while (low <= high)
    {
        mid = (low + high) / 2;
        if (target < arr[mid])
            high = mid - 1;
        else if (target > arr[mid])
            low = mid + 1;
        else
            return mid;
    }
    return -1;
}

pangulu_int64_t binarysearch_inblock_idx(pangulu_int64_t begin, pangulu_int64_t end, pangulu_int64_t aim, pangulu_inblock_idx *array)
{
    end = end - 1;
    pangulu_int64_t middle = (end + begin) / 2;
    pangulu_int64_t left = begin;
    pangulu_int64_t right = end;
    while (left <= right)
    {
        if (array[middle] > aim)
        {
            right = middle - 1;
        }
        else if (array[middle] < aim)
        {
            left = middle + 1;
        }
        else
        {
            return middle;
        }
        middle = (right + left) / 2;
    }
    return -1; // not find
}

void pangulu_get_common(pangulu_common *common,
                        pangulu_init_options *init_options, pangulu_int32_t size)
{
    common->p = 0;
    common->q = 0;
    common->sum_rank_size = size;
    common->omp_thread = 64;

    pangulu_int64_t tmp_p = sqrt(common->sum_rank_size);
    while (((common->sum_rank_size) % tmp_p) != 0)
    {
        tmp_p--;
    }

    common->p = tmp_p;
    common->q = common->sum_rank_size / tmp_p;
    if ((common->nb) == 0)
    {
        printf(PANGULU_E_NB_IS_ZERO);
        exit(4);
    }
}

void pangulu_init_pangulu_block_smatrix(pangulu_block_smatrix *block_smatrix)
{
    // reorder array
    block_smatrix->row_perm = NULL;
    block_smatrix->col_perm = NULL;
    block_smatrix->metis_perm = NULL;
    block_smatrix->row_scale = NULL;
    block_smatrix->col_scale = NULL;

    // symbolic
    block_smatrix->symbolic_rowpointer = NULL;
    block_smatrix->symbolic_columnindex = NULL;

    // LU
    block_smatrix->BIP = NULL;
    block_smatrix->block_smatrix_non_zero_vector_l = NULL;
    block_smatrix->block_smatrix_non_zero_vector_u = NULL;
    block_smatrix->big_pangulu_smatrix_value = NULL;
    block_smatrix->l_pangulu_smatrix_columnpointer = NULL;
    block_smatrix->l_pangulu_smatrix_rowindex = NULL;
    block_smatrix->l_smatrix_nzz = 0;
    block_smatrix->l_pangulu_smatrix_value = NULL;
    block_smatrix->u_pangulu_smatrix_rowpointer = NULL;
    block_smatrix->u_pangulu_smatrix_columnindex = NULL;
    block_smatrix->u_smatrix_nzz = 0;
    block_smatrix->u_pangulu_smatrix_value = NULL;
    block_smatrix->mapper_diagonal = NULL;
    block_smatrix->diagonal_smatrix_l = NULL;
    block_smatrix->diagonal_smatrix_u = NULL;
    block_smatrix->calculate_l = NULL;
    block_smatrix->calculate_u = NULL;
    block_smatrix->calculate_x = NULL;

    block_smatrix->task_level_length = 0;
    block_smatrix->task_level_num = NULL;
    block_smatrix->heap = NULL;
    block_smatrix->now_level_l_length = NULL;
    block_smatrix->now_level_u_length = NULL;
    block_smatrix->save_now_level_l = NULL;
    block_smatrix->save_now_level_u = NULL;
    block_smatrix->send_flag = NULL;
    block_smatrix->send_diagonal_flag_l = NULL;
    block_smatrix->send_diagonal_flag_u = NULL;
    block_smatrix->grid_process_id = NULL;
    block_smatrix->save_send_rank_flag = NULL;
    block_smatrix->receive_level_num = NULL;
    block_smatrix->save_tmp = NULL;

    block_smatrix->level_index = NULL;
    block_smatrix->level_index_reverse = NULL;
    block_smatrix->mapper_mpi_reverse = NULL;
    block_smatrix->mpi_level_num = NULL;

    block_smatrix->flag_save_l = NULL;
    block_smatrix->flag_save_u = NULL;
    block_smatrix->flag_dignon_l = NULL;
    block_smatrix->flag_dignon_u = NULL;

#ifdef OVERLAP
    block_smatrix->run_bsem1 = NULL;
    block_smatrix->run_bsem2 = NULL;

#endif

    // sptrsv
    block_smatrix->big_row_vector = NULL;
    block_smatrix->big_col_vector = NULL;
    block_smatrix->diagonal_flag = NULL;
    block_smatrix->l_row_task_nnz = NULL;
    block_smatrix->l_col_task_nnz = NULL;
    block_smatrix->u_row_task_nnz = NULL;
    block_smatrix->u_col_task_nnz = NULL;
    block_smatrix->sptrsv_heap = NULL;
    block_smatrix->save_vector = NULL;
    block_smatrix->l_send_flag = NULL;
    block_smatrix->u_send_flag = NULL;
    block_smatrix->l_sptrsv_task_columnpointer = NULL;
    block_smatrix->l_sptrsv_task_rowindex = NULL;
    block_smatrix->u_sptrsv_task_columnpointer = NULL;
    block_smatrix->u_sptrsv_task_rowindex = NULL;
}

void pangulu_init_pangulu_smatrix(pangulu_smatrix *s)
{
    s->value = NULL;
    s->value_csc = NULL;
    s->csr_to_csc_index = NULL;
    s->csc_to_csr_index = NULL;
    s->rowpointer = NULL;
    s->columnindex = NULL;
    s->columnpointer = NULL;
    s->rowindex = NULL;
    s->column = 0;
    s->row = 0;
    s->nnz = 0;

    s->nnzu = NULL;
    s->bin_rowpointer = NULL;
    s->bin_rowindex = NULL;
    s->zip_flag = 0;
    s->zip_id = 0;

#ifdef GPU_OPEN
    s->cuda_rowpointer = NULL;
    s->cuda_columnindex = NULL;
    s->cuda_value = NULL;
    s->cuda_nnzu = NULL;
    s->cuda_bin_rowpointer = NULL;
    s->cuda_bin_rowindex = NULL;
#else
    s->num_lev = 0;
    s->level_idx = NULL;
    s->level_size = NULL;

#endif
}

void pangulu_init_pangulu_origin_smatrix(pangulu_origin_smatrix *s)
{
    s->value = NULL;
    s->value_csc = NULL;
    s->csr_to_csc_index = NULL;
    s->csc_to_csr_index = NULL;
    s->rowpointer = NULL;
    s->columnindex = NULL;
    s->columnpointer = NULL;
    s->rowindex = NULL;
    s->column = 0;
    s->row = 0;
    s->nnz = 0;

    s->nnzu = NULL;
    s->bin_rowpointer = NULL;
    s->bin_rowindex = NULL;
    s->zip_flag = 0;
    s->zip_id = 0;

#ifdef GPU_OPEN
    // s->CUDA_rowpointer = NULL;
    // s->CUDA_columnindex = NULL;
    // s->CUDA_value = NULL;
    // s->CUDA_nnzU = NULL;
    // s->CUDA_bin_rowpointer = NULL;
    // s->CUDA_bin_rowindex = NULL;
#else
    s->num_lev = 0;
    s->level_idx = NULL;
    s->level_size = NULL;

#endif
}

void pangulu_read_pangulu_origin_smatrix(pangulu_origin_smatrix *s, int wcs_n, long long wcs_nnz, pangulu_exblock_ptr *csr_rowptr, pangulu_exblock_idx *csr_colidx, calculate_type *csr_value)
{
    s->row = wcs_n;
    s->column = wcs_n;
    s->rowpointer = csr_rowptr;
    s->columnindex = csr_colidx;
    s->nnz = wcs_nnz;
    s->value = csr_value;
}

void pangulu_time_start(pangulu_common *common)
{
    gettimeofday(&(common->start_time), NULL);
}

void pangulu_time_stop(pangulu_common *common)
{
    gettimeofday(&(common->stop_time), NULL);
}

void pangulu_memcpy_zero_pangulu_smatrix_csc_value(pangulu_smatrix *s)
{
    for (pangulu_int64_t i = 0; i < s->nnz; i++)
    {
        s->value_csc[i] = 0.0;
    }
}
void pangulu_memcpy_zero_pangulu_smatrix_csr_value(pangulu_smatrix *s)
{
    for (pangulu_int64_t i = 0; i < s->nnz; i++)
    {
        s->value[i] = 0.0;
    }
}
void pangulu_display_pangulu_smatrix_csc(pangulu_smatrix *s)
{
    printf("------------------\n\n\n");
    if (s == NULL)
    {
        printf("\nno i am null\n");
        return;
    }
    printf("row is " FMT_PANGULU_INBLOCK_IDX " column is " FMT_PANGULU_INBLOCK_IDX "\n", s->row, s->column);
    printf("columnpointer:");
    for (pangulu_int64_t i = 0; i < s->row + 1; i++)
    {
        printf("%u ", s->columnpointer[i]);
    }
    printf("\n");
    printf("rowindex:\n");
    for (pangulu_int64_t i = 0; i < s->row; i++)
    {
        for (pangulu_int64_t j = s->columnpointer[i]; j < s->columnpointer[i + 1]; j++)
        {
            printf("%hu ", s->rowindex[j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("value_csc:\n");
    for (pangulu_int64_t i = 0; i < s->row; i++)
    {
        for (pangulu_int64_t j = s->columnpointer[i]; j < s->columnpointer[i + 1]; j++)
        {
            printf("%lf ", s->value_csc[j]);
        }
        printf("\n");
    }
    printf("\n\n\n--------------------");
}

double pangulu_get_spend_time(pangulu_common *common)
{
    double time = (common->stop_time.tv_sec - common->start_time.tv_sec) * 1000.0 + (common->stop_time.tv_usec - common->start_time.tv_usec) / 1000.0;
    return time / 1000.0;
}

void pangulu_transport_pangulu_smatrix_csc_to_csr(pangulu_smatrix *s)
{
    for (pangulu_int64_t i = 0; i < s->nnz; i++)
    {
        pangulu_int64_t index = s->csc_to_csr_index[i];
        s->value[index] = s->value_csc[i];
    }
}
void pangulu_transport_pangulu_smatrix_csr_to_csc(pangulu_smatrix *s)
{
    for (pangulu_int64_t i = 0; i < s->nnz; i++)
    {
        pangulu_int64_t index = s->csc_to_csr_index[i];
        s->value_csc[i] = s->value[index];
    }
}

void pangulu_pangulu_smatrix_memcpy_rowpointer_csr(pangulu_smatrix *s, pangulu_smatrix *copy_S)
{
    pangulu_int64_t n = s->row;
    for (pangulu_int64_t i = 0; i < (n + 1); i++)
    {
        s->rowpointer[i] = copy_S->rowpointer[i];
    }
}

void pangulu_pangulu_smatrix_memcpy_value_csr(pangulu_smatrix *s, pangulu_smatrix *copy_S)
{
    for (pangulu_int64_t i = 0; i < s->nnz; i++)
    {
        s->value[i] = copy_S->value[i];
    }
}

void pangulu_pangulu_smatrix_memcpy_value_csr_copy_length(pangulu_smatrix *s, pangulu_smatrix *copy_S)
{
    memcpy(s->value, copy_S->value, sizeof(calculate_type) * copy_S->nnz);
}

void pangulu_pangulu_smatrix_memcpy_value_csc_copy_length(pangulu_smatrix *s, pangulu_smatrix *copy_S)
{
    memcpy(s->value_csc, copy_S->value_csc, sizeof(calculate_type) * copy_S->nnz);
}

void pangulu_pangulu_smatrix_memcpy_struct_csc(pangulu_smatrix *s, pangulu_smatrix *copy_S)
{
    s->column = copy_S->column;
    s->row = copy_S->row;
    s->nnz = copy_S->nnz;
    for (pangulu_int64_t i = 0; i < s->column + 1; i++)
    {
        s->columnpointer[i] = copy_S->columnpointer[i];
    }
    for (pangulu_int64_t i = 0; i < s->nnz; i++)
    {
        s->rowindex[i] = copy_S->rowindex[i];
    }
}

void pangulu_pangulu_smatrix_memcpy_columnpointer_csc(pangulu_smatrix *s, pangulu_smatrix *copy_S)
{
    memcpy(s->columnpointer, copy_S->columnpointer, sizeof(pangulu_inblock_ptr) * (copy_S->row + 1));
}

void pangulu_pangulu_smatrix_memcpy_value_csc(pangulu_smatrix *s, pangulu_smatrix *copy_S)
{
    for (pangulu_int64_t i = 0; i < s->nnz; i++)
    {

        s->value_csc[i] = copy_S->value_csc[i];
    }
}

void pangulu_pangulu_smatrix_memcpy_complete_csc(pangulu_smatrix *s, pangulu_smatrix *copy_S)
{
    pangulu_pangulu_smatrix_memcpy_struct_csc(s, copy_S);
    pangulu_pangulu_smatrix_memcpy_value_csc(s, copy_S);
}

void pangulu_pangulu_smatrix_multiple_pangulu_vector_csr(pangulu_smatrix *a,
                                                         pangulu_vector *x,
                                                         pangulu_vector *b)
{
    pangulu_int64_t n = a->row;
    calculate_type *X_value = x->value;
    calculate_type *B_value = b->value;
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        B_value[i] = 0.0;
        for (pangulu_int64_t j = a->rowpointer[i]; j < a->rowpointer[i + 1]; j++)
        {
            pangulu_int64_t col = a->columnindex[j];
            B_value[i] += a->value[j] * X_value[col];
        }
    }
}

void pangulu_origin_smatrix_multiple_pangulu_vector_csr(pangulu_origin_smatrix *a,
                                                        pangulu_vector *x,
                                                        pangulu_vector *b)
{
    pangulu_exblock_idx n = a->row;
    calculate_type *X_value = x->value;
    calculate_type *B_value = b->value;
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        B_value[i] = 0.0;
        for (pangulu_exblock_ptr j = a->rowpointer[i]; j < a->rowpointer[i + 1]; j++)
        {
            pangulu_exblock_idx col = a->columnindex[j];
            B_value[i] += a->value[j] * X_value[col];
        }
    }
}

void pangulu_pangulu_smatrix_multiple_pangulu_vector(pangulu_smatrix *a,
                                                     pangulu_vector *x,
                                                     pangulu_vector *b)
{
    pangulu_int64_t n = a->row;
    calculate_type *X_value = x->value;
    calculate_type *B_value = b->value;
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        B_value[i] = 0.0;
    }
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        for (pangulu_int64_t j = a->rowpointer[i]; j < a->rowpointer[i + 1]; j++)
        {
            pangulu_int64_t row = a->columnindex[j];
            B_value[row] += a->value[j] * X_value[i];
        }
    }
}

void pangulu_pangulu_smatrix_multiply_block_pangulu_vector_csr(pangulu_smatrix *a,
                                                               calculate_type *x,
                                                               calculate_type *b)
{
    pangulu_int64_t n = a->row;
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        for (pangulu_int64_t j = a->rowpointer[i]; j < a->rowpointer[i + 1]; j++)
        {
            pangulu_int64_t col = a->columnindex[j];
            b[i] += a->value[j] * x[col];
        }
    }
}

void pangulu_pangulu_smatrix_multiple_pangulu_vector_csc(pangulu_smatrix *a,
                                                         pangulu_vector *x,
                                                         pangulu_vector *b)
{
    pangulu_int64_t n = a->column;
    calculate_type *X_value = x->value;
    calculate_type *B_value = b->value;
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        for (pangulu_int64_t j = a->columnpointer[i]; j < a->columnpointer[i + 1]; j++)
        {
            pangulu_int64_t row = a->rowindex[j];
            B_value[row] += a->value_csc[j] * X_value[i];
        }
    }
}

void pangulu_pangulu_smatrix_multiply_block_pangulu_vector_csc(pangulu_smatrix *a,
                                                               calculate_type *x,
                                                               calculate_type *b)
{
    pangulu_inblock_idx n = a->column;
    for (pangulu_inblock_idx i = 0; i < n; i++)
    {
        for (pangulu_inblock_ptr j = a->columnpointer[i]; j < a->columnpointer[i + 1]; j++)
        {
            pangulu_inblock_idx row = a->rowindex[j];
            b[row] += a->value_csc[j] * x[i];
        }
    }
}

void pangulu_get_init_value_pangulu_vector(pangulu_vector *x, pangulu_int64_t n)
{
    x->value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        // x->value[i] = (calculate_type)i;
        x->value[i] = 2.0;
    }
    x->row = n;
}

void pangulu_init_pangulu_vector(pangulu_vector *b, pangulu_int64_t n)
{
    b->value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        b->value[i] = (calculate_type)0.0;
    }
    b->row = n;
}

void pangulu_zero_pangulu_vector(pangulu_vector *v)
{
    for (int i = 0; i < v->row; i++)
    {

        v->value[i] = 0.0;
    }
}

void pangulu_add_diagonal_element(pangulu_origin_smatrix *s)
{
    pangulu_int64_t diagonal_add = 0;
    pangulu_exblock_idx n = s->row;
    pangulu_exblock_ptr *new_rowpointer = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 5));
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        char flag = 0;
        for (pangulu_exblock_ptr j = s->rowpointer[i]; j < s->rowpointer[i + 1]; j++)
        {
            if (s->columnindex[j] == i)
            {
                flag = 1;
                break;
            }
        }
        new_rowpointer[i] = s->rowpointer[i] + diagonal_add;
        diagonal_add += (!flag);
    }
    new_rowpointer[n] = s->rowpointer[n] + diagonal_add;

    pangulu_exblock_idx *new_columnindex = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * new_rowpointer[n]);
    calculate_type *new_value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * new_rowpointer[n]);

    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        if ((new_rowpointer[i + 1] - new_rowpointer[i]) == (s->rowpointer[i + 1] - s->rowpointer[i]))
        {
            for (pangulu_exblock_ptr j = new_rowpointer[i], k = s->rowpointer[i]; j < new_rowpointer[i + 1]; j++, k++)
            {
                new_columnindex[j] = s->columnindex[k];
                new_value[j] = s->value[k];
            }
        }
        else
        {
            char flag = 0;
            for (pangulu_exblock_ptr j = new_rowpointer[i], k = s->rowpointer[i]; k < s->rowpointer[i + 1]; j++, k++)
            {
                if (s->columnindex[k] < i)
                {
                    new_columnindex[j] = s->columnindex[k];
                    new_value[j] = s->value[k];
                }
                else if (s->columnindex[k] > i)
                {
                    if (flag == 0)
                    {
                        new_columnindex[j] = i;
                        new_value[j] = ZERO_ELEMENT;
                        k--;
                        flag = 1;
                    }
                    else
                    {
                        new_columnindex[j] = s->columnindex[k];
                        new_value[j] = s->value[k];
                    }
                }
                else
                {
                    printf(PANGULU_E_ADD_DIA);
                    pangulu_exit(1);
                }
            }
            if (flag == 0)
            {
                new_columnindex[new_rowpointer[i + 1] - 1] = i;
                new_value[new_rowpointer[i + 1] - 1] = ZERO_ELEMENT;
            }
        }
    }

    pangulu_free(__FILE__, __LINE__, s->rowpointer);
    pangulu_free(__FILE__, __LINE__, s->columnindex);
    pangulu_free(__FILE__, __LINE__, s->value);
    s->rowpointer = new_rowpointer;
    s->columnindex = new_columnindex;
    s->value = new_value;
    s->nnz = new_rowpointer[n];
}

void pangulu_send_pangulu_vector_value(pangulu_vector *s,
                                       pangulu_int64_t send_id, pangulu_int64_t signal, pangulu_int64_t vector_length)
{
    MPI_Send(s->value, vector_length, MPI_VAL_TYPE, send_id, signal, MPI_COMM_WORLD);
}

void pangulu_isend_pangulu_vector_value(pangulu_vector *s,
                                        int send_id, int signal, int vector_length)
{
    MPI_Request req;
    MPI_Isend(s->value, vector_length, MPI_VAL_TYPE, send_id, signal, MPI_COMM_WORLD, &req);
}

void pangulu_recv_pangulu_vector_value(pangulu_vector *s, pangulu_int64_t receive_id, pangulu_int64_t signal, pangulu_int64_t vector_length)
{
    MPI_Status status;
    MPI_Recv(s->value, vector_length, MPI_VAL_TYPE, receive_id, signal, MPI_COMM_WORLD, &status);
}

void pangulu_init_vector_int(pangulu_int64_t *vector, pangulu_int64_t length)
{
    for (pangulu_int64_t i = 0; i < length; i++)
    {
        vector[i] = 0;
    }
}

pangulu_int64_t pangulu_choose_pivot(pangulu_int64_t i, pangulu_int64_t j)
{
    return (i + j) / 2;
}

void pangulu_swap_int(pangulu_int64_t *a, pangulu_int64_t *b)
{
    pangulu_int64_t tmp = *a;
    *a = *b;
    *b = tmp;
}

void pangulu_quicksort_keyval(pangulu_int64_t *key, pangulu_int64_t *val, pangulu_int64_t start, pangulu_int64_t end)
{
    pangulu_int64_t pivot;
    pangulu_int64_t i, j, k;

    if (start < end)
    {
        k = pangulu_choose_pivot(start, end);
        pangulu_swap_int(&key[start], &key[k]);
        pangulu_swap_int(&val[start], &val[k]);
        pivot = key[start];

        i = start + 1;
        j = end;
        while (i <= j)
        {
            while ((i <= end) && (key[i] <= pivot))
                i++;
            while ((j >= start) && (key[j] > pivot))
                j--;
            if (i < j)
            {
                pangulu_swap_int(&key[i], &key[j]);
                pangulu_swap_int(&val[i], &val[j]);
            }
        }

        // swap two elements
        pangulu_swap_int(&key[start], &key[j]);
        pangulu_swap_int(&val[start], &val[j]);

        // recursively sort the lesser key
        pangulu_quicksort_keyval(key, val, start, j - 1);
        pangulu_quicksort_keyval(key, val, j + 1, end);
    }
}

double pangulu_standard_deviation(pangulu_int64_t *p, pangulu_int64_t num)
{
    double average = 0.0;
    for (pangulu_int64_t i = 0; i < num; i++)
    {
        average += (double)p[i];
    }
    average /= (double)num;
    double answer = 0.0;
    for (pangulu_int64_t i = 0; i < num; i++)
    {
        answer += (double)(((double)p[i] - average) * ((double)p[i] - average));
    }
    return sqrt(answer / (double)(num));
}

#ifndef GPU_OPEN
void pangulu_init_level_array(pangulu_smatrix *a, pangulu_int64_t *work_space)
{
    pangulu_int64_t n = a->row;
    pangulu_int64_t *level_size = a->level_size;
    pangulu_int64_t *level_idx = a->level_idx;
    pangulu_int64_t index_inlevel = 0;
    pangulu_int64_t index_level_ptr = 0;
    pangulu_int64_t num_lev = 0;

    pangulu_int64_t *l_col_ptr = work_space;
    pangulu_int64_t *csr_diag_ptr = work_space + n + 1;
    pangulu_int64_t *inlevel = work_space + (n + 1) * 2;
    pangulu_int64_t *level_ptr = work_space + (n + 1) * 3;

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        level_idx[i] = 0;
        level_size[i] = 0;
        inlevel[i] = 0;
        level_ptr[i] = 0;
        l_col_ptr[i] = 0;
        csr_diag_ptr[i] = 0;
    }

    for (pangulu_int64_t i = 0; i < n; i++) // each csc column
    {
        for (pangulu_int64_t j = a->columnpointer[i]; j < a->columnpointer[i + 1]; j++)
        {
            pangulu_int64_t row = a->rowindex[j];
            if (row == i)
            {
                l_col_ptr[i] = j;
            }
        }
    }

    for (pangulu_int64_t i = 0; i < n; i++) // each csr row
    {
        for (pangulu_int64_t j = a->rowpointer[i]; j < a->rowpointer[i + 1]; j++)
        {
            pangulu_int64_t column = a->columnindex[j];
            if (column == i)
            {
                csr_diag_ptr[i] = j;
                continue;
            }
            else
            {
                csr_diag_ptr[i] = -1;
            }
        }
    }

    for (pangulu_int64_t i = 0; i < n; i++) // each csc column
    {
        pangulu_int64_t max_lv = -1;
        pangulu_int64_t lv;
        // search dependent columns on the left
        for (pangulu_int64_t j = a->columnpointer[i]; j < l_col_ptr[i]; j++)
        {
            unsigned nz_idx = a->rowindex[j]; // Nonzero row in col i, u part

            // l part of col nz_idx exists , u-dependency found
            if (l_col_ptr[nz_idx] + 1 != a->columnpointer[nz_idx + 1])
            {
                lv = inlevel[nz_idx];
                if (lv > max_lv)
                {
                    max_lv = lv;
                }
            }
        }
        for (pangulu_int64_t j = a->rowpointer[i]; j < csr_diag_ptr[i]; j++)
        {
            unsigned nz_idx = a->columnindex[j];
            lv = inlevel[nz_idx];
            if (lv > max_lv)
            {
                max_lv = lv;
            }
        }
        lv = max_lv + 1;
        inlevel[index_inlevel++] = lv;
        ++level_size[lv];
        if (lv > num_lev)
        {
            num_lev = lv;
        }
    }

    ++num_lev;

    level_ptr[index_level_ptr++] = 0;
    for (pangulu_int64_t i = 0; i < num_lev; i++)
    {
        level_ptr[index_level_ptr++] = level_ptr[i] + level_size[i];
    }

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        level_idx[level_ptr[inlevel[i]]++] = i;
    }

    a->num_lev = num_lev;
}

#endif

pangulu_int64_t choose_pivot(pangulu_int64_t i, pangulu_int64_t j)
{
    return (i + j) / 2;
}

void swap_value(calculate_type *a, calculate_type *b)
{
    calculate_type tmp = *a;
    *a = *b;
    *b = tmp;
}

void swap_index_1(pangulu_exblock_idx *a, pangulu_exblock_idx *b)
{
    pangulu_exblock_idx tmp = *a;
    *a = *b;
    *b = tmp;
}
void swap_index_3(int32_t *a, int32_t *b)
{
    int32_t tmp = *a;
    *a = *b;
    *b = tmp;
}

void swap_index_2(pangulu_inblock_idx *a, pangulu_inblock_idx *b)
{
    pangulu_inblock_idx tmp = *a;
    *a = *b;
    *b = tmp;
}

void pangulu_sort(pangulu_exblock_idx *key, calculate_type *val, pangulu_int64_t start, pangulu_int64_t end)
{
    if (start < end)
    {
        pangulu_int64_t pivot;
        pangulu_int64_t i, j, k;

        k = choose_pivot(start, end);
        swap_index_1(&key[start], &key[k]);
        swap_value(&val[start], &val[k]);
        pivot = key[start];

        i = start + 1;
        j = end;
        while (i <= j)
        {
            while ((i <= end) && (key[i] <= pivot))
                i++;
            while ((j >= start) && (key[j] > pivot))
                j--;
            if (i < j)
            {
                swap_index_1(&key[i], &key[j]);
                swap_value(&val[i], &val[j]);
            }
        }

        // swap two elements
        swap_index_1(&key[start], &key[j]);
        swap_value(&val[start], &val[j]);

        // recursively sort the lesser key
        pangulu_sort(key, val, start, j - 1);
        pangulu_sort(key, val, j + 1, end);
    }
}

void pangulu_sort_struct_1(pangulu_exblock_idx *key, pangulu_exblock_ptr start, pangulu_exblock_ptr end)
{
    pangulu_int64_t pivot;
    pangulu_int64_t i, j, k;

    if (start < end)
    {
        k = choose_pivot(start, end);
        swap_index_1(&key[start], &key[k]);
        pivot = key[start];

        i = start + 1;
        j = end;
        while (i <= j)
        {
            while ((i <= end) && (key[i] <= pivot))
                i++;
            while ((j >= start) && (key[j] > pivot))
                j--;
            if (i < j)
            {
                swap_index_1(&key[i], &key[j]);
            }
        }

        // swap two elements
        swap_index_1(&key[start], &key[j]);

        // recursively sort the lesser key
        pangulu_sort_struct_1(key, start, j - 1);
        pangulu_sort_struct_1(key, j + 1, end);
    }
}

void pangulu_sort_struct_2(pangulu_inblock_idx *key, pangulu_int64_t start, pangulu_int64_t end)
{
    pangulu_int64_t pivot;
    pangulu_int64_t i, j, k;

    if (start < end)
    {
        k = choose_pivot(start, end);
        swap_index_2(&key[start], &key[k]);
        pivot = key[start];

        i = start + 1;
        j = end;
        while (i <= j)
        {
            while ((i <= end) && (key[i] <= pivot))
                i++;
            while ((j >= start) && (key[j] > pivot))
                j--;
            if (i < j)
            {
                swap_index_2(&key[i], &key[j]);
            }
        }

        // swap two elements
        swap_index_2(&key[start], &key[j]);

        // recursively sort the lesser key
        pangulu_sort_struct_2(key, start, j - 1);
        pangulu_sort_struct_2(key, j + 1, end);
    }
}

void pangulu_sort_pangulu_matrix(pangulu_int64_t n, pangulu_exblock_ptr *rowpointer, pangulu_exblock_idx *columnindex)
{
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        pangulu_sort_struct_1(columnindex, rowpointer[i], rowpointer[i + 1] - 1);
    }
}

void pangulu_sort_pangulu_origin_smatrix(pangulu_origin_smatrix *s)
{
    for (pangulu_exblock_idx i = 0; i < s->row; i++)
    {
        pangulu_sort(s->columnindex, s->value, s->rowpointer[i], s->rowpointer[i + 1] - 1);
    }
}
#ifdef GPU_OPEN
void triangle_pre_cpu(pangulu_inblock_idx *L_rowindex,
                      const pangulu_int64_t n,
                      const pangulu_int64_t nnzL,
                      int *d_graphindegree)
{
    for (int i = 0; i < nnzL; i++)
    {
        d_graphindegree[L_rowindex[i]] += 1;
    }
}

void pangulu_gessm_preprocess(pangulu_smatrix *l)
{
    pangulu_int64_t n = l->row;
    pangulu_int64_t nnzL = l->nnz;

    /**********************************l****************************************/

    int *graphindegree = l->graphindegree;
    memset(graphindegree, 0, n * sizeof(int));

    triangle_pre_cpu(l->rowindex, n, nnzL, graphindegree);
}

void pangulu_tstrf_preprocess(pangulu_smatrix *u)
{
    pangulu_int64_t n = u->row;
    pangulu_int64_t nnzU = u->nnz;

    /**********************************l****************************************/

    int *graphindegree = u->graphindegree;
    memset(graphindegree, 0, n * sizeof(int));

    triangle_pre_cpu(u->columnindex, n, nnzU, graphindegree);
}
#endif