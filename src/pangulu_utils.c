#include "pangulu_common.h"

void pangulu_sort_exblock_struct(
    pangulu_exblock_idx n,
    pangulu_exblock_ptr* pointer,
    pangulu_exblock_idx* index,
    pangulu_int32_t nthread
){
    if(nthread <= 0){
        nthread = sysconf(_SC_NPROCESSORS_ONLN);
    }
    
    #pragma omp parallel num_threads(nthread)
    {
        bind_to_core(omp_get_thread_num() % sysconf(_SC_NPROCESSORS_ONLN));
    }

    #pragma omp parallel for num_threads(nthread) schedule(guided)
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        pangulu_kvsort(index, NULL, pointer[i], pointer[i + 1] - 1);
    }
}

int cmp_exidx_asc(const void* a, const void* b){
    if((*(pangulu_exblock_idx*)a) > (*(pangulu_exblock_idx*)b)){
        return 1;
    }else if((*(pangulu_exblock_idx*)a) < (*(pangulu_exblock_idx*)b)){
        return -1;
    }else{
        return 0;
    }
}


void pangulu_kvsort(pangulu_exblock_idx *key, calculate_type *val, pangulu_int64_t start, pangulu_int64_t end)
{
    if(val){
        if (start < end)
        {
            pangulu_int64_t pivot;
            pangulu_int64_t i, j, k;

            // k = choose_pivot(start, end);
            k = (start + end) / 2;
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
            pangulu_kvsort(key, val, start, j - 1);
            pangulu_kvsort(key, val, j + 1, end);
        }
    }else{
        qsort(key + start, end + 1 - start , sizeof(pangulu_exblock_idx), cmp_exidx_asc);
    }
}

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
){
#define _PANGULU_SET_VALUE_SIZE(size) ((distcsc_value)?(size):(0))
#define _PANGULU_SET_BVALUE_SIZE(size) ((malloc_distbcsc_value)?(size):(0))

    if(distcsc_value){
        malloc_distbcsc_value = 1;
    }
    // printf("malloc_distbcsc_value = %d\n", malloc_distbcsc_value);

    struct timeval start_time;
    pangulu_time_start(&start_time);

    pangulu_int32_t rank, nproc;
    pangulu_cm_rank(&rank);
    pangulu_cm_size(&nproc);

    if(distcsc_proc_nnzptr){
        pangulu_free(__FILE__, __LINE__, distcsc_proc_nnzptr);
    }
    // printf("1.1\n");

    int preprocess_ompnum_separate_block = 2;

    bind_to_core((rank * preprocess_ompnum_separate_block) % sysconf(_SC_NPROCESSORS_ONLN));

    #pragma omp parallel num_threads(preprocess_ompnum_separate_block)
    {
        bind_to_core((rank * preprocess_ompnum_separate_block + omp_get_thread_num()) % sysconf(_SC_NPROCESSORS_ONLN));
    }

    pangulu_int64_t p = sqrt(nproc);
    while ((nproc % p) != 0)
    {
        p--;
    }
    pangulu_int64_t q = nproc / p;
    pangulu_exblock_idx col_per_rank = PANGULU_ICEIL(PANGULU_ICEIL(n_glo, block_order * q), nproc) * (block_order * q);
    pangulu_int64_t block_length = PANGULU_ICEIL(col_per_rank, block_order);
    pangulu_int64_t block_length_col = PANGULU_ICEIL(n_glo, block_order);
    pangulu_exblock_ptr nnz = distcsc_pointer[n_loc];
    pangulu_int64_t bit_length = (block_length_col + 31) / 32;
    pangulu_int64_t block_num = 0;
    pangulu_int64_t *block_nnz_pt;
    // printf("1.2\n");

    pangulu_int64_t avg_nnz = PANGULU_ICEIL(nnz, preprocess_ompnum_separate_block);
    pangulu_int64_t *block_row_nnz_pt = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (block_length + 1));
    for (int i = 0; i < block_length; i++)
    {
        block_row_nnz_pt[i] = distcsc_pointer[PANGULU_MIN(i * block_order, n_loc)];
    }
    block_row_nnz_pt[block_length] = distcsc_pointer[n_loc];

    int *thread_pt = (int *)pangulu_malloc(__FILE__, __LINE__, sizeof(int) * (preprocess_ompnum_separate_block + 1));
    thread_pt[0] = 0;
    for (int i = 1; i < preprocess_ompnum_separate_block + 1; i++)
    {
        thread_pt[i] = binarylowerbound(block_row_nnz_pt, block_length, avg_nnz * i);
    }
    pangulu_free(__FILE__, __LINE__, block_row_nnz_pt);
    block_row_nnz_pt = NULL;

    pangulu_int64_t *block_row_pt = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (block_length + 1));
    memset(block_row_pt, 0, sizeof(pangulu_int64_t) * (block_length + 1));

    unsigned int *bit_array = (unsigned int *)pangulu_malloc(__FILE__, __LINE__, sizeof(unsigned int) * bit_length * preprocess_ompnum_separate_block);

#pragma omp parallel num_threads(preprocess_ompnum_separate_block)
    {
        int tid = omp_get_thread_num();
        unsigned int *tmp_bit = bit_array + bit_length * tid;

        for (int level = thread_pt[tid]; level < thread_pt[tid + 1]; level++)
        {
            memset(tmp_bit, 0, sizeof(unsigned int) * bit_length);

            int start_row = level * block_order;
            int end_row = ((level + 1) * block_order) < n_loc ? ((level + 1) * block_order) : n_loc;

            for (int rid = start_row; rid < end_row; rid++)
            {
                for (pangulu_int64_t idx = distcsc_pointer[rid]; idx < distcsc_pointer[rid + 1]; idx++)
                {
                    pangulu_int32_t colidx = distcsc_index[idx];
                    pangulu_int32_t block_cid = colidx / block_order;
                    setbit(tmp_bit[block_cid / 32], block_cid % 32);
                }
            }

            pangulu_int64_t tmp_blocknum = 0;
            for (int i = 0; i < bit_length; i++)
            {
                tmp_blocknum += __builtin_popcount(tmp_bit[i]);
            }

            block_row_pt[level] = tmp_blocknum;
        }
    }
    exclusive_scan_1(block_row_pt, block_length + 1);
    block_num = block_row_pt[block_length];

    block_nnz_pt = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (block_num + 1));
    memset(block_nnz_pt, 0, sizeof(pangulu_int64_t) * (block_num + 1));
    pangulu_int32_t *block_col_idx = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * block_num);

    int *count_array = (int *)pangulu_malloc(__FILE__, __LINE__, sizeof(int) * block_length_col * preprocess_ompnum_separate_block);
#pragma omp parallel num_threads(preprocess_ompnum_separate_block)
    {
        int tid = omp_get_thread_num();
        unsigned int *tmp_bit = bit_array + bit_length * tid;
        int *tmp_count = count_array + block_length_col * tid;

        for (int level = thread_pt[tid]; level < thread_pt[tid + 1]; level++)
        {
            memset(tmp_bit, 0, sizeof(unsigned int) * bit_length);
            memset(tmp_count, 0, sizeof(int) * block_length_col);

            pangulu_int64_t *cur_block_nnz_pt = block_nnz_pt + block_row_pt[level];
            pangulu_int32_t *cur_block_col_idx = block_col_idx + block_row_pt[level];

            int start_row = level * block_order;
            int end_row = ((level + 1) * block_order) < n_loc ? ((level + 1) * block_order) : n_loc;

            for (int rid = start_row; rid < end_row; rid++)
            {
                for (pangulu_int64_t idx = distcsc_pointer[rid]; idx < distcsc_pointer[rid + 1]; idx++)
                {
                    pangulu_int32_t colidx = distcsc_index[idx];
                    pangulu_int32_t block_cid = colidx / block_order;
                    setbit(tmp_bit[block_cid / 32], block_cid % 32);
                    tmp_count[block_cid]++;
                }
            }

            pangulu_int64_t cnt = 0;
            for (int i = 0; i < block_length_col; i++)
            {
                if (getbit(tmp_bit[i / 32], i % 32))
                {
                    cur_block_nnz_pt[cnt] = tmp_count[i];
                    cur_block_col_idx[cnt] = i;
                    cnt++;
                }
            }
        }
    }
    pangulu_free(__FILE__, __LINE__, bit_array);
    bit_array = NULL;
    pangulu_free(__FILE__, __LINE__, count_array);
    count_array = NULL;
    exclusive_scan_1(block_nnz_pt, block_num + 1);
    
    // printf("1.3\n");
    
    pangulu_exblock_ptr* nzblk_each_rank_ptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (nproc + 1));
    pangulu_exblock_ptr* nnz_each_rank_ptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (nproc + 1));
    memset(nzblk_each_rank_ptr, 0, sizeof(pangulu_exblock_ptr) * (nproc + 1));
    memset(nnz_each_rank_ptr, 0, sizeof(pangulu_exblock_ptr) * (nproc + 1));
    for(pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++){
        for(pangulu_exblock_ptr bidx = block_row_pt[bcol]; bidx < block_row_pt[bcol+1]; bidx++){
            pangulu_exblock_idx brow = block_col_idx[bidx];
            nzblk_each_rank_ptr[(brow % p) * q + (bcol % q) + 1]++;
            nnz_each_rank_ptr[(brow % p) * q + (bcol % q) + 1] += (block_nnz_pt[bidx + 1] - block_nnz_pt[bidx]);
        }
    }
    for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
        nzblk_each_rank_ptr[remote_rank + 1] += nzblk_each_rank_ptr[remote_rank];
        nnz_each_rank_ptr[remote_rank + 1] += nnz_each_rank_ptr[remote_rank];
    }
    char** csc_draft_remote = pangulu_malloc(__FILE__, __LINE__, sizeof(char*) * nproc);
    for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
        csc_draft_remote[remote_rank] = pangulu_malloc(
            __FILE__, __LINE__, 
            sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
            sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]) +
            sizeof(pangulu_exblock_ptr) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank] + 1)
        );
        memset(csc_draft_remote[remote_rank], 0, 
            sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
            sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]) +
            sizeof(pangulu_exblock_ptr) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank] + 1)
        );
    }
    
    for(pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++){
        for(pangulu_exblock_ptr bidx = block_row_pt[bcol]; bidx < block_row_pt[bcol+1]; bidx++){
            pangulu_exblock_idx brow = block_col_idx[bidx];
            pangulu_inblock_ptr nnz_in_blk = block_nnz_pt[bidx + 1] - block_nnz_pt[bidx];
            pangulu_int32_t remote_rank = (brow % p) * q + (bcol % q);
            pangulu_exblock_ptr* remote_bcolptr = (pangulu_exblock_ptr*)csc_draft_remote[remote_rank];
            remote_bcolptr[bcol + 1]++;
        }
    }
    // printf("1.4\n");

    pangulu_exblock_ptr* aid_arr_colptr_remote = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (block_length + 1) * nproc);
    for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
        pangulu_exblock_ptr* remote_bcolptr = (pangulu_exblock_ptr*)csc_draft_remote[remote_rank];
        for(pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++){
            remote_bcolptr[bcol + 1] += remote_bcolptr[bcol];
        }
        memcpy(&aid_arr_colptr_remote[(block_length + 1) * remote_rank], remote_bcolptr, sizeof(pangulu_exblock_ptr) * (block_length + 1));
    }

    for(pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++){
        for(pangulu_exblock_ptr bidx = block_row_pt[bcol]; bidx < block_row_pt[bcol+1]; bidx++){
            pangulu_exblock_idx brow = block_col_idx[bidx];
            pangulu_inblock_ptr nnz_in_blk = block_nnz_pt[bidx + 1] - block_nnz_pt[bidx];
            pangulu_int32_t remote_rank = (brow % p) * q + (bcol % q);
            
            pangulu_exblock_idx* remote_browidx = (pangulu_exblock_idx*)
                (csc_draft_remote[remote_rank] + 
                sizeof(pangulu_exblock_ptr) * (block_length + 1));
            pangulu_exblock_ptr* remote_blknnzptr = (pangulu_exblock_ptr*)
                (csc_draft_remote[remote_rank] + 
                sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
                sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]));
            
            remote_browidx[aid_arr_colptr_remote[(block_length + 1) * remote_rank + bcol]] = brow;
            remote_blknnzptr[aid_arr_colptr_remote[(block_length + 1) * remote_rank + bcol] + 1] = nnz_in_blk;
            aid_arr_colptr_remote[(block_length + 1) * remote_rank + bcol]++;
        }
    }
    pangulu_free(__FILE__, __LINE__, aid_arr_colptr_remote);
    aid_arr_colptr_remote = NULL;
    pangulu_free(__FILE__, __LINE__, block_nnz_pt);
    block_nnz_pt = NULL;
    
    for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
        pangulu_exblock_ptr* remote_bcolptr = csc_draft_remote[remote_rank];
        pangulu_exblock_ptr* remote_blknnzptr = 
                csc_draft_remote[remote_rank] + 
                sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
                sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]);
        for(pangulu_inblock_ptr bidx = 0; bidx < remote_bcolptr[block_length]; bidx++){
            remote_blknnzptr[bidx + 1] += remote_blknnzptr[bidx];
        }
    }

    char** block_csc_remote = pangulu_malloc(__FILE__, __LINE__, sizeof(char*) * nproc);
    for(pangulu_int32_t target_rank = 0; target_rank < nproc; target_rank++){
        block_csc_remote[target_rank] = pangulu_malloc(
            __FILE__, __LINE__, 
            sizeof(pangulu_inblock_ptr) * (block_order + 1) * (nzblk_each_rank_ptr[target_rank + 1] - nzblk_each_rank_ptr[target_rank]) +
            sizeof(pangulu_inblock_idx) * (nnz_each_rank_ptr[target_rank + 1] - nnz_each_rank_ptr[target_rank]) + 
            _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type)) * (nnz_each_rank_ptr[target_rank + 1] - nnz_each_rank_ptr[target_rank])
        );
        memset(
            block_csc_remote[target_rank], 0, 
            sizeof(pangulu_inblock_ptr) * (block_order + 1) * (nzblk_each_rank_ptr[target_rank + 1] - nzblk_each_rank_ptr[target_rank]) +
            sizeof(pangulu_inblock_idx) * (nnz_each_rank_ptr[target_rank + 1] - nnz_each_rank_ptr[target_rank]) + 
            _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type)) * (nnz_each_rank_ptr[target_rank + 1] - nnz_each_rank_ptr[target_rank])
        );
    }


#pragma omp parallel num_threads(preprocess_ompnum_separate_block)
    {
        int tid = omp_get_thread_num();
        int* tmp_count = pangulu_malloc(__FILE__, __LINE__, sizeof(int) * block_length_col * q);

        for (int level = thread_pt[tid]; level < thread_pt[tid + 1]; level++)
        {

            memset(tmp_count, 0, sizeof(int) * block_length_col * q);
        
            pangulu_exblock_idx start_col = level * block_order;
            pangulu_exblock_idx end_col = ((level + 1) * block_order) < n_loc ? ((level + 1) * block_order) : n_loc;
            
            for (pangulu_exblock_idx col = start_col, col_in_blc = 0; col < end_col; col++, col_in_blc++)
            {
                pangulu_int64_t bidx_glo = block_row_pt[level];
                pangulu_exblock_idx brow = block_col_idx[bidx_glo];
                pangulu_int32_t target_rank = (brow % p) * q + (level % q);
                pangulu_exblock_ptr* remote_bcolptr = csc_draft_remote[target_rank];
                pangulu_exblock_idx* remote_browidx = csc_draft_remote[target_rank] + sizeof(pangulu_exblock_ptr) * (block_length + 1);
                pangulu_exblock_ptr* remote_bnnzptr = 
                    csc_draft_remote[target_rank] + 
                    sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
                    sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[target_rank + 1] - nzblk_each_rank_ptr[target_rank]);
                pangulu_int64_t bidx = remote_bcolptr[level];

                pangulu_int64_t arr_len = 
                    sizeof(pangulu_inblock_ptr) * bidx * (block_order + 1) + 
                    (sizeof(pangulu_inblock_idx) + _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type))) * remote_bnnzptr[bidx];
                pangulu_inblock_ptr *cur_block_rowptr = (pangulu_inblock_ptr *)(block_csc_remote[target_rank] + arr_len);
                pangulu_inblock_idx *cur_block_colidx = (pangulu_inblock_idx *)(block_csc_remote[target_rank] + arr_len + sizeof(pangulu_inblock_ptr) * (block_order + 1));
                calculate_type *cur_block_value = NULL;
                if(malloc_distbcsc_value){
                    cur_block_value = (calculate_type *)(
                        block_csc_remote[target_rank] + arr_len + 
                        sizeof(pangulu_inblock_ptr) * (block_order + 1) + 
                        sizeof(pangulu_inblock_idx) * (remote_bnnzptr[bidx + 1] - remote_bnnzptr[bidx])
                    );
                }

                pangulu_exblock_ptr reorder_matrix_idx = distcsc_pointer[col];
                pangulu_exblock_ptr reorder_matrix_idx_ub = distcsc_pointer[col + 1];
                for (pangulu_exblock_ptr idx = distcsc_pointer[col]; idx < distcsc_pointer[col + 1]; idx++)
                {
                    pangulu_exblock_idx row = distcsc_index[idx];
                    brow = row / block_order;
                    if (block_col_idx[bidx_glo] != brow)
                    {
                        bidx_glo = binarysearch(block_col_idx, bidx_glo, block_row_pt[level + 1], brow);
                        target_rank = (brow % p) * q + (level % q);
                        remote_bcolptr = csc_draft_remote[target_rank];
                        remote_browidx = csc_draft_remote[target_rank] + sizeof(pangulu_exblock_ptr) * (block_length + 1);
                        remote_bnnzptr = 
                            csc_draft_remote[target_rank] + 
                            sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
                            sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[target_rank + 1] - nzblk_each_rank_ptr[target_rank]);
                        bidx = binarysearch(remote_browidx, remote_bcolptr[level], remote_bcolptr[level + 1], brow);
                        arr_len = 
                            sizeof(pangulu_inblock_ptr) * bidx * (block_order + 1) + 
                            (sizeof(pangulu_inblock_idx) + _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type))) * remote_bnnzptr[bidx];
                        cur_block_rowptr = (pangulu_inblock_ptr *)(block_csc_remote[target_rank] + arr_len);
                        cur_block_colidx = (pangulu_inblock_idx *)(block_csc_remote[target_rank] + arr_len + sizeof(pangulu_inblock_ptr) * (block_order + 1));
                        if(malloc_distbcsc_value){
                            cur_block_value = (calculate_type *)(
                                block_csc_remote[target_rank] + arr_len + 
                                sizeof(pangulu_inblock_ptr) * (block_order + 1) + 
                                sizeof(pangulu_inblock_idx) * (remote_bnnzptr[bidx + 1] - remote_bnnzptr[bidx])
                            );
                        }
                    }
                    if(distcsc_value){
                        cur_block_value[tmp_count[(level % q) * block_length_col + brow]] = distcsc_value[reorder_matrix_idx];
                    }else if(malloc_distbcsc_value){
                        cur_block_value[tmp_count[(level % q) * block_length_col + brow]] = 0;
                    }
                    reorder_matrix_idx++;
                    cur_block_colidx[tmp_count[(level % q) * block_length_col + brow]++] = row % block_order;
                    cur_block_rowptr[col_in_blc]++;
                }
            }

            for(pangulu_int32_t target_rank = 0; target_rank < nproc; target_rank++){
                pangulu_exblock_ptr* remote_bcolptr = csc_draft_remote[target_rank];
                pangulu_exblock_ptr* remote_bnnzptr = 
                    csc_draft_remote[target_rank] + 
                    sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
                    sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[target_rank + 1] - nzblk_each_rank_ptr[target_rank]);
                for (pangulu_int64_t bidx = remote_bcolptr[level]; bidx < remote_bcolptr[level + 1]; bidx++)
                {
                    pangulu_int64_t tmp_stride = bidx * (block_order + 1) * sizeof(pangulu_inblock_ptr) + remote_bnnzptr[bidx] * (sizeof(pangulu_inblock_idx) + _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type)));
                    pangulu_inblock_ptr *cur_csr_rpt = (pangulu_inblock_ptr *)(block_csc_remote[target_rank] + tmp_stride);
                    exclusive_scan_3(cur_csr_rpt, block_order + 1);
                }
            }
        }
        pangulu_free(__FILE__, __LINE__, tmp_count);
        tmp_count = NULL;
    }
    pangulu_free(__FILE__, __LINE__, thread_pt);
    thread_pt = NULL;
    pangulu_free(__FILE__, __LINE__, block_row_pt);
    block_row_pt = NULL;
    pangulu_free(__FILE__, __LINE__, block_col_idx);
    block_col_idx = NULL;

    if(rootproc_free_originmatrix){
        pangulu_free(__FILE__, __LINE__, distcsc_pointer);
        pangulu_free(__FILE__, __LINE__, distcsc_index);
        if(distcsc_value){
            pangulu_free(__FILE__, __LINE__, distcsc_value); // Don't set distcsc_value to NULL.
        }
    }

    // comm TAG=0
    pangulu_cm_sync();
    pangulu_exblock_ptr* nzblk_remote = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * nproc);
    for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
        MPI_Request req;
        nzblk_remote[remote_rank] = nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank];
        MPI_Isend(&nzblk_remote[remote_rank], 1, MPI_PANGULU_EXBLOCK_PTR, remote_rank, 0, MPI_COMM_WORLD, &req);
    }
    pangulu_exblock_ptr* nzblk_fetch = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * nproc);
    for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
        MPI_Status stat;
        MPI_Recv(&nzblk_fetch[fetch_rank], 1, MPI_PANGULU_EXBLOCK_PTR, fetch_rank, 0, MPI_COMM_WORLD, &stat);
    }
    pangulu_cm_sync();
    pangulu_free(__FILE__, __LINE__, nzblk_remote);
    // printf("#%d 1.7\n", rank);

    // comm TAG=1 send csc_draft_remote
    pangulu_cm_sync();
    for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
        pangulu_cm_isend(
            csc_draft_remote[remote_rank], 
            sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
                sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]) +
                sizeof(pangulu_exblock_ptr) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank] + 1),
            remote_rank, 1, 10
        );
    }
    char** bstruct_csc_fetch = pangulu_malloc(__FILE__, __LINE__, sizeof(char*) * nproc);
    for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
        bstruct_csc_fetch[fetch_rank] = pangulu_malloc(
            __FILE__, __LINE__, 
            sizeof(pangulu_exblock_ptr) * (block_length+1) + 
            sizeof(pangulu_exblock_idx) * nzblk_fetch[fetch_rank] + 
            sizeof(pangulu_exblock_ptr) * (nzblk_fetch[fetch_rank] + 1)
        );
        pangulu_cm_recv(
            bstruct_csc_fetch[fetch_rank],
            sizeof(pangulu_exblock_ptr) * (block_length+1) + sizeof(pangulu_exblock_idx) * nzblk_fetch[fetch_rank] + sizeof(pangulu_exblock_ptr) * (nzblk_fetch[fetch_rank] + 1),
            fetch_rank, 1, 10
        );
    }
    pangulu_cm_sync();
    for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
        pangulu_free(__FILE__, __LINE__, csc_draft_remote[remote_rank]);
    }
    pangulu_free(__FILE__, __LINE__, csc_draft_remote);
    csc_draft_remote = NULL;
    
    // comm TAG=2 send block_csc_remote
    for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
        // MPI_Request req;
        pangulu_int64_t send_size = sizeof(pangulu_inblock_ptr) * (block_order + 1) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]) +
            sizeof(pangulu_inblock_idx) * (nnz_each_rank_ptr[remote_rank + 1] - nnz_each_rank_ptr[remote_rank]) + 
            _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type)) * (nnz_each_rank_ptr[remote_rank + 1] - nnz_each_rank_ptr[remote_rank]);
        pangulu_cm_isend(block_csc_remote[remote_rank], send_size, remote_rank, 2, 10);
    }
    pangulu_free(__FILE__, __LINE__, nzblk_each_rank_ptr);
    nzblk_each_rank_ptr = NULL;
    pangulu_free(__FILE__, __LINE__, nnz_each_rank_ptr);
    nnz_each_rank_ptr = NULL;
    char** block_csc_fetch = pangulu_malloc(__FILE__, __LINE__, sizeof(char*) * nproc);
    for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
        MPI_Status stat;
        pangulu_exblock_ptr* remote_bcolptr = bstruct_csc_fetch[fetch_rank];
        pangulu_exblock_idx* remote_browidx = bstruct_csc_fetch[fetch_rank] + sizeof(pangulu_exblock_ptr) * (block_length + 1);
        pangulu_exblock_ptr* remote_bnnzptr = 
            bstruct_csc_fetch[fetch_rank] + 
            sizeof(pangulu_exblock_ptr) * (block_length + 1) + 
            sizeof(pangulu_exblock_idx) * (remote_bcolptr[block_length]);
        
        pangulu_int64_t recv_size = 
            sizeof(pangulu_inblock_ptr) * (block_order + 1) * remote_bcolptr[block_length] + 
            sizeof(pangulu_inblock_idx) * remote_bnnzptr[remote_bcolptr[block_length]] + 
            _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type)) * remote_bnnzptr[remote_bcolptr[block_length]];
        block_csc_fetch[fetch_rank] = pangulu_malloc(__FILE__, __LINE__, recv_size);
        // MPI_Recv(block_csc_fetch[fetch_rank], recv_size, MPI_CHAR, fetch_rank, 2, MPI_COMM_WORLD, &stat);
        pangulu_cm_recv(block_csc_fetch[fetch_rank], recv_size, fetch_rank, 2, 10);
    }
    pangulu_cm_sync();
    for(pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++){
        pangulu_free(__FILE__, __LINE__, block_csc_remote[remote_rank]);
    }
    pangulu_free(__FILE__, __LINE__, block_csc_remote);
    block_csc_remote = NULL;
    
    pangulu_exblock_ptr* struct_bcolptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (block_length_col + 1));
    memset(struct_bcolptr, 0, sizeof(pangulu_exblock_ptr) * (block_length_col + 1));
    pangulu_exblock_ptr last_fetch_rank_ptr = 0;
    for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
        pangulu_exblock_ptr* bstruct_fetch_pointer = bstruct_csc_fetch[fetch_rank];
        for(pangulu_exblock_idx brow_offset = 0; brow_offset < block_length; brow_offset++){
            if(fetch_rank * block_length + brow_offset > block_length_col){
                break;
            }
            struct_bcolptr[fetch_rank * block_length + brow_offset] = bstruct_fetch_pointer[brow_offset] + last_fetch_rank_ptr;
        }
        last_fetch_rank_ptr += bstruct_fetch_pointer[block_length];
    }
    struct_bcolptr[block_length_col] = last_fetch_rank_ptr;
    
    pangulu_exblock_idx* struct_browidx = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * struct_bcolptr[block_length_col]);
    pangulu_exblock_ptr* struct_bnnzptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (struct_bcolptr[block_length_col] + 1));
    pangulu_exblock_ptr last_fetch_bnnz_ptr = 0;
    for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
        pangulu_exblock_ptr* bstruct_fetch_pointer = bstruct_csc_fetch[fetch_rank];
        pangulu_exblock_idx* bstruct_fetch_index = bstruct_csc_fetch[fetch_rank] + sizeof(pangulu_exblock_ptr) * (block_length+1);
        pangulu_exblock_ptr* bstruct_fetch_nnzptr = bstruct_csc_fetch[fetch_rank] + sizeof(pangulu_exblock_ptr) * (block_length+1) + sizeof(pangulu_exblock_idx) * nzblk_fetch[fetch_rank];
        for(pangulu_exblock_ptr bidx_offset = 0; bidx_offset < nzblk_fetch[fetch_rank]; bidx_offset++){
            struct_browidx[struct_bcolptr[fetch_rank * block_length] + bidx_offset] = bstruct_fetch_index[bidx_offset];
        }
        for(pangulu_exblock_ptr bidx_offset = 0; bidx_offset < nzblk_fetch[fetch_rank]; bidx_offset++){
            struct_bnnzptr[struct_bcolptr[fetch_rank * block_length] + bidx_offset] = bstruct_fetch_nnzptr[bidx_offset] + last_fetch_bnnz_ptr;
        }
        last_fetch_bnnz_ptr += bstruct_fetch_nnzptr[nzblk_fetch[fetch_rank]];
    }
    struct_bnnzptr[struct_bcolptr[block_length_col]] = last_fetch_bnnz_ptr;
    for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
        pangulu_free(__FILE__, __LINE__, bstruct_csc_fetch[fetch_rank]);
    }
    pangulu_free(__FILE__, __LINE__, bstruct_csc_fetch);
    bstruct_csc_fetch = NULL;
    pangulu_free(__FILE__, __LINE__, nzblk_fetch);
    nzblk_fetch = NULL;

    char* block_csc = pangulu_malloc(
        __FILE__, __LINE__,
        sizeof(pangulu_inblock_ptr) * (block_order + 1) * struct_bcolptr[block_length_col] + 
        sizeof(pangulu_inblock_idx) * struct_bnnzptr[struct_bcolptr[block_length_col]] + 
        _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type)) * struct_bnnzptr[struct_bcolptr[block_length_col]]
    );
    for(pangulu_int32_t fetch_rank = 0; fetch_rank < nproc; fetch_rank++){
        pangulu_exblock_idx bidx_ub = struct_bcolptr[PANGULU_MIN((fetch_rank + 1) * block_length, block_length_col)];
        pangulu_exblock_idx bidx_lb = struct_bcolptr[PANGULU_MIN(fetch_rank * block_length, block_length_col)];
        pangulu_int64_t offset = 
            sizeof(pangulu_inblock_ptr) * (block_order + 1) * bidx_lb + 
            sizeof(pangulu_inblock_idx) * struct_bnnzptr[bidx_lb] + 
            _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type)) * struct_bnnzptr[bidx_lb];
        pangulu_int64_t copy_size = 
            sizeof(pangulu_inblock_ptr) * (block_order + 1) * (bidx_ub - bidx_lb) +
            sizeof(pangulu_inblock_idx) * (struct_bnnzptr[bidx_ub] - struct_bnnzptr[bidx_lb]) + 
            _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type)) * (struct_bnnzptr[bidx_ub] - struct_bnnzptr[bidx_lb]);
        memcpy(block_csc + offset, block_csc_fetch[fetch_rank], copy_size);
        pangulu_free(__FILE__, __LINE__, block_csc_fetch[fetch_rank]);
    }
    pangulu_free(__FILE__, __LINE__, block_csc_fetch);
    block_csc_fetch = NULL;

    pangulu_inblock_ptr** inblock_pointers = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr*) * struct_bcolptr[block_length_col]);
    pangulu_inblock_idx** inblock_indeces = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx*) * struct_bcolptr[block_length_col]);
    calculate_type** inblock_values = NULL;
    if(malloc_distbcsc_value){
        inblock_values = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type*) * struct_bcolptr[block_length_col]);
    }

    for(pangulu_exblock_ptr bidx = 0; bidx < struct_bcolptr[block_length_col]; bidx++){
        pangulu_int64_t offset = 
            sizeof(pangulu_inblock_ptr) * (block_order + 1) * bidx + 
            sizeof(pangulu_inblock_idx) * struct_bnnzptr[bidx] + 
            _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type)) * struct_bnnzptr[bidx];
        inblock_pointers[bidx] = block_csc + offset;
        inblock_indeces[bidx] = block_csc + offset + 
            sizeof(pangulu_inblock_ptr) * (block_order + 1);
        if(malloc_distbcsc_value){
            inblock_values[bidx] = block_csc + offset + 
                sizeof(pangulu_inblock_ptr) * (block_order + 1) + 
                sizeof(pangulu_inblock_idx) * (struct_bnnzptr[bidx + 1] - struct_bnnzptr[bidx]);
        }
    }

    *bcsc_struct_pointer = struct_bcolptr;
    *bcsc_struct_index = struct_browidx;
    *bcsc_struct_nnzptr = struct_bnnzptr;
    *bcsc_inblock_pointers = inblock_pointers;
    *bcsc_inblock_indeces = inblock_indeces;
    if(malloc_distbcsc_value){
        *bcsc_values = inblock_values;
    }

    bind_to_core(rank % sysconf(_SC_NPROCESSORS_ONLN));

#undef _PANGULU_SET_BVALUE_SIZE
#undef _PANGULU_SET_VALUE_SIZE
}

void pangulu_cm_rank(pangulu_int32_t* rank){MPI_Comm_rank(MPI_COMM_WORLD, rank);}
void pangulu_cm_size(pangulu_int32_t* size){MPI_Comm_size(MPI_COMM_WORLD, size);}
void pangulu_cm_sync(){MPI_Barrier(MPI_COMM_WORLD);}
void pangulu_cm_bcast(void *buffer, int count, MPI_Datatype datatype, int root){MPI_Bcast(buffer, count, datatype, root, MPI_COMM_WORLD);}
void pangulu_cm_isend(char* buf, pangulu_int64_t count, pangulu_int32_t remote_rank, pangulu_int32_t tag, pangulu_int32_t tag_ub){
    MPI_Request req;
    const pangulu_int64_t send_maxlen = 0x7FFFFFFF;
    pangulu_int64_t send_times = PANGULU_ICEIL(count, send_maxlen);
    for(pangulu_int64_t iter = 0; iter < send_times; iter++){
        int count_current = PANGULU_MIN((iter + 1) * send_maxlen, count) - iter * send_maxlen;
        MPI_Isend(buf + iter * send_maxlen, count_current, MPI_CHAR, remote_rank, iter * tag_ub + tag, MPI_COMM_WORLD, &req);
    }
    if(send_times == 0){
        MPI_Isend(buf, 0, MPI_CHAR, remote_rank, tag, MPI_COMM_WORLD, &req);
    }
}
void pangulu_cm_send(char* buf, pangulu_int64_t count, pangulu_int32_t remote_rank, pangulu_int32_t tag, pangulu_int32_t tag_ub){
    const pangulu_int64_t send_maxlen = 0x7FFFFFFF;
    pangulu_int64_t send_times = PANGULU_ICEIL(count, send_maxlen);
    for(pangulu_int64_t iter = 0; iter < send_times; iter++){
        int count_current = PANGULU_MIN((iter + 1) * send_maxlen, count) - iter * send_maxlen;
        MPI_Send(buf + iter * send_maxlen, count_current, MPI_CHAR, remote_rank, iter * tag_ub + tag, MPI_COMM_WORLD);
    }
    if(send_times == 0){
        MPI_Send(buf, 0, MPI_CHAR, remote_rank, tag, MPI_COMM_WORLD);
    }
}
void pangulu_cm_recv(char* buf, pangulu_int64_t count, pangulu_int32_t fetch_rank, pangulu_int32_t tag, pangulu_int32_t tag_ub){
    MPI_Status stat;
    const pangulu_int64_t send_maxlen = 0x7FFFFFFF;
    pangulu_int64_t recv_times = PANGULU_ICEIL(count, send_maxlen);
    for(pangulu_int64_t iter = 0; iter < recv_times; iter++){
        int count_current = PANGULU_MIN((iter + 1) * send_maxlen, count) - iter * send_maxlen;
        MPI_Recv(buf + iter * send_maxlen, count_current, MPI_CHAR, fetch_rank, iter * tag_ub + tag, MPI_COMM_WORLD, &stat);
    }
    if(recv_times == 0){
        MPI_Recv(buf, 0, MPI_CHAR, fetch_rank, tag, MPI_COMM_WORLD, &stat);
    }
}
void pangulu_cm_sync_asym(int wake_rank)
{
    pangulu_int32_t sum_rank_size, rank;
    pangulu_cm_rank(&rank);
    pangulu_cm_size(&sum_rank_size);
    if (rank == wake_rank)
    {
        for (int i = 0; i < sum_rank_size; i++)
        {
            if (i != wake_rank)
            {
                MPI_Send(&sum_rank_size, 1, MPI_INT, i, 0xCAFE, MPI_COMM_WORLD);
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
            MPI_Iprobe(wake_rank, 0xCAFE, MPI_COMM_WORLD, &mpi_flag, &mpi_stat);
            if (mpi_flag != 0 && mpi_stat.MPI_TAG == 0xCAFE)
            {
                MPI_Recv(&mpi_buf_int, 1, MPI_INT, wake_rank, 0xCAFE, MPI_COMM_WORLD, &mpi_stat);
                if (mpi_buf_int == sum_rank_size)
                {
                    break;
                }
                else
                {
                    printf(PANGULU_E_ASYM);
                    exit(1);
                }
            }
            usleep(50);
        }
    }
    pangulu_cm_sync();
}
void pangulu_cm_probe(MPI_Status *status)
{
    int have_msg=0;
    do{
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &have_msg, status);
        if(have_msg){
            return;
        }
        usleep(10);
    }while(!have_msg);
}

// void pangulu_cm_distribute_bcsc_to_distbcsc(
//     pangulu_int32_t root_rank,
//     int rootproc_free_originmatrix,
//     pangulu_exblock_idx* n,

//     pangulu_exblock_ptr** bcsc_struct_pointer,
//     pangulu_exblock_idx** bcsc_struct_index,
//     pangulu_inblock_ptr** bcsc_struct_nnz,
//     pangulu_inblock_ptr*** bcsc_inblock_pointers,
//     pangulu_inblock_idx*** bcsc_inblock_indeces,
//     calculate_type*** bcsc_values
// ){
    

// }

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
){
    struct timeval start_time;
    pangulu_time_start(&start_time);
    // printf("1.1\n");


    pangulu_int32_t nproc = 0;
    pangulu_int32_t rank;
    pangulu_cm_size(&nproc);
    pangulu_cm_rank(&rank);
    *distcsc_nproc = nproc;
    pangulu_int64_t p = sqrt(nproc);
    while ((nproc % p) != 0)
    {
        p--;
    }
    pangulu_int64_t q = nproc / p;
    pangulu_cm_bcast(n, 1, MPI_PANGULU_EXBLOCK_IDX, root_rank);
    // printf("#%d 1.2\n", rank);

    rowchunk_align *= q;
    pangulu_exblock_idx col_per_rank = PANGULU_ICEIL(PANGULU_ICEIL(*n, rowchunk_align), nproc) * rowchunk_align;

    *distcsc_proc_nnzptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (nproc + 1));
    if(rank == root_rank){
        pangulu_exblock_ptr* columnpointer = *distcsc_pointer;
        pangulu_exblock_idx* rowindex = *distcsc_index;
        calculate_type* value_csc = NULL;
        if(distcsc_value){
            value_csc = *distcsc_value;
        }
        (*distcsc_proc_nnzptr)[0] = 0;
    // printf("#%d 1.3.1\n", rank);
        for(pangulu_int32_t target_rank = 0; target_rank < nproc; target_rank++){
    // printf("#%d 1.3.2\n", rank);
            pangulu_exblock_idx n_loc_remote = PANGULU_MIN(col_per_rank * (target_rank + 1), *n) - PANGULU_MIN(col_per_rank * target_rank, *n);
            (*distcsc_proc_nnzptr)[target_rank + 1] = columnpointer[PANGULU_MIN(col_per_rank * (target_rank + 1), *n)];
            if(rank == target_rank){
                *n_loc = n_loc_remote;
    // printf("#%d n_loc = %d\n", rank, *n_loc);

                *distcsc_pointer = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (*n_loc + 1));
                memcpy(*distcsc_pointer, &columnpointer[PANGULU_MIN(col_per_rank * target_rank, *n)], sizeof(pangulu_exblock_ptr) * (*n_loc + 1));
                pangulu_exblock_idx col_offset = (*distcsc_pointer)[0];
                for(pangulu_exblock_idx col = 0; col < *n_loc + 1; col++){
                    (*distcsc_pointer)[col] -= col_offset;
                }
                pangulu_exblock_ptr nnz_loc = (*distcsc_pointer)[*n_loc];
                *distcsc_index = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * nnz_loc);
                memcpy(*distcsc_index, &rowindex[(*distcsc_proc_nnzptr)[target_rank]], sizeof(pangulu_exblock_idx) * nnz_loc);

                if(distcsc_value){
                    *distcsc_value = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz_loc);
                    memcpy(*distcsc_value, &value_csc[(*distcsc_proc_nnzptr)[target_rank]], sizeof(calculate_type) * nnz_loc);
                }
            }else{
                MPI_Send(&n_loc_remote, 1, MPI_PANGULU_EXBLOCK_IDX, target_rank, 0, MPI_COMM_WORLD);
                MPI_Send(&columnpointer[PANGULU_MIN(col_per_rank * target_rank, *n)], n_loc_remote + 1, MPI_PANGULU_EXBLOCK_PTR, target_rank, 1, MPI_COMM_WORLD);
                // MPI_Send(
                //     &rowindex[(*distcsc_proc_nnzptr)[target_rank]], 
                //     (*distcsc_proc_nnzptr)[target_rank+1] - (*distcsc_proc_nnzptr)[target_rank], 
                //     MPI_PANGULU_EXBLOCK_IDX, target_rank, 2, MPI_COMM_WORLD
                // );
                pangulu_cm_send(&rowindex[(*distcsc_proc_nnzptr)[target_rank]], sizeof(pangulu_exblock_idx) * ((*distcsc_proc_nnzptr)[target_rank+1] - (*distcsc_proc_nnzptr)[target_rank]), target_rank, 2, 10);
                if(distcsc_value){
                    // MPI_Send(
                    //     &value_csc[(*distcsc_proc_nnzptr)[target_rank]],
                    //     (*distcsc_proc_nnzptr)[target_rank+1] - (*distcsc_proc_nnzptr)[target_rank],
                    //     MPI_VAL_TYPE, target_rank, 3, MPI_COMM_WORLD
                    // );
                    pangulu_cm_send(&value_csc[(*distcsc_proc_nnzptr)[target_rank]], sizeof(calculate_type) * ((*distcsc_proc_nnzptr)[target_rank+1] - (*distcsc_proc_nnzptr)[target_rank]), target_rank, 3, 10);
                    // printf("Sent %d->%d\n", rank, target_rank);
                }else{
                    int nouse = 0;
                    MPI_Send(&nouse, 1, MPI_INT, target_rank, 4, MPI_COMM_WORLD);
                }
            }
    // printf("#%d 1.3.3\n", rank);
        }
        MPI_Bcast(*distcsc_proc_nnzptr, nproc+1, MPI_PANGULU_EXBLOCK_PTR, root_rank, MPI_COMM_WORLD);
        if(rootproc_free_originmatrix){
            pangulu_free(__FILE__, __LINE__, columnpointer);
            pangulu_free(__FILE__, __LINE__, rowindex);
            if(distcsc_value){
                pangulu_free(__FILE__, __LINE__, value_csc);
            }
        }
    // printf("#%d 1.3.4\n", rank);
    }else{
    // printf("#%d 1.4.1\n", rank);
        MPI_Status mpi_stat;
        MPI_Recv(n_loc, 1, MPI_PANGULU_EXBLOCK_IDX, root_rank, 0, MPI_COMM_WORLD, &mpi_stat);
    // printf("#%d n_loc = %d\n", rank, *n_loc);
        
        *distcsc_pointer = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (*n_loc + 1));
        MPI_Recv(*distcsc_pointer, *n_loc+1, MPI_PANGULU_EXBLOCK_PTR, root_rank, 1, MPI_COMM_WORLD, &mpi_stat);
        pangulu_exblock_idx col_offset = (*distcsc_pointer)[0];
        for(pangulu_exblock_idx col = 0; col < *n_loc + 1; col++){
            (*distcsc_pointer)[col] -= col_offset;
        }
        pangulu_exblock_ptr nnz_loc = (*distcsc_pointer)[*n_loc];
        *distcsc_index = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * nnz_loc);
        // MPI_Recv(*distcsc_index, nnz_loc, MPI_PANGULU_EXBLOCK_IDX, root_rank, 2, MPI_COMM_WORLD, &mpi_stat);
        pangulu_cm_recv(*distcsc_index, sizeof(pangulu_exblock_idx) * nnz_loc, root_rank, 2, 10);
    // printf("#%d 1.4.2\n", rank);

        MPI_Probe(root_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &mpi_stat);
        // printf("Probe\n");
        if((mpi_stat.MPI_TAG%10) == 3){
            *distcsc_value = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz_loc);
            // MPI_Recv(*distcsc_value, nnz_loc, MPI_VAL_TYPE, root_rank, 3, MPI_COMM_WORLD, &mpi_stat);
            pangulu_cm_recv(*distcsc_value, sizeof(calculate_type) * nnz_loc, root_rank, 3, 10);
        }else if(mpi_stat.MPI_TAG == 4){
            int nouse = 0;
            MPI_Recv(&nouse, 1, MPI_INT, root_rank, 4, MPI_COMM_WORLD, &mpi_stat);
        }
    // printf("#%d 1.4.25\n", rank);

        MPI_Bcast(*distcsc_proc_nnzptr, nproc+1, MPI_PANGULU_EXBLOCK_PTR, root_rank, MPI_COMM_WORLD);
    // printf("#%d 1.4.3\n", rank);
    }
    // printf("#%d 1.5\n", rank);

    // printf("[PanguLU LOG] pangulu_cm_distribute_csc_to_distcsc time : %.2lf ms\n", pangulu_time_stop(&start_time) * 1e3);
}

void pangulu_convert_csr_to_csc(
    int free_csrmatrix,
    pangulu_exblock_idx n,
    pangulu_exblock_ptr** csr_pointer,
    pangulu_exblock_idx** csr_index,
    calculate_type** csr_value,
    pangulu_exblock_ptr** csc_pointer,
    pangulu_exblock_idx** csc_index,
    calculate_type** csc_value
){
    struct timeval start_time;
    pangulu_time_start(&start_time);

    pangulu_exblock_ptr* rowpointer = *csr_pointer;
    pangulu_exblock_idx* columnindex = *csr_index;
    calculate_type* value = NULL;
    if(csr_value){
        value = *csr_value;
    }

    pangulu_exblock_ptr nnz = rowpointer[n];
    
    pangulu_exblock_ptr* columnpointer = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1));
    pangulu_exblock_ptr* aid_ptr_arr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1));
    pangulu_exblock_idx* rowindex = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * nnz);
    calculate_type* value_csc = NULL;
    if(value){
        value_csc = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz);
    }

    memset(columnpointer, 0, sizeof(pangulu_exblock_ptr) * (n + 1));
    for(pangulu_exblock_idx row = 0; row < n; row++){
        for(pangulu_exblock_ptr idx = rowpointer[row]; idx < rowpointer[row+1]; idx++){
            pangulu_exblock_idx col = columnindex[idx];
            columnpointer[col+1]++;
        }
    }
    for(pangulu_exblock_idx col = 0; col < n; col++){
        columnpointer[col+1] += columnpointer[col];
    }
    memcpy(aid_ptr_arr, columnpointer, sizeof(pangulu_exblock_ptr) * (n + 1));

    if(value){
        for(pangulu_exblock_idx row = 0; row < n; row++){
            for(pangulu_exblock_ptr idx = rowpointer[row]; idx < rowpointer[row+1]; idx++){
                pangulu_exblock_idx col = columnindex[idx];
                rowindex[aid_ptr_arr[col]] = row;
                value_csc[aid_ptr_arr[col]] = value[idx];
                aid_ptr_arr[col]++;
            }
        }
    }else{
        for(pangulu_exblock_idx row = 0; row < n; row++){
            for(pangulu_exblock_ptr idx = rowpointer[row]; idx < rowpointer[row+1]; idx++){
                pangulu_exblock_idx col = columnindex[idx];
                rowindex[aid_ptr_arr[col]] = row;
                aid_ptr_arr[col]++;
            }
        }
    }

    pangulu_free(__FILE__, __LINE__, aid_ptr_arr);
    if(free_csrmatrix){
        pangulu_free(__FILE__, __LINE__, *csr_pointer);
        pangulu_free(__FILE__, __LINE__, *csr_index);
        pangulu_free(__FILE__, __LINE__, *csr_value);
        *csr_pointer = NULL;
        *csr_index = NULL;
        *csr_value = NULL;
    }
    *csc_pointer = columnpointer;
    *csc_index = rowindex;
    if(csc_value){
        *csc_value = value_csc;
    }
    // printf("[PanguLU LOG] pangulu_convert_csr_to_csc time : %.2lf ms\n", pangulu_time_stop(&start_time) * 1e3);
}

void pangulu_convert_ordered_halfsymcsc_to_csc_struct(
    int free_halfmatrix,
    int if_colsort,
    pangulu_exblock_idx n,
    pangulu_exblock_ptr** half_csc_pointer,
    pangulu_exblock_idx** half_csc_index,
    pangulu_exblock_ptr** csc_pointer,
    pangulu_exblock_idx** csc_index
){
    pangulu_exblock_ptr* half_ptr = *half_csc_pointer;
    pangulu_exblock_idx* half_idx = *half_csc_index;
    pangulu_exblock_ptr nnz = half_ptr[n];
    pangulu_exblock_ptr* at_ptr = (pangulu_exblock_ptr*)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1));
    memset(at_ptr, 0, sizeof(pangulu_exblock_ptr) * (n + 1));
    pangulu_exblock_idx* at_idx = (pangulu_exblock_idx*)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * nnz);
    for (pangulu_exblock_idx col = 0; col < n; col++) {
        for (pangulu_exblock_ptr row_idx = half_ptr[col]; row_idx < half_ptr[col + 1]; row_idx++) {
            pangulu_exblock_idx row = half_idx[row_idx];
            at_ptr[row + 1]++;
        }
    }
    for (pangulu_exblock_idx i = 1; i <= n; i++) {
        at_ptr[i] += at_ptr[i - 1];
    }

    pangulu_exblock_ptr* a_aid_ptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1));
    memcpy(a_aid_ptr, half_ptr, sizeof(pangulu_exblock_ptr) * (n + 1));
    pangulu_exblock_ptr* at_aid_ptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1));
    memcpy(at_aid_ptr, at_ptr, sizeof(pangulu_exblock_ptr) * (n + 1));
    pangulu_exblock_idx stride = 10000;
    // printf("stride = %d\n", stride);
    pangulu_exblock_idx n_tile = PANGULU_ICEIL(n,stride);
    for(pangulu_exblock_idx i_tile = 0; i_tile < n_tile; i_tile++){
        pangulu_exblock_idx col_end = PANGULU_MIN((i_tile + 1) * stride, n);
        for(pangulu_exblock_idx row = 0; row < n; row++){
            pangulu_exblock_idx col;
            while((a_aid_ptr[row] < half_ptr[row+1]) && ((col = half_idx[a_aid_ptr[row]]) < col_end)){
                at_idx[at_aid_ptr[col]] = row;
                at_aid_ptr[col]++;
                a_aid_ptr[row]++;
            }
        }
    }
    pangulu_free(__FILE__, __LINE__, at_aid_ptr);
    pangulu_free(__FILE__, __LINE__, a_aid_ptr);

    pangulu_exblock_ptr* full_ptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1));
    pangulu_exblock_idx* full_idx = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * (2 * nnz - n));
    full_ptr[0] = 0;
    for(pangulu_exblock_idx row = 0; row < n; row++){
        full_ptr[row + 1] = (half_ptr[row + 1] - half_ptr[row]) + (at_ptr[row + 1] - at_ptr[row]) - 1;
    }
    for(pangulu_exblock_idx row = 0; row < n; row++){
        full_ptr[row + 1] += full_ptr[row];
    }
    #pragma omp parallel for
    for(pangulu_exblock_idx row = 0; row < n; row++){
        memcpy(&full_idx[full_ptr[row]], &at_idx[at_ptr[row]], sizeof(pangulu_exblock_idx) * (at_ptr[row+1] - at_ptr[row]));
        memcpy(&full_idx[full_ptr[row] + (at_ptr[row+1] - at_ptr[row]) - 1], &half_idx[half_ptr[row]], sizeof(pangulu_exblock_idx) * (half_ptr[row+1] - half_ptr[row]));
    }
    
    pangulu_free(__FILE__, __LINE__, at_ptr);
    pangulu_free(__FILE__, __LINE__, at_idx);

    if (free_halfmatrix) {
        pangulu_free(__FILE__, __LINE__, *half_csc_pointer);
        pangulu_free(__FILE__, __LINE__, *half_csc_index);
        *half_csc_pointer = NULL;
        *half_csc_index = NULL;
    }

    if(if_colsort){
        pangulu_int32_t rank, nproc;
        pangulu_cm_rank(&rank);
        pangulu_cm_size(&nproc);
        int nthread_sort = 2;
        bind_to_core((rank * nthread_sort) % sysconf(_SC_NPROCESSORS_ONLN));
        #pragma omp parallel num_threads(nthread_sort)
        {
            bind_to_core((rank * nthread_sort + omp_get_thread_num()) % sysconf(_SC_NPROCESSORS_ONLN));
        }
        pangulu_sort_exblock_struct(n, at_ptr, at_idx, nthread_sort);
        bind_to_core(rank % sysconf(_SC_NPROCESSORS_ONLN));
    }
    
    *csc_pointer = full_ptr;
    *csc_index = full_idx;
}

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
){
    struct timeval start_time;
    pangulu_time_start(&start_time);

    pangulu_exblock_idx block_length = PANGULU_ICEIL(n , nb);

    for(pangulu_exblock_idx sp = 0; sp < block_length; sp++)
    {
        pangulu_exblock_ptr ssi = struct_bcsc_struct_pointer[sp];
        for(pangulu_exblock_ptr vsi = value_bcsc_struct_pointer[sp]; vsi < value_bcsc_struct_pointer[sp + 1]; vsi++)
        {
            while((struct_bcsc_struct_index[ssi] != value_bcsc_struct_index[vsi]) && (ssi < struct_bcsc_struct_pointer[sp+1]))
            {
                ssi++;
            }
            if(ssi >= struct_bcsc_struct_pointer[sp+1]){
                break;
            }
            for(pangulu_exblock_idx ip = 0; ip < nb; ip++)
            {
                pangulu_inblock_ptr sii = struct_bcsc_inblock_pointers[ssi][ip];
                for(pangulu_exblock_ptr vii = value_bcsc_inblock_pointers[vsi][ip]; vii < value_bcsc_inblock_pointers[vsi][ip + 1]; vii++)
                {
                    while((struct_bcsc_inblock_indeces[ssi][sii] != value_bcsc_inblock_indeces[vsi][vii]) && (sii < struct_bcsc_inblock_pointers[ssi][ip+1]))
                    {
                        sii++;
                    }
                    if(sii >= struct_bcsc_inblock_pointers[ssi][ip+1]){
                        break;
                    }
                    struct_bcsc_values[ssi][sii] = value_bcsc_values[vsi][vii];
                }
            }
        }
    }
    
    if(free_valuebcsc)
    {
        if(value_bcsc_struct_pointer[block_length] > 0){
            pangulu_free(__FILE__, __LINE__, value_bcsc_inblock_pointers[0]);            
        }
        pangulu_free(__FILE__, __LINE__, value_bcsc_inblock_pointers);
        pangulu_free(__FILE__, __LINE__, value_bcsc_inblock_indeces);
        pangulu_free(__FILE__, __LINE__, value_bcsc_values);
        pangulu_free(__FILE__, __LINE__, value_bcsc_struct_pointer);
        pangulu_free(__FILE__, __LINE__, value_bcsc_struct_index);
        pangulu_free(__FILE__, __LINE__, value_bcsc_struct_nnzptr);
    }

    // printf("[PanguLU LOG] pangulu_convert_bcsc_fill_value_to_struct time : %.2lf ms\n", pangulu_time_stop(&start_time) * 1e3);
}

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

// void pangulu_time_start(pangulu_common *common)
// {
//     gettimeofday(&(common->start_time), NULL);
// }

// void pangulu_time_stop(pangulu_common *common)
// {
//     gettimeofday(&(common->stop_time), NULL);
// }

void pangulu_time_start(struct timeval* start){
    gettimeofday(start, NULL);
}
double pangulu_time_stop(struct timeval* start){
    struct timeval end;
    gettimeofday(&end, NULL);
    double time = ((double)end.tv_sec - start->tv_sec) + ((double)end.tv_usec - start->tv_usec) * 1e-6;
    return time;
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

void pangulu_transpose_pangulu_smatrix_csc_to_csr(pangulu_smatrix *s)
{
    for (pangulu_int64_t i = 0; i < s->nnz; i++)
    {
        pangulu_int64_t index = s->csc_to_csr_index[i];
        s->value[index] = s->value_csc[i];
    }
}
void pangulu_transpose_pangulu_smatrix_csr_to_csc(pangulu_smatrix *s)
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
    #pragma omp parallel for
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