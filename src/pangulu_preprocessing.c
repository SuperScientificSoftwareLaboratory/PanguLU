#include "pangulu_common.h"

int cmp_int32t_asc(const void *a, const void *b)
{
    return *(int32_t *)a - *(int32_t *)b;
}

void pangulu_preprocessing(pangulu_block_common *block_common,
                           pangulu_block_smatrix *block_smatrix,
                           pangulu_origin_smatrix *reorder_matrix,
                           pangulu_int64_t nthread)
{
    struct timeval start_time;
    pangulu_exblock_idx n = block_common->n;
    pangulu_exblock_idx n_loc = 0;
    pangulu_exblock_idx nb = block_common->nb;
    pangulu_exblock_idx block_length = block_common->block_length;
    pangulu_int32_t rank = block_common->rank;
    pangulu_int32_t nproc = block_common->sum_rank_size;

    pangulu_exblock_ptr* distcsc_proc_nnzptr;
    pangulu_exblock_ptr* distcsc_symbolic_proc_nnzptr;
    pangulu_exblock_ptr* distcsc_pointer;
    pangulu_exblock_idx* distcsc_index;
    calculate_type* distcsc_value;
    pangulu_exblock_ptr* bcsc_nofill_pointer;
    pangulu_exblock_idx* bcsc_nofill_index;
    pangulu_exblock_ptr* bcsc_nofill_blknnzptr;
    pangulu_inblock_ptr** bcsc_nofill_inblk_pointers;
    pangulu_inblock_idx** bcsc_nofill_inblk_indeces;
    calculate_type** bcsc_nofill_inblk_values;
    pangulu_exblock_ptr* bcsc_pointer;
    pangulu_exblock_idx* bcsc_index;
    pangulu_exblock_ptr* bcsc_blknnzptr;
    pangulu_inblock_ptr** bcsc_inblk_pointers;
    pangulu_inblock_idx** bcsc_inblk_indeces;
    calculate_type** bcsc_inblk_values;

    pangulu_exblock_ptr* bcsc_related_pointer;
    pangulu_exblock_idx* bcsc_related_index;
    pangulu_uint64_t* bcsc_related_draft_info;

    const pangulu_int64_t preprocess_ompnum = nthread;
    pangulu_int64_t preprocess_ompnum_sort = preprocess_ompnum;
    pangulu_int64_t preprocess_ompnum_fill_columnindex = preprocess_ompnum;
    pangulu_int64_t preprocess_ompnum_set_value = preprocess_ompnum;
    pangulu_int64_t preprocess_ompnum_separate_block = preprocess_ompnum;
    pangulu_int64_t preprocess_ompnum_send_block = preprocess_ompnum;

    pangulu_int32_t p = block_common->p;
    pangulu_int32_t q = block_common->q;
    pangulu_block_info_pool *BIP = block_smatrix->BIP;

    pangulu_exblock_ptr *symbolic_rowpointer = NULL;
    pangulu_exblock_idx *symbolic_columnindex = NULL;

    bind_to_core((rank) % sysconf(_SC_NPROCESSORS_ONLN));

    // rank 0 only
    pangulu_int64_t block_num;
    char *block_csr;
    pangulu_int64_t *block_nnz_pt;
    // rank 0 only end

    if (rank == 0)
    {
        symbolic_rowpointer = block_smatrix->symbolic_rowpointer;
        symbolic_columnindex = block_smatrix->symbolic_columnindex;
    }

    pangulu_int64_t A_nnz_rowpointer_num = 0;
    pangulu_int32_t sum_rank_size = block_common->sum_rank_size;

    pangulu_int64_t *save_send_rank_flag = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * sum_rank_size);
    for (pangulu_int64_t i = 0; i < sum_rank_size; i++)
    {
        save_send_rank_flag[i] = 0;
    }

    pangulu_int32_t *grid_process_id = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * p * q);
    for (pangulu_int32_t i = 0; i < p; i++)
    {
        pangulu_int32_t offset = i % p;
        for (pangulu_int32_t j = 0; j < q; j++)
        {
            pangulu_int32_t now_rank = (j % q + offset * q);
            grid_process_id[i * q + j] = now_rank;
        }
    }

    pangulu_int64_t *every_rank_block_num = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * sum_rank_size);
    pangulu_int64_t *every_rank_block_nnz = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * sum_rank_size);

    pangulu_inblock_ptr *block_smatrix_non_zero_vector_L = NULL;
    pangulu_inblock_ptr *block_smatrix_non_zero_vector_U = NULL;

    if (rank == 0)
    {
        block_smatrix_non_zero_vector_L = block_smatrix->block_smatrix_non_zero_vector_l;
        block_smatrix_non_zero_vector_U = block_smatrix->block_smatrix_non_zero_vector_u;
    }
    else
    {
        block_smatrix_non_zero_vector_L = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * block_length);
        block_smatrix_non_zero_vector_U = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * block_length);
    }

    pangulu_int64_t rank0_BIP_length;
    pangulu_int64_t rank0_BIP_capacity;
    pangulu_int64_t rank0_BIP_index_upper_bound;
    if (rank == 0)
    {
        pangulu_inblock_ptr *block_smatrix_nnzA_num = block_smatrix->block_smatrix_nnza_num;
        for (pangulu_int64_t i = 0; i < block_length * block_length; i++)
        {
            if (block_smatrix_nnzA_num[i] != 0)
            {
                pangulu_bip_set(i, BIP)->block_smatrix_nnza_num = block_smatrix_nnzA_num[i];
            }
        }
        pangulu_free(__FILE__, __LINE__, block_smatrix_nnzA_num);
        block_smatrix->block_smatrix_nnza_num = NULL;
        rank0_BIP_length = BIP->length;
        rank0_BIP_capacity = BIP->capacity;
        rank0_BIP_index_upper_bound = BIP->index_upper_bound;
    }
    MPI_Bcast(&rank0_BIP_length, 1, MPI_PANGULU_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rank0_BIP_capacity, 1, MPI_PANGULU_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rank0_BIP_index_upper_bound, 1, MPI_PANGULU_INT64_T, 0, MPI_COMM_WORLD);
    if (rank != 0)
    {
        if (BIP->length != 0)
        {
            printf(PANGULU_E_BIP_NOT_EMPTY);
            pangulu_exit(1);
        }
        BIP->capacity = rank0_BIP_capacity;
        BIP->length = rank0_BIP_length;
        BIP->index_upper_bound = rank0_BIP_index_upper_bound;
        BIP->data = (pangulu_block_info *)pangulu_realloc(__FILE__, __LINE__, BIP->data, sizeof(pangulu_block_info) * BIP->capacity);
    }
    MPI_Bcast((char *)(BIP->data), sizeof(pangulu_block_info) * BIP->length, MPI_CHAR, 0, MPI_COMM_WORLD);             // containing block_smatrix_nnzA_num
    MPI_Bcast(BIP->block_map, PANGULU_BIP_MAP_LENGTH(BIP->index_upper_bound), MPI_PANGULU_INT64_T, 0, MPI_COMM_WORLD); // containing block_smatrix_nnzA_num


    if(rank == 0){
        pangulu_time_start(&start_time);
        pangulu_convert_csr_to_csc(
            1, n,
            &reorder_matrix->rowpointer,
            &reorder_matrix->columnindex,
            &reorder_matrix->value,
            &reorder_matrix->columnpointer,
            &reorder_matrix->rowindex,
            &reorder_matrix->value_csc
        );
        printf("[PanguLU Info] 6 PanguLU transpose reordered matrix time is %lf s.\n", pangulu_time_stop(&start_time));
        distcsc_pointer = reorder_matrix->columnpointer;
        distcsc_index = reorder_matrix->rowindex;
        distcsc_value = reorder_matrix->value_csc;
        reorder_matrix->columnpointer = NULL;
        reorder_matrix->rowindex = NULL;
        reorder_matrix->value_csc = NULL;
    }

    pangulu_cm_sync();

    double time_dist_prep = 0;

    if(rank == 0){
        pangulu_time_start(&start_time);
    }
    //  distribute to processes
    pangulu_cm_distribute_csc_to_distcsc(
        0, 1, &n, nb, &nproc, &n_loc,
        &distcsc_proc_nnzptr,
        &distcsc_pointer,
        &distcsc_index,
        &distcsc_value
    );

    pangulu_cm_sync();
    
    pangulu_cm_distribute_distcsc_to_distbcsc(
        1, 1, n, n_loc, nb,
        distcsc_proc_nnzptr,
        distcsc_pointer,
        distcsc_index,
        distcsc_value,
        &bcsc_nofill_pointer,
        &bcsc_nofill_index,
        &bcsc_nofill_blknnzptr,
        &bcsc_nofill_inblk_pointers,
        &bcsc_nofill_inblk_indeces,
        &bcsc_nofill_inblk_values
    );

    pangulu_cm_sync();

    if(rank == 0){
        time_dist_prep += pangulu_time_stop(&start_time);
    }

    if(rank == 0){
        //  generate full symbolic struct
        pangulu_sort_exblock_struct(n, block_smatrix->symbolic_rowpointer, block_smatrix->symbolic_columnindex, 0);
        pangulu_time_start(&start_time);
        pangulu_convert_ordered_halfsymcsc_to_csc_struct(
            1, 0, n,
            &block_smatrix->symbolic_rowpointer,
            &block_smatrix->symbolic_columnindex,
            &block_smatrix->symbolic_rowpointer,
            &block_smatrix->symbolic_columnindex
        );
        printf("[PanguLU Info] 7 PanguLU generate full symbolic matrix time is %lf s.\n", pangulu_time_stop(&start_time));
    }

    pangulu_cm_sync_asym(0);
    pangulu_cm_sync();

    if(rank == 0){
        pangulu_time_start(&start_time);
    }

    //  distribute to processes
    pangulu_cm_distribute_csc_to_distcsc(
        0, 1, &n, nb, &nproc, &n_loc,
        &distcsc_symbolic_proc_nnzptr,
        &block_smatrix->symbolic_rowpointer,
        &block_smatrix->symbolic_columnindex,
        NULL
    );
    
    pangulu_cm_sync();
    
    // pangulu_sort_exblock_struct(n_loc, block_smatrix->symbolic_rowpointer, block_smatrix->symbolic_columnindex, 0);

    // pangulu_cm_sync_asym(nproc - 1);

    pangulu_cm_distribute_distcsc_to_distbcsc(
        1, 1, n, n_loc, nb,
        distcsc_symbolic_proc_nnzptr,
        block_smatrix->symbolic_rowpointer,
        block_smatrix->symbolic_columnindex,
        NULL,
        &bcsc_pointer,
        &bcsc_index,
        &bcsc_blknnzptr,
        &bcsc_inblk_pointers,
        &bcsc_inblk_indeces,
        &bcsc_inblk_values
    );

    pangulu_cm_sync();

    pangulu_convert_bcsc_fill_value_to_struct(
        1, n, nb,
        bcsc_nofill_pointer,
        bcsc_nofill_index,
        bcsc_nofill_blknnzptr,
        bcsc_nofill_inblk_pointers,
        bcsc_nofill_inblk_indeces,
        bcsc_nofill_inblk_values,
        bcsc_pointer,
        bcsc_index,
        bcsc_blknnzptr,
        bcsc_inblk_pointers,
        bcsc_inblk_indeces,
        bcsc_inblk_values
    );
    bcsc_nofill_pointer = NULL;
    bcsc_nofill_index = NULL;
    bcsc_nofill_blknnzptr = NULL;
    bcsc_nofill_inblk_pointers = NULL;
    bcsc_nofill_inblk_indeces = NULL;
    bcsc_nofill_inblk_values = NULL;

    pangulu_bcast_vector(block_smatrix_non_zero_vector_L, block_length, 0);
    pangulu_bcast_vector(block_smatrix_non_zero_vector_U, block_length, 0);

    every_rank_block_num[rank] = bcsc_pointer[block_length];
    every_rank_block_nnz[rank] = bcsc_blknnzptr[bcsc_pointer[block_length]];
    pangulu_free(__FILE__, __LINE__, bcsc_blknnzptr);

    if (rank == 0)
    {
        MPI_Status mpi_stat;
        for (int32_t i = 1; i < sum_rank_size; i++)
        {
            MPI_Recv(&every_rank_block_num[i], 1, MPI_PANGULU_INT64_T, i, i, MPI_COMM_WORLD, &mpi_stat);
            MPI_Recv(&every_rank_block_nnz[i], 1, MPI_PANGULU_INT64_T, i, sum_rank_size + i, MPI_COMM_WORLD, &mpi_stat);
        }
    }
    else
    {
        MPI_Send(&every_rank_block_num[rank], 1, MPI_PANGULU_INT64_T, 0, rank, MPI_COMM_WORLD);
        MPI_Send(&every_rank_block_nnz[rank], 1, MPI_PANGULU_INT64_T, 0, sum_rank_size + rank, MPI_COMM_WORLD);
    }
    pangulu_bcast_vector_int64(every_rank_block_num, sum_rank_size, 0);
    pangulu_bcast_vector_int64(every_rank_block_nnz, sum_rank_size, 0);

    pangulu_int64_t current_rank_block_count = every_rank_block_num[rank];
    pangulu_int64_t current_rank_nnz_count = every_rank_block_nnz[rank];
    for (pangulu_exblock_idx row = 0; row < block_length; row++)
    {
        for (pangulu_exblock_idx col = 0; col < block_length; col++)
        {
            if (pangulu_bip_get(row * block_length + col, BIP)->block_smatrix_nnza_num != 0)
            {
                int32_t now_rank = (row % p) * q + (col % q);
                if (now_rank == rank)
                {
                    pangulu_bip_set(row * block_length + col, BIP)->sum_flag_block_num++;
                }
            }
        }
    }

    pangulu_free(__FILE__, __LINE__, every_rank_block_num);
    pangulu_free(__FILE__, __LINE__, every_rank_block_nnz);

    pangulu_smatrix *Big_smatrix_value = (pangulu_smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_smatrix) * current_rank_block_count);
    for (pangulu_int64_t i = 0; i < current_rank_block_count; i++)
    {
        pangulu_init_pangulu_smatrix(&Big_smatrix_value[i]);
    }

    current_rank_block_count = 0;
    current_rank_nnz_count = 0;

    pangulu_exblock_ptr nzblk_current_rank = 0;
    pangulu_exblock_ptr nnz_current_rank = 0;

    for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
    {
        for (pangulu_exblock_ptr bidx = bcsc_pointer[bcol]; bidx < bcsc_pointer[bcol + 1]; bidx++)
        {
            pangulu_exblock_idx brow = bcsc_index[bidx];
            pangulu_smatrix *current = &Big_smatrix_value[current_rank_block_count];
            current->row = nb;
            current->column = nb;
            current->nnz = bcsc_inblk_pointers[bidx][nb];
            current->columnpointer = bcsc_inblk_pointers[bidx];
            current->rowindex = bcsc_inblk_indeces[bidx];
            current->value_csc = bcsc_inblk_values[bidx];
            pangulu_bip_set(brow * block_length + bcol, BIP)->mapper_a = current_rank_block_count;
            current_rank_block_count++;
        }
    }
    pangulu_free(__FILE__, __LINE__, bcsc_pointer);
    pangulu_free(__FILE__, __LINE__, bcsc_index);
    pangulu_free(__FILE__, __LINE__, bcsc_inblk_pointers);
    pangulu_free(__FILE__, __LINE__, bcsc_inblk_indeces);
    pangulu_free(__FILE__, __LINE__, bcsc_inblk_values);


    for (pangulu_int64_t offset_block_index = 0; offset_block_index < p * q; offset_block_index++)
    {
        pangulu_int64_t offset_block_row = offset_block_index / q;
        pangulu_int64_t offset_block_col = offset_block_index % q;
        pangulu_int64_t now_rank = grid_process_id[offset_block_index];
        if (now_rank == rank)
        {
            for (pangulu_int64_t level = 0; level < block_length; level++)
            {
                pangulu_int64_t offset_row = calculate_offset(offset_block_row, level, p);
                pangulu_int64_t offset_col = calculate_offset(offset_block_col, level, q);
                for (pangulu_int64_t i = offset_col + level; i < block_length; i += q)
                {
                    pangulu_int64_t find_index = pangulu_bip_get(level * block_length + i, BIP)->block_smatrix_nnza_num;
                    if (find_index != 0)
                    {
                        pangulu_bip_set(level * block_length + i, BIP)->tmp_save_block_num = find_index;
                    }
                }
                for (pangulu_int64_t i = offset_row + level; i < block_length; i += p)
                {
                    pangulu_int64_t find_index = pangulu_bip_get(i * block_length + level, BIP)->block_smatrix_nnza_num;
                    if (find_index != 0)
                    {
                        pangulu_bip_set(i * block_length + level, BIP)->tmp_save_block_num = find_index;
                    }
                }
            }
        }
    }

    pangulu_exblock_ptr *U_rowpointer = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (block_length + 1));
    pangulu_inblock_ptr *L_columnpointer = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (block_length + 1));

    U_rowpointer[0] = 0;
    for (pangulu_int64_t i = 0; i < block_length + 1; i++)
    {
        L_columnpointer[i] = 0;
    }
    for (pangulu_int64_t level = 0; level < block_length; level++)
    {
        U_rowpointer[level + 1] = U_rowpointer[level];
        for (pangulu_int64_t i = level; i < block_length; i++)
        {
            if (pangulu_bip_get(level * block_length + i, BIP)->tmp_save_block_num != -1)
            {
                U_rowpointer[level + 1]++;
            }
        }
    }
    for (pangulu_int64_t level = 0; level < block_length; level++)
    {
        for (pangulu_int64_t i = 0; i <= level; i++)
        {
            if (pangulu_bip_get(level * block_length + i, BIP)->tmp_save_block_num != -1)
            {
                L_columnpointer[i + 1]++;
            }
        }
    }
    for (pangulu_int64_t i = 0; i < block_length; i++)
    {
        L_columnpointer[i + 1] += L_columnpointer[i];
    }

    pangulu_exblock_idx *U_columnindex = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * U_rowpointer[block_length]);
    pangulu_inblock_idx *L_rowindex = (pangulu_inblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx) * L_columnpointer[block_length]);

    for (pangulu_int64_t level = 0; level < block_length; level++)
    {
        for (pangulu_int64_t i = level; i < block_length; i++)
        {
            if (pangulu_bip_get(level * block_length + i, BIP)->tmp_save_block_num != -1)
            {
                U_columnindex[U_rowpointer[level]] = i;
                U_rowpointer[level]++;
            }
        }
        for (pangulu_int64_t i = 0; i <= level; i++)
        {
            if (pangulu_bip_get(level * block_length + i, BIP)->tmp_save_block_num != -1)
            {
                L_rowindex[L_columnpointer[i]] = level;
                L_columnpointer[i]++;
            }
        }
    }
    for (pangulu_int64_t i = block_length; i > 0; i--)
    {
        L_columnpointer[i] = L_columnpointer[i - 1];
    }
    for (pangulu_int64_t i = block_length; i > 0; i--)
    {
        U_rowpointer[i] = U_rowpointer[i - 1];
    }
    L_columnpointer[0] = 0;
    U_rowpointer[0] = 0;

    pangulu_int64_t every_level_length = block_common->every_level_length;

    pangulu_int64_t *level_index = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (block_length));
    pangulu_int64_t *level_index_reverse = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (block_length));

    for (pangulu_int64_t i = 0; i < block_length; i++)
    {
        level_index[i] = i;
    }

    pangulu_int64_t *tmp_diggonal_task_id = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (block_length));
    pangulu_int64_t now_num = 0;
    for (pangulu_int64_t i = 0; i < block_length; i++)
    {
        tmp_diggonal_task_id[i] = 0;
    }

    for (pangulu_int64_t i = 0; i < block_length; i++)
    {
        for (pangulu_int64_t j = i + 1; j < block_length; j++)
        {
            if ((pangulu_bip_get(i * block_length + j, BIP)->block_smatrix_nnza_num != 0) || (pangulu_bip_get(j * block_length + i, BIP)->block_smatrix_nnza_num != 0))
            {
                tmp_diggonal_task_id[j]++;
            }
        }
    }
    for (pangulu_int64_t i = 0; i < block_length; i++)
    {
        if (tmp_diggonal_task_id[i] == 0)
        {
            level_index[now_num++] = i;
        }
    }
    for (pangulu_int64_t i = 0; i < block_length; i++)
    {
        pangulu_int64_t now_flag = level_index[i];
        if (i >= now_num)
        {
            printf("error in level " FMT_PANGULU_INT64_T "\n", i);
            continue;
        }
        for (pangulu_int64_t j = now_flag + 1; j < block_length; j++)
        {
            if ((pangulu_bip_get(now_flag * block_length + j, BIP)->block_smatrix_nnza_num != 0) || (pangulu_bip_get(j * block_length + now_flag, BIP)->block_smatrix_nnza_num != 0))
            {
                tmp_diggonal_task_id[j]--;
                if (tmp_diggonal_task_id[j] == 0)
                {
                    level_index[now_num++] = j;
                }
                if (tmp_diggonal_task_id[j] < 0)
                {
                    printf("error in now flag " FMT_PANGULU_INT64_T " j " FMT_PANGULU_INT64_T "\n", now_flag, j);
                }
            }
        }
    }
    pangulu_free(__FILE__, __LINE__, tmp_diggonal_task_id);
    for (pangulu_int64_t i = 0; i < block_length; i++)
    {
        level_index_reverse[level_index[i]] = i;
    }

    pangulu_int64_t U_smatrix_nzz = 0;
    pangulu_int64_t L_smatrix_nzz = 0;
    for (pangulu_int64_t i = 0; i < block_length; i++)
    {
        U_smatrix_nzz = PANGULU_MAX(U_rowpointer[i + 1] - U_rowpointer[i], U_smatrix_nzz);
    }
    for (pangulu_int64_t i = 0; i < block_length; i++)
    {
        L_smatrix_nzz = PANGULU_MAX(L_columnpointer[i + 1] - L_columnpointer[i], L_smatrix_nzz);
    }
    pangulu_int64_t *now_level_L_length = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * every_level_length);
    pangulu_int64_t *now_level_U_length = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * every_level_length);

    for (pangulu_int64_t i = 0; i < every_level_length; i++)
    {
        now_level_L_length[i] = 0;
        now_level_U_length[i] = 0;
    }

    pangulu_int64_t *save_now_level_l = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * L_smatrix_nzz * every_level_length);
    pangulu_int64_t *save_now_level_u = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * U_smatrix_nzz * every_level_length);

    pangulu_int64_t now_nnz_L = 0;
    pangulu_int64_t now_nnz_U = 0;
    for (pangulu_int64_t i = 0; i < block_length; i++)
    {
        pangulu_int64_t mapper_level = level_index[i];
        if (i % every_level_length == 0)
        {
            now_nnz_L = 0;
            now_nnz_U = 0;
        }
        for (pangulu_int64_t j = U_rowpointer[mapper_level]; j < U_rowpointer[mapper_level + 1]; j++)
        {
            pangulu_int64_t mapper_index_U = mapper_level * block_length + U_columnindex[j];
            pangulu_bip_set(mapper_index_U, BIP)->mapper_lu = now_nnz_U++;
        }
        for (pangulu_int64_t j = L_columnpointer[mapper_level]; j < L_columnpointer[mapper_level + 1]; j++)
        {
            pangulu_int64_t mapper_index_L = L_rowindex[j] * block_length + mapper_level;
            pangulu_bip_set(mapper_index_L, BIP)->mapper_lu = now_nnz_L++;
        }
    }

    pangulu_int64_t MAX_all_nnzL = 0;
    pangulu_int64_t MAX_all_nnzU = 0;
    pangulu_int64_t *MAX_level_nnzL = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * L_smatrix_nzz * every_level_length);
    pangulu_int64_t *MAX_level_nnzU = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * U_smatrix_nzz * every_level_length);

    char *flag_save_L = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(char) * (L_smatrix_nzz + U_smatrix_nzz) * every_level_length);
    char *flag_save_U = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(char) * U_smatrix_nzz * every_level_length);

    block_smatrix->flag_save_l = flag_save_L;
    block_smatrix->flag_save_u = flag_save_U;

    for (pangulu_int64_t i = 0; i < L_smatrix_nzz * every_level_length; i++)
    {
        MAX_level_nnzL[i] = 0;
    }
    for (pangulu_int64_t i = 0; i < U_smatrix_nzz * every_level_length; i++)
    {
        MAX_level_nnzU[i] = 0;
    }
    pangulu_int64_t U_smatrix_index = 0;
    pangulu_int64_t L_smatrix_index = 0;

    for (pangulu_int64_t row = 0; row < block_length; row++)
    {
        pangulu_int64_t mapper_level = level_index[row];
        for (pangulu_int64_t index = U_rowpointer[mapper_level]; index < U_rowpointer[mapper_level + 1]; index++)
        {
            pangulu_int64_t col = U_columnindex[index];
            pangulu_int64_t find_index;
            if (mapper_level == col)
            {
                find_index = block_smatrix_non_zero_vector_U[mapper_level];
                if (find_index != 0)
                {
                    MAX_all_nnzU = PANGULU_MAX(MAX_all_nnzU, find_index);
                    U_smatrix_index++;
                }
            }
            else
            {
                find_index = pangulu_bip_get(mapper_level * block_length + col, BIP)->block_smatrix_nnza_num;
                if (find_index != 0)
                {
                    MAX_all_nnzU = PANGULU_MAX(MAX_all_nnzU, find_index);
                    pangulu_int64_t mapper_index_U = mapper_level * block_length + col;
                    pangulu_int64_t save_index = pangulu_bip_get(mapper_index_U, BIP)->mapper_lu;
                    MAX_level_nnzU[save_index] = PANGULU_MAX(MAX_level_nnzU[save_index], find_index);
                    U_smatrix_index++;
                }
            }
        }
    }

    for (pangulu_int64_t col = 0; col < block_length; col++)
    {
        pangulu_int64_t mapper_level = level_index[col];
        for (pangulu_int64_t index = L_columnpointer[mapper_level]; index < L_columnpointer[mapper_level + 1]; index++)
        {
            pangulu_int64_t row = L_rowindex[index];
            pangulu_int64_t find_index;
            if (row == mapper_level)
            {
                find_index = block_smatrix_non_zero_vector_L[mapper_level];
                if (find_index != 0)
                {
                    MAX_all_nnzL = PANGULU_MAX(MAX_all_nnzL, find_index);
                    L_smatrix_index++;
                }
            }
            else
            {
                find_index = pangulu_bip_get(row * block_length + mapper_level, BIP)->block_smatrix_nnza_num;
                if (find_index != 0)
                {
                    MAX_all_nnzL = PANGULU_MAX(MAX_all_nnzL, find_index);
                    pangulu_int64_t mapper_index_L = row * block_length + mapper_level;
                    pangulu_int64_t save_index = pangulu_bip_get(mapper_index_L, BIP)->mapper_lu;
                    MAX_level_nnzL[save_index] = PANGULU_MAX(MAX_level_nnzL[save_index], find_index);
                    L_smatrix_index++;
                }
            }
        }
    }

    pangulu_smatrix **U_value = (pangulu_smatrix **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_smatrix *) * U_smatrix_nzz * every_level_length);
    pangulu_smatrix **L_value = (pangulu_smatrix **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_smatrix *) * L_smatrix_nzz * every_level_length);

    for (pangulu_int64_t i = 0; i < U_smatrix_nzz * every_level_length; i++)
    {
        pangulu_smatrix *first_U = (pangulu_smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_smatrix));
        pangulu_init_pangulu_smatrix(first_U);
        pangulu_malloc_pangulu_smatrix_nnz_csc(first_U, nb, MAX_level_nnzU[i]);

#ifdef ADD_GPU_MEMORY
        pangulu_smatrix_cuda_memory_init(first_U, nb, MAX_level_nnzU[i]);
#endif
        U_value[i] = first_U;
    }
    for (pangulu_int64_t i = 0; i < L_smatrix_nzz * every_level_length; i++)
    {
        pangulu_smatrix *first_L = (pangulu_smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_smatrix));
        pangulu_init_pangulu_smatrix(first_L);
        pangulu_malloc_pangulu_smatrix_nnz_csc(first_L, nb, MAX_level_nnzL[i]);

#ifdef ADD_GPU_MEMORY
        pangulu_smatrix_cuda_memory_init(first_L, nb, MAX_level_nnzL[i]);

#endif
        L_value[i] = first_L;
    }

    pangulu_free(__FILE__, __LINE__, MAX_level_nnzL);
    pangulu_free(__FILE__, __LINE__, MAX_level_nnzU);

    pangulu_int64_t MAX_all_nnzX = 0;

    for (pangulu_int64_t i = 0; i < current_rank_block_count; i++)
    {
        pangulu_smatrix_add_more_memory_csr(&Big_smatrix_value[i]);

#ifdef GPU_OPEN
        pangulu_smatrix_add_cuda_memory(&Big_smatrix_value[i]);
        pangulu_smatrix_cuda_memcpy_a(&Big_smatrix_value[i]);
        pangulu_cuda_malloc((void **)&(Big_smatrix_value[i].d_left_sum), ((Big_smatrix_value[i].nnz)) * sizeof(calculate_type));
#endif

        MAX_all_nnzX = PANGULU_MAX(MAX_all_nnzX, Big_smatrix_value[i].nnz);
    }

#ifndef GPU_OPEN

    pangulu_int64_t *work_space = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (4 * nb + 8));
    for (pangulu_int64_t i = 0; i < block_length; i++)
    {

        pangulu_int64_t now_offset = i * block_length + i;
        pangulu_int64_t now_mapperA_offset = pangulu_bip_get(now_offset, BIP)->mapper_a;
        if (now_mapperA_offset != -1)
        {
            pangulu_malloc_smatrix_level(&Big_smatrix_value[now_mapperA_offset]);
            pangulu_init_level_array(&Big_smatrix_value[now_mapperA_offset], work_space);
        }
    }
    pangulu_free(__FILE__, __LINE__, work_space);
#endif

    pangulu_smatrix *calculate_L = (pangulu_smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_smatrix));
    pangulu_smatrix *calculate_U = (pangulu_smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_smatrix));
    pangulu_smatrix *calculate_X = (pangulu_smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_smatrix));

    pangulu_init_pangulu_smatrix(calculate_L);
    pangulu_init_pangulu_smatrix(calculate_U);
    pangulu_init_pangulu_smatrix(calculate_X);

#ifdef GPU_OPEN
    pangulu_smatrix_cuda_memory_init(calculate_L, nb, MAX_all_nnzL);
    pangulu_smatrix_cuda_memory_init(calculate_U, nb, MAX_all_nnzU);
    pangulu_smatrix_add_cuda_memory_u(calculate_U);
    pangulu_smatrix_cuda_memory_init(calculate_X, nb, MAX_all_nnzX);
#else
    calculate_L->row = nb;
    calculate_L->column = nb;
    calculate_L->nnz = MAX_all_nnzL;

    calculate_U->row = nb;
    calculate_U->column = nb;
    calculate_U->nnz = MAX_all_nnzU;

    calculate_X->row = nb;
    calculate_X->column = nb;
    calculate_X->nnz = MAX_all_nnzX;

#endif

    pangulu_malloc_pangulu_smatrix_value_csr(calculate_X, MAX_all_nnzX);
    pangulu_malloc_pangulu_smatrix_value_csc(calculate_X, MAX_all_nnzX);

    pangulu_int64_t diagonal_nnz = 0;
    pangulu_int64_t *mapper_diagonal_smatrix = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * block_length);

    for (pangulu_int64_t i = 0; i < block_length; i++)
    {
        mapper_diagonal_smatrix[i] = -1;
    }
    for (pangulu_int64_t i = 0; i < block_length; i++)
    {
        if (pangulu_bip_get(i * block_length + i, BIP)->tmp_save_block_num != -1)
        {
            mapper_diagonal_smatrix[i] = diagonal_nnz;
            diagonal_nnz++;
        }
    }

    pangulu_smatrix **diagonal_U = (pangulu_smatrix **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_smatrix *) * diagonal_nnz);
    pangulu_smatrix **diagonal_L = (pangulu_smatrix **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_smatrix *) * diagonal_nnz);

    char *flag_dignon_L = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(char) * diagonal_nnz);
    char *flag_dignon_U = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(char) * diagonal_nnz);

    for (pangulu_int64_t i = 0; i < diagonal_nnz; i++)
    {
        flag_dignon_L[i] = 0;
        flag_dignon_U[i] = 0;
    }

    block_smatrix->flag_dignon_l = flag_dignon_L;
    block_smatrix->flag_dignon_u = flag_dignon_U;

    for (pangulu_int64_t i = 0; i < diagonal_nnz; i++)
    {
        diagonal_U[i] = NULL;
    }
    for (pangulu_int64_t i = 0; i < diagonal_nnz; i++)
    {
        diagonal_L[i] = NULL;
    }

    for (pangulu_int64_t level = 0; level < block_length; level++)
    {
        pangulu_int64_t diagonal_index = mapper_diagonal_smatrix[level];
        if (diagonal_index != -1)
        {
            pangulu_int64_t now_rank = grid_process_id[(level % p) * q + level % q];
            if (now_rank == rank)
            {
                pangulu_int64_t first_index = pangulu_bip_get(level * block_length + level, BIP)->mapper_a;
                if (diagonal_U[diagonal_index] == NULL)
                {
                    pangulu_int64_t first_index = pangulu_bip_get(level * block_length + level, BIP)->mapper_a;
                    pangulu_smatrix *first_U = (pangulu_smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_smatrix));
                    pangulu_init_pangulu_smatrix(first_U);
                    pangulu_get_pangulu_smatrix_to_u(&Big_smatrix_value[first_index], first_U, nb);
                    pangulu_smatrix_add_csc(first_U);
                    pangulu_smatrix_add_memory_u(first_U);

#ifdef ADD_GPU_MEMORY
                    pangulu_smatrix_cuda_memory_init(first_U, first_U->row, first_U->nnz);
                    pangulu_smatrix_add_cuda_memory_u(first_U);
                    pangulu_smatrix_cuda_memcpy_nnzu(first_U, first_U);
                    pangulu_smatrix_cuda_memcpy_struct_csc(first_U, first_U);
                    pangulu_cuda_malloc((void **)&(first_U->d_graphindegree), nb * sizeof(int));
                    pangulu_cuda_malloc((void **)&(first_U->d_id_extractor), sizeof(int));
                    first_U->graphindegree = (int *)pangulu_malloc(__FILE__, __LINE__, nb * sizeof(int));
#endif
                    diagonal_U[diagonal_index] = first_U;
                }
                if (diagonal_L[diagonal_index] == NULL)
                {
                    pangulu_smatrix *first_L = (pangulu_smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_smatrix));
                    pangulu_init_pangulu_smatrix(first_L);
                    pangulu_get_pangulu_smatrix_to_l(&Big_smatrix_value[first_index], first_L, nb);
                    pangulu_smatrix_add_csc(first_L);

#ifdef ADD_GPU_MEMORY
                    pangulu_smatrix_cuda_memory_init(first_L, first_L->row, first_L->nnz);
                    pangulu_smatrix_cuda_memcpy_struct_csc(first_L, first_L);
                    pangulu_cuda_malloc((void **)&(first_L->d_graphindegree), nb * sizeof(int));
                    pangulu_cuda_malloc((void **)&(first_L->d_id_extractor), sizeof(int));
                    first_L->graphindegree = (int *)pangulu_malloc(__FILE__, __LINE__, nb * sizeof(int));
#endif
                    diagonal_L[diagonal_index] = first_L;
                }
            }
            else
            {
                if (diagonal_L[diagonal_index] == NULL)
                {
                    pangulu_smatrix *first_L = (pangulu_smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_smatrix));
                    pangulu_init_pangulu_smatrix(first_L);
                    pangulu_malloc_pangulu_smatrix_nnz_csc(first_L, nb, block_smatrix_non_zero_vector_L[level]);

#ifdef ADD_GPU_MEMORY
                    pangulu_smatrix_cuda_memory_init(first_L, nb, first_L->nnz);
                    pangulu_cuda_malloc((void **)&(first_L->d_graphindegree), nb * sizeof(int));
                    pangulu_cuda_malloc((void **)&(first_L->d_id_extractor), sizeof(int));
                    first_L->graphindegree = (int *)pangulu_malloc(__FILE__, __LINE__, nb * sizeof(int));
#endif
                    diagonal_L[diagonal_index] = first_L;
                }
                if (diagonal_U[diagonal_index] == NULL)
                {
                    pangulu_smatrix *first_U = (pangulu_smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_smatrix));
                    pangulu_init_pangulu_smatrix(first_U);
#ifdef GPU_TSTRF
                    pangulu_malloc_pangulu_smatrix_nnz_csr(first_U, nb, block_smatrix_non_zero_vector_U[level]);
#else
                    pangulu_malloc_pangulu_smatrix_nnz_csc(first_U, nb, block_smatrix_non_zero_vector_U[level]);
#endif

#ifdef ADD_GPU_MEMORY
                    pangulu_smatrix_cuda_memory_init(first_U, nb, first_U->nnz);
                    pangulu_cuda_malloc((void **)&(first_U->d_graphindegree), nb * sizeof(int));
                    pangulu_cuda_malloc((void **)&(first_U->d_id_extractor), sizeof(int));
                    first_U->graphindegree = (int *)pangulu_malloc(__FILE__, __LINE__, nb * sizeof(int));
#endif
                    diagonal_U[diagonal_index] = first_U;
                }
            }
        }
    }

    pangulu_int64_t task_level_length = block_length / every_level_length + (((block_length % every_level_length) == 0) ? 0 : 1);
    pangulu_int64_t *task_level_num = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * task_level_length);
    pangulu_int64_t *receive_level_num = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * task_level_length);

    for (pangulu_int64_t i = 0; i < task_level_length; i++)
    {
        task_level_num[i] = 0;
    }

    for (pangulu_int64_t i = 0; i < task_level_length; i++)
    {
        receive_level_num[i] = 0;
    }

    pangulu_int64_t *save_block_L = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * block_length);
    pangulu_int64_t *save_block_U = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * block_length);

    for (pangulu_int64_t k = 0; k < block_length; k++)
    {
        pangulu_int64_t level = level_index[k];
        pangulu_int64_t now_task_level = k / every_level_length;
        for (pangulu_int64_t i = level; i < block_length; i++)
        {
            save_block_L[i] = 0;
        }
        for (pangulu_int64_t i = level; i < block_length; i++)
        {
            save_block_U[i] = 0;
        }
        for (pangulu_int64_t i = L_columnpointer[level]; i < L_columnpointer[level + 1]; i++)
        {
            pangulu_int64_t row = L_rowindex[i];
            for (pangulu_int64_t j = U_rowpointer[level]; j < U_rowpointer[level + 1]; j++)
            {
                pangulu_int64_t col = U_columnindex[j];
                pangulu_int64_t now_offset_index = row * block_length + col;
                pangulu_int64_t now_offset_rank = grid_process_id[(row % p) * q + col % q];
                if (pangulu_bip_get(now_offset_index, BIP)->mapper_a != -1 && (now_offset_rank == rank))
                {
                    save_block_L[row] = 1;
                    save_block_U[col] = 1;
                }
            }
        }
        for (pangulu_int64_t i = level; i < block_length; i++)
        {
            if (save_block_L[i] == 1)
            {
                pangulu_int64_t mapper_index_A = pangulu_bip_get(i * block_length + level, BIP)->mapper_a;
                if (mapper_index_A == -1)
                {
                    receive_level_num[now_task_level]++;
                }
            }
        }
        for (pangulu_int64_t i = level; i < block_length; i++)
        {
            if (save_block_U[i] == 1)
            {
                pangulu_int64_t mapper_index_A = pangulu_bip_get(level * block_length + i, BIP)->mapper_a;
                if (mapper_index_A == -1)
                {
                    receive_level_num[now_task_level]++;
                }
            }
        }
    }
    pangulu_free(__FILE__, __LINE__, save_block_L);
    pangulu_free(__FILE__, __LINE__, save_block_U);

    for (pangulu_int64_t k = 0; k < block_length; k++)
    {
        pangulu_int64_t level = level_index[k];
        pangulu_int64_t now_task_level = k / every_level_length;
        for (pangulu_int64_t i = L_columnpointer[level]; i < L_columnpointer[level + 1]; i++)
        {
            pangulu_int64_t row = L_rowindex[i];
            for (pangulu_int64_t j = U_rowpointer[level]; j < U_rowpointer[level + 1]; j++)
            {
                pangulu_int64_t col = U_columnindex[j];
                pangulu_int64_t now_offset_index = row * block_length + col;
                pangulu_int64_t now_offset_rank = grid_process_id[(row % p) * q + col % q];
                if (pangulu_bip_get(now_offset_index, BIP)->mapper_a != -1 && (now_offset_rank == rank))
                {
                    pangulu_bip_set(row * block_length + col, BIP)->task_flag_id++;
                    task_level_num[now_task_level]++;
                }
            }
        }
    }

    pangulu_int64_t *save_flag = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * sum_rank_size);
    for (pangulu_int64_t i = 0; i < sum_rank_size; i++)
    {
        save_flag[i] = -1;
    }

    pangulu_int64_t *save_task_level_flag = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * task_level_length);

    for (pangulu_int64_t row = 0; row < block_length; row++)
    {
        for (pangulu_int64_t col = 0; col < block_length; col++)
        {
            pangulu_int64_t now_offset_index = row * block_length + col;
            if (pangulu_bip_get(now_offset_index, BIP)->mapper_a == -1)
            {
                continue;
            }
            if (pangulu_bip_get(now_offset_index, BIP)->sum_flag_block_num > 1)
            {
                pangulu_int64_t min_level = PANGULU_MIN(row, col);
                pangulu_int64_t now_offset_rank = grid_process_id[(row % p) * q + col % q];

                for (pangulu_int64_t i = 0; i < sum_rank_size; i++)
                {
                    save_flag[i] = -1;
                }

                for (pangulu_int64_t now_level = 0; now_level < min_level; now_level++)
                {
                    if ((pangulu_bip_get(now_level * block_length + col, BIP)->block_smatrix_nnza_num != 0) && (pangulu_bip_get(row * block_length + now_level, BIP)->block_smatrix_nnza_num != 0))
                    {
                        pangulu_int64_t now_rank = now_offset_rank;
                        save_flag[now_rank] = now_level;
                    }
                }
                for (pangulu_int64_t i = 0; i < task_level_length; i++)
                {
                    save_task_level_flag[i] = 0;
                }
                for (pangulu_int64_t i = 0; i < sum_rank_size; i++)
                {
                    if ((i != rank) && (save_flag[i] != -1))
                    {
                        save_task_level_flag[save_flag[i] / every_level_length]++;
                    }
                }
                for (pangulu_int64_t i = task_level_length - 1; i >= 0; i--)
                {
                    if (save_task_level_flag[i] != 0)
                    {
                        task_level_num[i]++;
                        break;
                    }
                }
                for (pangulu_int64_t i = 0; i < task_level_length; i++)
                {
                    receive_level_num[i] += save_task_level_flag[i];
                }
            }
        }
    }
    pangulu_free(__FILE__, __LINE__, save_flag);
    pangulu_free(__FILE__, __LINE__, save_task_level_flag);

    for (pangulu_int64_t row = 0; row < block_length; row++)
    {
        for (pangulu_int64_t col = 0; col < block_length; col++)
        {
            pangulu_int64_t now_offset_index = row * block_length + col;
            if (pangulu_bip_get(now_offset_index, BIP)->task_flag_id != 0 && pangulu_bip_get(now_offset_index, BIP)->sum_flag_block_num > 1)
            {
                pangulu_int64_t now_offset_rank = grid_process_id[(row % p) * q + col % q];
                if (now_offset_rank == rank)
                {
                    if (pangulu_bip_get(now_offset_index, BIP)->sum_flag_block_num > 1)
                    {
                        pangulu_bip_set(now_offset_index, BIP)->task_flag_id++;
                    }
                }
            }
        }
    }

    pangulu_int64_t max_PQ = block_common->max_pq;
    pangulu_int64_t *send_flag = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * current_rank_block_count * max_PQ);
    pangulu_int64_t *send_diagonal_flag_L = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * diagonal_nnz * max_PQ);
    pangulu_int64_t *send_diagonal_flag_U = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * diagonal_nnz * max_PQ);

    for (pangulu_int64_t i = 0; i < current_rank_block_count * max_PQ; i++)
    {
        send_flag[i] = 0;
    }
    for (pangulu_int64_t i = 0; i < diagonal_nnz * max_PQ; i++)
    {
        send_diagonal_flag_L[i] = 0;
    }
    for (pangulu_int64_t i = 0; i < diagonal_nnz * max_PQ; i++)
    {
        send_diagonal_flag_U[i] = 0;
    }

    for (pangulu_int64_t row = 0; row < block_length; row++)
    {
        for (pangulu_int64_t col = 0; col < block_length; col++)
        {
            pangulu_int64_t mapper_index = pangulu_bip_get(row * block_length + col, BIP)->mapper_a;
            if (mapper_index != -1)
            {
                if (row == col)
                {
                    pangulu_int64_t diagonal_index = mapper_diagonal_smatrix[row];
                    for (pangulu_int64_t i = row + 1; i < block_length; i++)
                    {
                        if (pangulu_bip_get(i * block_length + col, BIP)->block_smatrix_nnza_num != 0)
                        {
                            send_diagonal_flag_U[diagonal_index * max_PQ + (i - row) % p] = 1;
                        }
                    }
                    for (pangulu_int64_t i = col + 1; i < block_length; i++)
                    {
                        if (pangulu_bip_get(row * block_length + i, BIP)->block_smatrix_nnza_num != 0)
                        {
                            send_diagonal_flag_L[diagonal_index * max_PQ + (i - col) % q] = 1;
                        }
                    }
                }
                else if (row < col)
                {
                    for (pangulu_int64_t i = row + 1; i < block_length; i++)
                    {
                        if (pangulu_bip_get(i * block_length + row, BIP)->block_smatrix_nnza_num != 0 && pangulu_bip_get(i * block_length + col, BIP)->block_smatrix_nnza_num != 0)
                        {
                            send_flag[mapper_index * max_PQ + (i - row) % p] = 1;
                        }
                    }
                }
                else
                {
                    for (pangulu_int64_t i = col + 1; i < block_length; i++)
                    {
                        if (pangulu_bip_get(col * block_length + i, BIP)->block_smatrix_nnza_num != 0 && pangulu_bip_get(row * block_length + i, BIP)->block_smatrix_nnza_num != 0)
                        {
                            send_flag[mapper_index * max_PQ + (i - col) % q] = 1;
                        }
                    }
                }
            }
        }
    }

    pangulu_int64_t max_task_length = 0;
    for (pangulu_int64_t i = 0; i < task_level_length; i++)
    {
        max_task_length = PANGULU_MAX(task_level_num[i], max_task_length);
    }

    pangulu_heap *heap = (pangulu_heap *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_heap));
    pangulu_init_pangulu_heap(heap, max_task_length);

    pangulu_int64_t *mpi_level_num = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * task_level_length);
    for (pangulu_int64_t i = 0; i < task_level_length; i++)
    {
        mpi_level_num[i] = 0;
    }

    pangulu_int64_t block_non_zero_length = 0;
    for (pangulu_int64_t level = 0; level < block_length; level += every_level_length)
    {
        pangulu_int64_t block_non_zero_num = 0;
        pangulu_int64_t big_level = ((level + every_level_length) > block_length) ? block_length : (level + every_level_length);
        for (pangulu_int64_t k = level; k < big_level; k++)
        {
            pangulu_int64_t now_level = level_index[k];
            if (pangulu_bip_get(now_level * block_length + now_level, BIP)->block_smatrix_nnza_num != 0)
            {
                pangulu_bip_set(now_level * block_length + now_level, BIP)->mapper_mpi = block_non_zero_num;
                block_non_zero_num++;

                pangulu_bip_set(block_length * block_length + now_level, BIP)->mapper_mpi = block_non_zero_num;
                block_non_zero_num++;
            }
            else
            {
                printf("error diagnal is null\n");
            }
            for (pangulu_int64_t j = now_level + 1; j < block_length; j++)
            {
                if (pangulu_bip_get(now_level * block_length + j, BIP)->block_smatrix_nnza_num != 0)
                {
                    pangulu_bip_set(now_level * block_length + j, BIP)->mapper_mpi = block_non_zero_num;
                    block_non_zero_num++;
                }
                if (pangulu_bip_get(j * block_length + now_level, BIP)->block_smatrix_nnza_num != 0)
                {
                    pangulu_bip_set(j * block_length + now_level, BIP)->mapper_mpi = block_non_zero_num;
                    block_non_zero_num++;
                }
            }
        }
        mpi_level_num[level / every_level_length] = block_non_zero_length;
        block_non_zero_length += block_non_zero_num;
    }

    pangulu_int64_t *mapper_mpi_reverse = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * block_non_zero_length);

    block_non_zero_length = 0;
    for (pangulu_int64_t level = 0; level < block_length; level += every_level_length)
    {
        pangulu_int64_t big_level = ((level + every_level_length) > block_length) ? block_length : (level + every_level_length);
        for (pangulu_int64_t k = level; k < big_level; k++)
        {
            pangulu_int64_t now_level = level_index[k];
            if (pangulu_bip_get(now_level * block_length + now_level, BIP)->block_smatrix_nnza_num != 0)
            {
                mapper_mpi_reverse[block_non_zero_length++] = now_level * block_length + now_level;
                mapper_mpi_reverse[block_non_zero_length++] = block_length * block_length + now_level;
            }
            else
            {
                printf("error diagnal is null\n");
            }
            for (pangulu_int64_t j = now_level + 1; j < block_length; j++)
            {
                if (pangulu_bip_get(now_level * block_length + j, BIP)->block_smatrix_nnza_num != 0)
                {
                    mapper_mpi_reverse[block_non_zero_length++] = now_level * block_length + j;
                }
                if (pangulu_bip_get(j * block_length + now_level, BIP)->block_smatrix_nnza_num != 0)
                {
                    mapper_mpi_reverse[block_non_zero_length++] = j * block_length + now_level;
                }
            }
        }
    }

    temp_a_value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nb * nb);
    memset(temp_a_value, 0, sizeof(calculate_type) * nb * nb);
    ssssm_l_value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nb * nb);
    memset(ssssm_l_value, 0, sizeof(calculate_type) * nb * nb);
    ssssm_u_value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nb * nb);
    memset(ssssm_u_value, 0, sizeof(calculate_type) * nb * nb);
    ssssm_hash_l_row = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * (nb));
    ssssm_hash_u_col = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * (nb));
    ssssm_hash_lu = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * (nb));
    ssssm_col_ops_u = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (nb + 1));
    getrf_diagIndex_csc = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * (nb + 1));
    getrf_diagIndex_csr = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * (nb + 1));
    int omp_threads_num = pangu_omp_num_threads;
    ssssm_ops_pointer = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * (omp_threads_num + 1));

#ifdef GPU_OPEN
    pangulu_cuda_malloc((void **)&cuda_temp_value, nb * nb * sizeof(calculate_type));
    pangulu_cuda_malloc((void **)&cuda_b_idx_col, nb * nb * sizeof(pangulu_int32_t));
#endif

    block_smatrix->block_smatrix_non_zero_vector_l = block_smatrix_non_zero_vector_L;
    block_smatrix->block_smatrix_non_zero_vector_u = block_smatrix_non_zero_vector_U;
    block_smatrix->big_pangulu_smatrix_value = Big_smatrix_value;

    block_smatrix->l_pangulu_smatrix_columnpointer = L_columnpointer;
    block_smatrix->l_pangulu_smatrix_rowindex = L_rowindex;
    block_smatrix->l_pangulu_smatrix_value = L_value;
    block_smatrix->l_smatrix_nzz = L_smatrix_nzz;

    block_smatrix->u_pangulu_smatrix_rowpointer = U_rowpointer;
    block_smatrix->u_pangulu_smatrix_columnindex = U_columnindex;
    block_smatrix->u_pangulu_smatrix_value = U_value;
    block_smatrix->u_smatrix_nzz = U_smatrix_nzz;

    block_smatrix->mapper_diagonal = mapper_diagonal_smatrix;
    block_smatrix->diagonal_smatrix_l = diagonal_L;
    block_smatrix->diagonal_smatrix_u = diagonal_U;

    block_smatrix->calculate_l = calculate_L;
    block_smatrix->calculate_u = calculate_U;
    block_smatrix->calculate_x = calculate_X;

    block_smatrix->task_level_length = task_level_length;
    block_smatrix->task_level_num = task_level_num;
    block_smatrix->heap = heap;
    block_smatrix->now_level_l_length = now_level_L_length;
    block_smatrix->now_level_u_length = now_level_U_length;
    block_smatrix->save_now_level_l = save_now_level_l;
    block_smatrix->save_now_level_u = save_now_level_u;

    block_smatrix->send_flag = send_flag;
    block_smatrix->send_diagonal_flag_l = send_diagonal_flag_L;
    block_smatrix->send_diagonal_flag_u = send_diagonal_flag_U;

    block_smatrix->grid_process_id = grid_process_id;

    block_smatrix->save_send_rank_flag = save_send_rank_flag;

    block_smatrix->receive_level_num = receive_level_num;
    // block_smatrix->blocks_current_rank = blocks_current_rank;
    block_smatrix->save_tmp = NULL;

    block_smatrix->level_index = level_index;
    block_smatrix->level_index_reverse = level_index_reverse;

    block_smatrix->mapper_mpi_reverse = mapper_mpi_reverse;
    block_smatrix->mpi_level_num = mpi_level_num;
    block_smatrix->current_rank_block_count = current_rank_block_count;

#ifdef OVERLAP

    block_smatrix->run_bsem1 = (bsem *)pangulu_malloc(__FILE__, __LINE__, sizeof(bsem));
    block_smatrix->run_bsem2 = (bsem *)pangulu_malloc(__FILE__, __LINE__, sizeof(bsem));
    pangulu_bsem_init(block_smatrix->run_bsem1, 0);
    pangulu_bsem_init(block_smatrix->run_bsem2, 0);

    heap->heap_bsem = (bsem *)pangulu_malloc(__FILE__, __LINE__, sizeof(bsem));
    pangulu_bsem_init(heap->heap_bsem, 0);
#endif

    return;
}
