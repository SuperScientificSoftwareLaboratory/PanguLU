#include "pangulu_common.h"

void pangulu_solve_a_to_lu(pangulu_int64_t level, pangulu_int64_t row, pangulu_int64_t col,
                           pangulu_block_common *block_common,
                           pangulu_block_smatrix *block_smatrix,
                           pangulu_smatrix *calculate_L,
                           pangulu_smatrix *calculate_U)
{
    pangulu_int32_t block_length = block_common->block_length;
    pangulu_int32_t sum_rank_size = block_common->sum_rank_size;
    pangulu_int32_t p = block_common->p;
    pangulu_int32_t q = block_common->q;
    pangulu_int32_t *grid_process_id = block_smatrix->grid_process_id;
    pangulu_block_info_pool* BIP = block_smatrix->BIP;

    pangulu_int64_t mapper_index = pangulu_bip_get(block_length * row + col, BIP)->mapper_a;
    pangulu_smatrix *a = &block_smatrix->big_pangulu_smatrix_value[mapper_index];
    pangulu_int64_t diagonal_index = block_smatrix->mapper_diagonal[level];
    pangulu_smatrix *l = block_smatrix->diagonal_smatrix_l[diagonal_index];
    pangulu_smatrix *u = block_smatrix->diagonal_smatrix_u[diagonal_index];
    pangulu_heap *heap = block_smatrix->heap;

    pangulu_getrf_interface(a, l, u, calculate_L, calculate_U);

    #ifdef GPU_TSTRF
    pangulu_transpose_pangulu_smatrix_csc_to_csr(u);
    #endif

#ifdef OVERLAP
    pthread_mutex_t *heap_mutex = (heap->heap_bsem)->mutex;
    pthread_mutex_lock(heap_mutex);
#endif
    pangulu_int64_t begin_L = block_smatrix->l_pangulu_smatrix_columnpointer[col] + 1;
    pangulu_int64_t length_L = block_smatrix->l_pangulu_smatrix_columnpointer[col + 1] - begin_L;
    pangulu_inblock_idx *rowindex_L = block_smatrix->l_pangulu_smatrix_rowindex + begin_L;

    pangulu_int64_t begin_U = block_smatrix->u_pangulu_smatrix_rowpointer[row] + 1;
    pangulu_int64_t length_U = block_smatrix->u_pangulu_smatrix_rowpointer[row + 1] - begin_U;
    pangulu_exblock_idx *columnindex_U = block_smatrix->u_pangulu_smatrix_columnindex + begin_U;

    for (pangulu_int64_t i = 0; i < length_L; i++)
    {
        pangulu_int64_t now_row = rowindex_L[i];
        pangulu_int64_t now_col = col;
        pangulu_int64_t now_rank = grid_process_id[(now_row % p) * q + (now_col % q)];

        if (now_rank == rank)
        {
            pangulu_bip_set(block_length * rowindex_L[i] + col, BIP)->task_flag_id--;
            if (pangulu_bip_get(block_length * rowindex_L[i] + col, BIP)->task_flag_id == 0)
            {
                pangulu_heap_insert(heap, rowindex_L[i], col, level, 2, rowindex_L[i]);
            }
        }
    }

    for (pangulu_int64_t i = 0; i < length_U; i++)
    {
        pangulu_int64_t now_row = row;
        pangulu_int64_t now_col = columnindex_U[i];
        pangulu_int64_t now_rank = grid_process_id[(now_row % p) * q + (now_col % q)];

        if (now_rank == rank)
        {
            pangulu_bip_set(block_length * row + columnindex_U[i], BIP)->task_flag_id--;
            if (pangulu_bip_get(block_length * row + columnindex_U[i], BIP)->task_flag_id == 0)
            {
                pangulu_heap_insert(heap, row, columnindex_U[i], level, 3, columnindex_U[i]);
            }
        }
    }

#ifdef OVERLAP
    pthread_mutex_unlock(heap_mutex);
#endif

    pangulu_int64_t nb = block_common->nb;

    pangulu_int64_t signal = level * block_length + level;
    pangulu_int64_t index_signal = pangulu_bip_get(signal, BIP)->mapper_mpi;
    pangulu_int64_t *save_send_rank_flag = block_smatrix->save_send_rank_flag;

    pangulu_int64_t send_offset_row = level % p;
    pangulu_int64_t send_offset_col = level % q;

    pangulu_int64_t max_PQ = block_common->max_pq;
    pangulu_int64_t *now_send_diagonal_L = (block_smatrix->send_diagonal_flag_l) + diagonal_index * max_PQ;
    pangulu_init_vector_int(save_send_rank_flag, sum_rank_size);
    save_send_rank_flag[rank] = 1;

    for (pangulu_int64_t i = 0; i < ((q - 1) < (block_length - level - 1) ? (q - 1) : (block_length - level - 1)); i++)
    {
        send_offset_col++;
        send_offset_col = send_offset_col % q;

        pangulu_int64_t now_rank = grid_process_id[send_offset_row * q + send_offset_col];

        if (save_send_rank_flag[now_rank] == 0 && now_send_diagonal_L[i + 1] == 1)
        {
            save_send_rank_flag[now_rank] = 1;
            pangulu_isend_whole_pangulu_smatrix_csc(l, now_rank, index_signal, nb);
        }
    }

    send_offset_col = level % q;

    pangulu_int64_t *now_send_diagonal_U = (block_smatrix->send_diagonal_flag_u) + diagonal_index * max_PQ;
    pangulu_init_vector_int(save_send_rank_flag, sum_rank_size);
    save_send_rank_flag[rank] = 1;

    signal = block_length * block_length + level;
    index_signal = pangulu_bip_get(signal, BIP)->mapper_mpi;

    for (pangulu_int64_t i = 0; i < ((p - 1) < (block_length - level - 1) ? (p - 1) : (block_length - level - 1)); i++)
    {
        send_offset_row++;
        send_offset_row = send_offset_row % p;
        pangulu_int64_t now_rank = grid_process_id[send_offset_row * q + send_offset_col];
        if (save_send_rank_flag[now_rank] == 0 && now_send_diagonal_U[i + 1] == 1)
        {
            save_send_rank_flag[now_rank] = 1;
            #ifdef GPU_TSTRF
            pangulu_isend_whole_pangulu_smatrix_csr(u, now_rank, index_signal, nb);
            #else
            pangulu_isend_whole_pangulu_smatrix_csc(u, now_rank, index_signal, nb);
#endif
        }
    }

#ifdef ADD_GPU_MEMORY
    block_smatrix->flag_dignon_l[diagonal_index] = 1;
    pangulu_gessm_preprocess(l);
#endif

}
void pangulu_solve_xu_a(pangulu_int64_t level, pangulu_int64_t now_level, pangulu_int64_t row, pangulu_int64_t col,
                        pangulu_block_common *block_common,
                        pangulu_block_smatrix *block_smatrix,
                        pangulu_smatrix *calculate_U,
                        pangulu_smatrix *calculate_X)
{
    pangulu_int64_t block_length = block_common->block_length;
    pangulu_int64_t p = block_common->p;
    pangulu_int64_t q = block_common->q;
    pangulu_int64_t nb = block_common->nb;
    pangulu_int64_t sum_rank_size = block_common->sum_rank_size;
    pangulu_int32_t *grid_process_id = block_smatrix->grid_process_id;
    pangulu_heap *heap = block_smatrix->heap;
    pangulu_block_info_pool* BIP = block_smatrix->BIP;

    pangulu_int64_t mapper_index = pangulu_bip_get(block_length * row + col, BIP)->mapper_a;
    pangulu_smatrix *a = &block_smatrix->big_pangulu_smatrix_value[mapper_index];

    pangulu_int64_t diagonal_index = block_smatrix->mapper_diagonal[col];
    pangulu_smatrix *u = block_smatrix->diagonal_smatrix_u[diagonal_index];

    pangulu_int64_t mapper_index_X = pangulu_bip_get(block_length * row + col, BIP)->mapper_lu;
    pangulu_smatrix *save_X = block_smatrix->l_pangulu_smatrix_value[mapper_index_X];

    save_X->nnz = a->nnz;
    save_X->rowindex = a->rowindex;
    save_X->value_csc = a->value_csc;

    if (block_smatrix->flag_dignon_u[diagonal_index] == 0)
    {
#ifdef GPU_OPEN
        pangulu_smatrix_cuda_memcpy_complete_csr(u, u);
        pangulu_tstrf_preprocess(u);
#endif
        block_smatrix->flag_dignon_u[diagonal_index] = 1;
    }

    pangulu_tstrf_interface(a, save_X, u, calculate_X, calculate_U);

#ifdef OVERLAP
    pthread_mutex_t *heap_mutex = (heap->heap_bsem)->mutex;
    pthread_mutex_lock(heap_mutex);
#endif

    pangulu_int64_t offset_col = block_smatrix->level_index_reverse[col];
    pangulu_int64_t offset_L = (block_smatrix->l_smatrix_nzz) * (offset_col - now_level);
    pangulu_int64_t offset_U = (block_smatrix->u_smatrix_nzz) * (offset_col - now_level);
    block_smatrix->save_now_level_l[block_smatrix->now_level_l_length[offset_col - now_level] + offset_L] = row;

    block_smatrix->now_level_l_length[offset_col - now_level]++;
    pangulu_int64_t now_add_length = block_smatrix->now_level_u_length[offset_col - now_level];
    pangulu_int64_t *save_now_level_U = (block_smatrix->save_now_level_u) + offset_U;

    for (pangulu_int64_t i = 0; i < now_add_length; i++)
    {
        pangulu_int64_t now_row = row;
        pangulu_int64_t now_col = save_now_level_U[i];
        pangulu_int64_t now_rank = grid_process_id[(now_row % p) * q + (now_col % q)];
        if ((now_rank == rank) && (pangulu_bip_get(now_row * block_length + now_col, BIP)->mapper_a != -1))
        {
            pangulu_heap_insert(heap, row, save_now_level_U[i], level, 4, PANGULU_MAX(row, save_now_level_U[i]));
        }
    }

#ifdef OVERLAP
    pthread_mutex_unlock(heap_mutex);
#endif

    if (block_common->rank == -3)
    {
        pangulu_display_pangulu_smatrix_csc(save_X);
    }
    pangulu_int64_t signal = block_length * row + col;
    pangulu_int64_t index_signal = pangulu_bip_get(signal, BIP)->mapper_mpi;

    pangulu_int64_t send_offset_row = row % p;
    pangulu_int64_t send_offset_col = level % q;

    pangulu_int64_t max_PQ = block_common->max_pq;
    pangulu_int64_t *now_send_flag = (block_smatrix->send_flag) + max_PQ * mapper_index;

    pangulu_int64_t *save_send_rank_flag = block_smatrix->save_send_rank_flag;
    pangulu_init_vector_int(save_send_rank_flag, sum_rank_size);
    save_send_rank_flag[rank] = 1;

    for (pangulu_int64_t i = 0; i < ((q - 1) < ((block_length - level - 1)) ? (q - 1) : (block_length - level - 1)); i++)
    {
        send_offset_col++;
        send_offset_col = send_offset_col % q;

        pangulu_int64_t now_rank = grid_process_id[send_offset_row * q + send_offset_col];
        if (save_send_rank_flag[now_rank] == 0 && now_send_flag[i + 1] == 1)
        {
            save_send_rank_flag[now_rank] = 1;
            pangulu_isend_whole_pangulu_smatrix_csc(a, now_rank, index_signal, nb);
        }
    }
#ifdef ADD_GPU_MEMORY
    pangulu_pangulu_smatrix_memcpy_columnpointer_csc(save_X, a);
    pangulu_smatrix_cuda_memcpy_complete_csc(save_X, a);
    block_smatrix->flag_save_l[mapper_index_X] = 1;
#endif
}

void pangulu_solve_lx_a(pangulu_int64_t level, pangulu_int64_t now_level, pangulu_int64_t row, pangulu_int64_t col,
                        pangulu_block_common *block_common,
                        pangulu_block_smatrix *block_smatrix,
                        pangulu_smatrix *calculate_L,
                        pangulu_smatrix *calculate_X)
{
    pangulu_int64_t p = block_common->p;
    pangulu_int64_t block_length = block_common->block_length;
    pangulu_int64_t q = block_common->q;
    pangulu_int64_t nb = block_common->nb;
    pangulu_int64_t sum_rank_size = block_common->sum_rank_size;
    pangulu_int32_t *grid_process_id = block_smatrix->grid_process_id;
    pangulu_heap *heap = block_smatrix->heap;
    pangulu_block_info_pool* BIP = block_smatrix->BIP;

    pangulu_int64_t mapper_index = pangulu_bip_get(block_length * row + col, BIP)->mapper_a;
    pangulu_smatrix *a = &block_smatrix->big_pangulu_smatrix_value[mapper_index];

    pangulu_int64_t diagonal_index = block_smatrix->mapper_diagonal[row];
    pangulu_smatrix *l = block_smatrix->diagonal_smatrix_l[diagonal_index];

    pangulu_int64_t mapper_index_X = pangulu_bip_get(block_length * row + col, BIP)->mapper_lu;
    pangulu_smatrix *save_X = block_smatrix->u_pangulu_smatrix_value[mapper_index_X];

    save_X->nnz = a->nnz;
    save_X->rowindex = a->rowindex;
    save_X->value_csc = a->value_csc;

    if (block_smatrix->flag_dignon_l[diagonal_index] == 0)
    {
#ifdef GPU_OPEN
        pangulu_smatrix_cuda_memcpy_complete_csc(l, l);
        pangulu_gessm_preprocess(l);
#endif
        block_smatrix->flag_dignon_l[diagonal_index] = 1;
    }

    pangulu_gessm_interface(a, save_X, l, calculate_X, calculate_L);

#ifdef OVERLAP
    pthread_mutex_t *heap_mutex = (heap->heap_bsem)->mutex;
    pthread_mutex_lock(heap_mutex);
#endif

    pangulu_int64_t offset_row = block_smatrix->level_index_reverse[row];
    pangulu_int64_t offset_U = (block_smatrix->u_smatrix_nzz) * (offset_row - now_level);
    pangulu_int64_t offset_L = (block_smatrix->l_smatrix_nzz) * (offset_row - now_level);
    block_smatrix->save_now_level_u[block_smatrix->now_level_u_length[offset_row - now_level] + offset_U] = col;
    block_smatrix->now_level_u_length[offset_row - now_level]++;
    pangulu_int64_t now_add_length = block_smatrix->now_level_l_length[offset_row - now_level];
    pangulu_int64_t *save_now_level_L = (block_smatrix->save_now_level_l) + offset_L;

    for (pangulu_int64_t i = 0; i < now_add_length; i++)
    {
        pangulu_int64_t now_row = save_now_level_L[i];
        pangulu_int64_t now_col = col;
        pangulu_int64_t now_rank = grid_process_id[(now_row % p) * q + (now_col % q)];
        if ((now_rank == rank) && (pangulu_bip_get(now_row * block_length + now_col, BIP)->mapper_a != -1))
        {
            pangulu_heap_insert(heap, save_now_level_L[i], col, level, 4, PANGULU_MAX(save_now_level_L[i], col));
        }
    }
#ifdef OVERLAP
    pthread_mutex_unlock(heap_mutex);
#endif

    pangulu_int64_t signal = block_length * row + col;
    pangulu_int64_t index_signal = pangulu_bip_get(signal, BIP)->mapper_mpi;

    pangulu_int64_t send_offset_row = level % p;
    pangulu_int64_t send_offset_col = col % q;

    pangulu_int64_t max_PQ = block_common->max_pq;
    pangulu_int64_t *now_send_flag = (block_smatrix->send_flag) + max_PQ * mapper_index;

    pangulu_int64_t *save_send_rank_flag = block_smatrix->save_send_rank_flag;
    pangulu_init_vector_int(save_send_rank_flag, sum_rank_size);
    save_send_rank_flag[rank] = 1;

    for (pangulu_int64_t i = 0; i < ((p - 1) < (block_length - level - 1) ? (p - 1) : (block_length - level - 1)); i++)
    {
        send_offset_row++;
        send_offset_row = send_offset_row % p;
        pangulu_int64_t now_rank = grid_process_id[send_offset_row * q + send_offset_col];

        if (save_send_rank_flag[now_rank] == 0 && now_send_flag[i + 1] == 1)
        {
            save_send_rank_flag[now_rank] = 1;
            pangulu_isend_whole_pangulu_smatrix_csc(a, now_rank, index_signal, nb);
        }
    }

#ifdef ADD_GPU_MEMORY
    pangulu_pangulu_smatrix_memcpy_columnpointer_csc(save_X, a);
    pangulu_smatrix_cuda_memcpy_complete_csc(save_X, a);
    block_smatrix->flag_save_u[mapper_index_X] = 1;
#endif

    return;
}

void pangulu_solve_a_lu(pangulu_int64_t level, pangulu_int64_t row, pangulu_int64_t col,
                        pangulu_block_common *block_common,
                        pangulu_block_smatrix *block_smatrix,
                        pangulu_smatrix *calculate_L,
                        pangulu_smatrix *calculate_U)
{
    pangulu_int64_t block_length = block_common->block_length;
    pangulu_block_info_pool* BIP = block_smatrix->BIP;

    pangulu_int64_t mapper_index = pangulu_bip_get(block_length * row + col, BIP)->mapper_a;

    if (mapper_index == -1)
    {
        return;
    }

    pangulu_smatrix *a = &block_smatrix->big_pangulu_smatrix_value[mapper_index];
    pangulu_int64_t mapper_index_L = pangulu_bip_get(block_length * row + level, BIP)->mapper_lu;
    pangulu_int64_t mapper_index_U = pangulu_bip_get(block_length * level + col, BIP)->mapper_lu;
    pangulu_smatrix *l = block_smatrix->l_pangulu_smatrix_value[mapper_index_L];
    pangulu_smatrix *u = block_smatrix->u_pangulu_smatrix_value[mapper_index_U];

    if (block_common->rank == -1)
    {
#ifdef GPU_OPEN
        pangulu_smatrix_cuda_memcpy_value_csc(a, a);
#endif
        pangulu_display_pangulu_smatrix_csc(a);
    }

    if (block_smatrix->flag_save_l[mapper_index_L] == 0)
    {
#ifdef GPU_OPEN
        pangulu_smatrix_cuda_memcpy_complete_csc(l, l);
#endif
        block_smatrix->flag_save_l[mapper_index_L] = 1;
    }

    if (block_smatrix->flag_save_u[mapper_index_U] == 0)
    {
#ifdef GPU_OPEN
        pangulu_smatrix_cuda_memcpy_complete_csc(u, u);
#endif
        block_smatrix->flag_save_u[mapper_index_U] = 1;
    }

    pangulu_ssssm_interface(a, l, u, calculate_L, calculate_U);

    if (block_common->rank == -1)
    {
        pangulu_display_pangulu_smatrix_csc(l);
        pangulu_display_pangulu_smatrix_csc(u);
#ifdef GPU_OPEN
        pangulu_smatrix_cuda_memcpy_value_csc(a, a);
#endif
        pangulu_display_pangulu_smatrix_csc(a);
    }

    pangulu_heap *heap = block_smatrix->heap;

#ifdef OVERLAP
    pthread_mutex_t *heap_mutex = (heap->heap_bsem)->mutex;
    pthread_mutex_lock(heap_mutex);
#endif

    pangulu_bip_set(block_length * row + col, BIP)->task_flag_id--;
    if (pangulu_bip_get(block_length * row + col, BIP)->task_flag_id == 0)
    {

        pangulu_int32_t *grid_process_id = block_smatrix->grid_process_id;

        pangulu_int64_t p = block_common->p;
        pangulu_int64_t q = block_common->q;
        pangulu_int64_t now_rank = grid_process_id[(row % p) * q + (col % q)];

        if (now_rank == rank)
        {

            if (row == col)
            {
                pangulu_heap_insert(heap, row, col, row, 1, row);
            }
            else if (row > col)
            {
                pangulu_heap_insert(heap, row, col, col, 2, row);
            }
            else if (row < col)
            {
                pangulu_heap_insert(heap, row, col, row, 3, col);
            }

#ifdef OVERLAP
            pthread_mutex_unlock(heap_mutex);
#endif
        }
        else
        {
#ifdef OVERLAP
            pthread_mutex_unlock(heap_mutex);
#endif

            pangulu_int64_t nb = block_common->nb;
#ifdef GPU_OPEN
            pangulu_smatrix_cuda_memcpy_value_csc(a, a);
#endif
            pangulu_int64_t signal = block_length * row + col;
            pangulu_int64_t index_signal = pangulu_bip_get(signal, BIP)->mapper_mpi;
            pangulu_isend_pangulu_smatrix_value_csc_in_signal(a, now_rank, index_signal, nb);
        }
    }
    else
    {
#ifdef OVERLAP
        pthread_mutex_unlock(heap_mutex);
#endif
    }
}

void pangulu_numerical_work(compare_struct *flag,
                            pangulu_block_common *block_common,
                            pangulu_block_smatrix *block_smatrix,
                            pangulu_smatrix *calculate_L,
                            pangulu_smatrix *calculate_U,
                            pangulu_smatrix *calculate_X,
                            pangulu_int64_t now_level)
{
    pangulu_int64_t kernel_id = flag->kernel_id;
    pangulu_int64_t row = flag->row;
    pangulu_int64_t col = flag->col;
    pangulu_int64_t level = flag->task_level;
    pangulu_block_info_pool* BIP = block_smatrix->BIP;
    if (pangulu_bip_get(row * (block_common->block_length) + col, BIP)->mapper_a == -1)
    {
        printf(PANGULU_E_ERR_IN_RRCL);
        pangulu_exit(1);
        return;
    }

    if (kernel_id == 1)
    {
        pangulu_solve_a_to_lu(level, row, col,
                              block_common,
                              block_smatrix,
                              calculate_L,
                              calculate_U);
    }
    else if (kernel_id == 2)
    {
        pangulu_solve_xu_a(level, now_level, row, col,
                           block_common,
                           block_smatrix,
                           calculate_U,
                           calculate_X);
    }
    else if (kernel_id == 3)
    {
        pangulu_solve_lx_a(level, now_level, row, col,
                           block_common,
                           block_smatrix,
                           calculate_L,
                           calculate_X);
    }
    else if (kernel_id == 4)
    {
        pangulu_solve_a_lu(level, row, col,
                           block_common,
                           block_smatrix,
                           calculate_L,
                           calculate_U);
    }
    else
    {
        printf(PANGULU_E_K_ID);
        pangulu_exit(1);
    }
}

void pangulu_numerical_receive_message(MPI_Status status,
                                       pangulu_int64_t now_level,
                                       pangulu_block_common *block_common,
                                       pangulu_block_smatrix *block_smatrix)
{

    pangulu_int64_t block_length = block_common->block_length;
    pangulu_int64_t p = block_common->p;
    pangulu_int64_t q = block_common->q;
    pangulu_int64_t nb = block_common->nb;
    pangulu_int32_t *grid_process_id = block_smatrix->grid_process_id;
    pangulu_block_info_pool* BIP = block_smatrix->BIP;

    pangulu_heap *heap = block_smatrix->heap;

    pangulu_int64_t tag = (status.MPI_TAG);

    pangulu_int64_t *mpi_level_num = block_smatrix->mpi_level_num;
    pangulu_int64_t *mapper_mpi_reverse = block_smatrix->mapper_mpi_reverse;
    pangulu_int64_t index_tag = mapper_mpi_reverse[tag + mpi_level_num[now_level / block_common->every_level_length]];

    pangulu_int64_t receive_id = status.MPI_SOURCE;

    pangulu_int64_t row;
    pangulu_int64_t col;

    if (index_tag < (block_length * block_length))
    {
        row = index_tag / block_length;
        col = index_tag % block_length;
    }
    else
    {
        row = index_tag - (block_length * block_length);
        col = row;
    }


#ifdef OVERLAP
    pthread_mutex_t *heap_mutex = (heap->heap_bsem)->mutex;
#endif

    if (row == col)
    {

        pangulu_int64_t level = row;
        pangulu_int64_t diagonal_index = block_smatrix->mapper_diagonal[level];
        if (index_tag < (block_length * block_length))
        {
            pangulu_smatrix *l = block_smatrix->diagonal_smatrix_l[diagonal_index];
            pangulu_int64_t nnz = block_smatrix->block_smatrix_non_zero_vector_l[level];
            pangulu_recv_whole_pangulu_smatrix_csc(l, receive_id, tag, nnz, nb);
#ifdef ADD_GPU_MEMORY
#endif

            pangulu_exblock_ptr *U_rowpointer = block_smatrix->u_pangulu_smatrix_rowpointer;
            pangulu_exblock_idx *U_columnindex = block_smatrix->u_pangulu_smatrix_columnindex;
#ifdef OVERLAP
            pthread_mutex_lock(heap_mutex);
#endif
            for (pangulu_int64_t i = U_rowpointer[level]; i < U_rowpointer[level + 1]; i++)
            {
                pangulu_int64_t add_col = U_columnindex[i];

                pangulu_int64_t now_row = level;
                pangulu_int64_t now_col = add_col;
                pangulu_int64_t now_rank = grid_process_id[(now_row % p) * q + (now_col % q)];

                if (now_rank == rank)
                {
                    pangulu_bip_set(level * block_length + add_col, BIP)->task_flag_id--;
                    if (pangulu_bip_get(block_length * row + add_col, BIP)->task_flag_id == 0)
                    {
                        pangulu_heap_insert(heap, level, add_col, level, 3, add_col);
                    }
                }
            }

#ifdef OVERLAP
            pthread_mutex_unlock(heap_mutex);
#endif
        }
        else
        {
            pangulu_smatrix *u = block_smatrix->diagonal_smatrix_u[diagonal_index];
            pangulu_int64_t nnz = block_smatrix->block_smatrix_non_zero_vector_u[level];
            #ifdef GPU_TSTRF
            pangulu_recv_whole_pangulu_smatrix_csr(u, receive_id, tag, nnz, nb);
            #else
            pangulu_recv_whole_pangulu_smatrix_csc(u, receive_id, tag, nnz, nb);
            #endif

#ifdef ADD_GPU_MEMORY
#endif

            pangulu_inblock_ptr *L_columnpointer = block_smatrix->l_pangulu_smatrix_columnpointer;
            pangulu_inblock_idx *L_rowindex = block_smatrix->l_pangulu_smatrix_rowindex;

#ifdef OVERLAP
            pthread_mutex_lock(heap_mutex);
#endif

            for (pangulu_int64_t i = L_columnpointer[level]; i < L_columnpointer[level + 1]; i++)
            {
                pangulu_int64_t add_row = L_rowindex[i];

                pangulu_int64_t now_row = add_row;
                pangulu_int64_t now_col = level;
                pangulu_int64_t now_rank = grid_process_id[(now_row % p) * q + (now_col % q)];

                if (now_rank == rank)
                {
                    pangulu_bip_set(add_row * block_length + level, BIP)->task_flag_id--;
                    if (pangulu_bip_set(block_length * add_row + col, BIP)->task_flag_id == 0)
                    {
                        pangulu_heap_insert(heap, add_row, level, level, 2, add_row);
                    }
                }
            }

#ifdef OVERLAP
            pthread_mutex_unlock(heap_mutex);
#endif
        }
    }
    else if (row > col)
    {
        pangulu_int64_t level = col;

        pangulu_int64_t mapper_index_L = pangulu_bip_get(block_length * row + col, BIP)->mapper_lu;
        pangulu_smatrix *save_L = block_smatrix->l_pangulu_smatrix_value[mapper_index_L];
        pangulu_int64_t nnz = pangulu_bip_get(block_length * row + col, BIP)->block_smatrix_nnza_num;

        pangulu_recv_whole_pangulu_smatrix_csc(save_L, receive_id, tag, nnz, nb);
#ifdef ADD_GPU_MEMORY
#endif

#ifdef OVERLAP
        pthread_mutex_lock(heap_mutex);
#endif
        pangulu_int64_t offset_col = block_smatrix->level_index_reverse[col];
        pangulu_int64_t offset_L = (block_smatrix->l_smatrix_nzz) * (offset_col - now_level);
        pangulu_int64_t offset_U = (block_smatrix->u_smatrix_nzz) * (offset_col - now_level);
        block_smatrix->save_now_level_l[block_smatrix->now_level_l_length[offset_col - now_level] + offset_L] = row;

        block_smatrix->now_level_l_length[offset_col - now_level]++;
        pangulu_int64_t now_add_length = block_smatrix->now_level_u_length[offset_col - now_level];
        pangulu_int64_t *save_now_level_U = (block_smatrix->save_now_level_u) + offset_U;

        for (pangulu_int64_t i = 0; i < now_add_length; i++)
        {

            pangulu_int64_t now_row = row;
            pangulu_int64_t now_col = save_now_level_U[i];
            pangulu_int64_t now_rank = grid_process_id[(now_row % p) * q + (now_col % q)];
            if ((now_rank == rank) && (pangulu_bip_get(now_row * block_length + now_col, BIP)->mapper_a != -1))
            {
                pangulu_heap_insert(heap, row, save_now_level_U[i], level, 4, PANGULU_MAX(row, save_now_level_U[i]));
            }
        }

#ifdef OVERLAP
        pthread_mutex_unlock(heap_mutex);
#endif
    }
    else
    {
        pangulu_int64_t level = row;

        pangulu_int64_t mapper_index_U = pangulu_bip_get(block_length * row + col, BIP)->mapper_lu;
        pangulu_smatrix *save_U = block_smatrix->u_pangulu_smatrix_value[mapper_index_U];
        pangulu_int64_t nnz = pangulu_bip_get(block_length * row + col, BIP)->block_smatrix_nnza_num;

        pangulu_recv_whole_pangulu_smatrix_csc(save_U, receive_id, tag, nnz, nb);
#ifdef ADD_GPU_MEMORY
#endif

#ifdef OVERLAP
        pthread_mutex_lock(heap_mutex);
#endif
        pangulu_int64_t offset_row = block_smatrix->level_index_reverse[row];
        pangulu_int64_t offset_U = (block_smatrix->u_smatrix_nzz) * (offset_row - now_level);
        pangulu_int64_t offset_L = (block_smatrix->l_smatrix_nzz) * (offset_row - now_level);
        block_smatrix->save_now_level_u[block_smatrix->now_level_u_length[offset_row - now_level] + offset_U] = col;
        block_smatrix->now_level_u_length[offset_row - now_level]++;
        pangulu_int64_t now_add_length = block_smatrix->now_level_l_length[offset_row - now_level];
        pangulu_int64_t *save_now_level_L = (block_smatrix->save_now_level_l) + offset_L;

        for (pangulu_int64_t i = 0; i < now_add_length; i++)
        {
            pangulu_int64_t now_row = save_now_level_L[i];
            pangulu_int64_t now_col = col;
            pangulu_int64_t now_rank = grid_process_id[(now_row % p) * q + (now_col % q)];
            if ((now_rank == rank) && (pangulu_bip_get(now_row * block_length + now_col, BIP)->mapper_a != -1))
            {
                pangulu_heap_insert(heap, save_now_level_L[i], col, level, 4, PANGULU_MAX(save_now_level_L[i], col));
            }
        }

#ifdef OVERLAP
        pthread_mutex_unlock(heap_mutex);
#endif
    }
}

#ifdef OVERLAP

void *thread_GPU_work(void *param)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    pangulu_int64_t cpu_thread_count_per_node = sysconf(_SC_NPROCESSORS_ONLN);
    for (int i = 0; i < pangu_omp_num_threads; i++)
    {
#ifdef HT_IS_OPEN
        CPU_SET((2 * (pangu_omp_num_threads * rank + i)) % cpu_thread_count_per_node, &cpuset);
#else
        CPU_SET((pangu_omp_num_threads * rank + i) % cpu_thread_count_per_node, &cpuset);
#endif
    }
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0)
    {
        perror("pthread_setaffinity_np error");
    }
    
    // init
    thread_param *work_param = (thread_param *)param;
    pangulu_block_common *block_common = work_param->common;
    pangulu_block_smatrix *block_smatrix = work_param->smatrix;
#ifdef GPU_OPEN
    pangulu_cuda_device_init_thread(block_common->rank);
#endif
    pangulu_int64_t block_length = block_common->block_length;
    pangulu_int64_t every_level_length = block_common->every_level_length;

    pangulu_int64_t *task_level_num = block_smatrix->task_level_num;
    pangulu_smatrix *calculate_L = block_smatrix->calculate_l;
    pangulu_smatrix *calculate_U = block_smatrix->calculate_u;
    pangulu_smatrix *calculate_X = block_smatrix->calculate_x;

    bsem *run_bsem1 = block_smatrix->run_bsem1;
    bsem *run_bsem2 = block_smatrix->run_bsem2;
    pangulu_heap *heap = block_smatrix->heap;
    compare_struct *compare_queue = heap->comapre_queue;
    pthread_mutex_t *heap_mutex = (heap->heap_bsem)->mutex;

    pangulu_int64_t now_flag = 0;
    pangulu_bsem_synchronize(run_bsem2);

    for (pangulu_int64_t level = 0; level < block_length; level += every_level_length)
    {
        // communicate
        pangulu_int64_t now_task_num = task_level_num[level / every_level_length];

        while (now_task_num != 0)
        {

#ifdef CHECK_TIME
            struct timeval GET_TIME_START;
            pangulu_time_check_begin(&GET_TIME_START);
#endif
            pangulu_int64_t compare_flag = pangulu_bsem_wait(heap);
            now_task_num--;
            pangulu_numerical_work(compare_queue + compare_flag, block_common, block_smatrix,
                                   calculate_L, calculate_U, calculate_X, level);

#ifdef CHECK_TIME
            calculate_time_wait += pangulu_time_check_end(&GET_TIME_START);
#endif
        }
        pangulu_bsem_stop(heap);

        pthread_mutex_lock(heap_mutex);
        if (heap->length != 0)
        {
            printf(PANGULU_W_RANK_HEAP_DONT_NULL);

            while (!heap_empty(heap))
            {
                pangulu_int64_t compare_flag = pangulu_heap_delete(heap);
                pangulu_numerical_work(compare_queue + compare_flag, block_common, block_smatrix,
                                       calculate_L, calculate_U, calculate_X, level);
            }
        }
        pthread_mutex_unlock(heap_mutex);

        if ((now_flag % 2) == 0)
        {
            pangulu_bsem_synchronize(run_bsem1);
        }
        else
        {
            pangulu_bsem_synchronize(run_bsem2);
        }
        now_flag++;
    }
    return NULL;
}

void pangulu_create_pthread(pangulu_block_common *block_common,
                            pangulu_block_smatrix *block_smatrix)
{
    pthread_t pthread;
    thread_param param;
    param.common = block_common;
    param.smatrix = block_smatrix;
    pthread_create(&pthread, NULL, &thread_GPU_work, (void *)(&param));
    pangulu_bsem_synchronize(block_smatrix->run_bsem2);
}
#endif

void pangulu_numeric(pangulu_block_common *block_common,
                       pangulu_block_smatrix *block_smatrix)
{
    pangulu_int64_t cpu_thread_count_per_node = sysconf(_SC_NPROCESSORS_ONLN);
    #ifdef HT_IS_OPEN
    bind_to_core((2 * pangu_omp_num_threads * rank) % cpu_thread_count_per_node);
#else
    bind_to_core((pangu_omp_num_threads * rank) % cpu_thread_count_per_node);
#endif
    pangulu_int32_t block_length = block_common->block_length;
    pangulu_int32_t p = block_common->p;
    pangulu_int32_t q = block_common->q;
    pangulu_block_info_pool* BIP = block_smatrix->BIP;

    pangulu_int64_t every_level_length = block_common->every_level_length;

    pangulu_int64_t *receive_level_num = block_smatrix->receive_level_num;

    pangulu_int64_t *now_level_L_length = block_smatrix->now_level_l_length;
    pangulu_int64_t *now_level_U_length = block_smatrix->now_level_u_length;
    
    // init
    pangulu_heap *heap = block_smatrix->heap;

    pangulu_int32_t *grid_process_id = block_smatrix->grid_process_id;
    pangulu_int64_t *level_index = block_smatrix->level_index;

#ifdef OVERLAP

    bsem *run_bsem1 = block_smatrix->run_bsem1;
    bsem *run_bsem2 = block_smatrix->run_bsem2;

    bsem *heap_bsem = heap->heap_bsem;
    pthread_mutex_t *heap_mutex = heap_bsem->mutex;

    pangulu_int64_t now_flag = 0;

    for (pangulu_int64_t level = 0; level < block_length; level += every_level_length)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        for (pangulu_int64_t i = 0; i < block_smatrix->l_smatrix_nzz * every_level_length; i++)
        {
            block_smatrix->flag_save_l[i] = 0;
        }
        for (pangulu_int64_t i = 0; i < block_smatrix->u_smatrix_nzz * every_level_length; i++)
        {
            block_smatrix->flag_save_u[i] = 0;
        }

        global_level = level;
        for (pangulu_int64_t i = 0; i < block_smatrix->l_smatrix_nzz * every_level_length; i++)
        {
            (block_smatrix->l_pangulu_smatrix_value[i])->zip_flag = 0 ;
        }

        pthread_mutex_lock(heap_mutex);

        pangulu_zero_pangulu_heap(heap);
        // init

        pangulu_int64_t big_level = ((level + every_level_length) > block_length) ? block_length : (level + every_level_length);
        for (pangulu_int64_t k = level; k < big_level; k++)
        {
            pangulu_int64_t now_level = level_index[k];
            pangulu_int64_t now_rank = grid_process_id[(now_level % p) * q + (now_level % q)];
            if (now_rank == rank)
            {
                pangulu_bip_set(now_level * block_length + now_level, BIP)->task_flag_id--;
                if (pangulu_bip_get(now_level * block_length + now_level, BIP)->task_flag_id == 0)
                {
                    pangulu_heap_insert(heap, now_level, now_level, now_level, 1, now_level);
                }
            }
        }
        for (pangulu_int64_t i = 0; i < every_level_length; i++)
        {
            now_level_L_length[i] = 0;
        }
        for (pangulu_int64_t i = 0; i < every_level_length; i++)
        {
            now_level_U_length[i] = 0;
        }

        pthread_mutex_unlock(heap_mutex);

        pangulu_bsem_post(heap);

        pangulu_int64_t now_receive_num = receive_level_num[level / every_level_length];
        MPI_Status status;

        while (now_receive_num != 0)
        {
            pangulu_probe_message(&status);
            now_receive_num--;
            pangulu_numerical_receive_message(status, level, block_common, block_smatrix);
            pangulu_bsem_post(heap);
        }

        if ((now_flag % 2) == 0)
        {
            pangulu_bsem_synchronize(run_bsem1);
        }
        else
        {
            pangulu_bsem_synchronize(run_bsem2);
        }
        now_flag++;

        pangulu_int64_t flag = pangulu_iprobe_message(&status);
        if (flag == 1)
        {
            printf(PANGULU_W_ERR_RANK);
            pangulu_numerical_receive_message(status, level, block_common, block_smatrix);
        }
    }
#endif
}
