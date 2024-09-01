#include "pangulu_common.h"

void pangulu_sptrsv_preprocessing(pangulu_block_common *block_common,
                               pangulu_block_smatrix *block_smatrix,
                               pangulu_vector *vector)
{

    pangulu_int64_t n = block_common->n;
    pangulu_int32_t vector_number = 1;
    pangulu_int32_t rank = block_common->rank;
    pangulu_int32_t p = block_common->p;
    pangulu_int32_t q = block_common->q;
    pangulu_int32_t nb = block_common->nb;
    pangulu_int32_t block_length = block_common->block_length;
    pangulu_int32_t rank_row_length = block_common->rank_row_length;
    pangulu_int32_t rank_col_length = block_common->rank_col_length;
    pangulu_block_info_pool* BIP = block_smatrix->BIP;
    calculate_type *save_vector;
    pangulu_int64_t *diagonal_num = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (p * q + 1));
    if (rank == 0)
    {
        for (pangulu_int64_t i = 0; i < p * q + 1; i++)
        {
            diagonal_num[i] = 0;
        }
        for (pangulu_int64_t i = 0; i < block_length; i++)
        {
            pangulu_int64_t diagonal_rank = calculate_diagonal_rank(i, p, q);
            diagonal_num[diagonal_rank + 1] += nb;
        }
        for (pangulu_int64_t i = 0; i < p * q; i++)
        {
            diagonal_num[i + 1] += diagonal_num[i];
        }
    }
    pangulu_bcast_vector_int64(diagonal_num, p * q + 1, 0);
    if (rank == 0)
    {
        save_vector = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * (block_length * nb));

        pangulu_int64_t *save_diagonal_num = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (p * q + 1));
        for (pangulu_int64_t i = 0; i < p * q + 1; i++)
        {
            save_diagonal_num[i] = diagonal_num[i];
        }
        for (pangulu_int64_t i = 0; i < block_length; i++)
        {
            pangulu_int64_t now_length = ((i + 1) * nb) > n ? (n - i * nb) : nb;
            pangulu_int64_t now_add_rank = calculate_diagonal_rank(i, p, q);
            calculate_type *begin_diagnoal = save_vector + save_diagonal_num[now_add_rank];
            calculate_type *vector_begin = vector->value + i * nb;
            for (pangulu_int64_t j = 0; j < now_length; j++)
            {
                begin_diagnoal[j] = vector_begin[j];
            }
            save_diagonal_num[now_add_rank] += nb;
        }

        for (pangulu_int64_t i = 1; i < p * q; i++)
        {
            pangulu_send_vector_value(save_vector + diagonal_num[i],
                                      diagonal_num[i + 1] - diagonal_num[i], i, i);
        }
        pangulu_free(__FILE__, __LINE__, save_diagonal_num);
    }
    else
    {
        save_vector = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * (diagonal_num[rank + 1] - diagonal_num[rank]));
        pangulu_recv_vector_value(save_vector, diagonal_num[rank + 1] - diagonal_num[rank], 0, rank); 
    }
    pangulu_int64_t offset_row_init = rank / q;
    pangulu_int64_t offset_col_init = rank % q;
    pangulu_vector **big_col_vector = (pangulu_vector **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector *) * rank_col_length);
    char *diagonal_flag = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(char) * rank_col_length);
    for (pangulu_int64_t i = 0; i < rank_col_length; i++)
    {
        diagonal_flag[i] = 0;
    }
    for (pangulu_int64_t i = offset_col_init, now_diagonal_num = 0, k = 0; i < block_length; i += q, k++)
    {
        pangulu_int64_t offset_row = calculate_offset(offset_row_init, i, p);
        pangulu_vector *new_vector = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));
        pangulu_init_pangulu_vector(new_vector, nb * vector_number);
        big_col_vector[k] = new_vector;
        if (offset_row == 0)
        {
            diagonal_flag[k] = 1;
            for (pangulu_int64_t vector_index = 0; vector_index < vector_number; vector_index++)
            {
                calculate_type *now_value = vector_index * nb + new_vector->value;
                for (pangulu_int64_t j = 0; j < nb; j++)
                {
                    now_value[j] = save_vector[now_diagonal_num + j];
                }
            }
            now_diagonal_num += nb;
        }
    }
    pangulu_free(__FILE__, __LINE__, save_vector);
    pangulu_free(__FILE__, __LINE__, diagonal_num);

    pangulu_vector **big_row_vector = (pangulu_vector **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector *) * rank_row_length);
    for (pangulu_int64_t i = 0; i < rank_row_length; i++)
    {
        pangulu_vector *new_vector = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));
        pangulu_init_pangulu_vector(new_vector, nb * vector_number);
        big_row_vector[i] = new_vector;
    }

    pangulu_exblock_ptr *L_row_task_nnz = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * rank_row_length);
    pangulu_exblock_ptr *L_col_task_nnz = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * rank_col_length);

    pangulu_int64_t *row_flag = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * q);
    pangulu_int64_t *col_flag = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * p);

    char *L_send_flag = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(char) * rank_col_length * (p - 1));

    pangulu_exblock_ptr *U_row_task_nnz = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * rank_row_length);
    pangulu_exblock_ptr *U_col_task_nnz = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * rank_col_length);

    char *U_send_flag = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(char) * rank_col_length * (p - 1));

    for (pangulu_int64_t i = 0; i < rank_row_length; i++)
    {
        L_row_task_nnz[i] = 0;
    }

    for (pangulu_int64_t i = 0; i < rank_col_length; i++)
    {
        L_col_task_nnz[i] = 0;
    }

    for (pangulu_int64_t i = 0; i < rank_row_length; i++)
    {
        U_row_task_nnz[i] = 0;
    }

    for (pangulu_int64_t i = 0; i < rank_col_length; i++)
    {
        U_col_task_nnz[i] = 0;
    }

    for (pangulu_int64_t i = 0; i < rank_col_length * (p - 1); i++)
    {
        L_send_flag[i] = 0;
    }

    for (pangulu_int64_t i = 0; i < rank_col_length * (p - 1); i++)
    {
        U_send_flag[i] = 0;
    }

    pangulu_int64_t L_sum_task_num = 0;
    pangulu_exblock_ptr *L_columnpointer = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (rank_col_length + 1));

    for (pangulu_int64_t i = 0; i < (rank_col_length + 1); i++)
    {
        L_columnpointer[i] = 0;
    }
    for (pangulu_int64_t i = offset_row_init, k = 0; i < block_length; i += p, k++)
    {
        pangulu_int64_t offset_col = calculate_offset(offset_col_init, i, q);
        pangulu_int64_t length = 0;
        for (pangulu_int64_t j = offset_col_init; j <= i; j += q)
        {
            if (pangulu_bip_get(i * block_length + j, BIP)->block_smatrix_nnza_num != 0)
            {
                length++;
                L_columnpointer[j / q + 1]++;
            }
        }
        if (offset_col == 0)
        {
            L_row_task_nnz[k] = length - 1;
        }
        else
        {
            L_row_task_nnz[k] = length;
        }

        L_sum_task_num += length;
    }

    for (pangulu_int64_t i = offset_col_init, k = 0; i < block_length; i += q, k++)
    {
        pangulu_int64_t offset_row = calculate_offset(offset_row_init, i, p);
        if (offset_row == 0)
        {

            for (pangulu_int64_t j = 0; j < q; j++)
            {
                row_flag[j] = 0;
            }
            pangulu_int64_t nnz_row_recv = 0;
            for (pangulu_int64_t j = 0; j < i; j++)
            {
                if (pangulu_bip_get(j + i * block_length, BIP)->block_smatrix_nnza_num != 0)
                {
                    row_flag[j % q]++;
                }
            }
            for (pangulu_int64_t j = 0; j < q; j++)
            {
                if (row_flag[j] != 0)
                {
                    nnz_row_recv++;
                }
            }
            L_col_task_nnz[k] = nnz_row_recv;

            for (pangulu_int64_t j = 0; j < p; j++)
            {
                col_flag[j] = 0;
            }
            for (pangulu_int64_t j = i + 1, f = 1; j < block_length; j++, f++)
            {
                if (pangulu_bip_get(j * block_length + i, BIP)->block_smatrix_nnza_num != 0)
                {
                    col_flag[f % p] = 1;
                }
            }
            for (pangulu_int64_t j = 1; j < p; j++)
            {
                L_send_flag[k * (p - 1) + j - 1] = col_flag[j];
            }
        }
        else
        {
            L_col_task_nnz[k] = 1;
        }
    }

    pangulu_int64_t U_sum_task_num = 0;
    pangulu_exblock_ptr *U_columnpointer = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (rank_col_length + 1));

    for (pangulu_int64_t i = 0; i < (rank_col_length + 1); i++)
    {
        U_columnpointer[i] = 0;
    }

    for (pangulu_int64_t i = offset_row_init, k = 0; i < block_length; i += p, k++)
    {
        pangulu_int64_t offset_col = calculate_offset(offset_col_init, i, q);
        pangulu_int64_t length = 0;
        for (pangulu_int64_t j = i + offset_col; j < block_length; j += q)
        {
            if (pangulu_bip_get(i * block_length + j, BIP)->block_smatrix_nnza_num != 0)
            {
                length++;
                U_columnpointer[j / q + 1]++;
            }
        }
        if (offset_col == 0)
        {
            U_row_task_nnz[k] = length - 1;
        }
        else
        {
            U_row_task_nnz[k] = length;
        }

        U_sum_task_num += length;
    }

    for (pangulu_int64_t i = offset_col_init, k = 0; i < block_length; i += q, k++)
    {
        pangulu_int64_t offset_row = calculate_offset(offset_row_init, i, p);
        if (offset_row == 0)
        {

            for (pangulu_int64_t j = 0; j < q; j++)
            {
                row_flag[j] = 0;
            }
            pangulu_int64_t nnz_row_recv = 0;
            for (pangulu_int64_t j = i + 1; j < block_length; j++)
            {
                if (pangulu_bip_get(j + i * block_length, BIP)->block_smatrix_nnza_num != 0)
                {
                    row_flag[j % q]++;
                }
            }
            for (pangulu_int64_t j = 0; j < q; j++)
            {
                if (row_flag[j] != 0)
                {
                    nnz_row_recv++;
                }
            }
            U_col_task_nnz[k] = nnz_row_recv;

            for (pangulu_int64_t j = 0; j < p; j++)
            {
                col_flag[j] = 0;
            }
            for (pangulu_int64_t j = i - 1, f = 1; j >= 0; j--, f++)
            {
                if (pangulu_bip_get(j * block_length + i, BIP)->block_smatrix_nnza_num != 0)
                {
                    col_flag[f % p] = 1;
                }
            }
            for (pangulu_int64_t j = 1; j < p; j++)
            {
                U_send_flag[k * (p - 1) + j - 1] = col_flag[j];
            }
        }
        else
        {
            U_col_task_nnz[k] = 1;
        }
    }

    pangulu_free(__FILE__, __LINE__, row_flag);
    pangulu_free(__FILE__, __LINE__, col_flag);

    pangulu_exblock_idx *L_rowindex = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * L_sum_task_num);
    pangulu_exblock_idx *U_rowindex = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * U_sum_task_num);

    for (pangulu_int64_t i = 0; i < rank_col_length; i++)
    {
        L_columnpointer[i + 1] += L_columnpointer[i];
    }

    for (pangulu_int64_t i = 0; i < rank_col_length; i++)
    {
        U_columnpointer[i + 1] += U_columnpointer[i];
    }

    for (pangulu_int64_t i = offset_row_init, k = 0; i < block_length; i += p, k++)
    {
        for (pangulu_int64_t j = offset_col_init; j <= i; j += q)
        {
            if (pangulu_bip_get(i * block_length + j, BIP)->block_smatrix_nnza_num != 0)
            {
                L_rowindex[L_columnpointer[j / q]++] = i;
            }
        }
    }

    for (pangulu_int64_t i = offset_row_init, k = 0; i < block_length; i += p, k++)
    {
        pangulu_int64_t offset_col = calculate_offset(offset_col_init, i, q);
        for (pangulu_int64_t j = i + offset_col; j < block_length; j += q)
        {
            if (pangulu_bip_get(i * block_length + j, BIP)->block_smatrix_nnza_num != 0)
            {
                U_rowindex[U_columnpointer[j / q]++] = i;
            }
        }
    }
    for (pangulu_int64_t i = rank_col_length - 1; i > 0; i--)
    {
        L_columnpointer[i] = L_columnpointer[i - 1];
    }
    L_columnpointer[0] = 0;
    for (pangulu_int64_t i = rank_col_length - 1; i > 0; i--)
    {
        U_columnpointer[i] = U_columnpointer[i - 1];
    }
    U_columnpointer[0] = 0;

    pangulu_heap *sptrsv_heap = (pangulu_heap *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_heap));
    pangulu_init_pangulu_heap(sptrsv_heap, PANGULU_MAX(L_sum_task_num, U_sum_task_num));

    pangulu_vector *S_vector = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));
    pangulu_init_pangulu_vector(S_vector, nb);

    block_smatrix->big_row_vector = big_row_vector;
    block_smatrix->big_col_vector = big_col_vector;
    block_smatrix->diagonal_flag = diagonal_flag;

    block_smatrix->l_row_task_nnz = L_row_task_nnz;
    block_smatrix->l_col_task_nnz = L_col_task_nnz;

    block_smatrix->u_row_task_nnz = U_row_task_nnz;
    block_smatrix->u_col_task_nnz = U_col_task_nnz;

    block_smatrix->sptrsv_heap = sptrsv_heap;
    block_smatrix->save_vector = S_vector;

    block_smatrix->l_send_flag = L_send_flag;
    block_smatrix->u_send_flag = U_send_flag;

    block_smatrix->l_sptrsv_task_columnpointer = L_columnpointer;
    block_smatrix->l_sptrsv_task_rowindex = L_rowindex;

    block_smatrix->u_sptrsv_task_columnpointer = U_columnpointer;
    block_smatrix->u_sptrsv_task_rowindex = U_rowindex;
}

void L_pangulu_sptrsv_receive_message(MPI_Status status,
                                      pangulu_block_common *block_common,
                                      pangulu_block_smatrix *block_smatrix)
{

    pangulu_int32_t q = block_common->q;
    pangulu_int32_t nb = block_common->nb;
    pangulu_int64_t receive_id = status.MPI_SOURCE;
    pangulu_int64_t tag = status.MPI_TAG;
    pangulu_int64_t col = tag;
    pangulu_int64_t coloffset = col / q;
    char *diagonal_flag = block_smatrix->diagonal_flag;
    pangulu_vector *col_vector = block_smatrix->big_col_vector[coloffset];
    if (diagonal_flag[coloffset] == 0)
    {
        pangulu_recv_pangulu_vector_value(col_vector, receive_id, tag, nb);
    }
    else
    {
        pangulu_vector *save_vector = block_smatrix->save_vector;
        pangulu_recv_pangulu_vector_value(save_vector, receive_id, tag, nb);
        pangulu_vector_sub(col_vector, save_vector);
    }

    pangulu_exblock_ptr *L_col_task_nnz = block_smatrix->l_col_task_nnz;
    L_col_task_nnz[coloffset]--;
    if (L_col_task_nnz[coloffset] == 0)
    {
        pangulu_exblock_ptr *L_columnpointer = block_smatrix->l_sptrsv_task_columnpointer;
        pangulu_exblock_idx *L_rowindex = block_smatrix->l_sptrsv_task_rowindex;
        pangulu_heap *sptrsv_heap = block_smatrix->sptrsv_heap;

        for (pangulu_int64_t k = L_columnpointer[coloffset]; k < L_columnpointer[coloffset + 1]; k++)
        {
            pangulu_int64_t row = L_rowindex[k];
            pangulu_heap_insert(sptrsv_heap, row, col, 0, (row == col) ? 1 : 0, col);
        }
    }
}

void L_pangulu_sptrsv_work(compare_struct *flag,
                           pangulu_block_common *block_common,
                           pangulu_block_smatrix *block_smatrix,
                           pangulu_vector *save_vector)
{
    pangulu_int64_t kernel_id = flag->kernel_id;
    pangulu_int64_t row = flag->row;
    pangulu_int64_t col = flag->col;
    pangulu_int32_t p = block_common->p;
    pangulu_int32_t q = block_common->q;
    pangulu_int32_t block_length = block_common->block_length;
    pangulu_int32_t nb = block_common->nb;
    pangulu_int32_t rank = block_common->rank;
    pangulu_vector *row_vector = block_smatrix->big_row_vector[row / p];
    pangulu_vector *col_vector = block_smatrix->big_col_vector[col / q];
    pangulu_block_info_pool* BIP = block_smatrix->BIP;

    pangulu_zero_pangulu_vector(save_vector);
    if (kernel_id == 0)
    {
        pangulu_int64_t mapper_index = pangulu_bip_get(block_length * row + col, BIP)->mapper_a;
        pangulu_smatrix *a = &block_smatrix->big_pangulu_smatrix_value[mapper_index];

        pangulu_spmv(a, col_vector, save_vector, 1);
        pangulu_vector_add(row_vector, save_vector);

        pangulu_exblock_ptr *L_row_task_nnz = block_smatrix->l_row_task_nnz;
        pangulu_int64_t rowindex = row / p;
        L_row_task_nnz[rowindex]--;
        if (L_row_task_nnz[rowindex] == 0)
        {
            pangulu_int32_t diagonal_rank = row % p * q + row % q;
            pangulu_int32_t diagonal_task = row;
            pangulu_isend_pangulu_vector_value(row_vector,
                                               diagonal_rank, diagonal_task, nb);
        }
        return;
    }
    else if (kernel_id == 1)
    {
        pangulu_int64_t diagonal_index = block_smatrix->mapper_diagonal[row];
        pangulu_smatrix *l = block_smatrix->diagonal_smatrix_l[diagonal_index];
        pangulu_sptrsv(l, save_vector, col_vector, 1, 0);
        pangulu_int32_t diagonal_task = row;
        pangulu_vector_copy(col_vector, save_vector);
        char *L_send_flag = block_smatrix->l_send_flag + (col / q) * (p - 1);
        for (pangulu_int64_t k = 1; k < p; k++)
        {
            if (L_send_flag[k - 1] == 1)
            {
                int32_t diagonal_rank = (rank + k * q) % (p * q);
                pangulu_isend_pangulu_vector_value(col_vector,
                                                   diagonal_rank, diagonal_task, nb);
            }
        }
        return;
    }
    else
    {
        printf(PANGULU_E_WORK_ERR);
        pangulu_exit(1);
    }
    return;
}

void pangulu_sptrsv_L(pangulu_block_common *block_common,
                      pangulu_block_smatrix *block_smatrix)
{

    pangulu_exblock_ptr *col_task_nnz = block_smatrix->l_col_task_nnz;
    int32_t rank_col_length = block_common->rank_col_length;
    pangulu_int64_t reiceive_num = 0;
    pangulu_heap *sptrsv_heap = block_smatrix->sptrsv_heap;
    pangulu_zero_pangulu_heap(sptrsv_heap);
    compare_struct *compare_queue = sptrsv_heap->comapre_queue;

    char *diagonal_flag = block_smatrix->diagonal_flag;
    pangulu_exblock_ptr *L_columnpointer = block_smatrix->l_sptrsv_task_columnpointer;
    pangulu_exblock_idx *L_rowindex = block_smatrix->l_sptrsv_task_rowindex;
    pangulu_int32_t rank = block_common->rank;
    pangulu_int32_t q = block_common->q;

    for (pangulu_int64_t i = 0; i < block_common->rank_row_length; i++)
    {
        pangulu_zero_pangulu_vector(block_smatrix->big_row_vector[i]);
    }

    pangulu_vector *save_vector = block_smatrix->save_vector;

    for (pangulu_int32_t i = 0, j = rank % q; i < rank_col_length; i++, j += q)
    {
        if (col_task_nnz[i] == 0)
        {
            pangulu_int64_t col = j;

            for (pangulu_int64_t k = L_columnpointer[i]; k < L_columnpointer[i + 1]; k++)
            {
                pangulu_int64_t row = L_rowindex[k];
                pangulu_heap_insert(sptrsv_heap, row, col, 0, (row == col) ? 1 : 0, col);
            }
        }
        else
        {
            pangulu_int64_t length = L_columnpointer[i + 1] - L_columnpointer[i];
            if (diagonal_flag[i] == 1)
            {
                reiceive_num += col_task_nnz[i];
            }
            else
            {
                if (length > 0)
                {
                    reiceive_num += col_task_nnz[i];
                }
            }
        }
    }
    while (reiceive_num != 0)
    {
        if (heap_empty(sptrsv_heap) == 1)
        {
            MPI_Status status;
            pangulu_probe_message(&status);
            L_pangulu_sptrsv_receive_message(status, block_common, block_smatrix);
            reiceive_num--;
        }
        else
        {
            MPI_Status status;
            pangulu_int32_t flag = pangulu_iprobe_message(&status);
            if (flag == 1)
            {
                L_pangulu_sptrsv_receive_message(status, block_common, block_smatrix);
                reiceive_num--;
            }
        }
        if (heap_empty(sptrsv_heap) == 0)
        {
            pangulu_int64_t compare_flag = pangulu_heap_delete(sptrsv_heap);
            L_pangulu_sptrsv_work(compare_queue + compare_flag, block_common, block_smatrix, save_vector);
        }
    }
    while (heap_empty(sptrsv_heap) == 0)
    {
        pangulu_int64_t compare_flag = pangulu_heap_delete(sptrsv_heap);
        L_pangulu_sptrsv_work(compare_queue + compare_flag, block_common, block_smatrix, save_vector);
    }
}

void u_pangulu_sptrsv_receive_message(MPI_Status status,
                                      pangulu_block_common *block_common,
                                      pangulu_block_smatrix *block_smatrix)
{

    pangulu_int32_t q = block_common->q;
    pangulu_int32_t nb = block_common->nb;
    pangulu_int64_t receive_id = status.MPI_SOURCE;
    pangulu_int64_t tag = status.MPI_TAG;
    pangulu_int64_t col = tag;
    pangulu_int64_t coloffset = col / q;
    char *diagonal_flag = block_smatrix->diagonal_flag;
    pangulu_vector *col_vector = block_smatrix->big_col_vector[coloffset];
    if (diagonal_flag[coloffset] == 0)
    {
        pangulu_recv_pangulu_vector_value(col_vector, receive_id, tag, nb);
    }
    else
    {
        pangulu_vector *save_vector = block_smatrix->save_vector;
        pangulu_recv_pangulu_vector_value(save_vector, receive_id, tag, nb);
        pangulu_vector_sub(col_vector, save_vector);
    }

    pangulu_exblock_ptr *U_col_task_nnz = block_smatrix->u_col_task_nnz;
    U_col_task_nnz[coloffset]--;
    if (U_col_task_nnz[coloffset] == 0)
    {
        pangulu_exblock_ptr *U_columnpointer = block_smatrix->u_sptrsv_task_columnpointer;
        pangulu_exblock_idx *U_rowindex = block_smatrix->u_sptrsv_task_rowindex;
        pangulu_heap *sptrsv_heap = block_smatrix->sptrsv_heap;
        for (pangulu_int64_t k = U_columnpointer[coloffset]; k < U_columnpointer[coloffset + 1]; k++)
        {
            pangulu_int64_t row = U_rowindex[k];
            pangulu_heap_insert(sptrsv_heap, row, col, 0, (row == col) ? 1 : 0, col);
        }
    }
}

void u_pangulu_sptrsv_work(compare_struct *flag,
                           pangulu_block_common *block_common,
                           pangulu_block_smatrix *block_smatrix,
                           pangulu_vector *save_vector)
{
    pangulu_int64_t kernel_id = flag->kernel_id;
    pangulu_int64_t row = flag->row;
    pangulu_int64_t col = flag->col;
    pangulu_int32_t p = block_common->p;
    pangulu_int32_t q = block_common->q;
    pangulu_int32_t block_length = block_common->block_length;
    pangulu_int32_t nb = block_common->nb;
    pangulu_int32_t rank = block_common->rank;
    pangulu_block_info_pool* BIP = block_smatrix->BIP;

    pangulu_vector *row_vector = block_smatrix->big_row_vector[row / p];
    pangulu_vector *col_vector = block_smatrix->big_col_vector[col / q];

    pangulu_zero_pangulu_vector(save_vector);
    if (kernel_id == 0)
    {
        pangulu_int64_t mapper_index = pangulu_bip_get(block_length * row + col, BIP)->mapper_a;
        pangulu_smatrix *a = &block_smatrix->big_pangulu_smatrix_value[mapper_index];

        pangulu_spmv(a, col_vector, save_vector, 1);
        pangulu_vector_add(row_vector, save_vector);
        pangulu_exblock_ptr *U_row_task_nnz = block_smatrix->u_row_task_nnz;
        pangulu_int64_t rowindex = row / p;
        U_row_task_nnz[rowindex]--;
        if (U_row_task_nnz[rowindex] == 0)
        {
            pangulu_int32_t diagonal_rank = row % p * q + row % q;
            pangulu_int32_t diagonal_task = row;
            pangulu_isend_pangulu_vector_value(row_vector,
                                               diagonal_rank, diagonal_task, nb);
        }
        return;
    }
    else if (kernel_id == 1)
    {
        pangulu_int64_t diagonal_index = block_smatrix->mapper_diagonal[row];
        pangulu_smatrix *u = block_smatrix->diagonal_smatrix_u[diagonal_index];

        pangulu_sptrsv(u, save_vector, col_vector, 1, 1);
        pangulu_int32_t diagonal_task = row;
        pangulu_vector_copy(col_vector, save_vector);
        char *U_send_flag = block_smatrix->u_send_flag + (col / q) * (p - 1);
        for (pangulu_int32_t k = 1; k < p; k++)
        {
            if (U_send_flag[k - 1] == 1)
            {
                int32_t diagonal_rank = (rank + (p - k) * q) % (p * q);
                pangulu_isend_pangulu_vector_value(col_vector,
                                                   diagonal_rank, diagonal_task, nb);
            }
        }
        return;
    }
    else
    {
        printf(PANGULU_E_WORK_ERR);
        pangulu_exit(1);
    }
    return;
}

void pangulu_sptrsv_U(pangulu_block_common *block_common,
                      pangulu_block_smatrix *block_smatrix)
{
    pangulu_exblock_ptr *col_task_nnz = block_smatrix->u_col_task_nnz;
    int32_t rank_col_length = block_common->rank_col_length;
    pangulu_int64_t reiceive_num = 0;
    pangulu_heap *sptrsv_heap = block_smatrix->sptrsv_heap;
    pangulu_zero_pangulu_heap(sptrsv_heap);
    compare_struct *compare_queue = sptrsv_heap->comapre_queue;

    char *diagonal_flag = block_smatrix->diagonal_flag;
    pangulu_exblock_ptr *U_columnpointer = block_smatrix->u_sptrsv_task_columnpointer;
    pangulu_exblock_idx *U_rowindex = block_smatrix->u_sptrsv_task_rowindex;
    pangulu_int32_t rank = block_common->rank;
    pangulu_int32_t q = block_common->q;

    for (pangulu_int64_t i = 0; i < block_common->rank_row_length; i++)
    {
        pangulu_zero_pangulu_vector(block_smatrix->big_row_vector[i]);
    }

    pangulu_vector *save_vector = block_smatrix->save_vector;

    for (pangulu_int32_t i = 0, j = rank % q; i < rank_col_length; i++, j += q)
    {
        if (col_task_nnz[i] == 0)
        {
            pangulu_int64_t col = j;

            for (pangulu_int64_t k = U_columnpointer[i]; k < U_columnpointer[i + 1]; k++)
            {
                pangulu_int64_t row = U_rowindex[k];
                pangulu_heap_insert(sptrsv_heap, row, col, 0, (row == col) ? 1 : 0, col);
            }
        }
        else
        {
            pangulu_int64_t length = U_columnpointer[i + 1] - U_columnpointer[i];
            if (diagonal_flag[i] == 1)
            {
                reiceive_num += col_task_nnz[i];
            }
            else
            {
                if (length > 0)
                {
                    reiceive_num += col_task_nnz[i];
                }
            }
        }
    }
    while (reiceive_num != 0)
    {
        if (heap_empty(sptrsv_heap) == 1)
        {
            MPI_Status status;
            pangulu_probe_message(&status);
            u_pangulu_sptrsv_receive_message(status, block_common, block_smatrix);
            reiceive_num--;
        }
        else
        {
            MPI_Status status;
            pangulu_int32_t flag = pangulu_iprobe_message(&status);
            if (flag == 1)
            {
                u_pangulu_sptrsv_receive_message(status, block_common, block_smatrix);
                reiceive_num--;
            }
        }
        if (heap_empty(sptrsv_heap) == 0)
        {
            pangulu_int64_t compare_flag = pangulu_heap_delete(sptrsv_heap);
            u_pangulu_sptrsv_work(compare_queue + compare_flag, block_common, block_smatrix, save_vector);
        }
    }
    while (heap_empty(sptrsv_heap) == 0)
    {
        pangulu_int64_t compare_flag = pangulu_heap_delete(sptrsv_heap);
        u_pangulu_sptrsv_work(compare_queue + compare_flag, block_common, block_smatrix, save_vector);
    }
}

void pangulu_sptrsv_vector_gather(pangulu_block_common *block_common,
                                  pangulu_block_smatrix *block_smatrix,
                                  pangulu_vector *vector)
{

    pangulu_int64_t n = block_common->n;
    pangulu_int32_t vector_number = 1;
    pangulu_int32_t rank = block_common->rank;
    pangulu_int32_t p = block_common->p;
    pangulu_int32_t q = block_common->q;
    pangulu_int32_t nb = block_common->nb;
    pangulu_int32_t block_length = block_common->block_length;
    pangulu_int32_t rank_col_length = block_common->rank_col_length;
    pangulu_int64_t offset_row_init = rank / q;
    pangulu_int64_t offset_col_init = rank % q;

    calculate_type *save_vector;

    pangulu_int64_t *diagonal_num = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (p * q + 1));
    if (rank == 0)
    {
        for (pangulu_int64_t i = 0; i < p * q + 1; i++)
        {
            diagonal_num[i] = 0;
        }
        for (pangulu_int64_t i = 0; i < block_length; i++)
        {
            pangulu_int64_t diagonal_rank = calculate_diagonal_rank(i, p, q);
            diagonal_num[diagonal_rank + 1] += nb;
        }
        for (pangulu_int64_t i = 0; i < p * q; i++)
        {
            diagonal_num[i + 1] += diagonal_num[i];
        }
    }
    pangulu_bcast_vector_int64(diagonal_num, p * q + 1, 0);
    if (rank == 0)
    {
        save_vector = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * (diagonal_num[p * q]));
    }
    else
    {
        save_vector = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * (diagonal_num[rank + 1] - diagonal_num[rank]));
    }
    pangulu_int64_t *diagonal_flag = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * rank_col_length);
    for (pangulu_int64_t i = 0; i < rank_col_length; i++)
    {
        diagonal_flag[i] = 0;
    }
    for (pangulu_int64_t i = offset_col_init, now_diagonal_num = 0, k = 0; i < block_length; i += q, k++)
    {
        pangulu_int64_t offset_row = calculate_offset(offset_row_init, i, p);
        pangulu_vector *new_vector = block_smatrix->big_col_vector[k];
        if (offset_row == 0)
        {
            diagonal_flag[k] = 1;
            for (pangulu_int64_t vector_index = 0; vector_index < vector_number; vector_index++)
            {
                calculate_type *now_value = vector_index * nb + new_vector->value;
                for (pangulu_int64_t j = 0; j < nb; j++)
                {
                    save_vector[now_diagonal_num + j] = now_value[j];
                }
            }
            now_diagonal_num += nb;
        }
    }

    if (rank == 0)
    {
        for (pangulu_int64_t i = 1; i < p * q; i++)
        {
            pangulu_recv_vector_value(save_vector + diagonal_num[i],
                                      diagonal_num[i + 1] - diagonal_num[i], i, i);
        }
    }
    else
    {
        pangulu_send_vector_value(save_vector, diagonal_num[rank + 1] - diagonal_num[rank], 0, rank);
    }

    if (rank == 0)
    {
        pangulu_int64_t *save_diagonal_num = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (p * q + 1));
        for (pangulu_int64_t i = 0; i < p * q + 1; i++)
        {
            save_diagonal_num[i] = diagonal_num[i];
        }
        for (pangulu_int64_t i = 0; i < block_length; i++)
        {
            pangulu_int64_t now_length = ((i + 1) * nb) > n ? (n - i * nb) : nb;
            pangulu_int64_t now_add_rank = calculate_diagonal_rank(i, p, q);
            calculate_type *begin_diagnoal = save_vector + save_diagonal_num[now_add_rank];
            calculate_type *vector_begin = vector->value + i * nb;
            for (pangulu_int64_t j = 0; j < now_length; j++)
            {
                vector_begin[j] = begin_diagnoal[j];
            }
            save_diagonal_num[now_add_rank] += nb;
        }
        pangulu_free(__FILE__, __LINE__, save_diagonal_num);
    }

    pangulu_free(__FILE__, __LINE__, save_vector);
    pangulu_free(__FILE__, __LINE__, diagonal_num);
    pangulu_free(__FILE__, __LINE__, diagonal_flag);
}
