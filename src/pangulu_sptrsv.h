#ifndef PANGULU_SPTRSV_H
#define PANGULU_SPTRSV_H

#include "pangulu_common.h"
#include "pangulu_utils.h"

void pangulu_sptrsv_preprocessing(pangulu_block_common *block_common,
                               pangulu_block_Smatrix *block_Smatrix,
                               pangulu_vector *vector)
{

    int_t N = block_common->N;
    int_32t vector_number = 1;
    int_32t rank = block_common->rank;
    int_32t P = block_common->P;
    int_32t Q = block_common->Q;
    int_32t NB = block_common->NB;
    int_32t block_length = block_common->block_length;
    int_32t rank_row_length = block_common->rank_row_length;
    int_32t rank_col_length = block_common->rank_col_length;
    calculate_type *save_vector;
    int_t *diagonal_num = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (P * Q + 1));
    if (rank == 0)
    {
        for (int_t i = 0; i < P * Q + 1; i++)
        {
            diagonal_num[i] = 0;
        }
        for (int_t i = 0; i < block_length; i++)
        {
            int_t diagonal_rank = calculate_diagonal_rank(i, P, Q);
            diagonal_num[diagonal_rank + 1] += NB;
        }
        for (int_t i = 0; i < P * Q; i++)
        {
            diagonal_num[i + 1] += diagonal_num[i];
        }
    }
    pangulu_Bcast_vector(diagonal_num, P * Q + 1, 0);
    if (rank == 0)
    {
        save_vector = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * (block_length * NB));

        int_t *save_diagonal_num = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (P * Q + 1));
        for (int_t i = 0; i < P * Q + 1; i++)
        {
            save_diagonal_num[i] = diagonal_num[i];
        }
        for (int_t i = 0; i < block_length; i++)
        {
            int_t now_length = ((i + 1) * NB) > N ? (N - i * NB) : NB;
            int_t now_add_rank = calculate_diagonal_rank(i, P, Q);
            calculate_type *begin_diagnoal = save_vector + save_diagonal_num[now_add_rank];
            calculate_type *vector_begin = vector->value + i * NB;
            for (int_t j = 0; j < now_length; j++)
            {
                begin_diagnoal[j] = vector_begin[j];
            }
            save_diagonal_num[now_add_rank] += NB;
        }

        for (int_t i = 1; i < P * Q; i++)
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
    int_t offset_row_init = rank / Q;
    int_t offset_col_init = rank % Q;
    pangulu_vector **Big_col_vector = (pangulu_vector **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector *) * rank_col_length);
    int_t *diagonal_flag = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * rank_col_length);
    for (int_t i = 0; i < rank_col_length; i++)
    {
        diagonal_flag[i] = 0;
    }
    for (int_t i = offset_col_init, now_diagonal_num = 0, k = 0; i < block_length; i += Q, k++)
    {
        int_t offset_row = calculate_offset(offset_row_init, i, P);
        pangulu_vector *new_vector = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));
        pangulu_init_pangulu_vector(new_vector, NB * vector_number);
        Big_col_vector[k] = new_vector;
        if (offset_row == 0)
        {
            diagonal_flag[k] = 1;
            for (int_t vector_index = 0; vector_index < vector_number; vector_index++)
            {
                calculate_type *now_value = vector_index * NB + new_vector->value;
                for (int_t j = 0; j < NB; j++)
                {
                    now_value[j] = save_vector[now_diagonal_num + j];
                }
            }
            now_diagonal_num += NB;
        }
    }
    pangulu_free(__FILE__, __LINE__, save_vector);
    pangulu_free(__FILE__, __LINE__, diagonal_num);

    pangulu_vector **Big_row_vector = (pangulu_vector **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector *) * rank_row_length);
    for (int_t i = 0; i < rank_row_length; i++)
    {
        pangulu_vector *new_vector = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));
        pangulu_init_pangulu_vector(new_vector, NB * vector_number);
        Big_row_vector[i] = new_vector;
    }

    int_t *block_Smatrix_nnzA_num = block_Smatrix->block_Smatrix_nnzA_num;
    int_t *L_row_task_nnz = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * rank_row_length);
    int_t *L_col_task_nnz = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * rank_col_length);

    int_t *row_flag = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * Q);
    int_t *col_flag = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * P);

    int_t *L_send_flag = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * rank_col_length * (P - 1));

    int_t *U_row_task_nnz = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * rank_row_length);
    int_t *U_col_task_nnz = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * rank_col_length);

    int_t *U_send_flag = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * rank_col_length * (P - 1));

    for (int_t i = 0; i < rank_row_length; i++)
    {
        L_row_task_nnz[i] = 0;
    }

    for (int_t i = 0; i < rank_col_length; i++)
    {
        L_col_task_nnz[i] = 0;
    }

    for (int_t i = 0; i < rank_row_length; i++)
    {
        U_row_task_nnz[i] = 0;
    }

    for (int_t i = 0; i < rank_col_length; i++)
    {
        U_col_task_nnz[i] = 0;
    }

    for (int_t i = 0; i < rank_col_length * (P - 1); i++)
    {
        L_send_flag[i] = 0;
    }

    for (int_t i = 0; i < rank_col_length * (P - 1); i++)
    {
        U_send_flag[i] = 0;
    }

    int_t L_sum_task_num = 0;
    int_t *L_columnpointer = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (rank_col_length + 1));

    for (int_t i = 0; i < (rank_col_length + 1); i++)
    {
        L_columnpointer[i] = 0;
    }
    for (int_t i = offset_row_init, k = 0; i < block_length; i += P, k++)
    {
        int_t offset_col = calculate_offset(offset_col_init, i, Q);
        int_t length = 0;
        for (int_t j = offset_col_init; j <= i; j += Q)
        {
            if (block_Smatrix_nnzA_num[i * block_length + j] != 0)
            {
                length++;
                L_columnpointer[j / Q + 1]++;
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

    for (int_t i = offset_col_init, k = 0; i < block_length; i += Q, k++)
    {
        int_t offset_row = calculate_offset(offset_row_init, i, P);
        if (offset_row == 0)
        {

            for (int_t j = 0; j < Q; j++)
            {
                row_flag[j] = 0;
            }
            int_t nnz_row_recv = 0;
            int_t *now_block_Smatrix_nnzA = block_Smatrix_nnzA_num + i * block_length;
            for (int_t j = 0; j < i; j++)
            {
                if (now_block_Smatrix_nnzA[j] != 0)
                {
                    row_flag[j % Q]++;
                }
            }
            for (int_t j = 0; j < Q; j++)
            {
                if (row_flag[j] != 0)
                {
                    nnz_row_recv++;
                }
            }
            L_col_task_nnz[k] = nnz_row_recv;

            for (int_t j = 0; j < P; j++)
            {
                col_flag[j] = 0;
            }
            for (int_t j = i + 1, f = 1; j < block_length; j++, f++)
            {
                if (block_Smatrix_nnzA_num[j * block_length + i] != 0)
                {
                    col_flag[f % P] = 1;
                }
            }
            for (int_t j = 1; j < P; j++)
            {
                L_send_flag[k * (P - 1) + j - 1] = col_flag[j];
            }
        }
        else
        {
            L_col_task_nnz[k] = 1;
        }
    }
    // return ;

    int_t U_sum_task_num = 0;
    int_t *U_columnpointer = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (rank_col_length + 1));

    for (int_t i = 0; i < (rank_col_length + 1); i++)
    {
        U_columnpointer[i] = 0;
    }

    for (int_t i = offset_row_init, k = 0; i < block_length; i += P, k++)
    {
        int_t offset_col = calculate_offset(offset_col_init, i, Q);
        int_t length = 0;
        for (int_t j = i + offset_col; j < block_length; j += Q)
        {
            if (block_Smatrix_nnzA_num[i * block_length + j] != 0)
            {
                length++;
                U_columnpointer[j / Q + 1]++;
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

    for (int_t i = offset_col_init, k = 0; i < block_length; i += Q, k++)
    {
        int_t offset_row = calculate_offset(offset_row_init, i, P);
        if (offset_row == 0)
        {

            for (int_t j = 0; j < Q; j++)
            {
                row_flag[j] = 0;
            }
            int_t nnz_row_recv = 0;
            int_t *now_block_Smatrix_nnzA = block_Smatrix_nnzA_num + i * block_length;
            for (int_t j = i + 1; j < block_length; j++)
            {
                if (now_block_Smatrix_nnzA[j] != 0)
                {
                    row_flag[j % Q]++;
                }
            }
            for (int_t j = 0; j < Q; j++)
            {
                if (row_flag[j] != 0)
                {
                    nnz_row_recv++;
                }
            }
            U_col_task_nnz[k] = nnz_row_recv;

            for (int_t j = 0; j < P; j++)
            {
                col_flag[j] = 0;
            }
            for (int_t j = i - 1, f = 1; j >= 0; j--, f++)
            {
                if (block_Smatrix_nnzA_num[j * block_length + i] != 0)
                {
                    col_flag[f % P] = 1;
                }
            }
            for (int_t j = 1; j < P; j++)
            {
                U_send_flag[k * (P - 1) + j - 1] = col_flag[j];
            }
        }
        else
        {
            U_col_task_nnz[k] = 1;
        }
    }

    pangulu_free(__FILE__, __LINE__, row_flag);
    pangulu_free(__FILE__, __LINE__, col_flag);

    int_t *L_rowindex = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * L_sum_task_num);
    int_t *U_rowindex = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * U_sum_task_num);

    for (int_t i = 0; i < rank_col_length; i++)
    {
        L_columnpointer[i + 1] += L_columnpointer[i];
    }

    for (int_t i = 0; i < rank_col_length; i++)
    {
        U_columnpointer[i + 1] += U_columnpointer[i];
    }

    for (int_t i = offset_row_init, k = 0; i < block_length; i += P, k++)
    {
        for (int_t j = offset_col_init; j <= i; j += Q)
        {
            if (block_Smatrix_nnzA_num[i * block_length + j] != 0)
            {
                L_rowindex[L_columnpointer[j / Q]++] = i;
            }
        }
    }

    for (int_t i = offset_row_init, k = 0; i < block_length; i += P, k++)
    {
        int_t offset_col = calculate_offset(offset_col_init, i, Q);
        for (int_t j = i + offset_col; j < block_length; j += Q)
        {
            if (block_Smatrix_nnzA_num[i * block_length + j] != 0)
            {
                U_rowindex[U_columnpointer[j / Q]++] = i;
            }
        }
    }
    for (int_t i = rank_col_length - 1; i > 0; i--)
    {
        L_columnpointer[i] = L_columnpointer[i - 1];
    }
    L_columnpointer[0] = 0;
    for (int_t i = rank_col_length - 1; i > 0; i--)
    {
        U_columnpointer[i] = U_columnpointer[i - 1];
    }
    U_columnpointer[0] = 0;

    pangulu_heap *sptrsv_heap = (pangulu_heap *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_heap));
    pangulu_init_pangulu_heap(sptrsv_heap, PANGULU_MAX(L_sum_task_num, U_sum_task_num));

    pangulu_vector *S_vector = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));
    pangulu_init_pangulu_vector(S_vector, NB);

    block_Smatrix->Big_row_vector = Big_row_vector;
    block_Smatrix->Big_col_vector = Big_col_vector;
    block_Smatrix->diagonal_flag = diagonal_flag;

    block_Smatrix->L_row_task_nnz = L_row_task_nnz;
    block_Smatrix->L_col_task_nnz = L_col_task_nnz;

    block_Smatrix->U_row_task_nnz = U_row_task_nnz;
    block_Smatrix->U_col_task_nnz = U_col_task_nnz;

    block_Smatrix->sptrsv_heap = sptrsv_heap;
    block_Smatrix->save_vector = S_vector;

    block_Smatrix->L_send_flag = L_send_flag;
    block_Smatrix->U_send_flag = U_send_flag;

    block_Smatrix->L_sptrsv_task_columnpointer = L_columnpointer;
    block_Smatrix->L_sptrsv_task_rowindex = L_rowindex;

    block_Smatrix->U_sptrsv_task_columnpointer = U_columnpointer;
    block_Smatrix->U_sptrsv_task_rowindex = U_rowindex;
}

void L_pangulu_sptrsv_receive_message(MPI_Status status,
                                      pangulu_block_common *block_common,
                                      pangulu_block_Smatrix *block_Smatrix)
{

    int_32t Q = block_common->Q;
    int_32t NB = block_common->NB;
    int_t receive_id = status.MPI_SOURCE;
    int_t tag = status.MPI_TAG;
    int_t col = tag;
    int_t coloffset = col / Q;
    int_t *diagonal_flag = block_Smatrix->diagonal_flag;
    pangulu_vector *col_vector = block_Smatrix->Big_col_vector[coloffset];
    if (diagonal_flag[coloffset] == 0)
    {
        pangulu_recv_pangulu_vector_value(col_vector, receive_id, tag, NB);
    }
    else
    {
        pangulu_vector *save_vector = block_Smatrix->save_vector;
        pangulu_recv_pangulu_vector_value(save_vector, receive_id, tag, NB);
        pangulu_vector_sub(col_vector, save_vector);
    }

    int_t *L_col_task_nnz = block_Smatrix->L_col_task_nnz;
    L_col_task_nnz[coloffset]--;
    if (L_col_task_nnz[coloffset] == 0)
    {
        int_t *L_columnpointer = block_Smatrix->L_sptrsv_task_columnpointer;
        int_t *L_rowindex = block_Smatrix->L_sptrsv_task_rowindex;
        pangulu_heap *sptrsv_heap = block_Smatrix->sptrsv_heap;

        for (int_t k = L_columnpointer[coloffset]; k < L_columnpointer[coloffset + 1]; k++)
        {
            int_t row = L_rowindex[k];
            pangulu_heap_insert(sptrsv_heap, row, col, 0, (row == col) ? 1 : 0, col);
        }
    }
}

void L_pangulu_sptrsv_work(compare_struct *flag,
                           pangulu_block_common *block_common,
                           pangulu_block_Smatrix *block_Smatrix,
                           pangulu_vector *save_vector)
{
    int_t kernel_id = flag->kernel_id;
    int_t row = flag->row;
    int_t col = flag->col;
    int_32t P = block_common->P;
    int_32t Q = block_common->Q;
    int_32t block_length = block_common->block_length;
    int_32t NB = block_common->NB;
    int_32t rank = block_common->rank;
    pangulu_vector *row_vector = block_Smatrix->Big_row_vector[row / P];
    pangulu_vector *col_vector = block_Smatrix->Big_col_vector[col / Q];

    pangulu_zero_pangulu_vector(save_vector);
    if (kernel_id == 0)
    {
        int_t mapper_index = block_Smatrix->mapper_Big_pangulu_Smatrix[block_length * row + col];
        pangulu_Smatrix *A = &block_Smatrix->Big_pangulu_Smatrix_value[mapper_index];

        pangulu_spmv(A, col_vector, save_vector, 1);
        pangulu_vector_add(row_vector, save_vector);

        int_t *L_row_task_nnz = block_Smatrix->L_row_task_nnz;
        int_t rowindex = row / P;
        L_row_task_nnz[rowindex]--;
        if (L_row_task_nnz[rowindex] == 0)
        {
            int_32t diagonal_rank = row % P * Q + row % Q;
            int_32t diagonal_task = row;
            pangulu_isend_pangulu_vector_value(row_vector,
                                               diagonal_rank, diagonal_task, NB);
        }
        return;
    }
    else if (kernel_id == 1)
    {
        int_t diagonal_index = block_Smatrix->mapper_diagonal[row];
        pangulu_Smatrix *L = block_Smatrix->diagonal_Smatrix_L[diagonal_index];
        pangulu_sptrsv(L, save_vector, col_vector, 1, 0);
        int_32t diagonal_task = row;
        pangulu_vector_copy(col_vector, save_vector);
        int_t *L_send_flag = block_Smatrix->L_send_flag + (col / Q) * (P - 1);
        for (int_t k = 1; k < P; k++)
        {
            if (L_send_flag[k - 1] == 1)
            {
                int32_t diagonal_rank = (rank + k * Q) % (P * Q);
                pangulu_isend_pangulu_vector_value(col_vector,
                                                   diagonal_rank, diagonal_task, NB);
            }
        }
        return;
    }
    else
    {
        printf(PANGULU_E_WORK_ERR);
        exit(1);
    }
    return;
}

void pangulu_sptrsv_L(pangulu_block_common *block_common,
                      pangulu_block_Smatrix *block_Smatrix)
{

    int_t *col_task_nnz = block_Smatrix->L_col_task_nnz;
    int32_t rank_col_length = block_common->rank_col_length;
    int_t reiceive_num = 0;
    pangulu_heap *sptrsv_heap = block_Smatrix->sptrsv_heap;
    pangulu_zero_pangulu_heap(sptrsv_heap);
    compare_struct *compare_queue = sptrsv_heap->comapre_queue;

    int_t *diagonal_flag = block_Smatrix->diagonal_flag;
    int_t *L_columnpointer = block_Smatrix->L_sptrsv_task_columnpointer;
    int_t *L_rowindex = block_Smatrix->L_sptrsv_task_rowindex;
    int_32t rank = block_common->rank;
    int_32t Q = block_common->Q;

    for (int_t i = 0; i < block_common->rank_row_length; i++)
    {
        pangulu_zero_pangulu_vector(block_Smatrix->Big_row_vector[i]);
    }

    pangulu_vector *save_vector = block_Smatrix->save_vector;

    for (int_32t i = 0, j = rank % Q; i < rank_col_length; i++, j += Q)
    {
        if (col_task_nnz[i] == 0)
        {
            int_t col = j;

            for (int_t k = L_columnpointer[i]; k < L_columnpointer[i + 1]; k++)
            {
                int_t row = L_rowindex[k];
                pangulu_heap_insert(sptrsv_heap, row, col, 0, (row == col) ? 1 : 0, col);
            }
        }
        else
        {
            int_t length = L_columnpointer[i + 1] - L_columnpointer[i];
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
            L_pangulu_sptrsv_receive_message(status, block_common, block_Smatrix);
            reiceive_num--;
        }
        else
        {
            MPI_Status status;
            int_32t flag = pangulu_iprobe_message(&status);
            if (flag == 1)
            {
                L_pangulu_sptrsv_receive_message(status, block_common, block_Smatrix);
                reiceive_num--;
            }
        }
        if (heap_empty(sptrsv_heap) == 0)
        {
            int_t compare_flag = pangulu_heap_delete(sptrsv_heap);
            L_pangulu_sptrsv_work(compare_queue + compare_flag, block_common, block_Smatrix, save_vector);
        }
    }
    while (heap_empty(sptrsv_heap) == 0)
    {
        int_t compare_flag = pangulu_heap_delete(sptrsv_heap);
        L_pangulu_sptrsv_work(compare_queue + compare_flag, block_common, block_Smatrix, save_vector);
    }
}

void U_pangulu_sptrsv_receive_message(MPI_Status status,
                                      pangulu_block_common *block_common,
                                      pangulu_block_Smatrix *block_Smatrix)
{

    int_32t Q = block_common->Q;
    int_32t NB = block_common->NB;
    int_t receive_id = status.MPI_SOURCE;
    int_t tag = status.MPI_TAG;
    int_t col = tag;
    int_t coloffset = col / Q;
    int_t *diagonal_flag = block_Smatrix->diagonal_flag;
    pangulu_vector *col_vector = block_Smatrix->Big_col_vector[coloffset];
    if (diagonal_flag[coloffset] == 0)
    {
        pangulu_recv_pangulu_vector_value(col_vector, receive_id, tag, NB);
    }
    else
    {
        pangulu_vector *save_vector = block_Smatrix->save_vector;
        pangulu_recv_pangulu_vector_value(save_vector, receive_id, tag, NB);
        pangulu_vector_sub(col_vector, save_vector);
    }

    int_t *U_col_task_nnz = block_Smatrix->U_col_task_nnz;
    U_col_task_nnz[coloffset]--;
    if (U_col_task_nnz[coloffset] == 0)
    {
        int_t *U_columnpointer = block_Smatrix->U_sptrsv_task_columnpointer;
        int_t *U_rowindex = block_Smatrix->U_sptrsv_task_rowindex;
        pangulu_heap *sptrsv_heap = block_Smatrix->sptrsv_heap;
        for (int_t k = U_columnpointer[coloffset]; k < U_columnpointer[coloffset + 1]; k++)
        {
            int_t row = U_rowindex[k];
            pangulu_heap_insert(sptrsv_heap, row, col, 0, (row == col) ? 1 : 0, col);
        }
    }
}

void U_pangulu_sptrsv_work(compare_struct *flag,
                           pangulu_block_common *block_common,
                           pangulu_block_Smatrix *block_Smatrix,
                           pangulu_vector *save_vector)
{
    int_t kernel_id = flag->kernel_id;
    int_t row = flag->row;
    int_t col = flag->col;
    int_32t P = block_common->P;
    int_32t Q = block_common->Q;
    int_32t block_length = block_common->block_length;
    int_32t NB = block_common->NB;
    int_32t rank = block_common->rank;

    pangulu_vector *row_vector = block_Smatrix->Big_row_vector[row / P];
    pangulu_vector *col_vector = block_Smatrix->Big_col_vector[col / Q];

    pangulu_zero_pangulu_vector(save_vector);
    if (kernel_id == 0)
    {
        int_t mapper_index = block_Smatrix->mapper_Big_pangulu_Smatrix[block_length * row + col];
        pangulu_Smatrix *A = &block_Smatrix->Big_pangulu_Smatrix_value[mapper_index];

        pangulu_spmv(A, col_vector, save_vector, 1);
        pangulu_vector_add(row_vector, save_vector);
        int_t *U_row_task_nnz = block_Smatrix->U_row_task_nnz;
        int_t rowindex = row / P;
        U_row_task_nnz[rowindex]--;
        if (U_row_task_nnz[rowindex] == 0)
        {
            int_32t diagonal_rank = row % P * Q + row % Q;
            int_32t diagonal_task = row;
            pangulu_isend_pangulu_vector_value(row_vector,
                                               diagonal_rank, diagonal_task, NB);
        }
        return;
    }
    else if (kernel_id == 1)
    {
        int_t diagonal_index = block_Smatrix->mapper_diagonal[row];
        pangulu_Smatrix *U = block_Smatrix->diagonal_Smatrix_U[diagonal_index];

        pangulu_sptrsv(U, save_vector, col_vector, 1, 1);
        int_32t diagonal_task = row;
        pangulu_vector_copy(col_vector, save_vector);
        int_t *U_send_flag = block_Smatrix->U_send_flag + (col / Q) * (P - 1);
        for (int_32t k = 1; k < P; k++)
        {
            if (U_send_flag[k - 1] == 1)
            {
                int32_t diagonal_rank = (rank + (P - k) * Q) % (P * Q);
                pangulu_isend_pangulu_vector_value(col_vector,
                                                   diagonal_rank, diagonal_task, NB);
            }
        }
        return;
    }
    else
    {
        printf(PANGULU_E_WORK_ERR);
        exit(1);
    }
    return;
}

void pangulu_sptrsv_U(pangulu_block_common *block_common,
                      pangulu_block_Smatrix *block_Smatrix)
{
    int_t *col_task_nnz = block_Smatrix->U_col_task_nnz;
    int32_t rank_col_length = block_common->rank_col_length;
    int_t reiceive_num = 0;
    pangulu_heap *sptrsv_heap = block_Smatrix->sptrsv_heap;
    pangulu_zero_pangulu_heap(sptrsv_heap);
    compare_struct *compare_queue = sptrsv_heap->comapre_queue;

    int_t *diagonal_flag = block_Smatrix->diagonal_flag;
    int_t *U_columnpointer = block_Smatrix->U_sptrsv_task_columnpointer;
    int_t *U_rowindex = block_Smatrix->U_sptrsv_task_rowindex;
    int_32t rank = block_common->rank;
    int_32t Q = block_common->Q;

    for (int_t i = 0; i < block_common->rank_row_length; i++)
    {
        pangulu_zero_pangulu_vector(block_Smatrix->Big_row_vector[i]);
    }

    pangulu_vector *save_vector = block_Smatrix->save_vector;

    for (int_32t i = 0, j = rank % Q; i < rank_col_length; i++, j += Q)
    {
        if (col_task_nnz[i] == 0)
        {
            int_t col = j;

            for (int_t k = U_columnpointer[i]; k < U_columnpointer[i + 1]; k++)
            {
                int_t row = U_rowindex[k];
                pangulu_heap_insert(sptrsv_heap, row, col, 0, (row == col) ? 1 : 0, col);
            }
        }
        else
        {
            int_t length = U_columnpointer[i + 1] - U_columnpointer[i];
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
            U_pangulu_sptrsv_receive_message(status, block_common, block_Smatrix);
            reiceive_num--;
        }
        else
        {
            MPI_Status status;
            int_32t flag = pangulu_iprobe_message(&status);
            if (flag == 1)
            {
                U_pangulu_sptrsv_receive_message(status, block_common, block_Smatrix);
                reiceive_num--;
            }
        }
        if (heap_empty(sptrsv_heap) == 0)
        {
            int_t compare_flag = pangulu_heap_delete(sptrsv_heap);
            U_pangulu_sptrsv_work(compare_queue + compare_flag, block_common, block_Smatrix, save_vector);
        }
    }
    while (heap_empty(sptrsv_heap) == 0)
    {
        int_t compare_flag = pangulu_heap_delete(sptrsv_heap);
        U_pangulu_sptrsv_work(compare_queue + compare_flag, block_common, block_Smatrix, save_vector);
    }
}

void pangulu_sptrsv_vector_gather(pangulu_block_common *block_common,
                                  pangulu_block_Smatrix *block_Smatrix,
                                  pangulu_vector *vector)
{

    int_t N = block_common->N;
    int_32t vector_number = 1;
    int_32t rank = block_common->rank;
    int_32t P = block_common->P;
    int_32t Q = block_common->Q;
    int_32t NB = block_common->NB;
    int_32t block_length = block_common->block_length;
    int_32t rank_col_length = block_common->rank_col_length;
    int_t offset_row_init = rank / Q;
    int_t offset_col_init = rank % Q;

    calculate_type *save_vector;

    int_t *diagonal_num = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (P * Q + 1));
    if (rank == 0)
    {
        for (int_t i = 0; i < P * Q + 1; i++)
        {
            diagonal_num[i] = 0;
        }
        for (int_t i = 0; i < block_length; i++)
        {
            int_t diagonal_rank = calculate_diagonal_rank(i, P, Q);
            diagonal_num[diagonal_rank + 1] += NB;
        }
        for (int_t i = 0; i < P * Q; i++)
        {
            diagonal_num[i + 1] += diagonal_num[i];
        }
    }
    pangulu_Bcast_vector(diagonal_num, P * Q + 1, 0);
    if (rank == 0)
    {
        save_vector = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * (diagonal_num[P * Q]));
    }
    else
    {
        save_vector = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * (diagonal_num[rank + 1] - diagonal_num[rank]));
    }
    int_t *diagonal_flag = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * rank_col_length);
    for (int_t i = 0; i < rank_col_length; i++)
    {
        diagonal_flag[i] = 0;
    }
    for (int_t i = offset_col_init, now_diagonal_num = 0, k = 0; i < block_length; i += Q, k++)
    {
        int_t offset_row = calculate_offset(offset_row_init, i, P);
        pangulu_vector *new_vector = block_Smatrix->Big_col_vector[k];
        if (offset_row == 0)
        {
            diagonal_flag[k] = 1;
            for (int_t vector_index = 0; vector_index < vector_number; vector_index++)
            {
                calculate_type *now_value = vector_index * NB + new_vector->value;
                for (int_t j = 0; j < NB; j++)
                {
                    save_vector[now_diagonal_num + j] = now_value[j];
                }
            }
            now_diagonal_num += NB;
        }
    }

    if (rank == 0)
    {
        for (int_t i = 1; i < P * Q; i++)
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
        int_t *save_diagonal_num = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (P * Q + 1));
        for (int_t i = 0; i < P * Q + 1; i++)
        {
            save_diagonal_num[i] = diagonal_num[i];
        }
        for (int_t i = 0; i < block_length; i++)
        {
            int_t now_length = ((i + 1) * NB) > N ? (N - i * NB) : NB;
            int_t now_add_rank = calculate_diagonal_rank(i, P, Q);
            calculate_type *begin_diagnoal = save_vector + save_diagonal_num[now_add_rank];
            calculate_type *vector_begin = vector->value + i * NB;
            for (int_t j = 0; j < now_length; j++)
            {
                vector_begin[j] = begin_diagnoal[j];
            }
            save_diagonal_num[now_add_rank] += NB;
        }
        pangulu_free(__FILE__, __LINE__, save_diagonal_num);
    }

    pangulu_free(__FILE__, __LINE__, save_vector);
    pangulu_free(__FILE__, __LINE__, diagonal_num);
}
#endif