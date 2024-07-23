#ifndef PANGULU_NUMERICAL_H
#define PANGULU_NUMERICAL_H

#include "pangulu_common.h"
#include "pangulu_utils.h"
#include "pangulu_kernel_interface.h"
#include "pangulu_heap.h"
#include "pangulu_time.h"

#ifdef OVERLAP
#include "pangulu_thread.h"
#endif

void pangulu_solve_A_to_LU(int_t level, int_t row, int_t col,
                           pangulu_block_common *block_common,
                           pangulu_block_Smatrix *block_Smatrix,
                           pangulu_Smatrix *calculate_L,
                           pangulu_Smatrix *calculate_U)
{
    int_32t block_length = block_common->block_length;
    int_32t sum_rank_size = block_common->sum_rank_size;
    int_32t rank = block_common->rank;
    int_32t P = block_common->P;
    int_32t Q = block_common->Q;
    int_t *level_task_rank_id = block_Smatrix->level_task_rank_id;
    int_t *grid_process_id = block_Smatrix->grid_process_id;

    int_t mapper_index = block_Smatrix->mapper_Big_pangulu_Smatrix[block_length * row + col];
    pangulu_Smatrix *A = &block_Smatrix->Big_pangulu_Smatrix_value[mapper_index];
    int_t diagonal_index = block_Smatrix->mapper_diagonal[level];
    pangulu_Smatrix *L = block_Smatrix->diagonal_Smatrix_L[diagonal_index];
    pangulu_Smatrix *U = block_Smatrix->diagonal_Smatrix_U[diagonal_index];
    pangulu_heap *heap = block_Smatrix->heap;

    pangulu_getrf_interface(A, L, U, calculate_L, calculate_U);

    pangulu_transport_pangulu_Smatrix_CSC_to_CSR(U);

    int_t *task_flag_id = block_Smatrix->task_flag_id;

#ifdef OVERLAP
    pthread_mutex_t *heap_mutex = (heap->heap_bsem)->mutex;
    pthread_mutex_lock(heap_mutex);
#endif
    int_t begin_L = block_Smatrix->L_pangulu_Smatrix_columnpointer[col] + 1;
    int_t length_L = block_Smatrix->L_pangulu_Smatrix_columnpointer[col + 1] - begin_L;
    int_t *rowindex_L = block_Smatrix->L_pangulu_Smatrix_rowindex + begin_L;

    int_t begin_U = block_Smatrix->U_pangulu_Smatrix_rowpointer[row] + 1;
    int_t length_U = block_Smatrix->U_pangulu_Smatrix_rowpointer[row + 1] - begin_U;
    int_t *columnindex_U = block_Smatrix->U_pangulu_Smatrix_columnindex + begin_U;

    for (int_t i = 0; i < length_L; i++)
    {
        int_t now_row = rowindex_L[i];
        int_t now_col = col;
        int_t now_rank = grid_process_id[(now_row % P) * Q + (now_col % Q)];
        int_t attain_id = level_task_rank_id[level * (P * Q) + now_rank];

        if (attain_id == rank)
        {
            task_flag_id[block_length * rowindex_L[i] + col]--;
            if (task_flag_id[block_length * rowindex_L[i] + col] == 0)
            {
                pangulu_heap_insert(heap, rowindex_L[i], col, level, 2, rowindex_L[i]);
            }
        }
    }

    for (int_t i = 0; i < length_U; i++)
    {
        int_t now_row = row;
        int_t now_col = columnindex_U[i];
        int_t now_rank = grid_process_id[(now_row % P) * Q + (now_col % Q)];
        int_t attain_id = level_task_rank_id[level * (P * Q) + now_rank];

        if (attain_id == rank)
        {
            task_flag_id[block_length * row + columnindex_U[i]]--;
            if (task_flag_id[block_length * row + columnindex_U[i]] == 0)
            {
                pangulu_heap_insert(heap, row, columnindex_U[i], level, 3, columnindex_U[i]);
            }
        }
    }

#ifdef OVERLAP
    pthread_mutex_unlock(heap_mutex);
#endif

    int_t NB = block_common->NB;

    int_t signal = level * block_length + level;

    int_t *mapper_mpi = block_Smatrix->mapper_mpi;
    int_t index_signal = mapper_mpi[signal];

    int_t *save_send_rank_flag = block_Smatrix->save_send_rank_flag;

    int_t send_offset_row = level % P;
    int_t send_offset_col = level % Q;

    int_t max_PQ = block_common->max_PQ;
    int_t *now_send_diagonal_L = (block_Smatrix->send_diagonal_flag_L) + diagonal_index * max_PQ;
    pangulu_init_vector_int(save_send_rank_flag, sum_rank_size);
    save_send_rank_flag[rank] = 1;

    for (int_t i = 0; i < ((Q - 1) < (block_length - level - 1) ? (Q - 1) : (block_length - level - 1)); i++)
    {
        send_offset_col++;
        send_offset_col = send_offset_col % Q;

        int_t now_rank = grid_process_id[send_offset_row * Q + send_offset_col];
        int_t send_id = level_task_rank_id[level * (P * Q) + now_rank];

        if (save_send_rank_flag[send_id] == 0 && now_send_diagonal_L[i + 1] == 1)
        {
            save_send_rank_flag[send_id] = 1;
            pangulu_isend_whole_pangulu_Smatrix_CSC(L, send_id, index_signal, NB);
        }
    }

    send_offset_col = level % Q;

    int_t *now_send_diagonal_U = (block_Smatrix->send_diagonal_flag_U) + diagonal_index * max_PQ;
    pangulu_init_vector_int(save_send_rank_flag, sum_rank_size);
    save_send_rank_flag[rank] = 1;

    signal = block_length * block_length + level;
    index_signal = mapper_mpi[signal];

    for (int_t i = 0; i < ((P - 1) < (block_length - level - 1) ? (P - 1) : (block_length - level - 1)); i++)
    {
        send_offset_row++;
        send_offset_row = send_offset_row % P;
        int_t now_rank = grid_process_id[send_offset_row * Q + send_offset_col];
        int_t send_id = level_task_rank_id[level * (P * Q) + now_rank];
        if (save_send_rank_flag[send_id] == 0 && now_send_diagonal_U[i + 1] == 1)
        {
            save_send_rank_flag[send_id] = 1;
            pangulu_isend_whole_pangulu_Smatrix_CSR(U, send_id, index_signal, NB);
        }
    }

#ifdef ADD_GPU_MEMORY
    block_Smatrix->flag_dignon_L[diagonal_index] = 1;
    pangulu_gessm_preprocess(L);

#endif
}
void pangulu_solve_XU_A(int_t level, int_t now_level, int_t row, int_t col,
                        pangulu_block_common *block_common,
                        pangulu_block_Smatrix *block_Smatrix,
                        pangulu_Smatrix *calculate_U,
                        pangulu_Smatrix *calculate_X)
{
    int_t block_length = block_common->block_length;
    int_t P = block_common->P;
    int_t Q = block_common->Q;
    int_t NB = block_common->NB;
    int_t sum_rank_size = block_common->sum_rank_size;
    int_t *level_task_rank_id = block_Smatrix->level_task_rank_id;
    int_t *grid_process_id = block_Smatrix->grid_process_id;
    int_t rank = block_common->rank;
    int_t *mapper_A = block_Smatrix->mapper_Big_pangulu_Smatrix;
    pangulu_heap *heap = block_Smatrix->heap;

    int_t mapper_index = block_Smatrix->mapper_Big_pangulu_Smatrix[block_length * row + col];
    pangulu_Smatrix *A = &block_Smatrix->Big_pangulu_Smatrix_value[mapper_index];

    int_t diagonal_index = block_Smatrix->mapper_diagonal[col];
    pangulu_Smatrix *U = block_Smatrix->diagonal_Smatrix_U[diagonal_index];

    int_t mapper_index_X = block_Smatrix->mapper_LU[block_length * row + col];
    pangulu_Smatrix *save_X = block_Smatrix->L_pangulu_Smatrix_value[mapper_index_X];

    save_X->nnz = A->nnz;
    save_X->rowindex = A->rowindex;
    save_X->value_CSC = A->value_CSC;

    if (block_Smatrix->flag_dignon_U[diagonal_index] == 0)
    {
#ifdef GPU_OPEN
        pangulu_Smatrix_CUDA_memcpy_complete_CSR(U, U);
        pangulu_tstrf_preprocess(U);
#endif
        block_Smatrix->flag_dignon_U[diagonal_index] = 1;
    }

    pangulu_tstrf_interface(A, save_X, U, calculate_X, calculate_U);

#ifdef OVERLAP
    pthread_mutex_t *heap_mutex = (heap->heap_bsem)->mutex;
    pthread_mutex_lock(heap_mutex);
#endif

    int_t offset_col = block_Smatrix->level_index_reverse[col];
    int_t offset_L = (block_Smatrix->L_Smatrix_nzz) * (offset_col - now_level);
    int_t offset_U = (block_Smatrix->U_Smatrix_nzz) * (offset_col - now_level);
    block_Smatrix->save_now_level_L[block_Smatrix->now_level_L_length[offset_col - now_level] + offset_L] = row;

    block_Smatrix->now_level_L_length[offset_col - now_level]++;
    int_t now_add_length = block_Smatrix->now_level_U_length[offset_col - now_level];
    int_t *save_now_level_U = (block_Smatrix->save_now_level_U) + offset_U;

    for (int_t i = 0; i < now_add_length; i++)
    {
        int_t now_row = row;
        int_t now_col = save_now_level_U[i];
        int_t now_rank = grid_process_id[(now_row % P) * Q + (now_col % Q)];
        int_t attain_id = level_task_rank_id[level * (P * Q) + now_rank];
        if ((attain_id == rank) && (mapper_A[now_row * block_length + now_col] != -1))
        {
            pangulu_heap_insert(heap, row, save_now_level_U[i], level, 4, PANGULU_MAX(row, save_now_level_U[i]));
        }
    }

#ifdef OVERLAP
    pthread_mutex_unlock(heap_mutex);
#endif

    if (block_common->rank == -3)
    {
        pangulu_display_pangulu_Smatrix_CSC(save_X);
    }
    int_t signal = block_length * row + col;

    int_t *mapper_mpi = block_Smatrix->mapper_mpi;
    int_t index_signal = mapper_mpi[signal];

    int_t send_offset_row = row % P;
    int_t send_offset_col = level % Q;

    int_t max_PQ = block_common->max_PQ;
    int_t *now_send_flag = (block_Smatrix->send_flag) + max_PQ * mapper_index;

    int_t *save_send_rank_flag = block_Smatrix->save_send_rank_flag;
    pangulu_init_vector_int(save_send_rank_flag, sum_rank_size);
    save_send_rank_flag[rank] = 1;

    for (int_t i = 0; i < ((Q - 1) < ((block_length - level - 1)) ? (Q - 1) : (block_length - level - 1)); i++)
    {
        send_offset_col++;
        send_offset_col = send_offset_col % Q;

        int_t now_rank = grid_process_id[send_offset_row * Q + send_offset_col];
        int_t send_id = level_task_rank_id[level * (P * Q) + now_rank];
        if (save_send_rank_flag[send_id] == 0 && now_send_flag[i + 1] == 1)
        {
            save_send_rank_flag[send_id] = 1;
            pangulu_isend_whole_pangulu_Smatrix_CSC(A, send_id, index_signal, NB);
        }
    }
#ifdef ADD_GPU_MEMORY
    pangulu_pangulu_Smatrix_memcpy_columnpointer_CSC(save_X, A);
    pangulu_Smatrix_CUDA_memcpy_complete_CSC(save_X, A);
    block_Smatrix->flag_save_L[mapper_index_X] = 1;
#endif
}

void pangulu_solve_LX_A(int_t level, int_t now_level, int_t row, int_t col,
                        pangulu_block_common *block_common,
                        pangulu_block_Smatrix *block_Smatrix,
                        pangulu_Smatrix *calculate_L,
                        pangulu_Smatrix *calculate_X)
{
    int_t P = block_common->P;
    int_t block_length = block_common->block_length;
    int_t Q = block_common->Q;
    int_t NB = block_common->NB;
    int_t sum_rank_size = block_common->sum_rank_size;
    int_t *level_task_rank_id = block_Smatrix->level_task_rank_id;
    int_t *grid_process_id = block_Smatrix->grid_process_id;
    int_t rank = block_common->rank;
    int_t *mapper_A = block_Smatrix->mapper_Big_pangulu_Smatrix;
    pangulu_heap *heap = block_Smatrix->heap;

    int_t mapper_index = block_Smatrix->mapper_Big_pangulu_Smatrix[block_length * row + col];
    pangulu_Smatrix *A = &block_Smatrix->Big_pangulu_Smatrix_value[mapper_index];

    int_t diagonal_index = block_Smatrix->mapper_diagonal[row];
    pangulu_Smatrix *L = block_Smatrix->diagonal_Smatrix_L[diagonal_index];

    int_t mapper_index_X = block_Smatrix->mapper_LU[block_length * row + col];
    pangulu_Smatrix *save_X = block_Smatrix->U_pangulu_Smatrix_value[mapper_index_X];

    save_X->nnz = A->nnz;
    save_X->rowindex = A->rowindex;
    save_X->value_CSC = A->value_CSC;

    if (block_Smatrix->flag_dignon_L[diagonal_index] == 0)
    {
#ifdef GPU_OPEN
        pangulu_Smatrix_CUDA_memcpy_complete_CSC(L, L);
        pangulu_gessm_preprocess(L);
#endif
        block_Smatrix->flag_dignon_L[diagonal_index] = 1;
    }

    pangulu_gessm_interface(A, save_X, L, calculate_X, calculate_L);

#ifdef OVERLAP
    pthread_mutex_t *heap_mutex = (heap->heap_bsem)->mutex;
    pthread_mutex_lock(heap_mutex);
#endif

    int_t offset_row = block_Smatrix->level_index_reverse[row];
    int_t offset_U = (block_Smatrix->U_Smatrix_nzz) * (offset_row - now_level);
    int_t offset_L = (block_Smatrix->L_Smatrix_nzz) * (offset_row - now_level);
    block_Smatrix->save_now_level_U[block_Smatrix->now_level_U_length[offset_row - now_level] + offset_U] = col;
    block_Smatrix->now_level_U_length[offset_row - now_level]++;
    int_t now_add_length = block_Smatrix->now_level_L_length[offset_row - now_level];
    int_t *save_now_level_L = (block_Smatrix->save_now_level_L) + offset_L;

    for (int_t i = 0; i < now_add_length; i++)
    {
        int_t now_row = save_now_level_L[i];
        int_t now_col = col;
        int_t now_rank = grid_process_id[(now_row % P) * Q + (now_col % Q)];
        int_t attain_id = level_task_rank_id[level * (P * Q) + now_rank];
        if ((attain_id == rank) && (mapper_A[now_row * block_length + now_col] != -1))
        {
            pangulu_heap_insert(heap, save_now_level_L[i], col, level, 4, PANGULU_MAX(save_now_level_L[i], col));
        }
    }
#ifdef OVERLAP
    pthread_mutex_unlock(heap_mutex);
#endif

    int_t signal = block_length * row + col;

    int_t *mapper_mpi = block_Smatrix->mapper_mpi;
    int_t index_signal = mapper_mpi[signal];

    int_t send_offset_row = level % P;
    int_t send_offset_col = col % Q;

    int_t max_PQ = block_common->max_PQ;
    int_t *now_send_flag = (block_Smatrix->send_flag) + max_PQ * mapper_index;

    int_t *save_send_rank_flag = block_Smatrix->save_send_rank_flag;
    pangulu_init_vector_int(save_send_rank_flag, sum_rank_size);
    save_send_rank_flag[rank] = 1;

    for (int_t i = 0; i < ((P - 1) < (block_length - level - 1) ? (P - 1) : (block_length - level - 1)); i++)
    {
        send_offset_row++;
        send_offset_row = send_offset_row % P;
        int_t now_rank = grid_process_id[send_offset_row * Q + send_offset_col];
        int_t send_id = level_task_rank_id[level * (P * Q) + now_rank];

        if (save_send_rank_flag[send_id] == 0 && now_send_flag[i + 1] == 1)
        {
            save_send_rank_flag[send_id] = 1;
            pangulu_isend_whole_pangulu_Smatrix_CSC(A, send_id, index_signal, NB);
        }
    }

#ifdef ADD_GPU_MEMORY
    pangulu_pangulu_Smatrix_memcpy_columnpointer_CSC(save_X, A);
    pangulu_Smatrix_CUDA_memcpy_complete_CSC(save_X, A);
    block_Smatrix->flag_save_U[mapper_index_X] = 1;
#endif

    return;
}

void pangulu_solve_A_LU(int_t level, int_t row, int_t col,
                        pangulu_block_common *block_common,
                        pangulu_block_Smatrix *block_Smatrix,
                        pangulu_Smatrix *calculate_L,
                        pangulu_Smatrix *calculate_U)
{
    int_t block_length = block_common->block_length;

    int_t mapper_index = block_Smatrix->mapper_Big_pangulu_Smatrix[block_length * row + col];

    if (mapper_index == -1)
    {
        return;
    }

    pangulu_Smatrix *A = &block_Smatrix->Big_pangulu_Smatrix_value[mapper_index];
    int_t mapper_index_L = block_Smatrix->mapper_LU[block_length * row + level];
    int_t mapper_index_U = block_Smatrix->mapper_LU[block_length * level + col];
    pangulu_Smatrix *L = block_Smatrix->L_pangulu_Smatrix_value[mapper_index_L];
    pangulu_Smatrix *U = block_Smatrix->U_pangulu_Smatrix_value[mapper_index_U];

    if (block_common->rank == -1)
    {
#ifdef GPU_OPEN
        pangulu_Smatrix_CUDA_memcpy_value_CSC(A, A);
#endif
        pangulu_display_pangulu_Smatrix_CSC(A);
    }

    if (block_Smatrix->flag_save_L[mapper_index_L] == 0)
    {
#ifdef GPU_OPEN
        pangulu_Smatrix_CUDA_memcpy_complete_CSC(L, L);
#endif
        block_Smatrix->flag_save_L[mapper_index_L] = 1;
    }

    if (block_Smatrix->flag_save_U[mapper_index_U] == 0)
    {
#ifdef GPU_OPEN
        pangulu_Smatrix_CUDA_memcpy_complete_CSC(U, U);
#endif
        block_Smatrix->flag_save_U[mapper_index_U] = 1;
    }

    pangulu_ssssm_interface(A, L, U, calculate_L, calculate_U);

    if (block_common->rank == -1)
    {
        pangulu_display_pangulu_Smatrix_CSC(L);
        pangulu_display_pangulu_Smatrix_CSC(U);
#ifdef GPU_OPEN
        pangulu_Smatrix_CUDA_memcpy_value_CSC(A, A);
#endif
        pangulu_display_pangulu_Smatrix_CSC(A);
    }

    int_t *task_flag_id = block_Smatrix->task_flag_id;
    pangulu_heap *heap = block_Smatrix->heap;

#ifdef OVERLAP
    pthread_mutex_t *heap_mutex = (heap->heap_bsem)->mutex;
    pthread_mutex_lock(heap_mutex);
#endif

    task_flag_id[block_length * row + col]--;

    if (task_flag_id[block_length * row + col] == 0)
    {

        int_t *level_task_rank_id = block_Smatrix->level_task_rank_id;
        int_t *grid_process_id = block_Smatrix->grid_process_id;

        int_t P = block_common->P;
        int_t Q = block_common->Q;
        int_t now_rank = grid_process_id[(row % P) * Q + (col % Q)];
        int_t send_id = level_task_rank_id[PANGULU_MIN(row, col) * (P * Q) + now_rank];

        int_t rank = block_common->rank;

        if (send_id == rank)
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

            int_t NB = block_common->NB;
#ifdef GPU_OPEN
            pangulu_Smatrix_CUDA_memcpy_value_CSC(A, A);
#endif
            int_t signal = block_length * row + col;

            int_t *mapper_mpi = block_Smatrix->mapper_mpi;
            int_t index_signal = mapper_mpi[signal];
            pangulu_isend_pangulu_Smatrix_value_CSC_in_signal(A, send_id, index_signal, NB);
        }
    }
    else
    {
#ifdef OVERLAP
        pthread_mutex_unlock(heap_mutex);
#endif
    }
}

void pangulu_add_A_to_A_old(int_t level, int_t row, int_t col,
                            pangulu_block_common *block_common,
                            pangulu_block_Smatrix *block_Smatrix,
                            pangulu_Smatrix *calculate_X)
{
    printf(PANGULU_I_A_A_OLD);
    MPI_Abort(MPI_COMM_WORLD, 0);
    
    int_t *task_flag_id = block_Smatrix->task_flag_id;
    int_t block_length = block_common->block_length;

    int_t mapper_index = block_Smatrix->mapper_Big_pangulu_Smatrix[row * block_length + col];
    pangulu_Smatrix *receive_A = block_Smatrix->Big_pangulu_Smatrix_copy_value[mapper_index];
    pangulu_Smatrix *A = &block_Smatrix->Big_pangulu_Smatrix_value[mapper_index];

#ifdef GPU_OPEN
    pangulu_Smatrix_CUDA_memcpy_to_device_value_CSC(calculate_X, receive_A);
    pangulu_addmatrix_interface(A, calculate_X);
#else
    pangulu_addmatrix_interface_CPU(A, receive_A);
#endif
    pangulu_memcpy_zero_pangulu_Smatrix_CSC_value(receive_A);

    pangulu_heap *heap = block_Smatrix->heap;

#ifdef OVERLAP
    pthread_mutex_t *heap_mutex = (heap->heap_bsem)->mutex;
    pthread_mutex_lock(heap_mutex);
#endif

    task_flag_id[block_length * row + col]--;
    if (task_flag_id[block_length * row + col] == 0)
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
    }

#ifdef OVERLAP
    pthread_mutex_unlock(heap_mutex);
#endif
}

void pangulu_numerical_work(compare_struct *flag,
                            pangulu_block_common *block_common,
                            pangulu_block_Smatrix *block_Smatrix,
                            pangulu_Smatrix *calculate_L,
                            pangulu_Smatrix *calculate_U,
                            pangulu_Smatrix *calculate_X,
                            int_t now_level)
{
    int_t kernel_id = flag->kernel_id;
    int_t row = flag->row;
    int_t col = flag->col;
    int_t level = flag->task_level;

    if (block_Smatrix->mapper_Big_pangulu_Smatrix[row * (block_common->block_length) + col] == -1)
    {
        printf(PANGULU_E_ERR_IN_RRCL);
        exit(0);
        return;
    }

    if (kernel_id == 1)
    {
        pangulu_solve_A_to_LU(level, row, col,
                              block_common,
                              block_Smatrix,
                              calculate_L,
                              calculate_U);
    }
    else if (kernel_id == 2)
    {
        pangulu_solve_XU_A(level, now_level, row, col,
                           block_common,
                           block_Smatrix,
                           calculate_U,
                           calculate_X);
    }
    else if (kernel_id == 3)
    {
        pangulu_solve_LX_A(level, now_level, row, col,
                           block_common,
                           block_Smatrix,
                           calculate_L,
                           calculate_X);
    }
    else if (kernel_id == 4)
    {
        pangulu_solve_A_LU(level, row, col,
                           block_common,
                           block_Smatrix,
                           calculate_L,
                           calculate_U);
    }
    else if (kernel_id == 5)
    {
        pangulu_add_A_to_A_old(level, row, col,
                               block_common,
                               block_Smatrix,
                               calculate_X);
    }
    else
    {
        printf(PANGULU_E_K_ID);
        exit(0);
    }
}

void pangulu_numerical_receive_message(MPI_Status status,
                                       int_t now_level,
                                       pangulu_block_common *block_common,
                                       pangulu_block_Smatrix *block_Smatrix)
{

    int_t block_length = block_common->block_length;
    int_t P = block_common->P;
    int_t Q = block_common->Q;
    int_t rank = block_common->rank;
    int_t NB = block_common->NB;
    int_t *level_task_rank_id = block_Smatrix->level_task_rank_id;
    int_t *grid_process_id = block_Smatrix->grid_process_id;

    int_t *task_flag_id = block_Smatrix->task_flag_id;
    pangulu_heap *heap = block_Smatrix->heap;

    int_t tag = (status.MPI_TAG);

    int_t *mpi_level_num = block_Smatrix->mpi_level_num;
    int_t *mapper_mpi_reverse = block_Smatrix->mapper_mpi_reverse;
    int_t index_tag = mapper_mpi_reverse[tag + mpi_level_num[now_level / block_common->every_level_length]];

    int_t receive_id = status.MPI_SOURCE;

    int_t row;
    int_t col;

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

    int_t *real_matrix_flag = block_Smatrix->real_matrix_flag;
    int_t *mapper_A = block_Smatrix->mapper_Big_pangulu_Smatrix;

#ifdef OVERLAP
    pthread_mutex_t *heap_mutex = (heap->heap_bsem)->mutex;
#endif

    if ((mapper_A[row * block_length + col] != -1) && (real_matrix_flag[mapper_A[row * block_length + col]] == 1))
    {
        int_t level = PANGULU_MIN(row, col);

        pangulu_Smatrix *receive_A = block_Smatrix->Big_pangulu_Smatrix_copy_value[mapper_A[row * block_length + col]];

        pangulu_recv_pangulu_Smatrix_value_CSR_in_signal(receive_A, receive_id, tag, NB);

        // do the kernel
        pangulu_add_pangulu_Smatrix_CSR_to_CSC(receive_A);
        int_t flag = --(block_Smatrix->sum_flag_block_num[row * block_length + col]);

        if (flag == 1)
        {
            // add kernel 5
#ifdef OVERLAP
            pthread_mutex_lock(heap_mutex);
#endif
            pangulu_heap_insert(heap, row, col, level, 5, PANGULU_MAX(row, col));

#ifdef OVERLAP
            pthread_mutex_unlock(heap_mutex);
#endif
        }
        return;
    }

    if (row == col)
    {

        int_t level = row;
        int_t diagonal_index = block_Smatrix->mapper_diagonal[level];
        if (index_tag < (block_length * block_length))
        {
            pangulu_Smatrix *L = block_Smatrix->diagonal_Smatrix_L[diagonal_index];
            int_t nnz = block_Smatrix->block_Smatrix_non_zero_vector_L[level];
            pangulu_recv_whole_pangulu_Smatrix_CSC(L, receive_id, tag, nnz, NB);
#ifdef ADD_GPU_MEMORY
#endif

            int_t *U_rowpointer = block_Smatrix->U_pangulu_Smatrix_rowpointer;
            int_t *U_columnindex = block_Smatrix->U_pangulu_Smatrix_columnindex;
#ifdef OVERLAP
            pthread_mutex_lock(heap_mutex);
#endif
            for (int_t i = U_rowpointer[level]; i < U_rowpointer[level + 1]; i++)
            {
                int_t add_col = U_columnindex[i];

                int_t now_row = level;
                int_t now_col = add_col;
                int_t now_rank = grid_process_id[(now_row % P) * Q + (now_col % Q)];
                int_t attain_id = level_task_rank_id[level * (P * Q) + now_rank];

                if (attain_id == rank)
                {
                    task_flag_id[level * block_length + add_col]--;
                    if (task_flag_id[block_length * row + add_col] == 0)
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
            pangulu_Smatrix *U = block_Smatrix->diagonal_Smatrix_U[diagonal_index];
            int_t nnz = block_Smatrix->block_Smatrix_non_zero_vector_U[level];
            pangulu_recv_whole_pangulu_Smatrix_CSR(U, receive_id, tag, nnz, NB);

#ifdef ADD_GPU_MEMORY
#endif

            int_t *L_columnpointer = block_Smatrix->L_pangulu_Smatrix_columnpointer;
            int_t *L_rowindex = block_Smatrix->L_pangulu_Smatrix_rowindex;

#ifdef OVERLAP
            pthread_mutex_lock(heap_mutex);
#endif

            for (int_t i = L_columnpointer[level]; i < L_columnpointer[level + 1]; i++)
            {
                int_t add_row = L_rowindex[i];

                int_t now_row = add_row;
                int_t now_col = level;
                int_t now_rank = grid_process_id[(now_row % P) * Q + (now_col % Q)];
                int_t attain_id = level_task_rank_id[level * (P * Q) + now_rank];

                if (attain_id == rank)
                {
                    task_flag_id[add_row * block_length + level]--;
                    if (task_flag_id[block_length * add_row + col] == 0)
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
        int_t level = col;

        int_t mapper_index_L = block_Smatrix->mapper_LU[block_length * row + col];
        pangulu_Smatrix *save_L = block_Smatrix->L_pangulu_Smatrix_value[mapper_index_L];
        int_t nnz = block_Smatrix->block_Smatrix_nnzA_num[block_length * row + col];

        pangulu_recv_whole_pangulu_Smatrix_CSC(save_L, receive_id, tag, nnz, NB);
#ifdef ADD_GPU_MEMORY
#endif

#ifdef OVERLAP
        pthread_mutex_lock(heap_mutex);
#endif
        int_t offset_col = block_Smatrix->level_index_reverse[col];
        int_t offset_L = (block_Smatrix->L_Smatrix_nzz) * (offset_col - now_level);
        int_t offset_U = (block_Smatrix->U_Smatrix_nzz) * (offset_col - now_level);
        block_Smatrix->save_now_level_L[block_Smatrix->now_level_L_length[offset_col - now_level] + offset_L] = row;

        block_Smatrix->now_level_L_length[offset_col - now_level]++;
        int_t now_add_length = block_Smatrix->now_level_U_length[offset_col - now_level];
        int_t *save_now_level_U = (block_Smatrix->save_now_level_U) + offset_U;

        for (int_t i = 0; i < now_add_length; i++)
        {

            int_t now_row = row;
            int_t now_col = save_now_level_U[i];
            int_t now_rank = grid_process_id[(now_row % P) * Q + (now_col % Q)];
            int_t attain_id = level_task_rank_id[level * (P * Q) + now_rank];
            if ((attain_id == rank) && (mapper_A[now_row * block_length + now_col] != -1))
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
        int_t level = row;

        int_t mapper_index_U = block_Smatrix->mapper_LU[block_length * row + col];
        pangulu_Smatrix *save_U = block_Smatrix->U_pangulu_Smatrix_value[mapper_index_U];
        int_t nnz = block_Smatrix->block_Smatrix_nnzA_num[block_length * row + col];

        pangulu_recv_whole_pangulu_Smatrix_CSC(save_U, receive_id, tag, nnz, NB);
#ifdef ADD_GPU_MEMORY
#endif

#ifdef OVERLAP
        pthread_mutex_lock(heap_mutex);
#endif
        int_t offset_row = block_Smatrix->level_index_reverse[row];
        int_t offset_U = (block_Smatrix->U_Smatrix_nzz) * (offset_row - now_level);
        int_t offset_L = (block_Smatrix->L_Smatrix_nzz) * (offset_row - now_level);
        block_Smatrix->save_now_level_U[block_Smatrix->now_level_U_length[offset_row - now_level] + offset_U] = col;
        block_Smatrix->now_level_U_length[offset_row - now_level]++;
        int_t now_add_length = block_Smatrix->now_level_L_length[offset_row - now_level];
        int_t *save_now_level_L = (block_Smatrix->save_now_level_L) + offset_L;

        for (int_t i = 0; i < now_add_length; i++)
        {
            int_t now_row = save_now_level_L[i];
            int_t now_col = col;
            int_t now_rank = grid_process_id[(now_row % P) * Q + (now_col % Q)];
            int_t attain_id = level_task_rank_id[level * (P * Q) + now_rank];
            if ((attain_id == rank) && (mapper_A[now_row * block_length + now_col] != -1))
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
    int_t cpu_thread_count_per_node = sysconf(_SC_NPROCESSORS_ONLN);
    for(int i=0;i<PANGU_OMP_NUM_THREADS;i++){
    #ifdef HT_IS_OPEN
        CPU_SET((2*(PANGU_OMP_NUM_THREADS*RANK+i))%cpu_thread_count_per_node, &cpuset);
    #else
        CPU_SET((PANGU_OMP_NUM_THREADS*RANK+i)%cpu_thread_count_per_node, &cpuset);
    #endif
    }
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
        perror("pthread_setaffinity_np error");
    }
    
    // init
    thread_param *work_param = (thread_param *)param;
    pangulu_block_common *block_common = work_param->common;
    pangulu_block_Smatrix *block_Smatrix = work_param->Smatrix;
#ifdef GPU_OPEN
    pangulu_cuda_device_init_thread(block_common->rank);
#endif
    int_t block_length = block_common->block_length;
    int_t every_level_length = block_common->every_level_length;

    int_t *task_level_num = block_Smatrix->task_level_num;
    pangulu_Smatrix *calculate_L = block_Smatrix->calculate_L;
    pangulu_Smatrix *calculate_U = block_Smatrix->calculate_U;
    pangulu_Smatrix *calculate_X = block_Smatrix->calculate_X;

    bsem *run_bsem1 = block_Smatrix->run_bsem1;
    bsem *run_bsem2 = block_Smatrix->run_bsem2;
    pangulu_heap *heap = block_Smatrix->heap;
    compare_struct *compare_queue = heap->comapre_queue;
    pthread_mutex_t *heap_mutex = (heap->heap_bsem)->mutex;

    int_t rank = block_common->rank;
    int_t now_flag = 0;
    pangulu_bsem_synchronize(run_bsem2);

    for (int_t level = 0; level < block_length; level += every_level_length)
    {
        // communicate
        int_t now_task_num = task_level_num[level / every_level_length];

        while (now_task_num != 0)
        {

#ifdef CHECK_TIME
            struct timeval GET_TIME_START;
            pangulu_time_check_begin(&GET_TIME_START);
#endif
            int_t compare_flag = pangulu_bsem_wait(heap);
            now_task_num--;
            pangulu_numerical_work(compare_queue + compare_flag, block_common, block_Smatrix,
                                   calculate_L, calculate_U, calculate_X, level);

#ifdef CHECK_TIME
            calculate_TIME_wait += pangulu_time_check_end(&GET_TIME_START);
#endif
        }
        pangulu_bsem_stop(heap);

        pthread_mutex_lock(heap_mutex);
        if (heap->length != 0)
        {
            printf(PANGULU_W_RANK_HEAP_DONT_NULL);

            while (!heap_empty(heap))
            {
                int_t compare_flag = pangulu_heap_delete(heap);
                pangulu_numerical_work(compare_queue + compare_flag, block_common, block_Smatrix,
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
                            pangulu_block_Smatrix *block_Smatrix)
{
    pthread_t pthread;
    thread_param param;
    param.common = block_common;
    param.Smatrix = block_Smatrix;
    pthread_create(&pthread, NULL, &thread_GPU_work, (void *)(&param));
    pangulu_bsem_synchronize(block_Smatrix->run_bsem2);
}
#endif

void pangulu_numeric(pangulu_block_common *block_common,
                       pangulu_block_Smatrix *block_Smatrix)
{
    int_t cpu_thread_count_per_node = sysconf(_SC_NPROCESSORS_ONLN);
    #ifdef HT_IS_OPEN
        bind_to_core((2*PANGU_OMP_NUM_THREADS*RANK)%cpu_thread_count_per_node);
    #else
        bind_to_core((PANGU_OMP_NUM_THREADS*RANK)%cpu_thread_count_per_node);
    #endif

    int_32t rank = block_common->rank;
    int_32t block_length = block_common->block_length;
    int_32t P = block_common->P;
    int_32t Q = block_common->Q;

    int_t every_level_length = block_common->every_level_length;

    int_t *receive_level_num = block_Smatrix->receive_level_num;
    int_t *task_flag_id = block_Smatrix->task_flag_id;

    int_t *now_level_L_length = block_Smatrix->now_level_L_length;
    int_t *now_level_U_length = block_Smatrix->now_level_U_length;
    // init

    pangulu_heap *heap = block_Smatrix->heap;

    int_t *grid_process_id = block_Smatrix->grid_process_id;
    int_t *level_task_rank_id = block_Smatrix->level_task_rank_id;
    int_t *level_index = block_Smatrix->level_index;

#ifdef OVERLAP

    bsem *run_bsem1 = block_Smatrix->run_bsem1;
    bsem *run_bsem2 = block_Smatrix->run_bsem2;

    bsem *heap_bsem = heap->heap_bsem;
    pthread_mutex_t *heap_mutex = heap_bsem->mutex;

    int_t now_flag = 0;

    for (int_t level = 0; level < block_length; level += every_level_length)
    {

        MPI_Barrier(MPI_COMM_WORLD);
        for (int_t i = 0; i < block_Smatrix->L_Smatrix_nzz * every_level_length; i++)
        {
            block_Smatrix->flag_save_L[i] = 0;
        }
        for (int_t i = 0; i < block_Smatrix->U_Smatrix_nzz * every_level_length; i++)
        {
            block_Smatrix->flag_save_U[i] = 0;
        }

        LEVEL = level;
        for (int_t i = 0; i < block_Smatrix->L_Smatrix_nzz * every_level_length; i++)
        {
            (block_Smatrix->L_pangulu_Smatrix_value[i])->zip_flag = 0 ;
        }

        pthread_mutex_lock(heap_mutex);

        pangulu_zero_pangulu_heap(heap);
        // init

        int_t big_level = ((level + every_level_length) > block_length) ? block_length : (level + every_level_length);
        for (int_t k = level; k < big_level; k++)
        {
            int_t now_level = level_index[k];
            int_t now_rank = grid_process_id[(now_level % P) * Q + (now_level % Q)];
            int_t flag = level_task_rank_id[now_level * (P * Q) + now_rank];
            if (flag == rank)
            {
                task_flag_id[now_level * block_length + now_level]--;
                if (task_flag_id[now_level * block_length + now_level] == 0)
                {
                    pangulu_heap_insert(heap, now_level, now_level, now_level, 1, now_level);
                }
            }
        }
        for (int_t i = 0; i < every_level_length; i++)
        {
            now_level_L_length[i] = 0;
        }
        for (int_t i = 0; i < every_level_length; i++)
        {
            now_level_U_length[i] = 0;
        }

        pthread_mutex_unlock(heap_mutex);

        pangulu_bsem_post(heap);

        int_t now_receive_num = receive_level_num[level / every_level_length];
        MPI_Status status;

        while (now_receive_num != 0)
        {
            pangulu_probe_message(&status);
            now_receive_num--;
            pangulu_numerical_receive_message(status, level, block_common, block_Smatrix);
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

        int_t flag = pangulu_iprobe_message(&status);
        if (flag == 1)
        {
            printf(PANGULU_W_ERR_RANK);
            pangulu_numerical_receive_message(status, level, block_common, block_Smatrix);
        }
    }
#endif
}

#endif
