#ifndef PANGULU_PREPROCESS_H
#define PANGULU_PREPROCESS_H

#include "pangulu_common.h"

#ifdef GPU_OPEN
#include "pangulu_cuda_interface.h"
#endif

#include "pangulu_symbolic.h"
#include "pangulu_utils.h"
#include "pangulu_heap.h"
#include "pangulu_destroy.h"
#include "pangulu_reorder.h"

#ifdef OVERLAP
#include "pangulu_thread.h"
#endif

void pangulu_preprocess(pangulu_block_common *block_common,
                        pangulu_block_Smatrix *block_Smatrix,
                        pangulu_Smatrix *reorder_matrix)
{

    int_t N = block_common->N;
    int_32t rank = block_common->rank;
    int_32t P = block_common->P;
    int_32t Q = block_common->Q;
    int_32t NB = block_common->NB;

#ifdef SYMBOLIC
    int_t *L_symbolic_rowptr = NULL;
    int_32t *L_symbolic_colidx = NULL;
    int_t *U_symbolic_rowptr = NULL;
    int_32t *U_symbolic_colidx = NULL;

#endif

    if (rank == 0)
    {
#ifdef SYMBOLIC
#ifdef symmetric
        symbolic_sym_prune_get_U(block_Smatrix, N);
#endif
        L_symbolic_rowptr = block_Smatrix->L_rowpointer;
        L_symbolic_colidx = block_Smatrix->L_columnindex;
        U_symbolic_rowptr = block_Smatrix->U_rowpointer;
        U_symbolic_colidx = block_Smatrix->U_columnindex;

#endif
        long long int calculate_sum = 0;
        int_t *tmp_index = (int_t *)pangulu_malloc(sizeof(int_t) * (N + 5));
        for (int_t i = 0; i < N; i++)
        {
            tmp_index[i] = 1;
        }
        for (int_t i = 0; i < N; i++)
        {
            for (int_t j = U_symbolic_rowptr[i]; j < U_symbolic_rowptr[i + 1]; j++)
            {
                tmp_index[U_symbolic_colidx[j]]++;
            }
        }
        for (int_t i = 0; i < N; i++)
        {
            calculate_sum += (tmp_index[i] * (L_symbolic_rowptr[i + 1] - L_symbolic_rowptr[i]));
        }
        printf("Flop: %lld \n", calculate_sum * 2);
        free(tmp_index);
        FLOP = calculate_sum * 2;
    }

    int_32t block_length = block_common->block_length;
    int_t A_nnz_rowpointer_num = 0;
    int_32t sum_rank_size = block_common->sum_rank_size;

    int_t *level_task_rank_id = (int_t *)pangulu_malloc(sizeof(int_t) * (P * Q) * block_length);
    int_t *save_send_rank_flag = (int_t *)pangulu_malloc(sizeof(int_t) * sum_rank_size);
    for (int_t i = 0; i < sum_rank_size; i++)
    {
        save_send_rank_flag[i] = 0;
    }
    int_t *block_Smatrix_nnzA_num = (int_t *)pangulu_malloc(sizeof(int_t) * block_length * block_length);
    int_t *grid_process_id = (int_t *)pangulu_malloc(sizeof(int_t) * P * Q);
    int_t *block_Smatrix_non_zero_vector_L = (int_t *)pangulu_malloc(sizeof(int_t) * block_length);
    int_t *block_Smatrix_non_zero_vector_U = (int_t *)pangulu_malloc(sizeof(int_t) * block_length);
    int_t *save_block_Smatrix_columnpointer = NULL;
    idx_int *save_block_Smatrix_rowindex = NULL;
    calculate_type *save_value_csc = NULL;

    char *save_flag_block_num = NULL;
    int_t *save_block_Smatrix_csc_index = NULL;
    int_t *save_block_Smatrix_nnzA_num = NULL;
    int_t *every_rank_block_num = (int_t *)pangulu_malloc(sizeof(int_t) * sum_rank_size);
    int_t *every_rank_block_nnz = (int_t *)pangulu_malloc(sizeof(int_t) * sum_rank_size);
    int_t sum_send_num = 0;

    if (rank == 0)
    {

        for (int_t i = 0; i < block_length; i++)
        {
            for (int_t j = 0; j < (P * Q); j++)
            {
                level_task_rank_id[i * (P * Q) + j] = j;
            }
        }

        int_32t now_P = P;
        int_32t now_Q = Q;
        for (int_32t i = 0; i < P; i++)
        {
            int_32t offset = i % now_P;
            for (int_32t j = 0; j < Q; j++)
            {
                int_32t now_rank = (j % now_Q + offset * now_Q);
                grid_process_id[i * Q + j] = now_rank;
            }
        }

        save_block_Smatrix_nnzA_num = (int_t *)pangulu_malloc(sizeof(int_t) * block_length * block_length);

        for (int_t i = 0; i < block_length; i++)
        {
            block_Smatrix_non_zero_vector_L[i] = 0;
        }

        for (int_t i = 0; i < block_length; i++)
        {
            block_Smatrix_non_zero_vector_U[i] = 0;
        }

        for (int_t i = 0; i < block_length * block_length; i++)
        {
            block_Smatrix_nnzA_num[i] = 0;
        }

#ifdef SYMBOLIC
        // change begin
        for (int_t i = 0; i < N; i++)
        {
            for (int_t j = L_symbolic_rowptr[i]; j < L_symbolic_rowptr[i + 1]; j++)
            {
                int_t block_row = i / NB;
                int_t block_col = (L_symbolic_colidx[j]) / NB;
                block_Smatrix_nnzA_num[block_row * block_length + block_col]++;
                if (block_row == block_col)
                {
                    if (i >= L_symbolic_colidx[j])
                    {
                        block_Smatrix_non_zero_vector_L[block_row]++;
                    }
                    if (i <= L_symbolic_colidx[j])
                    {
                        block_Smatrix_non_zero_vector_U[block_col]++;
                    }
                }
            }
        }

        for (int_t i = 0; i < N; i++)
        {
            for (int_t j = U_symbolic_rowptr[i]; j < U_symbolic_rowptr[i + 1]; j++)
            {
                int_t block_row = i / NB;
                int_t block_col = (U_symbolic_colidx[j]) / NB;
                block_Smatrix_nnzA_num[block_row * block_length + block_col]++;
                if (block_row == block_col)
                {
                    if (i >= U_symbolic_colidx[j])
                    {
                        block_Smatrix_non_zero_vector_L[block_row]++;
                    }
                    if (i <= U_symbolic_colidx[j])
                    {
                        block_Smatrix_non_zero_vector_U[block_col]++;
                    }
                }
            }
        }

        // change end
#else
        for (int_t i = 0; i < N; i++)
        {
            for (int_t j = reorder_matrix->rowpointer[i]; j < reorder_matrix->rowpointer[i + 1]; j++)
            {
                int_t block_row = i / NB;
                int_t block_col = (reorder_matrix->columnindex[j]) / NB;
                block_Smatrix_nnzA_num[block_row * block_length + block_col]++;
                if (block_row == block_col)
                {
                    if (i >= reorder_matrix->columnindex[j])
                    {
                        block_Smatrix_non_zero_vector_L[block_row]++;
                    }
                    if (i <= reorder_matrix->columnindex[j])
                    {
                        block_Smatrix_non_zero_vector_U[block_col]++;
                    }
                }
            }
        }
#endif
        A_nnz_rowpointer_num = 0;

        int_t grid_id = 0;
        for (int_t offset_block_row = 0; offset_block_row < P; offset_block_row++)
        {
            for (int_t offset_block_col = 0; offset_block_col < Q; offset_block_col++)
            {
                for (int_t i = offset_block_row; i < block_length; i += P)
                {
                    for (int_t j = offset_block_col; j < block_length; j += Q)
                    {
                        if (block_Smatrix_nnzA_num[i * block_length + j] != 0)
                        {
                            save_block_Smatrix_nnzA_num[i * block_length + j] = A_nnz_rowpointer_num;
                            A_nnz_rowpointer_num++;
                        }
                        else
                        {
                            save_block_Smatrix_nnzA_num[i * block_length + j] = -1;
                        }
                    }
                }
                grid_id++;
            }
        }

        int_t *L_sum_columnpointer = (int_t *)pangulu_malloc(sizeof(int_t) * (block_length + 1));
        int_t *U_sum_rowpointer = (int_t *)pangulu_malloc(sizeof(int_t) * (block_length + 1));

        for (int_t i = 0; i < block_length + 1; i++)
        {
            L_sum_columnpointer[i] = 0;
        }
        for (int_t i = 0; i < block_length + 1; i++)
        {
            U_sum_rowpointer[i] = 0;
        }

        for (int_t i = 0; i < block_length; i++)
        {
            for (int_t j = 0; j < block_length; j++)
            {
                if (block_Smatrix_nnzA_num[i * block_length + j] != 0)
                {
                    if (i >= j)
                    {
                        L_sum_columnpointer[j + 1]++;
                    }
                    if (i <= j)
                    {
                        U_sum_rowpointer[i + 1]++;
                    }
                }
            }
        }

        for (int_t i = 0; i < block_length; i++)
        {
            L_sum_columnpointer[i + 1] += L_sum_columnpointer[i];
        }
        for (int_t i = 0; i < block_length; i++)
        {
            U_sum_rowpointer[i + 1] += U_sum_rowpointer[i];
        }

        int_t *L_sum_rowindex = (int_t *)pangulu_malloc(sizeof(int_t) * L_sum_columnpointer[block_length]);
        int_t *U_sum_columnindex = (int_t *)pangulu_malloc(sizeof(int_t) * U_sum_rowpointer[block_length]);
        for (int_t i = 0; i < block_length; i++)
        {
            for (int_t j = 0; j < block_length; j++)
            {
                if (block_Smatrix_nnzA_num[i * block_length + j] != 0)
                {
                    if (i >= j)
                    {
                        L_sum_rowindex[L_sum_columnpointer[j]++] = i;
                    }
                    if (i <= j)
                    {
                        U_sum_columnindex[U_sum_rowpointer[i]++] = j;
                    }
                }
            }
        }

        for (int_t i = block_length; i > 0; i--)
        {
            L_sum_columnpointer[i] = L_sum_columnpointer[i - 1];
        }
        for (int_t i = block_length; i > 0; i--)
        {
            U_sum_rowpointer[i] = U_sum_rowpointer[i - 1];
        }
        L_sum_columnpointer[0] = 0;
        U_sum_rowpointer[0] = 0;

        save_block_Smatrix_csc_index = (int_t *)pangulu_malloc(sizeof(int_t) * (A_nnz_rowpointer_num + 1));
        save_block_Smatrix_columnpointer = (int_t *)pangulu_malloc(sizeof(int_t) * (NB + 1) * A_nnz_rowpointer_num);
#ifdef SYMBOLIC
        save_block_Smatrix_rowindex = (idx_int *)pangulu_malloc(sizeof(idx_int) * (L_symbolic_rowptr[N] + U_symbolic_rowptr[N]));
        save_value_csc = (calculate_type *)pangulu_malloc(sizeof(calculate_type) * (L_symbolic_rowptr[N] + U_symbolic_rowptr[N]));

#else
        // change end

        save_block_Smatrix_rowindex = (idx_int *)pangulu_malloc(sizeof(idx_int) * reorder_matrix->rowpointer[N]);
        save_value_csc = (calculate_type *)pangulu_malloc(sizeof(calculate_type) * reorder_matrix->rowpointer[N]);
#endif

        for (int_t i = 0; i < (A_nnz_rowpointer_num + 1); i++)
        {
            save_block_Smatrix_csc_index[i] = 0;
        }
        for (int_t i = 0; i < A_nnz_rowpointer_num * (NB + 1); i++)
        {
            save_block_Smatrix_columnpointer[i] = 0;
        }

#ifdef SYMBOLIC
        // change begin
        for (int_t i = 0; i < (L_symbolic_rowptr[N] + U_symbolic_rowptr[N]); i++)
        {
            save_block_Smatrix_rowindex[i] = 0;
            save_value_csc[i] = 0.0;
        }
        // change end
#else
        for (int_t i = 0; i < reorder_matrix->rowpointer[N]; i++)
        {
            save_block_Smatrix_rowindex[i] = 0;
            save_value_csc[i] = 0.0;
        }
#endif

        A_nnz_rowpointer_num = 0;
        for (int_t i = 0; i < block_length; i++)
        {
            for (int_t j = 0; j < block_length; j++)
            {
                int_t now_mapper_index = save_block_Smatrix_nnzA_num[i * block_length + j];
                if (block_Smatrix_nnzA_num[i * block_length + j] != 0)
                {
                    save_block_Smatrix_csc_index[now_mapper_index + 1] = block_Smatrix_nnzA_num[i * block_length + j];
                    A_nnz_rowpointer_num++;
                }
            }
        }

        for (int_t i = 0; i < A_nnz_rowpointer_num; i++)
        {
            save_block_Smatrix_csc_index[i + 1] += save_block_Smatrix_csc_index[i];
        }

#ifdef SYMBOLIC

        // change begin
        for (int_t i = 0; i < N; i++)
        {
            for (int_t j = L_symbolic_rowptr[i]; j < L_symbolic_rowptr[i + 1]; j++)
            {
                int_t block_row = i / NB;
                int_t block_col = (L_symbolic_colidx[j]) / NB;
                save_block_Smatrix_columnpointer[save_block_Smatrix_nnzA_num[block_row * block_length + block_col] * (NB + 1) + (L_symbolic_colidx[j]) % NB + 1]++;
            }
        }

        for (int_t i = 0; i < N; i++)
        {
            for (int_t j = U_symbolic_rowptr[i]; j < U_symbolic_rowptr[i + 1]; j++)
            {
                int_t block_row = i / NB;
                int_t block_col = (U_symbolic_colidx[j]) / NB;
                save_block_Smatrix_columnpointer[save_block_Smatrix_nnzA_num[block_row * block_length + block_col] * (NB + 1) + (U_symbolic_colidx[j]) % NB + 1]++;
            }
        }
        // change end

#else
        for (int_t i = 0; i < N; i++)
        {
            for (int_t j = reorder_matrix->rowpointer[i]; j < reorder_matrix->rowpointer[i + 1]; j++)
            {
                int_t block_row = i / NB;
                int_t block_col = (reorder_matrix->columnindex[j]) / NB;
                save_block_Smatrix_columnpointer[save_block_Smatrix_nnzA_num[block_row * block_length + block_col] * (NB + 1) + (reorder_matrix->columnindex[j]) % NB + 1]++;
            }
        }
#endif
        for (int_t i = 0; i < A_nnz_rowpointer_num; i++)
        {
            int_t *now_Smatrix = save_block_Smatrix_columnpointer + i * (NB + 1);
            for (int_t j = 0; j < NB; j++)
            {
                now_Smatrix[j + 1] += now_Smatrix[j];
            }
        }
#ifdef SYMBOLIC
        // change begin

        for (int_t i = 0; i < N; i++)
        {
            int_t now_symbolic_index = reorder_matrix->rowpointer[i];
            int_t max_index = reorder_matrix->rowpointer[i + 1];
            for (int_t j = U_symbolic_rowptr[i]; j < U_symbolic_rowptr[i + 1]; j++)
            {
                int_t block_row = i / NB;
                int_t block_col = (U_symbolic_colidx[j]) / NB;
                int_t now_mapper_index = save_block_Smatrix_nnzA_num[block_row * block_length + block_col];
                int_t *now_columnpointer = save_block_Smatrix_columnpointer + now_mapper_index * (NB + 1);
                idx_int *now_rowindex = save_block_Smatrix_rowindex + save_block_Smatrix_csc_index[now_mapper_index];
                calculate_type *now_value_csc = save_value_csc + save_block_Smatrix_csc_index[now_mapper_index];
                now_rowindex[now_columnpointer[(U_symbolic_colidx[j]) % NB]] = i % NB;
                if (now_symbolic_index < max_index && U_symbolic_colidx[j] == reorder_matrix->columnindex[now_symbolic_index])
                {
                    now_value_csc[now_columnpointer[(U_symbolic_colidx[j]) % NB]] = reorder_matrix->value[now_symbolic_index];
                    now_symbolic_index++;
                }
                else
                {
                    now_value_csc[now_columnpointer[(U_symbolic_colidx[j]) % NB]] = 0.0;
                }
                now_columnpointer[(U_symbolic_colidx[j]) % NB]++;
            }
            for (int_t j = L_symbolic_rowptr[i]; j < L_symbolic_rowptr[i + 1]; j++)
            {
                int_t block_row = i / NB;
                int_t block_col = (L_symbolic_colidx[j]) / NB;
                int_t now_mapper_index = save_block_Smatrix_nnzA_num[block_row * block_length + block_col];
                int_t *now_columnpointer = save_block_Smatrix_columnpointer + now_mapper_index * (NB + 1);
                idx_int *now_rowindex = save_block_Smatrix_rowindex + save_block_Smatrix_csc_index[now_mapper_index];
                calculate_type *now_value_csc = save_value_csc + save_block_Smatrix_csc_index[now_mapper_index];
                now_rowindex[now_columnpointer[(L_symbolic_colidx[j]) % NB]] = i % NB;
                if (now_symbolic_index < max_index && L_symbolic_colidx[j] == reorder_matrix->columnindex[now_symbolic_index])
                {
                    now_value_csc[now_columnpointer[(L_symbolic_colidx[j]) % NB]] = reorder_matrix->value[now_symbolic_index];
                    now_symbolic_index++;
                }
                else
                {
                    now_value_csc[now_columnpointer[(L_symbolic_colidx[j]) % NB]] = 0.0;
                }
                now_columnpointer[(L_symbolic_colidx[j]) % NB]++;
            }
            if (now_symbolic_index != reorder_matrix->rowpointer[i + 1])
            {
                printf("restore the matrix error %ld %ld\n", now_symbolic_index, reorder_matrix->rowpointer[i + 1]);
                for (int_t j = U_symbolic_rowptr[i]; j < U_symbolic_rowptr[i + 1]; j++)
                {
                    printf("%d ", U_symbolic_colidx[j]);
                }
                printf("\n");
                for (int_t j = L_symbolic_rowptr[i]; j < L_symbolic_rowptr[i + 1]; j++)
                {
                    printf("%d ", L_symbolic_colidx[j]);
                }
                printf("\n");
                for (int_t j = reorder_matrix->rowpointer[i]; j < reorder_matrix->rowpointer[i + 1]; j++)
                {
                    printf("%d ", reorder_matrix->columnindex[j]);
                }
                printf("\n");
            }
        }

        free(L_symbolic_rowptr);
        free(L_symbolic_colidx);
        free(U_symbolic_rowptr);
        free(U_symbolic_colidx);

        L_symbolic_rowptr = NULL;
        L_symbolic_colidx = NULL;
        U_symbolic_rowptr = NULL;
        U_symbolic_colidx = NULL;

        block_Smatrix->L_rowpointer = NULL;
        block_Smatrix->L_columnindex = NULL;
        block_Smatrix->U_rowpointer = NULL;
        block_Smatrix->U_columnindex = NULL;

        // change end
#else

        for (int_t i = 0; i < N; i++)
        {
            for (int_t j = reorder_matrix->rowpointer[i]; j < reorder_matrix->rowpointer[i + 1]; j++)
            {
                int_t block_row = i / NB;
                int_t block_col = (reorder_matrix->columnindex[j]) / NB;
                int_t now_mapper_index = save_block_Smatrix_nnzA_num[block_row * block_length + block_col];
                int_t *now_columnpointer = save_block_Smatrix_columnpointer + now_mapper_index * (NB + 1);
                idx_int *now_rowindex = save_block_Smatrix_rowindex + save_block_Smatrix_csc_index[now_mapper_index];
                calculate_type *now_value_csc = save_value_csc + save_block_Smatrix_csc_index[now_mapper_index];
                now_rowindex[now_columnpointer[(reorder_matrix->columnindex[j]) % NB]] = i % NB;
                now_value_csc[now_columnpointer[(reorder_matrix->columnindex[j]) % NB]] = reorder_matrix->value[j];
                now_columnpointer[(reorder_matrix->columnindex[j]) % NB]++;
            }
        }
#endif

#ifndef CHECK_LU
        pangulu_destroy_part_pangulu_Smatrix(reorder_matrix);
#endif

        // return ;
        for (int_t i = 0; i < A_nnz_rowpointer_num; i++)
        {
            int_t *now_columnpointer = save_block_Smatrix_columnpointer + i * (NB + 1);
            for (int_t j = NB; j > 0; j--)
            {
                now_columnpointer[j] = now_columnpointer[j - 1];
            }
            now_columnpointer[0] = 0;
        }

        save_flag_block_num = (char *)pangulu_malloc(sizeof(char) * block_length * block_length * sum_rank_size);
        for (int_t i = 0; i < block_length * block_length * sum_rank_size; i++)
        {
            save_flag_block_num[i] = 0;
        }

        for (int_t level = 0; level < block_length; level++)
        {
            for (int_t L_num = L_sum_columnpointer[level]; L_num < L_sum_columnpointer[level + 1]; L_num++)
            {
                int_t L_row = L_sum_rowindex[L_num];
                for (int_t U_num = U_sum_rowpointer[level]; U_num < U_sum_rowpointer[level + 1]; U_num++)
                {
                    int_t U_col = U_sum_columnindex[U_num];
                    int_t block_index = L_row * block_length + U_col;
                    if (block_Smatrix_nnzA_num[block_index] != 0)
                    {
                        int_t now_rank = grid_process_id[(L_row % P) * Q + (U_col % Q)];
                        int_t now_task_rank = level_task_rank_id[level * (P * Q) + now_rank];
                        save_flag_block_num[now_task_rank * block_length * block_length + block_index] = 1;
                        // printf("block index %ld rank %ld\n",block_index,now_task_rank);
                    }
                }
            }
        }

        for (int_t i = 0; i < sum_rank_size; i++)
        {
            every_rank_block_num[i] = 0;
        }
        for (int_t i = 0; i < sum_rank_size; i++)
        {
            char *now_save_flag_block_num = save_flag_block_num + i * block_length * block_length;
            for (int_t j = 0; j < block_length * block_length; j++)
            {
                if (now_save_flag_block_num[j] != 0)
                {
                    every_rank_block_num[i]++;
                }
            }
        }

        for (int_t i = 0; i < sum_rank_size; i++)
        {
            every_rank_block_nnz[i] = 0;
        }
        for (int_t i = 0; i < sum_rank_size; i++)
        {
            char *now_save_flag_block_num = save_flag_block_num + i * block_length * block_length;
            for (int_t j = 0; j < block_length * block_length; j++)
            {
                if (now_save_flag_block_num[j] != 0)
                {
                    every_rank_block_nnz[i] += block_Smatrix_nnzA_num[j];
                }
            }
        }
        for (int_t i = 0; i < sum_rank_size; i++)
        {
            sum_send_num += every_rank_block_nnz[i];
        }
    }
    pangulu_Bcast_vector(block_Smatrix_nnzA_num, block_length * block_length, 0);
    pangulu_Bcast_vector(grid_process_id, P * Q, 0);
    pangulu_Bcast_vector(block_Smatrix_non_zero_vector_L, block_length, 0);
    pangulu_Bcast_vector(block_Smatrix_non_zero_vector_U, block_length, 0);
    pangulu_Bcast_vector(level_task_rank_id, (P * Q) * block_length, 0);
    pangulu_Bcast_vector(every_rank_block_num, sum_rank_size, 0);
    pangulu_Bcast_vector(every_rank_block_nnz, sum_rank_size, 0);

    int_t sum_process_grid_num = every_rank_block_num[rank];
    int_t sum_process_grid_nnz = every_rank_block_nnz[rank];

    char *max_tmp = NULL;
    int_32t send_max_length = 100000000;
    if (rank == 0)
    {

        int_t max_block_num = 0;
        int_t max_nnz = 0;
        for (int_t i = 0; i < sum_rank_size; i++)
        {
            max_block_num = PANGULU_MAX(max_block_num, every_rank_block_num[i]);
        }
        for (int_t i = 0; i < sum_rank_size; i++)
        {
            max_nnz = PANGULU_MAX(max_nnz, every_rank_block_nnz[i]);
        }

        max_tmp = (char *)pangulu_malloc(sizeof(int_t) * (max_block_num) * (NB + 1) + (sizeof(idx_int) + sizeof(calculate_type)) * max_nnz);

        for (int_t i = 1; i < sum_rank_size; i++)
        {
            int_t now_offset = 0;
            char *now_save_flag_block_num = save_flag_block_num + i * block_length * block_length;

            for (int_t now_row = 0; now_row < block_length; now_row++)
            {
                for (int_t now_col = 0; now_col < block_length; now_col++)
                {
                    if (now_save_flag_block_num[now_row * block_length + now_col] == 1)
                    {
                        int_t now_mapper_index = save_block_Smatrix_nnzA_num[now_row * block_length + now_col];
                        int_t *now_columnpointer = save_block_Smatrix_columnpointer + now_mapper_index * (NB + 1);
                        idx_int *now_rowindex = save_block_Smatrix_rowindex + save_block_Smatrix_csc_index[now_mapper_index];
                        calculate_type *now_value_csc = save_value_csc + save_block_Smatrix_csc_index[now_mapper_index];

                        char *now_tmp = max_tmp + now_offset;
                        int_t *save_send_columnpointer = (int_t *)(now_tmp);
                        idx_int *save_send_rowindex = (idx_int *)(now_tmp + (NB + 1) * sizeof(int_t));
                        calculate_type *save_send_value_csc = (calculate_type *)(now_tmp + (NB + 1) * sizeof(int_t) + now_columnpointer[NB] * sizeof(idx_int));

                        for (int_t k = 0; k < (NB + 1); k++)
                        {
                            save_send_columnpointer[k] = now_columnpointer[k];
                        }

                        for (int_t k = 0; k < now_columnpointer[NB]; k++)
                        {
                            save_send_rowindex[k] = now_rowindex[k];
                        }
                        for (int_t k = 0; k < now_columnpointer[NB]; k++)
                        {
                            save_send_value_csc[k] = now_value_csc[k];
                        }
                        now_offset += ((NB + 1) * sizeof(int_t) + now_columnpointer[NB] * (sizeof(idx_int) + sizeof(calculate_type)));
                    }
                }
            }
            int_t send_length = every_rank_block_num[i] * (NB + 1) * sizeof(int_t) + every_rank_block_nnz[i] * (sizeof(calculate_type) + sizeof(idx_int));
            for (int_t j = 0; j < (send_length + send_max_length - 1) / send_max_length; j++)
            {
                if ((send_length - j * send_max_length) >= send_max_length)
                {
                    pangulu_send_vector_char(max_tmp + j * send_max_length, send_max_length, i, i + sum_rank_size * j);
                }
                else
                {
                    pangulu_send_vector_char(max_tmp + j * send_max_length, (int_32t)(send_length - j * send_max_length), i, i + sum_rank_size * j);
                }
            }
        }

        int_t now_offset = 0;
        for (int_t now_row = 0; now_row < block_length; now_row++)
        {
            for (int_t now_col = 0; now_col < block_length; now_col++)
            {
                if (save_flag_block_num[now_row * block_length + now_col] == 1)
                {
                    int_t now_mapper_index = save_block_Smatrix_nnzA_num[now_row * block_length + now_col];
                    int_t *now_columnpointer = save_block_Smatrix_columnpointer + now_mapper_index * (NB + 1);
                    idx_int *now_rowindex = save_block_Smatrix_rowindex + save_block_Smatrix_csc_index[now_mapper_index];
                    calculate_type *now_value_csc = save_value_csc + save_block_Smatrix_csc_index[now_mapper_index];

                    char *now_tmp = max_tmp + now_offset;
                    int_t *save_send_columnpointer = (int_t *)(now_tmp);
                    idx_int *save_send_rowindex = (idx_int *)(now_tmp + (NB + 1) * sizeof(int_t));
                    calculate_type *save_send_value_csc = (calculate_type *)(now_tmp + (NB + 1) * sizeof(int_t) + now_columnpointer[NB] * sizeof(idx_int));

                    for (int_t k = 0; k < (NB + 1); k++)
                    {
                        save_send_columnpointer[k] = now_columnpointer[k];
                    }
                    for (int_t k = 0; k < now_columnpointer[NB]; k++)
                    {
                        save_send_rowindex[k] = now_rowindex[k];
                    }
                    for (int_t k = 0; k < now_columnpointer[NB]; k++)
                    {
                        save_send_value_csc[k] = now_value_csc[k];
                    }
                    now_offset += ((NB + 1) * sizeof(int_t) + now_columnpointer[NB] * (sizeof(idx_int) + sizeof(calculate_type)));
                }
            }
        }

        free(save_block_Smatrix_columnpointer);
        free(save_block_Smatrix_rowindex);
        free(save_value_csc);
        free(save_block_Smatrix_csc_index);
        free(save_block_Smatrix_nnzA_num);
    }
    else
    {
        max_tmp = (char *)pangulu_malloc(sizeof(int_t) * (sum_process_grid_num) * (NB + 1) + (sizeof(idx_int) + sizeof(calculate_type)) * sum_process_grid_nnz);
        int_t recv_length = sum_process_grid_num * (NB + 1) * sizeof(int_t) + sum_process_grid_nnz * (sizeof(idx_int) + sizeof(calculate_type));
        for (int_t j = 0; j < (recv_length + send_max_length - 1) / send_max_length; j++)
        {
            if ((recv_length - j * send_max_length) >= send_max_length)
            {
                pangulu_recv_vector_char(max_tmp + j * send_max_length, send_max_length, 0, rank + sum_rank_size * j);
            }
            else
            {
                pangulu_recv_vector_char(max_tmp + j * send_max_length, (int_32t)(recv_length - j * send_max_length), 0, rank + sum_rank_size * j);
            }
        }
    }

    if (rank == 0)
    {
        for (int_t i = 1; i < sum_rank_size; i++)
        {
            pangulu_send_vector_char(save_flag_block_num + i * block_length * block_length, block_length * block_length, i, i);
        }
    }
    else
    {
        save_flag_block_num = (char *)pangulu_malloc(sizeof(char) * block_length * block_length);
        pangulu_recv_vector_char(save_flag_block_num, block_length * block_length, 0, rank);
    }

    if (rank == -1)
    {
        printf("save flag block num:\n");
        for (int_t i = 0; i < block_length; i++)
        {
            for (int_t j = 0; j < block_length; j++)
            {
                printf("%d ", save_flag_block_num[i * block_length + j]);
            }
            printf("\n");
        }
    }
    int_t *sum_flag_block_num = (int_t *)pangulu_malloc(sizeof(int_t) * block_length * block_length);

    if (rank == 0)
    {
        for (int_t i = 0; i < block_length * block_length; i++)
        {
            sum_flag_block_num[i] = 0;
        }
        for (int_t i = 0; i < sum_rank_size; i++)
        {
            char *now_flag_block_num = save_flag_block_num + block_length * block_length * i;
            for (int_t j = 0; j < block_length * block_length; j++)
            {
                if (now_flag_block_num[j] == 1)
                {
                    sum_flag_block_num[j]++;
                }
            }
        }
    }

    pangulu_Bcast_vector(sum_flag_block_num, block_length * block_length, 0);

    int_t *mapper_A = (int_t *)pangulu_malloc(sizeof(int_t) * block_length * block_length);
    for (int_t i = 0; i < block_length * block_length; i++)
    {
        mapper_A[i] = -1;
    }

    pangulu_Smatrix **Big_Smatrix_value = (pangulu_Smatrix **)pangulu_malloc(sizeof(pangulu_Smatrix *) * sum_process_grid_num);

    for (int_t i = 0; i < sum_process_grid_num; i++)
    {
        Big_Smatrix_value[i] = NULL;
    }

    int_t *real_matrix_flag = (int_t *)pangulu_malloc(sizeof(int_t) * sum_process_grid_num);
    for (int_t i = 0; i < sum_process_grid_num; i++)
    {
        real_matrix_flag[i] = 0;
    }
    sum_process_grid_num = 0;
    sum_process_grid_nnz = 0;
    for (int_t now_row = 0; now_row < block_length; now_row++)
    {
        for (int_t now_col = 0; now_col < block_length; now_col++)
        {
            if (save_flag_block_num[now_row * block_length + now_col] != 0)
            {
                pangulu_Smatrix *tmp = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
                pangulu_init_pangulu_Smatrix(tmp);

                // new
                int_t offset = sum_process_grid_num * (NB + 1) * sizeof(int_t) + sum_process_grid_nnz * (sizeof(idx_int) + sizeof(calculate_type));
                int_t *now_columnpointer = (int_t *)(max_tmp + offset);
                idx_int *now_rowindex = (idx_int *)(max_tmp + offset + (NB + 1) * sizeof(int_t));
                calculate_type *now_value_csc = (calculate_type *)(max_tmp + offset + (NB + 1) * sizeof(int_t) + now_columnpointer[NB] * sizeof(idx_int));

                tmp->nnz = now_columnpointer[NB];
                tmp->columnpointer = now_columnpointer;
                tmp->rowindex = now_rowindex;
                tmp->value_CSC = now_value_csc;
                tmp->row = NB;
                tmp->column = NB;

                sum_process_grid_nnz += now_columnpointer[NB];
                mapper_A[now_row * block_length + now_col] = sum_process_grid_num;
                Big_Smatrix_value[sum_process_grid_num] = tmp;

                int_t now_rank = grid_process_id[(now_row % P) * Q + (now_col % Q)];
                int_t flag = level_task_rank_id[(P * Q) * PANGULU_MIN(now_row, now_col) + now_rank];

                if (flag == rank)
                {
                    real_matrix_flag[sum_process_grid_num] = 1;
                }

                sum_process_grid_num++;
            }
        }
    }

    free(save_flag_block_num);

    pangulu_Smatrix **Big_pangulu_Smatrix_copy_value = (pangulu_Smatrix **)pangulu_malloc(sizeof(pangulu_Smatrix *) * sum_process_grid_num);

    for (int_t i = 0; i < sum_process_grid_num; i++)
    {
        Big_pangulu_Smatrix_copy_value[i] = NULL;
    }

    for (int_t i = 0; i < block_length; i++)
    {
        for (int_t j = 0; j < block_length; j++)
        {
            int_t now_offset = i * block_length + j;
            int_t now_mapperA_offset = mapper_A[now_offset];
            if (now_mapperA_offset != -1 && sum_flag_block_num[now_offset] > 1)
            {
                if (real_matrix_flag[now_mapperA_offset] == 1)
                {
                    // pangulu_malloc
                    pangulu_Smatrix *tmp = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
                    pangulu_init_pangulu_Smatrix(tmp);
                    pangulu_malloc_pangulu_Smatrix_CSC_value(Big_Smatrix_value[now_mapperA_offset], tmp);
                    pangulu_malloc_pangulu_Smatrix_CSR_value(Big_Smatrix_value[now_mapperA_offset], tmp);
                    Big_pangulu_Smatrix_copy_value[now_mapperA_offset] = tmp;
                    pangulu_memcpy_zero_pangulu_Smatrix_CSC_value(tmp);
                    pangulu_memcpy_zero_pangulu_Smatrix_CSR_value(tmp);
                }
                else
                {
                    pangulu_memcpy_zero_pangulu_Smatrix_CSC_value(Big_Smatrix_value[now_mapperA_offset]);
                }
            }
        }
    }

    if (rank == -1)
    {
        for (int_t i = 0; i < block_length; i++)
        {
            for (int_t j = 0; j < block_length; j++)
            {
                int_t index = mapper_A[i * block_length + j];
                if (index != -1)
                {
                    pangulu_display_pangulu_Smatrix_CSC(Big_Smatrix_value[index]);
                }
            }
        }
    }

    int_t *tmp_save_block_num = (int_t *)pangulu_malloc(sizeof(int_t) * block_length * block_length);

    for (int_t i = 0; i < block_length * block_length; i++)
    {
        tmp_save_block_num[i] = -1;
    }

    for (int_t offset_block_index = 0; offset_block_index < P * Q; offset_block_index++)
    {
        int_t offset_block_row = offset_block_index / Q;
        int_t offset_block_col = offset_block_index % Q;
        int_t now_rank = grid_process_id[offset_block_index];
        for (int_t level = 0; level < block_length; level++)
        {
            if (level_task_rank_id[level * (P * Q) + now_rank] == rank)
            {
                int_t offset_row = calculate_offset(offset_block_row, level, P);
                int_t offset_col = calculate_offset(offset_block_col, level, Q);
                for (int_t i = offset_col + level; i < block_length; i += Q)
                {
                    int_t find_index = block_Smatrix_nnzA_num[level * block_length + i];
                    if (find_index != 0)
                    {
                        tmp_save_block_num[level * block_length + i] = find_index;
                    }
                }
                for (int_t i = offset_row + level; i < block_length; i += P)
                {
                    int_t find_index = block_Smatrix_nnzA_num[i * block_length + level];
                    if (find_index != 0)
                    {
                        tmp_save_block_num[i * block_length + level] = find_index;
                    }
                }
            }
        }
    }

    if (rank == -1)
    {
        printf("tmp save block num:\n");
        for (int_t i = 0; i < block_length; i++)
        {
            for (int_t j = 0; j < block_length; j++)
            {
                printf("%ld ", tmp_save_block_num[i * block_length + j]);
            }
            printf("\n");
        }
    }

    int_t *U_rowpointer = (int_t *)pangulu_malloc(sizeof(int_t) * (block_length + 1));
    int_t *L_columnpointer = (int_t *)pangulu_malloc(sizeof(int_t) * (block_length + 1));

    U_rowpointer[0] = 0;
    for (int_t i = 0; i < block_length + 1; i++)
    {
        L_columnpointer[i] = 0;
    }
    for (int_t level = 0; level < block_length; level++)
    {
        U_rowpointer[level + 1] = U_rowpointer[level];
        for (int_t i = level; i < block_length; i++)
        {
            if (tmp_save_block_num[level * block_length + i] != -1)
            {
                U_rowpointer[level + 1]++;
            }
        }
    }
    for (int_t level = 0; level < block_length; level++)
    {
        for (int_t i = 0; i <= level; i++)
        {
            if (tmp_save_block_num[level * block_length + i] != -1)
            {
                L_columnpointer[i + 1]++;
            }
        }
    }
    for (int_t i = 0; i < block_length; i++)
    {
        L_columnpointer[i + 1] += L_columnpointer[i];
    }

    int_t *U_columnindex = (int_t *)pangulu_malloc(sizeof(int_t) * U_rowpointer[block_length]);
    int_t *L_rowindex = (int_t *)pangulu_malloc(sizeof(int_t) * L_columnpointer[block_length]);

    for (int_t level = 0; level < block_length; level++)
    {
        for (int_t i = level; i < block_length; i++)
        {
            if (tmp_save_block_num[level * block_length + i] != -1)
            {
                U_columnindex[U_rowpointer[level]] = i;
                U_rowpointer[level]++;
            }
        }
        for (int_t i = 0; i <= level; i++)
        {
            if (tmp_save_block_num[level * block_length + i] != -1)
            {
                L_rowindex[L_columnpointer[i]] = level;
                L_columnpointer[i]++;
            }
        }
    }
    for (int_t i = block_length; i > 0; i--)
    {
        L_columnpointer[i] = L_columnpointer[i - 1];
    }
    for (int_t i = block_length; i > 0; i--)
    {
        U_rowpointer[i] = U_rowpointer[i - 1];
    }
    L_columnpointer[0] = 0;
    U_rowpointer[0] = 0;

    int_t every_level_length = block_common->every_level_length;

    // begin calculate level mapping
    int_t *level_index = (int_t *)pangulu_malloc(sizeof(int_t) * (block_length));
    int_t *level_index_reverse = (int_t *)pangulu_malloc(sizeof(int_t) * (block_length));

    for (int_t i = 0; i < block_length; i++)
    {
        level_index[i] = i;
    }

    // optimize begin
    int_t *tmp_diggonal_task_id = (int_t *)pangulu_malloc(sizeof(int_t) * (block_length));
    int_t now_num = 0;
    for (int_t i = 0; i < block_length; i++)
    {
        tmp_diggonal_task_id[i] = 0;
    }

    for (int_t i = 0; i < block_length; i++)
    {
        for (int_t j = i + 1; j < block_length; j++)
        {
            if ((block_Smatrix_nnzA_num[i * block_length + j] != 0) || (block_Smatrix_nnzA_num[j * block_length + i] != 0))
            {
                tmp_diggonal_task_id[j]++;
            }
        }
    }
    for (int_t i = 0; i < block_length; i++)
    {
        if (tmp_diggonal_task_id[i] == 0)
        {
            level_index[now_num++] = i;
        }
    }
    for (int_t i = 0; i < block_length; i++)
    {
        int_t now_flag = level_index[i];
        if (i >= now_num)
        {
            printf("error in level %ld\n", i);
            continue;
        }
        for (int_t j = now_flag + 1; j < block_length; j++)
        {
            if ((block_Smatrix_nnzA_num[now_flag * block_length + j] != 0) || (block_Smatrix_nnzA_num[j * block_length + now_flag] != 0))
            {
                tmp_diggonal_task_id[j]--;
                if (tmp_diggonal_task_id[j] == 0)
                {
                    level_index[now_num++] = j;
                }
                if (tmp_diggonal_task_id[j] < 0)
                {
                    printf("error in now flag %ld j %ld\n", now_flag, j);
                }
            }
        }
    }
    free(tmp_diggonal_task_id);
    for (int_t i = 0; i < block_length; i++)
    {
        level_index_reverse[level_index[i]] = i;
    }

    int_t U_Smatrix_nzz = 0;
    int_t L_Smatrix_nzz = 0;
    for (int_t i = 0; i < block_length; i++)
    {
        U_Smatrix_nzz = PANGULU_MAX(U_rowpointer[i + 1] - U_rowpointer[i], U_Smatrix_nzz);
    }
    for (int_t i = 0; i < block_length; i++)
    {
        L_Smatrix_nzz = PANGULU_MAX(L_columnpointer[i + 1] - L_columnpointer[i], L_Smatrix_nzz);
    }
    int_t *now_level_L_length = (int_t *)pangulu_malloc(sizeof(int_t) * every_level_length);
    int_t *now_level_U_length = (int_t *)pangulu_malloc(sizeof(int_t) * every_level_length);

    for (int_t i = 0; i < every_level_length; i++)
    {
        now_level_L_length[i] = 0;
        now_level_U_length[i] = 0;
    }

    int_t *save_now_level_L = (int_t *)pangulu_malloc(sizeof(int_t) * L_Smatrix_nzz * every_level_length);
    int_t *save_now_level_U = (int_t *)pangulu_malloc(sizeof(int_t) * U_Smatrix_nzz * every_level_length);

    int_t now_nnz_L = 0;
    int_t now_nnz_U = 0;
    int_t *mapper_LU = (int_t *)pangulu_malloc(sizeof(int_t) * block_length * block_length);
    for (int_t i = 0; i < block_length * block_length; i++)
    {
        mapper_LU[i] = -1;
    }
    for (int_t i = 0; i < block_length; i++)
    {
        int_t mapper_level = level_index[i];
        if (i % every_level_length == 0)
        {
            now_nnz_L = 0;
            now_nnz_U = 0;
        }
        for (int_t j = U_rowpointer[mapper_level]; j < U_rowpointer[mapper_level + 1]; j++)
        {
            int_t mapper_index_U = mapper_level * block_length + U_columnindex[j];
            mapper_LU[mapper_index_U] = now_nnz_U++;
        }
        for (int_t j = L_columnpointer[mapper_level]; j < L_columnpointer[mapper_level + 1]; j++)
        {
            int_t mapper_index_L = L_rowindex[j] * block_length + mapper_level;
            mapper_LU[mapper_index_L] = now_nnz_L++;
        }
    }

    int_t MAX_all_nnzL = 0;
    int_t MAX_all_nnzU = 0;
    int_t *MAX_level_nnzL = (int_t *)pangulu_malloc(sizeof(int_t) * L_Smatrix_nzz * every_level_length);
    int_t *MAX_level_nnzU = (int_t *)pangulu_malloc(sizeof(int_t) * U_Smatrix_nzz * every_level_length);

    char *flag_save_L = (char *)pangulu_malloc(sizeof(char) * L_Smatrix_nzz * every_level_length);
    char *flag_save_U = (char *)pangulu_malloc(sizeof(char) * U_Smatrix_nzz * every_level_length);

    block_Smatrix->flag_save_L = flag_save_L;
    block_Smatrix->flag_save_U = flag_save_U;

    for (int_t i = 0; i < L_Smatrix_nzz * every_level_length; i++)
    {
        MAX_level_nnzL[i] = 0;
    }
    for (int_t i = 0; i < U_Smatrix_nzz * every_level_length; i++)
    {
        MAX_level_nnzU[i] = 0;
    }
    int_t U_Smatrix_index = 0;
    int_t L_Smatrix_index = 0;

    for (int_t row = 0; row < block_length; row++)
    {
        int_t mapper_level = level_index[row];
        for (int_t index = U_rowpointer[mapper_level]; index < U_rowpointer[mapper_level + 1]; index++)
        {
            int_t col = U_columnindex[index];
            int_t find_index;
            if (mapper_level == col)
            {
                find_index = block_Smatrix_non_zero_vector_U[mapper_level];
                if (find_index != 0)
                {
                    MAX_all_nnzU = PANGULU_MAX(MAX_all_nnzU, find_index);
                    U_Smatrix_index++;
                }
            }
            else
            {
                find_index = block_Smatrix_nnzA_num[mapper_level * block_length + col];
                if (find_index != 0)
                {
                    MAX_all_nnzU = PANGULU_MAX(MAX_all_nnzU, find_index);
                    int_t mapper_index_U = mapper_level * block_length + col;
                    int_t save_index = mapper_LU[mapper_index_U];
                    MAX_level_nnzU[save_index] = PANGULU_MAX(MAX_level_nnzU[save_index], find_index);
                    U_Smatrix_index++;
                }
            }
        }
    }

    for (int_t col = 0; col < block_length; col++)
    {
        int_t mapper_level = level_index[col];
        for (int_t index = L_columnpointer[mapper_level]; index < L_columnpointer[mapper_level + 1]; index++)
        {
            int_t row = L_rowindex[index];
            int_t find_index;
            if (row == mapper_level)
            {
                find_index = block_Smatrix_non_zero_vector_L[mapper_level];
                if (find_index != 0)
                {
                    MAX_all_nnzL = PANGULU_MAX(MAX_all_nnzL, find_index);
                    L_Smatrix_index++;
                }
            }
            else
            {
                find_index = block_Smatrix_nnzA_num[row * block_length + mapper_level];
                if (find_index != 0)
                {
                    MAX_all_nnzL = PANGULU_MAX(MAX_all_nnzL, find_index);
                    int_t mapper_index_L = row * block_length + mapper_level;
                    int_t save_index = mapper_LU[mapper_index_L];
                    MAX_level_nnzL[save_index] = PANGULU_MAX(MAX_level_nnzL[save_index], find_index);
                    L_Smatrix_index++;
                }
            }
        }
    }

    pangulu_Smatrix **U_value = (pangulu_Smatrix **)pangulu_malloc(sizeof(pangulu_Smatrix *) * U_Smatrix_nzz * every_level_length);
    pangulu_Smatrix **L_value = (pangulu_Smatrix **)pangulu_malloc(sizeof(pangulu_Smatrix *) * L_Smatrix_nzz * every_level_length);

    for (int_t i = 0; i < U_Smatrix_nzz * every_level_length; i++)
    {
        pangulu_Smatrix *first_U = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
        pangulu_init_pangulu_Smatrix(first_U);
        pangulu_malloc_pangulu_Smatrix_nnz_CSC(first_U, NB, MAX_level_nnzU[i]);

#ifdef ADD_GPU_MEMORY
        pangulu_Smatrix_CUDA_memory_init(first_U, NB, MAX_level_nnzU[i]);
#endif
        U_value[i] = first_U;
    }
    for (int_t i = 0; i < L_Smatrix_nzz * every_level_length; i++)
    {
        pangulu_Smatrix *first_L = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
        pangulu_init_pangulu_Smatrix(first_L);
        pangulu_malloc_pangulu_Smatrix_nnz_CSC(first_L, NB, MAX_level_nnzL[i]);

#ifdef ADD_GPU_MEMORY
        pangulu_Smatrix_CUDA_memory_init(first_L, NB, MAX_level_nnzL[i]);

#endif
        L_value[i] = first_L;
    }

    free(MAX_level_nnzL);
    free(MAX_level_nnzU);

    int_t MAX_all_nnzX = 0;
    for (int_t i = 0; i < sum_process_grid_num; i++)
    {
        pangulu_Smatrix_add_more_memory_CSR(Big_Smatrix_value[i]);
#ifdef GPU_OPEN
        pangulu_Smatrix_add_CUDA_memory(Big_Smatrix_value[i]);
        pangulu_Smatrix_CUDA_memcpy_A(Big_Smatrix_value[i]);
        pangulu_cuda_malloc((void **)&(Big_Smatrix_value[i]->d_left_sum), ((Big_Smatrix_value[i]->nnz)) * sizeof(calculate_type));
#endif

        MAX_all_nnzX = PANGULU_MAX(MAX_all_nnzX, Big_Smatrix_value[i]->nnz);
    }

#ifndef GPU_OPEN

    int_t *work_space = (int_t *)malloc(sizeof(int_t) * (4 * NB + 8));
    for (int_t i = 0; i < block_length; i++)
    {

        int_t now_offset = i * block_length + i;
        int_t now_mapperA_offset = mapper_A[now_offset];
        if (now_mapperA_offset != -1 && real_matrix_flag[now_mapperA_offset] == 1)
        {
            pangulu_malloc_Smatrix_level(Big_Smatrix_value[now_mapperA_offset]);
            pangulu_init_level_array(Big_Smatrix_value[now_mapperA_offset], work_space);
        }
    }
    free(work_space);
#endif

    pangulu_Smatrix *calculate_L = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
    pangulu_Smatrix *calculate_U = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
    pangulu_Smatrix *calculate_X = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));

    pangulu_init_pangulu_Smatrix(calculate_L);
    pangulu_init_pangulu_Smatrix(calculate_U);
    pangulu_init_pangulu_Smatrix(calculate_X);

#ifdef GPU_OPEN
    pangulu_Smatrix_CUDA_memory_init(calculate_L, NB, MAX_all_nnzL);
    pangulu_Smatrix_CUDA_memory_init(calculate_U, NB, MAX_all_nnzU);
    pangulu_Smatrix_add_CUDA_memory_U(calculate_U);
    pangulu_Smatrix_CUDA_memory_init(calculate_X, NB, MAX_all_nnzX);
#else
    calculate_L->row = NB;
    calculate_L->column = NB;
    calculate_L->nnz = MAX_all_nnzL;

    calculate_U->row = NB;
    calculate_U->column = NB;
    calculate_U->nnz = MAX_all_nnzU;

    calculate_X->row = NB;
    calculate_X->column = NB;
    calculate_X->nnz = MAX_all_nnzX;

#endif

    pangulu_malloc_pangulu_Smatrix_value_CSR(calculate_X, MAX_all_nnzX);
    pangulu_malloc_pangulu_Smatrix_value_CSC(calculate_X, MAX_all_nnzX);

    int_t diagonal_nnz = 0;
    int_t *mapper_diagonal_Smatrix = (int_t *)pangulu_malloc(sizeof(int_t) * block_length);

    for (int_t i = 0; i < block_length; i++)
    {
        mapper_diagonal_Smatrix[i] = -1;
    }
    for (int_t i = 0; i < block_length; i++)
    {
        if (tmp_save_block_num[i * block_length + i] != -1)
        {
            mapper_diagonal_Smatrix[i] = diagonal_nnz;
            diagonal_nnz++;
        }
    }

    pangulu_Smatrix **diagonal_U = (pangulu_Smatrix **)pangulu_malloc(sizeof(pangulu_Smatrix *) * diagonal_nnz);
    pangulu_Smatrix **diagonal_L = (pangulu_Smatrix **)pangulu_malloc(sizeof(pangulu_Smatrix *) * diagonal_nnz);

    char *flag_dignon_L = (char *)pangulu_malloc(sizeof(char) * diagonal_nnz);
    char *flag_dignon_U = (char *)pangulu_malloc(sizeof(char) * diagonal_nnz);

    for (int_t i = 0; i < diagonal_nnz; i++)
    {
        flag_dignon_L[i] = 0;
        flag_dignon_U[i] = 0;
    }

    block_Smatrix->flag_dignon_L = flag_dignon_L;
    block_Smatrix->flag_dignon_U = flag_dignon_U;

    for (int_t i = 0; i < diagonal_nnz; i++)
    {
        diagonal_U[i] = NULL;
    }
    for (int_t i = 0; i < diagonal_nnz; i++)
    {
        diagonal_L[i] = NULL;
    }

    for (int_t level = 0; level < block_length; level++)
    {
        int_t diagonal_index = mapper_diagonal_Smatrix[level];
        if (diagonal_index != -1)
        {
            int_t now_rank = grid_process_id[(level % P) * Q + level % Q];
            if (level_task_rank_id[level * (P * Q) + now_rank] == rank)
            {
                int_t first_index = mapper_A[level * block_length + level];
                if (diagonal_U[diagonal_index] == NULL)
                {
                    int_t first_index = mapper_A[level * block_length + level];
                    pangulu_Smatrix *first_U = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
                    pangulu_init_pangulu_Smatrix(first_U);
                    pangulu_get_pangulu_Smatrix_to_U(Big_Smatrix_value[first_index], first_U, NB);
                    pangulu_Smatrix_add_CSC(first_U);
                    pangulu_Smatrix_add_memory_U(first_U);
#ifdef ADD_GPU_MEMORY
                    pangulu_Smatrix_CUDA_memory_init(first_U, first_U->row, first_U->nnz);
                    pangulu_Smatrix_add_CUDA_memory_U(first_U);
                    pangulu_Smatrix_CUDA_memcpy_nnzU(first_U, first_U);
                    pangulu_Smatrix_CUDA_memcpy_struct_CSC(first_U, first_U);
                    pangulu_cuda_malloc((void **)&(first_U->d_graphInDegree), NB * sizeof(int));
                    pangulu_cuda_malloc((void **)&(first_U->d_id_extractor), sizeof(int));
                    first_U->graphInDegree = (int *)pangulu_malloc(NB * sizeof(int));
#endif
                    diagonal_U[diagonal_index] = first_U;
                }
                if (diagonal_L[diagonal_index] == NULL)
                {
                    pangulu_Smatrix *first_L = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
                    pangulu_init_pangulu_Smatrix(first_L);
                    pangulu_get_pangulu_Smatrix_to_L(Big_Smatrix_value[first_index], first_L, NB);
                    pangulu_Smatrix_add_CSC(first_L);
#ifdef ADD_GPU_MEMORY
                    pangulu_Smatrix_CUDA_memory_init(first_L, first_L->row, first_L->nnz);
                    pangulu_Smatrix_CUDA_memcpy_struct_CSC(first_L, first_L);
                    pangulu_cuda_malloc((void **)&(first_L->d_graphInDegree), NB * sizeof(int));
                    pangulu_cuda_malloc((void **)&(first_L->d_id_extractor), sizeof(int));
                    first_L->graphInDegree = (int *)pangulu_malloc(NB * sizeof(int));
#endif
                    diagonal_L[diagonal_index] = first_L;
                }
            }
            else
            {
                if (diagonal_L[diagonal_index] == NULL)
                {
                    pangulu_Smatrix *first_L = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
                    pangulu_init_pangulu_Smatrix(first_L);
                    pangulu_malloc_pangulu_Smatrix_nnz_CSC(first_L, NB, block_Smatrix_non_zero_vector_L[level]);

#ifdef ADD_GPU_MEMORY
                    pangulu_Smatrix_CUDA_memory_init(first_L, NB, first_L->nnz);
                    pangulu_cuda_malloc((void **)&(first_L->d_graphInDegree), NB * sizeof(int));
                    pangulu_cuda_malloc((void **)&(first_L->d_id_extractor), sizeof(int));
                    first_L->graphInDegree = (int *)pangulu_malloc(NB * sizeof(int));
#endif
                    diagonal_L[diagonal_index] = first_L;
                }
                if (diagonal_U[diagonal_index] == NULL)
                {
                    pangulu_Smatrix *first_U = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
                    pangulu_init_pangulu_Smatrix(first_U);
                    pangulu_malloc_pangulu_Smatrix_nnz_CSR(first_U, NB, block_Smatrix_non_zero_vector_U[level]);

#ifdef ADD_GPU_MEMORY
                    pangulu_Smatrix_CUDA_memory_init(first_U, NB, first_U->nnz);
                    pangulu_cuda_malloc((void **)&(first_U->d_graphInDegree), NB * sizeof(int));
                    pangulu_cuda_malloc((void **)&(first_U->d_id_extractor), sizeof(int));
                    first_U->graphInDegree = (int *)pangulu_malloc(NB * sizeof(int));
#endif
                    diagonal_U[diagonal_index] = first_U;
                }
            }
        }
    }

    free(tmp_save_block_num);

    int_t *task_flag_id = (int_t *)pangulu_malloc(sizeof(int_t) * block_length * block_length);

    for (int_t i = 0; i < block_length * block_length; i++)
    {
        task_flag_id[i] = 0;
    }

    int_t task_level_length = block_length / every_level_length + (((block_length % every_level_length) == 0) ? 0 : 1);
    int_t *task_level_num = (int_t *)pangulu_malloc(sizeof(int_t) * task_level_length);
    int_t *receive_level_num = (int_t *)pangulu_malloc(sizeof(int_t) * task_level_length);

    for (int_t i = 0; i < task_level_length; i++)
    {
        task_level_num[i] = 0;
    }

    for (int_t i = 0; i < task_level_length; i++)
    {
        receive_level_num[i] = 0;
    }

    int_t *save_block_L = (int_t *)pangulu_malloc(sizeof(int_t) * block_length);
    int_t *save_block_U = (int_t *)pangulu_malloc(sizeof(int_t) * block_length);

    for (int_t k = 0; k < block_length; k++)
    {
        int_t level = level_index[k];
        int_t now_task_level = k / every_level_length;
        for (int_t i = level; i < block_length; i++)
        {
            save_block_L[i] = 0;
        }
        for (int_t i = level; i < block_length; i++)
        {
            save_block_U[i] = 0;
        }
        for (int_t i = L_columnpointer[level]; i < L_columnpointer[level + 1]; i++)
        {
            int_t row = L_rowindex[i];
            for (int_t j = U_rowpointer[level]; j < U_rowpointer[level + 1]; j++)
            {
                int_t col = U_columnindex[j];
                int_t now_offset_index = row * block_length + col;
                int_t now_offset_rank = grid_process_id[(row % P) * Q + col % Q];
                if (mapper_A[now_offset_index] != -1 && (level_task_rank_id[level * (P * Q) + now_offset_rank] == rank))
                {
                    save_block_L[row] = 1;
                    save_block_U[col] = 1;
                }
            }
        }
        for (int_t i = level; i < block_length; i++)
        {
            if (save_block_L[i] == 1)
            {
                int_t mapper_index_A = mapper_A[i * block_length + level];
                if (mapper_index_A == -1)
                {
                    receive_level_num[now_task_level]++;
                }
                else if (real_matrix_flag[mapper_index_A] != 1)
                {
                    receive_level_num[now_task_level]++;
                }
            }
        }
        for (int_t i = level; i < block_length; i++)
        {
            if (save_block_U[i] == 1)
            {
                int_t mapper_index_A = mapper_A[level * block_length + i];
                if (mapper_index_A == -1)
                {
                    receive_level_num[now_task_level]++;
                }
                else if (real_matrix_flag[mapper_index_A] != 1)
                {
                    receive_level_num[now_task_level]++;
                }
            }
        }
    }
    free(save_block_L);
    free(save_block_U);


    for (int_t k = 0; k < block_length; k++)
    {
        int_t level = level_index[k];
        int_t now_task_level = k / every_level_length;
        for (int_t i = L_columnpointer[level]; i < L_columnpointer[level + 1]; i++)
        {
            int_t row = L_rowindex[i];
            for (int_t j = U_rowpointer[level]; j < U_rowpointer[level + 1]; j++)
            {
                int_t col = U_columnindex[j];
                int_t now_offset_index = row * block_length + col;
                int_t now_offset_rank = grid_process_id[(row % P) * Q + col % Q];
                if (mapper_A[now_offset_index] != -1 && (level_task_rank_id[level * (P * Q) + now_offset_rank] == rank))
                {
                    task_flag_id[row * block_length + col]++;
                    task_level_num[now_task_level]++;
                }
            }
        }
    }

    int_t *save_flag = (int_t *)pangulu_malloc(sizeof(int_t) * sum_rank_size);
    for (int_t i = 0; i < sum_rank_size; i++)
    {
        save_flag[i] = -1;
    }

    int_t *save_task_level_flag = (int_t *)pangulu_malloc(sizeof(int_t) * task_level_length);

    for (int_t row = 0; row < block_length; row++)
    {
        for (int_t col = 0; col < block_length; col++)
        {
            int_t now_offset_index = row * block_length + col;
            if (mapper_A[now_offset_index] == -1)
            {
                continue;
            }
            int_t real_flag = real_matrix_flag[mapper_A[now_offset_index]];
            if ((real_flag != 0) && (sum_flag_block_num[now_offset_index] > 1))
            {
                int_t min_level = PANGULU_MIN(row, col);
                int_t now_offset_rank = grid_process_id[(row % P) * Q + col % Q];

                for (int_t i = 0; i < sum_rank_size; i++)
                {
                    save_flag[i] = -1;
                }

                for (int_t now_level = 0; now_level < min_level; now_level++)
                {
                    if ((block_Smatrix_nnzA_num[now_level * block_length + col] != 0) && (block_Smatrix_nnzA_num[row * block_length + now_level] != 0))
                    {
                        int_t now_rank = level_task_rank_id[now_level * (P * Q) + now_offset_rank];
                        save_flag[now_rank] = now_level;
                    }
                }
                for (int_t i = 0; i < task_level_length; i++)
                {
                    save_task_level_flag[i] = 0;
                }
                for (int_t i = 0; i < sum_rank_size; i++)
                {
                    if ((i != rank) && (save_flag[i] != -1))
                    {
                        // task_level_num[save_flag[i] / every_level_length]++;
                        save_task_level_flag[save_flag[i] / every_level_length]++;
                    }
                }
                for (int_t i = task_level_length - 1; i >= 0; i--)
                {
                    if (save_task_level_flag[i] != 0)
                    {
                        task_level_num[i]++;
                        break;
                    }
                }
                for (int_t i = 0; i < task_level_length; i++)
                {
                    receive_level_num[i] += save_task_level_flag[i];
                }
            }
        }
    }
    free(save_flag);
    free(save_task_level_flag);

    for (int_t row = 0; row < block_length; row++)
    {
        for (int_t col = 0; col < block_length; col++)
        {
            int_t now_offset_index = row * block_length + col;
            if (task_flag_id[now_offset_index] != 0 && sum_flag_block_num[now_offset_index] > 1)
            {
                int_t now_offset_rank = grid_process_id[(row % P) * Q + col % Q];
                if (level_task_rank_id[PANGULU_MIN(row, col) * (P * Q) + now_offset_rank] == rank)
                {
                    if (sum_flag_block_num[now_offset_index] > 1)
                    {
                        task_flag_id[now_offset_index]++;
                    }
                }
            }
        }
    }

    int_t max_PQ = block_common->max_PQ;
    int_t *send_flag = (int_t *)pangulu_malloc(sizeof(int_t) * sum_process_grid_num * max_PQ);
    int_t *send_diagonal_flag_L = (int_t *)pangulu_malloc(sizeof(int_t) * diagonal_nnz * max_PQ);
    int_t *send_diagonal_flag_U = (int_t *)pangulu_malloc(sizeof(int_t) * diagonal_nnz * max_PQ);

    for (int_t i = 0; i < sum_process_grid_num * max_PQ; i++)
    {
        send_flag[i] = 0;
    }
    for (int_t i = 0; i < diagonal_nnz * max_PQ; i++)
    {
        send_diagonal_flag_L[i] = 0;
    }
    for (int_t i = 0; i < diagonal_nnz * max_PQ; i++)
    {
        send_diagonal_flag_U[i] = 0;
    }

    for (int_t row = 0; row < block_length; row++)
    {
        for (int_t col = 0; col < block_length; col++)
        {
            int_t mapper_index = mapper_A[row * block_length + col];
            if (mapper_index != -1 && real_matrix_flag[mapper_index] != 0)
            {
                if (row == col)
                {
                    int_t diagonal_index = mapper_diagonal_Smatrix[row];
                    for (int_t i = row + 1; i < block_length; i++)
                    {
                        if (block_Smatrix_nnzA_num[i * block_length + col] != 0)
                        {
                            send_diagonal_flag_U[diagonal_index * max_PQ + (i - row) % P] = 1;
                        }
                    }
                    for (int_t i = col + 1; i < block_length; i++)
                    {
                        if (block_Smatrix_nnzA_num[row * block_length + i] != 0)
                        {
                            send_diagonal_flag_L[diagonal_index * max_PQ + (i - col) % Q] = 1;
                        }
                    }
                }
                else if (row < col)
                {
                    for (int_t i = row + 1; i < block_length; i++)
                    {
                        if (block_Smatrix_nnzA_num[i * block_length + row] != 0 && block_Smatrix_nnzA_num[i * block_length + col] != 0)
                        {
                            send_flag[mapper_index * max_PQ + (i - row) % P] = 1;
                        }
                    }
                }
                else
                {
                    for (int_t i = col + 1; i < block_length; i++)
                    {
                        if (block_Smatrix_nnzA_num[col * block_length + i] != 0 && block_Smatrix_nnzA_num[row * block_length + i] != 0)
                        {
                            send_flag[mapper_index * max_PQ + (i - col) % Q] = 1;
                        }
                    }
                }
            }
        }
    }

    int_t max_task_length = 0;
    for (int_t i = 0; i < task_level_length; i++)
    {
        max_task_length = PANGULU_MAX(task_level_num[i], max_task_length);
    }

    pangulu_heap *heap = (pangulu_heap *)pangulu_malloc(sizeof(pangulu_heap));
    pangulu_init_pangulu_heap(heap, max_task_length);

    int_t *mapper_mpi = (int_t *)pangulu_malloc(sizeof(int_t) * block_length * (block_length + 1));
    for (int_t i = 0; i < block_length * block_length; i++)
    {
        mapper_mpi[i] = -1;
    }

    int_t *mpi_level_num = (int_t *)pangulu_malloc(sizeof(int_t) * task_level_length);
    for (int_t i = 0; i < task_level_length; i++)
    {
        mpi_level_num[i] = 0;
    }

    int_t block_non_zero_length = 0;
    for (int_t level = 0; level < block_length; level += every_level_length)
    {
        int_t block_non_zero_num = 0;
        int_t big_level = ((level + every_level_length) > block_length) ? block_length : (level + every_level_length);
        for (int_t k = level; k < big_level; k++)
        {
            int_t now_level = level_index[k];
            if (block_Smatrix_nnzA_num[now_level * block_length + now_level] != 0)
            {
                mapper_mpi[now_level * block_length + now_level] = block_non_zero_num++;
                mapper_mpi[block_length * block_length + now_level] = block_non_zero_num++;
            }
            else
            {
                printf("error diagnal is null\n");
            }
            for (int_t j = now_level + 1; j < block_length; j++)
            {
                if (block_Smatrix_nnzA_num[now_level * block_length + j] != 0)
                {
                    mapper_mpi[now_level * block_length + j] = block_non_zero_num++;
                }
                if (block_Smatrix_nnzA_num[j * block_length + now_level] != 0)
                {
                    mapper_mpi[j * block_length + now_level] = block_non_zero_num++;
                }
            }
        }
        mpi_level_num[level / every_level_length] = block_non_zero_length;
        block_non_zero_length += block_non_zero_num;
    }

    int_t *mapper_mpi_reverse = (int_t *)pangulu_malloc(sizeof(int_t) * block_non_zero_length);

    block_non_zero_length = 0;
    for (int_t level = 0; level < block_length; level += every_level_length)
    {
        int_t big_level = ((level + every_level_length) > block_length) ? block_length : (level + every_level_length);
        for (int_t k = level; k < big_level; k++)
        {
            int_t now_level = level_index[k];
            if (block_Smatrix_nnzA_num[now_level * block_length + now_level] != 0)
            {
                mapper_mpi_reverse[block_non_zero_length++] = now_level * block_length + now_level;
                mapper_mpi_reverse[block_non_zero_length++] = block_length * block_length + now_level;
            }
            else
            {
                printf("error diagnal is null\n");
            }
            for (int_t j = now_level + 1; j < block_length; j++)
            {
                if (block_Smatrix_nnzA_num[now_level * block_length + j] != 0)
                {
                    mapper_mpi_reverse[block_non_zero_length++] = now_level * block_length + j;
                }
                if (block_Smatrix_nnzA_num[j * block_length + now_level] != 0)
                {
                    mapper_mpi_reverse[block_non_zero_length++] = j * block_length + now_level;
                }
            }
        }
    }

    if (rank == -1)
    {
        printf("ever length\n");
        for (int_t i = 0; i < task_level_length; i++)
        {
            printf("%ld ", task_level_num[i]);
        }
        printf("\ngrid :\n");
        for (int_t i = 0; i < P; i++)
        {
            for (int_t j = 0; j < Q; j++)
            {
                printf("%ld ", grid_process_id[i * Q + j]);
            }
            printf("\n");
        }
        printf("L:\n");
        for (int_t i = 0; i < block_length + 1; i++)
        {
            printf("%ld ", L_columnpointer[i]);
        }
        printf("\n");
        for (int_t i = 0; i < block_length; i++)
        {
            for (int_t j = L_columnpointer[i]; j < L_columnpointer[i + 1]; j++)
            {
                printf("%ld ", L_rowindex[j]);
            }
        }
        printf("\nU:\n");
        for (int_t i = 0; i < block_length + 1; i++)
        {
            printf("%ld ", U_rowpointer[i]);
        }
        printf("\n");
        for (int_t i = 0; i < block_length; i++)
        {
            for (int_t j = U_rowpointer[i]; j < U_rowpointer[i + 1]; j++)
            {
                printf("%ld ", U_columnindex[j]);
            }
        }
        printf("\nblock num:\n");
        for (int_t i = 0; i < block_length; i++)
        {
            for (int_t j = 0; j < block_length; j++)
            {
                printf("%ld ", block_Smatrix_nnzA_num[i * block_length + j]);
            }
            printf("\n");
        }
        printf("\ntask num:\n");
        for (int_t i = 0; i < block_length; i++)
        {
            for (int_t j = 0; j < block_length; j++)
            {
                printf("%ld ", task_flag_id[i * block_length + j]);
            }
            printf("\n");
        }
        printf("\nsum_flag_block_num:\n");
        for (int_t i = 0; i < block_length; i++)
        {
            for (int_t j = 0; j < block_length; j++)
            {
                printf("%ld ", sum_flag_block_num[i * block_length + j]);
            }
            printf("\n");
        }
        printf("\nmapper lu:\n");
        for (int_t i = 0; i < block_length; i++)
        {
            for (int_t j = 0; j < block_length; j++)
            {
                printf("%ld ", mapper_LU[i * block_length + j]);
            }
            printf("\n");
        }
        printf("max PQ : %ld\n", max_PQ);
        printf("send diagonal L\n");
        for (int_t i = 0; i < diagonal_nnz; i++)
        {
            for (int_t j = 0; j < max_PQ; j++)
            {
                printf("%ld ", send_diagonal_flag_L[i * max_PQ + j]);
            }
            printf("\n");
        }
        printf("send diagoanl U\n");
        for (int_t i = 0; i < diagonal_nnz; i++)
        {
            for (int_t j = 0; j < max_PQ; j++)
            {
                printf("%ld ", send_diagonal_flag_U[i * max_PQ + j]);
            }
            printf("\n");
        }
        printf("send flag\n");
        for (int_t i = 0; i < sum_process_grid_num; i++)
        {
            for (int_t j = 0; j < max_PQ; j++)
            {
                printf("%ld ", send_flag[i * max_PQ + j]);
            }
            printf("\n");
        }
        printf("real matrix flag\n");
        for (int_t i = 0; i < sum_process_grid_num; i++)
        {
            printf("%ld \n", real_matrix_flag[i]);
        }

        printf("\nlevel every length %ld\n", every_level_length);

        printf("task level num:\n");
        for (int_t i = 0; i < task_level_length; i++)
        {
            printf("%ld ", task_level_num[i]);
        }
        printf("\n");
        printf("reiceive level num:\n");
        for (int_t i = 0; i < task_level_length; i++)
        {
            printf("%ld ", receive_level_num[i]);
        }
        printf("\n");
    }

    TEMP_A_value = (calculate_type *)pangulu_malloc(sizeof(calculate_type) * NB * NB);
    ssssm_col_ops_u = (int_t *)pangulu_malloc(sizeof(int_t) * (NB + 1));
    getrf_diagIndex_csc = (idx_int *)pangulu_malloc(sizeof(idx_int) * (NB + 1));
    getrf_diagIndex_csr = (idx_int *)pangulu_malloc(sizeof(idx_int) * (NB + 1));
    int omp_threads_num = PANGU_OMP_NUM_THREADS;
    ssssm_ops_pointer = (idx_int *)pangulu_malloc(sizeof(idx_int) * (omp_threads_num + 1));
#ifdef GPU_OPEN

    pangulu_cuda_malloc((void **)&CUDA_TEMP_value, NB * NB * sizeof(calculate_type));
    pangulu_cuda_malloc((void **)&CUDA_B_idx_COL, NB * NB * sizeof(idx_int));
#endif

    block_Smatrix->mapper_Big_pangulu_Smatrix = mapper_A;
    block_Smatrix->block_Smatrix_nnzA_num = block_Smatrix_nnzA_num;
    block_Smatrix->block_Smatrix_non_zero_vector_L = block_Smatrix_non_zero_vector_L;
    block_Smatrix->block_Smatrix_non_zero_vector_U = block_Smatrix_non_zero_vector_U;
    block_Smatrix->Big_pangulu_Smatrix_value = Big_Smatrix_value;
    block_Smatrix->Big_pangulu_Smatrix_copy_value = Big_pangulu_Smatrix_copy_value;

    block_Smatrix->L_pangulu_Smatrix_columnpointer = L_columnpointer;
    block_Smatrix->L_pangulu_Smatrix_rowindex = L_rowindex;
    block_Smatrix->L_pangulu_Smatrix_value = L_value;
    block_Smatrix->L_Smatrix_nzz = L_Smatrix_nzz;

    block_Smatrix->U_pangulu_Smatrix_rowpointer = U_rowpointer;
    block_Smatrix->U_pangulu_Smatrix_columnindex = U_columnindex;
    block_Smatrix->U_pangulu_Smatrix_value = U_value;
    block_Smatrix->U_Smatrix_nzz = U_Smatrix_nzz;

    block_Smatrix->mapper_diagonal = mapper_diagonal_Smatrix;
    block_Smatrix->diagonal_Smatrix_L = diagonal_L;
    block_Smatrix->diagonal_Smatrix_U = diagonal_U;

    block_Smatrix->calculate_L = calculate_L;
    block_Smatrix->calculate_U = calculate_U;
    block_Smatrix->calculate_X = calculate_X;

    block_Smatrix->task_level_length = task_level_length;
    block_Smatrix->mapper_LU = mapper_LU;
    block_Smatrix->task_flag_id = task_flag_id;
    block_Smatrix->task_level_num = task_level_num;
    block_Smatrix->heap = heap;
    block_Smatrix->now_level_L_length = now_level_L_length;
    block_Smatrix->now_level_U_length = now_level_U_length;
    block_Smatrix->save_now_level_L = save_now_level_L;
    block_Smatrix->save_now_level_U = save_now_level_U;

    block_Smatrix->send_flag = send_flag;
    block_Smatrix->send_diagonal_flag_L = send_diagonal_flag_L;
    block_Smatrix->send_diagonal_flag_U = send_diagonal_flag_U;

    block_Smatrix->grid_process_id = grid_process_id;

    block_Smatrix->save_send_rank_flag = save_send_rank_flag;

    block_Smatrix->level_task_rank_id = level_task_rank_id;
    block_Smatrix->real_matrix_flag = real_matrix_flag;
    block_Smatrix->sum_flag_block_num = sum_flag_block_num;
    block_Smatrix->receive_level_num = receive_level_num;
    block_Smatrix->max_tmp = max_tmp;

    block_Smatrix->level_index = level_index;
    block_Smatrix->level_index_reverse = level_index_reverse;

    block_Smatrix->mapper_mpi = mapper_mpi;
    block_Smatrix->mapper_mpi_reverse = mapper_mpi_reverse;
    block_Smatrix->mpi_level_num = mpi_level_num;

#ifdef OVERLAP

    block_Smatrix->run_bsem1 = (bsem *)pangulu_malloc(sizeof(bsem));
    block_Smatrix->run_bsem2 = (bsem *)pangulu_malloc(sizeof(bsem));
    pangulu_bsem_init(block_Smatrix->run_bsem1, 0);
    pangulu_bsem_init(block_Smatrix->run_bsem2, 0);

    heap->heap_bsem = (bsem *)pangulu_malloc(sizeof(bsem));
    pangulu_bsem_init(heap->heap_bsem, 0);
#endif

    return;
}

#endif
