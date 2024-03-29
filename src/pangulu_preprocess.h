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

    int_t *symbolic_rowpointer = NULL;
    int_32t *symbolic_columnindex = NULL;


    if (rank == 0)
    {
        symbolic_rowpointer = block_Smatrix->symbolic_rowpointer;
        symbolic_columnindex = block_Smatrix->symbolic_columnindex;
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
    int_t *block_origin_nnzA_num = (int_t *)pangulu_malloc(sizeof(int_t) * block_length * block_length);
        for (int_t i = 0; i < block_length * block_length; i++)
    {
        block_origin_nnzA_num[i] = 0;
    }
    
    int_t *grid_process_id = (int_t *)pangulu_malloc(sizeof(int_t) * P * Q);
    
    char *save_flag_block_num = NULL;
    int_t *save_block_Smatrix_nnzA_num = NULL;
    int_t *every_rank_block_num = (int_t *)pangulu_malloc(sizeof(int_t) * sum_rank_size);
    int_t *every_rank_block_nnz = (int_t *)pangulu_malloc(sizeof(int_t) * sum_rank_size);
    int_t sum_send_num = 0;

    int_t *block_Smatrix_nnzA_num = NULL;
    int_t *block_Smatrix_non_zero_vector_L = NULL;
    int_t *block_Smatrix_non_zero_vector_U = NULL;
    
    if(rank==0){
        block_Smatrix_nnzA_num=block_Smatrix->block_Smatrix_nnzA_num;
        block_Smatrix_non_zero_vector_L=block_Smatrix->block_Smatrix_non_zero_vector_L;
        block_Smatrix_non_zero_vector_U=block_Smatrix->block_Smatrix_non_zero_vector_U;
    }
    else{
        block_Smatrix_nnzA_num = (int_t *)pangulu_malloc(sizeof(int_t) * block_length * block_length);
        block_Smatrix_non_zero_vector_L = (int_t *)pangulu_malloc(sizeof(int_t) * block_length);
        block_Smatrix_non_zero_vector_U = (int_t *)pangulu_malloc(sizeof(int_t) * block_length);
    }
    
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

        for (int_t i = 0; i < N; i++)
        {
            for (int_t j = reorder_matrix->rowpointer[i]; j < reorder_matrix->rowpointer[i + 1]; j++)
            {
                int_t block_row = i / NB;
                int_t block_col = (reorder_matrix->columnindex[j]) / NB;
                block_origin_nnzA_num[block_row * block_length + block_col]++;
            }
        }

        A_nnz_rowpointer_num = 0;

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
            }
        }

        save_flag_block_num = (char *)pangulu_malloc(sizeof(char) * block_length * block_length * sum_rank_size);
        for (int_t i = 0; i < block_length * block_length * sum_rank_size; i++)
        {
            save_flag_block_num[i] = 0;
        }

        for (int_t level = 0; level < block_length; level++)
        {
            for (int_t L_row = level; L_row < block_length; L_row++)
            {
                if (block_Smatrix_nnzA_num[L_row * block_length + level] == 0)
                {
                    continue;
                }
                for (int_t U_col = level; U_col < block_length; U_col++)
                {
                    if (block_Smatrix_nnzA_num[level * block_length + U_col] == 0)
                    {
                        continue;
                    }
                    int_t block_index = L_row * block_length + U_col;
                    if (block_Smatrix_nnzA_num[block_index] != 0)
                    {
                        int_t now_rank = grid_process_id[(L_row % P) * Q + (U_col % Q)];
                        int_t now_task_rank = level_task_rank_id[level * (P * Q) + now_rank];
                        save_flag_block_num[now_task_rank * block_length * block_length + block_index] = 1;
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
    pangulu_Bcast_vector(block_origin_nnzA_num, block_length * block_length, 0);
    pangulu_Bcast_vector(grid_process_id, P * Q, 0);
    pangulu_Bcast_vector(block_Smatrix_non_zero_vector_L, block_length, 0);
    pangulu_Bcast_vector(block_Smatrix_non_zero_vector_U, block_length, 0);
    pangulu_Bcast_vector(level_task_rank_id, (P * Q) * block_length, 0);
    pangulu_Bcast_vector(every_rank_block_num, sum_rank_size, 0);
    pangulu_Bcast_vector(every_rank_block_nnz, sum_rank_size, 0);

    int_t sum_process_grid_num = every_rank_block_num[rank];
    int_t sum_process_grid_nnz = every_rank_block_nnz[rank];

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

    char *save_tmp = (char *)pangulu_malloc(sizeof(int_t) * (sum_process_grid_num) * (NB + 1) + (sizeof(idx_int) + sizeof(calculate_type)) * sum_process_grid_nnz);


    sum_process_grid_num = 0;
    sum_process_grid_nnz = 0;
    int_t max_nnz = 0;
    int_t preprocess_ompnum = 4;

    if (rank == 0)
    {
        int req = 0;
        MPI_Request *Request = (MPI_Request *)pangulu_malloc(sizeof(MPI_Request) * block_length * preprocess_ompnum);
        for (int_t i = 0; i < block_length * block_length; i++)
        {
            max_nnz = PANGULU_MAX(max_nnz, block_Smatrix_nnzA_num[i]);
        }
        int_t max_origin_nnz = 0;
        for (int_t i = 0; i < block_length * block_length; i++)
        {
            max_origin_nnz = PANGULU_MAX(max_origin_nnz, block_origin_nnzA_num[i]);
        }

        int_t block_max_length = (sizeof(idx_int) + sizeof(calculate_type)) * max_origin_nnz;
        int_t tmp_length = PANGULU_MAX((unsigned int)block_max_length, (sizeof(idx_int)) * max_nnz);
        char *max_tmp = (char *)pangulu_malloc((block_length * 2 - 1) * (sizeof(int_t) * (NB + 1) + tmp_length) * preprocess_ompnum);
        
        int_t **tmp_rowpinter_all = (int_t **)pangulu_malloc(sizeof(int_t *) * (2 * block_length - 1) * preprocess_ompnum);
        idx_int **tmp_colindex_all = (idx_int **)pangulu_malloc(sizeof(idx_int *) * (2 * block_length - 1) * preprocess_ompnum);
        int_t index_array = 0;
        for (int_t i = 0; i < (2 * block_length - 1) * preprocess_ompnum; i++)
        {
            tmp_rowpinter_all[i] = (int_t *)(max_tmp + index_array);
            tmp_colindex_all[i] = (idx_int *)(max_tmp + index_array + sizeof(int_t) * (NB + 1));
            index_array += (sizeof(int_t) * (NB + 1) + (sizeof(idx_int)) * max_nnz);
        }
        for (int_t block_level = 0; block_level < block_length; block_level += preprocess_ompnum)
        {
            #pragma omp parallel num_threads(preprocess_ompnum)
            {
                int tid = omp_get_thread_num();

                int_t **tmp_rowpinter = tmp_rowpinter_all + (2 * block_length - 1) * tid;
                idx_int **tmp_colindex = tmp_colindex_all + (2 * block_length - 1) * tid;

                int_t now_level = tid + block_level;

                if (now_level < block_length)
                {
                    int_t row_min = now_level * NB;
                    int_t row_max = PANGULU_MIN(N, (now_level + 1) * NB);
                    
                    for (int_t i = 0; i < (2 * (block_length - now_level) - 1); i++)
                    {
                        int_t *now_tmp_rowpointer = tmp_rowpinter[i];
                        memset(now_tmp_rowpointer, 0, sizeof(int_t) * (NB + 1));
                    }
                    for (int_t now_col = row_min; now_col < row_max; now_col++)
                    {
                        for (int_t j = symbolic_rowpointer[now_col]; j < symbolic_rowpointer[now_col + 1]; j++)
                        {

                            int_t now_row = symbolic_columnindex[j];

                            int_t block_col = now_row / NB;

                            // add L
                            if (now_row == now_col)
                            {
                                // continue;
                            }
                            else if (now_row < row_max)
                            {
                                int_t *now_tmp_rowpointer = tmp_rowpinter[block_col - now_level];
                                if (preprocess_ompnum == 1)
                                    now_tmp_rowpointer[now_row % NB + 1]++;
                                else
                                {
                                    now_tmp_rowpointer[now_row % NB + 1]++;
                                }
                            }
                            else
                            {
                                int_t *now_tmp_rowpointer = tmp_rowpinter[block_col + block_length - 1 - 2 * now_level];
                                if (preprocess_ompnum == 1)
                                    now_tmp_rowpointer[now_row % NB + 1]++;
                                else
                                {
                                    now_tmp_rowpointer[now_row % NB + 1]++;
                                }
                            }
                        }
                    }

                    for (int_t now_row = row_min; now_row < row_max; now_row++)
                    {
                        for (int_t j = symbolic_rowpointer[now_row]; j < symbolic_rowpointer[now_row + 1]; j++)
                        {
                            int_t now_col = symbolic_columnindex[j];
                            int_t block_row = now_col / NB;
                            // add U
                            int_t *now_tmp_rowpointer = tmp_rowpinter[block_row - now_level];
                            now_tmp_rowpointer[now_row % NB + 1]++;
                        }
                    }

                    for (int_t i = 0; i < (2 * (block_length - now_level) - 1); i++)
                    {
                        int_t *now_tmp_rowpointer = tmp_rowpinter[i];
                        for (int_t j = 0; j < NB; j++)
                        {
                            now_tmp_rowpointer[j + 1] += now_tmp_rowpointer[j];
                        }
                    }
                    for (int_t now_col = row_min; now_col < row_max; now_col++)
                    {
                        for (int_t j = symbolic_rowpointer[now_col]; j < symbolic_rowpointer[now_col + 1]; j++)
                        {
                            int_t now_row = symbolic_columnindex[j];
                            int_t block_col = now_row / NB;
                            // add L
                            if (now_row == now_col)
                            {
                                // continue;
                            }
                            else if (now_row < row_max)
                            {
                                int_t *now_tmp_rowpointer = tmp_rowpinter[block_col - now_level];
                                idx_int *now_tmp_columnindex = tmp_colindex[block_col - now_level];
                                if (preprocess_ompnum == 1)
                                    now_tmp_columnindex[now_tmp_rowpointer[now_row % NB]++] = now_col % NB;
                                else
                                {
                                    int_t index;
                                    index = now_tmp_rowpointer[now_row % NB]++;
                                    now_tmp_columnindex[index] = now_col % NB;
                                }
                            }
                            else
                            {
                                int_t *now_tmp_rowpointer = tmp_rowpinter[block_col + block_length - 1 - 2 * now_level];
                                idx_int *now_tmp_columnindex = tmp_colindex[block_col + block_length - 1 - 2 * now_level];
                                if (preprocess_ompnum == 1)
                                    now_tmp_columnindex[now_tmp_rowpointer[now_row % NB]++] = now_col % NB;
                                else
                                {
                                    int_t index;
                                    index = now_tmp_rowpointer[now_row % NB]++;
                                    now_tmp_columnindex[index] = now_col % NB;
                                }
                            }
                        }
                    }

                    for (int_t now_row = row_min; now_row < row_max; now_row++)
                    {
                        for (int_t j = symbolic_rowpointer[now_row]; j < symbolic_rowpointer[now_row + 1]; j++)
                        {
                            int_t now_col = symbolic_columnindex[j];
                            int_t block_row = now_col / NB;
                            // add U
                            int_t *now_tmp_rowpointer = tmp_rowpinter[block_row - now_level];
                            idx_int *now_tmp_columnindex = tmp_colindex[block_row - now_level];
                            now_tmp_columnindex[now_tmp_rowpointer[now_row % NB]++] = now_col % NB;
                        }
                    }
                    for (int_t i = 0; i < (2 * (block_length - now_level) - 1); i++)
                    {
                        int_t *now_tmp_rowpointer = tmp_rowpinter[i];
                        for (int_t j = NB; j > 0; j--)
                        {
                            now_tmp_rowpointer[j] = now_tmp_rowpointer[j - 1];
                        }
                        now_tmp_rowpointer[0] = 0;
                    }
                }
            }
            req = 0;
            for (int_t now_level = block_level, k = 0; now_level < PANGULU_MIN(block_length, block_level + preprocess_ompnum); now_level++, k++)
            {

                int_t **tmp_rowpinter = tmp_rowpinter_all + (2 * block_length - 1) * k;
                
                for (int_t i = now_level; i < block_length; i++)
                {
                    int_t flag_index = now_level * block_length + i;
                    int_t length = sizeof(int_t) * (NB + 1) + (sizeof(idx_int)) * block_Smatrix_nnzA_num[flag_index];
                    int_t *now_tmp_rowpointer = tmp_rowpinter[i - now_level];
                    for (int_t j = 1; j < sum_rank_size; j++)
                    {
                        flag_index += block_length * block_length;
                        if (save_flag_block_num[flag_index] != 0)
                        {
                            pangulu_isend_vector_char_wait((char *)now_tmp_rowpointer, length, j, now_level * block_length + i, &Request[req++]);
                        }
                    }
                }
                
                // send L
                for (int_t i = now_level + 1; i < block_length; i++)
                {
                    int_t flag_index = i * block_length + now_level;
                    int_t length = sizeof(int_t) * (NB + 1) + (sizeof(idx_int)) * block_Smatrix_nnzA_num[flag_index];
                    int_t *now_tmp_rowpointer = tmp_rowpinter[i + block_length - 2 * now_level - 1];
                    for (int_t j = 1; j < sum_rank_size; j++)
                    {
                        flag_index += block_length * block_length;
                        if (save_flag_block_num[flag_index] != 0)
                        {
                            pangulu_isend_vector_char_wait((char *)now_tmp_rowpointer, length, j, i * block_length + now_level, &Request[req++]);
                        }
                    }
                }
                
                
            }
            
            for (int_t now_level = block_level, k = 0; now_level < PANGULU_MIN(block_length, block_level + preprocess_ompnum); now_level++, k++)
            {
                
                int_t **tmp_rowpinter = tmp_rowpinter_all + (2 * block_length - 1) * k;
                
                for (int_t i = now_level; i < block_length; i++)
                {
                    int_t block_row = now_level;
                    int_t block_col = i;
                    int_t flag_index = now_level * block_length + i;
                    int_t length = sizeof(int_t) * (NB + 1) + (sizeof(idx_int)) * block_Smatrix_nnzA_num[flag_index];
                    int_t *now_tmp_rowpointer = tmp_rowpinter[i - now_level];

                    if (save_flag_block_num[flag_index] != 0)
                    {
                        pangulu_Smatrix *tmp = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
                        pangulu_init_pangulu_Smatrix(tmp);

                        // new
                        int_t offset = sum_process_grid_num * (NB + 1) * sizeof(int_t) + sum_process_grid_nnz * (sizeof(idx_int) + sizeof(calculate_type));
                        memcpy(save_tmp + offset, now_tmp_rowpointer, length);

                        int_t *now_rowpinter = (int_t *)(save_tmp + offset);
                        idx_int *now_colindex = (idx_int *)(save_tmp + offset + (NB + 1) * sizeof(int_t));
                        calculate_type *now_value = (calculate_type *)(save_tmp + offset + (NB + 1) * sizeof(int_t) + now_rowpinter[NB] * sizeof(idx_int));
                        memset(now_value,now_rowpinter[NB]* sizeof(calculate_type),0.0);
                        // pangulu_sort_pangulu_matrix(NB,now_rowpinter,now_colindex);

                        tmp->nnz = now_rowpinter[NB];
                        tmp->rowpointer = now_rowpinter;
                        tmp->columnindex = now_colindex;
                        tmp->value = now_value;
                        tmp->row = NB;
                        tmp->column = NB;

                        sum_process_grid_nnz += now_rowpinter[NB];

                        mapper_A[block_row * block_length + block_col] = sum_process_grid_num;
                        Big_Smatrix_value[sum_process_grid_num] = tmp;

                        int_t now_rank = grid_process_id[(block_row % P) * Q + (block_col % Q)];
                        int_t flag = level_task_rank_id[(P * Q) * PANGULU_MIN(block_row, block_col) + now_rank];

                        if (flag == rank)
                        {
                            real_matrix_flag[sum_process_grid_num] = 1;
                        }
                        sum_process_grid_num++;
                    }
                }
                // get L
                for (int_t i = now_level + 1; i < block_length; i++)
                {
                    int_t block_row = i;
                    int_t block_col = now_level;
                    int_t flag_index = i * block_length + now_level;
                    int_t length = sizeof(int_t) * (NB + 1) + (sizeof(idx_int)) * block_Smatrix_nnzA_num[flag_index];
                    int_t *now_tmp_rowpointer = tmp_rowpinter[i + block_length - 2 * now_level - 1];

                    if (save_flag_block_num[flag_index] != 0)
                    {
                        pangulu_Smatrix *tmp = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
                        pangulu_init_pangulu_Smatrix(tmp);

                        // new
                        int_t offset = sum_process_grid_num * (NB + 1) * sizeof(int_t) + sum_process_grid_nnz * (sizeof(idx_int) + sizeof(calculate_type));
                        memcpy(save_tmp + offset, now_tmp_rowpointer, length);

                        int_t *now_rowpinter = (int_t *)(save_tmp + offset);
                        idx_int *now_colindex = (idx_int *)(save_tmp + offset + (NB + 1) * sizeof(int_t));
                        calculate_type *now_value = (calculate_type *)(save_tmp + offset + (NB + 1) * sizeof(int_t) + now_rowpinter[NB] * sizeof(idx_int));
                        memset(now_value,now_rowpinter[NB]* sizeof(calculate_type),0.0);
                        
                        // pangulu_sort_pangulu_matrix(NB,now_rowpinter,now_colindex);

                        tmp->nnz = now_rowpinter[NB];
                        tmp->rowpointer = now_rowpinter;
                        tmp->columnindex = now_colindex;
                        tmp->value = now_value;
                        tmp->row = NB;
                        tmp->column = NB;

                        sum_process_grid_nnz += now_rowpinter[NB];

                        mapper_A[block_row * block_length + block_col] = sum_process_grid_num;
                        Big_Smatrix_value[sum_process_grid_num] = tmp;

                        int_t now_rank = grid_process_id[(block_row % P) * Q + (block_col % Q)];
                        int_t flag = level_task_rank_id[(P * Q) * PANGULU_MIN(block_row, block_col) + now_rank];

                        if (flag == rank)
                        {
                            real_matrix_flag[sum_process_grid_num] = 1;
                        }
                        sum_process_grid_num++;
                    }
                }
            }
            pangulu_mpi_waitall(Request, req);
            
        }
        for (int_t block_row = 0; block_row < block_length; block_row++)
        {
            for (int_t block_col = 0; block_col < block_length; block_col++)
            {
                if (block_origin_nnzA_num[block_row * block_length + block_col] != 0)
                {
                    if (save_flag_block_num[block_row * block_length + block_col] != 0 && block_origin_nnzA_num[block_row * block_length + block_col] != 0)
                    {

                        int_t index = mapper_A[block_row * block_length + block_col];
                        pangulu_Smatrix *tmp = Big_Smatrix_value[index];
                        pangulu_sort_pangulu_matrix(NB, tmp->rowpointer, tmp->columnindex);
                    }
                }
            }
        }
        int_t *save_rowpointer = (int_t *)max_tmp;
        idx_int *save_colindex = (idx_int *)(max_tmp + sizeof(int_t) * (NB + 1));
        calculate_type *tmp_value = NULL;
        int_t *save_index = (int_t *)pangulu_malloc(sizeof(int_t) * N);
        for (int_t i = 0; i < N; i++)
        {
            save_index[i] = reorder_matrix->rowpointer[i];
        }
        for (int_t block_row = 0; block_row < block_length; block_row++)
        {
            for (int_t block_col = 0; block_col < block_length; block_col++)
            {
                if (block_origin_nnzA_num[block_row * block_length + block_col] != 0)
                {
                    tmp_value = (calculate_type *)(max_tmp + sizeof(int_t) * (NB + 1) + sizeof(idx_int) * block_origin_nnzA_num[block_row * block_length + block_col]);
                    int_t index_row_min = block_row * NB;
                    int_t index_row_max = PANGULU_MIN((block_row + 1) * NB, N);
                    int_t index_col_max = PANGULU_MIN((block_col + 1) * NB, N);
                    memset(max_tmp, 0, sizeof(int_t) * (NB + 1));

                    for (int_t index_row = index_row_min, i = 0; index_row < index_row_max; index_row++, i++)
                    {
                        int_t now_index = save_rowpointer[i];
                        int_t index_begin = save_index[index_row];
                        int_t col = reorder_matrix->columnindex[index_begin];
                        while ((col < index_col_max) && (index_begin < reorder_matrix->rowpointer[index_row + 1]))
                        {
                            tmp_value[now_index] = reorder_matrix->value[index_begin];
                            save_colindex[now_index++] = col % NB;

                            index_begin++;
                            col = reorder_matrix->columnindex[index_begin];
                        }
                        save_index[index_row] = index_begin;
                        save_rowpointer[i + 1] = now_index;
                    }

                    for (int_t i = index_row_max - block_row * NB; i < NB; i++)
                    {
                        save_rowpointer[i + 1] = save_rowpointer[i];
                    }

                    int_t length = sizeof(int_t) * (NB + 1) + (sizeof(idx_int) + sizeof(calculate_type)) * block_origin_nnzA_num[block_row * block_length + block_col];
                    int_t flag_index = block_row * block_length + block_col;
                    for (int_t i = 1; i < sum_rank_size; i++)
                    {
                        flag_index += block_length * block_length;
                        if (save_flag_block_num[flag_index] != 0 && block_origin_nnzA_num[block_row * block_length + block_col] != 0)
                        {
                            pangulu_send_vector_char(max_tmp, length, i, block_row * block_length + block_col);
                        }
                    }
                    if (save_flag_block_num[block_row * block_length + block_col] != 0 && block_origin_nnzA_num[block_row * block_length + block_col] != 0)
                    {
                        int_t index = mapper_A[block_row * block_length + block_col];
                        pangulu_Smatrix *tmp = Big_Smatrix_value[index];
                        for (int_t i = 0; i < NB; i++)
                        {
                            for (int_t j = tmp->rowpointer[i], k = save_rowpointer[i]; (j < tmp->rowpointer[i + 1]) && k < save_rowpointer[i + 1]; j++)
                            {
                                if (tmp->columnindex[j] == save_colindex[k])
                                {
                                    tmp->value[j] = tmp_value[k];
                                    k++;
                                }
                                else
                                {
                                    tmp->value[j] = 0.0;
                                }
                            }
                        }
                    }
                }
            }
        }
        free(tmp_rowpinter_all);
        free(tmp_colindex_all);
        free(Request);
        free(save_index);
        free(max_tmp);
        free(symbolic_rowpointer);
        free(symbolic_columnindex);
        block_Smatrix->symbolic_rowpointer=symbolic_rowpointer=NULL;
        block_Smatrix->symbolic_columnindex=symbolic_columnindex=NULL;
    }
    else
    {

        for (int_t now_level = 0; now_level < block_length; now_level++)
        {
            for (int_t i = now_level; i < block_length; i++)
            {
                int_t block_row = now_level;
                int_t block_col = i;
                int_t flag_index = now_level * block_length + i;
                int_t length = sizeof(int_t) * (NB + 1) + (sizeof(idx_int)) * block_Smatrix_nnzA_num[flag_index];

                if (save_flag_block_num[flag_index] != 0)
                {

                    int_t offset = sum_process_grid_num * (NB + 1) * sizeof(int_t) + sum_process_grid_nnz * (sizeof(idx_int) + sizeof(calculate_type));

                    pangulu_recv_vector_char(save_tmp + offset, length, 0, block_row * block_length + block_col);

                    pangulu_Smatrix *tmp = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
                    pangulu_init_pangulu_Smatrix(tmp);

                    int_t *now_rowpinter = (int_t *)(save_tmp + offset);
                    idx_int *now_colindex = (idx_int *)(save_tmp + offset + (NB + 1) * sizeof(int_t));
                    calculate_type *now_value = (calculate_type *)(save_tmp + offset + (NB + 1) * sizeof(int_t) + now_rowpinter[NB] * sizeof(idx_int));
                    memset(now_value,now_rowpinter[NB]* sizeof(calculate_type),0.0);
                        
                    tmp->nnz = now_rowpinter[NB];
                    tmp->rowpointer = now_rowpinter;
                    tmp->columnindex = now_colindex;
                    tmp->value = now_value;
                    tmp->row = NB;
                    tmp->column = NB;

                    sum_process_grid_nnz += now_rowpinter[NB];

                    mapper_A[block_row * block_length + block_col] = sum_process_grid_num;
                    Big_Smatrix_value[sum_process_grid_num] = tmp;

                    int_t now_rank = grid_process_id[(block_row % P) * Q + (block_col % Q)];
                    int_t flag = level_task_rank_id[(P * Q) * PANGULU_MIN(block_row, block_col) + now_rank];

                    if (flag == rank)
                    {
                        real_matrix_flag[sum_process_grid_num] = 1;
                    }
                    sum_process_grid_num++;
                }
            }
            // get L
            for (int_t i = now_level + 1; i < block_length; i++)
            {
                int_t block_row = i;
                int_t block_col = now_level;
                int_t flag_index = i * block_length + now_level;
                int_t length = sizeof(int_t) * (NB + 1) + (sizeof(idx_int)) * block_Smatrix_nnzA_num[flag_index];

                if (save_flag_block_num[flag_index] != 0)
                {
                    int_t offset = sum_process_grid_num * (NB + 1) * sizeof(int_t) + sum_process_grid_nnz * (sizeof(idx_int) + sizeof(calculate_type));

                    pangulu_recv_vector_char(save_tmp + offset, length, 0, block_row * block_length + block_col);

                    pangulu_Smatrix *tmp = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
                    pangulu_init_pangulu_Smatrix(tmp);

                    // new

                    int_t *now_rowpinter = (int_t *)(save_tmp + offset);
                    idx_int *now_colindex = (idx_int *)(save_tmp + offset + (NB + 1) * sizeof(int_t));
                    calculate_type *now_value = (calculate_type *)(save_tmp + offset + (NB + 1) * sizeof(int_t) + now_rowpinter[NB] * sizeof(idx_int));
                    memset(now_value,now_rowpinter[NB]* sizeof(calculate_type),0.0);
                        
                    // pangulu_sort_pangulu_matrix(NB,now_rowpinter,now_colindex);

                    tmp->nnz = now_rowpinter[NB];
                    tmp->rowpointer = now_rowpinter;
                    tmp->columnindex = now_colindex;
                    tmp->value = now_value;
                    tmp->row = NB;
                    tmp->column = NB;

                    sum_process_grid_nnz += now_rowpinter[NB];

                    mapper_A[block_row * block_length + block_col] = sum_process_grid_num;
                    Big_Smatrix_value[sum_process_grid_num] = tmp;

                    int_t now_rank = grid_process_id[(block_row % P) * Q + (block_col % Q)];
                    int_t flag = level_task_rank_id[(P * Q) * PANGULU_MIN(block_row, block_col) + now_rank];

                    if (flag == rank)
                    {
                        real_matrix_flag[sum_process_grid_num] = 1;
                    }
                    sum_process_grid_num++;
                }
            }
        }

        for (int_t block_row = 0; block_row < block_length; block_row++)
        {
            for (int_t block_col = 0; block_col < block_length; block_col++)
            {
                if (block_origin_nnzA_num[block_row * block_length + block_col] != 0)
                {
                    if (save_flag_block_num[block_row * block_length + block_col] != 0 && block_origin_nnzA_num[block_row * block_length + block_col] != 0)
                    {
                        int_t index = mapper_A[block_row * block_length + block_col];
                        pangulu_Smatrix *tmp = Big_Smatrix_value[index];
                        pangulu_sort_pangulu_matrix(NB, tmp->rowpointer, tmp->columnindex);
                    }
                }
            }
        }

        for (int_t i = 0; i < block_length * block_length; i++)
        {
            max_nnz = PANGULU_MAX(max_nnz, block_Smatrix_nnzA_num[i]);
        }
        int_t max_origin_nnz = 0;
        for (int_t i = 0; i < block_length * block_length; i++)
        {
            max_origin_nnz = PANGULU_MAX(max_origin_nnz, block_origin_nnzA_num[i]);
        }
        int_t tmp_length = PANGULU_MAX((sizeof(idx_int) + sizeof(calculate_type)) * max_origin_nnz, (sizeof(idx_int)) * max_nnz);
        char *max_tmp = (char *)pangulu_malloc(sizeof(int_t) * (NB + 1) + tmp_length);

        int_t *tmp_rowpinter = (int_t *)max_tmp;
        idx_int *tmp_colindex = (idx_int *)(max_tmp + sizeof(int_t) * (NB + 1));
        calculate_type *tmp_value = NULL;
        for (int_t block_row = 0; block_row < block_length; block_row++)
        {
            for (int_t block_col = 0; block_col < block_length; block_col++)
            {
                if (save_flag_block_num[block_row * block_length + block_col] != 0 && block_origin_nnzA_num[block_row * block_length + block_col] != 0)
                {
                    tmp_value = (calculate_type *)(max_tmp + sizeof(int_t) * (NB + 1) + sizeof(idx_int) * block_origin_nnzA_num[block_row * block_length + block_col]);
                    int_t length = sizeof(int_t) * (NB + 1) + (sizeof(idx_int) + sizeof(calculate_type)) * block_origin_nnzA_num[block_row * block_length + block_col];
                    pangulu_recv_vector_char(max_tmp, length, 0, block_row * block_length + block_col);

                    int_t index = mapper_A[block_row * block_length + block_col];
                    pangulu_Smatrix *tmp = Big_Smatrix_value[index];
                    for (int_t i = 0; i < NB; i++)
                    {
                        for (int_t j = tmp->rowpointer[i], k = tmp_rowpinter[i]; (j < tmp->rowpointer[i + 1]) && k < tmp_rowpinter[i + 1]; j++)
                        {
                            if (tmp->columnindex[j] == tmp_colindex[k])
                            {
                                tmp->value[j] = tmp_value[k];
                                k++;
                            }
                            else
                            {
                                tmp->value[j] = 0.0;
                            }
                        }
                    }
                }
            }
        }
        free(max_tmp);
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

    char *flag_save_L = (char *)pangulu_malloc(sizeof(char) * (L_Smatrix_nzz + U_Smatrix_nzz) * every_level_length);
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
        pangulu_Smatrix_add_more_memory(Big_Smatrix_value[i]);
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
    block_Smatrix->save_tmp = save_tmp;

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
