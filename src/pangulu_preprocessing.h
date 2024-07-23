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
#include "pangulu_utils.h"

#ifdef OVERLAP
#include "pangulu_thread.h"
#endif

int cmp_int32t_asc(const void *a, const void *b)
{
    return *(int32_t *)a - *(int32_t *)b;
}

void pangulu_preprocessing(pangulu_block_common *block_common,
                        pangulu_block_Smatrix *block_Smatrix,
                        pangulu_origin_Smatrix *reorder_matrix,
                        int_t nthread)
{
    const int_t preprocess_ompnum = nthread;
    int_t preprocess_ompnum_sort = preprocess_ompnum;
    int_t preprocess_ompnum_fill_columnindex = preprocess_ompnum;
    int_t preprocess_ompnum_set_value = preprocess_ompnum;
    int_t preprocess_ompnum_separate_block = preprocess_ompnum;
    int_t preprocess_ompnum_send_block = preprocess_ompnum;

    int_t N = block_common->N;
    int_32t rank = block_common->rank;
    int_32t P = block_common->P;
    int_32t Q = block_common->Q;
    int_32t NB = block_common->NB;

    int_t *symbolic_rowpointer = NULL;
    int_32t *symbolic_columnindex = NULL;

    int_32t block_length = block_common->block_length;

    // rank 0 only
    int_t block_num;
    char *block_csr;
    int_t *block_nnz_pt;
    // rank 0 only end

    if (rank == 0)
    {
        symbolic_rowpointer = block_Smatrix->symbolic_rowpointer;
        symbolic_columnindex = block_Smatrix->symbolic_columnindex;
    }


    int_t A_nnz_rowpointer_num = 0;
    int_32t sum_rank_size = block_common->sum_rank_size;

    int_t *level_task_rank_id = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (P * Q) * block_length);
    int_t *save_send_rank_flag = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * sum_rank_size);
    for (int_t i = 0; i < sum_rank_size; i++)
    {
        save_send_rank_flag[i] = 0;
    }
    int_t *block_origin_nnzA_num = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * block_length * block_length);
    for (int_t i = 0; i < block_length * block_length; i++)
    {
        block_origin_nnzA_num[i] = 0;
    }

    int_t *grid_process_id = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * P * Q);

    char *save_flag_block_num = NULL;
    int_t *save_block_Smatrix_nnzA_num = NULL;
    int_t *every_rank_block_num = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * sum_rank_size);
    int_t *every_rank_block_nnz = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * sum_rank_size);
    int_t sum_send_num = 0;

    int_t *block_Smatrix_nnzA_num = NULL;
    int_t *block_Smatrix_non_zero_vector_L = NULL;
    int_t *block_Smatrix_non_zero_vector_U = NULL;

    if (rank == 0)
    {
        block_Smatrix_nnzA_num = block_Smatrix->block_Smatrix_nnzA_num;
        block_Smatrix_non_zero_vector_L = block_Smatrix->block_Smatrix_non_zero_vector_L;
        block_Smatrix_non_zero_vector_U = block_Smatrix->block_Smatrix_non_zero_vector_U;
    }
    else
    {
        block_Smatrix_nnzA_num = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * block_length * block_length);
        block_Smatrix_non_zero_vector_L = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * block_length);
        block_Smatrix_non_zero_vector_U = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * block_length);
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

        save_block_Smatrix_nnzA_num = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * block_length * block_length);

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

        save_flag_block_num = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(char) * block_length * block_length * sum_rank_size);
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

    int_t current_rank_block_count = every_rank_block_num[rank];
    int_t current_rank_nnz_count = every_rank_block_nnz[rank];

    if (rank == 0)
    {
        for (int_t i = 1; i < sum_rank_size; i++)
        {
            pangulu_send_vector_char(save_flag_block_num + i * block_length * block_length, block_length * block_length, i, i);
        }
    }
    else
    {
        save_flag_block_num = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(char) * block_length * block_length);
        pangulu_recv_vector_char(save_flag_block_num, block_length * block_length, 0, rank);
    }
    int_t *sum_flag_block_num = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * block_length * block_length);

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
    pangulu_free(__FILE__, __LINE__, save_flag_block_num);
    pangulu_Bcast_vector(sum_flag_block_num, block_length * block_length, 0);

    int_t *mapper_A = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * block_length * block_length);
    for (int_t i = 0; i < block_length * block_length; i++)
    {
        mapper_A[i] = -1;
    }

    pangulu_Smatrix* Big_Smatrix_value = (pangulu_Smatrix*)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_Smatrix)*current_rank_block_count);
    for(int_t i=0;i<current_rank_block_count;i++){
        pangulu_init_pangulu_Smatrix(&Big_Smatrix_value[i]);
    }

    int_t *real_matrix_flag = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * current_rank_block_count);
    for (int_t i = 0; i < current_rank_block_count; i++)
    {
        real_matrix_flag[i] = 0;
    }

    // char *save_tmp = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (current_rank_block_count) * (NB + 1) + (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * current_rank_nnz_count);

    current_rank_block_count = 0;
    current_rank_nnz_count = 0;
    int_t max_nnz = 0;

    pangulu_exblock_idx nzblk_current_rank = 0; // 当前rank非零块数量
    pangulu_exblock_idx nnz_current_rank = 0; // 当前rank非零元数量
    char* blkcsr_current_rank = NULL; // 以块为单位的csr结构，矩阵中每个元素为块中非零元数量
    char* blocks_current_rank = NULL; // 一个一个地存储当前rank的块

    if (rank == 0)
    {

        struct timeval fill_start, fill_end;
        struct timeval fill_all_start, fill_all_end;

        // 将值填写到符号分解矩阵中
        gettimeofday(&fill_all_start, NULL);
        gettimeofday(&fill_start, NULL);
        int_t *symbolic_full_rowpointer = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (N + 1));
        int_32t *symbolic_full_columnindex = (int_32t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_32t) * block_Smatrix->symbolic_nnz);
        calculate_type *symbolic_full_value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * block_Smatrix->symbolic_nnz);
        int_t *symbolic_aid_array = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * N * (preprocess_ompnum_fill_columnindex + 1));
        memset(symbolic_aid_array, 0, sizeof(int_t) * N * (preprocess_ompnum_fill_columnindex + 1));
        memset(symbolic_full_rowpointer, 0, sizeof(int_t) * (N + 1));
        // memset(symbolic_full_value, 0, sizeof(calculate_type)*block_Smatrix->symbolic_nnz);
        gettimeofday(&fill_end, NULL);

        // 第一步 统计每列非零元数（不含对角元素），作为full每行非零元的一部分
        gettimeofday(&fill_start, NULL);
        for (int32_t row = 0; row < N; row++)
        {
            for (int_t j = symbolic_rowpointer[row]; j < symbolic_rowpointer[row + 1]; j++)
            {
                int32_t column = symbolic_columnindex[j];
                symbolic_aid_array[((row % preprocess_ompnum_fill_columnindex) + 1) * N + column]++;
                if (row != column)
                {
                    symbolic_full_rowpointer[column + 1]++;
                }
            }
        }
        gettimeofday(&fill_end, NULL);

        // 第二步 用symbolic_rowpointer补全symbolic_full_rowpointer，并做前缀和
        gettimeofday(&fill_start, NULL);
        for (int j = 0; j < N; j++)
        {
            for (int i = 1; i < preprocess_ompnum_fill_columnindex + 1; i++)
            {
                symbolic_aid_array[i * N + j] += symbolic_aid_array[(i - 1) * N + j];
            }
        }
        for (int_32t row = 1; row < N + 1; row++)
        {
            for (int i = 0; i < preprocess_ompnum_fill_columnindex + 1; i++)
            {
                symbolic_aid_array[i * N + (row - 1)] += symbolic_full_rowpointer[row - 1]; // row行下一个非零元index
            }
            symbolic_full_rowpointer[row] = symbolic_full_rowpointer[row] + symbolic_full_rowpointer[row - 1] + (symbolic_rowpointer[row] - symbolic_rowpointer[row - 1]);
        }
        gettimeofday(&fill_end, NULL);

        // 第三步 symbolic_columnindex每行前一半可直接搬到symbolic_full_columnindex
        gettimeofday(&fill_start, NULL);
        for (int_32t row = 0; row < N; row++)
        {
            memcpy(&symbolic_full_columnindex[symbolic_full_rowpointer[row + 1] - (symbolic_rowpointer[row + 1] - symbolic_rowpointer[row])], &symbolic_columnindex[symbolic_rowpointer[row]], sizeof(int_32t) * (symbolic_rowpointer[row + 1] - symbolic_rowpointer[row]));
        }
        gettimeofday(&fill_end, NULL);

        // 第四步 补全另一半symbolic_full_columnindex，上三角在哪列，下三角就在哪行
        gettimeofday(&fill_start, NULL);
#pragma omp parallel num_threads(preprocess_ompnum_fill_columnindex)
        {
            int tid = omp_get_thread_num();
            for (int_32t row = tid; row < N; row += preprocess_ompnum_fill_columnindex)
            {
                for (int_t j = symbolic_rowpointer[row]; j < symbolic_rowpointer[row + 1]; j++)
                {
                    int32_t column = symbolic_columnindex[j];
                    symbolic_full_columnindex[symbolic_aid_array[tid * N + column]++] = row;
                }
            }
        }
        gettimeofday(&fill_end, NULL);

        // 第五步 排序symbolic_full_columnindex
        gettimeofday(&fill_start, NULL);
        const int row_count_of_one_chunk = 500;
        int total_chunk_count = (N + row_count_of_one_chunk + 1) / row_count_of_one_chunk;
        int_32t *chunks_of_omp_rank = (int_32t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_32t) * total_chunk_count * preprocess_ompnum_sort);
        int_32t *chunk_count_of_omp_rank = (int_32t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_32t) * preprocess_ompnum_sort);
        int_t *omp_rank_total_nnz = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * preprocess_ompnum_sort);
        memset(chunk_count_of_omp_rank, 0, sizeof(int_32t) * preprocess_ompnum_sort);
        memset(omp_rank_total_nnz, 0, sizeof(int_t) * preprocess_ompnum_sort);
        for (int_32t chunk_id = 0; chunk_id < total_chunk_count; chunk_id++)
        {
            int_t chunk_nnz = symbolic_full_rowpointer[PANGULU_MIN((chunk_id + 1) * row_count_of_one_chunk, N)] - symbolic_full_rowpointer[chunk_id * row_count_of_one_chunk];
            int_t min_nnz = 0x7FFFFFFFFFFFFFFF;
            int min_nnz_rank = 0;
            for (int i = 0; i < preprocess_ompnum_sort; i++)
            {
                if (omp_rank_total_nnz[i] < min_nnz)
                {
                    min_nnz = omp_rank_total_nnz[i];
                    min_nnz_rank = i;
                }
            }
            omp_rank_total_nnz[min_nnz_rank] += chunk_nnz;
            chunks_of_omp_rank[min_nnz_rank * total_chunk_count + chunk_count_of_omp_rank[min_nnz_rank]] = chunk_id;
            chunk_count_of_omp_rank[min_nnz_rank]++;
        }
        gettimeofday(&fill_end, NULL);

        gettimeofday(&fill_start, NULL);
#pragma omp parallel num_threads(preprocess_ompnum_sort)
        {
            int omp_size = omp_get_num_threads();
            int omp_tid = omp_get_thread_num();
            for (int_32t i = 0; i < chunk_count_of_omp_rank[omp_tid]; i++)
            {
                for (int_32t row = chunks_of_omp_rank[omp_tid * total_chunk_count + i] * row_count_of_one_chunk; row < PANGULU_MIN((chunks_of_omp_rank[omp_tid * total_chunk_count + i] + 1) * row_count_of_one_chunk, N); row++)
                {
                    qsort(&symbolic_full_columnindex[symbolic_full_rowpointer[row]], symbolic_full_rowpointer[row + 1] - symbolic_full_rowpointer[row], sizeof(int_32t), cmp_int32t_asc);
                }
            }
        }
        gettimeofday(&fill_end, NULL);

        // 第六步 过一遍reorder_matrix，快慢指针补value
        gettimeofday(&fill_start, NULL);
#pragma omp parallel for num_threads(preprocess_ompnum_set_value)
        for (int_32t row = 0; row < N; row++)
        {
            int_t j_fast = symbolic_full_rowpointer[row];
            int_t j_fast_right_bound = symbolic_full_rowpointer[row + 1];
            for (int_t j_slow = reorder_matrix->rowpointer[row]; j_slow < reorder_matrix->rowpointer[row + 1]; j_slow++)
            {
                int_32t column = reorder_matrix->columnindex[j_slow];
                while (symbolic_full_columnindex[j_fast] != column)
                { // 没做鲁棒性
                    symbolic_full_value[j_fast] = 0.0;
                    j_fast++;
                }
                symbolic_full_value[j_fast] = reorder_matrix->value[j_slow];
                j_fast++;
            }
            while (j_fast < j_fast_right_bound)
            {
                symbolic_full_value[j_fast] = 0.0;
                j_fast++;
            }
            // memset(&symbolic_full_value[j_fast], 0, sizeof(calculate_type)*(j_fast_right_bound-j_fast));
        }
        gettimeofday(&fill_end, NULL);

        // 第七步 清理1
        gettimeofday(&fill_start, NULL);
        pangulu_free(__FILE__, __LINE__, chunks_of_omp_rank);
        pangulu_free(__FILE__, __LINE__, chunk_count_of_omp_rank);
        pangulu_free(__FILE__, __LINE__, omp_rank_total_nnz);
        pangulu_free(__FILE__, __LINE__, symbolic_aid_array);
        gettimeofday(&fill_end, NULL);

        gettimeofday(&fill_start, NULL);
        int_t nnz = symbolic_full_rowpointer[N];
        int bit_length = (block_length + 31) / 32;

        // ------负载均衡的预处理------
        // 计算出每个线程需要处理的非零元的数量
        int_t avg_nnz = (nnz + preprocess_ompnum_separate_block - 1) / preprocess_ompnum_separate_block;
        // 该数组记录大矩阵中每个行块第一个非零元的偏移
        int_t *block_row_nnz_pt = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (block_length + 1));
        for (int i = 0; i < block_length; i++)
        {
            block_row_nnz_pt[i] = symbolic_full_rowpointer[i * NB];
        }
        block_row_nnz_pt[block_length] = symbolic_full_rowpointer[N];
        // 根据avg_nnz，二分检索获取到每个线程需要计算的边界
        int *pt = (int *)pangulu_malloc(__FILE__, __LINE__, sizeof(int) * (preprocess_ompnum_separate_block + 1));
        pt[0] = 0;
        for (int i = 1; i < preprocess_ompnum_separate_block + 1; i++)
        {
            pt[i] = BinaryLowerBound(block_row_nnz_pt, block_length, avg_nnz * i);
        }

        // 该数组记录大矩阵中每个行块第一个非零块的偏移
        int_t *block_row_pt = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (block_length + 1));
        memset(block_row_pt, 0, sizeof(int_t) * (block_length + 1));

        unsigned int *bit_array = (unsigned int *)pangulu_malloc(__FILE__, __LINE__, sizeof(unsigned int) * bit_length * preprocess_ompnum_separate_block);

// step 1: get blocknum
#pragma omp parallel num_threads(preprocess_ompnum_separate_block)
        {
            int tid = omp_get_thread_num();
            unsigned int *tmp_bit = bit_array + bit_length * tid;

            for (int level = pt[tid]; level < pt[tid + 1]; level++)
            {
                memset(tmp_bit, 0, sizeof(unsigned int) * bit_length);

                int start_row = level * NB;
                int end_row = ((level + 1) * NB) < N ? ((level + 1) * NB) : N;

                for (int rid = start_row; rid < end_row; rid++)
                {
                    for (int_t idx = symbolic_full_rowpointer[rid]; idx < symbolic_full_rowpointer[rid + 1]; idx++)
                    {
                        idx_int colidx = symbolic_full_columnindex[idx];
                        idx_int block_cid = colidx / NB;
                        setbit(tmp_bit[block_cid / 32], block_cid % 32);
                    }
                }

                int_t tmp_blocknum = 0;
                for (int i = 0; i < bit_length; i++)
                {
                    tmp_blocknum += __builtin_popcount(tmp_bit[i]);
                }

                block_row_pt[level] = tmp_blocknum;
            }
        }
        exclusive_scan(block_row_pt, block_length + 1);
        block_num = block_row_pt[block_length]; // 大矩阵划分出的非零块的数量 //2

        // 该数组记录大矩阵中每个非零块中第一个非零元的偏移
        block_nnz_pt = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (block_num + 1)); // 3
        memset(block_nnz_pt, 0, sizeof(int_t) * (block_num + 1));
        // 该数组记录大矩阵中每个非零块的列索引
        idx_int *block_col_idx = (idx_int *)pangulu_malloc(__FILE__, __LINE__, sizeof(idx_int) * block_num);

        int *count_array = (int *)pangulu_malloc(__FILE__, __LINE__, sizeof(int) * block_length * preprocess_ompnum_separate_block);

// step 2: get nnz number in each block and colidx of each block
#pragma omp parallel num_threads(preprocess_ompnum_separate_block)
        {
            int tid = omp_get_thread_num();
            unsigned int *tmp_bit = bit_array + bit_length * tid;
            int *tmp_count = count_array + block_length * tid;

            for (int level = pt[tid]; level < pt[tid + 1]; level++)
            {
                memset(tmp_bit, 0, sizeof(unsigned int) * bit_length);
                memset(tmp_count, 0, sizeof(int) * block_length);

                int_t *cur_block_nnz_pt = block_nnz_pt + block_row_pt[level];
                idx_int *cur_block_col_idx = block_col_idx + block_row_pt[level];

                int start_row = level * NB;
                int end_row = ((level + 1) * NB) < N ? ((level + 1) * NB) : N;

                for (int rid = start_row; rid < end_row; rid++)
                {
                    for (int_t idx = symbolic_full_rowpointer[rid]; idx < symbolic_full_rowpointer[rid + 1]; idx++)
                    {
                        idx_int colidx = symbolic_full_columnindex[idx];
                        idx_int block_cid = colidx / NB;
                        setbit(tmp_bit[block_cid / 32], block_cid % 32);
                        tmp_count[block_cid]++;
                    }
                }

                int_t cnt = 0;
                for (int i = 0; i < block_length; i++)
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
        exclusive_scan(block_nnz_pt, block_num + 1);
        // 该数组为所有非零块的csr数据 //1
        block_csr = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (NB + 1) * block_num + sizeof(pangulu_inblock_idx) * nnz + sizeof(calculate_type) * nnz);

// step 3: get the complete csr of each block
#pragma omp parallel num_threads(preprocess_ompnum_separate_block)
        // for (int tid = 0; tid < 32; tid ++)
        {
            int tid = omp_get_thread_num();
            int *tmp_count = count_array + block_length * tid;

            for (int level = pt[tid]; level < pt[tid + 1]; level++)
            {

                memset(tmp_count, 0, sizeof(int) * block_length);

                for (int_t blc = block_row_pt[level]; blc < block_row_pt[level + 1]; blc++)
                {
                    int_t tmp_stride = blc * (NB + 1) * sizeof(pangulu_inblock_ptr) + block_nnz_pt[blc] * (sizeof(pangulu_inblock_idx) + sizeof(calculate_type));
                    pangulu_inblock_ptr *cur_csr_rpt = (pangulu_inblock_ptr *)(block_csr + tmp_stride);

                    memset(cur_csr_rpt, 0, sizeof(pangulu_inblock_ptr) * (NB + 1));
                }

                int start_row = level * NB;
                int end_row = ((level + 1) * NB) < N ? ((level + 1) * NB) : N;

                for (int rid = start_row, r_in_blc = 0; rid < end_row; rid++, r_in_blc++)
                {
                    int_t cur_block_idx = block_row_pt[level];

                    int_t arr_len = cur_block_idx * (NB + 1) * sizeof(pangulu_inblock_ptr) + block_nnz_pt[cur_block_idx] * (sizeof(pangulu_inblock_idx) + sizeof(calculate_type));
                    pangulu_inblock_ptr *cur_block_rowptr = (pangulu_inblock_ptr *)(block_csr + arr_len);
                    pangulu_inblock_idx *cur_block_colidx = (pangulu_inblock_idx *)(block_csr + arr_len + (NB + 1) * sizeof(pangulu_inblock_ptr));
                    calculate_type *cur_block_value = (calculate_type *)(block_csr + arr_len + (NB + 1) * sizeof(pangulu_inblock_ptr) + (block_nnz_pt[cur_block_idx + 1] - block_nnz_pt[cur_block_idx]) * sizeof(pangulu_inblock_idx));

                    for (int_t idx = symbolic_full_rowpointer[rid]; idx < symbolic_full_rowpointer[rid + 1]; idx++)
                    {
                        idx_int colidx = symbolic_full_columnindex[idx];
                        idx_int block_cid = colidx / NB;
                        if (block_col_idx[cur_block_idx] == block_cid)
                        {
                            cur_block_value[tmp_count[block_cid]] = symbolic_full_value[idx];
                            cur_block_colidx[tmp_count[block_cid]++] = colidx % NB;
                            cur_block_rowptr[r_in_blc]++;
                        }
                        else
                        {
                            cur_block_idx = BinarySearch(block_col_idx, cur_block_idx, block_row_pt[level + 1], block_cid);

                            arr_len = cur_block_idx * (NB + 1) * sizeof(pangulu_inblock_ptr) + block_nnz_pt[cur_block_idx] * (sizeof(pangulu_inblock_idx) + sizeof(calculate_type));
                            cur_block_rowptr = (pangulu_inblock_ptr *)(block_csr + arr_len);
                            cur_block_colidx = (pangulu_inblock_idx *)(block_csr + arr_len + (NB + 1) * sizeof(pangulu_inblock_ptr));
                            cur_block_value = (calculate_type *)(block_csr + arr_len + (NB + 1) * sizeof(pangulu_inblock_ptr) + (block_nnz_pt[cur_block_idx + 1] - block_nnz_pt[cur_block_idx]) * sizeof(pangulu_inblock_idx));

                            cur_block_value[tmp_count[block_cid]] = symbolic_full_value[idx];
                            cur_block_colidx[tmp_count[block_cid]++] = colidx % NB;
                            cur_block_rowptr[r_in_blc]++;
                        }
                    }
                }

                for (int_t blc = block_row_pt[level]; blc < block_row_pt[level + 1]; blc++)
                {
                    int_t tmp_stride = blc * (NB + 1) * sizeof(pangulu_inblock_ptr) + block_nnz_pt[blc] * (sizeof(pangulu_inblock_idx) + sizeof(calculate_type));
                    pangulu_inblock_ptr *cur_csr_rpt = (pangulu_inblock_ptr *)(block_csr + tmp_stride);
                    exclusive_scan(cur_csr_rpt, NB + 1);
                }
            }
        }

        gettimeofday(&fill_end, NULL);

        MPI_Barrier_asym(MPI_COMM_WORLD, 0, 1e5);

        gettimeofday(&fill_start, NULL);
        pangulu_exblock_idx* calculated_nzblk_count_each_rank = (pangulu_exblock_idx*)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx)*sum_rank_size);
        memset(calculated_nzblk_count_each_rank, 0, sizeof(pangulu_exblock_idx)*sum_rank_size);
        char* blkcsr_all_rank = (char*)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr)*(block_length+1)*sum_rank_size + (sizeof(pangulu_exblock_idx)/*blk_colidx*/ + sizeof(pangulu_inblock_ptr)/*blk_value(nnz_in_block)*/) * block_num);
        char** blkcsr_each_rank = (char**)pangulu_malloc(__FILE__, __LINE__, sizeof(char*)*sum_rank_size);
        int_t counted_block = 0;
        for(int i=0;i<sum_rank_size;i++){
            blkcsr_each_rank[i] = blkcsr_all_rank + sizeof(pangulu_exblock_ptr)*(block_length+1)*i + (sizeof(pangulu_exblock_idx)/*blk_colidx*/ + sizeof(pangulu_inblock_ptr)/*blk_value(nnz_in_block)*/) * counted_block;
            counted_block += every_rank_block_num[i];
        }

        for(int target_rank=0;target_rank<sum_rank_size;target_rank++){
            pangulu_exblock_ptr* blkcsr_rowptr = (pangulu_exblock_ptr*)blkcsr_each_rank[target_rank];
            blkcsr_rowptr[0] = 0;
        }
        for(int brow=0;brow<block_length;brow++){
            for(int_t bidx=block_row_pt[brow]; bidx<block_row_pt[brow+1]; bidx++){
                int bcol = block_col_idx[bidx];
                int target_rank = (brow%P)*Q + (bcol%Q);
                
                pangulu_exblock_idx* blkcsr_colidx = (pangulu_exblock_idx*)(blkcsr_each_rank[target_rank]+sizeof(pangulu_exblock_ptr)*(block_length+1));
                pangulu_inblock_ptr* blkcsr_value_blknnz = (pangulu_inblock_ptr*)(blkcsr_each_rank[target_rank]+sizeof(pangulu_exblock_ptr)*(block_length+1)+sizeof(pangulu_exblock_idx)*every_rank_block_num[target_rank]);
                
                pangulu_exblock_idx bidx_in_target_rank = calculated_nzblk_count_each_rank[target_rank];
                blkcsr_colidx[bidx_in_target_rank] = bcol;
                blkcsr_value_blknnz[bidx_in_target_rank] = block_nnz_pt[bidx+1] - block_nnz_pt[bidx];
                
                calculated_nzblk_count_each_rank[target_rank]++;
            }
            for(int target_rank=0;target_rank<sum_rank_size;target_rank++){
                pangulu_exblock_ptr* blkcsr_rowptr = (pangulu_exblock_ptr*)blkcsr_each_rank[target_rank];
                blkcsr_rowptr[brow+1] = calculated_nzblk_count_each_rank[target_rank];
            }
        }

        pangulu_exblock_idx* sent_block_count_each_rank = (pangulu_exblock_idx*)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx)*sum_rank_size);
        memset(sent_block_count_each_rank, 0, sizeof(pangulu_exblock_idx)*sum_rank_size);
#pragma omp parallel num_threads(preprocess_ompnum_send_block)
        {
            int tid = omp_get_thread_num();
            int nthread = omp_get_num_threads();
            // int each_thread_service_process_count = (sum_rank_size + nthread - 1) / nthread;
            for(int target_rank = tid; target_rank < sum_rank_size; target_rank += nthread){
                if(target_rank!=0){
                    MPI_Send(calculated_nzblk_count_each_rank+target_rank, 1, MPI_UNSIGNED_LONG_LONG, target_rank, 0xCAFE + 1, MPI_COMM_WORLD);
                    MPI_Send(every_rank_block_nnz+target_rank, 1, MPI_UNSIGNED_LONG_LONG, target_rank, 0xCAFE + 2, MPI_COMM_WORLD);
                    MPI_Send(blkcsr_each_rank[target_rank], sizeof(pangulu_exblock_ptr)*(block_length+1)+(sizeof(pangulu_exblock_idx)+sizeof(pangulu_inblock_ptr))*calculated_nzblk_count_each_rank[target_rank], MPI_CHAR, target_rank, 0xCAFE + 3, MPI_COMM_WORLD);
                }else{
                    nzblk_current_rank = calculated_nzblk_count_each_rank[target_rank];
                    nnz_current_rank = every_rank_block_nnz[target_rank];
                    // blkcsr_current_rank = (char*)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr)*(block_length+1)+(sizeof(pangulu_exblock_idx)+sizeof(pangulu_inblock_idx))*nzblk_current_rank);
                    blkcsr_current_rank = blkcsr_each_rank[0];
                    blocks_current_rank = (char*)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr)*(NB+1)*nzblk_current_rank + (sizeof(pangulu_inblock_idx)+sizeof(calculate_type))*nnz_current_rank);
                }
                int_t blocks_current_rank_receive_offset = 0; // only valid when target_rank==0
                for(pangulu_exblock_idx brow=0;brow<block_length;brow++){
                    if(brow%P != target_rank/Q){ // 进程网格brow%P行没有target_rank负责的块
                        continue;
                    }
                    for(pangulu_exblock_ptr bidx=block_row_pt[brow]; bidx<block_row_pt[brow+1]; bidx++){
                        pangulu_exblock_idx bcol = block_col_idx[bidx];
                        if(bcol%Q != target_rank%Q){ // 进程网格bcol%Q列没有target_rank负责的块
                            continue;
                        }
                        int_t block_byte_offset = sizeof(pangulu_inblock_ptr)*(bidx*(NB+1)) + (sizeof(pangulu_inblock_idx)+sizeof(calculate_type))*block_nnz_pt[bidx];
                        if(target_rank!=0){
                            MPI_Send(block_csr+block_byte_offset, sizeof(pangulu_inblock_ptr)*(NB+1)+(sizeof(pangulu_inblock_idx)+sizeof(calculate_type))*(block_nnz_pt[bidx+1] - block_nnz_pt[bidx]), MPI_CHAR, target_rank, sent_block_count_each_rank[target_rank], MPI_COMM_WORLD);
                            sent_block_count_each_rank[target_rank]++;
                        }else{
                            memcpy(blocks_current_rank+blocks_current_rank_receive_offset, block_csr+block_byte_offset, sizeof(pangulu_inblock_ptr)*(NB+1)+(sizeof(pangulu_inblock_idx)+sizeof(calculate_type))*(block_nnz_pt[bidx+1] - block_nnz_pt[bidx]));
                            
                            pangulu_Smatrix* current = &Big_Smatrix_value[current_rank_block_count];
                            current->row = NB;
                            current->column = NB;
                            current->rowpointer = (pangulu_inblock_ptr*)(blocks_current_rank+blocks_current_rank_receive_offset);
                            current->nnz = current->rowpointer[NB];
                            current->columnindex = (pangulu_inblock_idx*)(blocks_current_rank+blocks_current_rank_receive_offset+sizeof(pangulu_inblock_ptr)*(NB+1));
                            current->value = (calculate_type*)(blocks_current_rank+blocks_current_rank_receive_offset+sizeof(pangulu_inblock_ptr)*(NB+1)+sizeof(pangulu_inblock_idx)*current->nnz);

                            mapper_A[brow*block_length+bcol] = current_rank_block_count;
                            real_matrix_flag[current_rank_block_count] = 1;

                            blocks_current_rank_receive_offset += sizeof(pangulu_inblock_ptr)*(NB+1)+(sizeof(pangulu_inblock_idx)+sizeof(calculate_type))*(block_nnz_pt[bidx+1] - block_nnz_pt[bidx]);
                            current_rank_block_count++;
                        }
                    }
                }
            }
        }

        gettimeofday(&fill_end, NULL);

        // 第十步 清理2
        gettimeofday(&fill_start, NULL);
        pangulu_free(__FILE__, __LINE__, pt);
        pangulu_free(__FILE__, __LINE__, block_row_nnz_pt);
        pangulu_free(__FILE__, __LINE__, block_row_pt);
        pangulu_free(__FILE__, __LINE__, block_col_idx);
        pangulu_free(__FILE__, __LINE__, bit_array);
        pangulu_free(__FILE__, __LINE__, count_array);
        pangulu_free(__FILE__, __LINE__, block_csr);
        blkcsr_current_rank = (char*)pangulu_realloc(__FILE__, __LINE__, blkcsr_current_rank, sizeof(pangulu_exblock_ptr)*(block_length+1)+(sizeof(pangulu_exblock_idx)+sizeof(pangulu_inblock_ptr))*nzblk_current_rank);
        
        gettimeofday(&fill_end, NULL);

        gettimeofday(&fill_all_end, NULL);

        block_Smatrix->symbolic_full_rowpointer = symbolic_full_rowpointer;
        block_Smatrix->symbolic_full_columnindex = symbolic_full_columnindex;
        block_Smatrix->symbolic_full_value = symbolic_full_value;
    }
    else
    {
        MPI_Status mpi_stat;
        MPI_Barrier_asym(MPI_COMM_WORLD, 0, 1e5);
        MPI_Recv(&nzblk_current_rank, 1, MPI_UNSIGNED_LONG_LONG, 0, 0xCAFE + 1, MPI_COMM_WORLD, &mpi_stat);
        MPI_Recv(&nnz_current_rank, 1, MPI_UNSIGNED_LONG_LONG, 0, 0xCAFE + 2, MPI_COMM_WORLD, &mpi_stat);

        blkcsr_current_rank = (char*)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr)*(block_length+1)+(sizeof(pangulu_exblock_idx)+sizeof(pangulu_inblock_ptr))*nzblk_current_rank);
        MPI_Recv(blkcsr_current_rank, sizeof(pangulu_exblock_ptr)*(block_length+1)+(sizeof(pangulu_exblock_idx)+sizeof(pangulu_inblock_ptr))*nzblk_current_rank, MPI_CHAR, 0, 0xCAFE + 3, MPI_COMM_WORLD, &mpi_stat);
        blocks_current_rank = (char*)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr)*(NB+1)*nzblk_current_rank + (sizeof(pangulu_inblock_idx)+sizeof(calculate_type))*nnz_current_rank);

        pangulu_exblock_ptr* blkcsr_rowptr = (pangulu_exblock_ptr*)blkcsr_current_rank;
        pangulu_exblock_idx* blkcsr_colidx = (pangulu_exblock_idx*)(blkcsr_current_rank+sizeof(pangulu_exblock_ptr)*(block_length+1));
        pangulu_inblock_ptr* blkcsr_value_blknnz = (pangulu_inblock_ptr*)(blkcsr_current_rank+sizeof(pangulu_exblock_ptr)*(block_length+1)+sizeof(pangulu_exblock_idx)*nzblk_current_rank);

        int_t blocks_current_rank_receive_offset = 0;
        for(pangulu_exblock_idx brow = 0; brow<block_length; brow++){
            for(pangulu_exblock_ptr bidx = blkcsr_rowptr[brow]; bidx < blkcsr_rowptr[brow+1]; bidx++){
                pangulu_exblock_idx bcol = blkcsr_colidx[bidx];
                MPI_Recv(blocks_current_rank+blocks_current_rank_receive_offset, sizeof(pangulu_inblock_ptr)*(NB+1)+(sizeof(pangulu_inblock_idx)+sizeof(calculate_type))*blkcsr_value_blknnz[bidx], MPI_CHAR, 0, bidx, MPI_COMM_WORLD, &mpi_stat);

                pangulu_Smatrix* current = &Big_Smatrix_value[current_rank_block_count];
                current->row = NB;
                current->column = NB;
                current->rowpointer = (pangulu_inblock_ptr*)(blocks_current_rank+blocks_current_rank_receive_offset);
                current->nnz = current->rowpointer[NB];
                current->columnindex = (pangulu_inblock_idx*)(blocks_current_rank+blocks_current_rank_receive_offset+sizeof(pangulu_inblock_ptr)*(NB+1));
                current->value = (calculate_type*)(blocks_current_rank+blocks_current_rank_receive_offset+sizeof(pangulu_inblock_ptr)*(NB+1)+sizeof(pangulu_inblock_idx)*current->nnz);

                mapper_A[brow*block_length+bcol] = current_rank_block_count;
                real_matrix_flag[current_rank_block_count] = 1;

                blocks_current_rank_receive_offset += sizeof(pangulu_inblock_ptr)*(NB+1)+(sizeof(pangulu_inblock_idx)+sizeof(calculate_type))*blkcsr_value_blknnz[bidx];
                current_rank_block_count++;
            }
        }
    }

    
    pangulu_Smatrix **Big_pangulu_Smatrix_copy_value = (pangulu_Smatrix **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_Smatrix *) * current_rank_block_count);

    for (int_t i = 0; i < current_rank_block_count; i++)
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
                    pangulu_Smatrix *tmp = (pangulu_Smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_Smatrix));
                    pangulu_init_pangulu_Smatrix(tmp);
                    pangulu_malloc_pangulu_Smatrix_CSC_value(&Big_Smatrix_value[now_mapperA_offset], tmp);
                    pangulu_malloc_pangulu_Smatrix_CSR_value(&Big_Smatrix_value[now_mapperA_offset], tmp);
                    Big_pangulu_Smatrix_copy_value[now_mapperA_offset] = tmp;
                    pangulu_memcpy_zero_pangulu_Smatrix_CSC_value(tmp);
                    pangulu_memcpy_zero_pangulu_Smatrix_CSR_value(tmp);
                }
                else
                {
                    pangulu_memcpy_zero_pangulu_Smatrix_CSC_value(&Big_Smatrix_value[now_mapperA_offset]);
                }
            }
        }
    }

    int_t *tmp_save_block_num = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * block_length * block_length);

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

    int_t *U_rowpointer = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (block_length + 1));
    int_t *L_columnpointer = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (block_length + 1));

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

    int_t *U_columnindex = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * U_rowpointer[block_length]);
    int_t *L_rowindex = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * L_columnpointer[block_length]);

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
    int_t *level_index = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (block_length));
    int_t *level_index_reverse = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (block_length));

    for (int_t i = 0; i < block_length; i++)
    {
        level_index[i] = i;
    }

    // optimize begin
    int_t *tmp_diggonal_task_id = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (block_length));
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
    pangulu_free(__FILE__, __LINE__, tmp_diggonal_task_id);
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
    int_t *now_level_L_length = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * every_level_length);
    int_t *now_level_U_length = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * every_level_length);

    for (int_t i = 0; i < every_level_length; i++)
    {
        now_level_L_length[i] = 0;
        now_level_U_length[i] = 0;
    }

    int_t *save_now_level_L = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * L_Smatrix_nzz * every_level_length);
    int_t *save_now_level_U = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * U_Smatrix_nzz * every_level_length);

    int_t now_nnz_L = 0;
    int_t now_nnz_U = 0;
    int_t *mapper_LU = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * block_length * block_length);
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
    int_t *MAX_level_nnzL = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * L_Smatrix_nzz * every_level_length);
    int_t *MAX_level_nnzU = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * U_Smatrix_nzz * every_level_length);

    char *flag_save_L = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(char) * (L_Smatrix_nzz + U_Smatrix_nzz) * every_level_length);
    char *flag_save_U = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(char) * U_Smatrix_nzz * every_level_length);

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

    pangulu_Smatrix **U_value = (pangulu_Smatrix **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_Smatrix *) * U_Smatrix_nzz * every_level_length);
    pangulu_Smatrix **L_value = (pangulu_Smatrix **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_Smatrix *) * L_Smatrix_nzz * every_level_length);

    for (int_t i = 0; i < U_Smatrix_nzz * every_level_length; i++)
    {
        pangulu_Smatrix *first_U = (pangulu_Smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_Smatrix));
        pangulu_init_pangulu_Smatrix(first_U);
        pangulu_malloc_pangulu_Smatrix_nnz_CSC(first_U, NB, MAX_level_nnzU[i]);

#ifdef ADD_GPU_MEMORY
        pangulu_Smatrix_CUDA_memory_init(first_U, NB, MAX_level_nnzU[i]);
#endif
        U_value[i] = first_U;
    }
    for (int_t i = 0; i < L_Smatrix_nzz * every_level_length; i++)
    {
        pangulu_Smatrix *first_L = (pangulu_Smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_Smatrix));
        pangulu_init_pangulu_Smatrix(first_L);
        pangulu_malloc_pangulu_Smatrix_nnz_CSC(first_L, NB, MAX_level_nnzL[i]);

#ifdef ADD_GPU_MEMORY
        pangulu_Smatrix_CUDA_memory_init(first_L, NB, MAX_level_nnzL[i]);

#endif
        L_value[i] = first_L;
    }

    pangulu_free(__FILE__, __LINE__, MAX_level_nnzL);
    pangulu_free(__FILE__, __LINE__, MAX_level_nnzU);

    int_t MAX_all_nnzX = 0;

    for (int_t i = 0; i < current_rank_block_count; i++)
    {
        pangulu_Smatrix_add_more_memory(&Big_Smatrix_value[i]);
#ifdef GPU_OPEN
        pangulu_Smatrix_add_CUDA_memory(&Big_Smatrix_value[i]);
        pangulu_Smatrix_CUDA_memcpy_A(&Big_Smatrix_value[i]);
        pangulu_cuda_malloc((void **)&(Big_Smatrix_value[i].d_left_sum), ((Big_Smatrix_value[i].nnz)) * sizeof(calculate_type));
#endif

        MAX_all_nnzX = PANGULU_MAX(MAX_all_nnzX, Big_Smatrix_value[i].nnz);
    }

#ifndef GPU_OPEN

    int_t *work_space = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (4 * NB + 8));
    for (int_t i = 0; i < block_length; i++)
    {

        int_t now_offset = i * block_length + i;
        int_t now_mapperA_offset = mapper_A[now_offset];
        if (now_mapperA_offset != -1 && real_matrix_flag[now_mapperA_offset] == 1)
        {
            pangulu_malloc_Smatrix_level(&Big_Smatrix_value[now_mapperA_offset]);
            pangulu_init_level_array(&Big_Smatrix_value[now_mapperA_offset], work_space);
        }
    }
    pangulu_free(__FILE__, __LINE__, work_space);
#endif

    pangulu_Smatrix *calculate_L = (pangulu_Smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_Smatrix));
    pangulu_Smatrix *calculate_U = (pangulu_Smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_Smatrix));
    pangulu_Smatrix *calculate_X = (pangulu_Smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_Smatrix));

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
    int_t *mapper_diagonal_Smatrix = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * block_length);

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

    pangulu_Smatrix **diagonal_U = (pangulu_Smatrix **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_Smatrix *) * diagonal_nnz);
    pangulu_Smatrix **diagonal_L = (pangulu_Smatrix **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_Smatrix *) * diagonal_nnz);

    char *flag_dignon_L = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(char) * diagonal_nnz);
    char *flag_dignon_U = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(char) * diagonal_nnz);

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
                    pangulu_Smatrix *first_U = (pangulu_Smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_Smatrix));
                    pangulu_init_pangulu_Smatrix(first_U);
                    pangulu_get_pangulu_Smatrix_to_U(&Big_Smatrix_value[first_index], first_U, NB);
                    pangulu_Smatrix_add_CSC(first_U);
                    pangulu_Smatrix_add_memory_U(first_U);
#ifdef ADD_GPU_MEMORY
                    pangulu_Smatrix_CUDA_memory_init(first_U, first_U->row, first_U->nnz);
                    pangulu_Smatrix_add_CUDA_memory_U(first_U);
                    pangulu_Smatrix_CUDA_memcpy_nnzU(first_U, first_U);
                    pangulu_Smatrix_CUDA_memcpy_struct_CSC(first_U, first_U);
                    pangulu_cuda_malloc((void **)&(first_U->d_graphInDegree), NB * sizeof(int));
                    pangulu_cuda_malloc((void **)&(first_U->d_id_extractor), sizeof(int));
                    first_U->graphInDegree = (int *)pangulu_malloc(__FILE__, __LINE__, NB * sizeof(int));
#endif
                    diagonal_U[diagonal_index] = first_U;
                }
                if (diagonal_L[diagonal_index] == NULL)
                {
                    pangulu_Smatrix *first_L = (pangulu_Smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_Smatrix));
                    pangulu_init_pangulu_Smatrix(first_L);
                    pangulu_get_pangulu_Smatrix_to_L(&Big_Smatrix_value[first_index], first_L, NB);
                    pangulu_Smatrix_add_CSC(first_L);
#ifdef ADD_GPU_MEMORY
                    pangulu_Smatrix_CUDA_memory_init(first_L, first_L->row, first_L->nnz);
                    pangulu_Smatrix_CUDA_memcpy_struct_CSC(first_L, first_L);
                    pangulu_cuda_malloc((void **)&(first_L->d_graphInDegree), NB * sizeof(int));
                    pangulu_cuda_malloc((void **)&(first_L->d_id_extractor), sizeof(int));
                    first_L->graphInDegree = (int *)pangulu_malloc(__FILE__, __LINE__, NB * sizeof(int));
#endif
                    diagonal_L[diagonal_index] = first_L;
                }
            }
            else
            {
                if (diagonal_L[diagonal_index] == NULL)
                {
                    pangulu_Smatrix *first_L = (pangulu_Smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_Smatrix));
                    pangulu_init_pangulu_Smatrix(first_L);
                    pangulu_malloc_pangulu_Smatrix_nnz_CSC(first_L, NB, block_Smatrix_non_zero_vector_L[level]);

#ifdef ADD_GPU_MEMORY
                    pangulu_Smatrix_CUDA_memory_init(first_L, NB, first_L->nnz);
                    pangulu_cuda_malloc((void **)&(first_L->d_graphInDegree), NB * sizeof(int));
                    pangulu_cuda_malloc((void **)&(first_L->d_id_extractor), sizeof(int));
                    first_L->graphInDegree = (int *)pangulu_malloc(__FILE__, __LINE__, NB * sizeof(int));
#endif
                    diagonal_L[diagonal_index] = first_L;
                }
                if (diagonal_U[diagonal_index] == NULL)
                {
                    pangulu_Smatrix *first_U = (pangulu_Smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_Smatrix));
                    pangulu_init_pangulu_Smatrix(first_U);
                    pangulu_malloc_pangulu_Smatrix_nnz_CSR(first_U, NB, block_Smatrix_non_zero_vector_U[level]);

#ifdef ADD_GPU_MEMORY
                    pangulu_Smatrix_CUDA_memory_init(first_U, NB, first_U->nnz);
                    pangulu_cuda_malloc((void **)&(first_U->d_graphInDegree), NB * sizeof(int));
                    pangulu_cuda_malloc((void **)&(first_U->d_id_extractor), sizeof(int));
                    first_U->graphInDegree = (int *)pangulu_malloc(__FILE__, __LINE__, NB * sizeof(int));
#endif
                    diagonal_U[diagonal_index] = first_U;
                }
            }
        }
    }

    pangulu_free(__FILE__, __LINE__, tmp_save_block_num);

    int_t *task_flag_id = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * block_length * block_length);

    for (int_t i = 0; i < block_length * block_length; i++)
    {
        task_flag_id[i] = 0;
    }

    int_t task_level_length = block_length / every_level_length + (((block_length % every_level_length) == 0) ? 0 : 1);
    int_t *task_level_num = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * task_level_length);
    int_t *receive_level_num = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * task_level_length);

    for (int_t i = 0; i < task_level_length; i++)
    {
        task_level_num[i] = 0;
    }

    for (int_t i = 0; i < task_level_length; i++)
    {
        receive_level_num[i] = 0;
    }

    int_t *save_block_L = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * block_length);
    int_t *save_block_U = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * block_length);

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
    pangulu_free(__FILE__, __LINE__, save_block_L);
    pangulu_free(__FILE__, __LINE__, save_block_U);

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

    int_t *save_flag = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * sum_rank_size);
    for (int_t i = 0; i < sum_rank_size; i++)
    {
        save_flag[i] = -1;
    }

    int_t *save_task_level_flag = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * task_level_length);

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
    pangulu_free(__FILE__, __LINE__, save_flag);
    pangulu_free(__FILE__, __LINE__, save_task_level_flag);

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
    int_t *send_flag = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * current_rank_block_count * max_PQ);
    int_t *send_diagonal_flag_L = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * diagonal_nnz * max_PQ);
    int_t *send_diagonal_flag_U = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * diagonal_nnz * max_PQ);

    for (int_t i = 0; i < current_rank_block_count * max_PQ; i++)
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

    pangulu_heap *heap = (pangulu_heap *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_heap));
    pangulu_init_pangulu_heap(heap, max_task_length);

    int_t *mapper_mpi = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * block_length * (block_length + 1));
    for (int_t i = 0; i < block_length * block_length; i++)
    {
        mapper_mpi[i] = -1;
    }

    int_t *mpi_level_num = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * task_level_length);
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

    int_t *mapper_mpi_reverse = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * block_non_zero_length);

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

    TEMP_A_value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * NB * NB);

    SSSSM_L_value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * NB * NB);
    SSSSM_U_value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * NB * NB);
    zip_max_id = (block_common->N * block_common->every_level_length - 1) / NB + 1;
    zip_cols = (idx_int *)pangulu_malloc(__FILE__, __LINE__, sizeof(idx_int) * zip_max_id);
    zip_rows = (idx_int *)pangulu_malloc(__FILE__, __LINE__, sizeof(idx_int) * zip_max_id);
    SSSSM_hash_L_row = (idx_int *)pangulu_malloc(__FILE__, __LINE__, sizeof(idx_int) * (NB)*zip_max_id);
    SSSSM_hash_U_col = (idx_int *)pangulu_malloc(__FILE__, __LINE__, sizeof(idx_int) * (NB));
    SSSSM_hash_LU = (idx_int *)pangulu_malloc(__FILE__, __LINE__, sizeof(idx_int) * (NB)*zip_max_id);
    SSSSM_flag_LU = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(char) * (NB)*zip_max_id);
    SSSSM_flag_L_row = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(char) * (NB)*zip_max_id);

    ssssm_col_ops_u = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (NB + 1));
    getrf_diagIndex_csc = (idx_int *)pangulu_malloc(__FILE__, __LINE__, sizeof(idx_int) * (NB + 1));
    getrf_diagIndex_csr = (idx_int *)pangulu_malloc(__FILE__, __LINE__, sizeof(idx_int) * (NB + 1));
    int omp_threads_num = PANGU_OMP_NUM_THREADS;
    ssssm_ops_pointer = (idx_int *)pangulu_malloc(__FILE__, __LINE__, sizeof(idx_int) * (omp_threads_num + 1));
#ifdef GPU_OPEN

    pangulu_cuda_malloc((void **)&CUDA_TEMP_value, NB * NB * sizeof(calculate_type));
    pangulu_cuda_malloc((void **)&CUDA_B_idx_COL, NB * NB * sizeof(pangulu_inblock_idx));
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
    // block_Smatrix->save_tmp = save_tmp;
    block_Smatrix->blocks_current_rank = blocks_current_rank;
    block_Smatrix->save_tmp = NULL;

    block_Smatrix->level_index = level_index;
    block_Smatrix->level_index_reverse = level_index_reverse;

    block_Smatrix->mapper_mpi = mapper_mpi;
    block_Smatrix->mapper_mpi_reverse = mapper_mpi_reverse;
    block_Smatrix->mpi_level_num = mpi_level_num;

#ifdef OVERLAP

    block_Smatrix->run_bsem1 = (bsem *)pangulu_malloc(__FILE__, __LINE__, sizeof(bsem));
    block_Smatrix->run_bsem2 = (bsem *)pangulu_malloc(__FILE__, __LINE__, sizeof(bsem));
    pangulu_bsem_init(block_Smatrix->run_bsem1, 0);
    pangulu_bsem_init(block_Smatrix->run_bsem2, 0);

    heap->heap_bsem = (bsem *)pangulu_malloc(__FILE__, __LINE__, sizeof(bsem));
    pangulu_bsem_init(heap->heap_bsem, 0);
#endif

    return;
}

#endif
