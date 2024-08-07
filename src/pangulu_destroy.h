#ifndef PANGULU_DESTROY_H
#define PANGULU_DESTROY_H

#include "pangulu_common.h"
#include "pangulu_heap.h"

#ifdef GPU_OPEN
#include "pangulu_cuda_interface.h"
#endif

#ifdef OVERLAP
#include "pangulu_thread.h"
#endif

pangulu_vector *pangulu_destroy_pangulu_vector(pangulu_vector *V)
{
    pangulu_free(__FILE__, __LINE__, V->value);
    V->value = NULL;
    pangulu_free(__FILE__, __LINE__, V);
    return NULL;
}

pangulu_Smatrix *pangulu_destroy_part_pangulu_origin_Smatrix(pangulu_origin_Smatrix *S)
{
    pangulu_free(__FILE__, __LINE__, S->rowpointer);
    pangulu_free(__FILE__, __LINE__, S->columnindex);
    pangulu_free(__FILE__, __LINE__, S->value);
    S->rowpointer = NULL;
    S->columnindex = NULL;
    S->value = NULL;
    pangulu_free(__FILE__, __LINE__, S);
    return NULL;
}

pangulu_Smatrix *pangulu_destroy_pangulu_Smatrix(pangulu_Smatrix *S)
{
    S->nnz = 0;
    S->row = 0;
    S->column = 0;
    if (S->columnpointer != NULL)
    {
        pangulu_free(__FILE__, __LINE__, S->columnpointer);
    }
    S->columnpointer = NULL;
    S->rowindex = NULL;
    S->value_CSC = NULL;
    if (S->rowpointer != NULL)
    {
        pangulu_free(__FILE__, __LINE__, S->rowpointer);
    }
    S->rowpointer = NULL;
    S->columnindex = NULL;
    S->value = NULL;
    if (S->bin_rowpointer != NULL)
    {
        pangulu_free(__FILE__, __LINE__, S->bin_rowpointer);
    }
    S->bin_rowpointer = NULL;
    if (S->bin_rowindex != NULL)
    {
        pangulu_free(__FILE__, __LINE__, S->bin_rowindex);
    }
    S->bin_rowindex = NULL;
    if (S->nnzU != NULL)
    {
        pangulu_free(__FILE__, __LINE__, S->nnzU);
    }
    S->nnzU = NULL;
    if (S->CSC_to_CSR_index != NULL)
    {
        pangulu_free(__FILE__, __LINE__, S->CSC_to_CSR_index);
    }
    S->CSC_to_CSR_index = NULL;
    if (S != NULL)
    {
        pangulu_free(__FILE__, __LINE__, S);
    }
    return NULL;
}

pangulu_Smatrix *pangulu_destroy_copy_pangulu_Smatrix(pangulu_Smatrix *S)
{
    S->nnz = 0;
    S->row = 0;
    S->column = 0;
    if (S->value_CSC != NULL)
    {
        pangulu_free(__FILE__, __LINE__, S->value_CSC);
    }
    if (S->value != NULL)
    {
        pangulu_free(__FILE__, __LINE__, S->value);
    }
    if (S != NULL)
    {
        pangulu_free(__FILE__, __LINE__, S);
    }
    return NULL;
}

pangulu_Smatrix *pangulu_destroy_big_pangulu_Smatrix(pangulu_Smatrix *S)
{
    S->nnz = 0;
    S->row = 0;
    S->column = 0;
    if (S->columnpointer != NULL)
    {
        pangulu_free(__FILE__, __LINE__, S->columnpointer);
    }
    S->columnpointer = NULL;
    S->rowindex = NULL;
    S->value_CSC = NULL;
    if (S->rowpointer != NULL)
    {
        pangulu_free(__FILE__, __LINE__, S->rowpointer);
    }
    S->rowpointer = NULL;
    if (S->columnindex != NULL)
    {
        pangulu_free(__FILE__, __LINE__, S->columnindex);
    }
    S->columnindex = NULL;
    if (S->value != NULL)
    {
        pangulu_free(__FILE__, __LINE__, S->value);
    }
    S->value = NULL;
    if (S->bin_rowpointer != NULL)
    {
        pangulu_free(__FILE__, __LINE__, S->bin_rowpointer);
    }
    S->bin_rowpointer = NULL;
    if (S->bin_rowindex != NULL)
    {
        pangulu_free(__FILE__, __LINE__, S->bin_rowindex);
    }
    S->bin_rowindex = NULL;
    if (S->nnzU != NULL)
    {
        pangulu_free(__FILE__, __LINE__, S->nnzU);
    }
    S->nnzU = NULL;
    if (S->CSC_to_CSR_index != NULL)
    {
        pangulu_free(__FILE__, __LINE__, S->CSC_to_CSR_index);
    }
    S->CSC_to_CSR_index = NULL;
    if (S != NULL)
    {
        pangulu_free(__FILE__, __LINE__, S);
    }
    return NULL;
}

pangulu_Smatrix *pangulu_destroy_calculate_pangulu_Smatrix_X(pangulu_Smatrix *S)
{
    S->nnz = 0;
    S->row = 0;
    S->column = 0;
    S->columnpointer = NULL;
    S->rowindex = NULL;
    if (S->value_CSC != NULL)
    {
        pangulu_free(__FILE__, __LINE__, S->value_CSC);
    }
    S->value_CSC = NULL;
    S->rowpointer = NULL;
    S->columnindex = NULL;
    if (S->value != NULL)
    {
        pangulu_free(__FILE__, __LINE__, S->value);
    }
    S->value = NULL;
    if (S != NULL)
    {
        pangulu_free(__FILE__, __LINE__, S);
    }
    return NULL;
}

pangulu_common *pangulu_destroy_pangulu_common(pangulu_common *common)
{
    pangulu_free(__FILE__, __LINE__, common->file_name);
    pangulu_free(__FILE__, __LINE__, common);
    return NULL;
}

#ifdef GPU_OPEN
void pangulu_destroy_cuda_memory_pangulu_Smatrix(pangulu_Smatrix *S)
{
    if (S->CUDA_rowpointer != NULL)
    {
        pangulu_cuda_free_interface(S->CUDA_rowpointer);
    }
    S->CUDA_rowpointer = NULL;
    if (S->CUDA_columnindex != NULL)
    {
        pangulu_cuda_free_interface(S->CUDA_columnindex);
    }
    S->CUDA_columnindex = NULL;
    if (S->CUDA_value != NULL)
    {
        pangulu_cuda_free_interface(S->CUDA_value);
    }
    S->CUDA_value = NULL;
    if (S->CUDA_bin_rowpointer != NULL)
    {
        pangulu_cuda_free_interface(S->CUDA_bin_rowpointer);
    }
    S->CUDA_bin_rowpointer = NULL;
    if (S->CUDA_bin_rowindex != NULL)
    {
        pangulu_cuda_free_interface(S->CUDA_bin_rowindex);
    }
    S->CUDA_bin_rowindex = NULL;
    if (S->CUDA_nnzU != NULL)
    {
        pangulu_cuda_free_interface(S->CUDA_nnzU);
    }
    S->CUDA_nnzU = NULL;
}
#else
void pangulu_destroy_Smatrix_level(pangulu_Smatrix *A)
{
    pangulu_free(__FILE__, __LINE__, A->level_size);
    A->level_size = NULL;
    pangulu_free(__FILE__, __LINE__, A->level_idx);
    A->level_idx = NULL;
    A->num_lev = 0;
}

#endif

void pangulu_destroy(pangulu_block_common *block_common,
                     pangulu_block_Smatrix *block_Smatrix)
{
    int_t block_length = block_common->block_length;
    int_t L_Smatrix_nzz = block_Smatrix->L_Smatrix_nzz;
    int_t U_Smatrix_nzz = block_Smatrix->U_Smatrix_nzz;
    int_t every_level_length = block_common->every_level_length;
    int_t *mapper_A = block_Smatrix->mapper_Big_pangulu_Smatrix;

#ifndef GPU_OPEN
    for (int_t i = 0; i < block_length; i++)
    {

        int_t now_offset = i * block_length + i;
        int_t now_mapperA_offset = mapper_A[now_offset];
        if (now_mapperA_offset != -1 && block_Smatrix->real_matrix_flag[now_mapperA_offset] == 1)
        {
            pangulu_destroy_Smatrix_level(&block_Smatrix->Big_pangulu_Smatrix_value[now_mapperA_offset]);
        }
    }
#endif
    for (int_t i = 0; i < block_length * block_length; i++)
    {
        if (mapper_A[i] != -1)
        {
#ifdef GPU_OPEN
            pangulu_destroy_cuda_memory_pangulu_Smatrix(&(block_Smatrix->Big_pangulu_Smatrix_value[mapper_A[i]]));
#endif
            if (block_Smatrix->Big_pangulu_Smatrix_copy_value[mapper_A[i]] != NULL)
            {
                block_Smatrix->Big_pangulu_Smatrix_copy_value[mapper_A[i]] = pangulu_destroy_copy_pangulu_Smatrix(block_Smatrix->Big_pangulu_Smatrix_copy_value[mapper_A[i]]);
            }
        }
    }
    pangulu_free(__FILE__, __LINE__, block_Smatrix->blocks_current_rank);

    for (int_t i = 0; i < L_Smatrix_nzz * every_level_length; i++)
    {
        block_Smatrix->L_pangulu_Smatrix_value[i] = pangulu_destroy_pangulu_Smatrix(block_Smatrix->L_pangulu_Smatrix_value[i]);
    }

    for (int_t i = 0; i < U_Smatrix_nzz * every_level_length; i++)
    {
        block_Smatrix->U_pangulu_Smatrix_value[i] = pangulu_destroy_pangulu_Smatrix(block_Smatrix->U_pangulu_Smatrix_value[i]);
    }

    for (int_t i = 0; i < block_length; i++)
    {
        int_t index = block_Smatrix->mapper_diagonal[i];
        if (index != -1 && block_Smatrix->diagonal_Smatrix_L[index] != NULL)
        {
            block_Smatrix->diagonal_Smatrix_L[index] = pangulu_destroy_pangulu_Smatrix(block_Smatrix->diagonal_Smatrix_L[index]);
        }
    }

    for (int_t i = 0; i < block_length; i++)
    {
        int_t index = block_Smatrix->mapper_diagonal[i];
        if (index != -1 && block_Smatrix->diagonal_Smatrix_U[index] != NULL)
        {
            block_Smatrix->diagonal_Smatrix_U[index] = pangulu_destroy_pangulu_Smatrix(block_Smatrix->diagonal_Smatrix_U[index]);
        }
    }

#ifdef GPU_OPEN
    pangulu_cuda_free_interface(CUDA_B_idx_COL);

    pangulu_cuda_free_interface(CUDA_TEMP_value);

    pangulu_destroy_cuda_memory_pangulu_Smatrix(block_Smatrix->calculate_X);

    pangulu_destroy_cuda_memory_pangulu_Smatrix(block_Smatrix->calculate_L);

    pangulu_destroy_cuda_memory_pangulu_Smatrix(block_Smatrix->calculate_U);
#endif

    block_Smatrix->calculate_X = pangulu_destroy_calculate_pangulu_Smatrix_X(block_Smatrix->calculate_X);

#ifdef OVERLAP
    block_Smatrix->run_bsem1 = pangulu_bsem_destory(block_Smatrix->run_bsem1);
    block_Smatrix->run_bsem2 = pangulu_bsem_destory(block_Smatrix->run_bsem2);
    block_Smatrix->heap->heap_bsem = pangulu_bsem_destory(block_Smatrix->heap->heap_bsem);
#endif

    block_Smatrix->heap = pangulu_destory_pangulu_heap(block_Smatrix->heap);

    pangulu_free(__FILE__, __LINE__, TEMP_A_value);
    pangulu_free(__FILE__, __LINE__, ssssm_col_ops_u);
    pangulu_free(__FILE__, __LINE__, ssssm_ops_pointer);
    pangulu_free(__FILE__, __LINE__, getrf_diagIndex_csc);
    pangulu_free(__FILE__, __LINE__, getrf_diagIndex_csr);

    pangulu_free(__FILE__, __LINE__, SSSSM_hash_LU);
    pangulu_free(__FILE__, __LINE__, SSSSM_flag_LU);
    pangulu_free(__FILE__, __LINE__, SSSSM_flag_L_row);
    pangulu_free(__FILE__, __LINE__, SSSSM_hash_L_row);
    pangulu_free(__FILE__, __LINE__, SSSSM_L_value);
    pangulu_free(__FILE__, __LINE__, SSSSM_U_value);
    pangulu_free(__FILE__, __LINE__, zip_rows);
    pangulu_free(__FILE__, __LINE__, zip_cols);
    pangulu_free(__FILE__, __LINE__, SSSSM_hash_U_col);

    pangulu_free(__FILE__, __LINE__, block_Smatrix->row_perm);
    block_Smatrix->row_perm = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->col_perm);
    block_Smatrix->col_perm = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->metis_perm);
    block_Smatrix->metis_perm = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->row_scale);
    block_Smatrix->row_scale = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->col_scale);
    block_Smatrix->col_scale = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->mapper_Big_pangulu_Smatrix);
    block_Smatrix->mapper_Big_pangulu_Smatrix = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->save_tmp);
    block_Smatrix->save_tmp = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->block_Smatrix_nnzA_num);
    block_Smatrix->block_Smatrix_nnzA_num = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->block_Smatrix_non_zero_vector_L);
    block_Smatrix->block_Smatrix_non_zero_vector_L = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->block_Smatrix_non_zero_vector_U);
    block_Smatrix->block_Smatrix_non_zero_vector_U = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->Big_pangulu_Smatrix_value);
    block_Smatrix->Big_pangulu_Smatrix_value = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->Big_pangulu_Smatrix_copy_value);
    block_Smatrix->Big_pangulu_Smatrix_copy_value = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->L_pangulu_Smatrix_columnpointer);
    block_Smatrix->L_pangulu_Smatrix_columnpointer = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->L_pangulu_Smatrix_rowindex);
    block_Smatrix->L_pangulu_Smatrix_rowindex = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->L_pangulu_Smatrix_value);
    block_Smatrix->L_pangulu_Smatrix_value = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->U_pangulu_Smatrix_rowpointer);
    block_Smatrix->U_pangulu_Smatrix_rowpointer = NULL;
    
    pangulu_free(__FILE__, __LINE__, block_Smatrix->U_pangulu_Smatrix_columnindex);
    block_Smatrix->U_pangulu_Smatrix_columnindex = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->U_pangulu_Smatrix_value);
    block_Smatrix->U_pangulu_Smatrix_value = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->mapper_diagonal);
    block_Smatrix->mapper_diagonal = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->diagonal_Smatrix_L);
    block_Smatrix->diagonal_Smatrix_L = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->diagonal_Smatrix_U);
    block_Smatrix->diagonal_Smatrix_U = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->mapper_LU);
    block_Smatrix->mapper_LU = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->task_flag_id);
    block_Smatrix->task_flag_id = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->task_level_num);
    block_Smatrix->task_level_num = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->now_level_L_length);
    block_Smatrix->now_level_L_length = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->now_level_U_length);
    block_Smatrix->now_level_U_length = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->save_now_level_L);
    block_Smatrix->save_now_level_L = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->save_now_level_L);
    block_Smatrix->save_now_level_L = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->send_flag);
    block_Smatrix->send_flag = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->send_diagonal_flag_L);
    block_Smatrix->send_diagonal_flag_L = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->grid_process_id);
    block_Smatrix->grid_process_id = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->send_diagonal_flag_U);
    block_Smatrix->send_diagonal_flag_U = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->save_send_rank_flag);
    block_Smatrix->save_send_rank_flag = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->level_task_rank_id);
    block_Smatrix->level_task_rank_id = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->real_matrix_flag);
    block_Smatrix->real_matrix_flag = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->sum_flag_block_num);
    block_Smatrix->sum_flag_block_num = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->level_index);
    block_Smatrix->level_index = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->level_index_reverse);
    block_Smatrix->level_index_reverse = NULL;
    
    pangulu_free(__FILE__, __LINE__, block_Smatrix->flag_save_L);
    block_Smatrix->flag_save_L=NULL;
    block_Smatrix->flag_save_U=NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->mapper_mpi);
    block_Smatrix->mapper_mpi = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->mapper_mpi_reverse);
    block_Smatrix->mapper_mpi_reverse = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->mpi_level_num);
    block_Smatrix->mpi_level_num = NULL;

    pangulu_free(__FILE__, __LINE__, block_Smatrix->flag_dignon_L);
    block_Smatrix->flag_dignon_L=NULL;
    pangulu_free(__FILE__, __LINE__, block_Smatrix->flag_dignon_U);
    block_Smatrix->flag_dignon_U=NULL;
    return ;    
    
    
}

#endif