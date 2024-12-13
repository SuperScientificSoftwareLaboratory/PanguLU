#include "pangulu_common.h"

pangulu_vector *pangulu_destroy_pangulu_vector(pangulu_vector *v)
{
    pangulu_free(__FILE__, __LINE__, v->value);
    v->value = NULL;
    pangulu_free(__FILE__, __LINE__, v);
    return NULL;
}

pangulu_smatrix *pangulu_destroy_part_pangulu_origin_smatrix(pangulu_origin_smatrix *s)
{
    pangulu_free(__FILE__, __LINE__, s->rowpointer);
    pangulu_free(__FILE__, __LINE__, s->columnindex);
    pangulu_free(__FILE__, __LINE__, s->value);
    s->rowpointer = NULL;
    s->columnindex = NULL;
    s->value = NULL;
    pangulu_free(__FILE__, __LINE__, s);
    return NULL;
}

pangulu_smatrix *pangulu_destroy_pangulu_smatrix(pangulu_smatrix *s)
{
    s->nnz = 0;
    s->row = 0;
    s->column = 0;
    if (s->columnpointer != NULL)
    {
        pangulu_free(__FILE__, __LINE__, s->columnpointer);
    }
    s->columnpointer = NULL;
    s->rowindex = NULL;
    s->value_csc = NULL;
    if (s->rowpointer != NULL)
    {
        pangulu_free(__FILE__, __LINE__, s->rowpointer);
    }
    s->rowpointer = NULL;
    s->columnindex = NULL;
    s->value = NULL;
    if (s->bin_rowpointer != NULL)
    {
        pangulu_free(__FILE__, __LINE__, s->bin_rowpointer);
    }
    s->bin_rowpointer = NULL;
    if (s->bin_rowindex != NULL)
    {
        pangulu_free(__FILE__, __LINE__, s->bin_rowindex);
    }
    s->bin_rowindex = NULL;
    if (s->nnzu != NULL)
    {
        pangulu_free(__FILE__, __LINE__, s->nnzu);
    }
    s->nnzu = NULL;
    if (s->csc_to_csr_index != NULL)
    {
        pangulu_free(__FILE__, __LINE__, s->csc_to_csr_index);
    }
    s->csc_to_csr_index = NULL;
    pangulu_free(__FILE__, __LINE__, s);
    return NULL;
}

pangulu_smatrix *pangulu_destroy_copy_pangulu_smatrix(pangulu_smatrix *s)
{
    s->nnz = 0;
    s->row = 0;
    s->column = 0;
    if (s->value_csc != NULL)
    {
        pangulu_free(__FILE__, __LINE__, s->value_csc);
    }
    if (s->value != NULL)
    {
        pangulu_free(__FILE__, __LINE__, s->value);
    }
    if (s != NULL)
    {
        pangulu_free(__FILE__, __LINE__, s);
    }
    return NULL;
}

pangulu_smatrix *pangulu_destroy_big_pangulu_smatrix(pangulu_smatrix *s)
{
    s->nnz = 0;
    s->row = 0;
    s->column = 0;
    if (s->columnpointer != NULL)
    {
        pangulu_free(__FILE__, __LINE__, s->columnpointer);
    }
    s->columnpointer = NULL;
    s->rowindex = NULL;
    s->value_csc = NULL;
    if (s->rowpointer != NULL)
    {
        pangulu_free(__FILE__, __LINE__, s->rowpointer);
    }
    s->rowpointer = NULL;
    if (s->columnindex != NULL)
    {
        pangulu_free(__FILE__, __LINE__, s->columnindex);
    }
    s->columnindex = NULL;
    if (s->value != NULL)
    {
        pangulu_free(__FILE__, __LINE__, s->value);
    }
    s->value = NULL;
    if (s->bin_rowpointer != NULL)
    {
        pangulu_free(__FILE__, __LINE__, s->bin_rowpointer);
    }
    s->bin_rowpointer = NULL;
    if (s->bin_rowindex != NULL)
    {
        pangulu_free(__FILE__, __LINE__, s->bin_rowindex);
    }
    s->bin_rowindex = NULL;
    if (s->nnzu != NULL)
    {
        pangulu_free(__FILE__, __LINE__, s->nnzu);
    }
    s->nnzu = NULL;
    if (s->csc_to_csr_index != NULL)
    {
        pangulu_free(__FILE__, __LINE__, s->csc_to_csr_index);
    }
    s->csc_to_csr_index = NULL;
    pangulu_free(__FILE__, __LINE__, s);
    return NULL;
}

pangulu_smatrix *pangulu_destroy_calculate_pangulu_smatrix_X(pangulu_smatrix *s)
{
    s->nnz = 0;
    s->row = 0;
    s->column = 0;
    s->columnpointer = NULL;
    s->rowindex = NULL;
    if (s->value_csc != NULL)
    {
        pangulu_free(__FILE__, __LINE__, s->value_csc);
    }
    s->value_csc = NULL;
    s->rowpointer = NULL;
    s->columnindex = NULL;
    if (s->value != NULL)
    {
        pangulu_free(__FILE__, __LINE__, s->value);
    }
    s->value = NULL;
    pangulu_free(__FILE__, __LINE__, s);
    return NULL;
}

pangulu_common *pangulu_destroy_pangulu_common(pangulu_common *common)
{
    pangulu_free(__FILE__, __LINE__, common->file_name);
    pangulu_free(__FILE__, __LINE__, common);
    return NULL;
}

#ifdef GPU_OPEN
void pangulu_destroy_cuda_memory_pangulu_smatrix(pangulu_smatrix *s)
{
    if (s->cuda_rowpointer != NULL)
    {
        pangulu_cuda_free_interface(s->cuda_rowpointer);
    }
    s->cuda_rowpointer = NULL;
    if (s->cuda_columnindex != NULL)
    {
        pangulu_cuda_free_interface(s->cuda_columnindex);
    }
    s->cuda_columnindex = NULL;
    if (s->cuda_value != NULL)
    {
        pangulu_cuda_free_interface(s->cuda_value);
    }
    s->cuda_value = NULL;
    if (s->cuda_bin_rowpointer != NULL)
    {
        pangulu_cuda_free_interface(s->cuda_bin_rowpointer);
    }
    s->cuda_bin_rowpointer = NULL;
    if (s->cuda_bin_rowindex != NULL)
    {
        pangulu_cuda_free_interface(s->cuda_bin_rowindex);
    }
    s->cuda_bin_rowindex = NULL;
    if (s->cuda_nnzu != NULL)
    {
        pangulu_cuda_free_interface(s->cuda_nnzu);
    }
    s->cuda_nnzu = NULL;
}
#else // GPU_OPEN
void pangulu_destroy_smatrix_level(pangulu_smatrix *a)
{
    pangulu_free(__FILE__, __LINE__, a->level_size);
    a->level_size = NULL;
    pangulu_free(__FILE__, __LINE__, a->level_idx);
    a->level_idx = NULL;
    a->num_lev = 0;
}

#endif // GPU_OPEN

void pangulu_destroy(pangulu_block_common *block_common,
                     pangulu_block_smatrix *block_smatrix)
{
    pangulu_int64_t block_length = block_common->block_length;
    pangulu_int64_t L_smatrix_nzz = block_smatrix->l_smatrix_nzz;
    pangulu_int64_t U_smatrix_nzz = block_smatrix->u_smatrix_nzz;
    pangulu_int64_t every_level_length = block_common->every_level_length;
    pangulu_block_info_pool* BIP = block_smatrix->BIP;

    for(pangulu_exblock_ptr i=0;i<block_smatrix->current_rank_block_count;i++){
        pangulu_free(__FILE__, __LINE__, block_smatrix->big_pangulu_smatrix_value[i].csc_to_csr_index);
        pangulu_free(__FILE__, __LINE__, block_smatrix->big_pangulu_smatrix_value[i].bin_rowpointer);
        pangulu_free(__FILE__, __LINE__, block_smatrix->big_pangulu_smatrix_value[i].bin_rowindex);
        pangulu_free(__FILE__, __LINE__, block_smatrix->big_pangulu_smatrix_value[i].rowpointer);
        pangulu_free(__FILE__, __LINE__, block_smatrix->big_pangulu_smatrix_value[i].columnindex);
        pangulu_free(__FILE__, __LINE__, block_smatrix->big_pangulu_smatrix_value[i].value);
    }

    if(block_smatrix->current_rank_block_count){
        pangulu_free(__FILE__, __LINE__, block_smatrix->big_pangulu_smatrix_value[0].columnpointer);
    }

    for(pangulu_exblock_ptr i=0;i<block_common->rank_col_length;i++){
        pangulu_destroy_pangulu_vector(block_smatrix->big_col_vector[i]);
        block_smatrix->big_col_vector[i] = NULL;
    }

    for(pangulu_exblock_ptr i=0;i<block_common->rank_row_length;i++){
        pangulu_destroy_pangulu_vector(block_smatrix->big_row_vector[i]);
        block_smatrix->big_row_vector[i] = NULL;
    }

#ifndef GPU_OPEN
    for (pangulu_int64_t i = 0; i < block_length; i++)
    {

        pangulu_int64_t now_offset = i * block_length + i;
        pangulu_int64_t now_mapperA_offset = pangulu_bip_get(now_offset, BIP)->mapper_a;
        if (now_mapperA_offset != -1)
        {
            pangulu_destroy_smatrix_level(&block_smatrix->big_pangulu_smatrix_value[now_mapperA_offset]);
        }
    }
#endif
    for (pangulu_int64_t i = 0; i < block_length * block_length; i++)
    {
        if (pangulu_bip_get(i, BIP)->mapper_a != -1)
        {
#ifdef GPU_OPEN
            pangulu_destroy_cuda_memory_pangulu_smatrix(&(block_smatrix->big_pangulu_smatrix_value[pangulu_bip_get(i, BIP)->mapper_a]));
#endif
        }
    }
    // pangulu_free(__FILE__, __LINE__, block_smatrix->blocks_current_rank);

    for (pangulu_int64_t i = 0; i < L_smatrix_nzz * every_level_length; i++)
    {
        block_smatrix->l_pangulu_smatrix_value[i] = pangulu_destroy_pangulu_smatrix(block_smatrix->l_pangulu_smatrix_value[i]);
    }

    for (pangulu_int64_t i = 0; i < U_smatrix_nzz * every_level_length; i++)
    {
        block_smatrix->u_pangulu_smatrix_value[i] = pangulu_destroy_pangulu_smatrix(block_smatrix->u_pangulu_smatrix_value[i]);
    }

    for (pangulu_int64_t i = 0; i < block_length; i++)
    {
        pangulu_int64_t index = block_smatrix->mapper_diagonal[i];
        if (index != -1 && block_smatrix->diagonal_smatrix_l[index] != NULL)
        {
            block_smatrix->diagonal_smatrix_l[index] = pangulu_destroy_pangulu_smatrix(block_smatrix->diagonal_smatrix_l[index]);
        }
    }

    for (pangulu_int64_t i = 0; i < block_length; i++)
    {
        pangulu_int64_t index = block_smatrix->mapper_diagonal[i];
        if (index != -1 && block_smatrix->diagonal_smatrix_u[index] != NULL)
        {
            block_smatrix->diagonal_smatrix_u[index] = pangulu_destroy_pangulu_smatrix(block_smatrix->diagonal_smatrix_u[index]);
        }
    }

    pangulu_free(__FILE__, __LINE__, block_smatrix->save_vector->value);
    pangulu_free(__FILE__, __LINE__, block_smatrix->save_vector);
    block_smatrix->save_vector = NULL;

#ifdef GPU_OPEN
    pangulu_cuda_free_interface(cuda_b_idx_col);

    pangulu_cuda_free_interface(cuda_temp_value);

    pangulu_destroy_cuda_memory_pangulu_smatrix(block_smatrix->calculate_x);

    pangulu_destroy_cuda_memory_pangulu_smatrix(block_smatrix->calculate_l);

    pangulu_destroy_cuda_memory_pangulu_smatrix(block_smatrix->calculate_u);
#endif

    block_smatrix->calculate_x = pangulu_destroy_calculate_pangulu_smatrix_X(block_smatrix->calculate_x);

#ifdef OVERLAP
    block_smatrix->run_bsem1 = pangulu_bsem_destory(block_smatrix->run_bsem1);
    block_smatrix->run_bsem2 = pangulu_bsem_destory(block_smatrix->run_bsem2);
    block_smatrix->heap->heap_bsem = pangulu_bsem_destory(block_smatrix->heap->heap_bsem);
#endif

    block_smatrix->heap = pangulu_destory_pangulu_heap(block_smatrix->heap);
    block_smatrix->sptrsv_heap = pangulu_destory_pangulu_heap(block_smatrix->sptrsv_heap);

    pangulu_free(__FILE__, __LINE__, temp_a_value);
    pangulu_free(__FILE__, __LINE__, ssssm_col_ops_u);
    pangulu_free(__FILE__, __LINE__, ssssm_ops_pointer);
    pangulu_free(__FILE__, __LINE__, getrf_diagIndex_csc);
    pangulu_free(__FILE__, __LINE__, getrf_diagIndex_csr);

    pangulu_free(__FILE__, __LINE__, ssssm_hash_lu);
    pangulu_free(__FILE__, __LINE__, ssssm_hash_l_row);
    pangulu_free(__FILE__, __LINE__, ssssm_l_value);
    pangulu_free(__FILE__, __LINE__, ssssm_u_value);
    pangulu_free(__FILE__, __LINE__, ssssm_hash_u_col);

    pangulu_free(__FILE__, __LINE__, block_smatrix->row_perm);
    block_smatrix->row_perm = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->col_perm);
    block_smatrix->col_perm = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->metis_perm);
    block_smatrix->metis_perm = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->row_scale);
    block_smatrix->row_scale = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->col_scale);
    block_smatrix->col_scale = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->save_tmp);
    block_smatrix->save_tmp = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->block_smatrix_non_zero_vector_l);
    block_smatrix->block_smatrix_non_zero_vector_l = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->block_smatrix_non_zero_vector_u);
    block_smatrix->block_smatrix_non_zero_vector_u = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->big_pangulu_smatrix_value);
    block_smatrix->big_pangulu_smatrix_value = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->l_pangulu_smatrix_columnpointer);
    block_smatrix->l_pangulu_smatrix_columnpointer = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->l_pangulu_smatrix_rowindex);
    block_smatrix->l_pangulu_smatrix_rowindex = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->l_pangulu_smatrix_value);
    block_smatrix->l_pangulu_smatrix_value = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->u_pangulu_smatrix_rowpointer);
    block_smatrix->u_pangulu_smatrix_rowpointer = NULL;
    
    pangulu_free(__FILE__, __LINE__, block_smatrix->u_pangulu_smatrix_columnindex);
    block_smatrix->u_pangulu_smatrix_columnindex = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->u_pangulu_smatrix_value);
    block_smatrix->u_pangulu_smatrix_value = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->mapper_diagonal);
    block_smatrix->mapper_diagonal = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->diagonal_smatrix_l);
    block_smatrix->diagonal_smatrix_l = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->diagonal_smatrix_u);
    block_smatrix->diagonal_smatrix_u = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->task_level_num);
    block_smatrix->task_level_num = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->now_level_l_length);
    block_smatrix->now_level_l_length = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->now_level_u_length);
    block_smatrix->now_level_u_length = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->save_now_level_l);
    block_smatrix->save_now_level_l = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->save_now_level_l);
    block_smatrix->save_now_level_l = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->send_flag);
    block_smatrix->send_flag = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->send_diagonal_flag_l);
    block_smatrix->send_diagonal_flag_l = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->grid_process_id);
    block_smatrix->grid_process_id = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->send_diagonal_flag_u);
    block_smatrix->send_diagonal_flag_u = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->save_send_rank_flag);
    block_smatrix->save_send_rank_flag = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->level_index);
    block_smatrix->level_index = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->level_index_reverse);
    block_smatrix->level_index_reverse = NULL;
    
    pangulu_free(__FILE__, __LINE__, block_smatrix->flag_save_l);
    block_smatrix->flag_save_l=NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->flag_save_u);
    block_smatrix->flag_save_u=NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->mapper_mpi_reverse);
    block_smatrix->mapper_mpi_reverse = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->mpi_level_num);
    block_smatrix->mpi_level_num = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->flag_dignon_l);
    block_smatrix->flag_dignon_l=NULL;
    
    pangulu_free(__FILE__, __LINE__, block_smatrix->flag_dignon_u);
    block_smatrix->flag_dignon_u=NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->save_now_level_l);
    block_smatrix->save_now_level_l = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->save_now_level_u);
    block_smatrix->save_now_level_u = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->calculate_l);
    block_smatrix->calculate_l = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->calculate_u);
    block_smatrix->calculate_u = NULL;
    
    pangulu_free(__FILE__, __LINE__, block_smatrix->calculate_x);
    block_smatrix->calculate_x = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->receive_level_num);
    block_smatrix->receive_level_num = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->big_col_vector);
    block_smatrix->big_col_vector = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->big_row_vector);
    block_smatrix->big_row_vector = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->diagonal_flag);
    block_smatrix->diagonal_flag = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->l_row_task_nnz);
    block_smatrix->l_row_task_nnz = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->l_col_task_nnz);
    block_smatrix->l_col_task_nnz = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->l_send_flag);
    block_smatrix->l_send_flag = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->u_send_flag);
    block_smatrix->u_send_flag = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->u_row_task_nnz);
    block_smatrix->u_row_task_nnz = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->u_col_task_nnz);
    block_smatrix->u_col_task_nnz = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->l_sptrsv_task_columnpointer);
    block_smatrix->l_sptrsv_task_columnpointer = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->l_sptrsv_task_rowindex);
    block_smatrix->l_sptrsv_task_rowindex = NULL;
    
    pangulu_free(__FILE__, __LINE__, block_smatrix->u_sptrsv_task_columnpointer);
    block_smatrix->u_sptrsv_task_columnpointer = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->u_sptrsv_task_rowindex);
    block_smatrix->u_sptrsv_task_rowindex = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->heap);
    block_smatrix->heap = NULL;

    pangulu_free(__FILE__, __LINE__, block_smatrix->sptrsv_heap);
    block_smatrix->sptrsv_heap = NULL;

    pangulu_free(__FILE__, __LINE__, TEMP_calculate_type);
    TEMP_calculate_type = NULL;
    TEMP_calculate_type_len = 0;

    pangulu_free(__FILE__, __LINE__, TEMP_pangulu_inblock_ptr);
    TEMP_pangulu_inblock_ptr = NULL;
    TEMP_pangulu_inblock_ptr_len = 0;

    pangulu_bip_destroy(&(block_smatrix->BIP));

    return ;   
}
