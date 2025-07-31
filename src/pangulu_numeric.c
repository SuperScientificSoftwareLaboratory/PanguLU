#include "pangulu_common.h"

pangulu_task_t *working_task_buf = NULL;
pangulu_int64_t working_task_buf_capacity = 0;

void pangulu_numeric_receive_message(
    MPI_Status status,
    pangulu_block_common *block_common,
    pangulu_block_smatrix *block_smatrix)
{
    pangulu_exblock_idx block_length = block_common->block_length;
    pangulu_int32_t p = block_common->p;
    pangulu_int32_t q = block_common->q;
    pangulu_inblock_idx nb = block_common->nb;
    pangulu_task_queue_t *heap = block_smatrix->heap;
    pangulu_storage_t *storage = block_smatrix->storage;
    pangulu_int32_t rank = block_common->rank;
    pangulu_int32_t nproc = block_common->sum_rank_size;
    pangulu_exblock_ptr *related_nondiag_block_rowptr = block_smatrix->related_nondiag_block_rowptr;
    pangulu_exblock_idx *related_nondiag_block_colidx = block_smatrix->related_nondiag_block_colidx;
    pangulu_exblock_ptr *related_nondiag_block_csr_to_csc = block_smatrix->related_nondiag_block_csr_to_csc;
    pangulu_exblock_ptr *related_nondiag_block_colptr = block_smatrix->related_nondiag_block_colptr;
    pangulu_exblock_idx *related_nondiag_block_rowidx = block_smatrix->related_nondiag_block_rowidx;
    pangulu_uint64_t *related_nondiag_uniaddr = block_smatrix->related_nondiag_uniaddr;
    pangulu_uint32_t *nondiag_remain_task_count = block_smatrix->nondiag_remain_task_count;
    pangulu_uint32_t *diag_remain_task_count = block_smatrix->diag_remain_task_count;
    pangulu_uint64_t *diag_uniaddr = block_smatrix->diag_uniaddr;
    pangulu_exblock_ptr *related_nondiag_fstblk_idx_after_diag = block_smatrix->related_nondiag_fstblk_idx_after_diag;
    pangulu_exblock_ptr *related_nondiag_fstblk_idx_after_diag_csr = block_smatrix->related_nondiag_fstblk_idx_after_diag_csr;
    pangulu_exblock_idx bcol_pos = 0;
    pangulu_exblock_idx brow_pos = 0;

    int fetch_size;
    MPI_Get_count(&status, MPI_CHAR, &fetch_size);
    pangulu_uint64_t slot_addr = pangulu_storage_allocate_slot(storage, fetch_size);
    pthread_mutex_lock(block_smatrix->info_mutex);
    struct timeval start;
    pangulu_time_start(&start);
    pangulu_storage_slot_t *slot_recv = pangulu_storage_get_slot(storage, slot_addr);
    pangulu_cm_recv_block(&status, storage, slot_addr, block_length, nb, &bcol_pos, &brow_pos, related_nondiag_block_colptr, related_nondiag_block_rowidx, related_nondiag_uniaddr, diag_uniaddr);
    pangulu_exblock_idx level = PANGULU_MIN(bcol_pos, brow_pos);
    slot_recv->data_status = PANGULU_DATA_READY;
    if (bcol_pos == brow_pos)
    {
        if (slot_recv->is_upper)
        {
            for (pangulu_exblock_ptr bidx = related_nondiag_fstblk_idx_after_diag[level]; bidx < related_nondiag_block_colptr[level + 1]; bidx++)
            {
                pangulu_exblock_idx brow = related_nondiag_block_rowidx[bidx];
                pangulu_int32_t target_rank = (brow % p) * q + (level % q);
                if (target_rank == rank)
                {
                    if (nondiag_remain_task_count[bidx] == 1)
                    {
                        nondiag_remain_task_count[bidx]--;
                        pangulu_task_queue_push(
                            heap, brow, level, level, PANGULU_TASK_TSTRF, level,
                            pangulu_storage_get_slot(storage, related_nondiag_uniaddr[bidx]),
                            slot_recv, NULL, block_length, __FILE__, __LINE__);
                    }
                }
            }
        }
        else
        {
            for (pangulu_exblock_ptr bidx = related_nondiag_fstblk_idx_after_diag_csr[level]; bidx < related_nondiag_block_rowptr[level + 1]; bidx++)
            {
                pangulu_exblock_idx bcol = related_nondiag_block_colidx[bidx];
                pangulu_int32_t target_rank = (level % p) * q + (bcol % q);
                if (target_rank == rank)
                {
                    if (nondiag_remain_task_count[related_nondiag_block_csr_to_csc[bidx]] == 1)
                    {
                        nondiag_remain_task_count[related_nondiag_block_csr_to_csc[bidx]]--;
                        pangulu_task_queue_push(
                            heap, level, bcol, level, PANGULU_TASK_GESSM, level,
                            pangulu_storage_get_slot(storage, related_nondiag_uniaddr[related_nondiag_block_csr_to_csc[bidx]]),
                            slot_recv, NULL, block_length, __FILE__, __LINE__);
                    }
                }
            }
        }
    }
    else if (brow_pos > bcol_pos)
    {
        pangulu_exblock_ptr bidx_csr_diag = related_nondiag_fstblk_idx_after_diag_csr[level];
        pangulu_storage_slot_t *op2_updating_diag = NULL;
        for (pangulu_exblock_ptr bidx_csr = pangulu_binarysearch(related_nondiag_block_colidx, related_nondiag_block_rowptr[brow_pos], related_nondiag_block_rowptr[brow_pos + 1], bcol_pos) + 1;
             bidx_csr < related_nondiag_block_rowptr[brow_pos + 1]; bidx_csr++)
        {
            pangulu_exblock_idx bcol = related_nondiag_block_colidx[bidx_csr];
            pangulu_int32_t target_rank = (brow_pos % p) * q + (bcol % q);
            if (target_rank == rank)
            {
                while ((bidx_csr_diag < related_nondiag_block_rowptr[level + 1]) && (related_nondiag_block_colidx[bidx_csr_diag] < bcol))
                {
                    if (related_nondiag_block_colidx[bidx_csr_diag] == brow_pos)
                    {
                        op2_updating_diag = pangulu_storage_get_slot(storage, related_nondiag_uniaddr[related_nondiag_block_csr_to_csc[bidx_csr_diag]]);
                    }
                    bidx_csr_diag++;
                }
                if ((bidx_csr_diag < related_nondiag_block_rowptr[level + 1]) && (related_nondiag_block_colidx[bidx_csr_diag] == bcol))
                {
                    pangulu_storage_slot_t *ssssm_op2 = pangulu_storage_get_slot(storage, related_nondiag_uniaddr[related_nondiag_block_csr_to_csc[bidx_csr_diag]]);
                    if (ssssm_op2 && (ssssm_op2->data_status == PANGULU_DATA_READY))
                    {
                        nondiag_remain_task_count[related_nondiag_block_csr_to_csc[bidx_csr]]--;
                        pangulu_task_queue_push(
                            heap, brow_pos, bcol, level, PANGULU_TASK_SSSSM, level,
                            pangulu_storage_get_slot(storage, related_nondiag_uniaddr[related_nondiag_block_csr_to_csc[bidx_csr]]),
                            slot_recv, ssssm_op2, block_length, __FILE__, __LINE__);
                    }
                }
            }
        }
        while (bidx_csr_diag < related_nondiag_block_rowptr[level + 1])
        {
            if (related_nondiag_block_colidx[bidx_csr_diag] == brow_pos)
            {
                op2_updating_diag = pangulu_storage_get_slot(storage, related_nondiag_uniaddr[related_nondiag_block_csr_to_csc[bidx_csr_diag]]);
                break;
            }
            bidx_csr_diag++;
        }
        if (rank == (brow_pos % p) * q + (brow_pos % q))
        {
            pangulu_storage_slot_t *ssssm_opdst = pangulu_storage_get_diag(block_length, storage, diag_uniaddr[brow_pos]);
            if (op2_updating_diag && (op2_updating_diag->data_status == PANGULU_DATA_READY))
            {
                diag_remain_task_count[brow_pos]--;
                pangulu_task_queue_push(
                    heap, brow_pos, brow_pos, level, PANGULU_TASK_SSSSM, level,
                    ssssm_opdst,
                    slot_recv, op2_updating_diag, block_length, __FILE__, __LINE__);
            }
        }
    }
    else
    {
        pangulu_exblock_ptr bidx_diag = related_nondiag_fstblk_idx_after_diag[level];
        pangulu_storage_slot_t *op1_updating_diag = NULL;
        for (pangulu_exblock_ptr bidx = pangulu_binarysearch(related_nondiag_block_rowidx, related_nondiag_block_colptr[bcol_pos], related_nondiag_block_colptr[bcol_pos + 1], brow_pos) + 1;
             bidx < related_nondiag_block_colptr[bcol_pos + 1]; bidx++)
        {
            pangulu_exblock_idx brow = related_nondiag_block_rowidx[bidx];
            pangulu_int32_t target_rank = (brow % p) * q + (bcol_pos % q);
            if (target_rank == rank)
            {
                while ((bidx_diag < related_nondiag_block_colptr[level + 1]) && (related_nondiag_block_rowidx[bidx_diag] < brow))
                {
                    if (related_nondiag_block_rowidx[bidx_diag] == bcol_pos)
                    {
                        op1_updating_diag = pangulu_storage_get_slot(storage, related_nondiag_uniaddr[bidx_diag]);
                    }
                    bidx_diag++;
                }
                if ((bidx_diag < related_nondiag_block_colptr[level + 1]) && (related_nondiag_block_rowidx[bidx_diag] == brow))
                {
                    pangulu_storage_slot_t *ssssm_op1 = pangulu_storage_get_slot(storage, related_nondiag_uniaddr[bidx_diag]);
                    if (ssssm_op1 && (ssssm_op1->data_status == PANGULU_DATA_READY))
                    {
                        nondiag_remain_task_count[bidx]--;
                        pangulu_task_queue_push(
                            heap, brow, bcol_pos, level, PANGULU_TASK_SSSSM, level,
                            pangulu_storage_get_slot(storage, related_nondiag_uniaddr[bidx]),
                            ssssm_op1, slot_recv, block_length, __FILE__, __LINE__);
                    }
                }
            }
        }
        while (bidx_diag < related_nondiag_block_colptr[level + 1])
        {
            if (related_nondiag_block_rowidx[bidx_diag] == bcol_pos)
            {
                op1_updating_diag = pangulu_storage_get_slot(storage, related_nondiag_uniaddr[bidx_diag]);
                break;
            }
            bidx_diag++;
        }
        if (rank == (bcol_pos % p) * q + (bcol_pos % q))
        {
            pangulu_storage_slot_t *ssssm_opdst = pangulu_storage_get_diag(block_length, storage, diag_uniaddr[bcol_pos]);
            if (op1_updating_diag && (op1_updating_diag->data_status == PANGULU_DATA_READY))
            {
                diag_remain_task_count[bcol_pos]--;
                pangulu_task_queue_push(
                    heap, bcol_pos, bcol_pos, level, PANGULU_TASK_SSSSM, level,
                    ssssm_opdst,
                    op1_updating_diag, slot_recv, block_length, __FILE__, __LINE__);
            }
        }
    }
    pthread_mutex_unlock(block_smatrix->info_mutex);
}

int pangulu_execute_aggregated_ssssm(
    unsigned long long ntask,
    void *_task_descriptors,
    void *_extra_params)
{
    pangulu_task_t *tasks = (pangulu_task_t *)_task_descriptors;
    pangulu_numeric_thread_param *extra_params = (pangulu_numeric_thread_param *)_extra_params;
    pangulu_common *common = extra_params->pangulu_common;
    pangulu_block_common *block_common = extra_params->block_common;
    pangulu_block_smatrix *block_smatrix = extra_params->block_smatrix;
    pangulu_int32_t rank = block_common->rank;
    pangulu_int32_t nproc = block_common->sum_rank_size;
    pangulu_inblock_idx nb = block_common->nb;
    pangulu_int32_t p = block_common->p;
    pangulu_int32_t q = block_common->q;
    pangulu_task_queue_t *heap = block_smatrix->heap;
    pangulu_storage_t *storage = block_smatrix->storage;
    char *sent_rank_flag = block_smatrix->sent_rank_flag;
    pangulu_exblock_idx block_length = block_common->block_length;
    pangulu_exblock_ptr *related_nondiag_block_rowptr = block_smatrix->related_nondiag_block_rowptr;
    pangulu_exblock_idx *related_nondiag_block_colidx = block_smatrix->related_nondiag_block_colidx;
    pangulu_exblock_ptr *related_nondiag_block_csr_to_csc = block_smatrix->related_nondiag_block_csr_to_csc;
    pangulu_exblock_ptr *related_nondiag_block_colptr = block_smatrix->related_nondiag_block_colptr;
    pangulu_exblock_idx *related_nondiag_block_rowidx = block_smatrix->related_nondiag_block_rowidx;
    pangulu_uint64_t *related_nondiag_uniaddr = block_smatrix->related_nondiag_uniaddr;
    pangulu_uint32_t *nondiag_remain_task_count = block_smatrix->nondiag_remain_task_count;

    pangulu_hybrid_batched_interface(nb, ntask, tasks);
    pthread_mutex_lock(block_smatrix->info_mutex);
    for (pangulu_uint64_t i = 0; i < ntask; i++)
    {
        pangulu_task_t *task = &tasks[i];
        pangulu_int16_t kernel_id = task->kernel_id;
        pangulu_int64_t brow_task = task->row;
        pangulu_int64_t bcol_task = task->col;
        pangulu_int64_t level = task->task_level;
        if ((task->op1->brow_pos % p) * q + (task->op1->bcol_pos % q) != rank)
        {
            pangulu_exblock_ptr bidx_op1 = pangulu_binarysearch(related_nondiag_block_rowidx, related_nondiag_block_colptr[task->op1->bcol_pos], related_nondiag_block_colptr[task->op1->bcol_pos + 1], task->op1->brow_pos);
            nondiag_remain_task_count[bidx_op1]--;
            if (nondiag_remain_task_count[bidx_op1] == 0)
            {
                pangulu_storage_slot_queue_recycle(storage, &related_nondiag_uniaddr[bidx_op1]);
            }
        }
        if ((task->op2->brow_pos % p) * q + (task->op2->bcol_pos % q) != rank)
        {
            pangulu_exblock_ptr bidx_op2 = pangulu_binarysearch(related_nondiag_block_rowidx, related_nondiag_block_colptr[task->op2->bcol_pos], related_nondiag_block_colptr[task->op2->bcol_pos + 1], task->op2->brow_pos);
            nondiag_remain_task_count[bidx_op2]--;
            if (nondiag_remain_task_count[bidx_op2] == 0)
            {
                pangulu_storage_slot_queue_recycle(storage, &related_nondiag_uniaddr[bidx_op2]);
            }
        }
    }
    pthread_mutex_unlock(block_smatrix->info_mutex);
    return 0;
}

void pangulu_numeric_work_batched(
    pangulu_int64_t ntask,
    pangulu_task_t *tasks,
    pangulu_common *common,
    pangulu_block_common *block_common,
    pangulu_block_smatrix *block_smatrix)
{
    pangulu_int32_t rank = block_common->rank;
    pangulu_int32_t nproc = block_common->sum_rank_size;
    pangulu_inblock_idx nb = block_common->nb;
    pangulu_int32_t p = block_common->p;
    pangulu_int32_t q = block_common->q;
    pangulu_task_queue_t *heap = block_smatrix->heap;
    pangulu_storage_t *storage = block_smatrix->storage;
    char *sent_rank_flag = block_smatrix->sent_rank_flag;
    pangulu_exblock_idx block_length = block_common->block_length;
    pangulu_exblock_ptr *related_nondiag_block_rowptr = block_smatrix->related_nondiag_block_rowptr;
    pangulu_exblock_idx *related_nondiag_block_colidx = block_smatrix->related_nondiag_block_colidx;
    pangulu_exblock_ptr *related_nondiag_block_csr_to_csc = block_smatrix->related_nondiag_block_csr_to_csc;
    pangulu_exblock_ptr *related_nondiag_block_colptr = block_smatrix->related_nondiag_block_colptr;
    pangulu_exblock_idx *related_nondiag_block_rowidx = block_smatrix->related_nondiag_block_rowidx;
    pangulu_uint64_t *related_nondiag_uniaddr = block_smatrix->related_nondiag_uniaddr;
    pangulu_uint32_t *nondiag_remain_task_count = block_smatrix->nondiag_remain_task_count;
    pangulu_uint32_t *diag_remain_task_count = block_smatrix->diag_remain_task_count;
    pangulu_uint64_t *diag_uniaddr = block_smatrix->diag_uniaddr;
    pangulu_exblock_ptr *related_nondiag_fstblk_idx_after_diag = block_smatrix->related_nondiag_fstblk_idx_after_diag;
    pangulu_exblock_ptr *related_nondiag_fstblk_idx_after_diag_csr = block_smatrix->related_nondiag_fstblk_idx_after_diag_csr;
    pangulu_int64_t *aggregate_batch_tileid_capacity = &block_smatrix->aggregate_batch_tileid_capacity;
    pangulu_storage_slot_t ***aggregate_batch_tileid = &block_smatrix->aggregate_batch_tileid;

    if (ntask > *aggregate_batch_tileid_capacity)
    {
        *aggregate_batch_tileid_capacity = ntask;
        *aggregate_batch_tileid = pangulu_realloc(__FILE__, __LINE__, *aggregate_batch_tileid, sizeof(pangulu_storage_slot_t *) * *aggregate_batch_tileid_capacity);
    }
    pangulu_int64_t ntask_no_ssssm = 0;
    for (pangulu_int64_t i = 0; i < ntask; i++)
    {
        if (tasks[i].kernel_id != PANGULU_TASK_SSSSM)
        {
            (*aggregate_batch_tileid)[ntask_no_ssssm] = tasks[i].opdst;
            if (i != ntask_no_ssssm)
            {
                pangulu_task_t tmp = tasks[i];
                tasks[i] = tasks[ntask_no_ssssm];
                tasks[ntask_no_ssssm] = tmp;
            }
            ntask_no_ssssm++;
        }
    }

    pangulu_numeric_thread_param param;
    param.pangulu_common = common;
    param.block_common = block_common;
    param.block_smatrix = block_smatrix;
    pangulu_aggregate_task_compute_multi_tile(ntask_no_ssssm, *aggregate_batch_tileid, pangulu_execute_aggregated_ssssm, &param);

    pangulu_task_t *tasks_fetched = NULL;
    int ntasks_fetched = 0;
    while (ntask + ntasks_fetched > working_task_buf_capacity)
    {
        working_task_buf_capacity = (working_task_buf_capacity + ntask + ntasks_fetched) * 2;
        working_task_buf = pangulu_realloc(__FILE__, __LINE__, working_task_buf, sizeof(pangulu_task_t) * working_task_buf_capacity);
        tasks = working_task_buf;
    }
    for (int i = ntask - ntask_no_ssssm; i >= 0; i--)
    {
        tasks[ntask_no_ssssm + ntasks_fetched + i] = tasks[ntask_no_ssssm + i];
    }
    memcpy(tasks + ntask_no_ssssm, tasks_fetched, sizeof(pangulu_task_t) * ntasks_fetched);
    ntask_no_ssssm = ntask_no_ssssm + ntasks_fetched;
    ntask = ntask + ntasks_fetched;

    if (ntask_no_ssssm > 0)
    {
        pangulu_hybrid_batched_interface(nb, ntask_no_ssssm, tasks);
    }

    pangulu_int64_t ntask_fact = 0;
    pangulu_int64_t ntask_trsm = 0;
    for (pangulu_int64_t itask = 0; itask < ntask_no_ssssm; itask++)
    {
        pangulu_task_t *task = &tasks[itask];
        pangulu_int16_t kernel_id = task->kernel_id;
        if (kernel_id == PANGULU_TASK_GETRF)
        {
            ntask_fact++;
        }
        else if (kernel_id == PANGULU_TASK_TSTRF)
        {
            ntask_trsm++;
        }
        else if (kernel_id == PANGULU_TASK_GESSM)
        {
            ntask_trsm++;
        }
    }

    pthread_mutex_lock(block_smatrix->info_mutex);
    for (pangulu_int64_t itask = 0; itask < ntask; itask++)
    {
        pangulu_task_t *task = &tasks[itask];
        pangulu_int16_t kernel_id = task->kernel_id;
        pangulu_int64_t level = task->task_level;
        pangulu_int64_t brow_task = task->row;
        pangulu_int64_t bcol_task = task->col;
        if ((itask < ntask_no_ssssm) && (kernel_id == PANGULU_TASK_SSSSM))
        {
            continue;
        }
        memset(sent_rank_flag, 0, sizeof(char) * nproc);
        if (kernel_id == PANGULU_TASK_GETRF)
        {
            task->opdst->data_status = PANGULU_DATA_READY;
            if (task->opdst->related_block)
            {
                task->opdst->related_block->data_status = PANGULU_DATA_READY;
            }
            pangulu_storage_slot_t *slot_upper;
            pangulu_storage_slot_t *slot_lower;
            if (task->opdst->is_upper)
            {
                slot_upper = task->opdst;
            }
            else
            {
                slot_upper = task->opdst->related_block;
            }
            slot_lower = slot_upper->related_block;

            for (pangulu_exblock_ptr bidx = related_nondiag_fstblk_idx_after_diag[level]; bidx < related_nondiag_block_colptr[level + 1]; bidx++)
            {
                pangulu_exblock_idx brow = related_nondiag_block_rowidx[bidx];
                pangulu_int32_t target_rank = (brow % p) * q + (level % q);
                if (target_rank == rank)
                {
                    if (nondiag_remain_task_count[bidx] == 1)
                    {
                        nondiag_remain_task_count[bidx]--;
                        pangulu_task_queue_push(
                            heap, brow, level, level, PANGULU_TASK_TSTRF, level,
                            pangulu_storage_get_slot(storage, related_nondiag_uniaddr[bidx]),
                            slot_upper, NULL, block_length, __FILE__, __LINE__);
                    }
                }
                else
                {
                    if (sent_rank_flag[target_rank] == 0)
                    {
                        pangulu_cm_isend_block(slot_upper, nb, level, level, target_rank);
                        sent_rank_flag[target_rank] = 1;
                    }
                }
            }
            memset(sent_rank_flag, 0, sizeof(char) * nproc);
            for (pangulu_exblock_ptr bidx = related_nondiag_fstblk_idx_after_diag_csr[level]; bidx < related_nondiag_block_rowptr[level + 1]; bidx++)
            {
                pangulu_exblock_idx bcol = related_nondiag_block_colidx[bidx];
                pangulu_int32_t target_rank = (level % p) * q + (bcol % q);
                if (target_rank == rank)
                {
                    if (nondiag_remain_task_count[related_nondiag_block_csr_to_csc[bidx]] == 1)
                    {
                        nondiag_remain_task_count[related_nondiag_block_csr_to_csc[bidx]]--;
                        pangulu_task_queue_push(
                            heap, level, bcol, level, PANGULU_TASK_GESSM, level,
                            pangulu_storage_get_slot(storage, related_nondiag_uniaddr[related_nondiag_block_csr_to_csc[bidx]]),
                            slot_lower, NULL, block_length, __FILE__, __LINE__);
                    }
                }
                else
                {
                    if (sent_rank_flag[target_rank] == 0)
                    {
                        pangulu_cm_isend_block(slot_lower, nb, level, level, target_rank);
                        sent_rank_flag[target_rank] = 1;
                    }
                }
            }
        }
        else if (kernel_id == PANGULU_TASK_TSTRF)
        {
            task->opdst->data_status = PANGULU_DATA_READY;
            if ((level % p) * q + (level % q) != rank)
            {
                diag_remain_task_count[level]--;
                if (diag_remain_task_count[level] == 0)
                {
                    pangulu_storage_slot_queue_recycle_by_ptr(storage, task->op1->related_block);
                    task->op1->related_block = NULL;
                    pangulu_storage_slot_queue_recycle_by_ptr(storage, task->op1);
                    task->op1 = NULL;
                }
            }
            pangulu_exblock_ptr bidx_csr_diag = related_nondiag_fstblk_idx_after_diag_csr[level];
            pangulu_storage_slot_t *op2_updating_diag = NULL;
            for (pangulu_exblock_ptr bidx_csr = pangulu_binarysearch(related_nondiag_block_colidx, related_nondiag_block_rowptr[brow_task], related_nondiag_block_rowptr[brow_task + 1], bcol_task) + 1;
                 bidx_csr < related_nondiag_block_rowptr[brow_task + 1]; bidx_csr++)
            {
                pangulu_exblock_idx bcol = related_nondiag_block_colidx[bidx_csr];
                pangulu_int32_t target_rank = (brow_task % p) * q + (bcol % q);
                if (target_rank == rank)
                {
                    while ((bidx_csr_diag < related_nondiag_block_rowptr[level + 1]) && (related_nondiag_block_colidx[bidx_csr_diag] < bcol))
                    {
                        if (related_nondiag_block_colidx[bidx_csr_diag] == brow_task)
                        {
                            op2_updating_diag = pangulu_storage_get_slot(storage, related_nondiag_uniaddr[related_nondiag_block_csr_to_csc[bidx_csr_diag]]);
                        }
                        bidx_csr_diag++;
                    }
                    if ((bidx_csr_diag < related_nondiag_block_rowptr[level + 1]) && (related_nondiag_block_colidx[bidx_csr_diag] == bcol))
                    {
                        pangulu_storage_slot_t *ssssm_op2 = pangulu_storage_get_slot(storage, related_nondiag_uniaddr[related_nondiag_block_csr_to_csc[bidx_csr_diag]]);
                        if (ssssm_op2 && (ssssm_op2->data_status == PANGULU_DATA_READY))
                        {
                            nondiag_remain_task_count[related_nondiag_block_csr_to_csc[bidx_csr]]--;
                            pangulu_task_queue_push(
                                heap, brow_task, bcol, level, PANGULU_TASK_SSSSM, level,
                                pangulu_storage_get_slot(storage, related_nondiag_uniaddr[related_nondiag_block_csr_to_csc[bidx_csr]]),
                                task->opdst, ssssm_op2, block_length, __FILE__, __LINE__);
                        }
                    }
                }
                else
                {
                    if (sent_rank_flag[target_rank] == 0)
                    {
                        pangulu_cm_isend_block(task->opdst, nb, brow_task, bcol_task, target_rank);
                        sent_rank_flag[target_rank] = 1;
                    }
                }
            }
            while (bidx_csr_diag < related_nondiag_block_rowptr[level + 1])
            {
                if (related_nondiag_block_colidx[bidx_csr_diag] == brow_task)
                {
                    op2_updating_diag = pangulu_storage_get_slot(storage, related_nondiag_uniaddr[related_nondiag_block_csr_to_csc[bidx_csr_diag]]);
                    break;
                }
                bidx_csr_diag++;
            }
            if (rank == (brow_task % p) * q + (brow_task % q))
            {
                pangulu_storage_slot_t *ssssm_opdst = pangulu_storage_get_diag(block_length, storage, diag_uniaddr[brow_task]);
                if (op2_updating_diag && (op2_updating_diag->data_status == PANGULU_DATA_READY))
                {
                    diag_remain_task_count[brow_task]--;
                    pangulu_task_queue_push(
                        heap, brow_task, brow_task, level, PANGULU_TASK_SSSSM, level,
                        ssssm_opdst,
                        task->opdst, op2_updating_diag, block_length, __FILE__, __LINE__);
                }
            }
            else
            {
                if (sent_rank_flag[(brow_task % p) * q + (brow_task % q)] == 0)
                {
                    pangulu_cm_isend_block(task->opdst, nb, brow_task, bcol_task, (brow_task % p) * q + (brow_task % q));
                    sent_rank_flag[(brow_task % p) * q + (brow_task % q)] = 1;
                }
            }
        }
        else if (kernel_id == PANGULU_TASK_GESSM)
        {
            task->opdst->data_status = PANGULU_DATA_READY;
            if ((level % p) * q + (level % q) != rank)
            {
                diag_remain_task_count[level]--;
                if (diag_remain_task_count[level] == 0)
                {
                    pangulu_storage_slot_queue_recycle_by_ptr(storage, task->op1->related_block);
                    task->op1->related_block = NULL;
                    pangulu_storage_slot_queue_recycle_by_ptr(storage, task->op1);
                    task->op1 = NULL;
                }
            }
            pangulu_exblock_ptr bidx_diag = related_nondiag_fstblk_idx_after_diag[level];
            pangulu_storage_slot_t *op1_updating_diag = NULL;
            for (pangulu_exblock_ptr bidx = pangulu_binarysearch(related_nondiag_block_rowidx, related_nondiag_block_colptr[bcol_task], related_nondiag_block_colptr[bcol_task + 1], brow_task) + 1;
                 bidx < related_nondiag_block_colptr[bcol_task + 1]; bidx++)
            {
                pangulu_exblock_idx brow = related_nondiag_block_rowidx[bidx];
                pangulu_int32_t target_rank = (brow % p) * q + (bcol_task % q);
                if (target_rank == rank)
                {
                    while ((bidx_diag < related_nondiag_block_colptr[level + 1]) && (related_nondiag_block_rowidx[bidx_diag] < brow))
                    {
                        if (related_nondiag_block_rowidx[bidx_diag] == bcol_task)
                        {
                            op1_updating_diag = pangulu_storage_get_slot(storage, related_nondiag_uniaddr[bidx_diag]);
                        }
                        bidx_diag++;
                    }
                    if ((bidx_diag < related_nondiag_block_colptr[level + 1]) && (related_nondiag_block_rowidx[bidx_diag] == brow))
                    {
                        pangulu_storage_slot_t *ssssm_op1 = pangulu_storage_get_slot(storage, related_nondiag_uniaddr[bidx_diag]);
                        if (ssssm_op1 && (ssssm_op1->data_status == PANGULU_DATA_READY))
                        {
                            nondiag_remain_task_count[bidx]--;
                            pangulu_task_queue_push(
                                heap, brow, bcol_task, level, PANGULU_TASK_SSSSM, level,
                                pangulu_storage_get_slot(storage, related_nondiag_uniaddr[bidx]),
                                ssssm_op1, task->opdst, block_length, __FILE__, __LINE__);
                        }
                    }
                }
                else
                {
                    if (sent_rank_flag[target_rank] == 0)
                    {
                        pangulu_cm_isend_block(task->opdst, nb, brow_task, bcol_task, target_rank);
                        sent_rank_flag[target_rank] = 1;
                    }
                }
            }
            while (bidx_diag < related_nondiag_block_colptr[level + 1])
            {
                if (related_nondiag_block_rowidx[bidx_diag] == bcol_task)
                {
                    op1_updating_diag = pangulu_storage_get_slot(storage, related_nondiag_uniaddr[bidx_diag]);
                    break;
                }
                bidx_diag++;
            }
            if (rank == (bcol_task % p) * q + (bcol_task % q))
            {
                pangulu_storage_slot_t *ssssm_opdst = pangulu_storage_get_diag(block_length, storage, diag_uniaddr[bcol_task]);
                if (op1_updating_diag && (op1_updating_diag->data_status == PANGULU_DATA_READY))
                {
                    diag_remain_task_count[bcol_task]--;
                    pangulu_task_queue_push(
                        heap, bcol_task, bcol_task, level, PANGULU_TASK_SSSSM, level,
                        ssssm_opdst,
                        op1_updating_diag, task->opdst, block_length, __FILE__, __LINE__);
                }
            }
            else
            {
                if (sent_rank_flag[(bcol_task % p) * q + (bcol_task % q)] == 0)
                {
                    pangulu_cm_isend_block(task->opdst, nb, brow_task, bcol_task, (bcol_task % p) * q + (bcol_task % q));
                    sent_rank_flag[(bcol_task % p) * q + (bcol_task % q)] = 1;
                }
            }
        }
        else if (kernel_id == PANGULU_TASK_SSSSM)
        {
            if (brow_task == bcol_task)
            {
                if (diag_remain_task_count[brow_task] == 1)
                {
                    diag_remain_task_count[brow_task]--;
                    pangulu_task_queue_push(heap, brow_task, bcol_task, brow_task, PANGULU_TASK_GETRF, brow_task, task->opdst, NULL, NULL, block_length, __FILE__, __LINE__);
                }
            }
            else if (brow_task < bcol_task)
            {
                pangulu_exblock_ptr bidx_task = pangulu_binarysearch(related_nondiag_block_rowidx, related_nondiag_block_colptr[bcol_task], related_nondiag_block_colptr[bcol_task + 1], brow_task);
                if (nondiag_remain_task_count[bidx_task] == 1)
                {
                    pangulu_storage_slot_t *diag_block = pangulu_storage_get_diag(block_length, storage, diag_uniaddr[PANGULU_MIN(brow_task, bcol_task)]);
                    if (diag_block && (diag_block->data_status == PANGULU_DATA_READY))
                    {
                        if (diag_block->is_upper == 1)
                        {
                            diag_block = diag_block->related_block;
                        }
                        nondiag_remain_task_count[bidx_task]--;
                        pangulu_task_queue_push(
                            heap, brow_task, bcol_task, PANGULU_MIN(brow_task, bcol_task), PANGULU_TASK_GESSM,
                            PANGULU_MIN(brow_task, bcol_task), task->opdst, diag_block, NULL, block_length, __FILE__, __LINE__);
                    }
                }
            }
            else
            {
                pangulu_exblock_ptr bidx_task = pangulu_binarysearch(related_nondiag_block_rowidx, related_nondiag_block_colptr[bcol_task], related_nondiag_block_colptr[bcol_task + 1], brow_task);
                if (nondiag_remain_task_count[bidx_task] == 1)
                {
                    pangulu_storage_slot_t *diag_block = pangulu_storage_get_diag(block_length, storage, diag_uniaddr[PANGULU_MIN(brow_task, bcol_task)]);
                    if (diag_block && (diag_block->data_status == PANGULU_DATA_READY))
                    {
                        if (diag_block->is_upper == 0)
                        {
                            diag_block = diag_block->related_block;
                        }
                        nondiag_remain_task_count[bidx_task]--;
                        pangulu_task_queue_push(
                            heap, brow_task, bcol_task, PANGULU_MIN(brow_task, bcol_task), PANGULU_TASK_TSTRF,
                            PANGULU_MIN(brow_task, bcol_task), task->opdst, diag_block, NULL, block_length, __FILE__, __LINE__);
                    }
                }
            }
        }
    }
    pthread_mutex_unlock(block_smatrix->info_mutex);
}

void pangulu_numeric_work(
    pangulu_task_t *task,
    pangulu_common *common,
    pangulu_block_common *block_common,
    pangulu_block_smatrix *block_smatrix)
{
    pangulu_int16_t kernel_id = task->kernel_id;
    pangulu_int64_t brow_task = task->row;
    pangulu_int64_t bcol_task = task->col;
    pangulu_int64_t level = task->task_level;
    pangulu_int32_t rank = block_common->rank;
    pangulu_int32_t nproc = block_common->sum_rank_size;
    pangulu_inblock_idx nb = block_common->nb;
    pangulu_int32_t p = block_common->p;
    pangulu_int32_t q = block_common->q;
    pangulu_task_queue_t *heap = block_smatrix->heap;
    pangulu_storage_t *storage = block_smatrix->storage;
    char *sent_rank_flag = block_smatrix->sent_rank_flag;
    pangulu_exblock_idx block_length = block_common->block_length;
    pangulu_exblock_ptr *related_nondiag_block_rowptr = block_smatrix->related_nondiag_block_rowptr;
    pangulu_exblock_idx *related_nondiag_block_colidx = block_smatrix->related_nondiag_block_colidx;
    pangulu_exblock_ptr *related_nondiag_block_csr_to_csc = block_smatrix->related_nondiag_block_csr_to_csc;
    pangulu_exblock_ptr *related_nondiag_block_colptr = block_smatrix->related_nondiag_block_colptr;
    pangulu_exblock_idx *related_nondiag_block_rowidx = block_smatrix->related_nondiag_block_rowidx;
    pangulu_uint64_t *related_nondiag_uniaddr = block_smatrix->related_nondiag_uniaddr;
    pangulu_uint32_t *nondiag_remain_task_count = block_smatrix->nondiag_remain_task_count;
    pangulu_uint32_t *diag_remain_task_count = block_smatrix->diag_remain_task_count;
    pangulu_uint64_t *diag_uniaddr = block_smatrix->diag_uniaddr;
    pangulu_exblock_ptr *related_nondiag_fstblk_idx_after_diag = block_smatrix->related_nondiag_fstblk_idx_after_diag;
    pangulu_exblock_ptr *related_nondiag_fstblk_idx_after_diag_csr = block_smatrix->related_nondiag_fstblk_idx_after_diag_csr;

    memset(sent_rank_flag, 0, sizeof(char) * nproc);

    if (kernel_id != PANGULU_TASK_SSSSM)
    {
        pangulu_numeric_thread_param param;
        param.pangulu_common = common;
        param.block_common = block_common;
        param.block_smatrix = block_smatrix;
        pangulu_aggregate_task_compute(task->opdst, pangulu_execute_aggregated_ssssm, &param);
    }

    if (kernel_id == PANGULU_TASK_GETRF)
    {
        pangulu_getrf_interface(nb, task->opdst, 0);
        pthread_mutex_lock(block_smatrix->info_mutex);
        task->opdst->data_status = PANGULU_DATA_READY;
        if (task->opdst->related_block)
        {
            task->opdst->related_block->data_status = PANGULU_DATA_READY;
        }
        pangulu_storage_slot_t *slot_upper;
        pangulu_storage_slot_t *slot_lower;
        if (task->opdst->is_upper)
        {
            slot_upper = task->opdst;
        }
        else
        {
            slot_upper = task->opdst->related_block;
        }
        slot_lower = slot_upper->related_block;

        for (pangulu_exblock_ptr bidx = related_nondiag_fstblk_idx_after_diag[level]; bidx < related_nondiag_block_colptr[level + 1]; bidx++)
        {
            pangulu_exblock_idx brow = related_nondiag_block_rowidx[bidx];
            pangulu_int32_t target_rank = (brow % p) * q + (level % q);
            if (target_rank == rank)
            {
                if (nondiag_remain_task_count[bidx] == 1)
                {
                    nondiag_remain_task_count[bidx]--;
                    pangulu_task_queue_push(
                        heap, brow, level, level, PANGULU_TASK_TSTRF, level,
                        pangulu_storage_get_slot(storage, related_nondiag_uniaddr[bidx]),
                        slot_upper, NULL, block_length, __FILE__, __LINE__);
                }
            }
            else
            {
                if (sent_rank_flag[target_rank] == 0)
                {
                    pangulu_cm_isend_block(slot_upper, nb, level, level, target_rank);
                    sent_rank_flag[target_rank] = 1;
                }
            }
        }

        memset(sent_rank_flag, 0, sizeof(char) * nproc);
        for (pangulu_exblock_ptr bidx = related_nondiag_fstblk_idx_after_diag_csr[level]; bidx < related_nondiag_block_rowptr[level + 1]; bidx++)
        {
            pangulu_exblock_idx bcol = related_nondiag_block_colidx[bidx];
            pangulu_int32_t target_rank = (level % p) * q + (bcol % q);
            if (target_rank == rank)
            {
                if (nondiag_remain_task_count[related_nondiag_block_csr_to_csc[bidx]] == 1)
                {
                    nondiag_remain_task_count[related_nondiag_block_csr_to_csc[bidx]]--;
                    pangulu_task_queue_push(
                        heap, level, bcol, level, PANGULU_TASK_GESSM, level,
                        pangulu_storage_get_slot(storage, related_nondiag_uniaddr[related_nondiag_block_csr_to_csc[bidx]]),
                        slot_lower, NULL, block_length, __FILE__, __LINE__);
                }
            }
            else
            {
                if (sent_rank_flag[target_rank] == 0)
                {
                    pangulu_cm_isend_block(slot_lower, nb, level, level, target_rank);
                    sent_rank_flag[target_rank] = 1;
                }
            }
        }
        pthread_mutex_unlock(block_smatrix->info_mutex);
    }
    else if (kernel_id == PANGULU_TASK_TSTRF)
    {
        pangulu_tstrf_interface(nb, task->opdst, task->op1, 0);
        pthread_mutex_lock(block_smatrix->info_mutex);
        task->opdst->data_status = PANGULU_DATA_READY;
        if ((level % p) * q + (level % q) != rank)
        {
            diag_remain_task_count[level]--;
            if (diag_remain_task_count[level] == 0)
            {
                pangulu_storage_slot_queue_recycle_by_ptr(storage, task->op1->related_block);
                task->op1->related_block = NULL;
                pangulu_storage_slot_queue_recycle_by_ptr(storage, task->op1);
                task->op1 = NULL;
            }
        }
        pangulu_exblock_ptr bidx_csr_diag = related_nondiag_fstblk_idx_after_diag_csr[level];
        for (pangulu_exblock_ptr bidx_csr = pangulu_binarysearch(related_nondiag_block_colidx, related_nondiag_block_rowptr[brow_task], related_nondiag_block_rowptr[brow_task + 1], bcol_task) + 1;
             bidx_csr < related_nondiag_block_rowptr[brow_task + 1]; bidx_csr++)
        {
            pangulu_exblock_idx bcol = related_nondiag_block_colidx[bidx_csr];
            pangulu_int32_t target_rank = (brow_task % p) * q + (bcol % q);
            if (target_rank == rank)
            {
                while ((bidx_csr_diag < related_nondiag_block_rowptr[level + 1]) && (related_nondiag_block_colidx[bidx_csr_diag] < bcol))
                {
                    bidx_csr_diag++;
                }
                if ((bidx_csr_diag < related_nondiag_block_rowptr[level + 1]) && (related_nondiag_block_colidx[bidx_csr_diag] == bcol))
                {
                    pangulu_storage_slot_t *ssssm_op2 = pangulu_storage_get_slot(storage, related_nondiag_uniaddr[related_nondiag_block_csr_to_csc[bidx_csr_diag]]);
                    if (ssssm_op2 && (ssssm_op2->data_status == PANGULU_DATA_READY))
                    {
                        nondiag_remain_task_count[related_nondiag_block_csr_to_csc[bidx_csr]]--;
                        pangulu_task_queue_push(
                            heap, brow_task, bcol, level, PANGULU_TASK_SSSSM, level,
                            pangulu_storage_get_slot(storage, related_nondiag_uniaddr[related_nondiag_block_csr_to_csc[bidx_csr]]),
                            task->opdst, ssssm_op2, block_length, __FILE__, __LINE__);
                    }
                }
            }
            else
            {
                if (sent_rank_flag[target_rank] == 0)
                {
                    pangulu_cm_isend_block(task->opdst, nb, brow_task, bcol_task, target_rank);
                    sent_rank_flag[target_rank] = 1;
                }
            }
        }
        pthread_mutex_unlock(block_smatrix->info_mutex);
    }
    else if (kernel_id == PANGULU_TASK_GESSM)
    {
        pangulu_gessm_interface(nb, task->opdst, task->op1, 0);
        pthread_mutex_lock(block_smatrix->info_mutex);
        task->opdst->data_status = PANGULU_DATA_READY;
        if ((level % p) * q + (level % q) != rank)
        {
            diag_remain_task_count[level]--;
            if (diag_remain_task_count[level] == 0)
            {
                pangulu_storage_slot_queue_recycle_by_ptr(storage, task->op1->related_block);
                task->op1->related_block = NULL;
                pangulu_storage_slot_queue_recycle_by_ptr(storage, task->op1);
                task->op1 = NULL;
            }
        }
        pangulu_exblock_ptr bidx_diag = related_nondiag_fstblk_idx_after_diag[level];
        for (pangulu_exblock_ptr bidx = pangulu_binarysearch(related_nondiag_block_rowidx, related_nondiag_block_colptr[bcol_task], related_nondiag_block_colptr[bcol_task + 1], brow_task) + 1;
             bidx < related_nondiag_block_colptr[bcol_task + 1]; bidx++)
        {
            pangulu_exblock_idx brow = related_nondiag_block_rowidx[bidx];
            pangulu_int32_t target_rank = (brow % p) * q + (bcol_task % q);
            if (target_rank == rank)
            {
                while ((bidx_diag < related_nondiag_block_colptr[level + 1]) && (related_nondiag_block_rowidx[bidx_diag] < brow))
                {
                    bidx_diag++;
                }
                if ((bidx_diag < related_nondiag_block_colptr[level + 1]) && (related_nondiag_block_rowidx[bidx_diag] == brow))
                {
                    pangulu_storage_slot_t *ssssm_op1 = pangulu_storage_get_slot(storage, related_nondiag_uniaddr[bidx_diag]);
                    if (ssssm_op1 && (ssssm_op1->data_status == PANGULU_DATA_READY))
                    {
                        nondiag_remain_task_count[bidx]--;
                        pangulu_task_queue_push(
                            heap, brow, bcol_task, level, PANGULU_TASK_SSSSM, level,
                            pangulu_storage_get_slot(storage, related_nondiag_uniaddr[bidx]),
                            ssssm_op1, task->opdst, block_length, __FILE__, __LINE__);
                    }
                }
            }
            else
            {
                if (sent_rank_flag[target_rank] == 0)
                {
                    pangulu_cm_isend_block(task->opdst, nb, brow_task, bcol_task, target_rank);
                    sent_rank_flag[target_rank] = 1;
                }
            }
        }
        pthread_mutex_unlock(block_smatrix->info_mutex);
    }
    else if (kernel_id == PANGULU_TASK_SSSSM)
    {
        pthread_mutex_lock(block_smatrix->info_mutex);
        if (brow_task == bcol_task)
        {
            if (diag_remain_task_count[brow_task] == 1)
            {
                diag_remain_task_count[brow_task]--;
                pangulu_task_queue_push(heap, brow_task, bcol_task, brow_task, PANGULU_TASK_GETRF, brow_task, task->opdst, NULL, NULL, block_length, __FILE__, __LINE__);
            }
        }
        else if (brow_task < bcol_task)
        {
            pangulu_exblock_ptr bidx_task = pangulu_binarysearch(related_nondiag_block_rowidx, related_nondiag_block_colptr[bcol_task], related_nondiag_block_colptr[bcol_task + 1], brow_task);
            if (nondiag_remain_task_count[bidx_task] == 1)
            {
                pangulu_storage_slot_t *diag_block = pangulu_storage_get_diag(block_length, storage, diag_uniaddr[PANGULU_MIN(brow_task, bcol_task)]);
                if (diag_block && (diag_block->data_status == PANGULU_DATA_READY))
                {
                    if (diag_block->is_upper == 1)
                    {
                        diag_block = diag_block->related_block;
                    }
                    nondiag_remain_task_count[bidx_task]--;
                    pangulu_task_queue_push(
                        heap, brow_task, bcol_task, PANGULU_MIN(brow_task, bcol_task), PANGULU_TASK_GESSM,
                        PANGULU_MIN(brow_task, bcol_task), task->opdst, diag_block, NULL, block_length, __FILE__, __LINE__);
                }
            }
        }
        else
        {
            pangulu_exblock_ptr bidx_task = pangulu_binarysearch(related_nondiag_block_rowidx, related_nondiag_block_colptr[bcol_task], related_nondiag_block_colptr[bcol_task + 1], brow_task);
            if (nondiag_remain_task_count[bidx_task] == 1)
            {
                pangulu_storage_slot_t *diag_block = pangulu_storage_get_diag(block_length, storage, diag_uniaddr[PANGULU_MIN(brow_task, bcol_task)]);
                if (diag_block && (diag_block->data_status == PANGULU_DATA_READY))
                {
                    if (diag_block->is_upper == 0)
                    {
                        diag_block = diag_block->related_block;
                    }
                    nondiag_remain_task_count[bidx_task]--;
                    pangulu_task_queue_push(
                        heap, brow_task, bcol_task, PANGULU_MIN(brow_task, bcol_task), PANGULU_TASK_TSTRF,
                        PANGULU_MIN(brow_task, bcol_task), task->opdst, diag_block, NULL, block_length, __FILE__, __LINE__);
                }
            }
        }
        pthread_mutex_unlock(block_smatrix->info_mutex);
    }
    else
    {
        printf(PANGULU_E_K_ID);
        exit(1);
    }
}

void *pangulu_numeric_compute_thread(void *param)
{
    pangulu_numeric_thread_param *work_param = (pangulu_numeric_thread_param *)param;
    pangulu_common *common = work_param->pangulu_common;
    pangulu_block_common *block_common = work_param->block_common;
    pangulu_block_smatrix *block_smatrix = work_param->block_smatrix;
    pangulu_int32_t rank = block_common->rank;

    pangulu_int64_t cpu_thread_count_per_node = sysconf(_SC_NPROCESSORS_ONLN);
    int pangulu_omp_num_threads = common->omp_thread;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int i = 0; i < pangulu_omp_num_threads; i++)
    {
#ifdef HT_IS_OPEN
        CPU_SET((2 * (pangulu_omp_num_threads * rank + i)) % cpu_thread_count_per_node, &cpuset);
#else
        CPU_SET((pangulu_omp_num_threads * rank + i) % cpu_thread_count_per_node, &cpuset);
#endif
    }
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0)
    {
        perror("pthread_setaffinity_np error");
    }

#pragma omp parallel num_threads(common->omp_thread)
    {
        int tid = omp_get_thread_num();
#ifdef HT_IS_OPEN
        pangulu_bind_to_core((2 * (pangulu_omp_num_threads * rank + tid)) % cpu_thread_count_per_node);
#else
        pangulu_bind_to_core((pangulu_omp_num_threads * rank + tid) % cpu_thread_count_per_node);
#endif
    }

#ifdef GPU_OPEN
    int device_num;
    pangulu_platform_get_device_num(&device_num, PANGULU_DEFAULT_PLATFORM);
    pangulu_platform_set_default_device(rank % device_num, PANGULU_DEFAULT_PLATFORM);
#endif
    pangulu_int64_t *rank_remain_task_count = &(block_smatrix->rank_remain_task_count);
    pangulu_task_queue_t *heap = block_smatrix->heap;

    while ((*rank_remain_task_count) != 0)
    {
        if (pangulu_task_queue_empty(heap))
        {
            pangulu_aggregate_idle_batch(common, block_common, block_smatrix, 50, pangulu_execute_aggregated_ssssm, param);
            continue;
        }
        pangulu_int64_t ntask = 0;
        while (!pangulu_task_queue_empty(heap))
        {
            pangulu_task_t task = pangulu_task_queue_pop(heap);
            if (task.kernel_id == PANGULU_TASK_GETRF)
            {
                (*rank_remain_task_count)--;
                pangulu_numeric_work(&task, common, block_common, block_smatrix);
            }
            else
            {
                while (ntask + 1 > working_task_buf_capacity)
                {
                    working_task_buf_capacity = (working_task_buf_capacity + 1) * 2;
                    working_task_buf = pangulu_realloc(__FILE__, __LINE__, working_task_buf, sizeof(pangulu_task_t) * working_task_buf_capacity);
                }
                working_task_buf[ntask] = task;
                ntask++;
            }
        }
        (*rank_remain_task_count) -= ntask;
        if (ntask)
        {
            pangulu_numeric_work_batched(ntask, working_task_buf, common, block_common, block_smatrix);
        }
    }

    pangulu_free(__FILE__, __LINE__, working_task_buf);
    return NULL;
}

void pangulu_numeric(
    pangulu_common *common,
    pangulu_block_common *block_common,
    pangulu_block_smatrix *block_smatrix)
{
    pthread_t pthread;
    pangulu_numeric_thread_param param;
    param.pangulu_common = common;
    param.block_common = block_common;
    param.block_smatrix = block_smatrix;
    pthread_create(&pthread, NULL, pangulu_numeric_compute_thread, (void *)(&param));

    pangulu_int32_t block_length = block_common->block_length;
    pangulu_int32_t p = block_common->p;
    pangulu_int32_t q = block_common->q;
    pangulu_int32_t rank = block_common->rank;
    pangulu_int64_t *rank_remain_recv_block_count = &block_smatrix->rank_remain_recv_block_count;
    pangulu_task_queue_t *heap = block_smatrix->heap;
    pangulu_storage_t *storage = block_smatrix->storage;
    pangulu_exblock_ptr *related_nondiag_block_rowptr = block_smatrix->related_nondiag_block_rowptr;
    pangulu_exblock_idx *related_nondiag_block_colidx = block_smatrix->related_nondiag_block_colidx;
    pangulu_exblock_ptr *related_nondiag_block_csr_to_csc = block_smatrix->related_nondiag_block_csr_to_csc;
    pangulu_exblock_ptr *related_nondiag_block_colptr = block_smatrix->related_nondiag_block_colptr;
    pangulu_exblock_idx *related_nondiag_block_rowidx = block_smatrix->related_nondiag_block_rowidx;
    pangulu_uint64_t *related_nondiag_uniaddr = block_smatrix->related_nondiag_uniaddr;
    pangulu_uint64_t *diag_uniaddr = block_smatrix->diag_uniaddr;
    pangulu_int32_t *nondiag_remain_task_count = block_smatrix->nondiag_remain_task_count;
    pangulu_int32_t *diag_remain_task_count = block_smatrix->diag_remain_task_count;

    pangulu_int64_t cpu_thread_count_per_node = sysconf(_SC_NPROCESSORS_ONLN);
    int pangulu_omp_num_threads = common->omp_thread;
#ifdef HT_IS_OPEN
    pangulu_bind_to_core((2 * pangulu_omp_num_threads * rank) % cpu_thread_count_per_node);
#else
    pangulu_bind_to_core((pangulu_omp_num_threads * rank) % cpu_thread_count_per_node);
#endif

    pangulu_cm_sync();
    pangulu_task_queue_clear(heap);
    pthread_mutex_lock(block_smatrix->info_mutex);
    for (pangulu_int64_t level = 0; level < block_length; level++)
    {
        pangulu_int64_t now_rank = (level % p) * q + (level % q);
        if (now_rank == rank)
        {
            if (diag_remain_task_count[level] == 1)
            {
                diag_remain_task_count[level]--;
                pangulu_task_queue_push(
                    heap, level, level, level, PANGULU_TASK_GETRF, level,
                    pangulu_storage_get_diag(block_length, storage, diag_uniaddr[level]),
                    NULL, NULL, block_length, __FILE__, __LINE__);
            }
        }
    }
    pthread_mutex_unlock(block_smatrix->info_mutex);

    while ((*rank_remain_recv_block_count) != 0)
    {
        MPI_Status status;
        pangulu_cm_probe(&status);
        (*rank_remain_recv_block_count)--;
        pangulu_numeric_receive_message(status, block_common, block_smatrix);
    }

    pthread_join(pthread, NULL);
}

void pangulu_numeric_check(
    pangulu_common *common,
    pangulu_block_common *block_common,
    pangulu_block_smatrix *block_smatrix)
{
    pangulu_int32_t block_length = block_common->block_length;
    pangulu_int32_t p = block_common->p;
    pangulu_int32_t q = block_common->q;
    pangulu_int32_t rank = block_common->rank;
    pangulu_int64_t *rank_remain_recv_block_count = &block_smatrix->rank_remain_recv_block_count;
    pangulu_task_queue_t *heap = block_smatrix->heap;
    pangulu_storage_t *storage = block_smatrix->storage;
    pangulu_inblock_idx nb = block_common->nb;
    pangulu_int32_t nproc = block_common->sum_rank_size;
    pangulu_int32_t n = block_common->n;
    pangulu_exblock_ptr *related_nondiag_block_rowptr = block_smatrix->related_nondiag_block_rowptr;
    pangulu_exblock_idx *related_nondiag_block_colidx = block_smatrix->related_nondiag_block_colidx;
    pangulu_exblock_ptr *related_nondiag_block_csr_to_csc = block_smatrix->related_nondiag_block_csr_to_csc;
    pangulu_exblock_ptr *related_nondiag_block_colptr = block_smatrix->related_nondiag_block_colptr;
    pangulu_exblock_idx *related_nondiag_block_rowidx = block_smatrix->related_nondiag_block_rowidx;
    pangulu_uint64_t *related_nondiag_uniaddr = block_smatrix->related_nondiag_uniaddr;
    pangulu_uint32_t *nondiag_remain_task_count = block_smatrix->nondiag_remain_task_count;
    pangulu_uint64_t *diag_uniaddr = block_smatrix->diag_uniaddr;

    calculate_type *A_rowsum = block_smatrix->A_rowsum_reordered;
    calculate_type *Ux1 = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);
    calculate_type *LxUx1 = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);
    memset(Ux1, 0, sizeof(calculate_type) * n);
    memset(LxUx1, 0, sizeof(calculate_type) * n);

    for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
    {
        for (pangulu_exblock_idx bidx = related_nondiag_block_colptr[bcol]; bidx < related_nondiag_block_colptr[bcol + 1]; bidx++)
        {
            pangulu_exblock_idx brow = related_nondiag_block_rowidx[bidx];
            if (brow > bcol)
            {
                continue;
            }
            if ((brow % p) * q + (bcol % q) == rank)
            {
                pangulu_uint64_t slot_addr = related_nondiag_uniaddr[bidx];
                pangulu_storage_slot_t *slot = pangulu_storage_get_slot(storage, slot_addr);
                if (slot)
                {
                    pangulu_inblock_ptr *colptr = slot->columnpointer;
                    pangulu_inblock_idx *rowidx = slot->rowindex;
                    calculate_type *value = slot->value;
                    for (pangulu_inblock_idx col = 0; col < nb; col++)
                    {
                        for (pangulu_inblock_ptr idx = (col == 0 ? 0 : colptr[col]); idx < colptr[col + 1]; idx++)
                        {
                            pangulu_inblock_idx row = rowidx[idx];
                            if (brow * nb + row < n)
                            {
                                Ux1[brow * nb + row] += value[idx];
                            }
                        }
                    }
                }
            }
        }
    }
    for (pangulu_exblock_idx level = 0; level < block_length; level++)
    {
        if ((level % p) * q + (level % q) == rank)
        {
            pangulu_storage_slot_t *diag_upper = pangulu_storage_get_diag(block_length, storage, diag_uniaddr[level]);
            if (diag_upper->is_upper == 0)
            {
                diag_upper = diag_upper->related_block;
            }
            pangulu_inblock_ptr *rowptr = diag_upper->columnpointer;
            pangulu_inblock_idx *colidx = diag_upper->rowindex;
            calculate_type *value = diag_upper->value;
            for (pangulu_inblock_idx row = 0; row < nb; row++)
            {
                for (pangulu_inblock_ptr idx = rowptr[row]; idx < rowptr[row + 1]; idx++)
                {
                    pangulu_inblock_idx col = colidx[idx];
                    if (level * nb + row < n)
                    {
                        Ux1[level * nb + row] += value[idx];
                    }
                }
            }
        }
    }

    if (rank == 0)
    {
        MPI_Status mpi_stat;
        calculate_type *recv_buf = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);
        for (int fetch_rank = 1; fetch_rank < nproc; fetch_rank++)
        {
            MPI_Recv(recv_buf, n, MPI_VAL_TYPE, fetch_rank, 0, MPI_COMM_WORLD, &mpi_stat);
            for (pangulu_int32_t row = 0; row < n; row++)
            {
                Ux1[row] += recv_buf[row];
            }
        }
        for (int remote_rank = 1; remote_rank < nproc; remote_rank++)
        {
            MPI_Send(Ux1, n, MPI_VAL_TYPE, remote_rank, 0, MPI_COMM_WORLD);
        }
        pangulu_free(__FILE__, __LINE__, recv_buf);
    }
    else
    {
        MPI_Status mpi_stat;
        MPI_Send(Ux1, n, MPI_VAL_TYPE, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(Ux1, n, MPI_VAL_TYPE, 0, 0, MPI_COMM_WORLD, &mpi_stat);
    }

    for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
    {
        for (pangulu_exblock_idx bidx = related_nondiag_block_colptr[bcol]; bidx < related_nondiag_block_colptr[bcol + 1]; bidx++)
        {
            pangulu_exblock_idx brow = related_nondiag_block_rowidx[bidx];
            if (brow < bcol)
            {
                continue;
            }
            if ((brow % p) * q + (bcol % q) == rank)
            {
                pangulu_uint64_t slot_addr = related_nondiag_uniaddr[bidx];
                pangulu_storage_slot_t *slot = pangulu_storage_get_slot(storage, slot_addr);
                if (slot)
                {
                    pangulu_inblock_ptr *colptr = slot->columnpointer;
                    pangulu_inblock_idx *rowidx = slot->rowindex;
                    calculate_type *value = slot->value;
                    for (pangulu_inblock_idx col = 0; col < nb; col++)
                    {
                        for (pangulu_inblock_ptr idx = (col == 0 ? 0 : colptr[col]); idx < colptr[col + 1]; idx++)
                        {
                            pangulu_inblock_idx row = rowidx[idx];
                            if (brow * nb + row < n && bcol * nb + col < n)
                            {
                                LxUx1[brow * nb + row] += Ux1[bcol * nb + col] * value[idx];
                            }
                        }
                    }
                }
            }
        }
    }
    for (pangulu_exblock_idx level = 0; level < block_length; level++)
    {
        if ((level % p) * q + (level % q) == rank)
        {
            pangulu_storage_slot_t *diag_lower = pangulu_storage_get_diag(block_length, storage, diag_uniaddr[level]);
            if (diag_lower->is_upper == 1)
            {
                diag_lower = diag_lower->related_block;
            }
            pangulu_inblock_ptr *colptr = diag_lower->columnpointer;
            pangulu_inblock_idx *rowidx = diag_lower->rowindex;
            calculate_type *value = diag_lower->value;
            for (pangulu_inblock_idx col = 0; col < nb; col++)
            {
                for (pangulu_inblock_ptr idx = colptr[col]; idx < colptr[col + 1]; idx++)
                {
                    pangulu_inblock_idx row = rowidx[idx];
                    if (level * nb + row < n && level * nb + col < n)
                    {
                        LxUx1[level * nb + row] += Ux1[level * nb + col] * value[idx];
                    }
                }
            }
            for (pangulu_inblock_idx row = 0; row < nb; row++)
            {
                if (level * nb + row < n && level * nb + row < n)
                {
                    LxUx1[level * nb + row] += Ux1[level * nb + row];
                }
            }
        }
    }

    if (rank == 0)
    {
        MPI_Status mpi_stat;
        calculate_type *recv_buf = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);
        for (int fetch_rank = 1; fetch_rank < nproc; fetch_rank++)
        {
            MPI_Recv(recv_buf, n, MPI_VAL_TYPE, fetch_rank, 0, MPI_COMM_WORLD, &mpi_stat);
            for (pangulu_int32_t row = 0; row < n; row++)
            {
                LxUx1[row] += recv_buf[row];
            }
        }
        pangulu_free(__FILE__, __LINE__, recv_buf);
    }
    else
    {
        MPI_Send(LxUx1, n, MPI_VAL_TYPE, 0, 0, MPI_COMM_WORLD);
    }

#ifdef PANGULU_COMPLEX
    if (rank == 0)
    {
        double sum = 0.0;
        double c = 0.0;
        for (int i = 0; i < n; i++)
        {
            calculate_type diff = LxUx1[i] - A_rowsum[i];
            double num = creal(diff) * creal(diff) + cimag(diff) * cimag(diff);
            double z = num - c;
            double t = sum + z;
            c = (t - sum) - z;
            sum = t;
        }
        double residual_norm2 = sqrt(sum);

        sum = 0.0;
        c = 0.0;
        for (int i = 0; i < n; i++)
        {
            calculate_type val = A_rowsum[i];
            double num = creal(val) * creal(val) + cimag(val) * cimag(val);
            double z = num - c;
            double t = sum + z;
            c = (t - sum) - z;
            sum = t;
        }
        double rhs_norm2 = sqrt(sum);

        printf(PANGULU_I_NUMERIC_CHECK);
    }
#else
    if (rank == 0)
    {
        calculate_type sum = 0.0;
        calculate_type c = 0.0;
        for (int i = 0; i < n; i++)
        {
            calculate_type num = (LxUx1[i] - A_rowsum[i]) * (LxUx1[i] - A_rowsum[i]);
            calculate_type z = num - c;
            calculate_type t = sum + z;
            c = (t - sum) - z;
            sum = t;
        }
        calculate_type residual_norm2 = sqrt(sum);

        sum = 0.0;
        c = 0.0;
        for (int i = 0; i < n; i++)
        {
            calculate_type num = A_rowsum[i] * A_rowsum[i];
            calculate_type z = num - c;
            calculate_type t = sum + z;
            c = (t - sum) - z;
            sum = t;
        }
        calculate_type rhs_norm2 = sqrt(sum);
        printf(PANGULU_I_NUMERIC_CHECK);
    }
#endif
}
