#include "pangulu_common.h"

#ifdef PANGULU_PERF
void pangulu_getrf_flop(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    int tid)
{
    pangulu_storage_slot_t *upper_diag;
    pangulu_storage_slot_t *lower_diag;
    if (opdst->is_upper)
    {
        upper_diag = opdst;
        lower_diag = opdst->related_block;
    }
    else
    {
        upper_diag = opdst->related_block;
        lower_diag = opdst;
    }
    for (pangulu_int32_t level = 0; level < nb; level++)
    {
        if (upper_diag->columnpointer[level] == upper_diag->columnpointer[level + 1])
        {
            continue;
        }

        global_stat.flop += (lower_diag->columnpointer[level + 1] - lower_diag->columnpointer[level]);

        for (pangulu_int32_t csc_idx = lower_diag->columnpointer[level]; csc_idx < lower_diag->columnpointer[level + 1]; csc_idx++)
        {
            pangulu_int32_t row = lower_diag->rowindex[csc_idx];
            pangulu_int32_t csr_idx_op2 = upper_diag->columnpointer[level];
            pangulu_int32_t csr_idx_op2_ub = upper_diag->columnpointer[level + 1];
            pangulu_int32_t csr_idx_opdst = upper_diag->columnpointer[row];
            pangulu_int32_t csr_idx_opdst_ub = upper_diag->columnpointer[row + 1];
            while (csr_idx_op2 < csr_idx_op2_ub && csr_idx_opdst < csr_idx_opdst_ub)
            {
                if (upper_diag->rowindex[csr_idx_opdst] == upper_diag->rowindex[csr_idx_op2])
                {
                    global_stat.flop += 2;
                    csr_idx_op2++;
                    csr_idx_opdst++;
                }
                while (csr_idx_op2 < csr_idx_op2_ub && upper_diag->rowindex[csr_idx_op2] < upper_diag->rowindex[csr_idx_opdst])
                {
                    csr_idx_op2++;
                }
                while (csr_idx_opdst < csr_idx_opdst_ub && upper_diag->rowindex[csr_idx_opdst] < upper_diag->rowindex[csr_idx_op2])
                {
                    csr_idx_opdst++;
                }
            }
        }

        for (pangulu_int32_t csr_idx = upper_diag->columnpointer[level] + 1; csr_idx < upper_diag->columnpointer[level + 1]; csr_idx++)
        {
            pangulu_int32_t col = upper_diag->rowindex[csr_idx];
            pangulu_int32_t csc_idx_op1 = lower_diag->columnpointer[level];
            pangulu_int32_t csc_idx_op1_ub = lower_diag->columnpointer[level + 1];
            pangulu_int32_t csc_idx_opdst = lower_diag->columnpointer[col];
            pangulu_int32_t csc_idx_opdst_ub = lower_diag->columnpointer[col + 1];
            while (csc_idx_op1 < csc_idx_op1_ub && csc_idx_opdst < csc_idx_opdst_ub)
            {
                if (lower_diag->rowindex[csc_idx_opdst] == lower_diag->rowindex[csc_idx_op1])
                {
                    global_stat.flop += 2;
                    csc_idx_op1++;
                    csc_idx_opdst++;
                }
                while (csc_idx_op1 < csc_idx_op1_ub && lower_diag->rowindex[csc_idx_op1] < lower_diag->rowindex[csc_idx_opdst])
                {
                    csc_idx_op1++;
                }
                while (csc_idx_opdst < csc_idx_opdst_ub && lower_diag->rowindex[csc_idx_opdst] < lower_diag->rowindex[csc_idx_op1])
                {
                    csc_idx_opdst++;
                }
            }
        }
    }
}

void pangulu_tstrf_flop(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *opdiag,
    int tid)
{
    if (opdiag->is_upper == 0)
    {
        opdiag = opdiag->related_block;
    }
    for (pangulu_int32_t rhs_id = 0; rhs_id < nb; rhs_id++)
    {
        for (pangulu_int32_t rhs_idx = opdst->rowpointer[rhs_id]; rhs_idx < opdst->rowpointer[rhs_id + 1]; rhs_idx++)
        {
            pangulu_int32_t mul_row = opdst->columnindex[rhs_idx];
            pangulu_int32_t lsum_idx = rhs_idx + 1;
            pangulu_int32_t diag_idx = opdiag->columnpointer[mul_row];
            global_stat.flop++;
            while (lsum_idx < opdst->rowpointer[rhs_id + 1] && diag_idx < opdiag->columnpointer[mul_row + 1])
            {
                if (opdiag->rowindex[diag_idx] == opdst->columnindex[lsum_idx])
                {
                    global_stat.flop += 2;
                    lsum_idx++;
                    diag_idx++;
                }
                while (lsum_idx < opdst->rowpointer[rhs_id + 1] && opdst->columnindex[lsum_idx] < opdiag->rowindex[diag_idx])
                {
                    lsum_idx++;
                }
                while (diag_idx < opdiag->columnpointer[mul_row + 1] && opdiag->rowindex[diag_idx] < opdst->columnindex[lsum_idx])
                {
                    diag_idx++;
                }
            }
        }
    }
}

void pangulu_gessm_flop(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *opdiag,
    int tid)
{
    if (opdiag->is_upper == 1)
    {
        opdiag = opdiag->related_block;
    }
    for (pangulu_int32_t rhs_id = 0; rhs_id < nb; rhs_id++)
    {
        for (pangulu_int32_t rhs_idx = opdst->columnpointer[rhs_id]; rhs_idx < opdst->columnpointer[rhs_id + 1]; rhs_idx++)
        {
            pangulu_int32_t mul_row = opdst->rowindex[rhs_idx];
            pangulu_int32_t lsum_idx = rhs_idx + 1;
            pangulu_int32_t diag_idx = opdiag->columnpointer[mul_row];
            while (lsum_idx < opdst->columnpointer[rhs_id + 1] && diag_idx < opdiag->columnpointer[mul_row + 1])
            {
                if (opdiag->rowindex[diag_idx] == opdst->rowindex[lsum_idx])
                {
                    global_stat.flop += 2;
                    lsum_idx++;
                    diag_idx++;
                }
                while (lsum_idx < opdst->columnpointer[rhs_id + 1] && opdst->rowindex[lsum_idx] < opdiag->rowindex[diag_idx])
                {
                    lsum_idx++;
                }
                while (diag_idx < opdiag->columnpointer[mul_row + 1] && opdiag->rowindex[diag_idx] < opdst->rowindex[lsum_idx])
                {
                    diag_idx++;
                }
            }
        }
    }
}

void pangulu_ssssm_flop(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *op1,
    pangulu_storage_slot_t *op2,
    int tid)
{
    for (pangulu_int32_t col2 = 0; col2 < nb; col2++)
    {
        for (pangulu_int32_t idx2 = op2->columnpointer[col2]; idx2 < op2->columnpointer[col2 + 1]; idx2++)
        {
            pangulu_inblock_idx row2 = op2->rowindex[idx2];
            global_stat.flop += 2 * (op1->columnpointer[row2 + 1] - op1->columnpointer[row2]);
        }
    }
}

#endif

void pangulu_getrf_interface(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    int tid)
{
#ifdef PANGULU_PERF
    global_stat.kernel_cnt++;
    struct timeval start;
    pangulu_time_start(&start);
#endif
    pangulu_platform_getrf(nb, opdst, tid, PANGULU_DEFAULT_PLATFORM);
    pangulu_platform_synchronize(PANGULU_DEFAULT_PLATFORM);
#ifdef PANGULU_PERF
    global_stat.time_outer_kernel += pangulu_time_stop(&start);
    pangulu_getrf_flop(nb, opdst, tid);
#endif
}

void pangulu_tstrf_interface(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *opdiag,
    int tid)
{
#ifdef PANGULU_PERF
    global_stat.kernel_cnt++;
    struct timeval start;
    pangulu_time_start(&start);
#endif
    pangulu_platform_tstrf(nb, opdst, opdiag, tid, PANGULU_DEFAULT_PLATFORM);
    pangulu_platform_synchronize(PANGULU_DEFAULT_PLATFORM);
#ifdef PANGULU_PERF
    global_stat.time_outer_kernel += pangulu_time_stop(&start);
    pangulu_tstrf_flop(nb, opdst, opdiag, tid);
#endif
}

void pangulu_gessm_interface(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *opdiag,
    int tid)
{
#ifdef PANGULU_PERF
    global_stat.kernel_cnt++;
    struct timeval start;
    pangulu_time_start(&start);
#endif
    pangulu_platform_gessm(nb, opdst, opdiag, tid, PANGULU_DEFAULT_PLATFORM);
    pangulu_platform_synchronize(PANGULU_DEFAULT_PLATFORM);
#ifdef PANGULU_PERF
    global_stat.time_outer_kernel += pangulu_time_stop(&start);
    pangulu_gessm_flop(nb, opdst, opdiag, tid);
#endif
}

void pangulu_ssssm_interface(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *op1,
    pangulu_storage_slot_t *op2,
    int tid)
{
#ifdef PANGULU_PERF
    global_stat.kernel_cnt++;
    struct timeval start;
    pangulu_time_start(&start);
#endif
    pangulu_platform_ssssm(nb, opdst, op1, op2, tid, PANGULU_DEFAULT_PLATFORM);
    pangulu_platform_synchronize(PANGULU_DEFAULT_PLATFORM);
#ifdef PANGULU_PERF
    global_stat.time_outer_kernel += pangulu_time_stop(&start);
    pangulu_ssssm_flop(nb, opdst, op1, op2, tid);
#endif
}

void pangulu_hybrid_batched_interface(
    pangulu_inblock_idx nb,
    pangulu_uint64_t ntask,
    pangulu_task_t *tasks)
{
#ifdef PANGULU_PERF
    global_stat.kernel_cnt++;
    struct timeval start;
    pangulu_time_start(&start);
#endif
    pangulu_platform_hybrid_batched(nb, ntask, tasks, PANGULU_DEFAULT_PLATFORM);
    pangulu_platform_synchronize(PANGULU_DEFAULT_PLATFORM);
#ifdef PANGULU_PERF
    global_stat.time_outer_kernel += pangulu_time_stop(&start);

    for (pangulu_int64_t i = 0; i < ntask; i++)
    {
        switch (tasks[i].kernel_id)
        {
        case PANGULU_TASK_GETRF:
            pangulu_getrf_flop(nb, tasks[i].opdst, 0);
            break;
        case PANGULU_TASK_TSTRF:
            pangulu_tstrf_flop(nb, tasks[i].opdst, tasks[i].op1, 0);
            break;
        case PANGULU_TASK_GESSM:
            pangulu_gessm_flop(nb, tasks[i].opdst, tasks[i].op1, 0);
            break;
        case PANGULU_TASK_SSSSM:
            pangulu_ssssm_flop(nb, tasks[i].opdst, tasks[i].op1, tasks[i].op2, 0);
            break;
        }
    }
#endif
}

void pangulu_ssssm_batched_interface(
    pangulu_inblock_idx nb,
    pangulu_uint64_t ntask,
    pangulu_task_t *tasks)
{
#ifdef PANGULU_PERF
    global_stat.kernel_cnt++;
    struct timeval start;
    pangulu_time_start(&start);
#endif
    pangulu_platform_hybrid_batched(nb, ntask, tasks, PANGULU_DEFAULT_PLATFORM);
    pangulu_platform_synchronize(PANGULU_DEFAULT_PLATFORM);
#ifdef PANGULU_PERF
    global_stat.time_outer_kernel += pangulu_time_stop(&start);

    for (pangulu_int64_t i = 0; i < ntask; i++)
    {
        pangulu_ssssm_flop(nb, tasks[i].opdst, tasks[i].op1, tasks[i].op2, 0);
    }
#endif
}

void pangulu_spmv_interface(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *a,
    calculate_type *x,
    calculate_type *y)
{
    pangulu_platform_spmv(nb, a, x, y, PANGULU_DEFAULT_PLATFORM);
}

void pangulu_vecadd_interface(
    pangulu_int64_t length,
    calculate_type *bval,
    calculate_type *xval)
{
    pangulu_platform_vecadd(length, bval, xval, PANGULU_DEFAULT_PLATFORM);
}

void pangulu_sptrsv_interface(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *s,
    calculate_type *xval,
    pangulu_int64_t uplo)
{
    pangulu_platform_sptrsv(nb, s, xval, uplo, PANGULU_DEFAULT_PLATFORM);
}
