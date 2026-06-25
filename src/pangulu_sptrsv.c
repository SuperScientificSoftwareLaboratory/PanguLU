#include "pangulu_common.h"

void pangulu_sptrsv_preprocessing(
    pangulu_block_common *bcommon,
    pangulu_block_smatrix *bsmatrix,
    pangulu_vector *reordered_rhs)
{
    pangulu_exblock_idx block_length = bcommon->block_length;
    pangulu_inblock_idx nb = bcommon->nb;
    pangulu_int32_t rank = bcommon->rank;
    pangulu_int32_t nproc = bcommon->sum_rank_size;
    pangulu_int16_t p = bcommon->p;
    pangulu_int16_t q = bcommon->q;
    bsmatrix->rhs = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * block_length * nb);
    if (rank == 0)
    {
        memcpy(bsmatrix->rhs, reordered_rhs->value, sizeof(calculate_type) * reordered_rhs->row);
    }
    pangulu_cm_bcast(bsmatrix->rhs, block_length * nb, MPI_VAL_TYPE, 0);
    bsmatrix->spmv_acc = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nb);
    bsmatrix->recv_buf = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nb);
}

void pangulu_sptrsv_uplo(
    pangulu_block_common *bcommon,
    pangulu_block_smatrix *bsmatrix,
    pangulu_int32_t uplo)
{
    pangulu_task_queue_t *heap = bsmatrix->heap;
    pangulu_int32_t rank = bcommon->rank;
    pangulu_exblock_idx block_length = bcommon->block_length;
    pangulu_int32_t p = bcommon->p;
    pangulu_int32_t q = bcommon->q;
    pangulu_storage_t *storage = bsmatrix->storage;

    pangulu_uint64_t *diag_uniaddr = bsmatrix->diag_uniaddr;
    pangulu_exblock_ptr *nondiag_block_rowptr = bsmatrix->nondiag_block_rowptr;
    pangulu_exblock_idx *nondiag_block_colidx = bsmatrix->nondiag_block_colidx;
    pangulu_exblock_ptr *nondiag_block_csr_to_csc = bsmatrix->nondiag_block_csr_to_csc;
    pangulu_uint64_t *related_nondiag_uniaddr = bsmatrix->related_nondiag_uniaddr;
    pangulu_inblock_idx nb = bcommon->nb;
    calculate_type *rhs = bsmatrix->rhs;
    calculate_type *spmv_acc = bsmatrix->spmv_acc;
    calculate_type *recv_buf = bsmatrix->recv_buf;

    if (uplo == PANGULU_LOWER)
    {
        for (pangulu_exblock_idx brow = 0; brow < block_length; brow++)
        {
            calculate_type *target_rhs = rhs + brow * nb;
            memset(spmv_acc, 0, sizeof(calculate_type) * nb);
            for (pangulu_int64_t bidx_csr = nondiag_block_rowptr[brow]; bidx_csr < nondiag_block_rowptr[brow + 1]; bidx_csr++)
            {
                pangulu_exblock_idx bcol = nondiag_block_colidx[bidx_csr];
                if (((brow % p) != (rank / q)) || ((bcol % q) != (rank % q)))
                {
                    continue;
                }
                if (bcol < brow)
                {
                    calculate_type *local_rhs = rhs + bcol * nb;
                    pangulu_platform_spmv(nb, &storage->bins[0].slots[nondiag_block_csr_to_csc[bidx_csr]], local_rhs, spmv_acc, PANGULU_PLATFORM_CPU_NAIVE);
                }
                else if (bcol > brow)
                {
                    break;
                }
            }

            if (((brow % p) == (rank / q)) && ((brow % q) == (rank % q)))
            {
                for (int fetch_rank = (rank / q) * q; fetch_rank < ((rank / q) + 1) * q; fetch_rank++)
                {
                    if (fetch_rank == rank)
                    {
                        continue;
                    }
                    pangulu_cm_recv(recv_buf, sizeof(calculate_type) * nb, fetch_rank, brow, 10);
#pragma omp simd
                    for (pangulu_inblock_idx i = 0; i < nb; i++)
                    {
                        spmv_acc[i] += recv_buf[i];
                    }
                }
                for (pangulu_inblock_idx i = 0; i < nb; i++)
                {
                    target_rhs[i] += spmv_acc[i];
                }
                pangulu_storage_slot_t *lower_diag = pangulu_storage_get_diag(block_length, storage, diag_uniaddr[brow]);
                if (lower_diag->is_upper)
                {
                    lower_diag = lower_diag->related_block;
                }
                pangulu_platform_sptrsv(nb, lower_diag, target_rhs, PANGULU_LOWER, PANGULU_PLATFORM_CPU_NAIVE);
                pangulu_cm_bcast(target_rhs, nb, MPI_VAL_TYPE, rank);
            }

            int diagonal_rank = (brow % p) * q + (brow % q);
            if (diagonal_rank != rank)
            {
                if ((diagonal_rank / q) == (rank / q))
                {
                    pangulu_cm_isend(spmv_acc, sizeof(calculate_type) * nb, diagonal_rank, brow, 10);
                }
                pangulu_cm_bcast(target_rhs, nb, MPI_VAL_TYPE, diagonal_rank);
            }
        }
    }
    else if (uplo == PANGULU_UPPER)
    {
        for (pangulu_exblock_idx brow = block_length - 1; brow < block_length; brow--)
        {
            calculate_type *target_rhs = rhs + brow * nb;
            memset(spmv_acc, 0, sizeof(calculate_type) * nb);

            for (pangulu_int64_t bidx_csr = nondiag_block_rowptr[brow + 1] - 1; bidx_csr >= (pangulu_int64_t)nondiag_block_rowptr[brow]; bidx_csr--)
            {
                pangulu_exblock_idx bcol = nondiag_block_colidx[bidx_csr];
                if (((brow % p) != (rank / q)) || ((bcol % q) != (rank % q)))
                {
                    continue;
                }
                if (bcol > brow)
                {
                    calculate_type *local_rhs = rhs + bcol * nb;
                    pangulu_platform_spmv(nb, &storage->bins[0].slots[nondiag_block_csr_to_csc[bidx_csr]], local_rhs, spmv_acc, PANGULU_PLATFORM_CPU_NAIVE);
                }
                else if (bcol < brow)
                {
                    break;
                }
            }

            if (((brow % p) == (rank / q)) && ((brow % q) == (rank % q)))
            {
                for (int fetch_rank = (rank / q) * q; fetch_rank < ((rank / q) + 1) * q; fetch_rank++)
                {
                    if (fetch_rank == rank)
                    {
                        continue;
                    }
                    pangulu_cm_recv(recv_buf, sizeof(calculate_type) * nb, fetch_rank, 0, 10);
#pragma omp simd
                    for (pangulu_inblock_idx i = 0; i < nb; i++)
                    {
                        spmv_acc[i] += recv_buf[i];
                    }
                }

                for (pangulu_inblock_idx i = 0; i < nb; i++)
                {
                    target_rhs[i] += spmv_acc[i];
                }
                pangulu_storage_slot_t *upper_diag = pangulu_storage_get_diag(block_length, storage, diag_uniaddr[brow]);
                if (upper_diag->is_upper == 0)
                {
                    upper_diag = upper_diag->related_block;
                }
                pangulu_platform_sptrsv(nb, upper_diag, target_rhs, PANGULU_UPPER, PANGULU_PLATFORM_CPU_NAIVE);
                pangulu_cm_bcast(target_rhs, nb, MPI_VAL_TYPE, rank);
            }

            int diagonal_rank = (brow % p) * q + (brow % q);
            if (diagonal_rank != rank)
            {
                if (diagonal_rank / q == rank / q)
                {
                    pangulu_cm_isend(spmv_acc, sizeof(calculate_type) * nb, diagonal_rank, 0, 10);
                }
                pangulu_cm_bcast(target_rhs, nb, MPI_VAL_TYPE, diagonal_rank);
            }
        }
    }
}

void pangulu_solve(
    pangulu_block_common *block_common,
    pangulu_block_smatrix *block_smatrix,
    pangulu_vector *result)
{
    int rank = block_common->rank;
    pangulu_cm_sync();
    pangulu_sptrsv_uplo(block_common, block_smatrix, PANGULU_LOWER);
    pangulu_cm_sync();
    pangulu_sptrsv_uplo(block_common, block_smatrix, PANGULU_UPPER);
    pangulu_cm_sync();
    if (block_common->rank == 0)
    {
        memcpy(result->value, block_smatrix->rhs, sizeof(calculate_type) * result->row);
    }
}