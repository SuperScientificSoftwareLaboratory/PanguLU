#include "pangulu_common.h"

extern pangulu_int32_t **ssssm_hash_lu;
extern pangulu_int32_t **ssssm_hash_l_row;
extern pangulu_int32_t **ssssm_hash_l_row_inv;
extern pangulu_int32_t **ssssm_hash_u_col;
extern pangulu_int32_t **ssssm_hash_u_col_inv;
extern calculate_type **ssssm_l_value;
extern calculate_type **ssssm_u_value;
extern calculate_type **temp_a_value;
extern pangulu_int32_t **hd_getrf_nnzu;

void pangulu_preprocessing(
    pangulu_common *common,
    pangulu_block_common *bcommon,
    pangulu_block_smatrix *bsmatrix,
    pangulu_origin_smatrix *reorder_matrix,
    pangulu_int32_t nthread)
{
    pangulu_exblock_idx n = bcommon->n;
    pangulu_exblock_idx nb = bcommon->nb;
    pangulu_exblock_idx block_length = bcommon->block_length;
    pangulu_int32_t rank = bcommon->rank;
    pangulu_int32_t nproc = bcommon->sum_rank_size;
    pangulu_exblock_ptr *bcsc_reordered_pointer;
    pangulu_exblock_idx *bcsc_reordered_index;
    pangulu_exblock_ptr *bcsc_reordered_blknnzptr;
    pangulu_inblock_ptr **bcsc_reordered_inblk_pointers;
    pangulu_inblock_idx **bcsc_reordered_inblk_indeces;
    calculate_type **bcsc_reordered_inblk_values;
    pangulu_inblock_ptr **diag_upper_rowptr;
    pangulu_inblock_idx **diag_upper_colidx;
    calculate_type **diag_upper_csrvalue;
    pangulu_inblock_ptr **diag_lower_colptr;
    pangulu_inblock_idx **diag_lower_rowidx;
    calculate_type **diag_lower_cscvalue;
    calculate_type **nondiag_cscvalue;
    pangulu_inblock_ptr **nondiag_colptr;
    pangulu_inblock_idx **nondiag_rowidx;
    pangulu_inblock_ptr **nondiag_csr_to_csc;
    pangulu_inblock_ptr **nondiag_rowptr;
    pangulu_inblock_idx **nondiag_colidx;
    pangulu_exblock_ptr *nondiag_block_colptr;
    pangulu_exblock_idx *nondiag_block_rowidx;
    pangulu_exblock_ptr *nondiag_block_rowptr;
    pangulu_exblock_idx *nondiag_block_colidx;
    pangulu_exblock_ptr *nondiag_block_csr_to_csc;
    pangulu_exblock_ptr *related_nondiag_block_colptr;
    pangulu_exblock_idx *related_nondiag_block_rowidx;
    pangulu_exblock_ptr *related_nondiag_block_rowptr;
    pangulu_exblock_idx *related_nondiag_block_colidx;
    pangulu_exblock_ptr *related_nondiag_block_csr_to_csc;
    pangulu_uint64_t *diag_uniaddr;

    pangulu_int64_t p = sqrt(nproc);
    while ((nproc % p) != 0)
    {
        p--;
    }
    pangulu_int64_t q = nproc / p;

    if (rank == 0)
        pangulu_sort_exblock_struct(n, bsmatrix->symbolic_rowpointer, bsmatrix->symbolic_columnindex, 0);

    pangulu_cm_sync_asym(0);

    pangulu_cm_distribute_csc_to_distbcsc(
        n, nb,
        reorder_matrix->columnpointer,
        reorder_matrix->rowindex,
        reorder_matrix->value_csc,
        &bcsc_reordered_pointer,
        &bcsc_reordered_index,
        &bcsc_reordered_blknnzptr,
        &bcsc_reordered_inblk_pointers,
        &bcsc_reordered_inblk_indeces,
        &bcsc_reordered_inblk_values,
        NULL,
        NULL);

    pangulu_cm_distribute_csc_to_distbcsc_symb(
        n, nb,
        bsmatrix->symbolic_rowpointer,
        bsmatrix->symbolic_columnindex,
        &diag_upper_rowptr,
        &diag_upper_colidx,
        &diag_upper_csrvalue,
        &diag_lower_colptr,
        &diag_lower_rowidx,
        &diag_lower_cscvalue,
        &nondiag_cscvalue,
        &nondiag_colptr,
        &nondiag_rowidx,
        &nondiag_csr_to_csc,
        &nondiag_rowptr,
        &nondiag_colidx,
        &nondiag_block_colptr,
        &nondiag_block_rowidx,
        &nondiag_block_rowptr,
        &nondiag_block_colidx,
        &nondiag_block_csr_to_csc,
        &related_nondiag_block_colptr,
        &related_nondiag_block_rowidx,
        &related_nondiag_block_rowptr,
        &related_nondiag_block_colidx,
        &related_nondiag_block_csr_to_csc,
        &diag_uniaddr);

    pangulu_convert_block_fill_value_to_struct(
        p, q, rank, n, nb,
        bcsc_reordered_pointer,
        bcsc_reordered_index,
        bcsc_reordered_blknnzptr,
        bcsc_reordered_inblk_pointers,
        bcsc_reordered_inblk_indeces,
        bcsc_reordered_inblk_values,
        nondiag_block_colptr,
        nondiag_block_rowidx,
        nondiag_colptr,
        nondiag_rowidx,
        nondiag_cscvalue,
        diag_uniaddr,
        diag_upper_rowptr,
        diag_upper_colidx,
        diag_upper_csrvalue,
        diag_lower_colptr,
        diag_lower_rowidx,
        diag_lower_cscvalue);

    struct timeval start_time;
    pangulu_time_start(&start_time);
    pangulu_int64_t rank_remain_task_count = 0;
    pangulu_int32_t *nondiag_remain_task_count = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * related_nondiag_block_colptr[block_length]);
    memset(nondiag_remain_task_count, 0, sizeof(pangulu_int32_t) * related_nondiag_block_colptr[block_length]);
    for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
    {
        for (pangulu_exblock_ptr bidx = related_nondiag_block_colptr[bcol]; bidx < related_nondiag_block_colptr[bcol + 1]; bidx++)
        {
            pangulu_exblock_idx brow = related_nondiag_block_rowidx[bidx];
            if (((brow % p) * q + (bcol % q)) == rank)
            {
                pangulu_exblock_ptr bidx1 = related_nondiag_block_colptr[bcol];
                pangulu_exblock_ptr bidx2 = related_nondiag_block_rowptr[brow];
                pangulu_exblock_idx blevel_ub = PANGULU_MIN(brow, bcol);
                pangulu_int64_t current_block_remain_task_count = 1;
                while (related_nondiag_block_rowidx[bidx1] < blevel_ub && related_nondiag_block_colidx[bidx2] < blevel_ub)
                {
                    if (related_nondiag_block_rowidx[bidx1] < related_nondiag_block_colidx[bidx2])
                    {
                        bidx1++;
                    }
                    else if (related_nondiag_block_rowidx[bidx1] > related_nondiag_block_colidx[bidx2])
                    {
                        bidx2++;
                    }
                    else
                    {
                        bidx1++;
                        bidx2++;
                        current_block_remain_task_count++;
                    }
                }
                rank_remain_task_count += current_block_remain_task_count;
                nondiag_remain_task_count[bidx] += current_block_remain_task_count;
            }
        }
    }

    pangulu_int32_t *diag_remain_task_count = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * block_length);
    memset(diag_remain_task_count, 0, sizeof(pangulu_int32_t) * block_length);
    for (pangulu_exblock_idx level = 0; level < block_length; level++)
    {
        if ((level % p) * q + (level % q) == rank)
        {
            diag_remain_task_count[level] = 1;
            pangulu_exblock_ptr bidx1 = related_nondiag_block_colptr[level];
            pangulu_exblock_ptr bidx2 = related_nondiag_block_rowptr[level];
            while (
                bidx1 < related_nondiag_block_colptr[level + 1] &&
                bidx2 < related_nondiag_block_rowptr[level + 1] &&
                related_nondiag_block_rowidx[bidx1] < level &&
                related_nondiag_block_colidx[bidx2] < level)
            {
                if (related_nondiag_block_rowidx[bidx1] == related_nondiag_block_colidx[bidx2])
                {
                    diag_remain_task_count[level]++;
                    bidx1++;
                    bidx2++;
                }
                while (
                    bidx1 < related_nondiag_block_colptr[level + 1] &&
                    related_nondiag_block_rowidx[bidx1] < level &&
                    related_nondiag_block_rowidx[bidx1] < related_nondiag_block_colidx[bidx2])
                {
                    bidx1++;
                }
                while (
                    bidx2 < related_nondiag_block_rowptr[level + 1] &&
                    related_nondiag_block_colidx[bidx2] < level &&
                    related_nondiag_block_rowidx[bidx1] > related_nondiag_block_colidx[bidx2])
                {
                    bidx2++;
                }
            }
            rank_remain_task_count += diag_remain_task_count[level];
        }
    }

    char *bcsc_nondiag_remote_need_flag = pangulu_malloc(__FILE__, __LINE__, sizeof(char) * related_nondiag_block_colptr[block_length]);
    char *bcsc_diag_remote_need_flag = pangulu_malloc(__FILE__, __LINE__, sizeof(char) * block_length);
    memset(bcsc_nondiag_remote_need_flag, 0, sizeof(char) * related_nondiag_block_colptr[block_length]);
    memset(bcsc_diag_remote_need_flag, 0, sizeof(char) * block_length);
    pangulu_int64_t rank_remain_recv_block_count = 0;
    for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
    {
        pangulu_int64_t bidx_maxrow = related_nondiag_block_colptr[bcol + 1] - 1;
        char diag_need_flag = 1;
        while (bidx_maxrow >= (pangulu_int64_t)related_nondiag_block_colptr[bcol])
        {
            if ((related_nondiag_block_rowidx[bidx_maxrow] % p) * q + (bcol % q) == rank)
            {
                break;
            }
            if ((related_nondiag_block_rowidx[bidx_maxrow] < bcol) && ((bcol % p) * q + (bcol % q) == rank))
            {
                bidx_maxrow++;
                diag_need_flag = 0;
                break;
            }
            bidx_maxrow--;
        }
        if (bidx_maxrow < (pangulu_int64_t)related_nondiag_block_colptr[bcol])
        {
            continue;
        }
        for (pangulu_exblock_ptr bidx = related_nondiag_block_colptr[bcol]; bidx < bidx_maxrow; bidx++)
        {
            pangulu_exblock_idx brow = related_nondiag_block_rowidx[bidx];
            if (brow >= bcol)
            {
                break;
            }
            if ((brow % p) * q + (bcol % q) != rank)
            {
                bcsc_nondiag_remote_need_flag[bidx] = 1;
            }
        }
        if ((diag_need_flag == 1) && (related_nondiag_block_rowidx[bidx_maxrow] > bcol))
        {
            if ((bcol % p) * q + (bcol % q) != rank)
            {
                bcsc_diag_remote_need_flag[bcol] |= 1;
            }
        }
    }
    for (pangulu_exblock_idx brow = 0; brow < block_length; brow++)
    {
        pangulu_int64_t bidx_maxcol = related_nondiag_block_rowptr[brow + 1] - 1;
        char diag_need_flag = 1;
        while (bidx_maxcol >= (pangulu_int64_t)related_nondiag_block_rowptr[brow])
        {
            if ((brow % p) * q + (related_nondiag_block_colidx[bidx_maxcol] % q) == rank)
            {
                break;
            }
            if ((related_nondiag_block_colidx[bidx_maxcol] < brow) && ((brow % p) * q + (brow % q) == rank))
            {
                bidx_maxcol++;
                diag_need_flag = 0;
                break;
            }
            bidx_maxcol--;
        }
        if (bidx_maxcol < (pangulu_int64_t)related_nondiag_block_rowptr[brow])
        {
            continue;
        }
        for (pangulu_exblock_ptr bidx = related_nondiag_block_rowptr[brow]; bidx < bidx_maxcol; bidx++)
        {
            pangulu_exblock_idx bcol = related_nondiag_block_colidx[bidx];
            if (bcol >= brow)
            {
                break;
            }
            if ((brow % p) * q + (bcol % q) != rank)
            {
                bcsc_nondiag_remote_need_flag[related_nondiag_block_csr_to_csc[bidx]] = 1;
            }
        }
        if ((diag_need_flag == 1) && (related_nondiag_block_colidx[bidx_maxcol] > brow))
        {
            if ((brow % p) * q + (brow % q) != rank)
            {
                bcsc_diag_remote_need_flag[brow] |= 2;
            }
        }
    }
    for (pangulu_exblock_ptr bidx = 0; bidx < related_nondiag_block_colptr[block_length]; bidx++)
    {
        if (bcsc_nondiag_remote_need_flag[bidx] == 1)
        {
            rank_remain_recv_block_count++;
        }
    }
    for (pangulu_exblock_idx level = 0; level < block_length; level++)
    {
        if (bcsc_diag_remote_need_flag[level] & 1)
        {
            rank_remain_recv_block_count++;
        }
        if (bcsc_diag_remote_need_flag[level] & 2)
        {
            rank_remain_recv_block_count++;
        }
    }
    pangulu_free(__FILE__, __LINE__, bcsc_nondiag_remote_need_flag);
    pangulu_free(__FILE__, __LINE__, bcsc_diag_remote_need_flag);

    const int each_bin_minimum_capacity = 100;
    float basic_param = common->basic_param;
#define ALIGN_8(nbyte) (((nbyte) % 8) ? ((nbyte) / 8 * 8 + 8) : (nbyte))
#define BLOCK_MEMSIZE_CSCCSR(nnz) (sizeof(pangulu_inblock_ptr) * 2 * (nb + 1) + (sizeof(pangulu_inblock_idx) * 2 + sizeof(calculate_type) + sizeof(pangulu_inblock_ptr)) * (nnz) + 32)
#define BLOCK_MEMSIZE_CSC(nnz) (sizeof(pangulu_inblock_ptr) * (nb + 1) + (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * (nnz) + 32)
    pangulu_int64_t pangulu_heap_capacity = rank_remain_task_count + 1;
    pangulu_int64_t pangulu_storage_slot_capacity[7] = {
        0,
        ALIGN_8(BLOCK_MEMSIZE_CSC(PANGULU_MIN(5, nb * nb / 1024))),
        ALIGN_8(BLOCK_MEMSIZE_CSC(nb * PANGULU_MIN(nb, 4))),
        ALIGN_8(BLOCK_MEMSIZE_CSCCSR(nb * ((nb / 100) ? (nb / 100) : PANGULU_MIN(nb, 4)))),
        ALIGN_8(BLOCK_MEMSIZE_CSCCSR(nb * ((nb / 50) ? (nb / 50) : PANGULU_MIN(nb, 4)))),
        ALIGN_8(BLOCK_MEMSIZE_CSCCSR(nb * ((nb / 10) ? (nb / 10) : PANGULU_MIN(nb, 4)))),
        ALIGN_8(BLOCK_MEMSIZE_CSCCSR(nb * nb)) + 8};
#undef BLOCK_MEMSIZE_CSCCSR
#undef BLOCK_MEMSIZE_CSC
#undef ALIGN_8

    pangulu_int32_t pangulu_storage_slot_count[7];
    if (nproc == 1)
    {
        pangulu_int32_t tmp[7] = {
            0,
            0, 0, 0, 0, 0, 0};
        memcpy(pangulu_storage_slot_count, tmp, sizeof(pangulu_int32_t) * 7);
    }
    else
    {
        pangulu_int32_t tmp[7] = {
            0,
            basic_param * block_length / (PANGULU_MAX(1, nproc / (p + q))) * 0.1,
            basic_param * block_length / (PANGULU_MAX(1, nproc / (p + q))) * 1.5,
            basic_param * block_length / (PANGULU_MAX(1, nproc / (p + q))) * 0.5,
            basic_param * block_length / (PANGULU_MAX(1, nproc / (p + q))) * 0.25,
            basic_param * block_length / (PANGULU_MAX(1, nproc / (p + q))) * 0.5,
            PANGULU_MAX(60 * basic_param * block_length / nb / PANGULU_MAX(1, nproc / (p + q)), 5)};
        char warning_flag = 0;
        for (int bin = 1; bin < 7; bin++)
        {
            tmp[bin] = PANGULU_MAX(tmp[bin], each_bin_minimum_capacity);
            warning_flag = 1;
        }
        if (rank == 0)
        {
            printf(PANGULU_W_INSUFFICIENT_MPI_BUF);
        }
        memcpy(pangulu_storage_slot_count, tmp, sizeof(pangulu_int32_t) * 7);
    }

    bsmatrix->storage = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_storage_t));
    pangulu_storage_init(
        bsmatrix->storage,
        pangulu_storage_slot_capacity,
        pangulu_storage_slot_count,

        block_length,
        nondiag_block_colptr,
        nondiag_block_rowidx,
        nondiag_colptr,
        nondiag_rowidx,
        nondiag_cscvalue,
        nondiag_rowptr,
        nondiag_colidx,
        nondiag_csr_to_csc,

        diag_uniaddr,
        diag_upper_rowptr,
        diag_upper_colidx,
        diag_upper_csrvalue,
        diag_lower_colptr,
        diag_lower_rowidx,
        diag_lower_cscvalue,

        nb);
    pangulu_uint64_t *related_nondiag_uniaddr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_uint64_t) * related_nondiag_block_colptr[block_length]);
    pangulu_uint64_t local_block_cnt = 0;
    pangulu_exblock_ptr *related_nondiag_fstblk_idx_after_diag = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * block_length);
    for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
    {
        char cross_diag_flag = 0;
        for (pangulu_exblock_ptr bidx = related_nondiag_block_colptr[bcol]; bidx < related_nondiag_block_colptr[bcol + 1]; bidx++)
        {
            pangulu_exblock_idx brow = related_nondiag_block_rowidx[bidx];
            if ((brow % p == rank / q) && (bcol % q == rank % q))
            {
                pangulu_int64_t nnz = bsmatrix->storage->bins[0].slots[local_block_cnt].columnpointer[nb];
                related_nondiag_uniaddr[bidx] = (PANGULU_DIGINFO_SET_BINID(0) | PANGULU_DIGINFO_SET_NNZ(nnz) | PANGULU_DIGINFO_SET_SLOT_IDX(local_block_cnt));
                local_block_cnt++;
            }
            else
            {
                related_nondiag_uniaddr[bidx] = PANGULU_DIGINFO_SET_BINID(7);
            }
            if ((brow > bcol) && (cross_diag_flag == 0))
            {
                related_nondiag_fstblk_idx_after_diag[bcol] = bidx;
                cross_diag_flag = 1;
            }
        }
        if (!cross_diag_flag)
        {
            related_nondiag_fstblk_idx_after_diag[bcol] = related_nondiag_block_colptr[bcol + 1];
        }
    }
    pangulu_exblock_ptr *related_nondiag_fstblk_idx_after_diag_csr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * block_length);
    for (pangulu_exblock_idx brow = 0; brow < block_length; brow++)
    {
        char cross_diag_flag = 0;
        for (pangulu_exblock_ptr bidx = related_nondiag_block_rowptr[brow]; bidx < related_nondiag_block_rowptr[brow + 1]; bidx++)
        {
            pangulu_exblock_idx bcol = related_nondiag_block_colidx[bidx];
            if ((bcol > brow) && (cross_diag_flag == 0))
            {
                related_nondiag_fstblk_idx_after_diag_csr[brow] = bidx;
                cross_diag_flag = 1;
                break;
            }
        }
        if (!cross_diag_flag)
        {
            related_nondiag_fstblk_idx_after_diag_csr[brow] = related_nondiag_block_rowptr[brow + 1];
        }
    }

    for (pangulu_exblock_idx level = 0; level < block_length; level++)
    {
        if ((level % p) * q + (level % q) != rank)
        {
            for (pangulu_exblock_ptr csr_idx = related_nondiag_fstblk_idx_after_diag_csr[level]; csr_idx < related_nondiag_block_rowptr[level + 1]; csr_idx++)
            {
                pangulu_exblock_idx col = related_nondiag_block_colidx[csr_idx];
                if ((level % p) * q + (col % q) == rank)
                {
                    diag_remain_task_count[level] += 1;
                }
            }
            for (pangulu_exblock_ptr idx = related_nondiag_fstblk_idx_after_diag[level]; idx < related_nondiag_block_colptr[level + 1]; idx++)
            {
                pangulu_exblock_idx row = related_nondiag_block_rowidx[idx];
                if ((row % p) * q + (level % q) == rank)
                {
                    diag_remain_task_count[level] += 1;
                }
            }
        }
    }

    for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
    {
        for (pangulu_exblock_ptr bidx = related_nondiag_block_colptr[bcol]; bidx < related_nondiag_block_colptr[bcol + 1]; bidx++)
        {
            pangulu_exblock_idx brow = related_nondiag_block_rowidx[bidx];
            if (((brow % p) * q + (bcol % q)) != rank)
            {
                nondiag_remain_task_count[bidx] = 0;
                if (brow > bcol)
                {
                    pangulu_exblock_idx schur_level = bcol;
                    pangulu_exblock_ptr bidx1 = related_nondiag_fstblk_idx_after_diag_csr[schur_level];
                    pangulu_exblock_ptr bidx1_ub = related_nondiag_block_rowptr[schur_level + 1];
                    pangulu_exblock_ptr bidx2 = related_nondiag_block_rowptr[brow];
                    pangulu_exblock_ptr bidx2_ub = related_nondiag_block_rowptr[brow + 1];
                    while (bidx1 < bidx1_ub && bidx2 < bidx2_ub)
                    {
                        if (related_nondiag_block_colidx[bidx1] == related_nondiag_block_colidx[bidx2])
                        {
                            if (((brow % p) * q + (related_nondiag_block_colidx[bidx2] % q)) == rank)
                            {
                                nondiag_remain_task_count[bidx]++;
                            }
                            bidx1++;
                            bidx2++;
                        }
                        while (bidx1 < bidx1_ub && related_nondiag_block_colidx[bidx1] < related_nondiag_block_colidx[bidx2])
                        {
                            if ((related_nondiag_block_colidx[bidx1] == brow) && ((brow % p) * q + (brow % q) == rank))
                            {
                                nondiag_remain_task_count[bidx]++;
                            }
                            bidx1++;
                        }
                        while (bidx2 < bidx2_ub && related_nondiag_block_colidx[bidx2] < related_nondiag_block_colidx[bidx1])
                        {
                            bidx2++;
                        }
                    }
                    while (bidx1 < bidx1_ub)
                    {
                        if ((related_nondiag_block_colidx[bidx1] == brow) && ((brow % p) * q + (brow % q) == rank))
                        {
                            nondiag_remain_task_count[bidx]++;
                        }
                        bidx1++;
                    }
                }
                else
                {
                    pangulu_exblock_idx schur_level = brow;
                    pangulu_exblock_ptr bidx1 = related_nondiag_fstblk_idx_after_diag[schur_level];
                    pangulu_exblock_ptr bidx1_ub = related_nondiag_block_colptr[schur_level + 1];
                    pangulu_exblock_ptr bidx2 = related_nondiag_block_colptr[bcol];
                    pangulu_exblock_ptr bidx2_ub = related_nondiag_block_colptr[bcol + 1];
                    while (bidx1 < bidx1_ub && bidx2 < bidx2_ub)
                    {
                        if (related_nondiag_block_rowidx[bidx1] == related_nondiag_block_rowidx[bidx2])
                        {
                            if (((related_nondiag_block_rowidx[bidx2] % p) * q + (bcol % q)) == rank)
                            {
                                nondiag_remain_task_count[bidx]++;
                            }
                            bidx1++;
                            bidx2++;
                        }
                        while (bidx1 < bidx1_ub && related_nondiag_block_rowidx[bidx1] < related_nondiag_block_rowidx[bidx2])
                        {
                            if ((related_nondiag_block_rowidx[bidx1] == bcol) && ((bcol % p) * q + (bcol % q) == rank))
                            {
                                nondiag_remain_task_count[bidx]++;
                            }
                            bidx1++;
                        }
                        while (bidx2 < bidx2_ub && related_nondiag_block_rowidx[bidx2] < related_nondiag_block_rowidx[bidx1])
                        {
                            bidx2++;
                        }
                    }
                    while (bidx1 < bidx1_ub)
                    {
                        if ((related_nondiag_block_rowidx[bidx1] == bcol) && ((bcol % p) * q + (bcol % q) == rank))
                        {
                            nondiag_remain_task_count[bidx]++;
                        }
                        bidx1++;
                    }
                }
            }
        }
    }

    bsmatrix->heap = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_task_queue_t));
    pangulu_task_queue_init(bsmatrix->heap, pangulu_heap_capacity);

    bsmatrix->rank_remain_task_count = rank_remain_task_count;
    bsmatrix->rank_remain_recv_block_count = rank_remain_recv_block_count;
    bsmatrix->nondiag_remain_task_count = nondiag_remain_task_count;
    bsmatrix->diag_remain_task_count = diag_remain_task_count;
    bsmatrix->diag_uniaddr = diag_uniaddr;
    bsmatrix->diag_upper_rowptr = diag_upper_rowptr;
    bsmatrix->diag_upper_colidx = diag_upper_colidx;
    bsmatrix->diag_upper_csrvalue = diag_upper_csrvalue;
    bsmatrix->diag_lower_colptr = diag_lower_colptr;
    bsmatrix->diag_lower_rowidx = diag_lower_rowidx;
    bsmatrix->diag_lower_cscvalue = diag_lower_cscvalue;
    bsmatrix->nondiag_cscvalue = nondiag_cscvalue;
    bsmatrix->nondiag_colptr = nondiag_colptr;
    bsmatrix->nondiag_rowidx = nondiag_rowidx;
    bsmatrix->nondiag_csr_to_csc = nondiag_csr_to_csc;
    bsmatrix->nondiag_rowptr = nondiag_rowptr;
    bsmatrix->nondiag_colidx = nondiag_colidx;
    bsmatrix->nondiag_block_colptr = nondiag_block_colptr;
    bsmatrix->nondiag_block_rowidx = nondiag_block_rowidx;
    bsmatrix->nondiag_block_rowptr = nondiag_block_rowptr;
    bsmatrix->nondiag_block_colidx = nondiag_block_colidx;
    bsmatrix->nondiag_block_csr_to_csc = nondiag_block_csr_to_csc;
    bsmatrix->related_nondiag_block_colptr = related_nondiag_block_colptr;
    bsmatrix->related_nondiag_block_rowidx = related_nondiag_block_rowidx;
    bsmatrix->related_nondiag_uniaddr = related_nondiag_uniaddr;
    bsmatrix->related_nondiag_block_rowptr = related_nondiag_block_rowptr;
    bsmatrix->related_nondiag_block_colidx = related_nondiag_block_colidx;
    bsmatrix->related_nondiag_block_csr_to_csc = related_nondiag_block_csr_to_csc;
    bsmatrix->related_nondiag_fstblk_idx_after_diag = related_nondiag_fstblk_idx_after_diag;
    bsmatrix->related_nondiag_fstblk_idx_after_diag_csr = related_nondiag_fstblk_idx_after_diag_csr;
    bsmatrix->sent_rank_flag = pangulu_malloc(__FILE__, __LINE__, sizeof(char) * nproc);
    bsmatrix->info_mutex = pangulu_malloc(__FILE__, __LINE__, sizeof(pthread_mutex_t));
    pthread_mutex_init(bsmatrix->info_mutex, NULL);
    bsmatrix->aggregate_batch_tileid = NULL;
    bsmatrix->aggregate_batch_tileid_capacity = 0;
    ssssm_hash_lu = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t *) * common->omp_thread);
    memset(ssssm_hash_lu, 0, sizeof(pangulu_int32_t *) * common->omp_thread);
    ssssm_hash_l_row = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t *) * common->omp_thread);
    memset(ssssm_hash_l_row, 0, sizeof(pangulu_int32_t *) * common->omp_thread);
    ssssm_hash_l_row_inv = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t *) * common->omp_thread);
    memset(ssssm_hash_l_row_inv, 0, sizeof(pangulu_int32_t *) * common->omp_thread);
    ssssm_hash_u_col = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t *) * common->omp_thread);
    memset(ssssm_hash_u_col, 0, sizeof(pangulu_int32_t *) * common->omp_thread);
    ssssm_hash_u_col_inv = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t *) * common->omp_thread);
    memset(ssssm_hash_u_col_inv, 0, sizeof(pangulu_int32_t *) * common->omp_thread);
    ssssm_l_value = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type *) * common->omp_thread);
    memset(ssssm_l_value, 0, sizeof(calculate_type *) * common->omp_thread);
    ssssm_u_value = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type *) * common->omp_thread);
    memset(ssssm_u_value, 0, sizeof(calculate_type *) * common->omp_thread);
    temp_a_value = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type *) * common->omp_thread);
    memset(temp_a_value, 0, sizeof(calculate_type *) * common->omp_thread);
}
