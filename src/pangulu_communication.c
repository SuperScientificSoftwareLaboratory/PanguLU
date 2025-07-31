#include "pangulu_common.h"

void pangulu_cm_rank(pangulu_int32_t *rank) { MPI_Comm_rank(MPI_COMM_WORLD, rank); }
void pangulu_cm_size(pangulu_int32_t *size) { MPI_Comm_size(MPI_COMM_WORLD, size); }
void pangulu_cm_sync() { MPI_Barrier(MPI_COMM_WORLD); }
void pangulu_cm_bcast(void *buffer, int count, MPI_Datatype datatype, int root) { MPI_Bcast(buffer, count, datatype, root, MPI_COMM_WORLD); }
void pangulu_cm_isend(char *buf, pangulu_int64_t count, pangulu_int32_t remote_rank, pangulu_int32_t tag, pangulu_int32_t tag_ub)
{
    MPI_Request req;
    const pangulu_int64_t send_maxlen = 0x7FFFFFFF;
    pangulu_int64_t send_times = PANGULU_ICEIL(count, send_maxlen);
    for (pangulu_int64_t iter = 0; iter < send_times; iter++)
    {
        int count_current = PANGULU_MIN((iter + 1) * send_maxlen, count) - iter * send_maxlen;
        MPI_Isend(buf + iter * send_maxlen, count_current, MPI_CHAR, remote_rank, iter * tag_ub + tag, MPI_COMM_WORLD, &req);
    }
    if (send_times == 0)
    {
        MPI_Isend(buf, 0, MPI_CHAR, remote_rank, tag, MPI_COMM_WORLD, &req);
    }
}
void pangulu_cm_send(char *buf, pangulu_int64_t count, pangulu_int32_t remote_rank, pangulu_int32_t tag, pangulu_int32_t tag_ub)
{
    const pangulu_int64_t send_maxlen = 0x7FFFFFFF;
    pangulu_int64_t send_times = PANGULU_ICEIL(count, send_maxlen);
    for (pangulu_int64_t iter = 0; iter < send_times; iter++)
    {
        int count_current = PANGULU_MIN((iter + 1) * send_maxlen, count) - iter * send_maxlen;
        MPI_Send(buf + iter * send_maxlen, count_current, MPI_CHAR, remote_rank, iter * tag_ub + tag, MPI_COMM_WORLD);
    }
    if (send_times == 0)
    {
        MPI_Send(buf, 0, MPI_CHAR, remote_rank, tag, MPI_COMM_WORLD);
    }
}
void pangulu_cm_recv(char *buf, pangulu_int64_t count, pangulu_int32_t fetch_rank, pangulu_int32_t tag, pangulu_int32_t tag_ub)
{
    MPI_Status stat;
    const pangulu_int64_t send_maxlen = 0x7FFFFFFF;
    pangulu_int64_t recv_times = PANGULU_ICEIL(count, send_maxlen);
    for (pangulu_int64_t iter = 0; iter < recv_times; iter++)
    {
        int count_current = PANGULU_MIN((iter + 1) * send_maxlen, count) - iter * send_maxlen;
        MPI_Recv(buf + iter * send_maxlen, count_current, MPI_CHAR, fetch_rank, iter * tag_ub + tag, MPI_COMM_WORLD, &stat);
    }
    if (recv_times == 0)
    {
        MPI_Recv(buf, 0, MPI_CHAR, fetch_rank, tag, MPI_COMM_WORLD, &stat);
    }
}
void pangulu_cm_sync_asym(int wake_rank)
{
    pangulu_int32_t sum_rank_size, rank;
    pangulu_cm_rank(&rank);
    pangulu_cm_size(&sum_rank_size);
    if (rank == wake_rank)
    {
        for (int i = 0; i < sum_rank_size; i++)
        {
            if (i != wake_rank)
            {
                MPI_Send(&sum_rank_size, 1, MPI_INT, i, 0xCAFE, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        int mpi_buf_int;
        int mpi_flag;
        MPI_Status mpi_stat;
        while (1)
        {
            mpi_flag = 0;
            MPI_Iprobe(wake_rank, 0xCAFE, MPI_COMM_WORLD, &mpi_flag, &mpi_stat);
            if (mpi_flag != 0 && mpi_stat.MPI_TAG == 0xCAFE)
            {
                MPI_Recv(&mpi_buf_int, 1, MPI_INT, wake_rank, 0xCAFE, MPI_COMM_WORLD, &mpi_stat);
                if (mpi_buf_int == sum_rank_size)
                {
                    break;
                }
                else
                {
                    printf(PANGULU_E_ASYM);
                    exit(1);
                }
            }
            usleep(50);
        }
    }
    pangulu_cm_sync();
}
void pangulu_cm_probe(MPI_Status *status)
{
    int have_msg = 0;
    do
    {
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &have_msg, status);
        if (have_msg)
        {
            return;
        }
        usleep(10);
    } while (!have_msg);
}

void pangulu_cm_distribute_csc_to_distcsc(
    pangulu_int32_t root_rank,
    int rootproc_free_originmatrix,
    pangulu_exblock_idx *n,
    pangulu_inblock_idx rowchunk_align,
    pangulu_int32_t *distcsc_nproc,
    pangulu_exblock_idx *n_loc,
    pangulu_exblock_ptr **distcsc_proc_nnzptr,
    pangulu_exblock_ptr **distcsc_pointer,
    pangulu_exblock_idx **distcsc_index,
    calculate_type **distcsc_value)
{
    struct timeval start_time;
    pangulu_time_start(&start_time);
    pangulu_int32_t nproc = 0;
    pangulu_int32_t rank;
    pangulu_cm_size(&nproc);
    pangulu_cm_rank(&rank);
    *distcsc_nproc = nproc;
    pangulu_int64_t p = sqrt(nproc);
    while ((nproc % p) != 0)
    {
        p--;
    }
    pangulu_int64_t q = nproc / p;
    pangulu_cm_bcast(n, 1, MPI_PANGULU_EXBLOCK_IDX, root_rank);
    rowchunk_align *= q;
    pangulu_exblock_idx col_per_rank = PANGULU_ICEIL(PANGULU_ICEIL(*n, rowchunk_align), nproc) * rowchunk_align;

    *distcsc_proc_nnzptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (nproc + 1));
    if (rank == root_rank)
    {
        pangulu_exblock_ptr *columnpointer = *distcsc_pointer;
        pangulu_exblock_idx *rowindex = *distcsc_index;
        calculate_type *value_csc = NULL;
        if (distcsc_value)
        {
            value_csc = *distcsc_value;
        }
        (*distcsc_proc_nnzptr)[0] = 0;
        for (pangulu_int32_t target_rank = 0; target_rank < nproc; target_rank++)
        {
            pangulu_exblock_idx n_loc_remote = PANGULU_MIN(col_per_rank * (target_rank + 1), *n) - PANGULU_MIN(col_per_rank * target_rank, *n);
            (*distcsc_proc_nnzptr)[target_rank + 1] = columnpointer[PANGULU_MIN(col_per_rank * (target_rank + 1), *n)];
            if (rank == target_rank)
            {
                *n_loc = n_loc_remote;
                *distcsc_pointer = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (*n_loc + 1));
                memcpy(*distcsc_pointer, &columnpointer[PANGULU_MIN(col_per_rank * target_rank, *n)], sizeof(pangulu_exblock_ptr) * (*n_loc + 1));
                pangulu_exblock_idx col_offset = (*distcsc_pointer)[0];
                for (pangulu_exblock_idx col = 0; col < *n_loc + 1; col++)
                {
                    (*distcsc_pointer)[col] -= col_offset;
                }
                pangulu_exblock_ptr nnz_loc = (*distcsc_pointer)[*n_loc];
                *distcsc_index = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * nnz_loc);
                memcpy(*distcsc_index, &rowindex[(*distcsc_proc_nnzptr)[target_rank]], sizeof(pangulu_exblock_idx) * nnz_loc);

                if (distcsc_value)
                {
                    *distcsc_value = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz_loc);
                    memcpy(*distcsc_value, &value_csc[(*distcsc_proc_nnzptr)[target_rank]], sizeof(calculate_type) * nnz_loc);
                }
            }
            else
            {
                MPI_Send(&n_loc_remote, 1, MPI_PANGULU_EXBLOCK_IDX, target_rank, 0, MPI_COMM_WORLD);
                MPI_Send(&columnpointer[PANGULU_MIN(col_per_rank * target_rank, *n)], n_loc_remote + 1, MPI_PANGULU_EXBLOCK_PTR, target_rank, 1, MPI_COMM_WORLD);
                pangulu_cm_send(&rowindex[(*distcsc_proc_nnzptr)[target_rank]], sizeof(pangulu_exblock_idx) * ((*distcsc_proc_nnzptr)[target_rank + 1] - (*distcsc_proc_nnzptr)[target_rank]), target_rank, 2, 10);
                if (distcsc_value)
                {
                    pangulu_cm_send(&value_csc[(*distcsc_proc_nnzptr)[target_rank]], sizeof(calculate_type) * ((*distcsc_proc_nnzptr)[target_rank + 1] - (*distcsc_proc_nnzptr)[target_rank]), target_rank, 3, 10);
                }
                else
                {
                    int nouse = 0;
                    MPI_Send(&nouse, 1, MPI_INT, target_rank, 4, MPI_COMM_WORLD);
                }
            }
        }
        MPI_Bcast(*distcsc_proc_nnzptr, nproc + 1, MPI_PANGULU_EXBLOCK_PTR, root_rank, MPI_COMM_WORLD);
        if (rootproc_free_originmatrix)
        {
            pangulu_free(__FILE__, __LINE__, columnpointer);
            pangulu_free(__FILE__, __LINE__, rowindex);
            if (distcsc_value)
            {
                pangulu_free(__FILE__, __LINE__, value_csc);
            }
        }
    }
    else
    {
        MPI_Status mpi_stat;
        MPI_Recv(n_loc, 1, MPI_PANGULU_EXBLOCK_IDX, root_rank, 0, MPI_COMM_WORLD, &mpi_stat);
        *distcsc_pointer = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (*n_loc + 1));
        MPI_Recv(*distcsc_pointer, *n_loc + 1, MPI_PANGULU_EXBLOCK_PTR, root_rank, 1, MPI_COMM_WORLD, &mpi_stat);
        pangulu_exblock_idx col_offset = (*distcsc_pointer)[0];
        for (pangulu_exblock_idx col = 0; col < *n_loc + 1; col++)
        {
            (*distcsc_pointer)[col] -= col_offset;
        }
        pangulu_exblock_ptr nnz_loc = (*distcsc_pointer)[*n_loc];
        *distcsc_index = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * nnz_loc);
        pangulu_cm_recv(*distcsc_index, sizeof(pangulu_exblock_idx) * nnz_loc, root_rank, 2, 10);
        MPI_Probe(root_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &mpi_stat);
        if ((mpi_stat.MPI_TAG % 10) == 3)
        {
            *distcsc_value = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz_loc);
            pangulu_cm_recv(*distcsc_value, sizeof(calculate_type) * nnz_loc, root_rank, 3, 10);
        }
        else if (mpi_stat.MPI_TAG == 4)
        {
            int nouse = 0;
            MPI_Recv(&nouse, 1, MPI_INT, root_rank, 4, MPI_COMM_WORLD, &mpi_stat);
        }
        MPI_Bcast(*distcsc_proc_nnzptr, nproc + 1, MPI_PANGULU_EXBLOCK_PTR, root_rank, MPI_COMM_WORLD);
    }
}

void pangulu_cm_distribute_csc_to_distbcsc(
    pangulu_exblock_idx n,
    pangulu_inblock_idx nb,
    pangulu_exblock_ptr *csc_pointer,
    pangulu_exblock_idx *csc_index,
    calculate_type *csc_value,
    pangulu_exblock_ptr **bcsc_struct_pointer,
    pangulu_exblock_idx **bcsc_struct_index,
    pangulu_exblock_ptr **bcsc_struct_nnzptr,
    pangulu_inblock_ptr ***bcsc_inblock_pointers,
    pangulu_inblock_idx ***bcsc_inblock_indeces,
    calculate_type ***bcsc_values,
    pangulu_exblock_ptr **bcsc_global_pointer,
    pangulu_exblock_idx **bcsc_global_index)
{
#define _PANGULU_SET_BVALUE_SIZE(size) ((csc_value) ? (size) : (0))

    struct timeval start_time, total_time;
    pangulu_time_start(&total_time);

    pangulu_uint64_t nzblk;
    char *bstruct_csc = NULL;
    char *block_csc = NULL;
    char have_value;

    pangulu_int32_t rank, nproc;
    pangulu_cm_rank(&rank);
    pangulu_cm_size(&nproc);

    pangulu_int64_t p = sqrt(nproc);
    while ((nproc % p) != 0)
    {
        p--;
    }
    pangulu_int64_t q = nproc / p;
    pangulu_int64_t block_length = PANGULU_ICEIL(n, nb);

    if (rank == 0)
    {
        pangulu_time_start(&start_time);
        int nthreads = sysconf(_SC_NPROCESSORS_ONLN);
#pragma omp parallel num_threads(nthreads)
        {
            pangulu_bind_to_core(omp_get_thread_num());
        }

        pangulu_exblock_ptr nnz = csc_pointer[n];
        pangulu_int64_t bit_length = (block_length + 31) / 32;
        pangulu_int64_t block_num = 0;
        pangulu_int64_t *block_nnz_pt;

        pangulu_int64_t avg_nnz = PANGULU_ICEIL(nnz, nthreads);
        pangulu_int64_t *block_row_nnz_pt = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (block_length + 1));
        for (int i = 0; i < block_length; i++)
        {
            block_row_nnz_pt[i] = csc_pointer[PANGULU_MIN(i * nb, n)];
        }
        block_row_nnz_pt[block_length] = csc_pointer[n];

        int *thread_pt = (int *)pangulu_malloc(__FILE__, __LINE__, sizeof(int) * (nthreads + 1));
        thread_pt[0] = 0;
        for (int i = 1; i < nthreads + 1; i++)
        {
            thread_pt[i] = pangulu_binarylowerbound(block_row_nnz_pt, block_length, avg_nnz * i);
        }
        pangulu_free(__FILE__, __LINE__, block_row_nnz_pt);
        block_row_nnz_pt = NULL;

        pangulu_int64_t *block_row_pt = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (block_length + 1));
        memset(block_row_pt, 0, sizeof(pangulu_int64_t) * (block_length + 1));

        unsigned int *bit_array = (unsigned int *)pangulu_malloc(__FILE__, __LINE__, sizeof(unsigned int) * bit_length * nthreads);

#pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            unsigned int *tmp_bit = bit_array + bit_length * tid;

            for (int level = thread_pt[tid]; level < thread_pt[tid + 1]; level++)
            {
                memset(tmp_bit, 0, sizeof(unsigned int) * bit_length);

                int start_row = level * nb;
                int end_row = ((level + 1) * nb) < n ? ((level + 1) * nb) : n;

                for (int rid = start_row; rid < end_row; rid++)
                {
                    for (pangulu_int64_t idx = csc_pointer[rid]; idx < csc_pointer[rid + 1]; idx++)
                    {
                        pangulu_int32_t colidx = csc_index[idx];
                        pangulu_int32_t block_cid = colidx / nb;
                        PANGULU_SETBIT(tmp_bit[block_cid / 32], block_cid % 32);
                    }
                }

                pangulu_int64_t tmp_blocknum = 0;
                for (int i = 0; i < bit_length; i++)
                {
                    tmp_blocknum += __builtin_popcount(tmp_bit[i]);
                }

                block_row_pt[level] = tmp_blocknum;
            }
        }
        pangulu_exclusive_scan_1(block_row_pt, block_length + 1);
        block_num = block_row_pt[block_length];

        block_nnz_pt = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (block_num + 1));
        memset(block_nnz_pt, 0, sizeof(pangulu_int64_t) * (block_num + 1));
        pangulu_int32_t *block_col_idx = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * block_num);

        int *count_array = (int *)pangulu_malloc(__FILE__, __LINE__, sizeof(int) * block_length * nthreads);
#pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            unsigned int *tmp_bit = bit_array + bit_length * tid;
            int *tmp_count = count_array + block_length * tid;

            for (int level = thread_pt[tid]; level < thread_pt[tid + 1]; level++)
            {
                memset(tmp_bit, 0, sizeof(unsigned int) * bit_length);
                memset(tmp_count, 0, sizeof(int) * block_length);

                pangulu_int64_t *cur_block_nnz_pt = block_nnz_pt + block_row_pt[level];
                pangulu_int32_t *cur_block_col_idx = block_col_idx + block_row_pt[level];

                int start_row = level * nb;
                int end_row = ((level + 1) * nb) < n ? ((level + 1) * nb) : n;

                for (int rid = start_row; rid < end_row; rid++)
                {
                    for (pangulu_int64_t idx = csc_pointer[rid]; idx < csc_pointer[rid + 1]; idx++)
                    {
                        pangulu_int32_t colidx = csc_index[idx];
                        pangulu_int32_t block_cid = colidx / nb;
                        PANGULU_SETBIT(tmp_bit[block_cid / 32], block_cid % 32);
                        tmp_count[block_cid]++;
                    }
                }

                pangulu_int64_t cnt = 0;
                for (int i = 0; i < block_length; i++)
                {
                    if (PANGULU_GETBIT(tmp_bit[i / 32], i % 32))
                    {
                        cur_block_nnz_pt[cnt] = tmp_count[i];
                        cur_block_col_idx[cnt] = i;
                        cnt++;
                    }
                }
            }
        }
        pangulu_free(__FILE__, __LINE__, bit_array);
        bit_array = NULL;
        pangulu_free(__FILE__, __LINE__, count_array);
        count_array = NULL;
        pangulu_exclusive_scan_1(block_nnz_pt, block_num + 1);

        pangulu_time_start(&start_time);
        pangulu_exblock_ptr *nzblk_each_rank_ptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (nproc + 1));
        pangulu_exblock_ptr *nnz_each_rank_ptr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (nproc + 1));
        memset(nzblk_each_rank_ptr, 0, sizeof(pangulu_exblock_ptr) * (nproc + 1));
        memset(nnz_each_rank_ptr, 0, sizeof(pangulu_exblock_ptr) * (nproc + 1));
        for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
        {
            for (pangulu_exblock_ptr bidx = block_row_pt[bcol]; bidx < block_row_pt[bcol + 1]; bidx++)
            {
                pangulu_exblock_idx brow = block_col_idx[bidx];
                nzblk_each_rank_ptr[(brow % p) * q + (bcol % q) + 1]++;
                nnz_each_rank_ptr[(brow % p) * q + (bcol % q) + 1] += (block_nnz_pt[bidx + 1] - block_nnz_pt[bidx]);
            }
        }
        for (pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++)
        {
            nzblk_each_rank_ptr[remote_rank + 1] += nzblk_each_rank_ptr[remote_rank];
            nnz_each_rank_ptr[remote_rank + 1] += nnz_each_rank_ptr[remote_rank];
        }
        char **csc_draft_remote = pangulu_malloc(__FILE__, __LINE__, sizeof(char *) * nproc);
        for (pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++)
        {
            csc_draft_remote[remote_rank] = pangulu_malloc(
                __FILE__, __LINE__,
                sizeof(pangulu_exblock_ptr) * (block_length + 1) +
                    sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]) +
                    sizeof(pangulu_exblock_ptr) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank] + 1));
            memset(csc_draft_remote[remote_rank], 0,
                   sizeof(pangulu_exblock_ptr) * (block_length + 1) +
                       sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]) +
                       sizeof(pangulu_exblock_ptr) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank] + 1));
        }

        for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
        {
            for (pangulu_exblock_ptr bidx = block_row_pt[bcol]; bidx < block_row_pt[bcol + 1]; bidx++)
            {
                pangulu_exblock_idx brow = block_col_idx[bidx];
                pangulu_inblock_ptr nnz_in_blk = block_nnz_pt[bidx + 1] - block_nnz_pt[bidx];
                pangulu_int32_t remote_rank = (brow % p) * q + (bcol % q);
                pangulu_exblock_ptr *remote_bcolptr = csc_draft_remote[remote_rank];
                remote_bcolptr[bcol + 1]++;
            }
        }

        pangulu_exblock_ptr *aid_arr_colptr_remote = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (block_length + 1) * nproc);
        for (pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++)
        {
            pangulu_exblock_ptr *remote_bcolptr = csc_draft_remote[remote_rank];
            for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
            {
                remote_bcolptr[bcol + 1] += remote_bcolptr[bcol];
            }
            memcpy(&aid_arr_colptr_remote[(block_length + 1) * remote_rank], remote_bcolptr, sizeof(pangulu_exblock_ptr) * (block_length + 1));
        }

        for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
        {
            for (pangulu_exblock_ptr bidx = block_row_pt[bcol]; bidx < block_row_pt[bcol + 1]; bidx++)
            {
                pangulu_exblock_idx brow = block_col_idx[bidx];
                pangulu_inblock_ptr nnz_in_blk = block_nnz_pt[bidx + 1] - block_nnz_pt[bidx];
                pangulu_int32_t remote_rank = (brow % p) * q + (bcol % q);

                pangulu_exblock_idx *remote_browidx =
                    csc_draft_remote[remote_rank] +
                    sizeof(pangulu_exblock_ptr) * (block_length + 1);
                pangulu_exblock_ptr *remote_blknnzptr =
                    csc_draft_remote[remote_rank] +
                    sizeof(pangulu_exblock_ptr) * (block_length + 1) +
                    sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]);

                remote_browidx[aid_arr_colptr_remote[(block_length + 1) * remote_rank + bcol]] = brow;
                remote_blknnzptr[aid_arr_colptr_remote[(block_length + 1) * remote_rank + bcol] + 1] = nnz_in_blk;
                aid_arr_colptr_remote[(block_length + 1) * remote_rank + bcol]++;
            }
        }
        pangulu_free(__FILE__, __LINE__, aid_arr_colptr_remote);
        aid_arr_colptr_remote = NULL;
        pangulu_free(__FILE__, __LINE__, block_nnz_pt);
        block_nnz_pt = NULL;

        for (pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++)
        {
            pangulu_exblock_ptr *remote_bcolptr = csc_draft_remote[remote_rank];
            pangulu_exblock_ptr *remote_blknnzptr =
                csc_draft_remote[remote_rank] +
                sizeof(pangulu_exblock_ptr) * (block_length + 1) +
                sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]);
            for (pangulu_inblock_ptr bidx = 0; bidx < remote_bcolptr[block_length]; bidx++)
            {
                remote_blknnzptr[bidx + 1] += remote_blknnzptr[bidx];
            }
        }

        pangulu_time_start(&start_time);
        char **block_csc_remote = pangulu_malloc(__FILE__, __LINE__, sizeof(char *) * nproc);
#pragma omp parallel num_threads(nthreads)
        {
            pangulu_int32_t tid = omp_get_thread_num();
            if (tid < nproc)
            {
                pangulu_int32_t target_rank = tid;
                pangulu_int32_t blk_diff = nzblk_each_rank_ptr[target_rank + 1] - nzblk_each_rank_ptr[target_rank];
                pangulu_int32_t nnz_diff = nnz_each_rank_ptr[target_rank + 1] - nnz_each_rank_ptr[target_rank];

                size_t alloc_size = sizeof(pangulu_inblock_ptr) * (nb + 1) * blk_diff +
                                    sizeof(pangulu_inblock_idx) * nnz_diff +
                                    _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type)) * nnz_diff;

                block_csc_remote[target_rank] = pangulu_malloc(__FILE__, __LINE__, alloc_size);
                memset(block_csc_remote[target_rank], 0, alloc_size);
            }
        }

        pangulu_time_start(&start_time);
#pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            int *tmp_count = pangulu_malloc(__FILE__, __LINE__, sizeof(int) * block_length * q);

            for (int level = thread_pt[tid]; level < thread_pt[tid + 1]; level++)
            {
                memset(tmp_count, 0, sizeof(int) * block_length * q);
                pangulu_exblock_idx start_col = level * nb;
                pangulu_exblock_idx end_col = ((level + 1) * nb) < n ? ((level + 1) * nb) : n;
                for (pangulu_exblock_idx col = start_col, col_in_blc = 0; col < end_col; col++, col_in_blc++)
                {
                    pangulu_int64_t bidx_glo = block_row_pt[level];
                    pangulu_exblock_idx brow = block_col_idx[bidx_glo];
                    pangulu_int32_t target_rank = (brow % p) * q + (level % q);

                    pangulu_exblock_ptr *remote_bcolptr = csc_draft_remote[target_rank];
                    pangulu_exblock_idx *remote_browidx = csc_draft_remote[target_rank] + sizeof(pangulu_exblock_ptr) * (block_length + 1);
                    pangulu_exblock_ptr *remote_bnnzptr =
                        csc_draft_remote[target_rank] +
                        sizeof(pangulu_exblock_ptr) * (block_length + 1) +
                        sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[target_rank + 1] - nzblk_each_rank_ptr[target_rank]);
                    pangulu_int64_t bidx = remote_bcolptr[level];

                    pangulu_inblock_ptr *cur_block_colptr = (pangulu_inblock_ptr *)(block_csc_remote[target_rank] + sizeof(pangulu_inblock_ptr) * bidx * (nb + 1));
                    pangulu_inblock_idx *cur_block_rowidx = (pangulu_inblock_idx *)(block_csc_remote[target_rank] + sizeof(pangulu_inblock_ptr) * remote_bcolptr[block_length] * (nb + 1) + sizeof(pangulu_inblock_idx) * remote_bnnzptr[bidx]);
                    calculate_type *cur_block_value = NULL;
                    if (csc_value)
                    {
                        cur_block_value = (calculate_type *)(block_csc_remote[target_rank] + sizeof(pangulu_inblock_ptr) * remote_bcolptr[block_length] * (nb + 1) + sizeof(pangulu_inblock_idx) * remote_bnnzptr[remote_bcolptr[block_length]] + sizeof(calculate_type) * remote_bnnzptr[bidx]);
                    }

                    pangulu_exblock_ptr reorder_matrix_idx = csc_pointer[col];
                    pangulu_exblock_ptr reorder_matrix_idx_ub = csc_pointer[col + 1];

                    for (pangulu_exblock_ptr idx = csc_pointer[col]; idx < csc_pointer[col + 1]; idx++)
                    {
                        pangulu_exblock_idx row = csc_index[idx];
                        brow = row / nb;
                        if (block_col_idx[bidx_glo] != brow)
                        {
                            bidx_glo++;
                            target_rank = (brow % p) * q + (level % q);

                            remote_bcolptr = csc_draft_remote[target_rank];
                            remote_browidx = csc_draft_remote[target_rank] + sizeof(pangulu_exblock_ptr) * (block_length + 1);
                            remote_bnnzptr =
                                csc_draft_remote[target_rank] +
                                sizeof(pangulu_exblock_ptr) * (block_length + 1) +
                                sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[target_rank + 1] - nzblk_each_rank_ptr[target_rank]);
                            bidx = pangulu_binarysearch(remote_browidx, remote_bcolptr[level], remote_bcolptr[level + 1], brow);

                            cur_block_colptr = (pangulu_inblock_ptr *)(block_csc_remote[target_rank] + sizeof(pangulu_inblock_ptr) * bidx * (nb + 1));
                            cur_block_rowidx = (pangulu_inblock_idx *)(block_csc_remote[target_rank] + sizeof(pangulu_inblock_ptr) * remote_bcolptr[block_length] * (nb + 1) + sizeof(pangulu_inblock_idx) * remote_bnnzptr[bidx]);
                            if (csc_value)
                            {
                                cur_block_value = (calculate_type *)(block_csc_remote[target_rank] + sizeof(pangulu_inblock_ptr) * remote_bcolptr[block_length] * (nb + 1) + sizeof(pangulu_inblock_idx) * remote_bnnzptr[remote_bcolptr[block_length]] + sizeof(calculate_type) * remote_bnnzptr[bidx]);
                            }
                        }
                        if (csc_value)
                        {
                            cur_block_value[tmp_count[(level % q) * block_length + brow]] = csc_value[reorder_matrix_idx];
                        }
                        reorder_matrix_idx++;
                        cur_block_rowidx[tmp_count[(level % q) * block_length + brow]++] = row % nb;
                        cur_block_colptr[col_in_blc]++;
                    }
                }
                for (pangulu_int32_t target_rank = 0; target_rank < nproc; target_rank++)
                {
                    pangulu_exblock_ptr *remote_bcolptr = csc_draft_remote[target_rank];
                    for (pangulu_int64_t bidx = remote_bcolptr[level]; bidx < remote_bcolptr[level + 1]; bidx++)
                    {
                        pangulu_inblock_ptr *cur_block_colptr = (pangulu_inblock_ptr *)(block_csc_remote[target_rank] + sizeof(pangulu_inblock_ptr) * bidx * (nb + 1));
                        pangulu_exclusive_scan_3(cur_block_colptr, nb + 1);
                    }
                }
            }

            pangulu_free(__FILE__, __LINE__, tmp_count);
            tmp_count = NULL;
        }
        pangulu_free(__FILE__, __LINE__, thread_pt);
        thread_pt = NULL;

        if (bcsc_global_pointer)
        {
            *bcsc_global_pointer = block_row_pt;
        }
        else
        {
            pangulu_free(__FILE__, __LINE__, block_row_pt);
            block_row_pt = NULL;
        }
        if (bcsc_global_index)
        {
            *bcsc_global_index = block_col_idx;
        }
        else
        {
            pangulu_free(__FILE__, __LINE__, block_col_idx);
            block_col_idx = NULL;
        }

        pangulu_free(__FILE__, __LINE__, csc_pointer);
        pangulu_free(__FILE__, __LINE__, csc_index);
        if (csc_value)
        {
            pangulu_free(__FILE__, __LINE__, csc_value); // Don't set csc_value to NULL.
        }

        pangulu_time_start(&start_time);
        for (pangulu_int32_t remote_rank = 1; remote_rank < nproc; remote_rank++)
        {
            MPI_Request req;
            pangulu_uint64_t nzblk_remote = nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank];
            if (csc_value)
            {
                nzblk_remote |= (1ULL << 63);
            }
            MPI_Isend(&nzblk_remote, 1, MPI_PANGULU_UINT64_T, remote_rank, 0, MPI_COMM_WORLD, &req);
        }
        nzblk = nzblk_each_rank_ptr[1] - nzblk_each_rank_ptr[0];
        pangulu_cm_sync();
        fflush(stdout);

        pangulu_time_start(&start_time);
        for (pangulu_int32_t remote_rank = 1; remote_rank < nproc; remote_rank++)
        {
            pangulu_cm_isend(
                csc_draft_remote[remote_rank],
                sizeof(pangulu_exblock_ptr) * (block_length + 1) +
                    sizeof(pangulu_exblock_idx) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]) +
                    sizeof(pangulu_exblock_ptr) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank] + 1),
                remote_rank, 1, 10);
        }
        bstruct_csc = csc_draft_remote[0];
        pangulu_cm_sync();

        for (pangulu_int32_t remote_rank = 1; remote_rank < nproc; remote_rank++)
        {
            pangulu_free(__FILE__, __LINE__, csc_draft_remote[remote_rank]);
        }
        pangulu_free(__FILE__, __LINE__, csc_draft_remote);
        csc_draft_remote = NULL;

        pangulu_time_start(&start_time);
        for (pangulu_int32_t remote_rank = 1; remote_rank < nproc; remote_rank++)
        {
            pangulu_int64_t send_size =
                sizeof(pangulu_inblock_ptr) * (nb + 1) * (nzblk_each_rank_ptr[remote_rank + 1] - nzblk_each_rank_ptr[remote_rank]) +
                sizeof(pangulu_inblock_idx) * (nnz_each_rank_ptr[remote_rank + 1] - nnz_each_rank_ptr[remote_rank]) +
                _PANGULU_SET_BVALUE_SIZE(sizeof(calculate_type)) * (nnz_each_rank_ptr[remote_rank + 1] - nnz_each_rank_ptr[remote_rank]);
            pangulu_cm_isend(block_csc_remote[remote_rank], send_size, remote_rank, 2, 10);
        }
        pangulu_free(__FILE__, __LINE__, nzblk_each_rank_ptr);
        nzblk_each_rank_ptr = NULL;
        pangulu_free(__FILE__, __LINE__, nnz_each_rank_ptr);
        nnz_each_rank_ptr = NULL;
        block_csc = block_csc_remote[0];
        pangulu_cm_sync();
        fflush(stdout);

        for (pangulu_int32_t remote_rank = 1; remote_rank < nproc; remote_rank++)
        {
            pangulu_free(__FILE__, __LINE__, block_csc_remote[remote_rank]);
        }
        pangulu_free(__FILE__, __LINE__, block_csc_remote);
        block_csc_remote = NULL;

        pangulu_bind_to_core(rank % sysconf(_SC_NPROCESSORS_ONLN));

        if (csc_value)
        {
            have_value = 1;
        }
        else
        {
            have_value = 0;
        }
    }
    else
    {
        MPI_Status stat;
        MPI_Recv(&nzblk, 1, MPI_PANGULU_UINT64_T, 0, 0, MPI_COMM_WORLD, &stat);
        have_value = (nzblk >> 63);
        nzblk &= ~(1ULL << 63);
        pangulu_cm_sync();

        bstruct_csc = pangulu_malloc(
            __FILE__, __LINE__,
            sizeof(pangulu_exblock_ptr) * (block_length + 1) +
                sizeof(pangulu_exblock_idx) * nzblk +
                sizeof(pangulu_exblock_ptr) * (nzblk + 1));
        pangulu_cm_recv(
            bstruct_csc,
            sizeof(pangulu_exblock_ptr) * (block_length + 1) + sizeof(pangulu_exblock_idx) * nzblk + sizeof(pangulu_exblock_ptr) * (nzblk + 1),
            0, 1, 10);
        pangulu_cm_sync();

        pangulu_exblock_ptr *remote_bcolptr = bstruct_csc;
        pangulu_exblock_idx *remote_browidx = bstruct_csc + sizeof(pangulu_exblock_ptr) * (block_length + 1);
        pangulu_exblock_ptr *remote_bnnzptr =
            bstruct_csc +
            sizeof(pangulu_exblock_ptr) * (block_length + 1) +
            sizeof(pangulu_exblock_idx) * (remote_bcolptr[block_length]);
        pangulu_int64_t recv_size =
            sizeof(pangulu_inblock_ptr) * (nb + 1) * remote_bcolptr[block_length] +
            sizeof(pangulu_inblock_idx) * remote_bnnzptr[remote_bcolptr[block_length]] +
            (have_value ? (sizeof(calculate_type)) : 0) * remote_bnnzptr[remote_bcolptr[block_length]];
        block_csc = pangulu_malloc(__FILE__, __LINE__, recv_size);
        pangulu_cm_recv(block_csc, recv_size, 0, 2, 10);
        pangulu_cm_sync();
    }

    pangulu_time_start(&start_time);
    *bcsc_struct_pointer = bstruct_csc;
    *bcsc_struct_index = bstruct_csc + sizeof(pangulu_exblock_ptr) * (block_length + 1);
    *bcsc_struct_nnzptr = bstruct_csc + sizeof(pangulu_exblock_ptr) * (block_length + 1) +
                          +sizeof(pangulu_exblock_idx) * nzblk;
    pangulu_inblock_ptr **inblock_pointers = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr *) * nzblk);
    pangulu_inblock_idx **inblock_indeces = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx *) * nzblk);
    calculate_type **inblock_values = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type *) * nzblk);
    pangulu_int64_t nnz_skiped = 0;
    pangulu_int64_t nnz_local = 0;
    if (have_value)
    {
        for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
        {
            for (pangulu_exblock_ptr bidx = (*bcsc_struct_pointer)[bcol]; bidx < (*bcsc_struct_pointer)[bcol + 1]; bidx++)
            {
                inblock_pointers[bidx] = block_csc + sizeof(pangulu_inblock_ptr) * bidx * (nb + 1);
                nnz_local += inblock_pointers[bidx][nb];
            }
        }
    }
    for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
    {
        for (pangulu_exblock_ptr bidx = (*bcsc_struct_pointer)[bcol]; bidx < (*bcsc_struct_pointer)[bcol + 1]; bidx++)
        {
            inblock_pointers[bidx] = block_csc + sizeof(pangulu_inblock_ptr) * bidx * (nb + 1);
            inblock_indeces[bidx] = block_csc + sizeof(pangulu_inblock_ptr) * nzblk * (nb + 1) + sizeof(pangulu_inblock_idx) * nnz_skiped;
            if (have_value)
            {
                inblock_values[bidx] = block_csc + sizeof(pangulu_inblock_ptr) * nzblk * (nb + 1) + sizeof(pangulu_inblock_idx) * nnz_local + sizeof(calculate_type) * nnz_skiped;
            }
            nnz_skiped += inblock_pointers[bidx][nb];
        }
    }
    *bcsc_inblock_pointers = inblock_pointers;
    *bcsc_inblock_indeces = inblock_indeces;
    if (have_value)
    {
        *bcsc_values = inblock_values;
    }
    else
    {
        pangulu_free(__FILE__, __LINE__, inblock_values);
    }
#undef _PANGULU_SET_BVALUE_SIZE
}

void pangulu_set_omp_threads_per_process(int total_threads, int total_processes, int process_rank)
{
    int total_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    int base_cpus = total_cpus / total_processes;
    int extra = total_cpus % total_processes;

    int my_cpu_count = base_cpus + (process_rank < extra ? 1 : 0);

    int start_cpu = process_rank * base_cpus + (process_rank < extra ? process_rank : extra);

    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int i = 0; i < my_cpu_count; ++i)
    {
        CPU_SET(start_cpu + i, &mask);
    }

    if (sched_setaffinity(0, sizeof(mask), &mask) != 0)
    {
        perror("sched_setaffinity failed");
    }

    int base_threads = total_threads / total_processes;
    int extra_threads = total_threads % total_processes;
    int threads_for_this_process = base_threads + (process_rank < extra_threads ? 1 : 0);

    omp_set_num_threads(threads_for_this_process);
}

void pangulu_cm_distribute_csc_to_distbcsc_symb(
    pangulu_exblock_idx n,
    pangulu_inblock_idx nb,
    pangulu_exblock_ptr *csc_pointer,
    pangulu_exblock_idx *csc_index,
    pangulu_inblock_ptr ***out_diag_upper_rowptr,
    pangulu_inblock_idx ***out_diag_upper_colidx,
    calculate_type ***out_diag_upper_csrvalue,
    pangulu_inblock_ptr ***out_diag_lower_colptr,
    pangulu_inblock_idx ***out_diag_lower_rowidx,
    calculate_type ***out_diag_lower_cscvalue,
    calculate_type ***out_nondiag_cscvalue,
    pangulu_inblock_ptr ***out_nondiag_colptr,
    pangulu_inblock_idx ***out_nondiag_rowidx,
    pangulu_inblock_ptr ***out_nondiag_csr_to_csc,
    pangulu_inblock_ptr ***out_nondiag_rowptr,
    pangulu_inblock_idx ***out_nondiag_colidx,
    pangulu_exblock_ptr **out_nondiag_block_colptr,
    pangulu_exblock_idx **out_nondiag_block_rowidx,
    pangulu_exblock_ptr **out_nondiag_block_rowptr,
    pangulu_exblock_idx **out_nondiag_block_colidx,
    pangulu_exblock_ptr **out_nondiag_block_csr_to_csc,
    pangulu_exblock_ptr **out_related_nondiag_block_colptr,
    pangulu_exblock_idx **out_related_nondiag_block_rowidx,
    pangulu_exblock_ptr **out_related_nondiag_block_rowptr,
    pangulu_exblock_idx **out_related_nondiag_block_colidx,
    pangulu_exblock_ptr **out_related_nondiag_block_csr_to_csc,
    pangulu_uint64_t **out_diag_uniaddr)
{
    struct timeval start_time, total_time;

    pangulu_int32_t rank, nproc;
    pangulu_cm_rank(&rank);
    pangulu_cm_size(&nproc);

    pangulu_int64_t p = sqrt(nproc);
    while ((nproc % p) != 0)
    {
        p--;
    }
    pangulu_int64_t q = nproc / p;
    pangulu_int64_t block_length = PANGULU_ICEIL(n, nb);
    int nthreads = sysconf(_SC_NPROCESSORS_ONLN);

    pangulu_set_omp_threads_per_process(nthreads, nproc, rank);

    if (rank == 0)
    {
        pangulu_exblock_ptr nnz = csc_pointer[n];
        pangulu_int64_t bit_length = (block_length + 31) / 32;
        pangulu_int64_t block_num = 0;
        pangulu_int64_t *block_nnz_pt;

        pangulu_int64_t avg_nnz = PANGULU_ICEIL(nnz, nthreads);
        pangulu_int64_t *block_row_nnz_pt = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (block_length + 1));
        for (int i = 0; i < block_length; i++)
        {
            block_row_nnz_pt[i] = csc_pointer[PANGULU_MIN(i * nb, n)];
        }
        block_row_nnz_pt[block_length] = csc_pointer[n];

        int *thread_pt = (int *)pangulu_malloc(__FILE__, __LINE__, sizeof(int) * (nthreads + 1));
        thread_pt[0] = 0;
        for (int i = 1; i < nthreads + 1; i++)
        {
            thread_pt[i] = pangulu_binarylowerbound(block_row_nnz_pt, block_length, avg_nnz * i);
        }
        pangulu_free(__FILE__, __LINE__, block_row_nnz_pt);
        block_row_nnz_pt = NULL;

        pangulu_int64_t *block_col_pt = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (block_length + 1));
        memset(block_col_pt, 0, sizeof(pangulu_int64_t) * (block_length + 1));

        unsigned int *bit_array = (unsigned int *)pangulu_malloc(__FILE__, __LINE__, sizeof(unsigned int) * bit_length * nthreads);

#pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            unsigned int *tmp_bit = bit_array + bit_length * tid;

            for (int level = thread_pt[tid]; level < thread_pt[tid + 1]; level++)
            {
                memset(tmp_bit, 0, sizeof(unsigned int) * bit_length);

                int start_row = level * nb;
                int end_row = ((level + 1) * nb) < n ? ((level + 1) * nb) : n;

                for (int rid = start_row; rid < end_row; rid++)
                {
                    for (pangulu_int64_t idx = csc_pointer[rid]; idx < csc_pointer[rid + 1]; idx++)
                    {
                        pangulu_int32_t colidx = csc_index[idx];
                        pangulu_int32_t block_cid = colidx / nb;
                        PANGULU_SETBIT(tmp_bit[block_cid / 32], block_cid % 32);
                    }
                }

                pangulu_int64_t tmp_blocknum = 0;
                for (int i = 0; i < bit_length; i++)
                {
                    tmp_blocknum += __builtin_popcount(tmp_bit[i]);
                }

                block_col_pt[level] = tmp_blocknum;
            }
        }
        pangulu_exclusive_scan_1(block_col_pt, block_length + 1);
        block_num = block_col_pt[block_length];

        block_nnz_pt = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (block_num + 1));
        memset(block_nnz_pt, 0, sizeof(pangulu_int64_t) * (block_num + 1));
        pangulu_int32_t *block_row_idx = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * block_num);

        int *count_array = (int *)pangulu_malloc(__FILE__, __LINE__, sizeof(int) * block_length * nthreads);
#pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            unsigned int *tmp_bit = bit_array + bit_length * tid;
            int *tmp_count = count_array + block_length * tid;

            for (int level = thread_pt[tid]; level < thread_pt[tid + 1]; level++)
            {
                memset(tmp_bit, 0, sizeof(unsigned int) * bit_length);
                memset(tmp_count, 0, sizeof(int) * block_length);

                pangulu_int64_t *cur_block_nnz_pt = block_nnz_pt + block_col_pt[level];
                pangulu_int32_t *cur_block_row_idx = block_row_idx + block_col_pt[level];

                int start_row = level * nb;
                int end_row = ((level + 1) * nb) < n ? ((level + 1) * nb) : n;

                for (int rid = start_row; rid < end_row; rid++)
                {
                    for (pangulu_int64_t idx = csc_pointer[rid]; idx < csc_pointer[rid + 1]; idx++)
                    {
                        pangulu_int32_t colidx = csc_index[idx];
                        pangulu_int32_t block_cid = colidx / nb;
                        PANGULU_SETBIT(tmp_bit[block_cid / 32], block_cid % 32);
                        tmp_count[block_cid]++;
                    }
                }

                pangulu_int64_t cnt = 0;
                for (int i = 0; i < block_length; i++)
                {
                    if (PANGULU_GETBIT(tmp_bit[i / 32], i % 32))
                    {
                        cur_block_nnz_pt[cnt] = tmp_count[i];
                        cur_block_row_idx[cnt] = i;
                        cnt++;
                    }
                }
            }
        }
        pangulu_free(__FILE__, __LINE__, bit_array);
        bit_array = NULL;
        pangulu_free(__FILE__, __LINE__, count_array);
        count_array = NULL;
        pangulu_int64_t block_num_half = block_num;
        block_num = block_num * 2 - block_length;

        pangulu_int64_t *block_col_pt_full = NULL;
        pangulu_int32_t *block_row_idx_full = NULL;
        pangulu_convert_halfsymcsc_to_csc_struct(1, 0, block_length, &block_col_pt, &block_row_idx, &block_col_pt_full, &block_row_idx_full);

        pangulu_int64_t *block_nnz_pt_full = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, (block_num + 1) * sizeof(pangulu_int64_t));
        pangulu_int64_t *block_col_pt_tmp = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, (block_length + 1) * sizeof(pangulu_int64_t));
        memcpy(block_col_pt_tmp, block_col_pt_full, (block_length + 1) * sizeof(pangulu_int64_t));
        memset(block_nnz_pt_full, 0, block_num * sizeof(pangulu_int64_t));

        for (pangulu_int32_t bcolid = 0; bcolid < block_length; bcolid++)
        {
            for (pangulu_int64_t ptr = block_col_pt[bcolid]; ptr < block_col_pt[bcolid + 1]; ptr++)
            {
                pangulu_int32_t index = block_row_idx[ptr];
                if (index > bcolid)
                {
                    pangulu_int64_t nnz_num = block_nnz_pt[ptr];

                    pangulu_int64_t cvt_ptr = block_col_pt_tmp[index];
                    pangulu_int64_t org_ptr = block_col_pt_tmp[bcolid];

                    block_nnz_pt_full[org_ptr] = nnz_num;
                    block_nnz_pt_full[cvt_ptr] = nnz_num;

                    block_col_pt_tmp[bcolid]++;
                    block_col_pt_tmp[index]++;
                }
                else if (index == bcolid)
                {
                    pangulu_int64_t nnz_num = block_nnz_pt[ptr] * 2 - nb;
                    pangulu_int64_t org_ptr = block_col_pt_tmp[bcolid];
                    block_nnz_pt_full[org_ptr] = nnz_num;
                    block_col_pt_tmp[bcolid]++;
                }
            }
        }

        pangulu_free(__FILE__, __LINE__, block_col_pt_tmp);
        pangulu_exclusive_scan_1(block_nnz_pt_full, block_num + 1);
        block_nnz_pt_full[block_num] += nb - n % nb;
        pangulu_exclusive_scan_1(block_nnz_pt, block_num_half + 1);

        pangulu_exblock_ptr bcsc_colptr_length = block_num_half * (nb + 1);
        pangulu_inblock_ptr *block_bcsc_colptr = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * bcsc_colptr_length);
        pangulu_inblock_idx *block_bcsc_rowidx = (pangulu_inblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx) * nnz);
        memset(block_bcsc_colptr, 0, sizeof(pangulu_inblock_ptr) * bcsc_colptr_length);
        pangulu_int64_t *block_nnz_pt_tmp = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (block_num_half + 1));
        memcpy(block_nnz_pt_tmp, block_nnz_pt, sizeof(pangulu_int64_t) * (block_num_half + 1));

#pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();

            for (int level = thread_pt[tid]; level < thread_pt[tid + 1]; level++)
            {
                pangulu_int32_t start_col = level * nb;
                pangulu_int32_t end_col = ((level + 1) * nb) < n ? ((level + 1) * nb) : n;
                pangulu_int32_t off = 0;
                for (pangulu_int32_t cid = start_col; cid < end_col; cid++)
                {
                    off++;
                    for (pangulu_int64_t idx = csc_pointer[cid]; idx < csc_pointer[cid + 1]; idx++)
                    {
                        pangulu_exblock_idx rowidx = csc_index[idx];
                        pangulu_int32_t block_num_ = pangulu_binarysearch_in_column(block_col_pt, block_row_idx, block_length, level, rowidx / nb);

                        block_bcsc_colptr[(block_num_) * (nb + 1) + off]++;
                        block_bcsc_rowidx[block_nnz_pt_tmp[block_num_]] = rowidx % nb;
                        block_nnz_pt_tmp[block_num_]++;
                    }
                }
                for (pangulu_int64_t bcolid = block_col_pt[level]; bcolid < block_col_pt[level + 1]; bcolid++)
                {
                    pangulu_int32_t browid = block_row_idx[bcolid];
                    pangulu_int32_t remote_rankid0 = browid / p * q + bcolid / q;
                }
                for (pangulu_int32_t blockid = block_col_pt[level]; blockid < block_col_pt[level + 1]; blockid++)
                {
                    for (pangulu_int32_t j = 0; j < nb; j++)
                    {
                        block_bcsc_colptr[blockid * (nb + 1) + j + 1] += block_bcsc_colptr[blockid * (nb + 1) + j];
                    }
                }
            }
        }
        pangulu_free(__FILE__, __LINE__, block_nnz_pt_tmp);
        pangulu_free(__FILE__, __LINE__, csc_pointer);
        pangulu_free(__FILE__, __LINE__, csc_index);
        pangulu_exblock_ptr *nondiag_block_colptr = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (block_length + 1) * nproc);
        pangulu_exblock_ptr *nondiag_block_colptr_tmp = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (block_length + 1) * nproc);
        pangulu_exblock_ptr *nondiag_block_rowptr = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (block_length + 1));
        pangulu_exblock_idx *nondiag_block_rowidx = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * (block_num - block_length));
        memset(nondiag_block_colptr, 0, sizeof(pangulu_exblock_ptr) * (block_length + 1) * nproc);
        memset(nondiag_block_rowptr, 0, sizeof(pangulu_exblock_ptr) * (block_length + 1));
        pangulu_exblock_ptr *related_nondiag_block_colptr = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * ((block_length + 1) * nproc));
        pangulu_exblock_ptr *related_nondiag_block_rowptr = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * ((block_length + 1)));
        memset(related_nondiag_block_colptr, 0, sizeof(pangulu_exblock_ptr) * (block_length + 1) * nproc);
        memset(related_nondiag_block_rowptr, 0, sizeof(pangulu_exblock_ptr) * (block_length + 1));
        pangulu_uint64_t *diag_uniaddr = (pangulu_uint64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_uint64_t) * block_length);

        pangulu_int32_t *procgrid = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * block_num);
        pangulu_int64_t *procgrid_blockpos = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * block_num);
        pangulu_exblock_ptr block_pointer_length = (block_length + 1) * nproc;
        pangulu_exblock_ptr *malloc_size_each_rank = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (7 * nproc));
        memset(malloc_size_each_rank, 0, sizeof(pangulu_exblock_ptr) * (7 * nproc));

        for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
        {
            for (pangulu_exblock_ptr bidx = block_col_pt_full[bcol]; bidx < block_col_pt_full[bcol + 1]; bidx++)
            {
                pangulu_exblock_idx brow = block_row_idx_full[bidx];
                pangulu_int32_t remote_rankid = (brow % p) * q + (bcol % q);
                procgrid[bidx] = remote_rankid;
                pangulu_exblock_ptr *ptr = &malloc_size_each_rank[7 * remote_rankid];

                if (bcol == brow)
                {
                    diag_uniaddr[bcol] = remote_rankid + 1;
                    pangulu_inblock_idx nb_ = bcol == block_length - 1 ? n % nb : nb;
                    ptr[0]++;
                    ptr[2] += 2;
                    ptr[3] += nb_;
                    ptr[4] += ((block_nnz_pt_full[bidx + 1] - block_nnz_pt_full[bidx]) - nb_) / 2;
                }
                else
                {
                    nondiag_block_colptr[remote_rankid * (block_length + 1) + bcol + 1]++;
                    pangulu_int32_t pos_remote_rank_x = remote_rankid / q;
                    pangulu_int32_t pos_remote_rank_y = remote_rankid % p;
                    for (pangulu_int32_t rankid = 0; rankid < nproc; rankid++)
                    {
                        pangulu_int32_t pos_x = rankid / q;
                        pangulu_int32_t pos_y = rankid % p;
                        if ((pos_remote_rank_x == pos_x) || (pos_remote_rank_y == pos_y))
                        {
                            related_nondiag_block_colptr[rankid * (block_length + 1) + bcol + 1]++;
                        }
                    }
                    ptr[1]++;
                    ptr[2]++;
                    ptr[5] += (block_nnz_pt_full[bidx + 1] - block_nnz_pt_full[bidx]);
                }
            }
        }
        for (pangulu_exblock_ptr i = 0; i < nproc; i++)
        {
            pangulu_exblock_ptr *ptr = &nondiag_block_colptr[i * (block_length + 1)];
            for (pangulu_exblock_ptr j = 0; j < block_length; j++)
            {
                ptr[j + 1] += ptr[j];
            }
        }
        memcpy(nondiag_block_colptr_tmp, nondiag_block_colptr, sizeof(pangulu_exblock_ptr) * (block_length + 1) * nproc);

        MPI_Request *req_tag4 = (MPI_Request *)malloc(sizeof(MPI_Request) * (nproc - 1));
        for (pangulu_int32_t remote_rank = 1; remote_rank < nproc; remote_rank++)
        {
            MPI_Isend(diag_uniaddr, block_length, MPI_PANGULU_UINT64_T, remote_rank, 4, MPI_COMM_WORLD, &req_tag4[remote_rank - 1]);
        }
        MPI_Waitall(nproc - 1, req_tag4, MPI_STATUS_IGNORE);
        pangulu_exblock_ptr *related_nondiag_block_length = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (nproc + 1));
        memset(related_nondiag_block_length, 0, sizeof(pangulu_exblock_ptr) * (nproc + 1));
        for (int i = 0; i < nproc; i++)
        {
            pangulu_exblock_ptr *ptr1 = &related_nondiag_block_colptr[i * (block_length + 1)];
            for (int j = 0; j < block_length; j++)
            {
                ptr1[j + 1] += ptr1[j];
            }
            malloc_size_each_rank[i * 7 + 6] = ptr1[block_length];
            related_nondiag_block_length[i + 1] += ptr1[block_length] + related_nondiag_block_length[i];
        }
        pangulu_exblock_idx *related_nondiag_block_rowidx = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * related_nondiag_block_length[nproc]);
        pangulu_exblock_idx *related_nondiag_block_colidx = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * related_nondiag_block_length[nproc]);
        pangulu_exblock_ptr *related_nondiag_block_colptr_tmp = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * ((block_length + 1) * nproc));
        pangulu_exblock_ptr *related_nondiag_block_csr_to_csc = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * related_nondiag_block_length[1]);
        pangulu_exblock_ptr *op_nondiag_loc = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * malloc_size_each_rank[1]);

        memcpy(related_nondiag_block_colptr_tmp, related_nondiag_block_colptr, sizeof(pangulu_exblock_ptr) * ((block_length + 1) * nproc));

        pangulu_exblock_ptr *blocknum_each_rank_ptr = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (nproc + 1));
        pangulu_exblock_ptr *nondiag_blocknum_each_rank_ptr = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (nproc + 1));
        pangulu_exblock_ptr *nondiag_blocknum_each_rank_ptr_tmp = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (nproc + 1));
        blocknum_each_rank_ptr[0] = 0;
        nondiag_blocknum_each_rank_ptr[0] = 0;

        for (pangulu_int32_t i = 0; i < nproc; i++)
        {
            blocknum_each_rank_ptr[i + 1] = blocknum_each_rank_ptr[i] + malloc_size_each_rank[i * 7] + malloc_size_each_rank[i * 7 + 1];
            nondiag_blocknum_each_rank_ptr[i + 1] = nondiag_blocknum_each_rank_ptr[i] + malloc_size_each_rank[i * 7 + 1];
        }
        memcpy(nondiag_blocknum_each_rank_ptr_tmp, nondiag_blocknum_each_rank_ptr, sizeof(pangulu_exblock_ptr) * (nproc + 1));

        MPI_Request *req_tag0 = (MPI_Request *)malloc((nproc - 1) * sizeof(MPI_Request));
        for (pangulu_int32_t remote_rank = 1; remote_rank < nproc; remote_rank++)
        {
            pangulu_exblock_ptr *ptr = &malloc_size_each_rank[remote_rank * 7];
            MPI_Isend(ptr, 7, MPI_UINT64_T, remote_rank, 0, MPI_COMM_WORLD, &req_tag0[remote_rank - 1]);
        }
        MPI_Waitall(nproc - 1, req_tag0, MPI_STATUS_IGNORE);

        pangulu_exblock_ptr *blknz_ptr_pre = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (nproc * 3 + 1));
        pangulu_exblock_ptr *blknz_ptr_pre_tmp = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (nproc * 3 + 1));
        pangulu_exblock_ptr *nzblk_each_rank_ptr = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (nproc + 1));
        pangulu_exblock_ptr *nzblk_each_rank_ptr_tmp = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (nproc + 1));
        nzblk_each_rank_ptr[0] = 0;
        nzblk_each_rank_ptr_tmp[0] = 0;
        blknz_ptr_pre[0] = 0;
        for (pangulu_int32_t remote_rank = 0; remote_rank < nproc; remote_rank++)
        {
            pangulu_exblock_ptr *ptr = &malloc_size_each_rank[(remote_rank) * 7];
            blknz_ptr_pre[remote_rank * 3 + 1] = blknz_ptr_pre[remote_rank * 3] + ptr[0];
            blknz_ptr_pre[remote_rank * 3 + 2] = blknz_ptr_pre[remote_rank * 3 + 1] + ptr[0];
            blknz_ptr_pre[remote_rank * 3 + 3] = blknz_ptr_pre[remote_rank * 3 + 2] + ptr[1];
            nzblk_each_rank_ptr[remote_rank + 1] = nzblk_each_rank_ptr[remote_rank] + ptr[0] + ptr[1];
        }
        memcpy(blknz_ptr_pre_tmp, blknz_ptr_pre, sizeof(pangulu_exblock_ptr) * (nproc * 3 + 1));
        memcpy(nzblk_each_rank_ptr_tmp, nzblk_each_rank_ptr, sizeof(pangulu_exblock_ptr) * (nproc + 1));

        pangulu_int32_t *block_op_rank = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * block_num);
        pangulu_exblock_ptr *blknz_ptr_rank = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (block_num + block_length + 1));
        memset(blknz_ptr_rank, 0, sizeof(pangulu_exblock_ptr) * (block_num + block_length + 1));
        memset(block_op_rank, -1, sizeof(pangulu_int32_t) * block_num);
        pangulu_int32_t *tag_nproc = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, nproc * sizeof(pangulu_int32_t));
        pangulu_int32_t *tag_par = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, block_num * sizeof(pangulu_int32_t));
        memset(tag_nproc, 0, nproc * sizeof(pangulu_int32_t));

        for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
        {
            for (pangulu_exblock_ptr bidx = block_col_pt_full[bcol]; bidx < block_col_pt_full[bcol + 1]; bidx++)
            {
                pangulu_exblock_idx brow = block_row_idx_full[bidx];
                pangulu_int32_t remote_rankid = procgrid[bidx];
                pangulu_exblock_ptr block_id = nzblk_each_rank_ptr_tmp[remote_rankid];
                pangulu_exblock_ptr *ptr_pre = &blknz_ptr_pre_tmp[remote_rankid * 3];
                pangulu_exblock_idx pos = 0;
                tag_par[bidx] = tag_nproc[remote_rankid];
                tag_nproc[remote_rankid]++;
                if (brow >= bcol)
                {
                    pos = pangulu_binarysearch_in_column(block_col_pt, block_row_idx, block_length, bcol, brow);
                    if (brow == bcol)
                    {
                        pangulu_inblock_idx nb_ = bcol == block_length - 1 ? n % nb : nb;
                        block_op_rank[block_id] = 0;
                        blknz_ptr_rank[ptr_pre[0]] = block_nnz_pt[pos + 1] - block_nnz_pt[pos];
                        blknz_ptr_rank[ptr_pre[1]] = block_nnz_pt[pos + 1] - block_nnz_pt[pos] - nb_;
                        blknz_ptr_pre_tmp[remote_rankid * 3 + 0]++;
                        blknz_ptr_pre_tmp[remote_rankid * 3 + 1]++;
                    }
                    else
                    {
                        block_op_rank[block_id] = 1;
                        blknz_ptr_rank[ptr_pre[2]] = block_nnz_pt[pos + 1] - block_nnz_pt[pos];
                        blknz_ptr_pre_tmp[remote_rankid * 3 + 2]++;
                    }
                }
                else
                {
                    pos = pangulu_binarysearch_in_column(block_col_pt, block_row_idx, block_length, brow, bcol);
                    block_op_rank[block_id] = 2;
                    blknz_ptr_rank[ptr_pre[2]] = block_nnz_pt[pos + 1] - block_nnz_pt[pos];
                    blknz_ptr_pre_tmp[remote_rankid * 3 + 2]++;
                }
                procgrid_blockpos[bidx] = pos;
                nzblk_each_rank_ptr_tmp[remote_rankid]++;
                if (brow != bcol)
                {
                    nondiag_block_rowidx[nondiag_blocknum_each_rank_ptr[remote_rankid] + nondiag_block_colptr_tmp[remote_rankid * (block_length + 1) + bcol]] = brow;
                    nondiag_blocknum_each_rank_ptr_tmp[remote_rankid]++;
                    nondiag_block_colptr_tmp[remote_rankid * (block_length + 1) + bcol]++;
                    pangulu_int32_t pos_remote_rank_x = remote_rankid / q;
                    pangulu_int32_t pos_remote_rank_y = remote_rankid % p;
                    for (pangulu_int32_t rankid = 0; rankid < nproc; rankid++)
                    {
                        pangulu_int32_t pos_x = rankid / q;
                        pangulu_int32_t pos_y = rankid % p;
                        if ((pos_remote_rank_x == pos_x) || (pos_remote_rank_y == pos_y))
                        {
                            related_nondiag_block_rowidx[related_nondiag_block_length[rankid] + related_nondiag_block_colptr_tmp[rankid * (block_length + 1) + bcol]] = brow;
                            related_nondiag_block_colptr_tmp[rankid * (block_length + 1) + bcol]++;
                        }
                    }
                }
            }
        }
        pangulu_free(__FILE__, __LINE__, related_nondiag_block_colptr_tmp);
        pangulu_free(__FILE__, __LINE__, nondiag_block_colptr_tmp);
        pangulu_free(__FILE__, __LINE__, nondiag_blocknum_each_rank_ptr_tmp);

        MPI_Request *req_tag5 = (MPI_Request *)malloc(sizeof(MPI_Request) * ((nproc - 1) * 4));

        for (int i = 1; i < nproc; i++)
        {
            MPI_Isend(&related_nondiag_block_colptr[i * (block_length + 1)], block_length + 1, MPI_UINT64_T, i, 5, MPI_COMM_WORLD, &req_tag5[(i - 1) * 4]);
            MPI_Isend(&related_nondiag_block_rowidx[related_nondiag_block_length[i]], related_nondiag_block_length[i + 1] - related_nondiag_block_length[i], MPI_UINT32_T, i, 5, MPI_COMM_WORLD, &req_tag5[(i - 1) * 4 + 1]);
            MPI_Isend(&nondiag_block_rowidx[nondiag_blocknum_each_rank_ptr[i]], nondiag_blocknum_each_rank_ptr[i + 1] - nondiag_blocknum_each_rank_ptr[i], MPI_UINT32_T, i, 5, MPI_COMM_WORLD, &req_tag5[(i - 1) * 4 + 2]);
            MPI_Isend(&nondiag_block_colptr[i * (block_length + 1)], block_length + 1, MPI_UINT64_T, i, 5, MPI_COMM_WORLD, &req_tag5[(i - 1) * 4 + 3]);
        }

        pangulu_exblock_ptr *aid_inptr = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * block_length);
        pangulu_transpose_struct_with_valueidx_exblock(block_length, related_nondiag_block_colptr, related_nondiag_block_rowidx, related_nondiag_block_rowptr, related_nondiag_block_colidx, related_nondiag_block_csr_to_csc, aid_inptr);
        MPI_Waitall((nproc - 1) * 4, req_tag5, MPI_STATUS_IGNORE);
        pangulu_exblock_idx *nondiag_block_colidx = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * malloc_size_each_rank[1]);
        pangulu_exblock_ptr *nondiag_block_csr_to_csc = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * nondiag_block_colptr[block_length]);
        pangulu_convert_csr_to_csc_block_with_index(
            block_length, nondiag_block_colptr, nondiag_block_rowidx, nondiag_block_rowptr, nondiag_block_colidx, nondiag_block_csr_to_csc);

        pangulu_free(__FILE__, __LINE__, nzblk_each_rank_ptr_tmp);
        pangulu_free(__FILE__, __LINE__, blknz_ptr_pre_tmp);
        pangulu_free(__FILE__, __LINE__, tag_nproc);

        MPI_Request *req_tag1 = (MPI_Request *)malloc(sizeof(MPI_Request) * (nproc - 1));
        MPI_Request *req_tag2 = (MPI_Request *)malloc(sizeof(MPI_Request) * (nproc - 1));

        pangulu_exblock_ptr off1 = malloc_size_each_rank[2];
        for (pangulu_int32_t remote_rank = 1; remote_rank < nproc; remote_rank++)
        {
            pangulu_exblock_ptr off2 = blocknum_each_rank_ptr[remote_rank];
            pangulu_exblock_ptr length1 = malloc_size_each_rank[remote_rank * 7 + 2];
            pangulu_exblock_ptr length2 = blocknum_each_rank_ptr[remote_rank + 1] - off2;
            MPI_Isend(&blknz_ptr_rank[off1], length1, MPI_UINT64_T, remote_rank, 1, MPI_COMM_WORLD, &req_tag1[remote_rank - 1]);
            MPI_Isend(&block_op_rank[off2], length2, MPI_INT, remote_rank, 1, MPI_COMM_WORLD, &req_tag2[remote_rank - 1]);
            off1 += length1;
        }
        MPI_Waitall(nproc - 1, req_tag1, MPI_STATUS_IGNORE);
        MPI_Waitall(nproc - 1, req_tag2, MPI_STATUS_IGNORE);

        char *nondiag_csc = NULL;
        pangulu_exblock_ptr nondiag_block_num = malloc_size_each_rank[1];
        calculate_type **nondiag_cscvalue = (calculate_type **)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type *) * nondiag_block_num);
        pangulu_inblock_ptr **nondiag_colptr = (pangulu_inblock_ptr **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr *) * nondiag_block_num);
        pangulu_inblock_idx **nondiag_rowidx = (pangulu_inblock_idx **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx *) * nondiag_block_num);
        pangulu_inblock_ptr **nondiag_csr_to_csc = (pangulu_inblock_ptr **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr *) * nondiag_block_num);
        pangulu_inblock_ptr **nondiag_rowptr = (pangulu_inblock_ptr **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr *) * nondiag_block_num);
        pangulu_inblock_idx **nondiag_colidx = (pangulu_inblock_idx **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx *) * nondiag_block_num);

        char *memptr = nondiag_csc;
        pangulu_exblock_ptr *nzblk_size_ptr = &blknz_ptr_rank[2 * malloc_size_each_rank[0]];

        pangulu_exblock_ptr nondiag_cnt = 0;
        for (pangulu_int32_t i = 0; i < blocknum_each_rank_ptr[1]; i++)
        {
            if (block_op_rank[i] == 0)
                continue;
            pangulu_exblock_ptr length = nzblk_size_ptr[nondiag_cnt];
            memptr += sizeof(pangulu_int64_t) * 4;
            nondiag_cscvalue[nondiag_cnt] = memptr;
            memptr += sizeof(calculate_type) * length;
            nondiag_colptr[nondiag_cnt] = memptr;
            memptr += sizeof(pangulu_inblock_ptr) * (nb + 1);
            nondiag_rowidx[nondiag_cnt] = memptr;
            memptr += sizeof(pangulu_inblock_idx) * length;
            if (block_op_rank[i] == 1)
            {
                size_t align = 8 - ((uintptr_t)memptr % 8);
                if (align != 8)
                    memptr += align;
                nondiag_csr_to_csc[nondiag_cnt] = memptr;
                memptr += sizeof(pangulu_inblock_ptr) * length;
                nondiag_rowptr[nondiag_cnt] = memptr;
                memptr += sizeof(pangulu_inblock_ptr) * (nb + 1);
                nondiag_colidx[nondiag_cnt] = memptr;
                memptr += sizeof(pangulu_inblock_idx) * length;
            }
            size_t align = 8 - ((uintptr_t)memptr % 8);
            if (align != 8)
                memptr += align;
            nondiag_cnt++;
        }

        nondiag_csc = (char *)pangulu_malloc(__FILE__, __LINE__, (size_t)memptr);
        for (pangulu_int32_t i = 0; i < nondiag_block_num; i++)
        {
            nondiag_cscvalue[i] = nondiag_csc + (size_t)(nondiag_cscvalue[i]);
            nondiag_colptr[i] = nondiag_csc + (size_t)(nondiag_colptr[i]);
            nondiag_rowidx[i] = nondiag_csc + (size_t)(nondiag_rowidx[i]);
            nondiag_csr_to_csc[i] = nondiag_csc + (size_t)(nondiag_csr_to_csc[i]);
            nondiag_rowptr[i] = nondiag_csc + (size_t)(nondiag_rowptr[i]);
            nondiag_colidx[i] = nondiag_csc + (size_t)(nondiag_colidx[i]);
        }
        char *diag_upper_csr = NULL;
        pangulu_exblock_ptr diag_block_num = malloc_size_each_rank[0];
        calculate_type **diag_upper_csrvalue = (calculate_type **)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type *) * diag_block_num);
        pangulu_inblock_ptr **diag_upper_rowptr = (pangulu_inblock_ptr **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr *) * diag_block_num);
        pangulu_inblock_idx **diag_upper_colidx = (pangulu_inblock_idx **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx *) * diag_block_num);

        char *umemptr = diag_upper_csr;
        nzblk_size_ptr = blknz_ptr_rank;
        for (pangulu_int32_t i = 0; i < diag_block_num; i++)
        {
            pangulu_exblock_ptr length = nzblk_size_ptr[i];
            umemptr += sizeof(pangulu_int64_t) * 4;
            diag_upper_csrvalue[i] = umemptr;
            umemptr += sizeof(calculate_type) * length;
            diag_upper_rowptr[i] = umemptr;
            umemptr += sizeof(pangulu_inblock_ptr) * (nb + 1);
            diag_upper_colidx[i] = umemptr;
            umemptr += sizeof(pangulu_inblock_idx) * length;
            size_t align = 8 - ((uintptr_t)umemptr % 8);
            if (align != 8)
                umemptr += align;
        }

        diag_upper_csr = (char *)pangulu_malloc(__FILE__, __LINE__, (size_t)umemptr);
        for (pangulu_int32_t i = 0; i < diag_block_num; i++)
        {
            diag_upper_csrvalue[i] = diag_upper_csr + (size_t)(diag_upper_csrvalue[i]);
            diag_upper_rowptr[i] = diag_upper_csr + (size_t)(diag_upper_rowptr[i]);
            diag_upper_colidx[i] = diag_upper_csr + (size_t)(diag_upper_colidx[i]);
        }

        char *diag_lower_csc = NULL;
        calculate_type **diag_lower_cscvalue = (calculate_type **)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type *) * diag_block_num);
        pangulu_inblock_ptr **diag_lower_colptr = (pangulu_inblock_ptr **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr *) * diag_block_num);
        pangulu_inblock_idx **diag_lower_rowidx = (pangulu_inblock_idx **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx *) * diag_block_num);

        char *lmemptr = diag_lower_csc;
        nzblk_size_ptr = &blknz_ptr_rank[malloc_size_each_rank[0]];
        for (pangulu_int32_t i = 0; i < diag_block_num; i++)
        {
            pangulu_exblock_ptr length = nzblk_size_ptr[i];
            lmemptr += sizeof(pangulu_int64_t) * 4;
            diag_lower_cscvalue[i] = lmemptr;
            lmemptr += sizeof(calculate_type) * length;
            diag_lower_colptr[i] = lmemptr;
            lmemptr += sizeof(pangulu_inblock_ptr) * (nb + 1);
            diag_lower_rowidx[i] = lmemptr;
            lmemptr += sizeof(pangulu_inblock_idx) * length;
            size_t align = 8 - ((uintptr_t)lmemptr % 8);
            if (align != 8)
                lmemptr += align;
        }

        diag_lower_csc = (char *)pangulu_malloc(__FILE__, __LINE__, (size_t)lmemptr);
        for (pangulu_int32_t i = 0; i < diag_block_num; i++)
        {
            diag_lower_cscvalue[i] = diag_lower_csc + (size_t)(diag_lower_cscvalue[i]);
            diag_lower_colptr[i] = diag_lower_csc + (size_t)(diag_lower_colptr[i]);
            diag_lower_rowidx[i] = diag_lower_csc + (size_t)(diag_lower_rowidx[i]);
        }

        MPI_Request *send_reqs = (MPI_Request *)malloc(sizeof(MPI_Request) * (block_num * 2));
        pangulu_int32_t reqs = 0;
        pangulu_exblock_ptr diag_cnt = 0;
        nondiag_cnt = 0;

#pragma omp parallel num_threads(nproc)
        {
            for (pangulu_int32_t level = 0; level < block_length; level++)
            {
                for (pangulu_int64_t idx = block_col_pt_full[level]; idx < block_col_pt_full[level + 1]; idx++)
                {
                    pangulu_int32_t remote_rankid = procgrid[idx];
                    pangulu_int32_t block_num_thread = tag_par[idx];

                    pangulu_int64_t cor_rankid = procgrid_blockpos[idx];
                    pangulu_inblock_idx *send_buffer_idx = &block_bcsc_rowidx[block_nnz_pt[cor_rankid]];
                    pangulu_inblock_ptr *send_buffer_ptr = &block_bcsc_colptr[cor_rankid * (nb + 1)];
                    pangulu_exblock_ptr length = 0;

                    if (remote_rankid == 0 && omp_get_thread_num() == 0)
                    {
                        pangulu_exblock_ptr *ptr = blknz_ptr_rank;
                        if (block_op_rank[block_num_thread] == 0)
                        {
                            length = ptr[diag_cnt];
                            memcpy(diag_upper_colidx[diag_cnt], send_buffer_idx, sizeof(pangulu_inblock_idx) * length);
                            memcpy(diag_upper_rowptr[diag_cnt], send_buffer_ptr, sizeof(pangulu_inblock_ptr) * (nb + 1));
                            diag_cnt++;
                        }
                        else
                        {
                            length = ptr[malloc_size_each_rank[0] * 2 + nondiag_cnt];
                            memcpy(nondiag_rowidx[nondiag_cnt], send_buffer_idx, sizeof(pangulu_inblock_idx) * length);
                            memcpy(nondiag_colptr[nondiag_cnt], send_buffer_ptr, sizeof(pangulu_inblock_ptr) * (nb + 1));
                            op_nondiag_loc[nondiag_cnt] = block_op_rank[block_num_thread];
                            nondiag_cnt++;
                        }
                    }
                    else if (remote_rankid != 0 && omp_get_thread_num() == remote_rankid)
                    {
                        if (block_op_rank[blocknum_each_rank_ptr[remote_rankid] + block_num_thread] == 0)
                        {
                            length = blknz_ptr_rank[blknz_ptr_pre[remote_rankid * 3]];
                            blknz_ptr_pre[remote_rankid * 3]++;
                        }
                        else
                        {
                            length = blknz_ptr_rank[blknz_ptr_pre[remote_rankid * 3 + 2]];
                            blknz_ptr_pre[remote_rankid * 3 + 2]++;
                        }
                        MPI_Isend(send_buffer_idx, length, MPI_UINT16_T, remote_rankid, 2, MPI_COMM_WORLD, &send_reqs[reqs++]);
                        MPI_Isend(send_buffer_ptr, nb + 1, MPI_UINT32_T, remote_rankid, 3, MPI_COMM_WORLD, &send_reqs[reqs++]);
                    }
                }
            }
        }

        pangulu_time_start(&start_time);
        {
#pragma omp parallel for schedule(guided)
            for (pangulu_exblock_ptr i = 0; i < malloc_size_each_rank[0]; i++)
            {
                pangulu_diag_block_trans(nb, diag_upper_rowptr[i], diag_upper_colidx[i], diag_lower_colptr[i], diag_lower_rowidx[i]);
            }
        }

        {
#pragma omp parallel for schedule(guided)
            for (pangulu_exblock_ptr i = 0; i < malloc_size_each_rank[1]; i++)
            {
                pangulu_int32_t op = op_nondiag_loc[i];
                if (op == 1)
                {
                    pangulu_transpose_struct_with_valueidx_inblock(nb, nondiag_colptr[i], nondiag_rowidx[i], nondiag_rowptr[i], nondiag_colidx[i], nondiag_csr_to_csc[i]);
                }
                else if (op == 2)
                {
                    pangulu_inblock_ptr *ptr = nondiag_colptr[i];
                    pangulu_inblock_ptr **double_ptr = &ptr;
                    pangulu_inblock_idx *ptr1 = nondiag_rowidx[i];
                    pangulu_inblock_idx **double_idx = &ptr1;
                    pangulu_convert_csr_to_csc_block(0, nb, double_ptr, double_idx, NULL, double_ptr, double_idx, NULL);
                    memcpy(nondiag_colptr[i], ptr, (nb + 1) * sizeof(pangulu_inblock_ptr));
                    memcpy(nondiag_rowidx[i], ptr1, ptr[nb] * sizeof(pangulu_inblock_idx));
                }
            }
        }

        diag_cnt = 0;
        for (pangulu_int32_t level = 0; level < block_length; level++)
        {
            if (diag_uniaddr[level] - 1 == rank)
            {
                diag_uniaddr[level] = diag_cnt + 1;
                diag_cnt++;
            }
            else
            {
                diag_uniaddr[level] = 0;
            }
        }

        pangulu_free(__FILE__, __LINE__, block_col_pt_full);
        pangulu_free(__FILE__, __LINE__, block_row_idx_full);

        MPI_Waitall(reqs, send_reqs, MPI_STATUSES_IGNORE);
        *out_diag_upper_rowptr = diag_upper_rowptr;
        *out_diag_upper_colidx = diag_upper_colidx;
        *out_diag_upper_csrvalue = diag_upper_csrvalue;

        *out_diag_lower_colptr = diag_lower_colptr;
        *out_diag_lower_rowidx = diag_lower_rowidx;
        *out_diag_lower_cscvalue = diag_lower_cscvalue;

        *out_nondiag_cscvalue = nondiag_cscvalue;
        *out_nondiag_colptr = nondiag_colptr;
        *out_nondiag_rowidx = nondiag_rowidx;
        *out_nondiag_csr_to_csc = nondiag_csr_to_csc;
        *out_nondiag_rowptr = nondiag_rowptr;
        *out_nondiag_colidx = nondiag_colidx;

        *out_nondiag_block_colptr = nondiag_block_colptr;
        *out_nondiag_block_rowidx = nondiag_block_rowidx;
        *out_nondiag_block_rowptr = nondiag_block_rowptr;
        *out_nondiag_block_colidx = nondiag_block_colidx;
        *out_nondiag_block_csr_to_csc = nondiag_block_csr_to_csc;

        *out_related_nondiag_block_colptr = related_nondiag_block_colptr;
        *out_related_nondiag_block_rowidx = related_nondiag_block_rowidx;
        *out_related_nondiag_block_rowptr = related_nondiag_block_rowptr;
        *out_related_nondiag_block_colidx = related_nondiag_block_colidx;
        *out_related_nondiag_block_csr_to_csc = related_nondiag_block_csr_to_csc;
        *out_diag_uniaddr = diag_uniaddr;
    }
    else
    {
        MPI_Request req_tag0, req_tag1, req_tag4;
        MPI_Request *req_tag5 = (MPI_Request *)malloc(sizeof(MPI_Request) * 5);
        pangulu_uint64_t *diag_uniaddr = (pangulu_uint64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_uint64_t) * block_length);
        pangulu_exblock_ptr *related_nondiag_block_colptr = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (block_length + 1));
        pangulu_exblock_ptr *related_nondiag_block_rowptr = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (block_length + 1));
        pangulu_exblock_ptr *nondiag_block_colptr = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (block_length + 1));
        pangulu_exblock_ptr *nondiag_block_rowptr = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (block_length + 1));
        memset(nondiag_block_rowptr, 0, sizeof(pangulu_exblock_ptr) * (block_length + 1));
        MPI_Irecv(diag_uniaddr, block_length, MPI_PANGULU_UINT64_T, 0, 4, MPI_COMM_WORLD, &req_tag4);
        pangulu_exblock_ptr *malloc_size = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * 6);
        MPI_Recv(malloc_size, 7, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pangulu_exblock_idx *related_nondiag_block_rowidx = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * malloc_size[6]);
        pangulu_exblock_idx *related_nondiag_block_colidx = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * malloc_size[6]);
        pangulu_exblock_idx *nondiag_block_rowidx = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * malloc_size[1]);
        pangulu_exblock_idx *nondiag_block_colidx = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * malloc_size[1]);
        pangulu_exblock_ptr *related_nondiag_block_csr_to_csc = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * malloc_size[6]);
        pangulu_exblock_ptr *op_nondiag_loc = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * malloc_size[1]);

        MPI_Irecv(related_nondiag_block_colptr, block_length + 1, MPI_UINT64_T, 0, 5, MPI_COMM_WORLD, &req_tag5[0]);
        MPI_Irecv(related_nondiag_block_rowidx, malloc_size[6], MPI_UINT32_T, 0, 5, MPI_COMM_WORLD, &req_tag5[1]);
        MPI_Irecv(nondiag_block_rowidx, malloc_size[1], MPI_UINT32_T, 0, 5, MPI_COMM_WORLD, &req_tag5[2]);
        MPI_Irecv(nondiag_block_colptr, block_length + 1, MPI_UINT64_T, 0, 5, MPI_COMM_WORLD, &req_tag5[3]);
        MPI_Waitall(4, req_tag5, MPI_STATUS_IGNORE);

        pangulu_exblock_ptr *aid_inptr = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * block_length);
        pangulu_transpose_struct_with_valueidx_exblock(block_length, related_nondiag_block_colptr, related_nondiag_block_rowidx, related_nondiag_block_rowptr, related_nondiag_block_colidx, related_nondiag_block_csr_to_csc, aid_inptr);

        pangulu_exblock_ptr *nondiag_block_csr_to_csc = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * nondiag_block_colptr[block_length]);
        pangulu_convert_csr_to_csc_block_with_index(block_length, nondiag_block_colptr, nondiag_block_rowidx, nondiag_block_rowptr, nondiag_block_colidx, nondiag_block_csr_to_csc);

        pangulu_exblock_ptr block_num = malloc_size[0] + malloc_size[1];
        pangulu_exblock_ptr *nzblk_size = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * malloc_size[2]);
        pangulu_int32_t *block_op = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * (block_num));
        MPI_Recv(nzblk_size, malloc_size[2], MPI_UINT64_T, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(block_op, block_num, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Wait(&req_tag4, MPI_STATUS_IGNORE);
        pangulu_exblock_ptr diag_cnt = 0;
        pangulu_int64_t cnt_tmp = 0;
        for (pangulu_int32_t level = 0; level < block_length; level++)
        {
            if (diag_uniaddr[level] - 1 == rank)
            {
                diag_uniaddr[level] = diag_cnt + 1;
                diag_cnt++;
            }
            else
            {
                diag_uniaddr[level] = 0;
            }
        }
        char *nondiag_csc = NULL;
        pangulu_exblock_ptr nondiag_block_num = malloc_size[1];
        calculate_type **nondiag_cscvalue = (calculate_type **)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type *) * nondiag_block_num);
        pangulu_inblock_ptr **nondiag_colptr = (pangulu_inblock_ptr **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr *) * nondiag_block_num);
        pangulu_inblock_idx **nondiag_rowidx = (pangulu_inblock_idx **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx *) * nondiag_block_num);
        pangulu_inblock_ptr **nondiag_csr_to_csc = (pangulu_inblock_ptr **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr *) * nondiag_block_num);
        pangulu_inblock_ptr **nondiag_rowptr = (pangulu_inblock_ptr **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr *) * nondiag_block_num);
        pangulu_inblock_idx **nondiag_colidx = (pangulu_inblock_idx **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx *) * nondiag_block_num);

        char *memptr = nondiag_csc;
        pangulu_exblock_ptr *nzblk_size_ptr = &nzblk_size[2 * malloc_size[0]];

        pangulu_exblock_ptr nondiag_cnt = 0;
        for (pangulu_int32_t i = 0; i < block_num; i++)
        {
            if (block_op[i] == 0)
                continue;
            pangulu_exblock_ptr length = nzblk_size_ptr[nondiag_cnt];
            memptr += sizeof(pangulu_int64_t) * 4;
            nondiag_cscvalue[nondiag_cnt] = memptr;
            memptr += sizeof(calculate_type) * length;
            nondiag_colptr[nondiag_cnt] = memptr;
            memptr += sizeof(pangulu_inblock_ptr) * (nb + 1);
            nondiag_rowidx[nondiag_cnt] = memptr;
            memptr += sizeof(pangulu_inblock_idx) * length;
            if (block_op[i] == 1)
            {
                size_t align = 8 - ((uintptr_t)memptr % 8);
                if (align != 8)
                    memptr += align;
                nondiag_csr_to_csc[nondiag_cnt] = memptr;
                memptr += sizeof(pangulu_inblock_ptr) * length;
                nondiag_rowptr[nondiag_cnt] = memptr;
                memptr += sizeof(pangulu_inblock_ptr) * (nb + 1);
                nondiag_colidx[nondiag_cnt] = memptr;
                memptr += sizeof(pangulu_inblock_idx) * length;
            }
            size_t align = 8 - ((uintptr_t)memptr % 8);
            if (align != 8)
                memptr += align;
            nondiag_cnt++;
        }

        nondiag_csc = (char *)pangulu_malloc(__FILE__, __LINE__, (size_t)memptr);
        for (pangulu_int32_t i = 0; i < nondiag_block_num; i++)
        {
            nondiag_cscvalue[i] = nondiag_csc + (size_t)(nondiag_cscvalue[i]);
            nondiag_colptr[i] = nondiag_csc + (size_t)(nondiag_colptr[i]);
            nondiag_rowidx[i] = nondiag_csc + (size_t)(nondiag_rowidx[i]);
            nondiag_csr_to_csc[i] = nondiag_csc + (size_t)(nondiag_csr_to_csc[i]);
            nondiag_rowptr[i] = nondiag_csc + (size_t)(nondiag_rowptr[i]);
            nondiag_colidx[i] = nondiag_csc + (size_t)(nondiag_colidx[i]);
        }
        char *diag_upper_csr = NULL;
        pangulu_exblock_ptr diag_block_num = malloc_size[0];
        calculate_type **diag_upper_csrvalue = (calculate_type **)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type *) * diag_block_num);
        pangulu_inblock_ptr **diag_upper_rowptr = (pangulu_inblock_ptr **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr *) * diag_block_num);
        pangulu_inblock_idx **diag_upper_colidx = (pangulu_inblock_idx **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx *) * diag_block_num);

        char *umemptr = diag_upper_csr;
        nzblk_size_ptr = nzblk_size;
        for (pangulu_int32_t i = 0; i < diag_block_num; i++)
        {
            pangulu_exblock_ptr length = nzblk_size_ptr[i];
            umemptr += sizeof(pangulu_int64_t) * 4;
            diag_upper_csrvalue[i] = umemptr;
            umemptr += sizeof(calculate_type) * length;
            diag_upper_rowptr[i] = umemptr;
            umemptr += sizeof(pangulu_inblock_ptr) * (nb + 1);
            diag_upper_colidx[i] = umemptr;
            umemptr += sizeof(pangulu_inblock_idx) * length;
            size_t align = 8 - ((uintptr_t)umemptr % 8);
            if (align != 8)
                umemptr += align;
        }

        diag_upper_csr = (char *)pangulu_malloc(__FILE__, __LINE__, (size_t)umemptr);
        for (pangulu_int32_t i = 0; i < diag_block_num; i++)
        {
            diag_upper_csrvalue[i] = diag_upper_csr + (size_t)(diag_upper_csrvalue[i]);
            diag_upper_rowptr[i] = diag_upper_csr + (size_t)(diag_upper_rowptr[i]);
            diag_upper_colidx[i] = diag_upper_csr + (size_t)(diag_upper_colidx[i]);
        }

        char *diag_lower_csc = NULL;
        calculate_type **diag_lower_cscvalue = (calculate_type **)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type *) * diag_block_num);
        pangulu_inblock_ptr **diag_lower_colptr = (pangulu_inblock_ptr **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr *) * diag_block_num);
        pangulu_inblock_idx **diag_lower_rowidx = (pangulu_inblock_idx **)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx *) * diag_block_num);

        char *lmemptr = diag_lower_csc;
        nzblk_size_ptr = &nzblk_size[malloc_size[0]];
        for (pangulu_int32_t i = 0; i < diag_block_num; i++)
        {
            pangulu_exblock_ptr length = nzblk_size_ptr[i];
            lmemptr += sizeof(pangulu_int64_t) * 4;
            diag_lower_cscvalue[i] = lmemptr;
            lmemptr += sizeof(calculate_type) * length;
            diag_lower_colptr[i] = lmemptr;
            lmemptr += sizeof(pangulu_inblock_ptr) * (nb + 1);
            diag_lower_rowidx[i] = lmemptr;
            lmemptr += sizeof(pangulu_inblock_idx) * length;
            size_t align = 8 - ((uintptr_t)lmemptr % 8);
            if (align != 8)
                lmemptr += align;
        }

        diag_lower_csc = (char *)pangulu_malloc(__FILE__, __LINE__, (size_t)lmemptr);

        for (pangulu_int32_t i = 0; i < diag_block_num; i++)
        {
            diag_lower_cscvalue[i] = (diag_lower_csc + (size_t)(diag_lower_cscvalue[i]));
            diag_lower_colptr[i] = (diag_lower_csc + (size_t)(diag_lower_colptr[i]));
            diag_lower_rowidx[i] = (diag_lower_csc + (size_t)(diag_lower_rowidx[i]));
        }

        MPI_Request *recv_requests = (MPI_Request *)malloc(sizeof(MPI_Request) * (2 * block_num));
        for (int i = 0; i < 2 * block_num; i++)
        {
            recv_requests[i] = MPI_REQUEST_NULL;
        }

        diag_cnt = 0;
        nondiag_cnt = 0;
        for (pangulu_int32_t idx = 0; idx < block_num; idx++)
        {
            if (block_op[idx] == 0)
            {
                pangulu_exblock_ptr length = nzblk_size[diag_cnt];
                MPI_Irecv(diag_upper_colidx[diag_cnt], length, MPI_UINT16_T, 0, 2, MPI_COMM_WORLD, &recv_requests[idx * 2]);
                MPI_Irecv(diag_upper_rowptr[diag_cnt], nb + 1, MPI_UINT32_T, 0, 3, MPI_COMM_WORLD, &recv_requests[idx * 2 + 1]);
                diag_cnt++;
            }
            else
            {
                pangulu_exblock_ptr length = nzblk_size[malloc_size[0] * 2 + nondiag_cnt];
                MPI_Irecv(nondiag_rowidx[nondiag_cnt], length, MPI_UINT16_T, 0, 2, MPI_COMM_WORLD, &recv_requests[idx * 2]);
                MPI_Irecv(nondiag_colptr[nondiag_cnt], nb + 1, MPI_UINT32_T, 0, 3, MPI_COMM_WORLD, &recv_requests[idx * 2 + 1]);
                op_nondiag_loc[nondiag_cnt] = block_op[idx];
                nondiag_cnt++;
            }
        }
        MPI_Waitall(block_num * 2, recv_requests, MPI_STATUSES_IGNORE);

        {
#pragma omp parallel for schedule(guided)
            for (pangulu_exblock_ptr i = 0; i < diag_block_num; i++)
            {
                pangulu_diag_block_trans(nb, diag_upper_rowptr[i], diag_upper_colidx[i], diag_lower_colptr[i], diag_lower_rowidx[i]);
            }
        }

        {
#pragma omp parallel for schedule(guided)
            for (pangulu_exblock_ptr i = 0; i < nondiag_block_num; i++)
            {
                pangulu_int32_t op = op_nondiag_loc[i];
                if (op == 1)
                {
                    pangulu_transpose_struct_with_valueidx_inblock(nb, nondiag_colptr[i], nondiag_rowidx[i], nondiag_rowptr[i], nondiag_colidx[i], nondiag_csr_to_csc[i]);
                }
                else if (op == 2)
                {
                    pangulu_inblock_ptr *ptr = nondiag_colptr[i];
                    pangulu_inblock_ptr **double_ptr = &ptr;
                    pangulu_inblock_idx *ptr1 = nondiag_rowidx[i];
                    pangulu_inblock_idx **double_idx = &ptr1;
                    pangulu_convert_csr_to_csc_block(0, nb, double_ptr, double_idx, NULL, double_ptr, double_idx, NULL);
                    memcpy(nondiag_colptr[i], ptr, (nb + 1) * sizeof(pangulu_inblock_ptr));
                    memcpy(nondiag_rowidx[i], ptr1, ptr[nb] * sizeof(pangulu_inblock_idx));
                }
            }
        }

        *out_diag_upper_rowptr = diag_upper_rowptr;
        *out_diag_upper_colidx = diag_upper_colidx;
        *out_diag_upper_csrvalue = diag_upper_csrvalue;

        *out_diag_lower_colptr = diag_lower_colptr;
        *out_diag_lower_rowidx = diag_lower_rowidx;
        *out_diag_lower_cscvalue = diag_lower_cscvalue;

        *out_nondiag_cscvalue = nondiag_cscvalue;
        *out_nondiag_colptr = nondiag_colptr;
        *out_nondiag_rowidx = nondiag_rowidx;
        *out_nondiag_csr_to_csc = nondiag_csr_to_csc;
        *out_nondiag_rowptr = nondiag_rowptr;
        *out_nondiag_colidx = nondiag_colidx;

        *out_nondiag_block_colptr = nondiag_block_colptr;
        *out_nondiag_block_rowidx = nondiag_block_rowidx;
        *out_nondiag_block_rowptr = nondiag_block_rowptr;
        *out_nondiag_block_colidx = nondiag_block_colidx;
        *out_nondiag_block_csr_to_csc = nondiag_block_csr_to_csc;

        *out_related_nondiag_block_colptr = related_nondiag_block_colptr;
        *out_related_nondiag_block_rowidx = related_nondiag_block_rowidx;
        *out_related_nondiag_block_rowptr = related_nondiag_block_rowptr;
        *out_related_nondiag_block_colidx = related_nondiag_block_colidx;
        *out_related_nondiag_block_csr_to_csc = related_nondiag_block_csr_to_csc;
        *out_diag_uniaddr = diag_uniaddr;
    }
#undef _PANGULU_SET_BVALUE_SIZE
}

void pangulu_cm_recv_block(
    MPI_Status *msg_stat,
    pangulu_storage_t *storage,
    pangulu_uint64_t slot_addr,
    pangulu_exblock_idx block_length,
    pangulu_inblock_idx nb,
    pangulu_exblock_idx *bcol_pos,
    pangulu_exblock_idx *brow_pos,
    pangulu_exblock_ptr *related_nondiag_block_colptr,
    pangulu_exblock_idx *related_nondiag_block_rowidx,
    pangulu_uint64_t *related_nondiag_uniaddr,
    pangulu_uint64_t *diag_uniaddr)
{
#ifdef PANGULU_PERF
    global_stat.recv_cnt++;
    struct timeval start;
    pangulu_time_start(&start);
#endif

    pangulu_storage_slot_t *slot = pangulu_storage_get_slot(storage, slot_addr);
    MPI_Status mpi_stat;
    int mpi_count = 0;
    MPI_Get_count(msg_stat, MPI_CHAR, &mpi_count);
    MPI_Recv((char *)(slot->value) - 32, mpi_count, MPI_CHAR, msg_stat->MPI_SOURCE, msg_stat->MPI_TAG, MPI_COMM_WORLD, &mpi_stat);
    pangulu_inblock_ptr nnz = *(unsigned long long *)((char *)(slot->value) - 32);
    *brow_pos = *(unsigned int *)((char *)(slot->value) - 24);
    *bcol_pos = *(unsigned int *)((char *)(slot->value) - 20);

    size_t offset_from_val = 0;
    offset_from_val += sizeof(calculate_type) * nnz;
    slot->columnpointer = (char *)(slot->value) + offset_from_val;
    offset_from_val += sizeof(pangulu_inblock_ptr) * (nb + 1);
    slot->rowindex = ((char *)(slot->value)) + offset_from_val;
    offset_from_val += sizeof(pangulu_inblock_idx) * nnz;
    offset_from_val = (offset_from_val % 8) ? (offset_from_val / 8 * 8 + 8) : offset_from_val;
    if (*brow_pos > *bcol_pos)
    {
        slot->idx_of_csc_value_for_csr = ((char *)(slot->value)) + offset_from_val;
        offset_from_val += sizeof(pangulu_inblock_ptr) * nnz;
        slot->rowpointer = ((char *)(slot->value)) + offset_from_val;
        offset_from_val += sizeof(pangulu_inblock_ptr) * (nb + 1);
        slot->columnindex = ((char *)(slot->value)) + offset_from_val;
        offset_from_val += sizeof(pangulu_inblock_idx) * nnz;
        offset_from_val = (offset_from_val % 8) ? (offset_from_val / 8 * 8 + 8) : offset_from_val;
    }

    if (*brow_pos == *bcol_pos)
    {
        int is_upper = *(unsigned int *)((char *)(slot->value) - 16);
        slot->is_upper = is_upper;
        if (diag_uniaddr[*brow_pos] > block_length)
        {
            ((pangulu_storage_slot_t *)(diag_uniaddr[*brow_pos]))->related_block = slot;
            slot->related_block = ((pangulu_storage_slot_t *)(diag_uniaddr[*brow_pos]));
        }
        else
        {
            diag_uniaddr[*brow_pos] = (pangulu_uint64_t)slot;
            slot->related_block = NULL;
        }
        slot->brow_pos = *brow_pos;
        slot->bcol_pos = *bcol_pos;

#ifdef PANGULU_NONSHAREDMEM
        pangulu_platform_memcpy_async((char *)(slot->d_value) - 32, (char *)(slot->value) - 32, mpi_count, 0, NULL, PANGULU_DEFAULT_PLATFORM);
        if (is_upper)
        {
            slot->d_rowpointer = (((char *)(slot->d_value)) + sizeof(calculate_type) * nnz);
            slot->d_columnindex = (((char *)(slot->d_value)) + sizeof(calculate_type) * nnz + sizeof(pangulu_inblock_ptr) * (nb + 1));
        }
        else
        {
            slot->d_columnpointer = (((char *)(slot->d_value)) + sizeof(calculate_type) * nnz);
            slot->d_rowindex = (((char *)(slot->d_value)) + sizeof(calculate_type) * nnz + sizeof(pangulu_inblock_ptr) * (nb + 1));
        }
#endif
    }
    else
    {
        pangulu_exblock_ptr bidx = pangulu_binarysearch(
            related_nondiag_block_rowidx,
            related_nondiag_block_colptr[*bcol_pos],
            related_nondiag_block_colptr[(*bcol_pos) + 1],
            *brow_pos);
        related_nondiag_uniaddr[bidx] =
            PANGULU_DIGINFO_SET_BINID(PANGULU_DIGINFO_GET_BINID(slot_addr)) |
            PANGULU_DIGINFO_SET_NNZ(nnz) |
            PANGULU_DIGINFO_SET_SLOT_IDX(PANGULU_DIGINFO_GET_SLOT_IDX(slot_addr));
        slot->brow_pos = *brow_pos;
        slot->bcol_pos = *bcol_pos;

#ifdef PANGULU_NONSHAREDMEM
        slot->d_columnpointer = (((char *)(slot->d_value)) + sizeof(calculate_type) * nnz);
        slot->d_rowindex = (((char *)(slot->d_value)) + sizeof(calculate_type) * nnz + sizeof(pangulu_inblock_ptr) * (nb + 1));
        pangulu_platform_memcpy_async((char *)(slot->d_value) - 32, (char *)(slot->value) - 32, mpi_count, 0, NULL, PANGULU_DEFAULT_PLATFORM);

        if (*brow_pos > *bcol_pos)
        {
            pangulu_uint64_t byte_offset = sizeof(calculate_type) * nnz + sizeof(pangulu_inblock_ptr) * (nb + 1) + sizeof(pangulu_inblock_idx) * nnz;
            if (byte_offset % 8)
            {
                byte_offset = (byte_offset / 8) * 8 + 8;
            }
            slot->d_idx_of_csc_value_for_csr = (char *)(slot->d_value) + byte_offset;
            byte_offset += sizeof(pangulu_inblock_ptr) * nnz;
            slot->d_rowpointer = (char *)(slot->d_value) + byte_offset;
            byte_offset += sizeof(pangulu_inblock_ptr) * (nb + 1);
            slot->d_columnindex = (char *)(slot->d_value) + byte_offset;
        }
#endif
    }
#ifdef PANGULU_PERF
    global_stat.time_recv += pangulu_time_stop(&start);
#endif
}

void pangulu_cm_isend_block(
    pangulu_storage_slot_t *slot,
    pangulu_inblock_idx nb,
    pangulu_exblock_idx brow_pos,
    pangulu_exblock_idx bcol_pos,
    pangulu_int32_t target_rank)
{
    MPI_Request mpi_req;
    pangulu_inblock_ptr nnz = slot->columnpointer[nb];
    pangulu_int64_t size = 32 +
                           sizeof(pangulu_inblock_ptr) * (nb + 1) +
                           sizeof(pangulu_inblock_idx) * nnz +
                           sizeof(calculate_type) * nnz;
    if (size % 8)
    {
        size = (size / 8) * 8 + 8;
    }
    if (brow_pos > bcol_pos)
    {
        size += sizeof(pangulu_inblock_ptr) * (nb + 1) +
                sizeof(pangulu_inblock_idx) * nnz +
                sizeof(pangulu_inblock_ptr) * nnz;
        if (size % 8)
        {
            size = (size / 8) * 8 + 8;
        }
    }
    if (brow_pos == bcol_pos)
    {
        if (slot->is_upper)
        {
            *(unsigned int *)((char *)(slot->value) - 16) = 1;
        }
        else
        {
            *(unsigned int *)((char *)(slot->value) - 16) = 0;
        }
    }
    *(unsigned long *)((char *)(slot->value) - 32) = nnz;
    *(unsigned int *)((char *)(slot->value) - 24) = brow_pos;
    *(unsigned int *)((char *)(slot->value) - 20) = bcol_pos;
    MPI_Isend((char *)(slot->value) - 32, size, MPI_CHAR, target_rank, 0, MPI_COMM_WORLD, &mpi_req);
}
