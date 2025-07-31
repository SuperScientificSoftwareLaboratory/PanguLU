#include "pangulu_common.h"

int storage_full_warning_flag = 0;
long long wait_time = 10000;
int wait_count = 0;

pangulu_int32_t pangulu_storage_bin_navail(pangulu_storage_bin_t *bin)
{
    return (bin->queue_tail + (bin->slot_count + 1) - bin->queue_head) % (bin->slot_count + 1);
}

pangulu_int32_t pangulu_storage_slot_queue_alloc(pangulu_storage_bin_t *bin)
{
    if (bin->queue_head == bin->queue_tail)
    {
        printf(PANGULU_E_BIN_FULL);
        exit(1);
    }
    pangulu_int32_t slot_idx = bin->avail_slot_queue[bin->queue_head];
    bin->queue_head = (bin->queue_head + 1) % (bin->slot_count + 1);
    return slot_idx;
}

void pangulu_storage_slot_queue_recycle(
    pangulu_storage_t *storage,
    pangulu_uint64_t *slot_addr)
{
    pthread_mutex_lock(storage->mutex);
    pangulu_storage_bin_t *bin = &storage->bins[PANGULU_DIGINFO_GET_BINID(*slot_addr)];
    pangulu_storage_slot_t *slot = &bin->slots[PANGULU_DIGINFO_GET_SLOT_IDX(*slot_addr)];
    calculate_type *value = slot->value;
#ifdef PANGULU_NONSHAREDMEM
    calculate_type *d_value = slot->d_value;
#endif
    memset(slot, 0, sizeof(pangulu_storage_slot_t));
    slot->data_status = PANGULU_DATA_INVALID;
    slot->value = value;
#ifdef PANGULU_NONSHAREDMEM
    slot->d_value = d_value;
#endif
    if (bin->queue_head == ((bin->queue_tail + 1) % (bin->slot_count + 1)))
    {
        printf(PANGULU_E_RECYCLE_QUEUE_FULL);
        exit(1);
    }
    bin->queue_head = (bin->queue_head - 1 + (bin->slot_count + 1)) % (bin->slot_count + 1);
    bin->avail_slot_queue[bin->queue_head] = PANGULU_DIGINFO_GET_SLOT_IDX(*slot_addr);
    *slot_addr = PANGULU_DIGINFO_SET_NNZ(PANGULU_DIGINFO_GET_NNZ(*slot_addr));
    pthread_mutex_unlock(storage->mutex);
}

void pangulu_storage_slot_queue_recycle_by_ptr(
    pangulu_storage_t *storage,
    pangulu_storage_slot_t *slot)
{
    if (slot == NULL)
    {
        return;
    }
    pthread_mutex_lock(storage->mutex);
    pangulu_storage_bin_t *bin = &storage->bins[slot->bin_id];
    calculate_type *value = slot->value;
#ifdef PANGULU_NONSHAREDMEM
    calculate_type *d_value = slot->d_value;
#endif
    pangulu_int32_t slot_idx = slot->slot_idx;
    memset(slot, 0, sizeof(pangulu_storage_slot_t));
    slot->data_status = PANGULU_DATA_INVALID;
    slot->value = value;
#ifdef PANGULU_NONSHAREDMEM
    slot->d_value = d_value;
#endif
    if (bin->queue_head == ((bin->queue_tail + 1) % (bin->slot_count + 1)))
    {
        printf(PANGULU_E_RECYCLE_QUEUE_FULL);
        exit(1);
    }
    bin->queue_head = (bin->queue_head - 1 + (bin->slot_count + 1)) % (bin->slot_count + 1);
    bin->avail_slot_queue[bin->queue_head] = slot_idx;
    pthread_mutex_unlock(storage->mutex);
}

pangulu_uint64_t pangulu_storage_allocate_slot(
    pangulu_storage_t *storage,
    pangulu_int64_t size)
{
    pthread_mutex_lock(storage->mutex);
    pangulu_uint64_t slot_addr = 0xFFFFFFFFFFFFFFFF;
    for (pangulu_int32_t bin_id = 1; bin_id < storage->n_bin; bin_id++)
    {
        if (storage->bins[bin_id].slot_capacity >= size)
        {
            if (pangulu_storage_bin_navail(&storage->bins[bin_id]) == 0)
            {
                continue;
            }
            pangulu_int32_t slot_idx = pangulu_storage_slot_queue_alloc(&storage->bins[bin_id]);
            storage->bins[bin_id].slots[slot_idx].data_status = PANGULU_DATA_PREPARING;
            storage->bins[bin_id].slots[slot_idx].bin_id = bin_id;
            storage->bins[bin_id].slots[slot_idx].slot_idx = slot_idx;
            slot_addr = 0;
            slot_addr |= PANGULU_DIGINFO_SET_SLOT_IDX(slot_idx);
            slot_addr |= PANGULU_DIGINFO_SET_BINID(bin_id);
            break;
        }
    }
    pthread_mutex_unlock(storage->mutex);

    if (slot_addr == 0xFFFFFFFFFFFFFFFF)
    {
        if (storage_full_warning_flag == 0)
        {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0)
            {
                printf(PANGULU_W_MPI_BUF_FULL);
            }
            storage_full_warning_flag = 1;
        }
        usleep(wait_time);
        if (wait_time < 1000000)
        {
            wait_time *= 2;
        }
        wait_count++;
        if(wait_count >= 10){
            printf(PANGULU_E_MPI_BUF_WAIT_EXCEED);
            exit(1);
        }
        return pangulu_storage_allocate_slot(storage, size);
    }
    else
    {
        wait_time = 10000;
        wait_count = 0;
        return slot_addr;
    }
}

pangulu_storage_slot_t *pangulu_storage_get_slot(
    pangulu_storage_t *storage,
    pangulu_uint64_t slot_addr)
{
    if (PANGULU_DIGINFO_GET_BINID(slot_addr) == 7)
    {
        return NULL;
    }
    return &(storage->bins[PANGULU_DIGINFO_GET_BINID(slot_addr)].slots[PANGULU_DIGINFO_GET_SLOT_IDX(slot_addr)]);
}

pangulu_storage_slot_t *pangulu_storage_get_diag(
    pangulu_exblock_idx block_length,
    pangulu_storage_t *storage,
    pangulu_uint64_t diag_addr)
{
    if (!diag_addr)
    {
        return NULL;
    }
    if (diag_addr > block_length)
    {
        return (pangulu_storage_slot_t *)diag_addr;
    }
    else
    {
        return &(storage->bins[0].slots[storage->bins[0].nondiag_slot_cnt + 2 * (diag_addr - 1)]);
    }
}

void pangulu_storage_bin_init(
    pangulu_storage_bin_t *bin,
    pangulu_int32_t bin_id,
    pangulu_int64_t slot_capacity,
    pangulu_int32_t slot_count)
{
    bin->slot_count = slot_count;
    bin->slot_capacity = slot_capacity;
    bin->slots = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_storage_slot_t) * slot_count);
    bin->avail_slot_queue = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * (slot_count + 1));
    bin->queue_head = 0;
    bin->queue_tail = slot_count;
    memset(bin->slots, 0, sizeof(pangulu_storage_slot_t) * slot_count);
    char *h_bin_buffer = pangulu_malloc(__FILE__, __LINE__, slot_capacity * slot_count);
#ifdef PANGULU_NONSHAREDMEM
    char *d_bin_buffer = NULL;
    pangulu_platform_malloc(&(d_bin_buffer), slot_capacity * slot_count, PANGULU_DEFAULT_PLATFORM);
#endif
    for (pangulu_int32_t i = 0; i < slot_count; i++)
    {
        bin->slots[i].value = (h_bin_buffer + i * slot_capacity + 32);
#ifdef PANGULU_NONSHAREDMEM
        bin->slots[i].d_value = (d_bin_buffer + i * slot_capacity + 32);
#endif
        bin->avail_slot_queue[i] = i;
    }
}

void pangulu_storage_init(
    pangulu_storage_t *storage,
    pangulu_int64_t *slot_capacity,
    pangulu_int32_t *slot_count,
    pangulu_exblock_idx block_length,
    pangulu_exblock_ptr *bcsc_pointer,
    pangulu_exblock_idx *bcsc_index,
    pangulu_inblock_ptr **bcsc_inblk_pointers,
    pangulu_inblock_idx **bcsc_inblk_indeces,
    calculate_type **bcsc_inblk_values,
    pangulu_inblock_ptr **bcsr_inblk_pointers,
    pangulu_inblock_idx **bcsr_inblk_indeces,
    pangulu_inblock_ptr **bcsr_inblk_valueindices,
    pangulu_uint64_t *diag_uniaddr,
    pangulu_inblock_ptr **diag_upper_rowptr,
    pangulu_inblock_idx **diag_upper_colidx,
    calculate_type **diag_upper_csrvalue,
    pangulu_inblock_ptr **diag_lower_colptr,
    pangulu_inblock_idx **diag_lower_rowidx,
    calculate_type **diag_lower_cscvalue,
    pangulu_inblock_idx nb)
{
    pangulu_int64_t storage_size = sizeof(pangulu_storage_bin_t) * storage->n_bin;
    storage->n_bin = 7;
    storage->bins = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_storage_bin_t) * storage->n_bin);
    for (pangulu_int32_t i_bin = 1; i_bin < storage->n_bin; i_bin++)
    {
        pangulu_storage_bin_init(&storage->bins[i_bin], i_bin, slot_capacity[i_bin], slot_count[i_bin]);
        storage_size += sizeof(pangulu_storage_slot_t) * slot_count[i_bin];
        storage_size += sizeof(pangulu_int32_t) * (slot_count[i_bin] + 1);
        storage_size += slot_capacity[i_bin] * slot_count[i_bin];
    }
    pangulu_exblock_idx ndiag = 0;
    for (pangulu_exblock_idx level = 0; level < block_length; level++)
    {
        if (diag_uniaddr[level])
        {
            ndiag++;
        }
    }
    pangulu_storage_bin_t *bin0 = &storage->bins[0];
    bin0->slot_count = (bcsc_pointer[block_length] + ndiag * 2);
    bin0->nondiag_slot_cnt = bcsc_pointer[block_length];
    bin0->slot_capacity = 0;
    bin0->slots = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_storage_slot_t) * bin0->slot_count);
    bin0->avail_slot_queue = NULL;
    bin0->queue_head = 0;
    bin0->queue_tail = 0;
    for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
    {
        for (pangulu_exblock_ptr bidx = bcsc_pointer[bcol]; bidx < bcsc_pointer[bcol + 1]; bidx++)
        {
            bin0->slots[bidx].columnpointer = bcsc_inblk_pointers[bidx];
            bin0->slots[bidx].rowindex = bcsc_inblk_indeces[bidx];
            bin0->slots[bidx].value = bcsc_inblk_values[bidx];
            bin0->slots[bidx].data_status = PANGULU_DATA_PREPARING;
            bin0->slots[bidx].brow_pos = bcsc_index[bidx];
            bin0->slots[bidx].bcol_pos = bcol;
            bin0->slots[bidx].task_queue = NULL;
            if (bin0->slots[bidx].brow_pos > bin0->slots[bidx].bcol_pos)
            {
                bin0->slots[bidx].rowpointer = bcsr_inblk_pointers[bidx];
                bin0->slots[bidx].columnindex = bcsr_inblk_indeces[bidx];
                bin0->slots[bidx].idx_of_csc_value_for_csr = bcsr_inblk_valueindices[bidx];
            }
        }
    }
    for (pangulu_exblock_idx level = 0; level < block_length; level++)
    {
        if (!diag_uniaddr[level])
        {
            continue;
        }
        pangulu_exblock_idx diag_local_idx = diag_uniaddr[level] - 1;

        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].value = diag_lower_cscvalue[diag_local_idx];
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].columnpointer = diag_lower_colptr[diag_local_idx];
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].rowindex = diag_lower_rowidx[diag_local_idx];
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].data_status = PANGULU_DATA_PREPARING;
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].brow_pos = level;
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].bcol_pos = level;
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].is_upper = 0;
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].related_block = &(bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1]);
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].task_queue = NULL;

        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].value = diag_upper_csrvalue[diag_local_idx];
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].columnpointer = diag_upper_rowptr[diag_local_idx];
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].rowindex = diag_upper_colidx[diag_local_idx];
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].data_status = PANGULU_DATA_PREPARING;
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].brow_pos = level;
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].bcol_pos = level;
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].is_upper = 1;
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].related_block = &(bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx]);
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].task_queue = NULL;
    }

#ifdef PANGULU_NONSHAREDMEM
    char *d_memptr = NULL;
    for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
    {
        for (pangulu_exblock_ptr bidx = bcsc_pointer[bcol]; bidx < bcsc_pointer[bcol + 1]; bidx++)
        {
            pangulu_exblock_idx brow = bcsc_index[bidx];
            pangulu_exblock_ptr nnz = bin0->slots[bidx].columnpointer[nb];
            d_memptr += sizeof(pangulu_int64_t) * 4;
            bin0->slots[bidx].d_value = d_memptr;
            d_memptr += sizeof(calculate_type) * nnz;
            bin0->slots[bidx].d_columnpointer = d_memptr;
            d_memptr += sizeof(pangulu_inblock_ptr) * (nb + 1);
            bin0->slots[bidx].d_rowindex = d_memptr;
            d_memptr += sizeof(pangulu_inblock_idx) * nnz;
            if (brow > bcol)
            {
                size_t align = 8 - ((uintptr_t)d_memptr % 8);
                if (align != 8)
                    d_memptr += align;
                bin0->slots[bidx].d_idx_of_csc_value_for_csr = d_memptr;
                d_memptr += sizeof(pangulu_inblock_ptr) * nnz;
                bin0->slots[bidx].d_rowpointer = d_memptr;
                d_memptr += sizeof(pangulu_inblock_ptr) * (nb + 1);
                bin0->slots[bidx].d_columnindex = d_memptr;
                d_memptr += sizeof(pangulu_inblock_idx) * nnz;
            }
            size_t align = 8 - ((uintptr_t)d_memptr % 8);
            if (align != 8)
                d_memptr += align;
        }
    }
    size_t nondiag_mem_size = (size_t)d_memptr;
    for (pangulu_exblock_idx level = 0; level < block_length; level++)
    {
        if (!diag_uniaddr[level])
        {
            continue;
        }
        pangulu_exblock_idx diag_local_idx = diag_uniaddr[level] - 1;

        d_memptr += sizeof(pangulu_int64_t) * 4;
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].d_value = d_memptr;
        d_memptr += sizeof(calculate_type) * bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].columnpointer[nb];
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].d_columnpointer = d_memptr;
        d_memptr += sizeof(pangulu_inblock_ptr) * (nb + 1);
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].d_rowindex = d_memptr;
        d_memptr += sizeof(pangulu_inblock_idx) * bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].columnpointer[nb];
        size_t align = 8 - ((uintptr_t)d_memptr % 8);
        if (align != 8)
            d_memptr += align;
        d_memptr += sizeof(pangulu_int64_t) * 4;
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].d_value = d_memptr;
        d_memptr += sizeof(calculate_type) * bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].columnpointer[nb];
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].d_rowpointer = d_memptr;
        d_memptr += sizeof(pangulu_inblock_ptr) * (nb + 1);
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].d_columnindex = d_memptr;
        d_memptr += sizeof(pangulu_inblock_idx) * bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].columnpointer[nb];
        align = 8 - ((uintptr_t)d_memptr % 8);
        if (align != 8)
            d_memptr += align;
    }
    size_t mem_size = (size_t)d_memptr;

    pangulu_platform_malloc(&(d_memptr), mem_size, PANGULU_DEFAULT_PLATFORM);
    char *h_memptr = (char *)(bin0->slots[0].value) - 32;
    pangulu_platform_memcpy(d_memptr, h_memptr, nondiag_mem_size, 0, PANGULU_DEFAULT_PLATFORM);
    for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
    {
        for (pangulu_exblock_ptr bidx = bcsc_pointer[bcol]; bidx < bcsc_pointer[bcol + 1]; bidx++)
        {
            pangulu_exblock_idx brow = bcsc_index[bidx];
            bin0->slots[bidx].d_value = d_memptr + (size_t)(bin0->slots[bidx].d_value);
            bin0->slots[bidx].d_columnpointer = d_memptr + (size_t)(bin0->slots[bidx].d_columnpointer);
            bin0->slots[bidx].d_rowindex = d_memptr + (size_t)(bin0->slots[bidx].d_rowindex);
            if (brow > bcol)
            {
                bin0->slots[bidx].d_idx_of_csc_value_for_csr = d_memptr + (size_t)(bin0->slots[bidx].d_idx_of_csc_value_for_csr);
                bin0->slots[bidx].d_rowpointer = d_memptr + (size_t)(bin0->slots[bidx].d_rowpointer);
                bin0->slots[bidx].d_columnindex = d_memptr + (size_t)(bin0->slots[bidx].d_columnindex);
            }
        }
    }
    for (pangulu_exblock_idx level = 0; level < block_length; level++)
    {
        if (!diag_uniaddr[level])
        {
            continue;
        }
        pangulu_exblock_idx diag_local_idx = diag_uniaddr[level] - 1;

        pangulu_inblock_ptr nnz = bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].columnpointer[nb];
        size_t size = 32 + sizeof(calculate_type) * nnz + sizeof(pangulu_inblock_ptr) * (nb + 1) + sizeof(pangulu_inblock_idx) * nnz;
        if (size % 8)
        {
            size = (size / 8) * 8 + 8;
        }
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].d_value = d_memptr + (size_t)(bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].d_value);
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].d_columnpointer = d_memptr + (size_t)(bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].d_columnpointer);
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].d_rowindex = d_memptr + (size_t)(bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].d_rowindex);
        pangulu_platform_memcpy(
            bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].d_value,
            bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].value,
            sizeof(calculate_type) * nnz, 0, PANGULU_DEFAULT_PLATFORM);
        pangulu_platform_memcpy(
            bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].d_columnpointer,
            bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].columnpointer,
            sizeof(pangulu_inblock_ptr) * (nb + 1), 0, PANGULU_DEFAULT_PLATFORM);
        pangulu_platform_memcpy(
            bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].d_rowindex,
            bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx].rowindex,
            sizeof(pangulu_inblock_idx) * nnz, 0, PANGULU_DEFAULT_PLATFORM);

        nnz = bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].columnpointer[nb];
        size = 32 + sizeof(calculate_type) * nnz + sizeof(pangulu_inblock_ptr) * (nb + 1) + sizeof(pangulu_inblock_idx) * nnz;
        if (size % 8)
        {
            size = (size / 8) * 8 + 8;
        }
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].d_value = d_memptr + (size_t)(bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].d_value);
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].d_rowpointer = d_memptr + (size_t)(bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].d_rowpointer);
        bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].d_columnindex = d_memptr + (size_t)(bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].d_columnindex);
        pangulu_platform_memcpy(
            (char *)(bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].d_value) - 32,
            (char *)(bin0->slots[bcsc_pointer[block_length] + 2 * diag_local_idx + 1].value) - 32,
            size, 0, PANGULU_DEFAULT_PLATFORM);
    }
#endif

    storage->mutex = pangulu_malloc(__FILE__, __LINE__, sizeof(pthread_mutex_t));
    pthread_mutex_init(storage->mutex, NULL);
}