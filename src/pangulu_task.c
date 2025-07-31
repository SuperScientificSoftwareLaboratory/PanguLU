#include "pangulu_common.h"

pthread_mutex_t pangulu_aggregate_map_mutex;
void *pangulu_aggregate_task_buf = NULL;
unsigned long long pangulu_aggregate_task_buf_capacity = 0;

int pangulu_aggregate_init()
{
    pthread_mutex_init(&pangulu_aggregate_map_mutex, NULL);
    return 0;
}

int pangulu_aggregate_task_store(pangulu_storage_slot_t *opdst, pangulu_task_t *task_descriptor)
{
    pthread_mutex_lock(&pangulu_aggregate_map_mutex);
    pangulu_aggregate_queue_t *tq = opdst->task_queue;
    if (!tq)
    {
        tq = (pangulu_aggregate_queue_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_aggregate_queue_t));
        memset(tq, 0, sizeof(pangulu_aggregate_queue_t));
        tq->capacity = PANGULU_AGGR_QUEUE_MIN_CAP;
        tq->task_descriptors = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_task_t) * tq->capacity);
        opdst->task_queue = tq;
    }
    if (tq->length >= tq->capacity)
    {
        while (tq->length >= tq->capacity)
        {
            tq->capacity *= PANGULU_AGGR_QUEUE_INCREASE_SPEED;
        }
        tq->task_descriptors = pangulu_realloc(__FILE__, __LINE__, tq->task_descriptors, sizeof(pangulu_task_t) * tq->capacity);
    }
    memcpy((char *)tq->task_descriptors + sizeof(pangulu_task_t) * tq->length, task_descriptor, sizeof(pangulu_task_t));
    tq->length++;
    pthread_mutex_unlock(&pangulu_aggregate_map_mutex);
    return 0;
}

int pangulu_aggregate_task_compute(
    pangulu_storage_slot_t *opdst,
    int (*compute_callback)(unsigned long long, void *, void *),
    void *extra_params)
{
    pthread_mutex_lock(&pangulu_aggregate_map_mutex);
    pangulu_aggregate_queue_t *tq = opdst->task_queue;
    if (!tq)
    {
        pthread_mutex_unlock(&pangulu_aggregate_map_mutex);
        return 0;
    }
    unsigned long long ntasks = tq->length;
    void *task_descriptors = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_task_t) * ntasks);
    memcpy(task_descriptors, tq->task_descriptors, sizeof(pangulu_task_t) * ntasks);
    pangulu_free(__FILE__, __LINE__, opdst->task_queue->task_descriptors);
    pangulu_free(__FILE__, __LINE__, opdst->task_queue);
    opdst->task_queue = NULL;
    pthread_mutex_unlock(&pangulu_aggregate_map_mutex);
    return compute_callback(ntasks, task_descriptors, extra_params);
}

int pangulu_aggregate_task_compute_multi_tile(
    unsigned long long ntile,
    pangulu_storage_slot_t **opdst,
    int (*compute_callback)(unsigned long long, void *, void *),
    void *extra_params)
{
    pthread_mutex_lock(&pangulu_aggregate_map_mutex);
    unsigned long long ntasks_batched = 0;
    for (unsigned long long i = 0; i < ntile; ++i)
    {
        pangulu_aggregate_queue_t *tq = opdst[i]->task_queue;
        if (!tq)
            continue;
        unsigned long long ntasks = tq->length;
        if (ntasks_batched + ntasks > pangulu_aggregate_task_buf_capacity)
        {
            pangulu_aggregate_task_buf_capacity = ntasks_batched + ntasks;
            pangulu_aggregate_task_buf = pangulu_realloc(__FILE__, __LINE__, pangulu_aggregate_task_buf, sizeof(pangulu_task_t) * pangulu_aggregate_task_buf_capacity);
        }
        memcpy((char *)pangulu_aggregate_task_buf + sizeof(pangulu_task_t) * ntasks_batched,
               tq->task_descriptors, sizeof(pangulu_task_t) * ntasks);
        ntasks_batched += ntasks;
        pangulu_free(__FILE__, __LINE__, opdst[i]->task_queue->task_descriptors);
        pangulu_free(__FILE__, __LINE__, opdst[i]->task_queue);
        opdst[i]->task_queue = NULL;
    }
    pthread_mutex_unlock(&pangulu_aggregate_map_mutex);
    if (ntasks_batched == 0)
        return 0;
    return compute_callback(ntasks_batched, pangulu_aggregate_task_buf, extra_params);
}

int pangulu_aggregate_idle_batch(
    pangulu_common *common,
    pangulu_block_common *block_common,
    pangulu_block_smatrix *block_smatrix,
    int chunk_ntask,
    int (*compute_callback)(unsigned long long, void *, void *),
    void *extra_params)
{
    pthread_mutex_lock(&pangulu_aggregate_map_mutex);
    unsigned long long ntasks_batched = 0;
    for (size_t i = 0; i < block_common->block_length && chunk_ntask > 0; ++i)
    {
        for (int _ = 0; _ < 1; _++)
        {
            pangulu_storage_slot_t *slot = pangulu_storage_get_diag(block_common->block_length, block_smatrix->storage, block_smatrix->diag_uniaddr[i]);
            if (!slot)
                continue;
            if (!slot->task_queue)
                continue;
            pangulu_aggregate_queue_t *tq = slot->task_queue;
            unsigned long long ntasks = tq->length;
            if (ntasks_batched + ntasks > pangulu_aggregate_task_buf_capacity)
            {
                pangulu_aggregate_task_buf_capacity = ntasks_batched + ntasks;
                pangulu_aggregate_task_buf = pangulu_realloc(__FILE__, __LINE__, pangulu_aggregate_task_buf, sizeof(pangulu_task_t) * pangulu_aggregate_task_buf_capacity);
            }
            memcpy((char *)pangulu_aggregate_task_buf + sizeof(pangulu_task_t) * ntasks_batched,
                   tq->task_descriptors, sizeof(pangulu_task_t) * ntasks);
            ntasks_batched += ntasks;
            chunk_ntask -= ntasks;
            free(tq->task_descriptors);
            free(tq);
            slot->task_queue = NULL;
        }
        for (size_t j = block_smatrix->nondiag_block_colptr[i]; j < block_smatrix->nondiag_block_colptr[i + 1] && chunk_ntask > 0; ++j)
        {
            pangulu_storage_slot_t *slot = &block_smatrix->storage->bins[0].slots[j];
            if (!slot->task_queue)
                continue;
            pangulu_aggregate_queue_t *tq = slot->task_queue;
            unsigned long long ntasks = tq->length;
            if (ntasks_batched + ntasks > pangulu_aggregate_task_buf_capacity)
            {
                pangulu_aggregate_task_buf_capacity = ntasks_batched + ntasks;
                pangulu_aggregate_task_buf = pangulu_realloc(__FILE__, __LINE__, pangulu_aggregate_task_buf, sizeof(pangulu_task_t) * pangulu_aggregate_task_buf_capacity);
            }
            memcpy((char *)pangulu_aggregate_task_buf + sizeof(pangulu_task_t) * ntasks_batched,
                   tq->task_descriptors, sizeof(pangulu_task_t) * ntasks);
            ntasks_batched += ntasks;
            chunk_ntask -= ntasks;
            free(tq->task_descriptors);
            free(tq);
            slot->task_queue = NULL;
        }
        for (size_t j = block_smatrix->nondiag_block_rowptr[i]; j < block_smatrix->nondiag_block_rowptr[i + 1] && chunk_ntask > 0; ++j)
        {
            pangulu_storage_slot_t *slot = &block_smatrix->storage->bins[0].slots[block_smatrix->nondiag_block_csr_to_csc[j]];
            if (!slot->task_queue)
                continue;
            pangulu_aggregate_queue_t *tq = slot->task_queue;
            unsigned long long ntasks = tq->length;
            if (ntasks_batched + ntasks > pangulu_aggregate_task_buf_capacity)
            {
                pangulu_aggregate_task_buf_capacity = ntasks_batched + ntasks;
                pangulu_aggregate_task_buf = pangulu_realloc(__FILE__, __LINE__, pangulu_aggregate_task_buf, sizeof(pangulu_task_t) * pangulu_aggregate_task_buf_capacity);
            }
            memcpy((char *)pangulu_aggregate_task_buf + sizeof(pangulu_task_t) * ntasks_batched,
                   tq->task_descriptors, sizeof(pangulu_task_t) * ntasks);
            ntasks_batched += ntasks;
            chunk_ntask -= ntasks;
            free(tq->task_descriptors);
            free(tq);
            slot->task_queue = NULL;
        }
        if (chunk_ntask <= 0)
        {
            break;
        }
    }

    pthread_mutex_unlock(&pangulu_aggregate_map_mutex);
    if (ntasks_batched == 0)
        return 0;
    return compute_callback(ntasks_batched, pangulu_aggregate_task_buf, extra_params);
}

pangulu_int64_t pangulu_task_queue_alloc(pangulu_task_queue_t *tq)
{
    if (tq->task_storage_avail_queue_head == tq->task_storage_avail_queue_tail)
    {
        printf(PANGULU_E_TASK_QUEUE_FULL);
        exit(1);
    }
    pangulu_int64_t store_idx = tq->task_storage_avail_queue_head;
    tq->task_storage_avail_queue_head = (tq->task_storage_avail_queue_head + 1) % (tq->capacity + 1);
    return tq->task_storage_avail_queue[store_idx];
}

void pangulu_task_queue_recycle(
    pangulu_task_queue_t *tq,
    pangulu_int64_t store_idx)
{
    if (tq->task_storage_avail_queue_head == (tq->task_storage_avail_queue_tail + 1) % (tq->capacity + 1))
    {
        printf(PANGULU_E_RECYCLE_QUEUE_FULL);
        exit(1);
    }
    tq->task_storage_avail_queue[tq->task_storage_avail_queue_tail] = store_idx;
    tq->task_storage_avail_queue_tail = (tq->task_storage_avail_queue_tail + 1) % (tq->capacity + 1);
}

void pangulu_task_queue_cmp_strategy(
    pangulu_task_queue_t *tq,
    pangulu_int32_t cmp_strategy)
{
    tq->cmp_strategy = cmp_strategy;
}

void pangulu_task_queue_init(
    pangulu_task_queue_t *heap,
    pangulu_int64_t capacity)
{
    pangulu_int64_t size = 0;
    pangulu_task_t *compare_queue = (pangulu_task_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_task_t) * capacity);
    size += sizeof(pangulu_task_t) * capacity;
    pangulu_int64_t *task_index_heap = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * capacity);
    size += sizeof(pangulu_int64_t) * capacity;
    heap->task_storage = compare_queue;
    heap->task_index_heap = task_index_heap;
    heap->capacity = capacity;
    heap->length = 0;
    heap->heap_bsem = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_bsem_t));
    heap->heap_bsem->mutex = pangulu_malloc(__FILE__, __LINE__, sizeof(pthread_mutex_t));
    pthread_mutex_init(heap->heap_bsem->mutex, NULL);
    heap->task_storage_avail_queue = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (capacity + 1));
    size += sizeof(pangulu_int64_t) * (capacity + 1);
    heap->task_storage_avail_queue_head = 0;
    heap->task_storage_avail_queue_tail = capacity;
    for (pangulu_int64_t i = 0; i < capacity; i++)
    {
        heap->task_storage_avail_queue[i] = i;
    }
    pangulu_task_queue_cmp_strategy(heap, 0);
}

pangulu_task_queue_t *pangulu_task_queue_destory(pangulu_task_queue_t *heap)
{
    if (heap != NULL)
    {
        pangulu_free(__FILE__, __LINE__, heap->task_storage);
        pangulu_free(__FILE__, __LINE__, heap->task_index_heap);
        heap->length = 0;
        heap->capacity = 0;
    }
    pangulu_free(__FILE__, __LINE__, heap);
    return NULL;
}

void pangulu_task_queue_clear(pangulu_task_queue_t *heap)
{
    heap->length = 0;
    heap->task_storage_avail_queue_head = 0;
    heap->task_storage_avail_queue_tail = heap->capacity;
    for (pangulu_int64_t i = 0; i < heap->capacity; i++)
    {
        heap->task_storage_avail_queue[i] = i;
    }
}

char pangulu_task_compare(
    pangulu_task_t *compare_queue,
    pangulu_int64_t a,
    pangulu_int64_t b,
    pangulu_int32_t heap_select)
{
    if (0 == heap_select)
    {
        if (compare_queue[a].compare_flag == compare_queue[b].compare_flag)
        {
            pangulu_int64_t compare_flag_a = compare_queue[a].row + compare_queue[a].col - compare_queue[a].compare_flag;
            pangulu_int64_t compare_flag_b = compare_queue[b].row + compare_queue[b].col - compare_queue[b].compare_flag;

            return compare_flag_a < compare_flag_b;
        }
        else
        {
            return compare_queue[a].compare_flag < compare_queue[b].compare_flag;
        }
    }
    else if (1 == heap_select)
    {
        if (compare_queue[a].kernel_id == compare_queue[b].kernel_id)
        {

            pangulu_int64_t compare_flag_a = compare_queue[a].row + compare_queue[a].col - compare_queue[a].compare_flag;
            pangulu_int64_t compare_flag_b = compare_queue[b].row + compare_queue[b].col - compare_queue[b].compare_flag;

            if (compare_flag_a == compare_flag_b)
            {
                return compare_queue[a].compare_flag < compare_queue[b].compare_flag;
            }
            else
            {
                return compare_flag_a < compare_flag_b;
            }
        }
        else
        {
            return compare_queue[a].kernel_id < compare_queue[b].kernel_id;
        }
    }
    else if (2 == heap_select)
    {
        if (compare_queue[a].kernel_id == compare_queue[b].kernel_id)
        {
            if (compare_queue[a].compare_flag == compare_queue[b].compare_flag)
            {
                pangulu_int64_t compare_flag_a = compare_queue[a].row + compare_queue[a].col - compare_queue[a].compare_flag;
                pangulu_int64_t compare_flag_b = compare_queue[b].row + compare_queue[b].col - compare_queue[b].compare_flag;
                return compare_flag_a < compare_flag_b;
            }
            else
            {
                return compare_queue[a].compare_flag < compare_queue[b].compare_flag;
            }
        }
        else
        {
            return compare_queue[a].kernel_id < compare_queue[b].kernel_id;
        }
    }
    else if (3 == heap_select)
    {
        pangulu_int64_t compare_flag_a = compare_queue[a].row + compare_queue[a].col - compare_queue[a].compare_flag;
        pangulu_int64_t compare_flag_b = compare_queue[b].row + compare_queue[b].col - compare_queue[b].compare_flag;
        return compare_flag_a < compare_flag_b;
    }
    else if (4 == heap_select)
    {
        if (compare_queue[a].compare_flag == compare_queue[b].compare_flag)
        {
            pangulu_int64_t compare_flag_a = compare_queue[a].row + compare_queue[a].col - compare_queue[a].compare_flag;
            pangulu_int64_t compare_flag_b = compare_queue[b].row + compare_queue[b].col - compare_queue[b].compare_flag;

            return compare_flag_a > compare_flag_b;
        }
        else
        {
            return compare_queue[a].compare_flag > compare_queue[b].compare_flag;
        }
    }
    else
    {
        printf(PANGULU_E_INVALID_HEAP_SELECT);
        exit(1);
    }
}

void pangulu_task_swap(pangulu_int64_t *task_index_heap, pangulu_int64_t a, pangulu_int64_t b)
{
    pangulu_int64_t temp = task_index_heap[a];
    task_index_heap[a] = task_index_heap[b];
    task_index_heap[b] = temp;
}

void pangulu_task_queue_push(
    pangulu_task_queue_t *heap,
    pangulu_int64_t row,
    pangulu_int64_t col,
    pangulu_int64_t task_level,
    pangulu_int64_t kernel_id,
    pangulu_int64_t compare_flag,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *op1,
    pangulu_storage_slot_t *op2,
    pangulu_int64_t block_length,
    const char *file,
    int line)
{
    pthread_mutex_lock(heap->heap_bsem->mutex);
    if (kernel_id == PANGULU_TASK_SSSSM)
    {
        pangulu_task_t store_task;
        store_task.row = row;
        store_task.col = col;
        store_task.task_level = task_level;
        store_task.kernel_id = kernel_id;
        store_task.compare_flag = compare_flag;
        store_task.opdst = opdst;
        store_task.op1 = op1;
        store_task.op2 = op2;
        pangulu_aggregate_task_store(opdst, &store_task);
    }
    pangulu_task_t *task_storage = heap->task_storage;
    pangulu_int64_t *task_index_heap = heap->task_index_heap;
    if (heap->length >= heap->capacity)
    {
        printf(PANGULU_E_HEAP_FULL);
        exit(1);
    }
    pangulu_int64_t store_idx = pangulu_task_queue_alloc(heap);
    task_storage[store_idx].row = row;
    task_storage[store_idx].col = col;
    task_storage[store_idx].task_level = task_level;
    task_storage[store_idx].kernel_id = kernel_id;
    task_storage[store_idx].compare_flag = compare_flag;
    task_storage[store_idx].opdst = opdst;
    task_storage[store_idx].op1 = op1;
    task_storage[store_idx].op2 = op2;
    task_index_heap[heap->length] = store_idx;
    pangulu_int64_t son = heap->length;
    pangulu_int64_t parent = (son - 1) / 2;
    while (son != 0 && parent >= 0)
    {
        if (pangulu_task_compare(task_storage, task_index_heap[son], task_index_heap[parent], heap->cmp_strategy))
        {
            pangulu_task_swap(task_index_heap, son, parent);
        }
        else
        {
            break;
        }
        son = parent;
        parent = (son - 1) / 2;
    }
    heap->length++;
    pthread_mutex_unlock(heap->heap_bsem->mutex);
}

char pangulu_task_queue_empty(pangulu_task_queue_t *heap)
{
    return !(heap->length);
}

pangulu_task_t pangulu_task_queue_delete(pangulu_task_queue_t *heap)
{
    pthread_mutex_lock(heap->heap_bsem->mutex);
    if (pangulu_task_queue_empty(heap))
    {
        printf(PANGULU_E_HEAP_EMPTY);
        exit(1);
    }
    pangulu_int64_t *task_index_heap = heap->task_index_heap;
    pangulu_task_swap(task_index_heap, heap->length - 1, 0);
    pangulu_task_t *task_storage = heap->task_storage;
    pangulu_int64_t top = 0;
    pangulu_int64_t left = top * 2 + 1;
    while (left < (heap->length - 1))
    {
        if ((left + 1) < (heap->length - 1) && pangulu_task_compare(task_storage, task_index_heap[left + 1], task_index_heap[left], heap->cmp_strategy))
        {
            left = left + 1;
        }
        if (pangulu_task_compare(task_storage, task_index_heap[left], task_index_heap[top], heap->cmp_strategy))
        {
            pangulu_task_swap(task_index_heap, left, top);
            top = left;
            left = 2 * top + 1;
        }
        else
        {
            break;
        }
    }
    heap->length--;
    pangulu_task_t ret = heap->task_storage[task_index_heap[heap->length]];
    pangulu_task_queue_recycle(heap, task_index_heap[heap->length]);
    pthread_mutex_unlock(heap->heap_bsem->mutex);
    return ret;
}

pangulu_task_t pangulu_task_queue_pop(pangulu_task_queue_t *heap)
{
    while (pangulu_task_queue_empty(heap))
    {
        usleep(1);
    }
    pangulu_task_t task = pangulu_task_queue_delete(heap);
    return task;
}