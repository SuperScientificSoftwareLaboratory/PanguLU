#include "pangulu_common.h"
void pangulu_init_heap_select(pangulu_int64_t select)
{
    heap_select = select;
}

void pangulu_init_pangulu_heap(pangulu_heap *heap, pangulu_int64_t max_length)
{
    compare_struct *compare_queue = (compare_struct *)pangulu_malloc(__FILE__, __LINE__, sizeof(compare_struct) * max_length);
    pangulu_int64_t *heap_queue = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * max_length);
    heap->comapre_queue = compare_queue;
    heap->heap_queue = heap_queue;
    heap->max_length = max_length;
    heap->length = 0;
    heap->nnz_flag = 0;
#ifdef OVERLAP
    heap->heap_bsem = NULL;
#endif
}

pangulu_heap *pangulu_destory_pangulu_heap(pangulu_heap *heap)
{
    if (heap != NULL)
    {
        pangulu_free(__FILE__, __LINE__, heap->comapre_queue);
        pangulu_free(__FILE__, __LINE__, heap->heap_queue);
        heap->length = 0;
        heap->nnz_flag = 0;
        heap->max_length = 0;
    }
    pangulu_free(__FILE__, __LINE__, heap);
    return NULL;
}

void pangulu_zero_pangulu_heap(pangulu_heap *heap)
{
    heap->length = 0;
    heap->nnz_flag = 0;
}

pangulu_int64_t pangulu_compare(compare_struct *compare_queue, pangulu_int64_t a, pangulu_int64_t b)
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
        pangulu_exit(1);
    }
}

void pangulu_swap(pangulu_int64_t *heap_queue, pangulu_int64_t a, pangulu_int64_t b)
{
    pangulu_int64_t temp = heap_queue[a];
    heap_queue[a] = heap_queue[b];
    heap_queue[b] = temp;
}

void pangulu_heap_insert(pangulu_heap *heap, pangulu_int64_t row, pangulu_int64_t col, pangulu_int64_t task_level, pangulu_int64_t kernel_id, pangulu_int64_t compare_flag)
{

    compare_struct *compare_queue = heap->comapre_queue;
    pangulu_int64_t *heap_queue = heap->heap_queue;
    pangulu_int64_t length = heap->length;
    pangulu_int64_t nnz_flag = heap->nnz_flag;

    if (rank == -1)
    {
        printf(PANGULU_I_TASK_INFO);
    }

    if ((nnz_flag) >= heap->max_length)
    {
        printf(PANGULU_E_HEAP_FULL);
        pangulu_exit(1);
    }
    compare_queue[nnz_flag].row = row;
    compare_queue[nnz_flag].col = col;
    compare_queue[nnz_flag].task_level = task_level;
    compare_queue[nnz_flag].kernel_id = kernel_id;
    compare_queue[nnz_flag].compare_flag = compare_flag;
    heap_queue[length] = nnz_flag;
    (heap->nnz_flag)++;
    pangulu_int64_t now = length;
    pangulu_int64_t before = (now - 1) / 2;
    while (now != 0 && before >= 0)
    {
        if (pangulu_compare(compare_queue, heap_queue[now], heap_queue[before]))
        {
            pangulu_swap(heap_queue, now, before);
        }
        else
        {
            break;
        }
        now = before;
        before = (now - 1) / 2;
    }
    heap->length = length + 1;
}

pangulu_int64_t heap_empty(pangulu_heap *heap)
{
    return !(heap->length);
}

void pangulu_heap_adjust(pangulu_heap *heap, pangulu_int64_t top, pangulu_int64_t n)
{
    compare_struct *compare_queue = heap->comapre_queue;
    pangulu_int64_t *heap_queue = heap->heap_queue;
    pangulu_int64_t left = top * 2 + 1;

    while (left < n)
    {
        if ((left + 1) < n && pangulu_compare(compare_queue, heap_queue[left + 1], heap_queue[left]))
        {
            left = left + 1;
        }
        if (pangulu_compare(compare_queue, heap_queue[left], heap_queue[top]))
        {
            pangulu_swap(heap_queue, left, top);
            top = left;
            left = 2 * top + 1;
        }
        else
        {
            break;
        }
    }
}

pangulu_int64_t pangulu_heap_delete(pangulu_heap *heap)
{
    if (heap_empty(heap))
    {
        printf(PANGULU_E_HEAP_EMPTY);
        pangulu_exit(1);
    }
    pangulu_int64_t length = heap->length;
    pangulu_int64_t *heap_queue = heap->heap_queue;
    pangulu_swap(heap_queue, length - 1, 0);
    pangulu_heap_adjust(heap, 0, length - 1);
    heap->length = length - 1;
    return heap_queue[length - 1];
}

void pangulu_display_heap(pangulu_heap *heap)
{
    printf(PANGULU_I_HEAP_LEN);
    for (pangulu_int64_t i = 0; i < heap->length; i++)
    {
        printf(FMT_PANGULU_INT64_T " ", heap->heap_queue[i]);
    }
    printf("\n");
    for (pangulu_int64_t i = 0; i < heap->length; i++)
    {
        pangulu_int64_t now = heap->heap_queue[i];
        printf("row is " FMT_PANGULU_EXBLOCK_IDX 
            " col is " FMT_PANGULU_EXBLOCK_IDX 
            " level is " FMT_PANGULU_EXBLOCK_IDX
            " compare_flag is " FMT_PANGULU_INT64_T
            " do the kernel " FMT_PANGULU_INT16_T "\n",
            heap->comapre_queue[now].row,
            heap->comapre_queue[now].col,
            heap->comapre_queue[now].task_level,
            heap->comapre_queue[now].compare_flag,
            heap->comapre_queue[now].kernel_id);
    }
}