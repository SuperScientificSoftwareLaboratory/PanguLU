#include "pangulu_common.h"

void *pangulu_malloc(const char* file, pangulu_int64_t line, pangulu_int64_t size)
{
    void *malloc_address = NULL;
    malloc_address = (void *)malloc(size);
    if (size != 0 && malloc_address == NULL)
    {
        printf(PANGULU_E_CPU_MEM);
        exit(1);
    }
    return malloc_address;
}

void *pangulu_realloc(const char* file, pangulu_int64_t line, void* oldptr, pangulu_int64_t size)
{
    void *realloc_address = NULL;
    realloc_address = (void *)realloc(oldptr, size);
    if (size != 0 && realloc_address == NULL)
    {
        printf(PANGULU_E_CPU_MEM);
        exit(1);
    }
    return realloc_address;
}

void pangulu_free(const char* file, pangulu_int64_t line, void* ptr){
    if(ptr==NULL){
        return;
    }
    free(ptr);
}

void pangulu_origin_smatrix_add_csc(pangulu_origin_smatrix *a)
{
    pangulu_exblock_ptr nnzA = a->rowpointer[a->row];
    pangulu_exblock_idx n = a->row;
    char *now_malloc_space = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1) + sizeof(pangulu_exblock_idx) * nnzA + sizeof(calculate_type) * nnzA);

    pangulu_exblock_ptr *rowpointer = (pangulu_exblock_ptr *)now_malloc_space;
    pangulu_exblock_idx *columnindex = (pangulu_exblock_idx *)(now_malloc_space + sizeof(pangulu_exblock_ptr) * (n + 1));
    calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(pangulu_exblock_ptr) * (n + 1) + sizeof(pangulu_exblock_idx) * nnzA);
    pangulu_exblock_ptr *csc_to_csr_index = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * nnzA);
    for (pangulu_exblock_ptr i = 0; i < nnzA; i++)
    {
        value[i] = 0.0;
    }

    for (pangulu_exblock_idx i = 0; i < (n + 1); i++)
    {
        rowpointer[i] = 0;
    }
    for (pangulu_exblock_ptr i = 0; i < nnzA; i++)
    {
        rowpointer[a->columnindex[i] + 1]++;
    }
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        rowpointer[i + 1] += rowpointer[i];
    }
    pangulu_exblock_ptr *index_rowpointer = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1));
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        index_rowpointer[i] = rowpointer[i];
    }
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        for (pangulu_exblock_ptr j = a->rowpointer[i]; j < a->rowpointer[i + 1]; j++)
        {

            pangulu_exblock_idx col = a->columnindex[j];
            pangulu_exblock_ptr index = index_rowpointer[col];
            columnindex[index] = i;
            value[index] = a->value[j];
            csc_to_csr_index[index] = j;
            index_rowpointer[col]++;
        }
    }
    a->columnpointer = rowpointer;
    a->rowindex = columnindex;
    a->value_csc = value;
    a->csc_to_csr_index = csc_to_csr_index;
    pangulu_free(__FILE__, __LINE__, index_rowpointer);
}