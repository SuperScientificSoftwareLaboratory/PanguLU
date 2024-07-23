#ifndef PANGULU_MALLOC_H
#define PANGULU_MALLOC_H

#include "pangulu_common.h"
#include "pangulu_utils.h"

#include <map>
#include <string>
std::map<std::string, int_t> malloc_map;
std::map<void*, int_t> malloc_size_records;
char msg[100];

void *pangulu_malloc(const char* file, int_t line, int_t size)
{
    if(size==0){
        return NULL;
    }
    
    sprintf(msg, "%s:%ld", file, line);
    // malloc_map[msg] += size;
    CPU_MEMORY += size;
    if(CPU_MEMORY > CPU_PEAK_MEMORY){
        CPU_PEAK_MEMORY = CPU_MEMORY;
    }
    void *malloc_address = NULL;
    malloc_address = (void *)malloc(size);
    // malloc_size_records[malloc_address] = size;
    if (malloc_address == NULL)
    {
        printf(PANGULU_E_CPU_MEM);
    }
    return malloc_address;
}

void *pangulu_realloc(const char* file, int_t line, void* oldptr, int_t size)
{
    sprintf(msg, "%s:%ld", file, line);
    int_t old_size = malloc_map[msg];
    // malloc_map[msg] = size;
    CPU_MEMORY -= old_size;
    CPU_MEMORY += size;
    if(CPU_MEMORY > CPU_PEAK_MEMORY){
        CPU_PEAK_MEMORY = CPU_MEMORY;
    }
    void *realloc_address = NULL;
    realloc_address = (void *)realloc(oldptr, size);
    // malloc_size_records.erase(oldptr);
    // malloc_size_records[realloc_address] = size;
    if (realloc_address == NULL)
    {
        printf(PANGULU_E_CPU_MEM);
    }
    return realloc_address;
}

void pangulu_free(const char* file, int_t line, void* ptr){
    if(ptr==NULL){
        return;
    }
    CPU_MEMORY -= malloc_size_records[ptr];
    // malloc_size_records.erase(ptr);
    free(ptr);
}

int_t pangulu_bin_map(int_t nnz)
{
    int_t log_nnz = ceil(log2((double)nnz));
    if (log_nnz >= 14)
    {
        return 11;
    }
    else if (log_nnz > 5)
    {
        return log_nnz - 2;
    }
    else if (nnz > 1)
    {
        return 3;
    }
    else if (nnz == 1)
    {
        return 2;
    }
    else
    {
        return 1;
    }
}

void pangulu_get_pangulu_Smatrix_to_U(pangulu_Smatrix *S,
                                      pangulu_Smatrix *U,
                                      int_t NB)
{
    int_t nnz = 0;
    for (int_t i = 0; i < NB; i++)
    {
        for (int_t j = S->rowpointer[i]; j < S->rowpointer[i + 1]; j++)
        {
            int_t col = S->columnindex[j];
            if (i <= col)
            {
                nnz++;
            }
        }
    }
    char *now_malloc_space = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (NB + 1) + sizeof(pangulu_inblock_idx) * nnz + sizeof(calculate_type) * nnz);
    pangulu_inblock_ptr *rowpointer = (pangulu_inblock_ptr *)now_malloc_space;
    rowpointer[0] = 0;
    for (int_t i = 0; i < NB; i++)
    {
        rowpointer[i + 1] = rowpointer[i];
        for (int_t j = S->rowpointer[i]; j < S->rowpointer[i + 1]; j++)
        {
            int_t col = S->columnindex[j];
            if (i <= col)
            {
                rowpointer[i + 1]++;
            }
        }
    }
    pangulu_inblock_idx *columnindex = (pangulu_inblock_idx *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (NB + 1));
    calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (NB + 1) + sizeof(pangulu_inblock_idx) * nnz);
    for (int_t i = 0; i < rowpointer[NB]; i++)
    {
        value[i] = 0.0;
    }
    int_t nzz = 0;
    for (int_t i = 0; i < NB; i++)
    {
        for (int_t j = S->rowpointer[i]; j < S->rowpointer[i + 1]; j++)
        {
            int_t col = S->columnindex[j];
            if (i <= col)
            {
                columnindex[nzz] = col;
                // value[nzz] = S->value[j];
                nzz++;
            }
        }
    }
    U->nnz = rowpointer[NB];
    U->rowpointer = rowpointer;
    U->columnindex = columnindex;
    U->value = value;
    U->row = NB;
    U->column = NB;
    return;
}

void pangulu_get_pangulu_Smatrix_to_L(pangulu_Smatrix *S,
                                      pangulu_Smatrix *L,
                                      int_t NB)
{
    int_t nnz = 0;
    for (int_t i = 0; i < NB; i++)
    {
        for (int_t j = S->rowpointer[i]; j < S->rowpointer[i + 1]; j++)
        {
            int_t col = S->columnindex[j];
            if (i >= col)
            {
                nnz++;
            }
        }
    }
    char *now_malloc_space = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (NB + 1) + sizeof(pangulu_inblock_idx) * nnz + sizeof(calculate_type) * nnz);
    pangulu_inblock_ptr *rowpointer = (pangulu_inblock_ptr *)now_malloc_space;
    rowpointer[0] = 0;
    for (int_t i = 0; i < NB; i++)
    {
        rowpointer[i + 1] = rowpointer[i];
        for (int_t j = S->rowpointer[i]; j < S->rowpointer[i + 1]; j++)
        {
            int_t col = S->columnindex[j];
            if (i >= col)
            {
                rowpointer[i + 1]++;
            }
        }
    }
    pangulu_inblock_idx *columnindex = (pangulu_inblock_idx *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (NB + 1));
    calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (NB + 1) + sizeof(pangulu_inblock_idx) * nnz);
    for (int_t i = 0; i < rowpointer[NB]; i++)
    {
        value[i] = 0.0;
    }
    int_t nzz = 0;
    for (int_t i = 0; i < NB; i++)
    {
        for (int_t j = S->rowpointer[i]; j < S->rowpointer[i + 1]; j++)
        {
            int_t col = S->columnindex[j];
            if (i >= col)
            {
                columnindex[nzz] = col;
                // value[nzz] = S->value[j];
                nzz++;
            }
        }
    }
    L->nnz = rowpointer[NB];
    L->rowpointer = rowpointer;
    L->columnindex = columnindex;
    L->value = value;
    L->row = NB;
    L->column = NB;
    return;
}

void pangulu_Smatrix_add_more_memory(pangulu_Smatrix *A)
{
    // add CPU moemory
    int_t nnzA = A->rowpointer[A->row];
    int_t n = A->row;
    char *now_malloc_space = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (n + 1) + sizeof(pangulu_inblock_idx) * nnzA + sizeof(calculate_type) * nnzA);

    pangulu_inblock_ptr *rowpointer = (pangulu_inblock_ptr *)now_malloc_space;
    pangulu_inblock_idx *columnindex = (pangulu_inblock_idx *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (n + 1));
    calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (n + 1) + sizeof(pangulu_inblock_idx) * nnzA);
    int_t *CSC_to_CSR_index = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * nnzA);
    for (int_t i = 0; i < nnzA; i++)
    {
        value[i] = (calculate_type)0.0;
    }

    for (int_t i = 0; i < (n + 1); i++)
    {
        rowpointer[i] = 0;
    }
    for (int_t i = 0; i < nnzA; i++)
    {
        rowpointer[A->columnindex[i] + 1]++;
    }
    for (int_t i = 0; i < n; i++)
    {
        rowpointer[i + 1] += rowpointer[i];
    }
    pangulu_inblock_ptr *index_rowpointer = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (n + 1));
    for (int_t i = 0; i < n; i++)
    {
        index_rowpointer[i] = rowpointer[i];
    }
    for (int_t i = 0; i < n; i++)
    {
        for (int_t j = A->rowpointer[i]; j < A->rowpointer[i + 1]; j++)
        {

            int_t col = A->columnindex[j];
            pangulu_inblock_ptr index = index_rowpointer[col];
            columnindex[index] = i;
            value[index] = A->value[j];
            CSC_to_CSR_index[index] = j;
            index_rowpointer[col]++;
        }
    }
    pangulu_inblock_ptr *bin_rowpointer = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * BIN_LENGTH);
    pangulu_inblock_idx *bin_rowindex = (pangulu_inblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx) * n);
    for (int_t i = 0; i < BIN_LENGTH; i++)
    {
        bin_rowpointer[i] = 0;
    }
    for (int_t i = 0; i < n; i++)
    {
        int_t row_length = rowpointer[i + 1] - rowpointer[i];
        bin_rowpointer[pangulu_bin_map(row_length)]++;
    }
    pangulu_inblock_ptr *save_bin_rowpointer = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (BIN_LENGTH + 1));
    for (int_t i = 0; i < BIN_LENGTH - 1; i++)
    {
        bin_rowpointer[i + 1] += bin_rowpointer[i];
    }
    for (int_t i = 0; i < BIN_LENGTH; i++)
    {
        save_bin_rowpointer[i + 1] = bin_rowpointer[i];
    }
    for (int_t i = 0; i < n; i++)
    {
        int_t row_length = rowpointer[i + 1] - rowpointer[i];
        int_t index = pangulu_bin_map(row_length);
        bin_rowindex[save_bin_rowpointer[index]] = i;
        save_bin_rowpointer[index]++;
    }
    A->bin_rowpointer = bin_rowpointer;
    A->bin_rowindex = bin_rowindex;
    A->columnpointer = rowpointer;
    A->rowindex = columnindex;
    A->value_CSC = value;
    A->CSC_to_CSR_index = CSC_to_CSR_index;
    pangulu_free(__FILE__, __LINE__, index_rowpointer);
    pangulu_free(__FILE__, __LINE__, save_bin_rowpointer);

    // add GPU memory
}

void pangulu_Smatrix_add_more_memory_CSR(pangulu_Smatrix *A)
{
    // add CPU moemory

    int_t nnzA = A->nnz;
    int_t n = A->row;

    pangulu_inblock_ptr *rowpointer = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (n + 1));
    pangulu_inblock_idx *columnindex = (pangulu_inblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx) * (nnzA));
    calculate_type *value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * (nnzA));

    int_t *CSC_to_CSR_index = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * nnzA);
    for (int_t i = 0; i < nnzA; i++)
    {
        value[i] = 0.0;
    }

    for (int_t i = 0; i < (n + 1); i++)
    {
        rowpointer[i] = 0;
    }
    for (int_t i = 0; i < nnzA; i++)
    {
        rowpointer[A->rowindex[i] + 1]++;
    }
    for (int_t i = 0; i < n; i++)
    {
        rowpointer[i + 1] += rowpointer[i];
    }
    pangulu_inblock_ptr *index_rowpointer = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (n + 1));
    for (int_t i = 0; i < n; i++)
    {
        index_rowpointer[i] = rowpointer[i];
    }
    for (int_t i = 0; i < n; i++)
    {
        for (int_t j = A->columnpointer[i]; j < A->columnpointer[i + 1]; j++)
        {

            int_t col = A->rowindex[j];
            int_t index = index_rowpointer[col];
            columnindex[index] = i;
            value[index] = A->value_CSC[j];
            CSC_to_CSR_index[j] = index;
            index_rowpointer[col]++;
        }
    }
    pangulu_inblock_ptr *bin_rowpointer = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * BIN_LENGTH);
    pangulu_inblock_idx *bin_rowindex = (pangulu_inblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx) * n);
    for (int_t i = 0; i < BIN_LENGTH; i++)
    {
        bin_rowpointer[i] = 0;
    }
    for (int_t i = 0; i < n; i++)
    {
        int_t row_length = A->columnpointer[i + 1] - A->columnpointer[i];
        bin_rowpointer[pangulu_bin_map(row_length)]++;
    }
    pangulu_inblock_ptr *save_bin_rowpointer = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (BIN_LENGTH + 1));
    for (int_t i = 0; i < BIN_LENGTH - 1; i++)
    {
        bin_rowpointer[i + 1] += bin_rowpointer[i];
    }
    for (int_t i = 0; i < BIN_LENGTH; i++)
    {
        save_bin_rowpointer[i + 1] = bin_rowpointer[i];
    }
    for (int_t i = 0; i < n; i++)
    {
        int_t row_length = A->columnpointer[i + 1] - A->columnpointer[i];
        int_t index = pangulu_bin_map(row_length);
        bin_rowindex[save_bin_rowpointer[index]] = i;
        save_bin_rowpointer[index]++;
    }
    A->bin_rowpointer = bin_rowpointer;
    A->bin_rowindex = bin_rowindex;
    A->rowpointer = rowpointer;
    A->columnindex = columnindex;
    A->value = value;
    A->CSC_to_CSR_index = CSC_to_CSR_index;
    pangulu_free(__FILE__, __LINE__, index_rowpointer);
    pangulu_free(__FILE__, __LINE__, save_bin_rowpointer);

    // add GPU memory
}

void pangulu_Smatrix_add_CSC(pangulu_Smatrix *A)
{
    // add CSC moemory
    int_t nnzA = A->rowpointer[A->row];
    int_t n = A->row;
    char *now_malloc_space = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (n + 1) + sizeof(pangulu_inblock_idx) * nnzA + sizeof(calculate_type) * nnzA);

    pangulu_inblock_ptr *rowpointer = (pangulu_inblock_ptr *)now_malloc_space;
    pangulu_inblock_idx *columnindex = (pangulu_inblock_idx *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (n + 1));
    calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (n + 1) + sizeof(pangulu_inblock_idx) * nnzA);
    int_t *CSC_to_CSR_index = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * nnzA);
    for (int_t i = 0; i < nnzA; i++)
    {
        value[i] = 0.0;
    }

    for (int_t i = 0; i < (n + 1); i++)
    {
        rowpointer[i] = 0;
    }
    for (int_t i = 0; i < nnzA; i++)
    {
        rowpointer[A->columnindex[i] + 1]++;
    }
    for (int_t i = 0; i < n; i++)
    {
        rowpointer[i + 1] += rowpointer[i];
    }
    int_t *index_rowpointer = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (n + 1));
    for (int_t i = 0; i < n; i++)
    {
        index_rowpointer[i] = rowpointer[i];
    }
    for (int_t i = 0; i < n; i++)
    {
        for (int_t j = A->rowpointer[i]; j < A->rowpointer[i + 1]; j++)
        {

            int_t col = A->columnindex[j];
            int_t index = index_rowpointer[col];
            columnindex[index] = i;
            value[index] = A->value[j];
            CSC_to_CSR_index[index] = j;
            index_rowpointer[col]++;
        }
    }
    A->columnpointer = rowpointer;
    A->rowindex = columnindex;
    A->value_CSC = value;
    A->CSC_to_CSR_index = CSC_to_CSR_index;
    pangulu_free(__FILE__, __LINE__, index_rowpointer);
}

void pangulu_origin_Smatrix_add_CSC(pangulu_origin_Smatrix *A)
{
    // add CSC moemory
    int_t nnzA = A->rowpointer[A->row];
    int_t n = A->row;
    char *now_malloc_space = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (n + 1) + sizeof(idx_int) * nnzA + sizeof(calculate_type) * nnzA);

    int_t *rowpointer = (int_t *)now_malloc_space;
    idx_int *columnindex = (idx_int *)(now_malloc_space + sizeof(int_t) * (n + 1));
    calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(int_t) * (n + 1) + sizeof(idx_int) * nnzA);
    int_t *CSC_to_CSR_index = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * nnzA);
    for (int_t i = 0; i < nnzA; i++)
    {
        value[i] = 0.0;
    }

    for (int_t i = 0; i < (n + 1); i++)
    {
        rowpointer[i] = 0;
    }
    for (int_t i = 0; i < nnzA; i++)
    {
        rowpointer[A->columnindex[i] + 1]++;
    }
    for (int_t i = 0; i < n; i++)
    {
        rowpointer[i + 1] += rowpointer[i];
    }
    int_t *index_rowpointer = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (n + 1));
    for (int_t i = 0; i < n; i++)
    {
        index_rowpointer[i] = rowpointer[i];
    }
    for (int_t i = 0; i < n; i++)
    {
        for (int_t j = A->rowpointer[i]; j < A->rowpointer[i + 1]; j++)
        {

            int_t col = A->columnindex[j];
            int_t index = index_rowpointer[col];
            columnindex[index] = i;
            value[index] = A->value[j];
            CSC_to_CSR_index[index] = j;
            index_rowpointer[col]++;
        }
    }
    A->columnpointer = rowpointer;
    A->rowindex = columnindex;
    A->value_CSC = value;
    A->CSC_to_CSR_index = CSC_to_CSR_index;
    pangulu_free(__FILE__, __LINE__, index_rowpointer);
}

void pangulu_malloc_pangulu_Smatrix_CSC(pangulu_Smatrix *S,
                                        int_t NB, int_t *save_columnpointer)
{

    int_t nnzA = save_columnpointer[NB];
    char *now_malloc_space = (char *)pangulu_malloc(__FILE__, __LINE__, (sizeof(pangulu_inblock_ptr) * (NB + 1)) + (sizeof(pangulu_inblock_idx) * nnzA) + (sizeof(calculate_type) * nnzA));
    pangulu_inblock_ptr *columnpointer = (pangulu_inblock_ptr *)now_malloc_space;
    pangulu_inblock_idx *rowindex = (pangulu_inblock_idx *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (NB + 1));
    calculate_type *value_csc = (calculate_type *)(now_malloc_space + (sizeof(pangulu_inblock_ptr) * (NB + 1) + sizeof(pangulu_inblock_idx) * nnzA));

    S->row = NB;
    S->column = NB;
    for (int_t i = 0; i <= NB; i++)
    {
        columnpointer[i] = save_columnpointer[i];
    }
    for (int_t i = 0; i < nnzA; i++)
    {
        rowindex[i] = 0;
    }

    for (int_t i = 0; i < nnzA; i++)
    {
        value_csc[i] = 0.0;
    }
    S->nnz = nnzA;
    S->columnpointer = columnpointer;
    S->rowindex = rowindex;
    S->value_CSC = value_csc;
}

void pangulu_malloc_pangulu_Smatrix_CSC_value(pangulu_Smatrix *S,
                                              pangulu_Smatrix *B)
{
    B->row = S->row;
    B->column = S->column;
    B->nnz = S->nnz;
    calculate_type *value_csc = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * S->nnz);
    B->value_CSC = value_csc;
}

void pangulu_malloc_pangulu_Smatrix_CSR_value(pangulu_Smatrix *S,
                                              pangulu_Smatrix *B)
{
    B->row = S->row;
    B->column = S->column;
    B->nnz = S->nnz;
    calculate_type *value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * S->nnz);
    B->value = value;
}

void pangulu_malloc_pangulu_Smatrix_nnz_CSC(pangulu_Smatrix *S,
                                            int_t NB, int_t nnz)
{
    char *now_malloc_space = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (NB + 1) + sizeof(pangulu_inblock_idx) * nnz + sizeof(calculate_type) * nnz);
    pangulu_inblock_ptr *rowpointer = (pangulu_inblock_ptr *)now_malloc_space;
    S->row = NB;
    S->column = NB;
    pangulu_inblock_idx *columnindex = (pangulu_inblock_idx *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (NB + 1));
    calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (NB + 1) + sizeof(pangulu_inblock_idx) * nnz);
    S->columnpointer = rowpointer;
    S->rowindex = columnindex;
    S->value_CSC = value;
    S->nnz = nnz;
}

void pangulu_malloc_pangulu_Smatrix_nnz_CSR(pangulu_Smatrix *S,
                                            int_t NB, int_t nnz)
{
    char *now_malloc_space = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (NB + 1) + sizeof(pangulu_inblock_idx) * nnz + sizeof(calculate_type) * nnz);
    pangulu_inblock_ptr *rowpointer = (pangulu_inblock_ptr *)now_malloc_space;
    S->row = NB;
    S->column = NB;
    pangulu_inblock_idx *columnindex = (pangulu_inblock_idx *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (NB + 1));
    calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (NB + 1) + sizeof(pangulu_inblock_idx) * nnz);
    S->rowpointer = rowpointer;
    S->columnindex = columnindex;
    S->value = value;
    S->nnz = nnz;
}

void pangulu_malloc_pangulu_Smatrix_value_CSC(pangulu_Smatrix *S, int_t nnz)
{
    calculate_type *value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz);
    S->value_CSC = value;
    S->nnz = nnz;
}

void pangulu_malloc_pangulu_Smatrix_value_CSR(pangulu_Smatrix *S, int_t nnz)
{
    calculate_type *value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz);
    S->value = value;
    S->nnz = nnz;
}

void pangulu_Smatrix_add_memory_U(pangulu_Smatrix *U)
{
    int_32t *nnzU = (int_32t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_32t) * (U->row));
    for (int_t j = 0; j < U->row; j++)
    {
        nnzU[j] = U->columnpointer[j + 1] - U->columnpointer[j] - 1;
    }
    U->nnzU = nnzU;
}

#ifndef GPU_OPEN

void pangulu_malloc_Smatrix_level(pangulu_Smatrix *A)
{
    int_t n = A->row;
    int_t *level_size = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (n + 1));
    int_t *level_idx = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (n + 1));
    A->level_size = level_size;
    A->level_idx = level_idx;
    A->num_lev = 0;
}

#endif

#endif