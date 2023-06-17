#ifndef PANGULU_MALLOC_H
#define PANGULU_MALLOC_H

#include "pangulu_utils.h"
#include "pangulu_common.h"

void *pangulu_malloc(int_t size)
{

    CPU_MEMORY += size;
    void *malloc_address = NULL;
    malloc_address = (void *)malloc(size);
    if (malloc_address == NULL)
    {
        printf("error ------------ don't have cpu memory\n");
    }
    return malloc_address;
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
    char *now_malloc_space = (char *)pangulu_malloc(sizeof(int_t) * (NB + 1) + sizeof(idx_int) * nnz + sizeof(calculate_type) * nnz);
    int_t *rowpointer = (int_t *)now_malloc_space;
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
    idx_int *columnindex = (idx_int *)(now_malloc_space + sizeof(int_t) * (NB + 1));
    calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(int_t) * (NB + 1) + sizeof(idx_int) * nnz);
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
    char *now_malloc_space = (char *)pangulu_malloc(sizeof(int_t) * (NB + 1) + sizeof(idx_int) * nnz + sizeof(calculate_type) * nnz);
    int_t *rowpointer = (int_t *)now_malloc_space;
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
    idx_int *columnindex = (idx_int *)(now_malloc_space + sizeof(int_t) * (NB + 1));
    calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(int_t) * (NB + 1) + sizeof(idx_int) * nnz);
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
    char *now_malloc_space = (char *)pangulu_malloc(sizeof(int_t) * (n + 1) + sizeof(idx_int) * nnzA + sizeof(calculate_type) * nnzA);

    int_t *rowpointer = (int_t *)now_malloc_space;
    idx_int *columnindex = (idx_int *)(now_malloc_space + sizeof(int_t) * (n + 1));
    calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(int_t) * (n + 1) + sizeof(idx_int) * nnzA);
    int_t *CSR_to_CSC_index = (int_t *)pangulu_malloc(sizeof(int_t) * nnzA);
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
    int_t *index_rowpointer = (int_t *)pangulu_malloc(sizeof(int_t) * (n + 1));
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
            CSR_to_CSC_index[j] = index;
            index_rowpointer[col]++;
        }
    }
    int_t *bin_rowpointer = (int_t *)pangulu_malloc(sizeof(int_t) * BIN_LENGTH);
    int_t *bin_rowindex = (int_t *)pangulu_malloc(sizeof(int_t) * n);
    for (int_t i = 0; i < BIN_LENGTH; i++)
    {
        bin_rowpointer[i] = 0;
    }
    for (int_t i = 0; i < n; i++)
    {
        int_t row_length = rowpointer[i + 1] - rowpointer[i];
        bin_rowpointer[pangulu_bin_map(row_length)]++;
    }
    int_t *save_bin_rowpointer = (int_t *)pangulu_malloc(sizeof(int_t) * (BIN_LENGTH + 1));
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
    A->CSR_to_CSC_index = CSR_to_CSC_index;
    free(index_rowpointer);
    free(save_bin_rowpointer);

    // add GPU memory
}

void pangulu_Smatrix_add_more_memory_CSR(pangulu_Smatrix *A)
{
    // add CPU moemory

    int_t nnzA = A->nnz;
    int_t n = A->row;

    int_t *rowpointer = (int_t *)pangulu_malloc(sizeof(int_t) * (n + 1));
    idx_int *columnindex = (idx_int *)pangulu_malloc(sizeof(idx_int) * (nnzA));
    calculate_type *value = (calculate_type *)pangulu_malloc(sizeof(calculate_type) * (nnzA));

    int_t *CSC_to_CSR_index = (int_t *)pangulu_malloc(sizeof(int_t) * nnzA);
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
    int_t *index_rowpointer = (int_t *)pangulu_malloc(sizeof(int_t) * (n + 1));
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
    int_t *bin_rowpointer = (int_t *)pangulu_malloc(sizeof(int_t) * BIN_LENGTH);
    int_t *bin_rowindex = (int_t *)pangulu_malloc(sizeof(int_t) * n);
    for (int_t i = 0; i < BIN_LENGTH; i++)
    {
        bin_rowpointer[i] = 0;
    }
    for (int_t i = 0; i < n; i++)
    {
        int_t row_length = A->columnpointer[i + 1] - A->columnpointer[i];
        bin_rowpointer[pangulu_bin_map(row_length)]++;
    }
    int_t *save_bin_rowpointer = (int_t *)pangulu_malloc(sizeof(int_t) * (BIN_LENGTH + 1));
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
    free(index_rowpointer);
    free(save_bin_rowpointer);

    // add GPU memory
}

void pangulu_Smatrix_add_CSC(pangulu_Smatrix *A)
{
    // add CSC moemory
    int_t nnzA = A->rowpointer[A->row];
    int_t n = A->row;
    char *now_malloc_space = (char *)pangulu_malloc(sizeof(int_t) * (n + 1) + sizeof(idx_int) * nnzA + sizeof(calculate_type) * nnzA);

    int_t *rowpointer = (int_t *)now_malloc_space;
    idx_int *columnindex = (idx_int *)(now_malloc_space + sizeof(int_t) * (n + 1));
    calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(int_t) * (n + 1) + sizeof(idx_int) * nnzA);
    int_t *CSC_to_CSR_index = (int_t *)pangulu_malloc(sizeof(int_t) * nnzA);
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
    int_t *index_rowpointer = (int_t *)pangulu_malloc(sizeof(int_t) * (n + 1));
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
    free(index_rowpointer);
}

void pangulu_malloc_pangulu_Smatrix_CSC(pangulu_Smatrix *S,
                                        int_t NB, int_t *save_columnpointer)
{

    int_t nnzA = save_columnpointer[NB];
    char *now_malloc_space = (char *)pangulu_malloc((sizeof(int_t) * (NB + 1)) + (sizeof(idx_int) * nnzA) + (sizeof(calculate_type) * nnzA));
    int_t *columnpointer = (int_t *)now_malloc_space;
    idx_int *rowindex = (idx_int *)(now_malloc_space + sizeof(int_t) * (NB + 1));
    calculate_type *value_csc = (calculate_type *)(now_malloc_space + (sizeof(int_t) * (NB + 1) + sizeof(idx_int) * nnzA));

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
    // printf(" nnz is %ld\n",S->nnz);
    B->row = S->row;
    B->column = S->column;
    B->nnz = S->nnz;
    calculate_type *value_csc = (calculate_type *)pangulu_malloc(sizeof(calculate_type) * S->nnz);
    B->value_CSC = value_csc;
}

void pangulu_malloc_pangulu_Smatrix_CSR_value(pangulu_Smatrix *S,
                                              pangulu_Smatrix *B)
{
    // printf(" nnz is %ld\n",S->nnz);
    B->row = S->row;
    B->column = S->column;
    B->nnz = S->nnz;
    calculate_type *value = (calculate_type *)pangulu_malloc(sizeof(calculate_type) * S->nnz);
    B->value = value;
}

void pangulu_malloc_pangulu_Smatrix_nnz_CSC(pangulu_Smatrix *S,
                                            int_t NB, int_t nnz)
{
    char *now_malloc_space = (char *)pangulu_malloc(sizeof(int_t) * (NB + 1) + sizeof(idx_int) * nnz + sizeof(calculate_type) * nnz);
    int_t *rowpointer = (int_t *)now_malloc_space;
    S->row = NB;
    S->column = NB;
    idx_int *columnindex = (idx_int *)(now_malloc_space + sizeof(int_t) * (NB + 1));
    calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(int_t) * (NB + 1) + sizeof(idx_int) * nnz);
    S->columnpointer = rowpointer;
    S->rowindex = columnindex;
    S->value_CSC = value;
    S->nnz = nnz;
}

void pangulu_malloc_pangulu_Smatrix_nnz_CSR(pangulu_Smatrix *S,
                                            int_t NB, int_t nnz)
{
    char *now_malloc_space = (char *)pangulu_malloc(sizeof(int_t) * (NB + 1) + sizeof(idx_int) * nnz + sizeof(calculate_type) * nnz);
    int_t *rowpointer = (int_t *)now_malloc_space;
    S->row = NB;
    S->column = NB;
    idx_int *columnindex = (idx_int *)(now_malloc_space + sizeof(int_t) * (NB + 1));
    calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(int_t) * (NB + 1) + sizeof(idx_int) * nnz);
    S->rowpointer = rowpointer;
    S->columnindex = columnindex;
    S->value = value;
    S->nnz = nnz;
}

void pangulu_malloc_pangulu_Smatrix_value_CSC(pangulu_Smatrix *S, int_t nnz)
{
    calculate_type *value = (calculate_type *)pangulu_malloc(sizeof(calculate_type) * nnz);
    S->value_CSC = value;
    S->nnz = nnz;
}

void pangulu_malloc_pangulu_Smatrix_value_CSR(pangulu_Smatrix *S, int_t nnz)
{
    calculate_type *value = (calculate_type *)pangulu_malloc(sizeof(calculate_type) * nnz);
    S->value = value;
    S->nnz = nnz;
}

void pangulu_Smatrix_add_memory_U(pangulu_Smatrix *U)
{
    int_32t *nnzU = (int_32t *)pangulu_malloc(sizeof(int_32t) * (U->row));
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
    int_t *level_size = (int_t *)malloc(sizeof(int_t) * (n + 1));
    int_t *level_idx = (int_t *)malloc(sizeof(int_t) * (n + 1));
    A->level_size = level_size;
    A->level_idx = level_idx;
    A->num_lev = 0;
}

#endif

#endif