#ifndef PANGULU_SPMV_FP64_H
#define PANGULU_SPMV_FP64_H

#include "pangulu_common.h"
#include "pangulu_utils.h"

void pangulu_spmv_cpu_choumi(pangulu_Smatrix *S, pangulu_vector *X, pangulu_vector *B)
{
    calculate_type *value = S->value;
    calculate_type *b = B->value;
    calculate_type *x = X->value;
    int_t N = S->column;
    int_t M = S->row;
    for (int_t i = 0; i < M; i++)
        b[i] = 0.0;
    for (int_t i = 0; i < M; i++)
    {
        for (int_t j = 0; j < N; j++)
        {
            b[i] += value[i * N + j] * x[j];
        }
    }
}

void pangulu_spmv_cpu_xishu(pangulu_Smatrix *S, pangulu_vector *X, pangulu_vector *B, int_t vector_number)
{
    int_t m = S->row;
    int_t *csrRowPtr_tmp = S->rowpointer;
    int_32t *csrColIdx_tmp = S->columnindex;
    calculate_type *csrVal_tmp = S->value;
    for (int_t vector_index = 0; vector_index < vector_number; vector_index++)
    {
        calculate_type *x = X->value + vector_index * m;
        calculate_type *y = B->value + vector_index * m;
        for (int_t i = 0; i < m; i++)
        {
            for (int_t j = csrRowPtr_tmp[i]; j < csrRowPtr_tmp[i + 1]; j++)
            {
                y[i] += csrVal_tmp[j] * x[csrColIdx_tmp[j]];
            }
        }
    }
}

void pangulu_spmv_cpu_xishu_csc(pangulu_Smatrix *S, pangulu_vector *X, pangulu_vector *B, int_t vector_number)
{
    int_t m = S->row;
    int_t *csccolumnPtr_tmp = S->columnpointer;
    int_32t *cscrowIdx_tmp = S->rowindex;
    calculate_type *cscVal_tmp = S->value_CSC;
    for (int_t vector_index = 0; vector_index < vector_number; vector_index++)
    {
        calculate_type *x = X->value + vector_index * m;
        calculate_type *y = B->value + vector_index * m;
        for (int_t i = 0; i < m; i++)
        {
            for (int_t j = csccolumnPtr_tmp[i]; j < csccolumnPtr_tmp[i + 1]; j++)
            {
                int_32t row = cscrowIdx_tmp[j];
                y[row] += cscVal_tmp[j] * x[i];
            }
        }
    }
}

void pangulu_vector_add_cpu(pangulu_vector *B, pangulu_vector *X)
{

    calculate_type *x = X->value;
    calculate_type *b = B->value;
    int_t n = X->row;
    for (int_t i = 0; i < n; i++)
    {
        b[i] += x[i];
    }
}

void pangulu_vector_sub_cpu(pangulu_vector *B, pangulu_vector *X)
{

    calculate_type *x = X->value;
    calculate_type *b = B->value;
    int_t n = X->row;
    for (int_t i = 0; i < n; i++)
    {
        b[i] -= x[i];
    }
}

void pangulu_vector_copy_cpu(pangulu_vector *B, pangulu_vector *X)
{

    calculate_type *x = X->value;
    calculate_type *b = B->value;
    int_t n = X->row;
    for (int_t i = 0; i < n; i++)
    {
        b[i] = x[i];
    }
}

#endif