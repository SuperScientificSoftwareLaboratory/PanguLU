#ifndef PANGULU_SPTRSV_FP64_H
#define PANGULU_SPTRSV_FP64_H

#include "pangulu_common.h"
#include "pangulu_utils.h"

void pangulu_sptrsv_cpu_choumi(pangulu_Smatrix *S, pangulu_vector *X, pangulu_vector *B)
{
    calculate_type *value = S->value;
    calculate_type *b = B->value;
    calculate_type *x = X->value;
    int_t N = S->column;
    for (int_t i = 0; i < N; i++)
    {
        for (int_t j = 0; j < N; j++)
        {
            if (i == j)
            {
                x[i] = b[i] / value[i * N + j];
                for (int_t k = i + 1; k < N; k++)
                {
                    b[k] -= x[i] * value[k * N + j];
                }
            }
        }
    }
}

void pangulu_sptrsv_cpu_xishu(pangulu_Smatrix *S, pangulu_vector *X, pangulu_vector *B, int_t vector_number)
{
    int_t row = S->row;
    int_t *csrRowPtr_tmp = S->rowpointer;
    int_32t *csrColIdx_tmp = S->columnindex;
    calculate_type *csrVal_tmp = S->value;
    for (int_t vector_index = 0; vector_index < vector_number; vector_index++)
    {
        calculate_type *x = X->value + vector_index * row;
        calculate_type *b = B->value + vector_index * row;
        for (int_t i = 0; i < row; i++)
        {
            calculate_type sumf = 0;
            int_t have = 0;
            for (int_t j = csrRowPtr_tmp[i]; j < csrRowPtr_tmp[i + 1]; j++)
            {
                if (i != csrColIdx_tmp[j])
                    sumf += csrVal_tmp[j] * x[csrColIdx_tmp[j]];
                else
                    have = 1;
            }
            if (have == 0)
            {
                x[i] = 0.0;
            }
            else
            {
                x[i] = (b[i] - sumf) / csrVal_tmp[csrRowPtr_tmp[i + 1] - 1];
            }
        }
    }
}
void pangulu_sptrsv_cpu_xishu_csc(pangulu_Smatrix *S, pangulu_vector *X, pangulu_vector *B, int_t vector_number, int_t tag)
{
    int_t col = S->column;
    int_t *cscColumnPtr_tmp = S->columnpointer;
    int_32t *cscRowIdx_tmp = S->rowindex;
    calculate_type *cscVal_tmp = S->value_CSC;
    if (tag == 0)
    {
        for (int_t vector_index = 0; vector_index < vector_number; vector_index++)
        {
            calculate_type *x = X->value + vector_index * col;
            calculate_type *b = B->value + vector_index * col;
            for (int_t i = 0; i < col; i++)
            {
                if (cscRowIdx_tmp[cscColumnPtr_tmp[i]] == i)
                {
                    x[i] = b[i] / cscVal_tmp[cscColumnPtr_tmp[i]];
                }
                else
                {
                    x[i] = 0.0;
                    continue;
                }
                for (int_t j = cscColumnPtr_tmp[i] + 1; j < cscColumnPtr_tmp[i + 1]; j++)
                {
                    int_32t row = cscRowIdx_tmp[j];
                    b[row] -= cscVal_tmp[j] * x[i];
                }
            }
        }
    }
    else
    {
        for (int_t vector_index = 0; vector_index < vector_number; vector_index++)
        {
            calculate_type *x = X->value + vector_index * col;
            calculate_type *b = B->value + vector_index * col;
            for (int_t i = col - 1; i >= 0; i--)
            {
                if (cscRowIdx_tmp[cscColumnPtr_tmp[i + 1] - 1] == i)
                {
                    x[i] = b[i] / cscVal_tmp[cscColumnPtr_tmp[i + 1] - 1];
                }
                else
                {
                    x[i] = 0.0;
                    continue;
                }
                for (int_t j = cscColumnPtr_tmp[i + 1] - 2; j >= cscColumnPtr_tmp[i]; j--)
                {

                    int_32t row = cscRowIdx_tmp[j];
                    b[row] -= cscVal_tmp[j] * x[i];
                }
            }
        }
    }
}

#endif