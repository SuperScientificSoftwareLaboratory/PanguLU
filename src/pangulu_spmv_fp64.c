#include "pangulu_common.h"

void pangulu_spmv_cpu_choumi(pangulu_smatrix *s, pangulu_vector *x, pangulu_vector *b)
{
    calculate_type *value = s->value;
    calculate_type *bval = b->value;
    calculate_type *xval = x->value;
    pangulu_int64_t n = s->column;
    pangulu_int64_t m = s->row;
    for (pangulu_int64_t i = 0; i < m; i++)
        bval[i] = 0.0;
    for (pangulu_int64_t i = 0; i < m; i++)
    {
        for (pangulu_int64_t j = 0; j < n; j++)
        {
            bval[i] += value[i * n + j] * xval[j];
        }
    }
}

void pangulu_spmv_cpu_xishu(pangulu_smatrix *s, pangulu_vector *x, pangulu_vector *b, pangulu_int64_t vector_number)
{
    pangulu_int64_t m = s->row;
    pangulu_inblock_ptr *csrRowPtr_tmp = s->rowpointer;
    pangulu_inblock_idx *csrColIdx_tmp = s->columnindex;
    calculate_type *csrVal_tmp = s->value;
    for (pangulu_int64_t vector_index = 0; vector_index < vector_number; vector_index++)
    {
        calculate_type *xval = x->value + vector_index * m;
        calculate_type *yval = b->value + vector_index * m;
        for (pangulu_int64_t i = 0; i < m; i++)
        {
            for (pangulu_int64_t j = csrRowPtr_tmp[i]; j < csrRowPtr_tmp[i + 1]; j++)
            {
                yval[i] += csrVal_tmp[j] * xval[csrColIdx_tmp[j]];
            }
        }
    }
}

void pangulu_spmv_cpu_xishu_csc(pangulu_smatrix *s, pangulu_vector *x, pangulu_vector *b, pangulu_int64_t vector_number)
{
    pangulu_int64_t m = s->row;
    pangulu_inblock_ptr *csccolumnPtr_tmp = s->columnpointer;
    pangulu_inblock_idx *cscrowIdx_tmp = s->rowindex;
    calculate_type *cscVal_tmp = s->value_csc;
    for (pangulu_int64_t vector_index = 0; vector_index < vector_number; vector_index++)
    {
        calculate_type *xval = x->value + vector_index * m;
        calculate_type *yval = b->value + vector_index * m;
        for (pangulu_int64_t i = 0; i < m; i++)
        {
            for (pangulu_int64_t j = csccolumnPtr_tmp[i]; j < csccolumnPtr_tmp[i + 1]; j++)
            {
                pangulu_inblock_idx row = cscrowIdx_tmp[j];
                yval[row] += cscVal_tmp[j] * xval[i];
            }
        }
    }
}

void pangulu_vector_add_cpu(pangulu_vector *b, pangulu_vector *x)
{

    calculate_type *xval = x->value;
    calculate_type *bval = b->value;
    pangulu_int64_t n = x->row;
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        bval[i] += xval[i];
    }
}

void pangulu_vector_sub_cpu(pangulu_vector *b, pangulu_vector *x)
{

    calculate_type *xval = x->value;
    calculate_type *bval = b->value;
    pangulu_int64_t n = x->row;
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        bval[i] -= xval[i];
    }
}

void pangulu_vector_copy_cpu(pangulu_vector *b, pangulu_vector *x)
{

    calculate_type *xval = x->value;
    calculate_type *bval = b->value;
    pangulu_int64_t n = x->row;
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        bval[i] = xval[i];
    }
}