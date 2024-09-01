#include "pangulu_common.h"

void pangulu_add_pangulu_smatrix_cpu(pangulu_smatrix *a,
                                     pangulu_smatrix *b)
{
    for (pangulu_int64_t i = 0; i < a->nnz; i++)
    {
        a->value_csc[i] += b->value_csc[i];
    }
}

void pangulu_add_pangulu_smatrix_csr_to_csc(pangulu_smatrix *a)
{
    for (pangulu_int64_t i = 0; i < a->nnz; i++)
    {
        a->value_csc[i] += a->value[i];
    }
}