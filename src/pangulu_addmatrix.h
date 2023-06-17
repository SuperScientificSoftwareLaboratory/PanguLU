#ifndef PANGULU_ADD_MATRIX_H
#define PANGULU_ADD_MATRIX_H

#include "pangulu_common.h"

void pangulu_add_pangulu_Smatrix_cpu(pangulu_Smatrix *A,
                                     pangulu_Smatrix *B)
{
    for (int_t i = 0; i < A->nnz; i++)
    {
        A->value_CSC[i] += B->value_CSC[i];
    }
}

void pangulu_add_pangulu_Smatrix_CSR_to_CSC(pangulu_Smatrix *A)
{
    for (int_t i = 0; i < A->nnz; i++)
    {
        A->value_CSC[i] += A->value[i];
    }
}

#endif