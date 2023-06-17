#ifndef PANGULU_ADD_MATRIX_CUDA_H
#define PANGULU_ADD_MATRIX_CUDA_H

#include "pangulu_common.h"
#include "pangulu_cuda.h"

void pangulu_add_pangulu_Smatrix_cuda(pangulu_Smatrix *A,
                                      pangulu_Smatrix *B)
{
    pangulu_cuda_vector_add_kernel(A->nnz, A->CUDA_value, B->CUDA_value);
}

#endif