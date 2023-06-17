#ifndef PANGULU_GETRF_FP64_CUDA_H
#define PANGULU_GETRF_FP64_CUDA_H

#include "pangulu_common.h"
#include "pangulu_cuda.h"
#include "pangulu_cuda_interface.h"

void pangulu_getrf_fp64_cuda(pangulu_Smatrix *A,
                             pangulu_Smatrix *L,
                             pangulu_Smatrix *U)
{

    if (A->nnz > 1e4)
    {
        pangulu_getrf_cuda_dense_kernel(A->row,
                                        A->rowpointer[A->row],
                                        U->CUDA_nnzU,
                                        A->CUDA_rowpointer,
                                        A->CUDA_columnindex,
                                        A->CUDA_value,
                                        L->CUDA_rowpointer,
                                        L->CUDA_columnindex,
                                        L->CUDA_value,
                                        U->CUDA_rowpointer,
                                        U->CUDA_columnindex,
                                        U->CUDA_value);
    }
    else
    {
        pangulu_getrf_cuda_kernel(A->row,
                                  A->rowpointer[A->row],
                                  U->CUDA_nnzU,
                                  A->CUDA_rowpointer,
                                  A->CUDA_columnindex,
                                  A->CUDA_value,
                                  L->CUDA_rowpointer,
                                  L->CUDA_columnindex,
                                  L->CUDA_value,
                                  U->CUDA_rowpointer,
                                  U->CUDA_columnindex,
                                  U->CUDA_value);
    }
}

void pangulu_getrf_interface_G_V1(pangulu_Smatrix *A,
                                  pangulu_Smatrix *L,
                                  pangulu_Smatrix *U)
{
    pangulu_getrf_cuda_kernel(A->row,
                              A->rowpointer[A->row],
                              U->CUDA_nnzU,
                              A->CUDA_rowpointer,
                              A->CUDA_columnindex,
                              A->CUDA_value,
                              L->CUDA_rowpointer,
                              L->CUDA_columnindex,
                              L->CUDA_value,
                              U->CUDA_rowpointer,
                              U->CUDA_columnindex,
                              U->CUDA_value);
    pangulu_Smatrix_CUDA_memcpy_value_CSC(L, L);
    pangulu_Smatrix_CUDA_memcpy_value_CSC(U, U);
}
void pangulu_getrf_interface_G_V2(pangulu_Smatrix *A,
                                  pangulu_Smatrix *L,
                                  pangulu_Smatrix *U)
{
    pangulu_getrf_cuda_dense_kernel(A->row,
                                    A->rowpointer[A->row],
                                    U->CUDA_nnzU,
                                    A->CUDA_rowpointer,
                                    A->CUDA_columnindex,
                                    A->CUDA_value,
                                    L->CUDA_rowpointer,
                                    L->CUDA_columnindex,
                                    L->CUDA_value,
                                    U->CUDA_rowpointer,
                                    U->CUDA_columnindex,
                                    U->CUDA_value);
    pangulu_Smatrix_CUDA_memcpy_value_CSC(L, L);
    pangulu_Smatrix_CUDA_memcpy_value_CSC(U, U);
}

#endif