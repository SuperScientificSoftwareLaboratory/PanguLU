#ifndef PANGULU_SSSSM_FP64_CUDA_H
#define PANGULU_SSSSM_FP64_CUDA_H

#include "pangulu_common.h"
#include "pangulu_cuda.h"

void pangulu_ssssm_fp64_cuda(pangulu_Smatrix *A,
                             pangulu_Smatrix *L,
                             pangulu_Smatrix *U)
{
    int n = A->row;
    int nnz_a = A->columnpointer[n] - A->columnpointer[0];
    double sparsity_A = (double)nnz_a / (double)(n * n);

    if (sparsity_A < 0.001)
    {
        pangulu_ssssm_cuda_kernel(A->row,
                                  A->bin_rowpointer,
                                  A->CUDA_bin_rowpointer,
                                  A->CUDA_bin_rowindex,
                                  U->CUDA_rowpointer,
                                  U->CUDA_columnindex,
                                  U->CUDA_value,
                                  L->CUDA_rowpointer,
                                  L->CUDA_columnindex,
                                  L->CUDA_value,
                                  A->CUDA_rowpointer,
                                  A->CUDA_columnindex,
                                  A->CUDA_value);
    }
    else
    {
        pangulu_ssssm_dense_cuda_kernel(A->row,
                                        A->columnpointer[A->row],
                                        U->columnpointer[U->row],
                                        L->CUDA_rowpointer,
                                        L->CUDA_columnindex,
                                        L->CUDA_value,
                                        U->CUDA_rowpointer,
                                        U->CUDA_columnindex,
                                        U->CUDA_value,
                                        A->CUDA_rowpointer,
                                        A->CUDA_columnindex,
                                        A->CUDA_value);
    }
}

void pangulu_ssssm_interface_G_V1(pangulu_Smatrix *A,
                                  pangulu_Smatrix *L,
                                  pangulu_Smatrix *U)
{
    pangulu_ssssm_cuda_kernel(A->row,
                              A->bin_rowpointer,
                              A->CUDA_bin_rowpointer,
                              A->CUDA_bin_rowindex,
                              U->CUDA_rowpointer,
                              U->CUDA_columnindex,
                              U->CUDA_value,
                              L->CUDA_rowpointer,
                              L->CUDA_columnindex,
                              L->CUDA_value,
                              A->CUDA_rowpointer,
                              A->CUDA_columnindex,
                              A->CUDA_value);
}
void pangulu_ssssm_interface_G_V2(pangulu_Smatrix *A,
                                  pangulu_Smatrix *L,
                                  pangulu_Smatrix *U)
{
    pangulu_ssssm_dense_cuda_kernel(A->row,
                                    A->columnpointer[A->row],
                                    U->columnpointer[U->row],
                                    L->CUDA_rowpointer,
                                    L->CUDA_columnindex,
                                    L->CUDA_value,
                                    U->CUDA_rowpointer,
                                    U->CUDA_columnindex,
                                    U->CUDA_value,
                                    A->CUDA_rowpointer,
                                    A->CUDA_columnindex,
                                    A->CUDA_value);
}
#endif