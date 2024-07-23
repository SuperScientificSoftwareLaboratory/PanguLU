#ifndef PANGULU_TSTRF_FP64_CUDA_H
#define PANGULU_TSTRF_FP64_CUDA_H

#include "pangulu_common.h"
#include "platforms/02_GPU/01_CUDA/000_CUDA/pangulu_cuda.h"

void pangulu_tstrf_fp64_cuda_v8(pangulu_Smatrix *A,
                                pangulu_Smatrix *X,
                                pangulu_Smatrix *U)
{
    int_t n = A->row;
    int_t nnzU = U->nnz;
    int_t nnzA = A->nnz;

    /*********************************U****************************************/
    int *d_graphInDegree = U->d_graphInDegree;
    cudaMemcpy(d_graphInDegree, U->graphInDegree, n * sizeof(int), cudaMemcpyHostToDevice);
    int *d_id_extractor = U->d_id_extractor;
    cudaMemset(d_id_extractor, 0, sizeof(int));
    calculate_type *d_left_sum = A->d_left_sum;
    cudaMemset(d_left_sum, 0, nnzA * sizeof(calculate_type));
    /*****************************************************************************/
    pangulu_tstrf_cuda_kernel_v8(n,
                                 nnzU,
                                 d_graphInDegree,
                                 d_id_extractor,
                                 d_left_sum,
                                 U->CUDA_rowpointer,
                                 U->CUDA_columnindex,
                                 U->CUDA_value,
                                 A->CUDA_rowpointer,
                                 A->CUDA_columnindex,
                                 X->CUDA_value,
                                 A->CUDA_rowpointer,
                                 A->CUDA_columnindex,
                                 A->CUDA_value);
}

void pangulu_tstrf_fp64_cuda_v9(pangulu_Smatrix *A,
                                pangulu_Smatrix *X,
                                pangulu_Smatrix *U)
{
    int_t n = A->row;
    int_t nnzU = U->nnz;
    int_t nnzA = A->nnz;

    int *d_graphInDegree = U->d_graphInDegree;
    cudaMemcpy(d_graphInDegree, U->graphInDegree, n * sizeof(int), cudaMemcpyHostToDevice);
    int *d_id_extractor = U->d_id_extractor;
    cudaMemset(d_id_extractor, 0, sizeof(int));

    int *d_while_profiler;
    cudaMalloc((void **)&d_while_profiler, sizeof(int) * n);
    cudaMemset(d_while_profiler, 0, sizeof(int) * n);
    pangulu_inblock_ptr *Spointer = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (n + 1));
    memset(Spointer, 0, sizeof(int_t) * (n + 1));
    int_t rhs = 0;
    for (int i = 0; i < n; i++)
    {
        if (A->rowpointer[i] != A->rowpointer[i + 1])
        {
            Spointer[rhs] = i;
            rhs++;
        }
    }
    calculate_type *d_left_sum;
    cudaMalloc((void **)&d_left_sum, n * rhs * sizeof(calculate_type));
    cudaMemset(d_left_sum, 0, n * rhs * sizeof(calculate_type));

    calculate_type *d_x, *d_b;
    cudaMalloc((void **)&d_x, n * rhs * sizeof(calculate_type));
    cudaMalloc((void **)&d_b, n * rhs * sizeof(calculate_type));
    cudaMemset(d_x, 0, n * rhs * sizeof(calculate_type));
    cudaMemset(d_b, 0, n * rhs * sizeof(calculate_type));

    pangulu_inblock_ptr *d_Spointer;
    cudaMalloc((void **)&d_Spointer, sizeof(pangulu_inblock_ptr) * (n + 1));
    cudaMemset(d_Spointer, 0, sizeof(pangulu_inblock_ptr) * (n + 1));
    cudaMemcpy(d_Spointer, Spointer, sizeof(pangulu_inblock_ptr) * (n + 1), cudaMemcpyHostToDevice);

    pangulu_gessm_cuda_kernel_v9(n,
                                 nnzU,
                                 rhs,
                                 nnzA,
                                 d_Spointer,
                                 d_graphInDegree,
                                 d_id_extractor,
                                 d_while_profiler,
                                 U->CUDA_rowpointer,
                                 U->CUDA_columnindex,
                                 U->CUDA_value,
                                 A->CUDA_rowpointer,
                                 A->CUDA_columnindex,
                                 X->CUDA_value,

                                 A->CUDA_rowpointer,
                                 A->CUDA_columnindex,
                                 A->CUDA_value,
                                 d_left_sum,
                                 d_x,
                                 d_b);

    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_left_sum);
    cudaFree(d_while_profiler);
}

void pangulu_tstrf_fp64_cuda_v7(pangulu_Smatrix *A,
                                pangulu_Smatrix *X,
                                pangulu_Smatrix *U)
{
    int_t n = A->row;
    int_t nnzU = U->nnz;
    pangulu_tstrf_cuda_kernel_v7(n,
                                 nnzU,
                                 U->CUDA_rowpointer,
                                 U->CUDA_columnindex,
                                 U->CUDA_value,
                                 A->CUDA_rowpointer,
                                 A->CUDA_columnindex,
                                 X->CUDA_value,
                                 A->CUDA_rowpointer,
                                 A->CUDA_columnindex,
                                 A->CUDA_value);
}

void pangulu_tstrf_fp64_cuda_v10(pangulu_Smatrix *A,
                                 pangulu_Smatrix *X,
                                 pangulu_Smatrix *U)
{
    int_t n = A->row;
    int_t nnzU = U->nnz;
    pangulu_tstrf_cuda_kernel_v10(n,
                                  nnzU,
                                  U->CUDA_rowpointer,
                                  U->CUDA_columnindex,
                                  U->CUDA_value,
                                  A->CUDA_rowpointer,
                                  A->CUDA_columnindex,
                                  X->CUDA_value,
                                  A->CUDA_rowpointer,
                                  A->CUDA_columnindex,
                                  A->CUDA_value);
}

void pangulu_tstrf_fp64_cuda_v11(pangulu_Smatrix *A,
                                 pangulu_Smatrix *X,
                                 pangulu_Smatrix *U)
{
    int_t n = A->row;
    int_t nnzU = U->nnz;
    int_t nnzA = A->nnz;

    /*********************************U****************************************/
    int *d_graphInDegree = U->d_graphInDegree;
    cudaMemcpy(d_graphInDegree, U->graphInDegree, n * sizeof(int), cudaMemcpyHostToDevice);
    int *d_id_extractor = U->d_id_extractor;
    cudaMemset(d_id_extractor, 0, sizeof(int));
    calculate_type *d_left_sum = A->d_left_sum;
    cudaMemset(d_left_sum, 0, nnzA * sizeof(calculate_type));
    /*****************************************************************************/
    pangulu_tstrf_cuda_kernel_v11(n,
                                  nnzU,
                                  d_graphInDegree,
                                  d_id_extractor,
                                  d_left_sum,
                                  U->CUDA_rowpointer,
                                  U->CUDA_columnindex,
                                  U->CUDA_value,
                                  A->CUDA_rowpointer,
                                  A->CUDA_columnindex,
                                  X->CUDA_value,
                                  A->CUDA_rowpointer,
                                  A->CUDA_columnindex,
                                  A->CUDA_value);
}

void pangulu_tstrf_interface_G_V1(pangulu_Smatrix *A,
                                  pangulu_Smatrix *X,
                                  pangulu_Smatrix *U)
{
    pangulu_Smatrix_CUDA_memcpy_value_CSC(A, A);
    pangulu_transport_pangulu_Smatrix_CSC_to_CSR(A);
    pangulu_Smatrix_CUDA_memcpy_complete_CSR(A, A);
    pangulu_tstrf_fp64_cuda_v7(A, X, U);
    pangulu_Smatrix_CUDA_memcpy_value_CSR(A, X);
    pangulu_transport_pangulu_Smatrix_CSR_to_CSC(A);
}
void pangulu_tstrf_interface_G_V2(pangulu_Smatrix *A,
                                  pangulu_Smatrix *X,
                                  pangulu_Smatrix *U)
{
    pangulu_tstrf_fp64_cuda_v8(A, X, U);
    pangulu_Smatrix_CUDA_memcpy_value_CSC(A, X);
}
void pangulu_tstrf_interface_G_V3(pangulu_Smatrix *A,
                                  pangulu_Smatrix *X,
                                  pangulu_Smatrix *U)
{
    pangulu_Smatrix_CUDA_memcpy_value_CSC(A, A);
    pangulu_transport_pangulu_Smatrix_CSC_to_CSR(A);
    pangulu_Smatrix_CUDA_memcpy_complete_CSR(A, A);
    pangulu_tstrf_fp64_cuda_v10(A, X, U);
    pangulu_Smatrix_CUDA_memcpy_value_CSR(A, X);
    pangulu_transport_pangulu_Smatrix_CSR_to_CSC(A);
}

#endif