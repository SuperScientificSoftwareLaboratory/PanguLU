#include "pangulu_common.h"

#ifdef GPU_OPEN
void pangulu_tstrf_fp64_cuda_v8(pangulu_smatrix *a,
                                pangulu_smatrix *x,
                                pangulu_smatrix *u)
{
    pangulu_int64_t n = a->row;
    pangulu_int64_t nnzU = u->nnz;
    pangulu_int64_t nnzA = a->nnz;

    /*********************************u****************************************/
    int *d_graphindegree = u->d_graphindegree;
    cudaMemcpy(d_graphindegree, u->graphindegree, n * sizeof(int), cudaMemcpyHostToDevice);
    int *d_id_extractor = u->d_id_extractor;
    cudaMemset(d_id_extractor, 0, sizeof(int));
    calculate_type *d_left_sum = a->d_left_sum;
    cudaMemset(d_left_sum, 0, nnzA * sizeof(calculate_type));
    /*****************************************************************************/
    pangulu_tstrf_cuda_kernel_v8(n,
                                 nnzU,
                                 d_graphindegree,
                                 d_id_extractor,
                                 d_left_sum,
                                 u->cuda_rowpointer,
                                 u->cuda_columnindex,
                                 u->cuda_value,
                                 a->cuda_rowpointer,
                                 a->cuda_columnindex,
                                 x->cuda_value,
                                 a->cuda_rowpointer,
                                 a->cuda_columnindex,
                                 a->cuda_value);
}

void pangulu_tstrf_fp64_cuda_v9(pangulu_smatrix *a,
                                pangulu_smatrix *x,
                                pangulu_smatrix *u)
{
    pangulu_int64_t n = a->row;
    pangulu_int64_t nnzU = u->nnz;
    pangulu_int64_t nnzA = a->nnz;

    int *d_graphindegree = u->d_graphindegree;
    cudaMemcpy(d_graphindegree, u->graphindegree, n * sizeof(int), cudaMemcpyHostToDevice);
    int *d_id_extractor = u->d_id_extractor;
    cudaMemset(d_id_extractor, 0, sizeof(int));

    int *d_while_profiler;
    cudaMalloc((void **)&d_while_profiler, sizeof(int) * n);
    cudaMemset(d_while_profiler, 0, sizeof(int) * n);
    pangulu_inblock_ptr *Spointer = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (n + 1));
    memset(Spointer, 0, sizeof(pangulu_int64_t) * (n + 1));
    pangulu_int64_t rhs = 0;
    for (int i = 0; i < n; i++)
    {
        if (a->rowpointer[i] != a->rowpointer[i + 1])
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
                                 d_graphindegree,
                                 d_id_extractor,
                                 d_while_profiler,
                                 u->cuda_rowpointer,
                                 u->cuda_columnindex,
                                 u->cuda_value,
                                 a->cuda_rowpointer,
                                 a->cuda_columnindex,
                                 x->cuda_value,

                                 a->cuda_rowpointer,
                                 a->cuda_columnindex,
                                 a->cuda_value,
                                 d_left_sum,
                                 d_x,
                                 d_b);

    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_left_sum);
    cudaFree(d_while_profiler);
}

void pangulu_tstrf_fp64_cuda_v7(pangulu_smatrix *a,
                                pangulu_smatrix *x,
                                pangulu_smatrix *u)
{
    pangulu_int64_t n = a->row;
    pangulu_int64_t nnzU = u->nnz;
    pangulu_tstrf_cuda_kernel_v7(n,
                                 nnzU,
                                 u->cuda_rowpointer,
                                 u->cuda_columnindex,
                                 u->cuda_value,
                                 a->cuda_rowpointer,
                                 a->cuda_columnindex,
                                 x->cuda_value,
                                 a->cuda_rowpointer,
                                 a->cuda_columnindex,
                                 a->cuda_value);
}

void pangulu_tstrf_fp64_cuda_v10(pangulu_smatrix *a,
                                 pangulu_smatrix *x,
                                 pangulu_smatrix *u)
{
    pangulu_int64_t n = a->row;
    pangulu_int64_t nnzU = u->nnz;
    pangulu_tstrf_cuda_kernel_v10(n,
                                  nnzU,
                                  u->cuda_rowpointer,
                                  u->cuda_columnindex,
                                  u->cuda_value,
                                  a->cuda_rowpointer,
                                  a->cuda_columnindex,
                                  x->cuda_value,
                                  a->cuda_rowpointer,
                                  a->cuda_columnindex,
                                  a->cuda_value);
}

void pangulu_tstrf_fp64_cuda_v11(pangulu_smatrix *a,
                                 pangulu_smatrix *x,
                                 pangulu_smatrix *u)
{
    pangulu_int64_t n = a->row;
    pangulu_int64_t nnzU = u->nnz;
    pangulu_int64_t nnzA = a->nnz;

    /*********************************u****************************************/
    int *d_graphindegree = u->d_graphindegree;
    cudaMemcpy(d_graphindegree, u->graphindegree, n * sizeof(int), cudaMemcpyHostToDevice);
    int *d_id_extractor = u->d_id_extractor;
    cudaMemset(d_id_extractor, 0, sizeof(int));
    calculate_type *d_left_sum = a->d_left_sum;
    cudaMemset(d_left_sum, 0, nnzA * sizeof(calculate_type));
    /*****************************************************************************/
    pangulu_tstrf_cuda_kernel_v11(n,
                                  nnzU,
                                  d_graphindegree,
                                  d_id_extractor,
                                  d_left_sum,
                                  u->cuda_rowpointer,
                                  u->cuda_columnindex,
                                  u->cuda_value,
                                  a->cuda_rowpointer,
                                  a->cuda_columnindex,
                                  x->cuda_value,
                                  a->cuda_rowpointer,
                                  a->cuda_columnindex,
                                  a->cuda_value);
}

void pangulu_tstrf_interface_G_V1(pangulu_smatrix *a,
                                  pangulu_smatrix *x,
                                  pangulu_smatrix *u)
{
    pangulu_smatrix_cuda_memcpy_value_csc(a, a);
    pangulu_transpose_pangulu_smatrix_csc_to_csr(a);
    pangulu_smatrix_cuda_memcpy_complete_csr(a, a);
    pangulu_tstrf_fp64_cuda_v7(a, x, u);
    pangulu_smatrix_cuda_memcpy_value_csr(a, x);
    pangulu_transpose_pangulu_smatrix_csr_to_csc(a);
}
void pangulu_tstrf_interface_G_V2(pangulu_smatrix *a,
                                  pangulu_smatrix *x,
                                  pangulu_smatrix *u)
{
    pangulu_tstrf_fp64_cuda_v8(a, x, u);
    pangulu_smatrix_cuda_memcpy_value_csc(a, x);
}
void pangulu_tstrf_interface_G_V3(pangulu_smatrix *a,
                                  pangulu_smatrix *x,
                                  pangulu_smatrix *u)
{
    pangulu_smatrix_cuda_memcpy_value_csc(a, a);
    pangulu_transpose_pangulu_smatrix_csc_to_csr(a);
    pangulu_smatrix_cuda_memcpy_complete_csr(a, a);
    pangulu_tstrf_fp64_cuda_v10(a, x, u);
    pangulu_smatrix_cuda_memcpy_value_csr(a, x);
    pangulu_transpose_pangulu_smatrix_csr_to_csc(a);
}
#endif