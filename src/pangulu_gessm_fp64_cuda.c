#include "pangulu_common.h"

#ifdef GPU_OPEN
void pangulu_gessm_fp64_cuda_v9(pangulu_smatrix *a,
                                pangulu_smatrix *l,
                                pangulu_smatrix *x)
{

    pangulu_int64_t n = a->row;
    pangulu_int64_t nnzl = l->nnz;
    pangulu_int64_t nnza = a->nnz;

    int *d_graphindegree = l->d_graphindegree;
    cudaMemcpy(d_graphindegree, l->graphindegree, n * sizeof(int), cudaMemcpyHostToDevice);
    int *d_id_extractor = l->d_id_extractor;
    cudaMemset(d_id_extractor, 0, sizeof(int));

    int *d_while_profiler;
    cudaMalloc((void **)&d_while_profiler, sizeof(int) * n);
    cudaMemset(d_while_profiler, 0, sizeof(int) * n);
    pangulu_int64_t *spointer = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (n + 1));
    memset(spointer, 0, sizeof(pangulu_int64_t) * (n + 1));
    pangulu_int64_t rhs = 0;
    for (int i = 0; i < n; i++)
    {
        if (a->columnpointer[i] != a->columnpointer[i + 1])
        {
            spointer[rhs] = i;
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

    pangulu_inblock_ptr *d_spointer;
    cudaMalloc((void **)&d_spointer, sizeof(pangulu_inblock_ptr) * (n + 1));
    cudaMemset(d_spointer, 0, sizeof(pangulu_inblock_ptr) * (n + 1));
    cudaMemcpy(d_spointer, spointer, sizeof(pangulu_inblock_ptr) * (n + 1), cudaMemcpyHostToDevice);

    pangulu_gessm_cuda_kernel_v9(n,
                                 nnzl,
                                 rhs,
                                 nnza,
                                 d_spointer,
                                 d_graphindegree,
                                 d_id_extractor,
                                 d_while_profiler,
                                 l->cuda_rowpointer,
                                 l->cuda_columnindex,
                                 l->cuda_value,
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

void pangulu_gessm_fp64_cuda_v11(pangulu_smatrix *a,
                                 pangulu_smatrix *l,
                                 pangulu_smatrix *x)
{
    pangulu_int64_t n = a->row;
    pangulu_int64_t nnzl = l->nnz;
    pangulu_int64_t nnza = a->nnz;
    /**********************************l****************************************/
    int *d_graphindegree = l->d_graphindegree;
    cudaMemcpy(d_graphindegree, l->graphindegree, n * sizeof(int), cudaMemcpyHostToDevice);
    int *d_id_extractor = l->d_id_extractor;
    cudaMemset(d_id_extractor, 0, sizeof(int));

    calculate_type *d_left_sum = a->d_left_sum;
    cudaMemset(d_left_sum, 0, nnza * sizeof(calculate_type));
    /*****************************************************************************/
    pangulu_gessm_cuda_kernel_v11(n,
                                  nnzl,
                                  nnza,
                                  d_graphindegree,
                                  d_id_extractor,
                                  d_left_sum,
                                  l->cuda_rowpointer,
                                  l->cuda_columnindex,
                                  l->cuda_value,
                                  a->cuda_rowpointer,
                                  a->cuda_columnindex,
                                  x->cuda_value,
                                  a->cuda_rowpointer,
                                  a->cuda_columnindex,
                                  a->cuda_value);
    cudaDeviceSynchronize();
}

void pangulu_gessm_fp64_cuda_v7(pangulu_smatrix *a,
                                pangulu_smatrix *l,
                                pangulu_smatrix *x)
{

    pangulu_int64_t n = a->row;
    pangulu_int64_t nnzl = l->nnz;
    pangulu_gessm_cuda_kernel_v7(n,
                                 nnzl,
                                 l->cuda_rowpointer,
                                 l->cuda_columnindex,
                                 l->cuda_value,
                                 a->cuda_rowpointer,
                                 a->cuda_columnindex,
                                 x->cuda_value,
                                 a->cuda_rowpointer,
                                 a->cuda_columnindex,
                                 a->cuda_value);
}

void pangulu_gessm_fp64_cuda_v8(pangulu_smatrix *a,
                                pangulu_smatrix *l,
                                pangulu_smatrix *x)
{
    pangulu_int64_t n = a->row;
    pangulu_int64_t nnzl = l->nnz;
    pangulu_int64_t nnza = a->nnz;
    /**********************************l****************************************/
    int *d_graphindegree = l->d_graphindegree;
    cudaMemcpy(d_graphindegree, l->graphindegree, n * sizeof(int), cudaMemcpyHostToDevice);
    int *d_id_extractor = l->d_id_extractor;
    cudaMemset(d_id_extractor, 0, sizeof(int));

    calculate_type *d_left_sum = a->d_left_sum;
    cudaMemset(d_left_sum, 0, nnza * sizeof(calculate_type));
    /*****************************************************************************/
    pangulu_gessm_cuda_kernel_v8(n,
                                 nnzl,
                                 nnza,
                                 d_graphindegree,
                                 d_id_extractor,
                                 d_left_sum,
                                 l->cuda_rowpointer,
                                 l->cuda_columnindex,
                                 l->cuda_value,
                                 a->cuda_rowpointer,
                                 a->cuda_columnindex,
                                 x->cuda_value,
                                 a->cuda_rowpointer,
                                 a->cuda_columnindex,
                                 a->cuda_value);
    cudaDeviceSynchronize();
}

void pangulu_gessm_fp64_cuda_v10(pangulu_smatrix *a,
                                 pangulu_smatrix *l,
                                 pangulu_smatrix *x)
{

    pangulu_int64_t n = a->row;
    pangulu_int64_t nnzl = l->nnz;
    pangulu_gessm_cuda_kernel_v10(n,
                                  nnzl,
                                  l->cuda_rowpointer,
                                  l->cuda_columnindex,
                                  l->cuda_value,
                                  a->cuda_rowpointer,
                                  a->cuda_columnindex,
                                  x->cuda_value,
                                  a->cuda_rowpointer,
                                  a->cuda_columnindex,
                                  a->cuda_value);
}

void pangulu_gessm_interface_g_v1(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *x)
{
    pangulu_gessm_fp64_cuda_v7(a, l, x);
    pangulu_smatrix_cuda_memcpy_value_csc(a, x);
}
void pangulu_gessm_interface_g_v2(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *x)
{
    pangulu_smatrix_cuda_memcpy_value_csc(a, a);
    pangulu_transpose_pangulu_smatrix_csc_to_csr(a);
    pangulu_smatrix_cuda_memcpy_complete_csr(a, a);

    pangulu_gessm_fp64_cuda_v8(a, l, x);

    pangulu_smatrix_cuda_memcpy_value_csr(a, x);
    pangulu_transpose_pangulu_smatrix_csr_to_csc(a);
}
void pangulu_gessm_interface_g_v3(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *x)
{
    pangulu_gessm_fp64_cuda_v10(a, l, x);
    pangulu_smatrix_cuda_memcpy_value_csc(a, x);
}
#endif