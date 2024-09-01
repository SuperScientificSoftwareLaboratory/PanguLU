#include "pangulu_common.h"

#ifdef GPU_OPEN
void pangulu_ssssm_fp64_cuda(pangulu_smatrix *a,
                             pangulu_smatrix *l,
                             pangulu_smatrix *u)
{
    int n = a->row;
    int nnz_a = a->columnpointer[n] - a->columnpointer[0];
    double sparsity_A = (double)nnz_a / (double)(n * n);

    if (sparsity_A < 0.001)
    {
        pangulu_ssssm_cuda_kernel(a->row,
                                  a->bin_rowpointer,
                                  a->cuda_bin_rowpointer,
                                  a->cuda_bin_rowindex,
                                  u->cuda_rowpointer,
                                  u->cuda_columnindex,
                                  u->cuda_value,
                                  l->cuda_rowpointer,
                                  l->cuda_columnindex,
                                  l->cuda_value,
                                  a->cuda_rowpointer,
                                  a->cuda_columnindex,
                                  a->cuda_value);
    }
    else
    {
        pangulu_ssssm_dense_cuda_kernel(a->row,
                                        a->columnpointer[a->row],
                                        u->columnpointer[u->row],
                                        l->cuda_rowpointer,
                                        l->cuda_columnindex,
                                        l->cuda_value,
                                        u->cuda_rowpointer,
                                        u->cuda_columnindex,
                                        u->cuda_value,
                                        a->cuda_rowpointer,
                                        a->cuda_columnindex,
                                        a->cuda_value);
    }
}

void pangulu_ssssm_interface_G_V1(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *u)
{
    pangulu_ssssm_cuda_kernel(a->row,
                              a->bin_rowpointer,
                              a->cuda_bin_rowpointer,
                              a->cuda_bin_rowindex,
                              u->cuda_rowpointer,
                              u->cuda_columnindex,
                              u->cuda_value,
                              l->cuda_rowpointer,
                              l->cuda_columnindex,
                              l->cuda_value,
                              a->cuda_rowpointer,
                              a->cuda_columnindex,
                              a->cuda_value);
}
void pangulu_ssssm_interface_G_V2(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *u)
{
    pangulu_ssssm_dense_cuda_kernel(a->row,
                                    a->columnpointer[a->row],
                                    u->columnpointer[u->row],
                                    l->cuda_rowpointer,
                                    l->cuda_columnindex,
                                    l->cuda_value,
                                    u->cuda_rowpointer,
                                    u->cuda_columnindex,
                                    u->cuda_value,
                                    a->cuda_rowpointer,
                                    a->cuda_columnindex,
                                    a->cuda_value);
}
#endif