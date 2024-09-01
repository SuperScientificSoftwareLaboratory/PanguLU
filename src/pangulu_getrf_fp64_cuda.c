#include "pangulu_common.h"

#ifdef GPU_OPEN
void pangulu_getrf_fp64_cuda(pangulu_smatrix *a,
                             pangulu_smatrix *l,
                             pangulu_smatrix *u)
{

    if (a->nnz > 1e4)
    {
        pangulu_getrf_cuda_dense_kernel(a->row,
                                        a->rowpointer[a->row],
                                        u->cuda_nnzu,
                                        a->cuda_rowpointer,
                                        a->cuda_columnindex,
                                        a->cuda_value,
                                        l->cuda_rowpointer,
                                        l->cuda_columnindex,
                                        l->cuda_value,
                                        u->cuda_rowpointer,
                                        u->cuda_columnindex,
                                        u->cuda_value);
    }
    else
    {
        pangulu_getrf_cuda_kernel(a->row,
                                  a->rowpointer[a->row],
                                  u->cuda_nnzu,
                                  a->cuda_rowpointer,
                                  a->cuda_columnindex,
                                  a->cuda_value,
                                  l->cuda_rowpointer,
                                  l->cuda_columnindex,
                                  l->cuda_value,
                                  u->cuda_rowpointer,
                                  u->cuda_columnindex,
                                  u->cuda_value);
    }
}

void pangulu_getrf_interface_G_V1(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *u)
{
    pangulu_getrf_cuda_kernel(a->row,
                              a->rowpointer[a->row],
                              u->cuda_nnzu,
                              a->cuda_rowpointer,
                              a->cuda_columnindex,
                              a->cuda_value,
                              l->cuda_rowpointer,
                              l->cuda_columnindex,
                              l->cuda_value,
                              u->cuda_rowpointer,
                              u->cuda_columnindex,
                              u->cuda_value);
    pangulu_smatrix_cuda_memcpy_value_csc(l, l);
    pangulu_smatrix_cuda_memcpy_value_csc(u, u);
}
void pangulu_getrf_interface_G_V2(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *u)
{
    pangulu_getrf_cuda_dense_kernel(a->row,
                                    a->rowpointer[a->row],
                                    u->cuda_nnzu,
                                    a->cuda_rowpointer,
                                    a->cuda_columnindex,
                                    a->cuda_value,
                                    l->cuda_rowpointer,
                                    l->cuda_columnindex,
                                    l->cuda_value,
                                    u->cuda_rowpointer,
                                    u->cuda_columnindex,
                                    u->cuda_value);
    pangulu_smatrix_cuda_memcpy_value_csc(l, l);
    pangulu_smatrix_cuda_memcpy_value_csc(u, u);
}

#endif