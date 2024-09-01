#include "pangulu_common.h"

void pangulu_add_pangulu_smatrix_cuda(pangulu_smatrix *a,
                                      pangulu_smatrix *b)
{
#ifdef GPU_OPEN
    pangulu_cuda_vector_add_kernel(a->nnz, a->cuda_value, b->cuda_value);
#endif
}