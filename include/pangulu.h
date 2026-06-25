#ifndef PANGULU_H
#define PANGULU_H

#include "pangulu_interface_common.h"

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

    void pangulu_init(sparse_index_t pangulu_n, sparse_pointer_t pangulu_nnz, sparse_pointer_t *csc_colptr, sparse_index_t *csc_rowidx, sparse_value_t *csc_value, pangulu_init_options *init_options, void **pangulu_handle);
    void pangulu_gstrf(pangulu_gstrf_options *gstrf_options, void **pangulu_handle);
    void pangulu_gstrs(sparse_value_t *rhs, pangulu_gstrs_options *gstrs_options, void **pangulu_handle);
    void pangulu_gssv(sparse_value_t *rhs, pangulu_gstrf_options *gstrf_options, pangulu_gstrs_options *gstrs_options, void **pangulu_handle);
    void pangulu_finalize(void **pangulu_handle);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // PANGULU_H
