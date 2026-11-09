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

    /* Additive checked interface; legacy void symbols remain available.
     * Init requires *pangulu_handle == NULL. CSC input remains caller-owned. */
    pangulu_status_t pangulu_init_checked(sparse_index_t n, sparse_pointer_t nnz,
        sparse_pointer_t *colptr, sparse_index_t *rowidx, sparse_value_t *values,
        pangulu_init_options *options, void **pangulu_handle);
    pangulu_status_t pangulu_gstrf_checked(pangulu_gstrf_options *options, void **pangulu_handle);
    pangulu_status_t pangulu_gstrs_checked(sparse_value_t *rhs, pangulu_gstrs_options *options, void **pangulu_handle);
    pangulu_status_t pangulu_gssv_checked(sparse_value_t *rhs, pangulu_gstrf_options *factor_options,
        pangulu_gstrs_options *solve_options, void **pangulu_handle);
    pangulu_status_t pangulu_get_status(void *pangulu_handle);
    const char *pangulu_status_string(pangulu_status_t status);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // PANGULU_H
