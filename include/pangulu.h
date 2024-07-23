#ifndef PANGULU_H
#define PANGULU_H
#include "pangulu_interface_common.h"
void pangulu_init(int pangulu_n, long long pangulu_nnz, long *csr_rowptr, int *csr_colidx, double *csr_value, pangulu_init_options *init_options, void **pangulu_handle);
void pangulu_gstrf(pangulu_gstrf_options *gstrf_options, void **pangulu_handle);
void pangulu_gstrs(double *rhs, pangulu_gstrs_options *gstrs_options, void** pangulu_handle);
void pangulu_gssv(double *rhs, pangulu_gstrf_options *gstrf_options, pangulu_gstrs_options *gstrs_options, void **pangulu_handle);
void pangulu_init(int pangulu_n, long long pangulu_nnz, long *csr_rowptr, int *csr_colidx, double _Complex*csr_value, pangulu_init_options *init_options, void **pangulu_handle);
void pangulu_gstrf(pangulu_gstrf_options *gstrf_options, void **pangulu_handle);
void pangulu_gstrs(double _Complex *rhs, pangulu_gstrs_options *gstrs_options, void** pangulu_handle);
void pangulu_gssv(double _Complex *rhs, pangulu_gstrf_options *gstrf_options, pangulu_gstrs_options *gstrs_options, void **pangulu_handle);
#endif
