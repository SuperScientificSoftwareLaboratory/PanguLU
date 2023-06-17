#ifndef PANGULU_CUDA_H
#define PANGULU_CUDA_H

#define WARP_SIZE 32
#define WARP_PER_BLOCK_GEMM 2
#define WARP_NUM_WARPLU 2
#define SM_LEN_WARPLEV 96
#define WARP_PER_BLOCK 8
#define LONGROW_THRESHOLD 2048
#define SM_LEN_DGESSM 1024
#define SM_LEN_DTSTRF 1024
#define WARP_PER_BLOCK_DGESSM 2
#define WARP_PER_BLOCK_DTSTRF 2
#include "pangulu_common.h"

void pangulu_cudaMemcpyAsync_host_to_device(void *gpu_address, void *cpu_address, int_t size, cudaStream_t *stream);
void pangulu_cudaMemcpyAsync_device_to_host(void *cpu_address, void *gpu_address, int_t size, cudaStream_t *stream);

void pangulu_create_cudastream(cudaStream_t *stream);
void pangulu_destroy_cudastream(cudaStream_t *stream);
void pangulu_create_cudaevent(cudaEvent_t *event);
void pangulu_destroy_cudaevent(cudaEvent_t *event);
void pangulu_eventrecord(cudaEvent_t *event, cudaStream_t *stream);
void pangulu_eventsynchronize(cudaEvent_t *event);

void pangulu_cudamemcpyAsync_device_to_device(void *gpu_address1, void *gpu_address2, int_t size, cudaStream_t *stream);
void pangulu_cudamemcpy_device_to_device(void *gpu_address1, void *gpu_address2, int_t size);

void pangulu_cuda_malloc(void **cuda_address, size_t size);
void pangulu_cuda_free(void *cuda_address);
void pangulu_cuda_memcpy_host_to_device_value(calculate_type *cpu_address, calculate_type *cuda_address, size_t size);
void pangulu_cuda_memcpy_device_to_host_value(calculate_type *cuda_address, calculate_type *cpu_address, size_t size);
void pangulu_cuda_devicesynchronize();
void pangulu_cuda_memcpy_host_to_device_int(int_t *cuda_address, int_t *cpu_address, size_t size);
void pangulu_cuda_memcpy_device_to_host_int(int_t *cpu_address, int_t *cuda_address, size_t size);
void pangulu_cuda_memcpy_host_to_device_int32(int_32t *cuda_address, int_32t *cpu_address, size_t size);
void pangulu_cuda_getDevicenum(int_32t *gpu_num);
int_32t pangulu_cuda_setDevice(int_32t gpu_num, int_32t rank);
void pangulu_cuda_vector_add_kernel(int_t n, calculate_type *d_now_val_csc, calculate_type *d_old_val_csc);
__global__ void cuda_Transform_s_to_d_col(int_t n,
                                          int stride,
                                          int_t *d_rowPtrA,
                                          idx_int *d_colIdxA,
                                          calculate_type *d_valueA,
                                          calculate_type *temp_value_A);
__global__ void WarpLevel_sflu(int_t n,
                               int_t *d_nnzU,
                               int_t *d_cscColPtrA,
                               idx_int *d_cscRowIdxA,
                               calculate_type *d_cscValueA,
                               int_t *d_cscColPtrL,
                               idx_int *d_cscRowIdxL,
                               calculate_type *d_cscValueL,
                               int_t *d_cscColPtrU,
                               idx_int *d_cscRowIdxU,
                               calculate_type *d_cscValueU);

__global__ void BlockLevel_sflu_L1(int_t n,
                                   int_t *d_nnzU,
                                   int_t *d_cscColPtrA,
                                   idx_int *d_cscRowIdxA,
                                   calculate_type *d_cscValueA,
                                   int_t *d_cscColPtrL,
                                   idx_int *d_cscRowIdxL,
                                   calculate_type *d_cscValueL,
                                   int_t *d_cscColPtrU,
                                   idx_int *d_cscRowIdxU,
                                   calculate_type *d_cscValueU);

__global__ void BlockLevel_sflu_L2(int_t n,
                                   int_t *d_nnzU,
                                   int_t *d_cscColPtrA,
                                   idx_int *d_cscRowIdxA,
                                   calculate_type *d_cscValueA,
                                   int_t *d_cscColPtrL,
                                   idx_int *d_cscRowIdxL,
                                   calculate_type *d_cscValueL,
                                   int_t *d_cscColPtrU,
                                   idx_int *d_cscRowIdxU,
                                   calculate_type *d_cscValueU);

__global__ void BlockLevel_sflu_L3(int_t n,
                                   int_t *d_nnzU,
                                   int_t *d_cscColPtrA,
                                   idx_int *d_cscRowIdxA,
                                   calculate_type *d_cscValueA,
                                   int_t *d_cscColPtrL,
                                   idx_int *d_cscRowIdxL,
                                   calculate_type *d_cscValueL,
                                   int_t *d_cscColPtrU,
                                   idx_int *d_cscRowIdxU,
                                   calculate_type *d_cscValueU);

__global__ void BlockLevel_sflu_L4(int_t n,
                                   int_t *d_nnzU,
                                   int_t *d_cscColPtrA,
                                   idx_int *d_cscRowIdxA,
                                   calculate_type *d_cscValueA,
                                   int_t *d_cscColPtrL,
                                   idx_int *d_cscRowIdxL,
                                   calculate_type *d_cscValueL,
                                   int_t *d_cscColPtrU,
                                   idx_int *d_cscRowIdxU,
                                   calculate_type *d_cscValueU);

int_t binarySearch(int_t *ridx, int_t left, int_t right, int_t target);

idx_int binarySearch_idx(idx_int *ridx, int_t left, int_t right, idx_int target);

__global__ void trans_cuda_CSC_to_CSR(int_t n, calculate_type *d_val_csr, int_t *d_idx, calculate_type *d_val_csc);

__global__ void trans_cuda_CSR_to_CSC(int_t n, calculate_type *d_val_csc, int_t *d_idx, calculate_type *d_val_csr);

__global__ void ThreadLevel_spgemm(int_t n,
                                   int_t layer,
                                   int_t *d_bin_rowpointer,
                                   int_t *d_bin_rowindex,
                                   int_t *d_rowPtrA,
                                   idx_int *d_colIdxA,
                                   calculate_type *d_valueA,
                                   int_t *d_rowPtrB,
                                   idx_int *d_colIdxB,
                                   calculate_type *d_valueB,
                                   int_t *d_rowPtrC,
                                   idx_int *d_colIdxC,
                                   calculate_type *d_valueC);

__global__ void WarpLevel_spgemm_32(int_t n,
                                    int_t layer,
                                    int_t *d_bin_rowpointer,
                                    int_t *d_bin_rowindex,
                                    int_t *d_rowPtrA,
                                    idx_int *d_colIdxA,
                                    calculate_type *d_valueA,
                                    int_t *d_rowPtrB,
                                    idx_int *d_colIdxB,
                                    calculate_type *d_valueB,
                                    int_t *d_rowPtrC,
                                    idx_int *d_colIdxC,
                                    calculate_type *d_valueC);

__global__ void WarpLevel_spgemm_64(int_t n,
                                    int_t layer,
                                    int_t *d_bin_rowpointer,
                                    int_t *d_bin_rowindex,
                                    int_t *d_rowPtrA,
                                    idx_int *d_colIdxA,
                                    calculate_type *d_valueA,
                                    int_t *d_rowPtrB,
                                    idx_int *d_colIdxB,
                                    calculate_type *d_valueB,
                                    int_t *d_rowPtrC,
                                    idx_int *d_colIdxC,
                                    calculate_type *d_valueC);

__global__ void WarpLevel_spgemm_128(int_t n,
                                     int_t layer,
                                     int_t *d_bin_rowpointer,
                                     int_t *d_bin_rowindex,
                                     int_t *d_rowPtrA,
                                     idx_int *d_colIdxA,
                                     calculate_type *d_valueA,
                                     int_t *d_rowPtrB,
                                     idx_int *d_colIdxB,
                                     calculate_type *d_valueB,
                                     int_t *d_rowPtrC,
                                     idx_int *d_colIdxC,
                                     calculate_type *d_valueC);

__global__ void BlockLevel_spgemm_256(int_t n,
                                      int_t layer,
                                      int_t *d_bin_rowpointer,
                                      int_t *d_bin_rowindex,
                                      int_t *d_rowPtrA,
                                      idx_int *d_colIdxA,
                                      calculate_type *d_valueA,
                                      int_t *d_rowPtrB,
                                      idx_int *d_colIdxB,
                                      calculate_type *d_valueB,
                                      int_t *d_rowPtrC,
                                      idx_int *d_colIdxC,
                                      calculate_type *d_valueC);

__global__ void BlockLevel_spgemm_512(int_t n,
                                      int_t layer,
                                      int_t *d_bin_rowpointer,
                                      int_t *d_bin_rowindex,
                                      int_t *d_rowPtrA,
                                      idx_int *d_colIdxA,
                                      calculate_type *d_valueA,
                                      int_t *d_rowPtrB,
                                      idx_int *d_colIdxB,
                                      calculate_type *d_valueB,
                                      int_t *d_rowPtrC,
                                      idx_int *d_colIdxC,
                                      calculate_type *d_valueC);

__global__ void BlockLevel_spgemm_1024(int_t n,
                                       int_t layer,
                                       int_t *d_bin_rowpointer,
                                       int_t *d_bin_rowindex,
                                       int_t *d_rowPtrA,
                                       idx_int *d_colIdxA,
                                       calculate_type *d_valueA,
                                       int_t *d_rowPtrB,
                                       idx_int *d_colIdxB,
                                       calculate_type *d_valueB,
                                       int_t *d_rowPtrC,
                                       idx_int *d_colIdxC,
                                       calculate_type *d_valueC);

__global__ void BlockLevel_spgemm_2048(int_t n,
                                       int_t layer,
                                       int_t *d_bin_rowpointer,
                                       int_t *d_bin_rowindex,
                                       int_t *d_rowPtrA,
                                       idx_int *d_colIdxA,
                                       calculate_type *d_valueA,
                                       int_t *d_rowPtrB,
                                       idx_int *d_colIdxB,
                                       calculate_type *d_valueB,
                                       int_t *d_rowPtrC,
                                       idx_int *d_colIdxC,
                                       calculate_type *d_valueC);

__global__ void BlockLevel_spgemm_4097(int_t n,
                                       int_t layer,
                                       int_t *d_bin_rowpointer,
                                       int_t *d_bin_rowindex,
                                       int_t *d_rowPtrA,
                                       idx_int *d_colIdxA,
                                       calculate_type *d_valueA,
                                       int_t *d_rowPtrB,
                                       idx_int *d_colIdxB,
                                       calculate_type *d_valueB,
                                       int_t *d_rowPtrC,
                                       idx_int *d_colIdxC,
                                       calculate_type *d_valueC);

__forceinline__ __device__ calculate_type sum_32_shfl(calculate_type sum);

__global__ void GESSM_Kernel_v2(int_t n,
                                int_t *L_columnpointer,
                                idx_int *L_rowindex,
                                calculate_type *L_VALUE,
                                int_t *X_columnpointer,
                                idx_int *X_rowindex,
                                calculate_type *X_VALUE,
                                int_t *A_columnpointer,
                                idx_int *A_rowindex,
                                calculate_type *A_VALUE);

__global__ void TSTRF_Kernel_v2(int_t n,
                                int_t *U_rowpointer,
                                idx_int *U_columnindex,
                                calculate_type *U_VALUE,
                                int_t *X_rowpointer,
                                idx_int *X_columnindex,
                                calculate_type *X_VALUE,
                                int_t *A_rowpointer,
                                idx_int *A_columnindex,
                                calculate_type *A_VALUE);

__global__ void GESSM_Kernel_v3(int_t n,
                                int_t *L_columnpointer,
                                idx_int *L_rowindex,
                                calculate_type *L_VALUE,
                                int_t *X_columnpointer,
                                idx_int *X_rowindex,
                                calculate_type *X_VALUE,
                                int_t *A_columnpointer,
                                idx_int *A_rowindex,
                                calculate_type *A_VALUE);

__global__ void TSTRF_Kernel_v3(int_t n,
                                int_t *U_rowpointer,
                                idx_int *U_columnindex,
                                calculate_type *U_VALUE,
                                int_t *X_rowpointer,
                                idx_int *X_columnindex,
                                calculate_type *X_VALUE,
                                int_t *A_rowpointer,
                                idx_int *A_columnindex,
                                calculate_type *A_VALUE);

void TRIANGLE_PRE(idx_int *L_rowindex,
                  const int_t n,
                  const int_t nnzL,
                  int *d_graphInDegree);

void pangulu_tstrf_cuda_kernel_v8(int_t n,
                                  int_t nnzU,
                                  int *degree,
                                  int *d_id_extractor,
                                  calculate_type *d_left_sum,
                                  int_t *U_rowpointer,
                                  idx_int *U_columnindex,
                                  calculate_type *U_VALUE,
                                  int_t *X_rowpointer,
                                  idx_int *X_columnindex,
                                  calculate_type *X_VALUE,
                                  int_t *A_rowpointer,
                                  idx_int *A_columnindex,
                                  calculate_type *A_VALUE);

void pangulu_gessm_cuda_kernel_v9(int_t n,
                                  int_t nnzL,
                                  int_t num,
                                  int_t nnzA,
                                  int_t *d_Spointer,
                                  int *d_graphInDegree,
                                  int *d_id_extractor,
                                  int *d_while_profiler,
                                  int_t *L_columnpointer,
                                  idx_int *L_rowindex,
                                  calculate_type *L_VALUE,
                                  int_t *X_columnpointer,
                                  idx_int *X_rowindex,
                                  calculate_type *X_VALUE,
                                  int_t *A_columnpointer,
                                  idx_int *A_rowindex,
                                  calculate_type *A_VALUE,
                                  calculate_type *d_left_sum,
                                  calculate_type *d_x,
                                  calculate_type *d_b);

void pangulu_tstrf_cuda_kernel_v9(int_t n,
                                  int_t nnzU,
                                  int_t num,
                                  int_t nnzA,
                                  int_t *d_Spointer,
                                  int *d_graphInDegree,
                                  int *d_id_extractor,
                                  int *d_while_profiler,
                                  int_t *U_columnpointer,
                                  idx_int *U_rowindex,
                                  calculate_type *U_VALUE,
                                  int_t *X_columnpointer,
                                  idx_int *X_rowindex,
                                  calculate_type *X_VALUE,
                                  int_t *A_columnpointer,
                                  idx_int *A_rowindex,
                                  calculate_type *A_VALUE,
                                  calculate_type *d_left_sum,
                                  calculate_type *d_x,
                                  calculate_type *d_b);

void pangulu_gessm_cuda_kernel_v11(int_t n,
                                   int_t nnzL,
                                   int_t nnzX,
                                   int *d_graphInDegree,
                                   int *d_id_extractor,
                                   calculate_type *d_left_sum,
                                   int_t *L_columnpointer,
                                   idx_int *L_rowindex,
                                   calculate_type *L_VALUE,
                                   int_t *X_columnpointer,
                                   idx_int *X_rowindex,
                                   calculate_type *X_VALUE,
                                   int_t *A_columnpointer,
                                   idx_int *A_rowindex,
                                   calculate_type *A_VALUE);

__global__ void WrapLevel_spgemm_dense_nnz(int_t n,
                                           int_t nnz,
                                           int_t *d_rowPtrA,
                                           idx_int *d_colIdxA,
                                           calculate_type *d_valueA,
                                           int_t *d_rowPtrB,
                                           idx_int *d_colIdxB,
                                           calculate_type *d_valueB,
                                           int_t *d_rowPtrC,
                                           idx_int *d_colIdxC,
                                           calculate_type *d_valueC,
                                           idx_int *coo_col_B,
                                           calculate_type *temp_value_C);

void pangulu_cuda_transport_kernel_CSC_to_CSR(int_t nnz, calculate_type *d_val_csr, int_t *d_idx, calculate_type *d_val_csc);

void pangulu_cuda_transport_kernel_CSR_to_CSC(int_t nnz, calculate_type *d_val_csc, int_t *d_idx, calculate_type *d_val_csr);

void pangulu_getrf_cuda_kernel(int_t n,
                               int_t nnzA,
                               int_32t *d_nnzU,
                               int_t *A_CUDA_rowpointer,
                               idx_int *A_CUDA_columnindex,
                               calculate_type *A_CUDA_value,
                               int_t *L_CUDA_rowpointer,
                               idx_int *L_CUDA_columnindex,
                               calculate_type *L_CUDA_value,
                               int_t *U_CUDA_rowpointer,
                               idx_int *U_CUDA_columnindex,
                               calculate_type *U_CUDA_value);

void pangulu_getrf_cuda_dense_kernel(int_t n,
                                     int_t nnzA,
                                     int_32t *d_nnzU,
                                     int_t *A_CUDA_rowpointer,
                                     idx_int *A_CUDA_columnindex,
                                     calculate_type *A_CUDA_value,
                                     int_t *L_CUDA_rowpointer,
                                     idx_int *L_CUDA_columnindex,
                                     calculate_type *L_CUDA_value,
                                     int_t *U_CUDA_rowpointer,
                                     idx_int *U_CUDA_columnindex,
                                     calculate_type *U_CUDA_value);

void pangulu_tstrf_cuda_kernel_v7(int_t n,
                                  int_t nnzU,
                                  int_t *U_rowpointer,
                                  idx_int *U_columnindex,
                                  calculate_type *U_VALUE,
                                  int_t *X_rowpointer,
                                  idx_int *X_columnindex,
                                  calculate_type *X_VALUE,
                                  int_t *A_rowpointer,
                                  idx_int *A_columnindex,
                                  calculate_type *A_VALUE);

void pangulu_tstrf_cuda_kernel_v10(int_t n,
                                   int_t nnzU,
                                   int_t *U_rowpointer,
                                   idx_int *U_columnindex,
                                   calculate_type *U_VALUE,
                                   int_t *X_rowpointer,
                                   idx_int *X_columnindex,
                                   calculate_type *X_VALUE,
                                   int_t *A_rowpointer,
                                   idx_int *A_columnindex,
                                   calculate_type *A_VALUE);

void pangulu_tstrf_cuda_kernel_v11(int_t n,
                                   int_t nnzU,
                                   int *degree,
                                   int *d_id_extractor,
                                   calculate_type *d_left_sum,
                                   int_t *U_rowpointer,
                                   idx_int *U_columnindex,
                                   calculate_type *U_VALUE,
                                   int_t *X_rowpointer,
                                   idx_int *X_columnindex,
                                   calculate_type *X_VALUE,
                                   int_t *A_rowpointer,
                                   idx_int *A_columnindex,
                                   calculate_type *A_VALUE);

void pangulu_gessm_cuda_kernel_v7(int_t n,
                                  int_t nnzL, int_t *L_columnpointer,
                                  idx_int *L_rowindex,
                                  calculate_type *L_VALUE,
                                  int_t *X_columnpointer,
                                  idx_int *X_rowindex,
                                  calculate_type *X_VALUE,
                                  int_t *A_columnpointer,
                                  idx_int *A_rowindex,
                                  calculate_type *A_VALUE);

void pangulu_gessm_cuda_kernel_v8(int_t n,
                                  int_t nnzL,
                                  int_t nnzX,
                                  int *d_graphInDegree,
                                  int *d_id_extractor,
                                  calculate_type *d_left_sum,
                                  int_t *L_columnpointer,
                                  idx_int *L_rowindex,
                                  calculate_type *L_VALUE,
                                  int_t *X_columnpointer,
                                  idx_int *X_rowindex,
                                  calculate_type *X_VALUE,
                                  int_t *A_columnpointer,
                                  idx_int *A_rowindex,
                                  calculate_type *A_VALUE);

void pangulu_gessm_cuda_kernel_v10(int_t n,
                                   int_t nnzL,
                                   int_t *L_columnpointer,
                                   idx_int *L_rowindex,
                                   calculate_type *L_VALUE,
                                   int_t *X_columnpointer,
                                   idx_int *X_rowindex,
                                   calculate_type *X_VALUE,
                                   int_t *A_columnpointer,
                                   idx_int *A_rowindex,
                                   calculate_type *A_VALUE);

void pangulu_ssssm_cuda_kernel(int_t n,
                               int_t *h_bin_rowpointer,
                               int_t *d_bin_rowpointer,
                               int_t *d_bin_rowindex,
                               int_t *d_rowPtrA,
                               idx_int *d_colIdxA,
                               calculate_type *d_valueA,
                               int_t *d_rowPtrB,
                               idx_int *d_colIdxB,
                               calculate_type *d_valueB,
                               int_t *d_rowPtrC,
                               idx_int *d_colIdxC,
                               calculate_type *d_valueC);

void pangulu_ssssm_dense_cuda_kernel(int_t n,
                                     int_t nnzC,
                                     int_t nnzB,
                                     int_t *d_rowPtrA,
                                     idx_int *d_colIdxA,
                                     calculate_type *d_valueA,
                                     int_t *d_rowPtrB,
                                     idx_int *d_colIdxB,
                                     calculate_type *d_valueB,
                                     int_t *d_rowPtrC,
                                     idx_int *d_colIdxC,
                                     calculate_type *d_valueC);

#endif