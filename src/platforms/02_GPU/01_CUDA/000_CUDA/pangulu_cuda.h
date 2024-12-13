#ifndef PANGULU_CUDA_H
#define PANGULU_CUDA_H

#define warp_size 32
#define warp_per_block_gemm 2
#define warp_num_warplu 2
#define sm_len_warplev 96
#define warp_per_block 8
#define longrow_threshold 2048
#define sm_len_dgessm 1024
#define sm_len_dtstrf 1024
#define warp_per_block_dgessm 2
#define warp_per_block_dtstrf 2
#include "../../../../pangulu_common.h"

#if defined(PANGULU_COMPLEX)
#warning Complex on GPU is comming soon. Fallback to CPU.
#else // defined(PANGULU_COMPLEX)

#ifdef __cplusplus
extern "C"{
#endif
void pangulu_cudamemcpyasync_host_to_device(void *gpu_address, void *cpu_address, pangulu_int64_t size, cudaStream_t *stream);
void pangulu_cudamemcpyasync_device_to_host(void *cpu_address, void *gpu_address, pangulu_int64_t size, cudaStream_t *stream);

void pangulu_create_cudastream(cudaStream_t *stream);
void pangulu_destroy_cudastream(cudaStream_t *stream);
void pangulu_create_cudaevent(cudaEvent_t *event);
void pangulu_destroy_cudaevent(cudaEvent_t *event);
void pangulu_eventrecord(cudaEvent_t *event, cudaStream_t *stream);
void pangulu_eventsynchronize(cudaEvent_t *event);

void pangulu_cudamemcpyasync_device_to_device(void *gpu_address1, void *gpu_address2, pangulu_int64_t size, cudaStream_t *stream);
void pangulu_cudamemcpy_device_to_device(void *gpu_address1, void *gpu_address2, pangulu_int64_t size);

void pangulu_cuda_malloc(void **cuda_address, size_t size);
void pangulu_cuda_free(void *cuda_address);
void pangulu_cuda_memcpy_host_to_device_value(calculate_type *cpu_address, calculate_type *cuda_address, size_t size);
void pangulu_cuda_memcpy_device_to_host_value(calculate_type *cuda_address, calculate_type *cpu_address, size_t size);
void pangulu_cuda_devicesynchronize();
void pangulu_cuda_memcpy_host_to_device_inblock_idx(pangulu_inblock_idx *cuda_address, pangulu_inblock_idx *cpu_address, size_t size);
void pangulu_cuda_memcpy_host_to_device_inblock_ptr(pangulu_inblock_ptr *cuda_address, pangulu_inblock_ptr *cpu_address, size_t size);
void pangulu_cuda_memcpy_device_to_host_int(pangulu_inblock_ptr *cpu_address, pangulu_inblock_ptr *cuda_address, size_t size);
void pangulu_cuda_memcpy_host_to_device_pangulu_inblock_idx(pangulu_inblock_idx *cuda_address, pangulu_inblock_idx *cpu_address, size_t size);
void pangulu_cuda_memcpy_host_to_device_int32(pangulu_int32_t *cuda_address,
                                            pangulu_int32_t *cpu_address,
                                            size_t size);
void pangulu_cuda_getdevicenum(pangulu_int32_t *gpu_num);
pangulu_int32_t pangulu_cuda_setdevice(pangulu_int32_t gpu_num, pangulu_int32_t rank);
void pangulu_cuda_vector_add_kernel(pangulu_int64_t n, calculate_type *d_now_val_csc, calculate_type *d_old_val_csc);

pangulu_int64_t binarysearch_device(pangulu_int64_t *ridx, pangulu_int64_t left, pangulu_int64_t right, pangulu_int64_t target);

pangulu_inblock_idx binarysearch_idx(pangulu_inblock_idx *ridx, pangulu_int64_t left, pangulu_int64_t right, pangulu_inblock_idx target);

void triangle_pre(pangulu_inblock_idx *l_rowindex,
                  const pangulu_int64_t n,
                  const pangulu_int64_t nnzl,
                  int *d_graphindegree);

void pangulu_tstrf_cuda_kernel_v8(pangulu_int64_t n,
                                  pangulu_int64_t nnzu,
                                  int *degree,
                                  int *d_id_extractor,
                                  calculate_type *d_left_sum,
                                  pangulu_inblock_ptr *u_rowpointer,
                                  pangulu_inblock_idx *u_columnindex,
                                  calculate_type *u_value,
                                  pangulu_inblock_ptr *x_rowpointer,
                                  pangulu_inblock_idx *x_columnindex,
                                  calculate_type *x_value,
                                  pangulu_inblock_ptr *a_rowpointer,
                                  pangulu_inblock_idx *a_columnindex,
                                  calculate_type *a_value);

void pangulu_gessm_cuda_kernel_v9(pangulu_int64_t n,
                                  pangulu_int64_t nnzl,
                                  pangulu_int64_t num,
                                  pangulu_int64_t nnza,
                                  pangulu_inblock_ptr *d_spointer,
                                  int *d_graphindegree,
                                  int *d_id_extractor,
                                  int *d_while_profiler,
                                  pangulu_inblock_ptr *l_columnpointer,
                                  pangulu_inblock_idx *l_rowindex,
                                  calculate_type *l_value,
                                  pangulu_inblock_ptr *x_columnpointer,
                                  pangulu_inblock_idx *x_rowindex,
                                  calculate_type *x_value,
                                  pangulu_inblock_ptr *a_columnpointer,
                                  pangulu_inblock_idx *a_rowindex,
                                  calculate_type *a_value,
                                  calculate_type *d_left_sum,
                                  calculate_type *d_x,
                                  calculate_type *d_b);

void pangulu_tstrf_cuda_kernel_v9(pangulu_int64_t n,
                                  pangulu_int64_t nnzu,
                                  pangulu_int64_t num,
                                  pangulu_int64_t nnza,
                                  pangulu_inblock_ptr *d_spointer,
                                  int *d_graphindegree,
                                  int *d_id_extractor,
                                  int *d_while_profiler,
                                  pangulu_inblock_ptr *u_columnpointer,
                                  pangulu_inblock_idx *u_rowindex,
                                  calculate_type *u_value,
                                  pangulu_inblock_ptr *x_columnpointer,
                                  pangulu_inblock_idx *x_rowindex,
                                  calculate_type *x_value,
                                  pangulu_inblock_ptr *a_columnpointer,
                                  pangulu_inblock_idx *a_rowindex,
                                  calculate_type *a_value,
                                  calculate_type *d_left_sum,
                                  calculate_type *d_x,
                                  calculate_type *d_b);

void pangulu_gessm_cuda_kernel_v11(pangulu_int64_t n,
                                   pangulu_int64_t nnzl,
                                   pangulu_int64_t nnzx,
                                   int *d_graphindegree,
                                   int *d_id_extractor,
                                   calculate_type *d_left_sum,
                                   pangulu_inblock_ptr *l_columnpointer,
                                   pangulu_inblock_idx *l_rowindex,
                                   calculate_type *l_value,
                                   pangulu_inblock_ptr *x_columnpointer,
                                   pangulu_inblock_idx *x_rowindex,
                                   calculate_type *x_value,
                                   pangulu_inblock_ptr *a_columnpointer,
                                   pangulu_inblock_idx *a_rowindex,
                                   calculate_type *a_value);

// void pangulu_cuda_transpose_kernel_csc_to_csr(pangulu_int64_t nnz, calculate_type *d_val_csr, pangulu_int64_t *d_idx, calculate_type *d_val_csc); 

// void pangulu_cuda_transpose_kernel_csr_to_csc(pangulu_int64_t nnz, calculate_type *d_val_csc, pangulu_int64_t *d_idx, calculate_type *d_val_csr);

void pangulu_getrf_cuda_kernel(pangulu_int64_t n,
                               pangulu_int64_t nnza,
                               pangulu_int32_t *d_nnzu,
                               pangulu_inblock_ptr *a_cuda_rowpointer,
                               pangulu_inblock_idx *a_cuda_columnindex,
                               calculate_type *a_cuda_value,
                               pangulu_inblock_ptr *l_cuda_rowpointer,
                               pangulu_inblock_idx *l_cuda_columnindex,
                               calculate_type *l_cuda_value,
                               pangulu_inblock_ptr *u_cuda_rowpointer,
                               pangulu_inblock_idx *u_cuda_columnindex,
                               calculate_type *u_cuda_value);

void pangulu_getrf_cuda_dense_kernel(pangulu_int64_t n,
                                     pangulu_int64_t nnza,
                                     pangulu_int32_t *d_nnzu,
                                     pangulu_inblock_ptr *a_cuda_rowpointer,
                                     pangulu_inblock_idx *a_cuda_columnindex,
                                     calculate_type *a_cuda_value,
                                     pangulu_inblock_ptr *l_cuda_rowpointer,
                                     pangulu_inblock_idx *l_cuda_columnindex,
                                     calculate_type *l_cuda_value,
                                     pangulu_inblock_ptr *u_cuda_rowpointer,
                                     pangulu_inblock_idx *u_cuda_columnindex,
                                     calculate_type *u_cuda_value);

void pangulu_tstrf_cuda_kernel_v7(pangulu_int64_t n,
                                  pangulu_int64_t nnzu,
                                  pangulu_inblock_ptr *u_rowpointer,
                                  pangulu_inblock_idx *u_columnindex,
                                  calculate_type *u_value,
                                  pangulu_inblock_ptr *x_rowpointer,
                                  pangulu_inblock_idx *x_columnindex,
                                  calculate_type *x_value,
                                  pangulu_inblock_ptr *a_rowpointer,
                                  pangulu_inblock_idx *a_columnindex,
                                  calculate_type *a_value);

void pangulu_tstrf_cuda_kernel_v10(pangulu_int64_t n,
                                   pangulu_int64_t nnzu,
                                   pangulu_inblock_ptr *u_rowpointer,
                                   pangulu_inblock_idx *u_columnindex,
                                   calculate_type *u_value,
                                   pangulu_inblock_ptr *x_rowpointer,
                                   pangulu_inblock_idx *x_columnindex,
                                   calculate_type *x_value,
                                   pangulu_inblock_ptr *a_rowpointer,
                                   pangulu_inblock_idx *a_columnindex,
                                   calculate_type *a_value);

void pangulu_tstrf_cuda_kernel_v11(pangulu_int64_t n,
                                   pangulu_int64_t nnzu,
                                   int *degree,
                                   int *d_id_extractor,
                                   calculate_type *d_left_sum,
                                   pangulu_inblock_ptr *u_rowpointer,
                                   pangulu_inblock_idx *u_columnindex,
                                   calculate_type *u_value,
                                   pangulu_inblock_ptr *x_rowpointer,
                                   pangulu_inblock_idx *x_columnindex,
                                   calculate_type *x_value,
                                   pangulu_inblock_ptr *a_rowpointer,
                                   pangulu_inblock_idx *a_columnindex,
                                   calculate_type *a_value);

void pangulu_gessm_cuda_kernel_v7(pangulu_int64_t n,
                                  pangulu_int64_t nnzl,
                                  pangulu_inblock_ptr *l_columnpointer,
                                  pangulu_inblock_idx *l_rowindex,
                                  calculate_type *l_value,
                                  pangulu_inblock_ptr *x_columnpointer,
                                  pangulu_inblock_idx *x_rowindex,
                                  calculate_type *x_value,
                                  pangulu_inblock_ptr *a_columnpointer,
                                  pangulu_inblock_idx *a_rowindex,
                                  calculate_type *a_value);

void pangulu_gessm_cuda_kernel_v8(pangulu_int64_t n,
                                  pangulu_int64_t nnzl,
                                  pangulu_int64_t nnzx,
                                  int *d_graphindegree,
                                  int *d_id_extractor,
                                  calculate_type *d_left_sum,
                                  pangulu_inblock_ptr *l_columnpointer,
                                  pangulu_inblock_idx *l_rowindex,
                                  calculate_type *l_value,
                                  pangulu_inblock_ptr *x_columnpointer,
                                  pangulu_inblock_idx *x_rowindex,
                                  calculate_type *x_value,
                                  pangulu_inblock_ptr *a_columnpointer,
                                  pangulu_inblock_idx *a_rowindex,
                                  calculate_type *a_value);

void pangulu_gessm_cuda_kernel_v10(pangulu_int64_t n,
                                   pangulu_int64_t nnzl,
                                   pangulu_inblock_ptr *l_columnpointer,
                                   pangulu_inblock_idx *l_rowindex,
                                   calculate_type *l_value,
                                   pangulu_inblock_ptr *x_columnpointer,
                                   pangulu_inblock_idx *x_rowindex,
                                   calculate_type *x_value,
                                   pangulu_inblock_ptr *a_columnpointer,
                                   pangulu_inblock_idx *a_rowindex,
                                   calculate_type *a_value);

void pangulu_ssssm_cuda_kernel(pangulu_int64_t n,
                               pangulu_inblock_ptr *h_bin_rowpointer,
                               pangulu_inblock_ptr *d_bin_rowpointer,
                               pangulu_inblock_idx *d_bin_rowindex,
                               pangulu_inblock_ptr *d_rowptra,
                               pangulu_inblock_idx *d_colidxa,
                               calculate_type *d_valuea,
                               pangulu_inblock_ptr *d_rowptrb,
                               pangulu_inblock_idx *d_colidxb,
                               calculate_type *d_valueb,
                               pangulu_inblock_ptr *d_rowptrc,
                               pangulu_inblock_idx *d_colidxc,
                               calculate_type *d_valuec);

void pangulu_ssssm_dense_cuda_kernel(pangulu_int64_t n,
                                     pangulu_int64_t nnzc,
                                     pangulu_int64_t nnzb,
                                     pangulu_inblock_ptr *d_rowptra,
                                     pangulu_inblock_idx *d_colidxa,
                                     calculate_type *d_valuea,
                                     pangulu_inblock_ptr *d_rowptrb,
                                     pangulu_inblock_idx *d_colidxb,
                                     calculate_type *d_valueb,
                                     pangulu_inblock_ptr *d_rowptrc,
                                     pangulu_inblock_idx *d_colidxc,
                                     calculate_type *d_valuec);
#ifdef __cplusplus
}
#endif

#ifdef PANGULU_PLATFORM_ENV
__global__ void cuda_transform_s_to_d_col(pangulu_int64_t n,
                                          int stride,
                                          pangulu_inblock_ptr *d_rowptra,
                                          pangulu_inblock_idx *d_colidxa,
                                          calculate_type *d_valuea,
                                          calculate_type *temp_value_a);
__global__ void warplevel_sflu(pangulu_int64_t n,
                               pangulu_int32_t *d_nnzu,
                               pangulu_inblock_ptr *d_csccolptra,
                               pangulu_inblock_idx *d_cscrowidxa,
                               calculate_type *d_cscvaluea,
                               pangulu_inblock_ptr *d_csccolptrl,
                               pangulu_inblock_idx *d_cscrowidxl,
                               calculate_type *d_cscvaluel,
                               pangulu_inblock_ptr *d_csccolptru,
                               pangulu_inblock_idx *d_cscrowidxu,
                               calculate_type *d_cscvalueu);

__global__ void blocklevel_sflu_l1(pangulu_int64_t n,
                                   pangulu_int64_t *d_nnzu,
                                   pangulu_inblock_ptr *d_csccolptra,
                                   pangulu_inblock_idx *d_cscrowidxa,
                                   calculate_type *d_cscvaluea,
                                   pangulu_inblock_ptr *d_csccolptrl,
                                   pangulu_inblock_idx *d_cscrowidxl,
                                   calculate_type *d_cscvaluel,
                                   pangulu_inblock_ptr *d_csccolptru,
                                   pangulu_inblock_idx *d_cscrowidxu,
                                   calculate_type *d_cscvalueu);

__global__ void blocklevel_sflu_l2(pangulu_int64_t n,
                                   pangulu_int64_t *d_nnzu,
                                   pangulu_inblock_ptr *d_csccolptra,
                                   pangulu_inblock_idx *d_cscrowidxa,
                                   calculate_type *d_cscvaluea,
                                   pangulu_inblock_ptr *d_csccolptrl,
                                   pangulu_inblock_idx *d_cscrowidxl,
                                   calculate_type *d_cscvaluel,
                                   pangulu_inblock_ptr *d_csccolptru,
                                   pangulu_inblock_idx *d_cscrowidxu,
                                   calculate_type *d_cscvalueu);

__global__ void blocklevel_sflu_l3(pangulu_int64_t n,
                                   pangulu_int64_t *d_nnzu,
                                   pangulu_inblock_ptr *d_csccolptra,
                                   pangulu_inblock_idx *d_cscrowidxa,
                                   calculate_type *d_cscvaluea,
                                   pangulu_inblock_ptr *d_csccolptrl,
                                   pangulu_inblock_idx *d_cscrowidxl,
                                   calculate_type *d_cscvaluel,
                                   pangulu_inblock_ptr *d_csccolptru,
                                   pangulu_inblock_idx *d_cscrowidxu,
                                   calculate_type *d_cscvalueu);

__global__ void blocklevel_sflu_l4(pangulu_int64_t n,
                                   pangulu_int64_t *d_nnzu,
                                   pangulu_inblock_ptr *d_csccolptra,
                                   pangulu_inblock_idx *d_cscrowidxa,
                                   calculate_type *d_cscvaluea,
                                   pangulu_inblock_ptr *d_csccolptrl,
                                   pangulu_inblock_idx *d_cscrowidxl,
                                   calculate_type *d_cscvaluel,
                                   pangulu_inblock_ptr *d_csccolptru,
                                   pangulu_inblock_idx *d_cscrowidxu,
                                   calculate_type *d_cscvalueu);

// __global__ void trans_cuda_csc_to_csr(pangulu_int64_t n, calculate_type *d_val_csr, pangulu_int64_t *d_idx, calculate_type *d_val_csc);

// __global__ void trans_cuda_csr_to_csc(pangulu_int64_t n, calculate_type *d_val_csc, pangulu_int64_t *d_idx, calculate_type *d_val_csr);

__global__ void threadlevel_spgemm(pangulu_int64_t n,
                                   pangulu_int64_t layer,
                                   pangulu_inblock_ptr *d_bin_rowpointer,
                                   pangulu_inblock_idx *d_bin_rowindex,
                                   pangulu_inblock_ptr *d_rowptra,
                                   pangulu_inblock_idx *d_colidxa,
                                   calculate_type *d_valuea,
                                   pangulu_inblock_ptr *d_rowptrb,
                                   pangulu_inblock_idx *d_colidxb,
                                   calculate_type *d_valueb,
                                   pangulu_inblock_ptr *d_rowptrc,
                                   pangulu_inblock_idx *d_colidxc,
                                   calculate_type *d_valuec);

__global__ void warplevel_spgemm_32(pangulu_int64_t n,
                                    pangulu_int64_t layer,
                                    pangulu_inblock_ptr *d_bin_rowpointer,
                                    pangulu_inblock_idx *d_bin_rowindex,
                                    pangulu_inblock_ptr *d_rowptra,
                                    pangulu_inblock_idx *d_colidxa,
                                    calculate_type *d_valuea,
                                    pangulu_inblock_ptr *d_rowptrb,
                                    pangulu_inblock_idx *d_colidxb,
                                    calculate_type *d_valueb,
                                    pangulu_inblock_ptr *d_rowptrc,
                                    pangulu_inblock_idx *d_colidxc,
                                    calculate_type *d_valuec);

__global__ void warplevel_spgemm_64(pangulu_int64_t n,
                                    pangulu_int64_t layer,
                                    pangulu_inblock_ptr *d_bin_rowpointer,
                                    pangulu_inblock_idx *d_bin_rowindex,
                                    pangulu_inblock_ptr *d_rowptra,
                                    pangulu_inblock_idx *d_colidxa,
                                    calculate_type *d_valuea,
                                    pangulu_inblock_ptr *d_rowptrb,
                                    pangulu_inblock_idx *d_colidxb,
                                    calculate_type *d_valueb,
                                    pangulu_inblock_ptr *d_rowptrc,
                                    pangulu_inblock_idx *d_colidxc,
                                    calculate_type *d_valuec);

__global__ void warplevel_spgemm_128(pangulu_int64_t n,
                                     pangulu_int64_t layer,
                                     pangulu_inblock_ptr *d_bin_rowpointer,
                                     pangulu_inblock_idx *d_bin_rowindex,
                                     pangulu_inblock_ptr *d_rowptra,
                                     pangulu_inblock_idx *d_colidxa,
                                     calculate_type *d_valuea,
                                     pangulu_inblock_ptr *d_rowptrb,
                                     pangulu_inblock_idx *d_colidxb,
                                     calculate_type *d_valueb,
                                     pangulu_inblock_ptr *d_rowptrc,
                                     pangulu_inblock_idx *d_colidxc,
                                     calculate_type *d_valuec);

__global__ void blocklevel_spgemm_256(pangulu_int64_t n,
                                      pangulu_int64_t layer,
                                      pangulu_inblock_ptr *d_bin_rowpointer,
                                      pangulu_inblock_idx *d_bin_rowindex,
                                      pangulu_inblock_ptr *d_rowptra,
                                      pangulu_inblock_idx *d_colidxa,
                                      calculate_type *d_valuea,
                                      pangulu_inblock_ptr *d_rowptrb,
                                      pangulu_inblock_idx *d_colidxb,
                                      calculate_type *d_valueb,
                                      pangulu_inblock_ptr *d_rowptrc,
                                      pangulu_inblock_idx *d_colidxc,
                                      calculate_type *d_valuec);

__global__ void blocklevel_spgemm_512(pangulu_int64_t n,
                                      pangulu_int64_t layer,
                                      pangulu_inblock_ptr *d_bin_rowpointer,
                                      pangulu_inblock_idx *d_bin_rowindex,
                                      pangulu_inblock_ptr *d_rowptra,
                                      pangulu_inblock_idx *d_colidxa,
                                      calculate_type *d_valuea,
                                      pangulu_inblock_ptr *d_rowptrb,
                                      pangulu_inblock_idx *d_colidxb,
                                      calculate_type *d_valueb,
                                      pangulu_inblock_ptr *d_rowptrc,
                                      pangulu_inblock_idx *d_colidxc,
                                      calculate_type *d_valuec);

__global__ void blocklevel_spgemm_1024(pangulu_int64_t n,
                                       pangulu_int64_t layer,
                                       pangulu_inblock_ptr *d_bin_rowpointer,
                                       pangulu_inblock_idx *d_bin_rowindex,
                                       pangulu_inblock_ptr *d_rowptra,
                                       pangulu_inblock_idx *d_colidxa,
                                       calculate_type *d_valuea,
                                       pangulu_inblock_ptr *d_rowptrb,
                                       pangulu_inblock_idx *d_colidxb,
                                       calculate_type *d_valueb,
                                       pangulu_inblock_ptr *d_rowptrc,
                                       pangulu_inblock_idx *d_colidxc,
                                       calculate_type *d_valuec);

__global__ void blocklevel_spgemm_2048(pangulu_int64_t n,
                                       pangulu_int64_t layer,
                                       pangulu_inblock_ptr *d_bin_rowpointer,
                                       pangulu_inblock_idx *d_bin_rowindex,
                                       pangulu_inblock_ptr *d_rowptra,
                                       pangulu_inblock_idx *d_colidxa,
                                       calculate_type *d_valuea,
                                       pangulu_inblock_ptr *d_rowptrb,
                                       pangulu_inblock_idx *d_colidxb,
                                       calculate_type *d_valueb,
                                       pangulu_inblock_ptr *d_rowptrc,
                                       pangulu_inblock_idx *d_colidxc,
                                       calculate_type *d_valuec);

__global__ void blocklevel_spgemm_4097(pangulu_int64_t n,
                                       pangulu_int64_t layer,
                                       pangulu_inblock_ptr *d_bin_rowpointer,
                                       pangulu_inblock_idx *d_bin_rowindex,
                                       pangulu_inblock_ptr *d_rowptra,
                                       pangulu_inblock_idx *d_colidxa,
                                       calculate_type *d_valuea,
                                       pangulu_inblock_ptr *d_rowptrb,
                                       pangulu_inblock_idx *d_colidxb,
                                       calculate_type *d_valueb,
                                       pangulu_inblock_ptr *d_rowptrc,
                                       pangulu_inblock_idx *d_colidxc,
                                       calculate_type *d_valuec);

__forceinline__ __device__ calculate_type sum_32_shfl(calculate_type sum);

__global__ void gessm_kernel_v2(pangulu_int64_t n,
                                pangulu_inblock_ptr *l_columnpointer,
                                pangulu_inblock_idx *l_rowindex,
                                calculate_type *l_value,
                                pangulu_inblock_ptr *x_columnpointer,
                                pangulu_inblock_idx *x_rowindex,
                                calculate_type *x_value,
                                pangulu_inblock_ptr *a_columnpointer,
                                pangulu_inblock_idx *a_rowindex,
                                calculate_type *a_value);

__global__ void tstrf_kernel_v2(pangulu_int64_t n,
                                pangulu_inblock_ptr *u_rowpointer,
                                pangulu_inblock_idx *u_columnindex,
                                calculate_type *u_value,
                                pangulu_inblock_ptr *x_rowpointer,
                                pangulu_inblock_idx *x_columnindex,
                                calculate_type *x_value,
                                pangulu_inblock_ptr *a_rowpointer,
                                pangulu_inblock_idx *a_columnindex,
                                calculate_type *a_value);

__global__ void gessm_kernel_v3(pangulu_int64_t n,
                                pangulu_inblock_ptr *l_columnpointer,
                                pangulu_inblock_idx *l_rowindex,
                                calculate_type *l_value,
                                pangulu_inblock_ptr *x_columnpointer,
                                pangulu_inblock_idx *x_rowindex,
                                calculate_type *x_value,
                                pangulu_inblock_ptr *a_columnpointer,
                                pangulu_inblock_idx *a_rowindex,
                                calculate_type *a_value);

__global__ void tstrf_kernel_v3(pangulu_int64_t n,
                                pangulu_inblock_ptr *u_rowpointer,
                                pangulu_inblock_idx *u_columnindex,
                                calculate_type *u_value,
                                pangulu_inblock_ptr *x_rowpointer,
                                pangulu_inblock_idx *x_columnindex,
                                calculate_type *x_value,
                                pangulu_inblock_ptr *a_rowpointer,
                                pangulu_inblock_idx *a_columnindex,
                                calculate_type *a_value);

__global__ void wraplevel_spgemm_dense_nnz(pangulu_int64_t n,
                                           pangulu_int64_t nnz,
                                           pangulu_inblock_ptr *d_rowptra,
                                           pangulu_inblock_idx *d_colidxa,
                                           calculate_type *d_valuea,
                                           pangulu_inblock_ptr *d_rowptrb,
                                           pangulu_inblock_idx *d_colidxb,
                                           calculate_type *d_valueb,
                                           pangulu_inblock_ptr *d_rowptrc,
                                           pangulu_inblock_idx *d_colidxc,
                                           calculate_type *d_valuec,
                                           pangulu_int32_t *coo_col_b,
                                           calculate_type *temp_value_c);

#endif // PANGULU_PLATFORM_ENV
#endif // defined(PANGULU_COMPLEX)
#endif // PANGULU_CUDA_H