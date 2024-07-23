#ifndef PANGULU_CUDA_INTERFACE_H
#define PANGULU_CUDA_INTERFACE_H

#include "pangulu_common.h"
#include "platforms/02_GPU/01_CUDA/000_CUDA/pangulu_cuda.h"

#ifdef CHECK_TIME
#include "pangulu_time.h"
#endif

void pangulu_cuda_device_init(int_32t rank)
{
        int_32t gpu_num;
        pangulu_cuda_getDevicenum(&gpu_num);
        int_32t usr_id = pangulu_cuda_setDevice(gpu_num, rank);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, usr_id);
        if (rank == 0)
                printf(PANGULU_I_DEV_IS);
}

void pangulu_cuda_device_init_thread(int_32t rank)
{
        int_32t gpu_num;
        pangulu_cuda_getDevicenum(&gpu_num);
        pangulu_cuda_setDevice(gpu_num, rank);
}

void pangulu_cuda_free_interface(void *cuda_address)
{
        pangulu_cuda_free(cuda_address);
}

void pangulu_Smatrix_add_CUDA_memory(pangulu_Smatrix *S)
{
        pangulu_cuda_malloc((void **)&(S->CUDA_rowpointer), ((S->row) + 1) * sizeof(int_t));
        pangulu_cuda_malloc((void **)&(S->CUDA_columnindex), (S->nnz) * sizeof(pangulu_inblock_idx));
        pangulu_cuda_malloc((void **)&(S->CUDA_value), (S->nnz) * sizeof(calculate_type));
        pangulu_cuda_malloc((void **)&(S->CUDA_bin_rowpointer), BIN_LENGTH * sizeof(int_t));
        pangulu_cuda_malloc((void **)&(S->CUDA_bin_rowindex), (S->row) * sizeof(pangulu_inblock_idx));
}

void pangulu_Smatrix_CUDA_memory_init(pangulu_Smatrix *S, int_t NB, int_t nnz)
{
        S->row = NB;
        S->column = NB;
        S->nnz = nnz;
        pangulu_cuda_malloc((void **)&(S->CUDA_rowpointer), (NB + 1) * sizeof(int_t));
        pangulu_cuda_malloc((void **)&(S->CUDA_columnindex), nnz * sizeof(pangulu_inblock_idx));
        pangulu_cuda_malloc((void **)&(S->CUDA_value), nnz * sizeof(calculate_type));
}

void pangulu_Smatrix_add_CUDA_memory_U(pangulu_Smatrix *U)
{
        pangulu_cuda_malloc((void **)&(U->CUDA_nnzU), (U->row) * sizeof(int));
}

void pangulu_Smatrix_CUDA_memcpy_A(pangulu_Smatrix *S)
{
        pangulu_cuda_memcpy_host_to_device_int(S->CUDA_rowpointer, S->columnpointer, (S->row) + 1);
        pangulu_cuda_memcpy_host_to_device_int32(S->CUDA_columnindex, S->rowindex, S->nnz);
        pangulu_cuda_memcpy_host_to_device_value(S->CUDA_value, S->value_CSC, S->nnz);
        pangulu_cuda_memcpy_host_to_device_int(S->CUDA_bin_rowpointer, S->bin_rowpointer, BIN_LENGTH);
        pangulu_cuda_memcpy_host_to_device_int(S->CUDA_bin_rowindex, S->bin_rowindex, S->row);
}

void pangulu_Smatrix_CUDA_memcpy_struct_CSR(pangulu_Smatrix *calculate_S, pangulu_Smatrix *S)
{
        calculate_S->nnz = S->nnz;
        pangulu_cuda_memcpy_host_to_device_int(calculate_S->CUDA_rowpointer, S->rowpointer, (S->row) + 1);
        pangulu_cuda_memcpy_host_to_device_int32(calculate_S->CUDA_columnindex, S->columnindex, S->nnz);
}

void pangulu_Smatrix_CUDA_memcpy_struct_CSC(pangulu_Smatrix *calculate_S, pangulu_Smatrix *S)
{
        calculate_S->nnz = S->nnz;
        pangulu_cuda_memcpy_host_to_device_int(calculate_S->CUDA_rowpointer, S->columnpointer, (S->row) + 1);
        pangulu_cuda_memcpy_host_to_device_int32(calculate_S->CUDA_columnindex, S->rowindex, S->nnz);
}

void pangulu_Smatrix_CUDA_memcpy_complete_CSR(pangulu_Smatrix *calculate_S, pangulu_Smatrix *S)
{
        calculate_S->nnz = S->nnz;
#ifdef CHECK_TIME
        struct timeval GET_TIME_START;
        pangulu_time_check_begin(&GET_TIME_START);
#endif
        pangulu_cuda_memcpy_host_to_device_int(calculate_S->CUDA_rowpointer, S->rowpointer, (S->row) + 1);
        pangulu_cuda_memcpy_host_to_device_int32(calculate_S->CUDA_columnindex, S->columnindex, S->nnz);
        pangulu_cuda_memcpy_host_to_device_value(calculate_S->CUDA_value, S->value, S->nnz);
#ifdef CHECK_TIME
        TIME_cuda_memcpy += pangulu_time_check_end(&GET_TIME_START);
#endif
}

void pangulu_Smatrix_CUDA_memcpy_nnzU(pangulu_Smatrix *calculate_U, pangulu_Smatrix *U)
{
        pangulu_cuda_memcpy_host_to_device_int32(calculate_U->CUDA_nnzU, U->nnzU, calculate_U->row);
}

void pangulu_Smatrix_CUDA_memcpy_value_CSR(pangulu_Smatrix *S, pangulu_Smatrix *calculate_S)
{
#ifdef CHECK_TIME
        struct timeval GET_TIME_START;
        pangulu_time_check_begin(&GET_TIME_START);
#endif
        pangulu_cuda_memcpy_device_to_host_value(S->value, calculate_S->CUDA_value, S->nnz);

#ifdef CHECK_TIME

        TIME_cuda_memcpy += pangulu_time_check_end(&GET_TIME_START);
#endif
}

void pangulu_Smatrix_CUDA_memcpy_value_CSR_Async(pangulu_Smatrix *S, pangulu_Smatrix *calculate_S, cudaStream_t *stream)
{
#ifdef CHECK_TIME
        struct timeval GET_TIME_START;
        pangulu_time_check_begin(&GET_TIME_START);
#endif
        pangulu_cudaMemcpyAsync_device_to_host(S->value, calculate_S->CUDA_value, (S->nnz) * sizeof(calculate_type), stream);

#ifdef CHECK_TIME

        TIME_cuda_memcpy += pangulu_time_check_end(&GET_TIME_START);
#endif
}

void pangulu_Smatrix_CUDA_memcpy_value_CSC(pangulu_Smatrix *S, pangulu_Smatrix *calculate_S)
{
#ifdef CHECK_TIME
        struct timeval GET_TIME_START;
        pangulu_time_check_begin(&GET_TIME_START);
#endif
        pangulu_cuda_memcpy_device_to_host_value(S->value_CSC, calculate_S->CUDA_value, S->nnz);
#ifdef CHECK_TIME

        TIME_cuda_memcpy += pangulu_time_check_end(&GET_TIME_START);
#endif
}

void pangulu_Smatrix_CUDA_memcpy_value_CSC_Async(pangulu_Smatrix *S, pangulu_Smatrix *calculate_S, cudaStream_t *stream)
{
#ifdef CHECK_TIME
        struct timeval GET_TIME_START;
        pangulu_time_check_begin(&GET_TIME_START);
#endif
        pangulu_cudaMemcpyAsync_device_to_host(S->value_CSC, calculate_S->CUDA_value, (S->nnz) * sizeof(calculate_type), stream);
#ifdef CHECK_TIME

        TIME_cuda_memcpy += pangulu_time_check_end(&GET_TIME_START);
#endif
}

void pangulu_Smatrix_CUDA_memcpy_value_CSC_cal_length(pangulu_Smatrix *S, pangulu_Smatrix *calculate_S)
{

        pangulu_cuda_memcpy_device_to_host_value(S->value_CSC, calculate_S->CUDA_value, calculate_S->nnz);
}

void pangulu_Smatrix_CUDA_memcpy_to_device_value_CSC_Async(pangulu_Smatrix *calculate_S, pangulu_Smatrix *S, cudaStream_t *stream)
{
#ifdef CHECK_TIME
        struct timeval GET_TIME_START;
        pangulu_time_check_begin(&GET_TIME_START);
#endif

        pangulu_cudaMemcpyAsync_host_to_device(calculate_S->CUDA_value, S->value_CSC, sizeof(calculate_type) * (S->nnz), stream);
        // pangulu_cuda_memcpy_host_to_device_value(calculate_S->CUDA_value, S->value_CSC, S->nnz);

#ifdef CHECK_TIME

        TIME_cuda_memcpy += pangulu_time_check_end(&GET_TIME_START);
#endif
}

void pangulu_Smatrix_CUDA_memcpy_to_device_value_CSC(pangulu_Smatrix *calculate_S, pangulu_Smatrix *S)
{
#ifdef CHECK_TIME
        struct timeval GET_TIME_START;
        pangulu_time_check_begin(&GET_TIME_START);
#endif

        pangulu_cuda_memcpy_host_to_device_value(calculate_S->CUDA_value, S->value_CSC, S->nnz);

#ifdef CHECK_TIME

        TIME_cuda_memcpy += pangulu_time_check_end(&GET_TIME_START);
#endif
}

void pangulu_Smatrix_CUDA_memcpy_complete_CSR_Async(pangulu_Smatrix *calculate_S, pangulu_Smatrix *S, cudaStream_t *stream)
{
        calculate_S->nnz = S->nnz;
#ifdef CHECK_TIME
        struct timeval GET_TIME_START;
        pangulu_time_check_begin(&GET_TIME_START);
#endif
   
        pangulu_cudaMemcpyAsync_host_to_device(calculate_S->CUDA_rowpointer, S->rowpointer, sizeof(int_t) * ((S->row) + 1), stream);
        pangulu_cudaMemcpyAsync_host_to_device(calculate_S->CUDA_columnindex, S->columnindex, sizeof(int_32t) * S->nnz, stream);
        pangulu_cudaMemcpyAsync_host_to_device(calculate_S->CUDA_value, S->value, sizeof(calculate_type) * S->nnz, stream);

#ifdef CHECK_TIME
        TIME_cuda_memcpy += pangulu_time_check_end(&GET_TIME_START);
#endif
}

void pangulu_Smatrix_CUDA_memcpy_complete_CSC_Async(pangulu_Smatrix *calculate_S, pangulu_Smatrix *S, cudaStream_t *stream)
{
        calculate_S->nnz = S->nnz;
#ifdef CHECK_TIME
        struct timeval GET_TIME_START;
        pangulu_time_check_begin(&GET_TIME_START);
#endif
        pangulu_cudaMemcpyAsync_host_to_device(calculate_S->CUDA_rowpointer, S->columnpointer, sizeof(int_t) * ((S->row) + 1), stream);
        pangulu_cudaMemcpyAsync_host_to_device(calculate_S->CUDA_columnindex, S->rowindex, sizeof(int_32t) * S->nnz, stream);
        pangulu_cudaMemcpyAsync_host_to_device(calculate_S->CUDA_value, S->value_CSC, sizeof(calculate_type) * S->nnz, stream);

#ifdef CHECK_TIME

        TIME_cuda_memcpy += pangulu_time_check_end(&GET_TIME_START);
#endif
}

void pangulu_Smatrix_CUDA_memcpy_complete_CSC(pangulu_Smatrix *calculate_S, pangulu_Smatrix *S)
{
        calculate_S->nnz = S->nnz;
#ifdef CHECK_TIME
        struct timeval GET_TIME_START;
        pangulu_time_check_begin(&GET_TIME_START);
#endif
        pangulu_cuda_memcpy_host_to_device_int(calculate_S->CUDA_rowpointer, S->columnpointer, (S->row) + 1);
        pangulu_cuda_memcpy_host_to_device_int32(calculate_S->CUDA_columnindex, S->rowindex, S->nnz);
        pangulu_cuda_memcpy_host_to_device_value(calculate_S->CUDA_value, S->value_CSC, S->nnz);
#ifdef CHECK_TIME

        TIME_cuda_memcpy += pangulu_time_check_end(&GET_TIME_START);
#endif
}

#endif