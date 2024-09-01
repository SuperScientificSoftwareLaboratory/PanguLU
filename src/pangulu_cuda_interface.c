#include "pangulu_common.h"

#ifdef GPU_OPEN
void pangulu_cuda_device_init(pangulu_int32_t rank)
{
        pangulu_int32_t gpu_num;
        pangulu_cuda_getdevicenum(&gpu_num);
        pangulu_int32_t usr_id = pangulu_cuda_setdevice(gpu_num, rank);
        struct cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, usr_id);
        if (rank == 0)
                printf(PANGULU_I_DEV_IS);
}

void pangulu_cuda_device_init_thread(pangulu_int32_t rank)
{
        pangulu_int32_t gpu_num;
        pangulu_cuda_getdevicenum(&gpu_num);
        pangulu_cuda_setdevice(gpu_num, rank);
}

void pangulu_cuda_free_interface(void *cuda_address)
{
        pangulu_cuda_free(cuda_address);
}

void pangulu_smatrix_add_cuda_memory(pangulu_smatrix *s)
{
        pangulu_cuda_malloc((void **)&(s->cuda_rowpointer), ((s->row) + 1) * sizeof(pangulu_int64_t));
        pangulu_cuda_malloc((void **)&(s->cuda_columnindex), (s->nnz) * sizeof(pangulu_inblock_idx));
        pangulu_cuda_malloc((void **)&(s->cuda_value), (s->nnz) * sizeof(calculate_type));
        pangulu_cuda_malloc((void **)&(s->cuda_bin_rowpointer), BIN_LENGTH * sizeof(pangulu_int64_t));
        pangulu_cuda_malloc((void **)&(s->cuda_bin_rowindex), (s->row) * sizeof(pangulu_inblock_idx));
}

void pangulu_smatrix_cuda_memory_init(pangulu_smatrix *s, pangulu_int64_t nb, pangulu_int64_t nnz)
{
        s->row = nb;
        s->column = nb;
        s->nnz = nnz;
        pangulu_cuda_malloc((void **)&(s->cuda_rowpointer), (nb + 1) * sizeof(pangulu_int64_t));
        pangulu_cuda_malloc((void **)&(s->cuda_columnindex), nnz * sizeof(pangulu_inblock_idx));
        pangulu_cuda_malloc((void **)&(s->cuda_value), nnz * sizeof(calculate_type));
}

void pangulu_smatrix_add_cuda_memory_u(pangulu_smatrix *u)
{
        pangulu_cuda_malloc((void **)&(u->cuda_nnzu), (u->row) * sizeof(int));
}

void pangulu_smatrix_cuda_memcpy_a(pangulu_smatrix *s)
{
        pangulu_cuda_memcpy_host_to_device_inblock_ptr(s->cuda_rowpointer, s->columnpointer, (s->row) + 1);
        pangulu_cuda_memcpy_host_to_device_inblock_idx(s->cuda_columnindex, s->rowindex, s->nnz);
        pangulu_cuda_memcpy_host_to_device_value(s->cuda_value, s->value_csc, s->nnz);
        pangulu_cuda_memcpy_host_to_device_inblock_ptr(s->cuda_bin_rowpointer, s->bin_rowpointer, BIN_LENGTH);
        pangulu_cuda_memcpy_host_to_device_inblock_idx(s->cuda_bin_rowindex, s->bin_rowindex, s->row);
}

void pangulu_smatrix_cuda_memcpy_struct_csr(pangulu_smatrix *calculate_s, pangulu_smatrix *s)
{
        calculate_s->nnz = s->nnz;
        pangulu_cuda_memcpy_host_to_device_inblock_ptr(calculate_s->cuda_rowpointer, s->rowpointer, (s->row) + 1);
        pangulu_cuda_memcpy_host_to_device_inblock_idx(calculate_s->cuda_columnindex, s->columnindex, s->nnz);
}

void pangulu_smatrix_cuda_memcpy_struct_csc(pangulu_smatrix *calculate_s, pangulu_smatrix *s)
{
        calculate_s->nnz = s->nnz;
        pangulu_cuda_memcpy_host_to_device_inblock_ptr(calculate_s->cuda_rowpointer, s->columnpointer, (s->row) + 1);
        pangulu_cuda_memcpy_host_to_device_inblock_idx(calculate_s->cuda_columnindex, s->rowindex, s->nnz);
}

void pangulu_smatrix_cuda_memcpy_complete_csr(pangulu_smatrix *calculate_s, pangulu_smatrix *s)
{
        calculate_s->nnz = s->nnz;
#ifdef check_time
        struct timeval get_time_start;
        pangulu_time_check_begin(&get_time_start);
#endif
        pangulu_cuda_memcpy_host_to_device_inblock_ptr(calculate_s->cuda_rowpointer, s->rowpointer, (s->row) + 1);
        pangulu_cuda_memcpy_host_to_device_inblock_idx(calculate_s->cuda_columnindex, s->columnindex, s->nnz);
        pangulu_cuda_memcpy_host_to_device_value(calculate_s->cuda_value, s->value, s->nnz);
#ifdef check_time
        time_cuda_memcpy += pangulu_time_check_end(&get_time_start);
#endif
}

void pangulu_smatrix_cuda_memcpy_nnzu(pangulu_smatrix *calculate_u, pangulu_smatrix *u)
{
        pangulu_cuda_memcpy_host_to_device_int32(calculate_u->cuda_nnzu, u->nnzu, calculate_u->row);
}

void pangulu_smatrix_cuda_memcpy_value_csr(pangulu_smatrix *s, pangulu_smatrix *calculate_s)
{
#ifdef check_time
        struct timeval get_time_start;
        pangulu_time_check_begin(&get_time_start);
#endif
        pangulu_cuda_memcpy_device_to_host_value(s->value, calculate_s->cuda_value, s->nnz);

#ifdef check_time

        time_cuda_memcpy += pangulu_time_check_end(&get_time_start);
#endif
}

void pangulu_smatrix_cuda_memcpy_value_csr_async(pangulu_smatrix *s, pangulu_smatrix *calculate_s, cudaStream_t *stream)
{
#ifdef check_time
        struct timeval get_time_start;
        pangulu_time_check_begin(&get_time_start);
#endif
        pangulu_cudamemcpyasync_device_to_host(s->value, calculate_s->cuda_value, (s->nnz) * sizeof(calculate_type), stream);

#ifdef check_time

        time_cuda_memcpy += pangulu_time_check_end(&get_time_start);
#endif
}

void pangulu_smatrix_cuda_memcpy_value_csc(pangulu_smatrix *s, pangulu_smatrix *calculate_s)
{
#ifdef check_time
        struct timeval get_time_start;
        pangulu_time_check_begin(&get_time_start);
#endif
        pangulu_cuda_memcpy_device_to_host_value(s->value_csc, calculate_s->cuda_value, s->nnz);
#ifdef check_time

        time_cuda_memcpy += pangulu_time_check_end(&get_time_start);
#endif
}

void pangulu_smatrix_cuda_memcpy_value_csc_async(pangulu_smatrix *s, pangulu_smatrix *calculate_s, cudaStream_t *stream)
{
#ifdef check_time
        struct timeval get_time_start;
        pangulu_time_check_begin(&get_time_start);
#endif
        pangulu_cudamemcpyasync_device_to_host(s->value_csc, calculate_s->cuda_value, (s->nnz) * sizeof(calculate_type), stream);
#ifdef check_time

        time_cuda_memcpy += pangulu_time_check_end(&get_time_start);
#endif
}

void pangulu_smatrix_cuda_memcpy_value_csc_cal_length(pangulu_smatrix *s, pangulu_smatrix *calculate_s)
{

        pangulu_cuda_memcpy_device_to_host_value(s->value_csc, calculate_s->cuda_value, calculate_s->nnz);
}

void pangulu_smatrix_cuda_memcpy_to_device_value_csc_async(pangulu_smatrix *calculate_s, pangulu_smatrix *s, cudaStream_t *stream)
{
#ifdef check_time
        struct timeval get_time_start;
        pangulu_time_check_begin(&get_time_start);
#endif

        pangulu_cudamemcpyasync_host_to_device(calculate_s->cuda_value, s->value_csc, sizeof(calculate_type) * (s->nnz), stream);

#ifdef check_time

        time_cuda_memcpy += pangulu_time_check_end(&get_time_start);
#endif
}

void pangulu_smatrix_cuda_memcpy_to_device_value_csc(pangulu_smatrix *calculate_s, pangulu_smatrix *s)
{
#ifdef check_time
        struct timeval get_time_start;
        pangulu_time_check_begin(&get_time_start);
#endif

        pangulu_cuda_memcpy_host_to_device_value(calculate_s->cuda_value, s->value_csc, s->nnz);

#ifdef check_time

        time_cuda_memcpy += pangulu_time_check_end(&get_time_start);
#endif
}

void pangulu_smatrix_cuda_memcpy_complete_csr_async(pangulu_smatrix *calculate_s, pangulu_smatrix *s, cudaStream_t *stream)
{
        calculate_s->nnz = s->nnz;
#ifdef check_time
        struct timeval get_time_start;
        pangulu_time_check_begin(&get_time_start);
#endif
   
        pangulu_cudamemcpyasync_host_to_device(calculate_s->cuda_rowpointer, s->rowpointer, sizeof(pangulu_int64_t) * ((s->row) + 1), stream);
        pangulu_cudamemcpyasync_host_to_device(calculate_s->cuda_columnindex, s->columnindex, sizeof(pangulu_int32_t) * s->nnz, stream);
        pangulu_cudamemcpyasync_host_to_device(calculate_s->cuda_value, s->value, sizeof(calculate_type) * s->nnz, stream);

#ifdef check_time
        time_cuda_memcpy += pangulu_time_check_end(&get_time_start);
#endif
}

void pangulu_smatrix_cuda_memcpy_complete_csc_async(pangulu_smatrix *calculate_s, pangulu_smatrix *s, cudaStream_t *stream)
{
        calculate_s->nnz = s->nnz;
#ifdef check_time
        struct timeval get_time_start;
        pangulu_time_check_begin(&get_time_start);
#endif
        pangulu_cudamemcpyasync_host_to_device(calculate_s->cuda_rowpointer, s->columnpointer, sizeof(pangulu_int64_t) * ((s->row) + 1), stream);
        pangulu_cudamemcpyasync_host_to_device(calculate_s->cuda_columnindex, s->rowindex, sizeof(pangulu_int32_t) * s->nnz, stream);
        pangulu_cudamemcpyasync_host_to_device(calculate_s->cuda_value, s->value_csc, sizeof(calculate_type) * s->nnz, stream);

#ifdef check_time

        time_cuda_memcpy += pangulu_time_check_end(&get_time_start);
#endif
}

void pangulu_smatrix_cuda_memcpy_complete_csc(pangulu_smatrix *calculate_s, pangulu_smatrix *s)
{
        calculate_s->nnz = s->nnz;
#ifdef check_time
        struct timeval get_time_start;
        pangulu_time_check_begin(&get_time_start);
#endif
        pangulu_cuda_memcpy_host_to_device_inblock_ptr(calculate_s->cuda_rowpointer, s->columnpointer, (s->row) + 1);
        pangulu_cuda_memcpy_host_to_device_inblock_idx(calculate_s->cuda_columnindex, s->rowindex, s->nnz);
        pangulu_cuda_memcpy_host_to_device_value(calculate_s->cuda_value, s->value_csc, s->nnz);
#ifdef check_time

        time_cuda_memcpy += pangulu_time_check_end(&get_time_start);
#endif
}
#endif