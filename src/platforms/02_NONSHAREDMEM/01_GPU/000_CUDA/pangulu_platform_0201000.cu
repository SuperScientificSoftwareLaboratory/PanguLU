#define PANGULU_PLATFORM_ENV
#include "../../../../pangulu_common.h"
#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)
#define CHECK_CUDA_LAST_ERROR() check_cuda_last_error(__FILE__, __LINE__)
#define PANGULU_WARP_SIZE 32
#define PANGULU_DATAMOV_THREADPERBLOCK (pangulu_gpu_data_move_warp_per_block * PANGULU_WARP_SIZE)
extern int pangulu_gpu_kernel_warp_per_block;
extern int pangulu_gpu_data_move_warp_per_block;
extern int pangulu_gpu_shared_mem_size;

#include <cublas_v2.h>
#include <cusolverDn.h>
cublasHandle_t cublas_handle;
cusolverDnHandle_t cusolver_handle;

FILE *result_file = NULL;
char mtx_name_glo[100];
calculate_type *getrf_dense_buf_d;
calculate_type *d_cusolver_work;
int *d_cusolver_info;
int *d_cusolver_pivot;
calculate_type *d_getrf_tag_buffer = NULL;
calculate_type *d_ssssm_dense_buf_opdst = NULL;
calculate_type *d_ssssm_dense_buf_op1 = NULL;
calculate_type *d_ssssm_dense_buf_op2 = NULL;

#ifdef PANGULU_PERF
extern pangulu_stat_t global_stat;
#endif

void check_cuda_last_error(const char *const file, int const line)
{
    cudaError_t result = cudaGetLastError();
    if (result)
    {
        fprintf(stderr, "[PanguLU ERROR] CUDA error at %s:%d %s (code=%d)\n",
                file, line, cudaGetErrorString(result), static_cast<unsigned int>(result));
        exit(EXIT_FAILURE);
    }
}

void check_cuda_error(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "[PanguLU ERROR] CUDA error at %s:%d %s (code=%d)\n",
                file, line, cudaGetErrorString(result), static_cast<unsigned int>(result));
        exit(EXIT_FAILURE);
    }
}

void pangulu_platform_0201000_malloc(void **platform_address, size_t size)
{
    cudaError_t err = cudaMalloc(platform_address, size);
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_malloc_pinned(void **platform_address, size_t size)
{
    cudaError_t err = cudaHostAlloc(platform_address, size, cudaHostAllocDefault);
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_synchronize()
{
    cudaError_t err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_memset(void *s, int c, size_t n)
{
    cudaError_t err = cudaMemset(s, c, n);
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_create_stream(void **stream)
{
    cudaError_t err = cudaStreamCreate((cudaStream_t *)stream);
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_memcpy(void *dst, const void *src, size_t count, unsigned int kind)
{
    cudaError_t err;
    if (kind == 0)
    {
        err = cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
    }
    else if (kind == 1)
    {
        err = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
    }
    else if (kind == 2)
    {
        err = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
    }
    else
    {
        fprintf(stderr, "Invalid memcpy kind value\n");
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_memcpy_async(void *dst, const void *src, size_t count, unsigned int kind, void *stream)
{
    cudaError_t err;
    if (kind == 0)
    {
        err = cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, (cudaStream_t)stream);
    }
    else if (kind == 1)
    {
        err = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
    }
    else if (kind == 2)
    {
        err = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    }
    else
    {
        fprintf(stderr, "Invalid memcpy kind value\n");
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_free(void *devptr)
{
    cudaError_t err = cudaFree(devptr);
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_get_device_num(int *device_num)
{
    cudaError_t err = cudaGetDeviceCount(device_num);
    CHECK_CUDA_ERROR(err);
}

void pangulu_platform_0201000_set_default_device(int device_num)
{
    cudaError_t err = cudaSetDevice(device_num);
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_num);
    CHECK_CUDA_ERROR(err);
    pangulu_gpu_shared_mem_size = prop.sharedMemPerBlock;
}

void pangulu_platform_0201000_get_device_name(char *name, int device_num)
{
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_num);
    CHECK_CUDA_ERROR(err);
    strcpy(name, prop.name);
}

void pangulu_platform_0201000_get_device_memory_usage(size_t *used_byte)
{
    size_t total_byte;
    cudaError_t err = cudaMemGetInfo(used_byte, &total_byte);
    *used_byte = total_byte - *used_byte;
    CHECK_CUDA_ERROR(err);
}



#ifdef PANGULU_NONSHAREDMEM

// __device__ pangulu_inblock_ptr
// binarysearch_inblk_cuda(
//     pangulu_inblock_idx *ridx,
//     pangulu_int32_t left,
//     pangulu_int32_t right,
//     pangulu_inblock_idx target)
// {
//     pangulu_int32_t mid;
//     while (left <= right)
//     {
//         mid = left + (right - left) / 2;
//         if (ridx[mid] == target)
//         {
//             return mid;
//         }
//         else if (ridx[mid] > target)
//         {
//             right = mid - 1;
//         }
//         else
//         {
//             left = mid + 1;
//         }
//     }
//     return 0xffffffff;
// }

void pangulu_cuda_download_block(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *slot)
{
    pangulu_platform_0201000_memcpy(slot->value, slot->d_value, sizeof(calculate_type) * slot->columnpointer[nb], 1);
}

__global__ void pangulu_load_dense(
    int nb,
    pangulu_inblock_ptr *columnpointer,
    pangulu_inblock_idx *rowindex,
    calculate_real_type *value_rp,
    calculate_real_type *dense_buf_rp)
{
    calculate_type *value = (calculate_type *)value_rp;
    calculate_type *dense_buf = (calculate_type *)dense_buf_rp;
    int col = blockIdx.x;
    if (col >= nb)
    {
        return;
    }
    for (pangulu_inblock_idx row = threadIdx.x; row < nb; row += blockDim.x)
    {
#ifdef PANGULU_COMPLEX
        __real__(dense_buf[col * nb + row]) = 0;
        __imag__(dense_buf[col * nb + row]) = 0;
#else
        dense_buf[col * nb + row] = 0;
#endif
    }
    __syncthreads();
    for (int idx = columnpointer[col] + threadIdx.x; idx < columnpointer[col + 1]; idx += blockDim.x)
    {
        int row = rowindex[idx];
        dense_buf[col * nb + row] = value[idx];
    }
}

__global__ void pangulu_load_dense_getrf(
    int nb,
    pangulu_inblock_ptr *columnpointer,
    pangulu_inblock_idx *rowindex,
    calculate_real_type *value_csc_rp,
    pangulu_inblock_ptr *rowpointer,
    pangulu_inblock_idx *columnindex,
    calculate_real_type *value_csr_rp,
    calculate_real_type *dense_buf_rp)
{
    calculate_type *value_csc = (calculate_type *)value_csc_rp;
    calculate_type *value_csr = (calculate_type *)value_csr_rp;
    calculate_type *dense_buf = (calculate_type *)dense_buf_rp;

    int rc = blockIdx.x;
    if (rc >= nb)
    {
        return;
    }
    // for(pangulu_inblock_idx row = threadIdx.x; row < nb; row+=blockDim.x){
    //     dense_buf[rc * nb + row] = 0;
    // }
    // __syncthreads();
    for (int idx = columnpointer[rc] + threadIdx.x; idx < columnpointer[rc + 1]; idx += blockDim.x)
    {
        int row = rowindex[idx];
        dense_buf[rc * nb + row] = value_csc[idx];
    }
    for (int idx = rowpointer[rc] + threadIdx.x; idx < rowpointer[rc + 1]; idx += blockDim.x)
    {
        int col = columnindex[idx];
        dense_buf[col * nb + rc] = value_csr[idx];
    }
}

__global__ void pangulu_store_dense_getrf(
    int nb,
    pangulu_inblock_ptr *columnpointer,
    pangulu_inblock_idx *rowindex,
    calculate_real_type *value_csc_rp,
    pangulu_inblock_ptr *rowpointer,
    pangulu_inblock_idx *columnindex,
    calculate_real_type *value_csr_rp,
    calculate_real_type *dense_buf_rp)
{
    calculate_type *value_csc = (calculate_type *)value_csc_rp;
    calculate_type *value_csr = (calculate_type *)value_csr_rp;
    calculate_type *dense_buf = (calculate_type *)dense_buf_rp;

    int rc = blockIdx.x;
    if (rc >= nb)
    {
        return;
    }
    for (int idx = columnpointer[rc] + threadIdx.x; idx < columnpointer[rc + 1]; idx += blockDim.x)
    {
        int row = rowindex[idx];
        value_csc[idx] = dense_buf[rc * nb + row];
    }
    for (int idx = rowpointer[rc] + threadIdx.x; idx < rowpointer[rc + 1]; idx += blockDim.x)
    {
        int col = columnindex[idx];
        value_csr[idx] = dense_buf[col * nb + rc];
    }
}

__global__ void clear_dense(
    pangulu_inblock_idx nb,
    calculate_real_type *dense_rp)
{
    calculate_type *dense = (calculate_type *)dense_rp;
    int col = blockIdx.x;
    if (col >= nb)
    {
        return;
    }
    for (int idx = col * nb + threadIdx.x; idx < (col + 1) * nb; idx += blockDim.x)
    {
#ifdef PANGULU_COMPLEX
        __real__(dense[idx]) = 0.0;
        __imag__(dense[idx]) = 0.0;
#else
        dense[idx] = 0.0;
#endif
    }
}

__global__ void tstrf_cuda(
    pangulu_inblock_idx n,
    pangulu_inblock_ptr *b_rowpointer,
    pangulu_inblock_idx *b_columnindex,
    pangulu_inblock_ptr *b_valueidx,
    calculate_real_type *b_value_rp,
    pangulu_inblock_ptr *l_rowpointer,
    pangulu_inblock_idx *l_columnindex,
    calculate_real_type *l_value_rp)
{
    extern __shared__ char shared_memory[];
    pangulu_inblock_idx *s_idxa = (pangulu_inblock_idx *)shared_memory;
    calculate_type *s_dense = (calculate_type *)(shared_memory + sizeof(pangulu_inblock_idx) * n * (blockDim.x / PANGULU_WARP_SIZE));

    calculate_type *b_value = (calculate_type *)b_value_rp;
    calculate_type *l_value = (calculate_type *)l_value_rp;

    pangulu_inblock_idx colidx = blockIdx.x * (blockDim.x / PANGULU_WARP_SIZE) + (threadIdx.x / PANGULU_WARP_SIZE);
    if (colidx >= n)
    {
        return;
    }
    pangulu_inblock_idx warp_thread = threadIdx.x % PANGULU_WARP_SIZE;
    pangulu_inblock_idx *s_idxa_warp = s_idxa + (threadIdx.x / PANGULU_WARP_SIZE) * n;
    calculate_type *s_dense_warp = s_dense + (threadIdx.x / PANGULU_WARP_SIZE) * n;

    pangulu_inblock_ptr b_col_start = b_rowpointer[colidx];
    pangulu_inblock_ptr b_col_end = b_rowpointer[colidx + 1];
    if (b_col_end == b_col_start)
    {
        return;
    }

    for (pangulu_inblock_idx i = warp_thread; i < b_col_end - b_col_start; i += PANGULU_WARP_SIZE)
    {
        s_idxa_warp[i] = b_columnindex[b_col_start + i];
        s_dense_warp[s_idxa_warp[i]] = b_value[b_valueidx[b_col_start + i]];
    }

    for (pangulu_inblock_ptr i = b_col_start, t = 0; i < b_col_end; i++, t++)
    {
        pangulu_inblock_idx rowa = s_idxa_warp[t];
        pangulu_inblock_ptr coll1 = l_rowpointer[rowa];
        pangulu_inblock_ptr coll2 = l_rowpointer[rowa + 1];

        calculate_type vala;
        vala = s_dense_warp[s_idxa_warp[t]];
#ifdef PANGULU_COMPLEX
        calculate_type z1 = vala;
        calculate_type z2 = l_value[coll1];
        __real__(vala) =
            (__real__(z1) * __real__(z2) + __imag__(z1) * __imag__(z2)) /
            (__real__(z2) * __real__(z2) + __imag__(z2) * __imag__(z2));
        __imag__(vala) =
            (__imag__(z1) * __real__(z2) - __real__(z1) * __imag__(z2)) /
            (__real__(z2) * __real__(z2) + __imag__(z2) * __imag__(z2));
#else
        vala /= l_value[coll1];
#endif
        if (warp_thread == 0)
        {
            s_dense_warp[s_idxa_warp[t]] = vala;
        }

        for (pangulu_inblock_ptr j = coll1 + 1 + warp_thread, p = warp_thread; j < coll2; j += PANGULU_WARP_SIZE, p += PANGULU_WARP_SIZE)
        {
#ifdef PANGULU_COMPLEX
            z1 = vala;
            z2 = l_value[j];
            __real__(s_dense_warp[l_columnindex[j]]) -= (__real__(z1) * __real__(z2) - __imag__(z1) * __imag__(z2));
            __imag__(s_dense_warp[l_columnindex[j]]) -= (__imag__(z1) * __real__(z2) + __real__(z1) * __imag__(z2));
#else
            s_dense_warp[l_columnindex[j]] -= vala * l_value[j];
#endif
        }
    }

    for (pangulu_inblock_idx i = warp_thread; i < b_col_end - b_col_start; i += PANGULU_WARP_SIZE)
    {
        b_value[b_valueidx[b_col_start + i]] = s_dense_warp[s_idxa_warp[i]];
    }
}

__global__ void gessm_cuda(
    pangulu_inblock_idx n,
    pangulu_inblock_ptr *b_columnpointer,
    pangulu_inblock_idx *b_rowindex,
    calculate_real_type *b_value_rp,
    pangulu_inblock_ptr *l_columnpointer,
    pangulu_inblock_idx *l_rowindex,
    calculate_real_type *l_value_rp)
{
    extern __shared__ char shared_memory[];
    pangulu_inblock_idx *s_idxa = (pangulu_inblock_idx *)shared_memory;
    calculate_type *s_dense = (calculate_type *)(shared_memory + sizeof(pangulu_inblock_idx) * n * (blockDim.x / PANGULU_WARP_SIZE));

    calculate_type *b_value = (calculate_type *)b_value_rp;
    calculate_type *l_value = (calculate_type *)l_value_rp;

    pangulu_inblock_idx colidx = blockIdx.x * (blockDim.x / PANGULU_WARP_SIZE) + (threadIdx.x / PANGULU_WARP_SIZE);
    if (colidx >= n)
    {
        return;
    }
    pangulu_inblock_idx warp_thread = threadIdx.x % PANGULU_WARP_SIZE;
    pangulu_inblock_idx *s_idxa_warp = s_idxa + (threadIdx.x / PANGULU_WARP_SIZE) * n;
    calculate_type *s_dense_warp = s_dense + (threadIdx.x / PANGULU_WARP_SIZE) * n;

    pangulu_inblock_ptr b_col_start = b_columnpointer[colidx];
    pangulu_inblock_ptr b_col_end = b_columnpointer[colidx + 1];
    if (b_col_end == b_col_start)
    {
        return;
    }

    for (pangulu_inblock_idx i = warp_thread; i < b_col_end - b_col_start; i += PANGULU_WARP_SIZE)
    {
        s_idxa_warp[i] = b_rowindex[b_col_start + i];
        s_dense_warp[s_idxa_warp[i]] = b_value[b_col_start + i];
    }

    for (pangulu_inblock_ptr i = b_col_start, t = 0; i < b_col_end; i++, t++)
    {
        pangulu_inblock_idx rowa = s_idxa_warp[t];
        calculate_type vala = s_dense_warp[s_idxa_warp[t]];
        pangulu_inblock_ptr coll1 = l_columnpointer[rowa];
        pangulu_inblock_ptr coll2 = l_columnpointer[rowa + 1];
        for (pangulu_int64_t j = coll1 + warp_thread, p = warp_thread; j < coll2; j += PANGULU_WARP_SIZE, p += PANGULU_WARP_SIZE)
        {
#ifdef PANGULU_COMPLEX
            calculate_type z1 = vala;
            calculate_type z2 = l_value[j];
            __real__(s_dense_warp[l_rowindex[j]]) -= (__real__(z1) * __real__(z2) - __imag__(z1) * __imag__(z2));
            __imag__(s_dense_warp[l_rowindex[j]]) -= (__imag__(z1) * __real__(z2) + __real__(z1) * __imag__(z2));
#else
            s_dense_warp[l_rowindex[j]] -= vala * l_value[j];
#endif
        }
    }

    for (pangulu_inblock_idx i = warp_thread; i < b_col_end - b_col_start; i += PANGULU_WARP_SIZE)
    {
        b_value[b_col_start + i] = s_dense_warp[s_idxa_warp[i]];
    }
}

__global__ void ssssm_cuda(
    pangulu_inblock_idx n,
    pangulu_inblock_ptr *d_rowptrc,
    pangulu_inblock_idx *d_colidxc,
    calculate_real_type *d_valuec_rp,
    pangulu_inblock_ptr *d_rowptrb,
    pangulu_inblock_idx *d_colidxb,
    calculate_real_type *d_valueb_rp,
    pangulu_inblock_ptr *d_rowptra,
    pangulu_inblock_idx *d_colidxa,
    calculate_real_type *d_valuea_rp)
{
    extern __shared__ calculate_type s_dense[];
    pangulu_inblock_idx row = blockIdx.x;
    const pangulu_inblock_idx thread_offset = threadIdx.x;

    calculate_type *d_valuec = (calculate_type *)d_valuec_rp;
    calculate_type *d_valueb = (calculate_type *)d_valueb_rp;
    calculate_type *d_valuea = (calculate_type *)d_valuea_rp;

    if (row >= n)
    {
        return;
    }

    pangulu_inblock_ptr therowc = d_rowptrc[row];
    pangulu_inblock_ptr nextrowc = d_rowptrc[row + 1];

    pangulu_inblock_ptr therow = d_rowptra[row];
    pangulu_inblock_ptr nextrow = d_rowptra[row + 1];

    for (pangulu_inblock_ptr idx = therowc + thread_offset; idx < nextrowc; idx += blockDim.x)
    {
        pangulu_inblock_idx col = d_colidxc[idx];
#ifdef PANGULU_COMPLEX
        __real__(s_dense[col]) = 0.0;
        __imag__(s_dense[col]) = 0.0;
#else
        s_dense[col] = 0.0;
#endif
    }

    __syncthreads();

    for (pangulu_inblock_ptr i = therow; i < nextrow; i++)
    {
        pangulu_inblock_idx cola = d_colidxa[i];
        calculate_type vala = d_valuea[i];

        pangulu_inblock_ptr therowb = d_rowptrb[cola];
        pangulu_inblock_ptr nextrowb = d_rowptrb[cola + 1];

        for (pangulu_inblock_ptr j = therowb + thread_offset; j < nextrowb; j += blockDim.x)
        {
            pangulu_inblock_idx colb = d_colidxb[j];
#ifdef PANGULU_COMPLEX
            calculate_type z1 = vala;
            calculate_type z2 = d_valueb[j];
            atomicAdd(&__real__(s_dense[colb]), (__real__(z1) * __real__(z2) - __imag__(z1) * __imag__(z2)));
            atomicAdd(&__imag__(s_dense[colb]), (__imag__(z1) * __real__(z2) + __real__(z1) * __imag__(z2)));
#else
            atomicAdd(&s_dense[colb], vala * d_valueb[j]);
#endif
        }
    }

    __syncthreads();

    for (pangulu_inblock_ptr idx = therowc + thread_offset; idx < nextrowc; idx += blockDim.x)
    {
        pangulu_inblock_idx col = d_colidxc[idx];
#ifdef PANGULU_COMPLEX
        atomicAdd(&__real__(d_valuec[idx]), -__real__(s_dense[col]));
        atomicAdd(&__imag__(d_valuec[idx]), -__imag__(s_dense[col]));
#else
        atomicAdd(&d_valuec[idx], -s_dense[col]);
#endif
    }
}

void pangulu_platform_0201000_getrf(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    int tid)
{
    if (!d_getrf_tag_buffer)
    {
        cudaMalloc(&d_getrf_tag_buffer, sizeof(calculate_type) * nb * nb);
        if (!d_getrf_tag_buffer)
        {
            printf("[PanguLU Error] cudaMalloc failed: requested %lld bytes but returned NULL.\n", sizeof(calculate_type) * nb * nb);
        }
    }

    if (!cusolver_handle)
    {
        cusolverDnCreate(&cusolver_handle);
        int work_size = 0;
#if defined(CALCULATE_TYPE_R64)
        cusolverDnDgetrf_bufferSize(cusolver_handle, nb, nb, d_getrf_tag_buffer, nb, &work_size);
#elif defined(CALCULATE_TYPE_R32)
        cusolverDnSgetrf_bufferSize(cusolver_handle, nb, nb, d_getrf_tag_buffer, nb, &work_size);
#elif defined(CALCULATE_TYPE_CR64)
        cusolverDnZgetrf_bufferSize(cusolver_handle, nb, nb, (cuDoubleComplex *)d_getrf_tag_buffer, nb, &work_size);
#elif defined(CALCULATE_TYPE_CR32)
        cusolverDnCgetrf_bufferSize(cusolver_handle, nb, nb, (cuFloatComplex *)d_getrf_tag_buffer, nb, &work_size);
#else
#error[PanguLU Compile Error] Unsupported CALCULATE_TYPE for the selected BLAS library. Please recompile with a compatible value.
#endif
        cudaMalloc(&d_cusolver_work, work_size * sizeof(calculate_type));
        cudaMalloc(&d_cusolver_info, sizeof(int));
        cudaMalloc(&d_cusolver_pivot, sizeof(int) * nb);
    }

    pangulu_storage_slot_t *diag_upper = NULL;
    pangulu_storage_slot_t *diag_lower = NULL;
    if (opdst->is_upper)
    {
        diag_upper = opdst;
        diag_lower = opdst->related_block;
    }
    else
    {
        diag_upper = opdst->related_block;
        diag_lower = opdst;
    }

    clear_dense<<<nb, PANGULU_DATAMOV_THREADPERBLOCK>>>(nb, (calculate_real_type *)d_getrf_tag_buffer);
    cudaDeviceSynchronize();
    pangulu_load_dense_getrf<<<nb, PANGULU_DATAMOV_THREADPERBLOCK>>>(
        nb,
        diag_lower->d_columnpointer,
        diag_lower->d_rowindex,
        (calculate_real_type *)diag_lower->d_value,
        diag_upper->d_rowpointer,
        diag_upper->d_columnindex,
        (calculate_real_type *)diag_upper->d_value,
        (calculate_real_type *)d_getrf_tag_buffer);

#ifdef PANGULU_PERF
    struct timeval start;
    cudaDeviceSynchronize();
    pangulu_time_start(&start);
#endif

#if defined(CALCULATE_TYPE_R64)
    cusolverDnDgetrf(cusolver_handle, nb, nb, d_getrf_tag_buffer, nb, d_cusolver_work, NULL, d_cusolver_info);
#elif defined(CALCULATE_TYPE_R32)
    cusolverDnSgetrf(cusolver_handle, nb, nb, d_getrf_tag_buffer, nb, d_cusolver_work, NULL, d_cusolver_info);
#elif defined(CALCULATE_TYPE_CR64)
    cusolverDnZgetrf(cusolver_handle, nb, nb, (cuDoubleComplex *)d_getrf_tag_buffer, nb, (cuDoubleComplex *)d_cusolver_work, NULL, d_cusolver_info);
#elif defined(CALCULATE_TYPE_CR32)
    cusolverDnCgetrf(cusolver_handle, nb, nb, (cuFloatComplex *)d_getrf_tag_buffer, nb, (cuFloatComplex *)d_cusolver_work, NULL, d_cusolver_info);
#else
#error[PanguLU Compile Error] Unsupported CALCULATE_TYPE for the selected BLAS library. Please recompile with a compatible value.
#endif

#ifdef PANGULU_PERF
    cudaDeviceSynchronize();
    global_stat.time_inner_kernel += pangulu_time_stop(&start);
#endif

    pangulu_store_dense_getrf<<<nb, PANGULU_DATAMOV_THREADPERBLOCK>>>(
        nb,
        diag_lower->d_columnpointer,
        diag_lower->d_rowindex,
        (calculate_real_type *)diag_lower->d_value,
        diag_upper->d_rowpointer,
        diag_upper->d_columnindex,
        (calculate_real_type *)diag_upper->d_value,
        (calculate_real_type *)d_getrf_tag_buffer);

    pangulu_cuda_download_block(nb, diag_upper);
    pangulu_cuda_download_block(nb, diag_lower);
}

void pangulu_platform_0201000_tstrf(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *opdiag,
    int tid)
{
#ifdef PANGULU_PERF
    global_stat.kernel_cnt++;
    struct timeval start;
    pangulu_platform_0201000_synchronize();
    pangulu_time_start(&start);
#endif

    if (opdiag->is_upper == 0)
    {
        opdiag = opdiag->related_block;
    }

    int shared_memory_size = (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * nb * pangulu_gpu_kernel_warp_per_block;
    if (shared_memory_size > pangulu_gpu_shared_mem_size)
    {
        printf("[PanguLU Error] Requested shared memory size %d bytes exceeds the maximum limit of %d bytes.\n", shared_memory_size, pangulu_gpu_shared_mem_size);
        printf("[PanguLU Error] Please reduce 'init_options.nb' and try again. Exiting.\n");
        exit(1);
    }
    tstrf_cuda<<<
        PANGULU_ICEIL(nb, pangulu_gpu_kernel_warp_per_block),
        pangulu_gpu_kernel_warp_per_block * PANGULU_WARP_SIZE,
        shared_memory_size>>>(
        nb,
        opdst->d_rowpointer, opdst->d_columnindex, opdst->d_idx_of_csc_value_for_csr, (calculate_real_type *)opdst->d_value,
        opdiag->d_rowpointer, opdiag->d_columnindex, (calculate_real_type *)opdiag->d_value);

#ifdef PANGULU_PERF
    pangulu_platform_0201000_synchronize();
    global_stat.time_inner_kernel += pangulu_time_stop(&start);
#endif
    pangulu_cuda_download_block(nb, opdst);
}

void pangulu_platform_0201000_gessm(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *opdiag,
    int tid)
{
#ifdef PANGULU_PERF
    global_stat.kernel_cnt++;
    struct timeval start;
    pangulu_platform_0201000_synchronize();
    pangulu_time_start(&start);
#endif

    if (opdiag->is_upper == 1)
    {
        opdiag = opdiag->related_block;
    }

    int shared_memory_size = (sizeof(pangulu_inblock_idx) + sizeof(calculate_type)) * nb * pangulu_gpu_kernel_warp_per_block;
    gessm_cuda<<<
        PANGULU_ICEIL(nb, pangulu_gpu_kernel_warp_per_block),
        pangulu_gpu_kernel_warp_per_block * PANGULU_WARP_SIZE,
        shared_memory_size>>>(
        nb,
        opdst->d_columnpointer, opdst->d_rowindex, (calculate_real_type *)opdst->d_value,
        opdiag->d_columnpointer, opdiag->d_rowindex, (calculate_real_type *)opdiag->d_value);

#ifdef PANGULU_PERF
    pangulu_platform_0201000_synchronize();
    global_stat.time_inner_kernel += pangulu_time_stop(&start);
#endif
    pangulu_cuda_download_block(nb, opdst);
}

void pangulu_platform_0201000_ssssm(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *op1,
    pangulu_storage_slot_t *op2,
    int tid)
{
#ifdef PANGULU_PERF
    struct timeval start;
    cudaDeviceSynchronize();
    pangulu_time_start(&start);
#endif

    if (!cublas_handle)
    {
        cublasCreate(&cublas_handle);
        int work_size = 0;
#if defined(CALCULATE_TYPE_R64)
        cusolverDnDgetrf_bufferSize(cusolver_handle, nb, nb, d_getrf_tag_buffer, nb, &work_size);
#elif defined(CALCULATE_TYPE_R32)
        cusolverDnSgetrf_bufferSize(cusolver_handle, nb, nb, d_getrf_tag_buffer, nb, &work_size);
#elif defined(CALCULATE_TYPE_CR64)
        cusolverDnZgetrf_bufferSize(cusolver_handle, nb, nb, (cuDoubleComplex *)d_getrf_tag_buffer, nb, &work_size);
#elif defined(CALCULATE_TYPE_CR32)
        cusolverDnCgetrf_bufferSize(cusolver_handle, nb, nb, (cuFloatComplex *)d_getrf_tag_buffer, nb, &work_size);
#else
#error [PanguLU ERROR] Invalid CALCULATE_TYPE marco.
#endif

        cudaMalloc(&d_cusolver_work, work_size * sizeof(calculate_type));
        cudaMalloc(&d_cusolver_info, sizeof(int));
        cudaMalloc(&d_cusolver_pivot, sizeof(int) * nb);
        cudaMalloc(&d_ssssm_dense_buf_opdst, sizeof(calculate_type) * nb * nb);
        cudaMalloc(&d_ssssm_dense_buf_op1, sizeof(calculate_type) * nb * nb);
        cudaMalloc(&d_ssssm_dense_buf_op2, sizeof(calculate_type) * nb * nb);
    }

    if (opdst->brow_pos == opdst->bcol_pos)
    {
        pangulu_load_dense<<<nb, PANGULU_DATAMOV_THREADPERBLOCK>>>(nb, op1->d_columnpointer, op1->d_rowindex, (calculate_real_type *)op1->d_value, (calculate_real_type *)d_ssssm_dense_buf_op1);
        pangulu_load_dense<<<nb, PANGULU_DATAMOV_THREADPERBLOCK>>>(nb, op2->d_columnpointer, op2->d_rowindex, (calculate_real_type *)op2->d_value, (calculate_real_type *)d_ssssm_dense_buf_op2);
        pangulu_storage_slot_t *upper_diag;
        pangulu_storage_slot_t *lower_diag;
        if (opdst->is_upper)
        {
            upper_diag = opdst;
            lower_diag = opdst->related_block;
        }
        else
        {
            upper_diag = opdst->related_block;
            lower_diag = opdst;
        }
        pangulu_load_dense_getrf<<<nb, PANGULU_DATAMOV_THREADPERBLOCK>>>(
            nb,
            lower_diag->d_columnpointer, lower_diag->d_rowindex, (calculate_real_type *)lower_diag->d_value,
            upper_diag->d_rowpointer, upper_diag->d_columnindex, (calculate_real_type *)upper_diag->d_value,
            (calculate_real_type *)d_ssssm_dense_buf_opdst);

        calculate_type alpha = -1.0;
        calculate_type beta = 1.0;
#if defined(CALCULATE_TYPE_R64)
        cublasDgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            nb, nb, nb,
            &alpha,
            d_ssssm_dense_buf_op1, nb,
            d_ssssm_dense_buf_op2, nb,
            &beta,
            d_ssssm_dense_buf_opdst, nb);
#elif defined(CALCULATE_TYPE_R32)
        cublasSgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            nb, nb, nb,
            &alpha,
            d_ssssm_dense_buf_op1, nb,
            d_ssssm_dense_buf_op2, nb,
            &beta,
            d_ssssm_dense_buf_opdst, nb);
#elif defined(CALCULATE_TYPE_CR64)
        cublasZgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            nb, nb, nb,
            (cuDoubleComplex *)&alpha,
            (cuDoubleComplex *)d_ssssm_dense_buf_op1, nb,
            (cuDoubleComplex *)d_ssssm_dense_buf_op2, nb,
            (cuDoubleComplex *)&beta,
            (cuDoubleComplex *)d_ssssm_dense_buf_opdst, nb);
#elif defined(CALCULATE_TYPE_CR32)
        cublasCgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            nb, nb, nb,
            (cuFloatComplex *)&alpha,
            (cuFloatComplex *)d_ssssm_dense_buf_op1, nb,
            (cuFloatComplex *)d_ssssm_dense_buf_op2, nb,
            (cuFloatComplex *)&beta,
            (cuFloatComplex *)d_ssssm_dense_buf_opdst, nb);
#else
#error [PanguLU ERROR] Invalid CALCULATE_TYPE marco.
#endif

        pangulu_store_dense_getrf<<<nb, PANGULU_DATAMOV_THREADPERBLOCK>>>(
            nb,
            lower_diag->d_columnpointer, lower_diag->d_rowindex, (calculate_real_type *)lower_diag->d_value,
            upper_diag->d_rowpointer, upper_diag->d_columnindex, (calculate_real_type *)upper_diag->d_value,
            (calculate_real_type *)d_ssssm_dense_buf_opdst);
    }
    else
    {
#ifndef PANGULU_COMPLEX
        if ((op1->columnpointer[nb] == nb * nb) && (op2->columnpointer[nb] == nb * nb) && (opdst->columnpointer[nb] == nb * nb))
        {
            calculate_type alpha = -1.0;
            calculate_type beta = 1.0;
#if defined(CALCULATE_TYPE_R64)
            cublasDgemm(
                cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                nb, nb, nb,
                &alpha,
                op1->d_value, nb,
                op2->d_value, nb,
                &beta,
                opdst->d_value, nb);
#elif defined(CALCULATE_TYPE_R32)
            cublasSgemm(
                cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                nb, nb, nb,
                &alpha,
                op1->d_value, nb,
                op2->d_value, nb,
                &beta,
                opdst->d_value, nb);
#else
#error [PanguLU ERROR] Invalid CALCULATE_TYPE marco.
#endif
        }
        else
        {
#endif
            ssssm_cuda<<<
                nb,
                pangulu_gpu_kernel_warp_per_block * PANGULU_WARP_SIZE,
                sizeof(calculate_type) * nb>>>(
                nb,
                opdst->d_columnpointer, opdst->d_rowindex, (calculate_real_type *)opdst->d_value,
                op1->d_columnpointer, op1->d_rowindex, (calculate_real_type *)op1->d_value,
                op2->d_columnpointer, op2->d_rowindex, (calculate_real_type *)op2->d_value);
#ifndef PANGULU_COMPLEX
        }
#endif
    }

#ifdef PANGULU_PERF
    cudaDeviceSynchronize();
    global_stat.time_inner_kernel += pangulu_time_stop(&start);
#endif
}

void pangulu_platform_0201000_hybrid_batched(
    pangulu_inblock_idx nb,
    pangulu_uint64_t ntask,
    pangulu_task_t *tasks)
{
    for (pangulu_uint64_t itask = 0; itask < ntask; itask++)
    {
        switch (tasks[itask].kernel_id)
        {
        case PANGULU_TASK_GETRF:
            pangulu_platform_0201000_getrf(nb, tasks[itask].opdst, 0);
            break;
        case PANGULU_TASK_TSTRF:
            pangulu_platform_0201000_tstrf(nb, tasks[itask].opdst, tasks[itask].op1, 0);
            break;
        case PANGULU_TASK_GESSM:
            pangulu_platform_0201000_gessm(nb, tasks[itask].opdst, tasks[itask].op1, 0);
            break;
        case PANGULU_TASK_SSSSM:
            pangulu_platform_0201000_ssssm(nb, tasks[itask].opdst, tasks[itask].op1, tasks[itask].op2, 0);
            break;
        }
    }
}

void pangulu_platform_0201000_ssssm_batched(
    pangulu_inblock_idx nb,
    pangulu_uint64_t ntask,
    pangulu_task_t *tasks)
{
    for (pangulu_uint64_t itask = 0; itask < ntask; itask++)
    {
        pangulu_platform_0201000_ssssm(nb, tasks[itask].opdst, tasks[itask].op1, tasks[itask].op2, 0);
    }
}

#else

void pangulu_platform_0201000_getrf(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    int tid)
{
}
void pangulu_platform_0201000_tstrf(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *opdiag,
    int tid)
{
}
void pangulu_platform_0201000_gessm(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *opdiag,
    int tid)
{
}
void pangulu_platform_0201000_ssssm(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *opdst,
    pangulu_storage_slot_t *op1,
    pangulu_storage_slot_t *op2,
    int tid)
{
}

void pangulu_platform_0201000_ssssm_batched(
    pangulu_inblock_idx nb,
    pangulu_uint64_t ntask,
    pangulu_task_t *tasks)
{
}

void pangulu_platform_0201000_hybrid_batched(
    pangulu_inblock_idx nb,
    pangulu_uint64_t ntask,
    pangulu_task_t *tasks)
{
}

#endif

void pangulu_platform_0201000_spmv(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *a,
    calculate_type *x,
    calculate_type *y)
{
}

void pangulu_platform_0201000_vecadd(
    pangulu_int64_t length,
    calculate_type *bval,
    calculate_type *xval)
{
}

void pangulu_platform_0201000_sptrsv(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *s,
    calculate_type *xval,
    pangulu_int64_t uplo)
{
}
