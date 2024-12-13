#define PANGULU_PLATFORM_ENV
#include "pangulu_cuda.h"

#if defined(PANGULU_COMPLEX)
#warning Complex on GPU is comming soon. Fallback to CPU.
#else // defined(PANGULU_COMPLEX)

#ifndef substitution_forward
#define substitution_forward 1
#endif

__global__ void gessm_kernel_v2(pangulu_int64_t n,
                                pangulu_inblock_ptr *l_columnpointer,
                                pangulu_inblock_idx *l_rowindex,
                                calculate_type *l_value,
                                pangulu_inblock_ptr *x_columnpointer,
                                pangulu_inblock_idx *x_rowindex,
                                calculate_type *x_value,
                                pangulu_inblock_ptr *a_columnpointer,
                                pangulu_inblock_idx *a_rowindex,
                                calculate_type *a_value)
{
    pangulu_int64_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    pangulu_int64_t lane_id = threadIdx.x % 32;
    pangulu_int64_t warp_local_id = threadIdx.x / 32;
    pangulu_int64_t flagx, flagb;

    if (warp_id >= n || x_columnpointer[warp_id] == x_columnpointer[warp_id + 1])
        return;

    pangulu_int64_t cola1 = a_columnpointer[warp_id];
    pangulu_int64_t cola2 = a_columnpointer[warp_id + 1];
    pangulu_int64_t colx1 = x_columnpointer[warp_id];
    pangulu_int64_t colx2 = x_columnpointer[warp_id + 1];

    if (cola2 - cola1 >= sm_len_dgessm)
    {
        for (pangulu_int64_t i = colx1, t = 0; i < colx2; i++, t++)
        {
            pangulu_int64_t rowx = x_rowindex[i];
            x_value[i] = a_value[i];
            calculate_type valx = (calculate_type)(x_value[i]);

            pangulu_int64_t coll1 = l_columnpointer[rowx];
            pangulu_int64_t coll2 = l_columnpointer[rowx + 1];
            for (pangulu_int64_t j = coll1 + 1 + lane_id, p = lane_id; j < coll2; j += warp_size, p += warp_size)
            {
                pangulu_inblock_idx f = binarysearch_idx(a_rowindex, cola1 + 1 + t + p, cola2 - coll2 + j, l_rowindex[j]);
                a_value[f] -= valx * l_value[j];
            }
            __syncthreads();
        }
    }
    else
    {
        __shared__ pangulu_inblock_idx s_rowidxa[sm_len_dgessm * warp_per_block_dgessm];
        __shared__ calculate_type s_valuea[sm_len_dgessm * warp_per_block_dgessm];

        pangulu_inblock_idx *idx_local = &s_rowidxa[warp_local_id * sm_len_dgessm];
        calculate_type *val_local = &s_valuea[warp_local_id * sm_len_dgessm];

        for (pangulu_int64_t i = lane_id; i < cola2 - cola1; i += warp_size)
        {
            idx_local[i] = a_rowindex[cola1 + i];
            val_local[i] = a_value[cola1 + i];
        }

        for (pangulu_int64_t i = colx1, t = 0; i < colx2; i++, t++)
        {
            pangulu_int64_t rowx = x_rowindex[i];
            x_value[i] = val_local[t];
            calculate_type valx = (calculate_type)(x_value[i]);

            pangulu_int64_t coll1 = l_columnpointer[rowx];
            pangulu_int64_t coll2 = l_columnpointer[rowx + 1];
            for (pangulu_int64_t j = coll1 + 1 + lane_id, p = lane_id; j < coll2; j += warp_size, p += warp_size)
            {
                // update a's value;
                pangulu_inblock_idx f = binarysearch_idx(idx_local, 1 + t + p, cola2 - cola1 - coll2 + j, l_rowindex[j]);
                if (f != 0xffff)
                    val_local[f] -= valx * l_value[j];
            }
            __syncthreads();
        }

        for (pangulu_int64_t i = lane_id; i < cola2 - cola1; i += warp_size)
        {
            a_value[cola1 + i] = val_local[i];
        }
    }
}

__global__ void gessm_kernel_v3(pangulu_int64_t n,
                                pangulu_inblock_ptr *l_columnpointer,
                                pangulu_inblock_idx *l_rowindex,
                                calculate_type *l_value,
                                pangulu_inblock_ptr *x_columnpointer,
                                pangulu_inblock_idx *x_rowindex,
                                calculate_type *x_value,
                                pangulu_inblock_ptr *a_columnpointer,
                                pangulu_inblock_idx *a_rowindex,
                                calculate_type *a_value)
{

    pangulu_int64_t colidx = blockIdx.x;
    pangulu_int64_t colx1 = x_columnpointer[colidx];
    pangulu_int64_t colx2 = x_columnpointer[colidx + 1];

    if (colidx >= n || colx2 == colx1)
        return;

    pangulu_int64_t cola1 = a_columnpointer[colidx];
    pangulu_int64_t cola2 = a_columnpointer[colidx + 1];

    if (cola2 - cola1 >= sm_len_dgessm)
    {
        for (pangulu_int64_t i = colx1, t = 0; i < colx2; i++, t++)
        {
            pangulu_int64_t rowx = x_rowindex[i];
            x_value[i] = a_value[i];
            calculate_type valx = x_value[i];

            pangulu_int64_t coll1 = l_columnpointer[rowx];
            pangulu_int64_t coll2 = l_columnpointer[rowx + 1];

            for (pangulu_int64_t j = coll1 + 1 + threadIdx.x, p = threadIdx.x; j < coll2; j += blockDim.x, p += blockDim.x)
            {
                // update a's value;
                pangulu_inblock_idx f = binarysearch_idx(a_rowindex, cola1 + 1 + t + p, cola2 - coll2 + j, l_rowindex[j]);
                a_value[f] -= valx * l_value[j];
            }
            __syncthreads();
        }
    }
    else
    {
        __shared__ pangulu_inblock_idx s_idxa[sm_len_dgessm];
        __shared__ calculate_type s_vala[sm_len_dgessm];

        for (pangulu_int64_t i = threadIdx.x; i < cola2 - cola1; i += blockDim.x)
        {
            s_idxa[i] = a_rowindex[cola1 + i];
            s_vala[i] = a_value[cola1 + i];
        }
        __syncthreads();

        for (pangulu_int64_t i = colx1, t = 0; i < colx2; i++, t++)
        {
            pangulu_int64_t rowx = x_rowindex[i];
            x_value[i] = s_vala[t];
            calculate_type valx = x_value[i];

            pangulu_int64_t coll1 = l_columnpointer[rowx];
            pangulu_int64_t coll2 = l_columnpointer[rowx + 1];

            for (pangulu_int64_t j = coll1 + 1 + threadIdx.x, p = threadIdx.x; j < coll2; j += blockDim.x, p += blockDim.x)
            {
                // update a's value;
                pangulu_inblock_idx f = binarysearch_idx(s_idxa, 1 + t + p, cola2 - cola1 - coll2 + j, l_rowindex[j]);
                s_vala[f] -= valx * l_value[j];
            }
            __syncthreads();
        }

        for (pangulu_int64_t i = threadIdx.x; i < cola2 - cola1; i += blockDim.x)
        {
            a_value[cola1 + i] = s_vala[i];
        }
        //__syncthreads();
    }
}

__global__ void tstrf_kernel_v2(pangulu_int64_t n,
                                pangulu_inblock_ptr *u_rowpointer,
                                pangulu_inblock_idx *u_columnindex,
                                calculate_type *u_value,
                                pangulu_inblock_ptr *x_rowpointer,
                                pangulu_inblock_idx *x_columnindex,
                                calculate_type *x_value,
                                pangulu_inblock_ptr *a_rowpointer,
                                pangulu_inblock_idx *a_columnindex,
                                calculate_type *a_value)
{
    pangulu_int64_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    pangulu_int64_t lane_id = threadIdx.x % 32;
    pangulu_int64_t warp_local_id = threadIdx.x / 32;

    if (warp_id >= n || x_rowpointer[warp_id] == x_rowpointer[warp_id + 1])
        return;

    pangulu_int64_t cola1 = a_rowpointer[warp_id];
    pangulu_int64_t cola2 = a_rowpointer[warp_id + 1];
    pangulu_int64_t colx1 = x_rowpointer[warp_id];
    pangulu_int64_t colx2 = x_rowpointer[warp_id + 1];

    if (cola2 - cola1 >= sm_len_dtstrf)
    {
        for (pangulu_int64_t i = colx1, t = 0; i < colx2; i++, t++)
        {
            pangulu_int64_t rowx = x_columnindex[i];
            pangulu_int64_t colu1 = u_rowpointer[rowx];
            pangulu_int64_t colu2 = u_rowpointer[rowx + 1];

            x_value[i] = a_value[i] / u_value[colu1];
            calculate_type valx = x_value[i];
            for (pangulu_int64_t j = colu1 + 1 + lane_id, p = lane_id; j < colu2; j += warp_size, p += warp_size)
            {
                // update a's value;
                pangulu_inblock_idx f = binarysearch_idx(a_columnindex, cola1 + t + 1 + p, cola2 - colu2 + j, u_columnindex[j]);
                a_value[f] -= valx * u_value[j];
            }
            __syncthreads();
        }
    }
    else
    {
        __shared__ pangulu_inblock_idx s_rowidxa[sm_len_dtstrf * warp_per_block_dtstrf];
        __shared__ calculate_type s_valuea[sm_len_dtstrf * warp_per_block_dtstrf];

        pangulu_inblock_idx *idx_local = &s_rowidxa[warp_local_id * sm_len_dtstrf];
        calculate_type *val_local = &s_valuea[warp_local_id * sm_len_dtstrf];

        for (pangulu_int64_t i = lane_id; i < cola2 - cola1; i += warp_size)
        {
            idx_local[i] = a_columnindex[cola1 + i];
            val_local[i] = a_value[cola1 + i];
        }

        for (pangulu_int64_t i = colx1, t = 0; i < colx2; i++, t++)
        {
            pangulu_int64_t rowx = x_columnindex[i];
            pangulu_int64_t colu1 = u_rowpointer[rowx];
            pangulu_int64_t colu2 = u_rowpointer[rowx + 1];

            x_value[i] = val_local[t] / u_value[colu1];
            calculate_type valx = x_value[i];
            for (pangulu_int64_t j = colu1 + 1 + lane_id, p = lane_id; j < colu2; j += warp_size, p += warp_size)
            {
                // update a's value;
                pangulu_inblock_idx f = binarysearch_idx(idx_local, 1 + t + p, cola2 - cola1 - colu2 + j, u_columnindex[j]);
                val_local[f] -= valx * u_value[j];
            }
            __syncthreads();
        }

        for (pangulu_int64_t i = lane_id; i < cola2 - cola1; i += warp_size)
        {
            a_value[cola1 + i] = val_local[i];
        }
    }
}
__global__ void tstrf_kernel_v3(pangulu_int64_t n,
                                pangulu_inblock_ptr *u_rowpointer,
                                pangulu_inblock_idx *u_columnindex,
                                calculate_type *u_value,
                                pangulu_inblock_ptr *x_rowpointer,
                                pangulu_inblock_idx *x_columnindex,
                                calculate_type *x_value,
                                pangulu_inblock_ptr *a_rowpointer,
                                pangulu_inblock_idx *a_columnindex,
                                calculate_type *a_value)
{
    pangulu_int64_t colidx = blockIdx.x;
    pangulu_int64_t colx1 = x_rowpointer[colidx];
    pangulu_int64_t colx2 = x_rowpointer[colidx + 1];

    if (colidx >= n || colx1 == colx2)
        return;

    pangulu_int64_t cola1 = a_rowpointer[colidx];
    pangulu_int64_t cola2 = a_rowpointer[colidx + 1];

    if (cola2 - cola1 >= sm_len_dtstrf)
    {
        for (pangulu_int64_t i = colx1, t = 0; i < colx2; i++, t++)
        {
            pangulu_int64_t rowx = x_columnindex[i];
            pangulu_int64_t colu1 = u_rowpointer[rowx];
            pangulu_int64_t colu2 = u_rowpointer[rowx + 1];

            x_value[i] = a_value[i] / u_value[colu1];
            calculate_type valx = x_value[i];
            for (pangulu_int64_t j = colu1 + 1 + threadIdx.x, p = threadIdx.x; j < colu2; j += blockDim.x, p += blockDim.x)
            {
                // update a's value;
                pangulu_inblock_idx f = binarysearch_idx(a_columnindex, cola1 + t + 1 + p, cola2 - colu2 + j, u_columnindex[j]);
                a_value[f] -= valx * u_value[j];
            }
            __syncthreads();
        }
    }
    else
    {
        __shared__ pangulu_inblock_idx s_rowidxa[sm_len_dtstrf];
        __shared__ calculate_type s_valuea[sm_len_dtstrf];

        for (pangulu_int64_t i = threadIdx.x; i < cola2 - cola1; i += blockDim.x)
        {
            s_rowidxa[i] = a_columnindex[cola1 + i];
            s_valuea[i] = a_value[cola1 + i];
        }
        __syncthreads();

        for (pangulu_int64_t i = colx1, t = 0; i < colx2; i++, t++)
        {
            pangulu_int64_t rowx = x_columnindex[i];
            pangulu_int64_t colu1 = u_rowpointer[rowx];
            pangulu_int64_t colu2 = u_rowpointer[rowx + 1];

            x_value[i] = s_valuea[t] / u_value[colu1];
            calculate_type valx = x_value[i];
            for (pangulu_int64_t j = colu1 + 1 + threadIdx.x, p = threadIdx.x; j < colu2; j += blockDim.x, p += blockDim.x)
            {
                // update a's value;
                pangulu_inblock_idx f = binarysearch_idx(s_rowidxa, 1 + t + p, cola2 - cola1 - colu2 + j, u_columnindex[j]);
                s_valuea[f] -= valx * u_value[j];
            }
            __syncthreads();
        }

        for (pangulu_int64_t i = threadIdx.x; i < cola2 - cola1; i += blockDim.x)
        {
            a_value[cola1 + i] = s_valuea[i];
        }
    }
}

__global__ void syncfree_cuda_csr(
    pangulu_int64_t n,
    pangulu_int64_t rhs,
    int *degree,
    pangulu_inblock_ptr *l_columnpointer,
    pangulu_inblock_idx *l_rowindex,
    calculate_type *l_value,
    pangulu_inblock_ptr *x_rowpointer,
    pangulu_inblock_idx *x_colindex,
    calculate_type *x_value,
    calculate_type *a_value,
    int *d_id_extractor,
    calculate_type *d_left_sum)
{
    const int local_warp_id = threadIdx.x / warp_size;
    const int lane_id = (warp_size - 1) & threadIdx.x;

    int global_x_id = 0;
    if (!lane_id)
        global_x_id = atomicAdd(d_id_extractor, 1);
    global_x_id = __shfl_sync(0xffffffff, global_x_id, 0);

    if (global_x_id >= n)
        return;
    do
    {
        __threadfence_block();
    } while (degree[global_x_id] != 1);

    const pangulu_int64_t loc1 = x_rowpointer[global_x_id];
    const pangulu_int64_t loc2 = x_rowpointer[global_x_id + 1];
    const pangulu_int64_t start_ptr = l_columnpointer[global_x_id] + 1;
    const pangulu_int64_t stop_ptr = l_columnpointer[global_x_id + 1];

    calculate_type res = 0;

    for (pangulu_int64_t m = loc1 + lane_id; m < loc2; m += warp_size)
    {
        x_value[m] = a_value[m] - d_left_sum[m];
    }
    if (loc1 != loc2)
        for (pangulu_int64_t jj = start_ptr + lane_id; jj < stop_ptr; jj += warp_size)
        {
            const pangulu_inblock_idx myidx = l_rowindex[jj];
            const pangulu_int64_t mypos = x_rowpointer[myidx];
            const pangulu_int64_t mypos2 = x_rowpointer[myidx + 1];
            const calculate_type l_id_value = l_value[jj];

            if (mypos2 - mypos > 200)
            {
                for (pangulu_int64_t k = loc1; k < loc2; k++)
                {
                    pangulu_inblock_idx f = binarysearch_idx(x_colindex, mypos, mypos2, x_colindex[k]);
                    res = x_value[k] * l_id_value;
                    // res = 0;
                    atomicAdd(&d_left_sum[f], res);
                }
            }
            else
            {
                for (pangulu_int64_t p = mypos, k = loc1; p < mypos2 && k < loc2; p++, k++)
                {

                    if (x_colindex[p] == x_colindex[k])
                    {
                        res = x_value[k] * l_id_value;
                        atomicAdd(&d_left_sum[p], res);
                    }
                    else
                    {
                        k--;
                    }
                }
            }
            atomicSub(&degree[myidx], 1);
        }
    else
    {
        for (pangulu_int64_t jj = start_ptr + lane_id; jj < stop_ptr; jj += warp_size)
        {
            atomicSub(&degree[l_rowindex[jj]], 1);
        }
    }
}

__global__ void syncfree_cuda_csr_u(
    pangulu_int64_t n,
    pangulu_int64_t rhs,
    int *degree,
    pangulu_inblock_ptr *l_columnpointer,
    pangulu_inblock_idx *l_rowindex,
    calculate_type *l_value,
    pangulu_inblock_ptr *x_rowpointer,
    pangulu_inblock_idx *x_colindex,
    calculate_type *x_value,
    calculate_type *a_value,
    int *d_id_extractor,
    calculate_type *d_left_sum)
{
    const int local_warp_id = threadIdx.x / warp_size;
    const int lane_id = (warp_size - 1) & threadIdx.x;
    int global_x_id = 0;
    if (!lane_id)
        global_x_id = atomicAdd(d_id_extractor, 1);
    global_x_id = __shfl_sync(0xffffffff, global_x_id, 0);
    if (global_x_id >= n)
        return;
    do
    {
        __threadfence_block();
    } while (degree[global_x_id] != 1);

    const pangulu_int64_t loc1 = x_rowpointer[global_x_id];
    const pangulu_int64_t loc2 = x_rowpointer[global_x_id + 1];
    const pangulu_int64_t start_ptr = l_columnpointer[global_x_id] + 1;
    const pangulu_int64_t stop_ptr = l_columnpointer[global_x_id + 1];

    calculate_type res = 0;

    for (pangulu_int64_t m = loc1 + lane_id; m < loc2; m += warp_size)
    {
        x_value[m] = (a_value[m] - d_left_sum[m]) / l_value[start_ptr - 1];
    }
    if (loc1 != loc2)
        for (pangulu_int64_t jj = start_ptr + lane_id; jj < stop_ptr; jj += warp_size)
        {
            const pangulu_inblock_idx myidx = l_rowindex[jj];
            const pangulu_int64_t mypos = x_rowpointer[myidx];
            const pangulu_int64_t mypos2 = x_rowpointer[myidx + 1];
            const calculate_type l_id_value = l_value[jj];

            if (mypos2 - mypos > 200)
            {
                for (pangulu_int64_t k = loc1; k < loc2; k++)
                {
                    pangulu_inblock_idx f = binarysearch_idx(x_colindex, mypos, mypos2, x_colindex[k]);
                    res = x_value[k] * l_id_value;
                    atomicAdd(&d_left_sum[f], res);
                }
            }
            else
            {
                for (pangulu_int64_t p = mypos, k = loc1; p < mypos2 && k < loc2; p++, k++)
                {
                    if (x_colindex[p] == x_colindex[k])
                    {
                        res = x_value[k] * l_id_value;
                        atomicAdd(&d_left_sum[p], res);
                    }
                    else
                    {
                        k--;
                    }
                }
            }

            atomicSub(&degree[myidx], 1);
        }
    else
    {
        for (pangulu_int64_t jj = start_ptr + lane_id; jj < stop_ptr; jj += warp_size)
        {
            atomicSub(&degree[l_rowindex[jj]], 1);
        }
    }
}
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
                                  calculate_type *a_value)
{

    int num_threads = warp_per_block_dgessm * warp_size;
    int num_blocks = ceil((calculate_type)n / (calculate_type)warp_per_block_dgessm);
    syncfree_cuda_csr_u<<<num_blocks, num_threads>>>(n, n, degree, u_rowpointer, u_columnindex, u_value,
                                                     a_rowpointer, a_columnindex, x_value, a_value, d_id_extractor, d_left_sum);
    cudaDeviceSynchronize();
}

__global__ void sptrsv_syncfree_cuda_executor_update(const pangulu_inblock_ptr *d_csccolptr,
                                                     const pangulu_inblock_idx *d_cscrowidx,
                                                     const calculate_type *d_cscval,
                                                     int *d_graphindegree,
                                                     calculate_type *d_left_sum,
                                                     const pangulu_int64_t m,
                                                     const pangulu_int64_t substitution,
                                                     const calculate_type *d_b,
                                                     calculate_type *d_x,
                                                     int *d_while_profiler,
                                                     int *d_id_extractor)
{
    const pangulu_int64_t global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // initialize
    const pangulu_int64_t local_warp_id = threadIdx.x / warp_size;
    const pangulu_int64_t lane_id = (warp_size - 1) & threadIdx.x;
    pangulu_int64_t global_x_id = 0;
    if (!lane_id)
        global_x_id = atomicAdd(d_id_extractor, 1);
    global_x_id = __shfl_sync(0xffffffff, global_x_id, 0);

    if (global_x_id >= m)
        return;

    // substitution is forward or backward
    global_x_id = substitution == substitution_forward ? global_x_id : m - 1 - global_x_id;
    // prefetch
    const pangulu_int64_t pos = substitution == substitution_forward ? d_csccolptr[global_x_id] : d_csccolptr[global_x_id + 1] - 1;
    const calculate_type coef = (calculate_type)1 / d_cscval[pos];
    // consumer
    do
    {
        __threadfence_block();
    } while (d_graphindegree[global_x_id] != 1);

    calculate_type xi = d_left_sum[global_x_id];
    xi = (d_b[global_x_id] - xi) * coef;

    // producer
    const pangulu_int64_t start_ptr = substitution == substitution_forward ? d_csccolptr[global_x_id] + 1 : d_csccolptr[global_x_id];
    const pangulu_int64_t stop_ptr = substitution == substitution_forward ? d_csccolptr[global_x_id + 1] : d_csccolptr[global_x_id + 1] - 1;
    for (pangulu_int64_t jj = start_ptr + lane_id; jj < stop_ptr; jj += warp_size)
    {
        const pangulu_int64_t j = substitution == substitution_forward ? jj : stop_ptr - 1 - (jj - start_ptr);
        const pangulu_inblock_idx rowidx = d_cscrowidx[j];

        atomicAdd(&d_left_sum[rowidx], xi * d_cscval[j]);
        __threadfence();
        atomicSub(&d_graphindegree[rowidx], 1);
    }

    // finish
    if (!lane_id)
        d_x[global_x_id] = xi;
}
__global__ void sptrsm_syncfree_cuda_executor_update(const pangulu_inblock_ptr *__restrict__ d_csccolptr,
                                                     const pangulu_inblock_idx *__restrict__ d_cscrowidx,
                                                     const calculate_type *__restrict__ d_cscval,
                                                     int *d_graphindegree,
                                                     calculate_type *d_left_sum,
                                                     const pangulu_int64_t m,
                                                     const int substitution,
                                                     const int rhs,

                                                     const calculate_type *__restrict__ d_b,
                                                     calculate_type *d_x,
                                                     int *d_while_profiler,
                                                     int *d_id_extractor)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    // initialize
    const int local_warp_id = threadIdx.x / warp_size;
    const int lane_id = (warp_size - 1) & threadIdx.x;
    int global_x_id = 0;
    if (!lane_id)
        global_x_id = atomicAdd(d_id_extractor, 1);
    global_x_id = __shfl_sync(0xffffffff, global_x_id, 0);

    if (global_x_id >= m)
        return;

    // substitution is forward or backward
    global_x_id = substitution == substitution_forward ? global_x_id : m - 1 - global_x_id;
    // prefetch
    const int pos = substitution == substitution_forward ? d_csccolptr[global_x_id] : d_csccolptr[global_x_id + 1] - 1;
    const calculate_type coef = (calculate_type)1 / d_cscval[pos];
    // consumer
    do
    {
        __threadfence_block();
    } while (1 != d_graphindegree[global_x_id]);

    for (int k = lane_id; k < rhs; k += warp_size)
    {
        const int pos = global_x_id * rhs + k;
        d_x[pos] = (d_b[pos] - d_left_sum[pos]) * coef;
    }

    // producer
    const pangulu_int64_t start_ptr = substitution == substitution_forward ? d_csccolptr[global_x_id] + 1 : d_csccolptr[global_x_id];
    const pangulu_int64_t stop_ptr = substitution == substitution_forward ? d_csccolptr[global_x_id + 1] : d_csccolptr[global_x_id + 1] - 1;

    for (pangulu_int64_t jj = start_ptr + lane_id; jj < stop_ptr; jj += warp_size)
    {
        const pangulu_int64_t j = substitution == substitution_forward ? jj : stop_ptr - 1 - (jj - start_ptr);
        const pangulu_inblock_idx rowidx = d_cscrowidx[j];
        for (int k = 0; k < rhs; k++)
            atomicAdd(&d_left_sum[rowidx * rhs + k], d_x[global_x_id * rhs + k] * d_cscval[j]);
        __threadfence();
        atomicSub(&d_graphindegree[rowidx], 1);
    }
}
__global__ void sptrsv_sparse_to_dense(const pangulu_inblock_ptr *d_cscptra,
                                       const pangulu_inblock_idx *d_cscidx,
                                       const calculate_type *d_value,
                                       pangulu_inblock_ptr *spointer,
                                       const pangulu_int64_t rhs,

                                       calculate_type *d_b)
{
    pangulu_int64_t global_x_id = blockIdx.x;

    if (global_x_id >= rhs)
        return;
    else
    {
        for (pangulu_int64_t i = d_cscptra[spointer[global_x_id]] + threadIdx.x; i < d_cscptra[spointer[global_x_id] + 1]; i += warp_size)
        {
            d_b[global_x_id + d_cscidx[i] * rhs] = d_value[i];
        }
    }
}
__global__ void sptrsv_dense_to_sparse(const pangulu_inblock_ptr *d_cscptra,
                                       const pangulu_inblock_idx *d_cscidx,
                                       calculate_type *d_value,
                                       pangulu_inblock_ptr *spointer,

                                       const pangulu_int64_t rhs,

                                       calculate_type *d_x)
{
    pangulu_int64_t global_x_id = blockIdx.x;

    if (global_x_id >= rhs)
        return;
    else
    {
        pangulu_int64_t index = spointer[global_x_id];
        for (pangulu_int64_t i = d_cscptra[index] + threadIdx.x; i < d_cscptra[index + 1]; i += warp_size)
        {
            d_value[i] = d_x[global_x_id + d_cscidx[i] * rhs];
        }
    }
}

void pangulu_gessm_cuda_kernel_v9(pangulu_int64_t n,
                                  pangulu_int64_t nnzl,
                                  pangulu_int64_t rhs,
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
                                  pangulu_inblock_ptr *a_columnpointer_cuda,
                                  pangulu_inblock_idx *a_rowindex_cuda,
                                  calculate_type *a_value_cuda,
                                  calculate_type *d_left_sum,
                                  calculate_type *d_x,
                                  calculate_type *d_b)
{

    /*********************************************pre***********************************************/
    pangulu_int64_t num_threads = 8 * warp_size;
    pangulu_int64_t num_blocks = rhs;
    sptrsv_sparse_to_dense<<<num_blocks, num_threads>>>(a_columnpointer_cuda, a_rowindex_cuda, a_value_cuda, d_spointer, rhs, d_b);
    cudaDeviceSynchronize();

    int substitution = 1;
    /*********************************************calculate***********************************************/
    if (rhs == 1)
    {
        num_threads = 16 * warp_size;

        num_blocks = ceil((calculate_type)n / (calculate_type)(num_threads / warp_size));
        sptrsv_syncfree_cuda_executor_update<<<num_blocks, num_threads>>>(l_columnpointer, l_rowindex, l_value,
                                                                          d_graphindegree, d_left_sum,
                                                                          n, substitution, d_b, d_x, d_while_profiler, d_id_extractor);
    }
    else
    {
        num_threads = 4 * warp_size;
        num_blocks = ceil((calculate_type)n / (calculate_type)(num_threads / warp_size));
        sptrsm_syncfree_cuda_executor_update<<<num_blocks, num_threads>>>(l_columnpointer, l_rowindex, l_value,
                                                                          d_graphindegree, d_left_sum,
                                                                          n, substitution, rhs,
                                                                          d_b, d_x, d_while_profiler, d_id_extractor);
    }
    cudaDeviceSynchronize();
    /*********************************************calculate***********************************************/

    num_threads = 8 * warp_size;
    num_blocks = rhs;
    sptrsv_dense_to_sparse<<<num_blocks, num_threads>>>(a_columnpointer_cuda, a_rowindex_cuda, x_value, d_spointer, rhs, d_x);
    cudaDeviceSynchronize();
}
void pangulu_tstrf_cuda_kernel_v9(pangulu_int64_t n,
                                  pangulu_int64_t nnzl,
                                  pangulu_int64_t rhs,
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
                                  pangulu_inblock_ptr *a_columnpointer_cuda,
                                  pangulu_inblock_idx *a_rowindex_cuda,
                                  calculate_type *a_value_cuda,
                                  calculate_type *d_left_sum,
                                  calculate_type *d_x,
                                  calculate_type *d_b)
{

    /*********************************************pre***********************************************/
    pangulu_int64_t num_threads = 8 * warp_size;
    pangulu_int64_t num_blocks = rhs;
    sptrsv_sparse_to_dense<<<num_blocks, num_threads>>>(a_columnpointer_cuda, a_rowindex_cuda, a_value_cuda, d_spointer, rhs, d_b);
    cudaDeviceSynchronize();
    int substitution = 1;
    /*********************************************calculate***********************************************/
    if (rhs == 1)
    {
        num_threads = 16 * warp_size;

        num_blocks = ceil((calculate_type)n / (calculate_type)(num_threads / warp_size));
        sptrsv_syncfree_cuda_executor_update<<<num_blocks, num_threads>>>(l_columnpointer, l_rowindex, l_value,
                                                                          d_graphindegree, d_left_sum,
                                                                          n, substitution, d_b, d_x, d_while_profiler, d_id_extractor);
    }
    else
    {
        num_threads = 4 * warp_size;
        num_blocks = ceil((calculate_type)n / (calculate_type)(num_threads / warp_size));
        sptrsm_syncfree_cuda_executor_update<<<num_blocks, num_threads>>>(l_columnpointer, l_rowindex, l_value,
                                                                          d_graphindegree, d_left_sum,
                                                                          n, substitution, rhs,
                                                                          d_b, d_x, d_while_profiler, d_id_extractor);
    }
    cudaDeviceSynchronize();
    /*********************************************calculate***********************************************/

    num_threads = 8 * warp_size;
    num_blocks = rhs;
    sptrsv_dense_to_sparse<<<num_blocks, num_threads>>>(a_columnpointer_cuda, a_rowindex_cuda, x_value, d_spointer, rhs, d_x);
    cudaDeviceSynchronize();
}

__global__ void gessm_kernel_dense(pangulu_int64_t n,
                                   pangulu_inblock_ptr *l_columnpointer,
                                   pangulu_inblock_idx *l_rowindex,
                                   calculate_type *l_value,
                                   pangulu_inblock_ptr *x_columnpointer,
                                   pangulu_inblock_idx *x_rowindex,
                                   calculate_type *x_value,
                                   pangulu_inblock_ptr *a_columnpointer,
                                   pangulu_inblock_idx *a_rowindex,
                                   calculate_type *a_value,
                                   calculate_type *dense)
{

    pangulu_int64_t colidx = blockIdx.x;
    pangulu_int64_t colx1 = x_columnpointer[colidx];
    pangulu_int64_t colx2 = x_columnpointer[colidx + 1];

    if (colidx >= n || colx2 == colx1)
        return;

    pangulu_int64_t cola1 = a_columnpointer[colidx];
    pangulu_int64_t cola2 = a_columnpointer[colidx + 1];

    for (pangulu_int64_t i = colx1, t = 0; i < colx2; i++, t++)
    {
        pangulu_int64_t rowx = x_rowindex[i];
        x_value[i] = dense[colidx * n + rowx];
        calculate_type valx = x_value[i];

        pangulu_int64_t coll1 = l_columnpointer[rowx];
        pangulu_int64_t coll2 = l_columnpointer[rowx + 1];

        for (pangulu_int64_t j = coll1 + 1 + threadIdx.x, p = threadIdx.x; j < coll2; j += blockDim.x, p += blockDim.x)
        {
            dense[colidx * n + l_rowindex[j]] -= valx * l_value[j];
        }
        __syncthreads();
    }
}

__global__ void tstrf_kernel_dense(pangulu_int64_t n,
                                   pangulu_inblock_ptr *u_rowpointer,
                                   pangulu_inblock_idx *u_columnindex,
                                   calculate_type *u_value,
                                   pangulu_inblock_ptr *x_rowpointer,
                                   pangulu_inblock_idx *x_columnindex,
                                   calculate_type *x_value,
                                   pangulu_inblock_ptr *a_rowpointer,
                                   pangulu_inblock_idx *a_columnindex,
                                   calculate_type *a_value,
                                   calculate_type *dense)
{
    pangulu_int64_t colidx = blockIdx.x;
    pangulu_int64_t colx1 = x_rowpointer[colidx];
    pangulu_int64_t colx2 = x_rowpointer[colidx + 1];

    if (colidx >= n || colx1 == colx2)
        return;

    pangulu_int64_t cola1 = a_rowpointer[colidx];
    pangulu_int64_t cola2 = a_rowpointer[colidx + 1];

    for (pangulu_int64_t i = colx1, t = 0; i < colx2; i++, t++)
    {
        pangulu_int64_t rowx = x_columnindex[i];
        pangulu_int64_t colu1 = u_rowpointer[rowx];
        pangulu_int64_t colu2 = u_rowpointer[rowx + 1];
        dense[colidx * n + rowx] /= u_value[colu1];
        x_value[i] = dense[colidx * n + rowx];

        calculate_type valx = x_value[i];
        for (pangulu_int64_t j = colu1 + 1 + threadIdx.x, p = threadIdx.x; j < colu2; j += blockDim.x, p += blockDim.x)
        {
            dense[colidx * n + u_columnindex[j]] -= valx * u_value[j];
        }
        __syncthreads();
    }
}

__global__ void syncfree_cuda_csr_dense_v11_l(
    pangulu_int64_t n,
    pangulu_int64_t rhs,
    int *degree,
    pangulu_inblock_ptr *l_columnpointer,
    pangulu_inblock_idx *l_rowindex,
    calculate_type *l_value,
    pangulu_inblock_ptr *x_rowpointer,
    pangulu_inblock_idx *x_colindex,
    calculate_type *x_value,
    calculate_type *a_value,
    int *d_id_extractor,
    calculate_type *d_left_sum,
    calculate_type *dense)
{
    const int local_warp_id = threadIdx.x / warp_size;
    const int lane_id = (warp_size - 1) & threadIdx.x;

    int global_x_id = 0;
    if (!lane_id)
        global_x_id = atomicAdd(d_id_extractor, 1);
    global_x_id = __shfl_sync(0xffffffff, global_x_id, 0);

    if (global_x_id >= n)
        return;
    do
    {
        __threadfence_block();
    } while (degree[global_x_id] != 1);

    const pangulu_int64_t loc1 = x_rowpointer[global_x_id];
    const pangulu_int64_t loc2 = x_rowpointer[global_x_id + 1];
    const pangulu_int64_t start_ptr = l_columnpointer[global_x_id] + 1;
    const pangulu_int64_t stop_ptr = l_columnpointer[global_x_id + 1];

    calculate_type res = 0;

    for (pangulu_int64_t m = loc1 + lane_id; m < loc2; m += warp_size)
    {
        x_value[m] = dense[global_x_id * n + x_colindex[m]];
    }
    if (loc1 != loc2)
        for (pangulu_int64_t jj = start_ptr + lane_id; jj < stop_ptr; jj += warp_size)
        {
            const pangulu_inblock_idx myidx = l_rowindex[jj];
            const pangulu_int64_t mypos = x_rowpointer[myidx];
            const pangulu_int64_t mypos2 = x_rowpointer[myidx + 1];
            const calculate_type l_id_value = l_value[jj];

            for (pangulu_int64_t k = loc1; k < loc2; k++)
            {
                res = -(x_value[k] * l_id_value);

                atomicAdd(&dense[myidx * n + x_colindex[k]], res);
            }
            atomicSub(&degree[myidx], 1);
        }
    else
    {
        for (pangulu_int64_t jj = start_ptr + lane_id; jj < stop_ptr; jj += warp_size)
        {
            atomicSub(&degree[l_rowindex[jj]], 1);
        }
    }
}

void pangulu_gessm_cuda_kernel_v11(pangulu_int64_t n,
                                   pangulu_int64_t nnzl,
                                   pangulu_int64_t nnzx,
                                   int *degree,
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
                                   calculate_type *a_value)
{
    pangulu_int64_t num_threads = warp_size * warp_per_block;
    pangulu_int64_t num_blocks;

    num_blocks = ceil((calculate_type)n / warp_per_block);
    cuda_transform_s_to_d_col<<<num_blocks, num_threads>>>(n, num_blocks, a_columnpointer, a_rowindex, a_value, cuda_temp_value);
    cudaDeviceSynchronize();

    num_threads = warp_per_block_dgessm * warp_size;
    num_blocks = ceil((calculate_type)n / (calculate_type)warp_per_block_dgessm);
    syncfree_cuda_csr_dense_v11_l<<<num_blocks, num_threads>>>(n, n, degree, l_columnpointer, l_rowindex, l_value,
                                                               a_columnpointer, a_rowindex, x_value, a_value, d_id_extractor, d_left_sum, cuda_temp_value);
    cudaDeviceSynchronize();
}

__global__ void syncfree_cuda_csr_dense_v11_u(
    pangulu_int64_t n,
    pangulu_int64_t rhs,
    int *degree,
    pangulu_inblock_ptr *l_columnpointer,
    pangulu_inblock_idx *l_rowindex,
    calculate_type *l_value,
    pangulu_inblock_ptr *x_rowpointer,
    pangulu_inblock_idx *x_colindex,
    calculate_type *x_value,
    calculate_type *a_value,
    int *d_id_extractor,
    calculate_type *d_left_sum,
    calculate_type *dense)
{
    const int local_warp_id = threadIdx.x / warp_size;
    const int lane_id = (warp_size - 1) & threadIdx.x;
    int global_x_id = 0;
    if (!lane_id)
        global_x_id = atomicAdd(d_id_extractor, 1);
    global_x_id = __shfl_sync(0xffffffff, global_x_id, 0);
    if (global_x_id >= n)
        return;
    do
    {
        __threadfence_block();
    } while (degree[global_x_id] != 1);

    const pangulu_int64_t loc1 = x_rowpointer[global_x_id];
    const pangulu_int64_t loc2 = x_rowpointer[global_x_id + 1];
    const pangulu_int64_t start_ptr = l_columnpointer[global_x_id] + 1;
    const pangulu_int64_t stop_ptr = l_columnpointer[global_x_id + 1];

    calculate_type res = 0;

    for (pangulu_int64_t m = loc1 + lane_id; m < loc2; m += warp_size)
    {
        dense[global_x_id * n + x_colindex[m]] /= l_value[start_ptr - 1];

        x_value[m] = dense[global_x_id * n + x_colindex[m]];
    }
    if (loc1 != loc2)
        for (pangulu_int64_t jj = start_ptr + lane_id; jj < stop_ptr; jj += warp_size)
        {
            const pangulu_inblock_idx myidx = l_rowindex[jj];
            const pangulu_int64_t mypos = x_rowpointer[myidx];
            const pangulu_int64_t mypos2 = x_rowpointer[myidx + 1];
            const calculate_type l_id_value = l_value[jj];
            for (pangulu_int64_t k = loc1; k < loc2; k++)
            {
                res = -(x_value[k] * l_id_value);

                atomicAdd(&dense[myidx * n + x_colindex[k]], res);
            }
            atomicSub(&degree[myidx], 1);
        }
    else
    {
        for (pangulu_int64_t jj = start_ptr + lane_id; jj < stop_ptr; jj += warp_size)
        {
            atomicSub(&degree[l_rowindex[jj]], 1);
        }
    }
}

void pangulu_cuda_malloc(void **cuda_address,
                         size_t size)
{
    gpu_memory += size;
    if (cudaSuccess != cudaMalloc((cuda_address), size))
    {
        printf(PANGULU_E_CUDA_MALLOC);
        pangulu_exit(1);
    }
}

void pangulu_cuda_free(void *cuda_address)
{
    cudaFree(cuda_address);
    return;
}

void pangulu_cuda_memcpy_host_to_device_value(calculate_type *cuda_address,
                                              calculate_type *cpu_address,
                                              size_t size)
{
    cudaMemcpy(cuda_address,
               cpu_address,
               size * sizeof(calculate_type),
               cudaMemcpyHostToDevice);
}

void pangulu_cuda_memcpy_device_to_host_value(calculate_type *cpu_address,
                                              calculate_type *cuda_address,
                                              size_t size)
{
    cudaMemcpy(cpu_address,
               cuda_address,
               size * sizeof(calculate_type),
               cudaMemcpyDeviceToHost);
}

void pangulu_cuda_memcpy_host_to_device_inblock_idx(
    pangulu_inblock_idx *cuda_address,
    pangulu_inblock_idx *cpu_address,
    size_t size
)
{
    cudaMemcpy(cuda_address,
               cpu_address,
               size * sizeof(pangulu_inblock_idx),
               cudaMemcpyHostToDevice);
}

void pangulu_cuda_memcpy_host_to_device_inblock_ptr(
    pangulu_inblock_ptr *cuda_address,
    pangulu_inblock_ptr *cpu_address,
    size_t size
)
{
    cudaMemcpy(cuda_address,
               cpu_address,
               size * sizeof(pangulu_inblock_ptr),
               cudaMemcpyHostToDevice);
}

void pangulu_cuda_memcpy_host_to_device_int32(pangulu_int32_t *cuda_address,
                                              pangulu_int32_t *cpu_address,
                                              size_t size)
{
    cudaMemcpy(cuda_address,
               cpu_address,
               size * sizeof(pangulu_int32_t),
               cudaMemcpyHostToDevice);
}

void pangulu_cuda_memcpy_device_to_host_int(pangulu_inblock_ptr *cpu_address,
                                            pangulu_inblock_ptr *cuda_address,
                                            size_t size)
{
    cudaMemcpy(cpu_address,
               cuda_address,
               size * sizeof(pangulu_inblock_ptr),
               cudaMemcpyDeviceToHost);
}

void pangulu_cuda_devicesynchronize()
{
    cudaDeviceSynchronize();
}

void pangulu_cuda_getdevicenum(pangulu_int32_t *gpu_num)
{
    cudaGetDeviceCount(gpu_num);
}

pangulu_int32_t pangulu_cuda_setdevice(pangulu_int32_t gpu_num,
                               pangulu_int32_t rank)
{
    pangulu_int32_t usr_id = rank % gpu_num;
    cudaSetDevice(usr_id);
    return usr_id;
}

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
                               calculate_type *d_cscvalueu)
{
    pangulu_int64_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    pangulu_int64_t warp_local_id = threadIdx.x / 32;
    pangulu_int64_t lane_id = threadIdx.x % 32;

    if (warp_id >= n)
        return;

    const pangulu_int64_t colidx = warp_id;

    pangulu_int64_t cola = d_csccolptra[colidx];
    pangulu_int64_t cola1 = d_csccolptra[colidx + 1];
    pangulu_int64_t colu = d_csccolptru[colidx];
    pangulu_int64_t colu1 = d_csccolptru[colidx + 1];
    pangulu_int64_t coll = d_csccolptrl[colidx];
    pangulu_int64_t coll1 = d_csccolptrl[colidx + 1];
    pangulu_int64_t loadlen = cola1 - cola;

    // use shared memory
    if (loadlen <= sm_len_warplev)
    {
        __shared__ pangulu_inblock_idx s_idxa[warp_num_warplu * sm_len_warplev];
        __shared__ calculate_type s_vala[warp_num_warplu * sm_len_warplev];

        pangulu_inblock_idx *s_idxa_local = &s_idxa[warp_local_id * sm_len_warplev];
        calculate_type *s_vala_local = &s_vala[warp_local_id * sm_len_warplev];

        for (pangulu_int64_t i = lane_id; i < loadlen; i += warp_size)
        {
            s_idxa_local[i] = d_cscrowidxa[cola + i];
            s_vala_local[i] = d_cscvaluea[cola + i];
        }

        // step one
        for (pangulu_int64_t i = cola; i < cola + colu1 - colu - 1; i++)
        {
            const pangulu_int64_t rowidx = d_cscrowidxa[i];
            calculate_type value3 = s_vala_local[i - cola];
            // busy-wait until nnzu[rowidx]==0
            do
            {
                __threadfence();
            } while (d_nnzu[rowidx] != -1);

            pangulu_int64_t rowa = d_csccolptra[rowidx];
            pangulu_int64_t rowa1 = d_csccolptra[rowidx + 1];
            pangulu_int64_t rowu = d_csccolptru[rowidx];
            pangulu_int64_t rowu1 = d_csccolptru[rowidx + 1];

            for (pangulu_int64_t j = rowa + rowu1 - rowu + lane_id; j < rowa1; j += warp_size)
            {
                const pangulu_int64_t lrowindex = d_cscrowidxa[j];
                const pangulu_int64_t thecolidx = rowidx;
                pangulu_int64_t flag1 = binarysearch_idx(s_idxa_local, 0, loadlen - 1, lrowindex);
                pangulu_int64_t flag2 = binarysearch_idx(d_cscrowidxa, d_csccolptra[thecolidx] + d_csccolptru[thecolidx + 1] - d_csccolptru[thecolidx], d_csccolptra[thecolidx + 1] - 1, lrowindex);

                s_vala_local[flag1] -= d_cscvaluea[flag2] * value3;
            }

            if (!lane_id)
            {
                atomicSub(&d_nnzu[colidx], 1);
            }
        }

        // step two
        pangulu_int64_t flag5 = colu1 - colu - 1;
        if (s_vala_local[flag5] > ERROR || s_vala_local[flag5] < -ERROR)
        {
        }
        else
        {
            s_vala_local[flag5] = ERROR;
        }
        calculate_type value5 = s_vala_local[flag5];

        d_cscvaluel[coll] = 1;
        for (pangulu_int64_t i = cola + colu1 - colu + lane_id; i < cola1; i += warp_size)
        {
            const pangulu_int64_t lrowindex = s_idxa_local[i - cola];
            pangulu_int64_t flagl = coll + i - (cola + colu1 - colu) + 1;
            s_vala_local[i - cola] = s_vala_local[i - cola] / value5;
            d_cscvaluel[flagl] = s_vala_local[i - cola];
        }

        for (pangulu_int64_t i = lane_id; i < colu1 - colu; i += warp_size)
        {
            d_cscvalueu[colu + i] = s_vala_local[i];
        }

        for (pangulu_int64_t i = lane_id; i < loadlen; i += warp_size)
        {
            d_cscvaluea[i + cola] = s_vala_local[i];
        }

        if (!lane_id)
        {
            atomicSub(&d_nnzu[colidx], 1);
        }
    }
    // do not use shared memory
    else
    {
        // step one
        for (pangulu_int64_t i = cola; i < cola + colu1 - colu - 1; i++)
        {
            const pangulu_int64_t rowidx = d_cscrowidxa[i];
            calculate_type value3 = d_cscvaluea[i];
            // busy-wait until nnzu[rowidx]==0
            do
            {
                __threadfence();
            } while (d_nnzu[rowidx] != -1);

            pangulu_int64_t rowa = d_csccolptra[rowidx];
            pangulu_int64_t rowa1 = d_csccolptra[rowidx + 1];
            pangulu_int64_t rowu = d_csccolptru[rowidx];
            pangulu_int64_t rowu1 = d_csccolptru[rowidx + 1];

            for (pangulu_int64_t j = rowa + rowu1 - rowu + lane_id; j < rowa1; j += warp_size)
            {
                const pangulu_int64_t lrowindex = d_cscrowidxa[j];
                const pangulu_int64_t thecolidx = rowidx;
                pangulu_int64_t flag1 = binarysearch_idx(d_cscrowidxa, cola, cola1 - 1, lrowindex);
                pangulu_int64_t flag2 = binarysearch_idx(d_cscrowidxa, d_csccolptra[thecolidx] + d_csccolptru[thecolidx + 1] - d_csccolptru[thecolidx], d_csccolptra[thecolidx + 1] - 1, lrowindex);

                d_cscvaluea[flag1] -= d_cscvaluea[flag2] * value3;
            }

            if (!lane_id)
            {
                atomicSub(&d_nnzu[colidx], 1);
            }
        }

        for (pangulu_int64_t i = lane_id; i < colu1 - colu; i += warp_size)
        {
            d_cscvalueu[colu + i] = d_cscvaluea[cola + i];
        }

        // step two
        pangulu_int64_t flag5 = cola + colu1 - colu - 1;
        if (d_cscvaluea[flag5] > ERROR || d_cscvaluea[flag5] < -ERROR)
        {
        }
        else
        {
            d_cscvaluea[flag5] = ERROR;
        }

        calculate_type value5 = d_cscvaluea[flag5];
        d_cscvaluel[coll] = 1;
        for (pangulu_int64_t i = cola + colu1 - colu + lane_id; i < cola1; i += warp_size)
        {
            const pangulu_int64_t lrowindex = d_cscrowidxa[i];
            pangulu_int64_t flagl = coll + i - (cola + colu1 - colu) + 1;
            d_cscvaluea[i] = d_cscvaluea[i] / value5;
            d_cscvaluel[flagl] = d_cscvaluea[i];
        }

        if (!lane_id)
        {
            atomicSub(&d_nnzu[colidx], 1);
        }
    }
}

__global__ void blocklevel_sflu_l1(pangulu_int64_t n,
                                   pangulu_int32_t *d_nnzu,
                                   pangulu_inblock_ptr *d_csccolptra,
                                   pangulu_inblock_idx *d_cscrowidxa,
                                   calculate_type *d_cscvaluea,
                                   pangulu_inblock_ptr *d_csccolptrl,
                                   pangulu_inblock_idx *d_cscrowidxl,
                                   calculate_type *d_cscvaluel,
                                   pangulu_inblock_ptr *d_csccolptru,
                                   pangulu_inblock_idx *d_cscrowidxu,
                                   calculate_type *d_cscvalueu)
{
    const pangulu_int64_t colidx = blockIdx.x;

    __shared__ int s_nnzu[1];

    pangulu_int64_t cola = d_csccolptra[colidx];
    pangulu_int64_t cola1 = d_csccolptra[colidx + 1];
    pangulu_int64_t smemlen = 256;
    pangulu_int64_t len_a = cola1 - cola;

    // use shared memory
    if (len_a <= smemlen)
    {
        __shared__ pangulu_inblock_idx s_rowidxa[256];
        __shared__ calculate_type s_valuea[256];

        for (pangulu_int64_t i = threadIdx.x; i < len_a; i += blockDim.x)
        {
            s_rowidxa[i] = d_cscrowidxa[i + cola];
            s_valuea[i] = d_cscvaluea[i + cola];
        }

        if (!threadIdx.x)
        {
            s_nnzu[0] = d_nnzu[colidx];
        }
        __syncthreads();

        pangulu_int64_t colu = d_csccolptru[colidx];
        pangulu_int64_t colu1 = d_csccolptru[colidx + 1];
        pangulu_int64_t coll = d_csccolptrl[colidx];
        pangulu_int64_t coll1 = d_csccolptrl[colidx + 1];

        // step one
        for (pangulu_int64_t i = cola; i < cola + colu1 - colu - 1; i++)
        {
            const pangulu_int64_t rowidx = d_cscrowidxa[i];
            calculate_type value3 = s_valuea[i - cola];

            // busy-wait until nnzu[rowidx]==0
            if (!threadIdx.x)
            {
                do
                {
                    __threadfence_block();
                } while (d_nnzu[rowidx] != -1);
            }
            __syncthreads();

            pangulu_int64_t rowa = d_csccolptra[rowidx];
            pangulu_int64_t rowa1 = d_csccolptra[rowidx + 1];
            pangulu_int64_t rowu = d_csccolptru[rowidx];
            pangulu_int64_t rowu1 = d_csccolptru[rowidx + 1];

            for (pangulu_int64_t j = rowa + rowu1 - rowu + threadIdx.x; j < rowa1; j += blockDim.x)
            {
                const pangulu_int64_t lrowindex = d_cscrowidxa[j];
                const pangulu_int64_t thecolidx = rowidx;

                pangulu_int64_t flag1 = binarysearch_idx(s_rowidxa, 0, len_a - 1, lrowindex);
                pangulu_int64_t flag2 = binarysearch_idx(d_cscrowidxa, d_csccolptra[thecolidx] + d_csccolptru[thecolidx + 1] - d_csccolptru[thecolidx], d_csccolptra[thecolidx + 1] - 1, lrowindex);

                s_valuea[flag1] -= d_cscvaluea[flag2] * value3;
            }

            if (!threadIdx.x)
            {
                atomicSub(&s_nnzu[0], 1);
            }
            __syncthreads();
        }

        // step two
        pangulu_int64_t flag5 = colu1 - colu - 1;
        if (s_valuea[flag5] > ERROR || s_valuea[flag5] < -ERROR)
        {
        }
        else
        {
            s_valuea[flag5] = ERROR;
        }
        calculate_type value5 = s_valuea[flag5];

        d_cscvaluel[coll] = 1;
        for (pangulu_int64_t i = cola + colu1 - colu + threadIdx.x; i < cola1; i += blockDim.x)
        {
            const pangulu_int64_t lrowindex = s_rowidxa[i - cola];
            pangulu_int64_t flagl = coll + i - (cola + colu1 - colu) + 1;
            s_valuea[i - cola] = s_valuea[i - cola] / value5;
            d_cscvaluel[flagl] = s_valuea[i - cola];
        }
        __syncthreads();

        for (pangulu_int64_t i = threadIdx.x; i < colu1 - colu; i += blockDim.x)
        {
            d_cscvalueu[colu + i] = s_valuea[i];
        }

        for (pangulu_int64_t i = threadIdx.x; i < len_a; i += blockDim.x)
        {
            d_cscvaluea[i + cola] = s_valuea[i];
        }

        if (!threadIdx.x)
        {
            atomicSub(&s_nnzu[0], 1);
            d_nnzu[colidx] = s_nnzu[0];
        }
    }
    // do not use shared memory
    else
    {
        if (!threadIdx.x)
        {
            s_nnzu[0] = d_nnzu[colidx];
        }
        __syncthreads();

        pangulu_int64_t colu = d_csccolptru[colidx];
        pangulu_int64_t colu1 = d_csccolptru[colidx + 1];
        pangulu_int64_t coll = d_csccolptrl[colidx];
        pangulu_int64_t coll1 = d_csccolptrl[colidx + 1];

        // step one
        for (pangulu_int64_t i = cola; i < cola + colu1 - colu - 1; i++)
        {
            const pangulu_int64_t rowidx = d_cscrowidxa[i];
            calculate_type value3 = d_cscvaluea[i];
            // busy-wait until nnzu[rowidx]==0
            if (!threadIdx.x)
            {
                do
                {
                    __threadfence_block();
                } while (d_nnzu[rowidx] != -1);
            }
            __syncthreads();

            pangulu_int64_t rowa = d_csccolptra[rowidx];
            pangulu_int64_t rowa1 = d_csccolptra[rowidx + 1];
            pangulu_int64_t rowu = d_csccolptru[rowidx];
            pangulu_int64_t rowu1 = d_csccolptru[rowidx + 1];

            for (pangulu_int64_t j = rowa + rowu1 - rowu + threadIdx.x; j < rowa1; j += blockDim.x)
            {
                const pangulu_int64_t lrowindex = d_cscrowidxa[j];
                const pangulu_int64_t thecolidx = rowidx;
                pangulu_int64_t flag1 = binarysearch_idx(d_cscrowidxa, cola, cola1 - 1, lrowindex);
                pangulu_int64_t flag2 = binarysearch_idx(d_cscrowidxa, d_csccolptra[thecolidx] + d_csccolptru[thecolidx + 1] - d_csccolptru[thecolidx], d_csccolptra[thecolidx + 1] - 1, lrowindex);

                d_cscvaluea[flag1] -= d_cscvaluea[flag2] * value3;
            }

            if (!threadIdx.x)
            {
                atomicSub(&s_nnzu[0], 1);
            }
            __syncthreads();
        }

        for (pangulu_int64_t i = threadIdx.x; i < colu1 - colu; i += blockDim.x)
        {
            d_cscvalueu[colu + i] = d_cscvaluea[cola + i];
        }

        // step two
        pangulu_int64_t flag5 = cola + colu1 - colu - 1;
        if (d_cscvaluea[flag5] > ERROR || d_cscvaluea[flag5] < -ERROR)
        {
        }
        else
        {
            d_cscvaluea[flag5] = ERROR;
        }
        calculate_type value5 = d_cscvaluea[flag5];
        d_cscvaluel[coll] = 1;

        for (pangulu_int64_t i = cola + colu1 - colu + threadIdx.x; i < cola1; i += blockDim.x)
        {
            const pangulu_int64_t lrowindex = d_cscrowidxa[i];
            pangulu_int64_t flagl = coll + i - (cola + colu1 - colu) + 1;
            d_cscvaluea[i] = d_cscvaluea[i] / value5;
            d_cscvaluel[flagl] = d_cscvaluea[i];
        }

        if (!threadIdx.x)
        {
            atomicSub(&s_nnzu[0], 1);
            d_nnzu[colidx] = s_nnzu[0];
        }
    }
}

__global__ void blocklevel_sflu_l2(pangulu_int64_t n,
                                   pangulu_int32_t *d_nnzu,
                                   pangulu_inblock_ptr *d_csccolptra,
                                   pangulu_inblock_idx *d_cscrowidxa,
                                   calculate_type *d_cscvaluea,
                                   pangulu_inblock_ptr *d_csccolptrl,
                                   pangulu_inblock_idx *d_cscrowidxl,
                                   calculate_type *d_cscvaluel,
                                   pangulu_inblock_ptr *d_csccolptru,
                                   pangulu_inblock_idx *d_cscrowidxu,
                                   calculate_type *d_cscvalueu)
{
    const pangulu_int64_t colidx = blockIdx.x;

    __shared__ pangulu_int32_t s_nnzu[1];

    pangulu_int64_t cola = d_csccolptra[colidx];
    pangulu_int64_t cola1 = d_csccolptra[colidx + 1];
    pangulu_int64_t smemlen = 512;
    pangulu_int64_t len_a = cola1 - cola;

    // use shared memory
    if (len_a <= smemlen)
    {
        __shared__ pangulu_inblock_idx s_rowidxa[512];
        __shared__ calculate_type s_valuea[512];

        for (pangulu_int64_t i = threadIdx.x; i < len_a; i += blockDim.x)
        {
            s_rowidxa[i] = d_cscrowidxa[i + cola];
            s_valuea[i] = d_cscvaluea[i + cola];
        }

        if (!threadIdx.x)
        {
            s_nnzu[0] = d_nnzu[colidx];
        }
        __syncthreads();

        pangulu_int64_t colu = d_csccolptru[colidx];
        pangulu_int64_t colu1 = d_csccolptru[colidx + 1];
        pangulu_int64_t coll = d_csccolptrl[colidx];
        pangulu_int64_t coll1 = d_csccolptrl[colidx + 1];

        // step one
        for (pangulu_int64_t i = cola; i < cola + colu1 - colu - 1; i++)
        {
            const pangulu_int64_t rowidx = d_cscrowidxa[i];
            calculate_type value3 = s_valuea[i - cola];

            // busy-wait until nnzu[rowidx]==0
            if (!threadIdx.x)
            {
                do
                {
                    __threadfence_block();
                } while (d_nnzu[rowidx] != -1);
            }
            __syncthreads();

            pangulu_int64_t rowa = d_csccolptra[rowidx];
            pangulu_int64_t rowa1 = d_csccolptra[rowidx + 1];
            pangulu_int64_t rowu = d_csccolptru[rowidx];
            pangulu_int64_t rowu1 = d_csccolptru[rowidx + 1];

            for (pangulu_int64_t j = rowa + rowu1 - rowu + threadIdx.x; j < rowa1; j += blockDim.x)
            {
                const pangulu_int64_t lrowindex = d_cscrowidxa[j];
                const pangulu_int64_t thecolidx = rowidx;

                pangulu_int64_t flag1 = binarysearch_idx(s_rowidxa, 0, len_a - 1, lrowindex);
                pangulu_int64_t flag2 = binarysearch_idx(d_cscrowidxa, d_csccolptra[thecolidx] + d_csccolptru[thecolidx + 1] - d_csccolptru[thecolidx], d_csccolptra[thecolidx + 1] - 1, lrowindex);

                s_valuea[flag1] -= d_cscvaluea[flag2] * value3;
            }

            if (!threadIdx.x)
            {
                atomicSub(&s_nnzu[0], 1);
            }
            __syncthreads();
        }

        // step two
        pangulu_int64_t flag5 = colu1 - colu - 1;
        if (s_valuea[flag5] > ERROR || s_valuea[flag5] < -ERROR)
        {
        }
        else
        {
            s_valuea[flag5] = ERROR;
        }
        calculate_type value5 = s_valuea[flag5];

        d_cscvaluel[coll] = 1;
        for (pangulu_int64_t i = cola + colu1 - colu + threadIdx.x; i < cola1; i += blockDim.x)
        {
            const pangulu_int64_t lrowindex = s_rowidxa[i - cola];
            pangulu_int64_t flagl = coll + i - (cola + colu1 - colu) + 1;
            s_valuea[i - cola] = s_valuea[i - cola] / value5;
            d_cscvaluel[flagl] = s_valuea[i - cola];
        }
        __syncthreads();

        for (pangulu_int64_t i = threadIdx.x; i < colu1 - colu; i += blockDim.x)
        {
            d_cscvalueu[colu + i] = s_valuea[i];
        }

        for (pangulu_int64_t i = threadIdx.x; i < len_a; i += blockDim.x)
        {
            d_cscvaluea[i + cola] = s_valuea[i];
        }

        if (!threadIdx.x)
        {
            atomicSub(&s_nnzu[0], 1);
            d_nnzu[colidx] = s_nnzu[0];
        }
    }
    // do not use shared memory
    else
    {
        if (!threadIdx.x)
        {
            s_nnzu[0] = d_nnzu[colidx];
        }
        __syncthreads();

        pangulu_int64_t colu = d_csccolptru[colidx];
        pangulu_int64_t colu1 = d_csccolptru[colidx + 1];
        pangulu_int64_t coll = d_csccolptrl[colidx];
        pangulu_int64_t coll1 = d_csccolptrl[colidx + 1];

        // step one
        for (pangulu_int64_t i = cola; i < cola + colu1 - colu - 1; i++)
        {
            const pangulu_int64_t rowidx = d_cscrowidxa[i];

            calculate_type value3 = d_cscvaluea[i];

            // busy-wait until nnzu[rowidx]==0
            if (!threadIdx.x)
            {
                do
                {
                    __threadfence_block();
                } while (d_nnzu[rowidx] != -1);
            }
            __syncthreads();

            pangulu_int64_t rowa = d_csccolptra[rowidx];
            pangulu_int64_t rowa1 = d_csccolptra[rowidx + 1];
            pangulu_int64_t rowu = d_csccolptru[rowidx];
            pangulu_int64_t rowu1 = d_csccolptru[rowidx + 1];

            for (pangulu_int64_t j = rowa + rowu1 - rowu + threadIdx.x; j < rowa1; j += blockDim.x)
            {
                const pangulu_int64_t lrowindex = d_cscrowidxa[j];
                const pangulu_int64_t thecolidx = rowidx;

                pangulu_int64_t flag1 = binarysearch_idx(d_cscrowidxa, cola, cola1 - 1, lrowindex);
                pangulu_int64_t flag2 = binarysearch_idx(d_cscrowidxa, d_csccolptra[thecolidx] + d_csccolptru[thecolidx + 1] - d_csccolptru[thecolidx], d_csccolptra[thecolidx + 1] - 1, lrowindex);

                d_cscvaluea[flag1] -= d_cscvaluea[flag2] * value3;
            }

            if (!threadIdx.x)
            {
                atomicSub(&s_nnzu[0], 1);
            }
            __syncthreads();
        }

        for (pangulu_int64_t i = threadIdx.x; i < colu1 - colu; i += blockDim.x)
        {
            d_cscvalueu[colu + i] = d_cscvaluea[cola + i];
        }

        // step two
        pangulu_int64_t flag5 = cola + colu1 - colu - 1;
        if (d_cscvaluea[flag5] > ERROR || d_cscvaluea[flag5] < -ERROR)
        {
        }
        else
        {
            d_cscvaluea[flag5] = ERROR;
        }
        calculate_type value5 = d_cscvaluea[flag5];

        d_cscvaluel[coll] = 1;
        for (pangulu_int64_t i = cola + colu1 - colu + threadIdx.x; i < cola1; i += blockDim.x)
        {
            const pangulu_int64_t lrowindex = d_cscrowidxa[i];
            pangulu_int64_t flagl = coll + i - (cola + colu1 - colu) + 1;
            d_cscvaluea[i] = d_cscvaluea[i] / value5;
            d_cscvaluel[flagl] = d_cscvaluea[i];
        }

        if (!threadIdx.x)
        {
            atomicSub(&s_nnzu[0], 1);
            d_nnzu[colidx] = s_nnzu[0];
        }
    }
}

__global__ void blocklevel_sflu_l3(pangulu_int64_t n,
                                   pangulu_int32_t *d_nnzu,
                                   pangulu_inblock_ptr *d_csccolptra,
                                   pangulu_inblock_idx *d_cscrowidxa,
                                   calculate_type *d_cscvaluea,
                                   pangulu_inblock_ptr *d_csccolptrl,
                                   pangulu_inblock_idx *d_cscrowidxl,
                                   calculate_type *d_cscvaluel,
                                   pangulu_inblock_ptr *d_csccolptru,
                                   pangulu_inblock_idx *d_cscrowidxu,
                                   calculate_type *d_cscvalueu)
{
    const pangulu_int64_t colidx = blockIdx.x;

    __shared__ pangulu_int32_t s_nnzu[1];

    pangulu_int64_t cola = d_csccolptra[colidx];
    pangulu_int64_t cola1 = d_csccolptra[colidx + 1];
    pangulu_int64_t smemlen = 1024;
    pangulu_int64_t len_a = cola1 - cola;

    // use shared memory
    if (len_a <= smemlen)
    {
        __shared__ pangulu_inblock_idx s_rowidxa[1024];
        __shared__ calculate_type s_valuea[1024];

        for (pangulu_int64_t i = threadIdx.x; i < len_a; i += blockDim.x)
        {
            s_rowidxa[i] = d_cscrowidxa[i + cola];
            s_valuea[i] = d_cscvaluea[i + cola];
        }

        if (!threadIdx.x)
        {
            s_nnzu[0] = d_nnzu[colidx];
        }
        __syncthreads();

        pangulu_int64_t colu = d_csccolptru[colidx];
        pangulu_int64_t colu1 = d_csccolptru[colidx + 1];
        pangulu_int64_t coll = d_csccolptrl[colidx];
        pangulu_int64_t coll1 = d_csccolptrl[colidx + 1];

        // step one
        for (pangulu_int64_t i = cola; i < cola + colu1 - colu - 1; i++)
        {
            const pangulu_int64_t rowidx = d_cscrowidxa[i];
            calculate_type value3 = s_valuea[i - cola];

            // busy-wait until nnzu[rowidx]==0
            if (!threadIdx.x)
            {
                do
                {
                    __threadfence_block();
                } while (d_nnzu[rowidx] != -1);
            }
            __syncthreads();

            pangulu_int64_t rowa = d_csccolptra[rowidx];
            pangulu_int64_t rowa1 = d_csccolptra[rowidx + 1];
            pangulu_int64_t rowu = d_csccolptru[rowidx];
            pangulu_int64_t rowu1 = d_csccolptru[rowidx + 1];

            for (pangulu_int64_t j = rowa + rowu1 - rowu + threadIdx.x; j < rowa1; j += blockDim.x)
            {
                const pangulu_int64_t lrowindex = d_cscrowidxa[j];
                const pangulu_int64_t thecolidx = rowidx;

                pangulu_int64_t flag1 = binarysearch_idx(s_rowidxa, 0, len_a - 1, lrowindex);
                pangulu_int64_t flag2 = binarysearch_idx(d_cscrowidxa, d_csccolptra[thecolidx] + d_csccolptru[thecolidx + 1] - d_csccolptru[thecolidx], d_csccolptra[thecolidx + 1] - 1, lrowindex);

                s_valuea[flag1] -= d_cscvaluea[flag2] * value3;
            }

            if (!threadIdx.x)
            {
                atomicSub(&s_nnzu[0], 1);
            }
            __syncthreads();
        }

        // step two
        pangulu_int64_t flag5 = colu1 - colu - 1;
        if (s_valuea[flag5] > ERROR || s_valuea[flag5] < -ERROR)
        {
        }
        else
        {
            s_valuea[flag5] = ERROR;
        }
        calculate_type value5 = s_valuea[flag5];

        d_cscvaluel[coll] = 1;
        for (pangulu_int64_t i = cola + colu1 - colu + threadIdx.x; i < cola1; i += blockDim.x)
        {
            const pangulu_int64_t lrowindex = s_rowidxa[i - cola];
            pangulu_int64_t flagl = coll + i - (cola + colu1 - colu) + 1;
            s_valuea[i - cola] = s_valuea[i - cola] / value5;
            d_cscvaluel[flagl] = s_valuea[i - cola];
        }
        __syncthreads();

        for (pangulu_int64_t i = threadIdx.x; i < colu1 - colu; i += blockDim.x)
        {
            d_cscvalueu[colu + i] = s_valuea[i];
        }

        for (pangulu_int64_t i = threadIdx.x; i < len_a; i += blockDim.x)
        {
            d_cscvaluea[i + cola] = s_valuea[i];
        }

        if (!threadIdx.x)
        {
            atomicSub(&s_nnzu[0], 1);
            d_nnzu[colidx] = s_nnzu[0];
        }
    }
    // do not use shared memory
    else
    {
        if (!threadIdx.x)
        {
            s_nnzu[0] = d_nnzu[colidx];
        }
        __syncthreads();

        pangulu_int64_t colu = d_csccolptru[colidx];
        pangulu_int64_t colu1 = d_csccolptru[colidx + 1];
        pangulu_int64_t coll = d_csccolptrl[colidx];
        pangulu_int64_t coll1 = d_csccolptrl[colidx + 1];

        // step one
        for (pangulu_int64_t i = cola; i < cola + colu1 - colu - 1; i++)
        {
            const pangulu_int64_t rowidx = d_cscrowidxa[i];

            calculate_type value3 = d_cscvaluea[i];

            // busy-wait until nnzu[rowidx]==0
            if (!threadIdx.x)
            {
                do
                {
                    __threadfence_block();
                } while (d_nnzu[rowidx] != -1);
            }
            __syncthreads();

            pangulu_int64_t rowa = d_csccolptra[rowidx];
            pangulu_int64_t rowa1 = d_csccolptra[rowidx + 1];
            pangulu_int64_t rowu = d_csccolptru[rowidx];
            pangulu_int64_t rowu1 = d_csccolptru[rowidx + 1];

            for (pangulu_int64_t j = rowa + rowu1 - rowu + threadIdx.x; j < rowa1; j += blockDim.x)
            {
                const pangulu_int64_t lrowindex = d_cscrowidxa[j];
                const pangulu_int64_t thecolidx = rowidx;

                pangulu_int64_t flag1 = binarysearch_idx(d_cscrowidxa, cola, cola1 - 1, lrowindex);
                pangulu_int64_t flag2 = binarysearch_idx(d_cscrowidxa, d_csccolptra[thecolidx] + d_csccolptru[thecolidx + 1] - d_csccolptru[thecolidx], d_csccolptra[thecolidx + 1] - 1, lrowindex);

                d_cscvaluea[flag1] -= d_cscvaluea[flag2] * value3;
            }

            if (!threadIdx.x)
            {
                atomicSub(&s_nnzu[0], 1);
            }
            __syncthreads();
        }

        for (pangulu_int64_t i = threadIdx.x; i < colu1 - colu; i += blockDim.x)
        {
            d_cscvalueu[colu + i] = d_cscvaluea[cola + i];
        }

        // step two
        pangulu_int64_t flag5 = cola + colu1 - colu - 1;
        if (d_cscvaluea[flag5] > ERROR || d_cscvaluea[flag5] < -ERROR)
        {
        }
        else
        {
            d_cscvaluea[flag5] = ERROR;
        }
        calculate_type value5 = d_cscvaluea[flag5];

        d_cscvaluel[coll] = 1;
        for (pangulu_int64_t i = cola + colu1 - colu + threadIdx.x; i < cola1; i += blockDim.x)
        {
            const pangulu_int64_t lrowindex = d_cscrowidxa[i];
            pangulu_int64_t flagl = coll + i - (cola + colu1 - colu) + 1;
            d_cscvaluea[i] = d_cscvaluea[i] / value5;
            d_cscvaluel[flagl] = d_cscvaluea[i];
        }

        if (!threadIdx.x)
        {
            atomicSub(&s_nnzu[0], 1);
            d_nnzu[colidx] = s_nnzu[0];
        }
    }
}

__global__ void blocklevel_sflu_l4(pangulu_int64_t n,
                                   pangulu_int32_t *d_nnzu,
                                   pangulu_inblock_ptr *d_csccolptra,
                                   pangulu_inblock_idx *d_cscrowidxa,
                                   calculate_type *d_cscvaluea,
                                   pangulu_inblock_ptr *d_csccolptrl,
                                   pangulu_inblock_idx *d_cscrowidxl,
                                   calculate_type *d_cscvaluel,
                                   pangulu_inblock_ptr *d_csccolptru,
                                   pangulu_inblock_idx *d_cscrowidxu,
                                   calculate_type *d_cscvalueu)
{
    const pangulu_int64_t colidx = blockIdx.x;

    __shared__ pangulu_int32_t s_nnzu[1];

    pangulu_int64_t cola = d_csccolptra[colidx];
    pangulu_int64_t cola1 = d_csccolptra[colidx + 1];
    pangulu_int64_t smemlen = 2048;
    pangulu_int64_t len_a = cola1 - cola;

    // use shared memory
    if (len_a <= smemlen)
    {
        __shared__ pangulu_inblock_idx s_rowidxa[2048];
        __shared__ calculate_type s_valuea[2048];

        for (pangulu_int64_t i = threadIdx.x; i < len_a; i += blockDim.x)
        {
            s_rowidxa[i] = d_cscrowidxa[i + cola];
            s_valuea[i] = d_cscvaluea[i + cola];
        }

        if (!threadIdx.x)
        {
            s_nnzu[0] = d_nnzu[colidx];
        }
        __syncthreads();

        pangulu_int64_t colu = d_csccolptru[colidx];
        pangulu_int64_t colu1 = d_csccolptru[colidx + 1];
        pangulu_int64_t coll = d_csccolptrl[colidx];
        pangulu_int64_t coll1 = d_csccolptrl[colidx + 1];

        // step one
        for (pangulu_int64_t i = cola; i < cola + colu1 - colu - 1; i++)
        {
            const pangulu_int64_t rowidx = d_cscrowidxa[i];
            calculate_type value3 = s_valuea[i - cola];

            // busy-wait until nnzu[rowidx]==0
            if (!threadIdx.x)
            {
                do
                {
                    __threadfence_block();
                } while (d_nnzu[rowidx] != -1);
            }
            __syncthreads();

            pangulu_int64_t rowa = d_csccolptra[rowidx];
            pangulu_int64_t rowa1 = d_csccolptra[rowidx + 1];
            pangulu_int64_t rowu = d_csccolptru[rowidx];
            pangulu_int64_t rowu1 = d_csccolptru[rowidx + 1];

            for (pangulu_int64_t j = rowa + rowu1 - rowu + threadIdx.x; j < rowa1; j += blockDim.x)
            {
                const pangulu_int64_t lrowindex = d_cscrowidxa[j];
                const pangulu_int64_t thecolidx = rowidx;

                pangulu_int64_t flag1 = binarysearch_idx(s_rowidxa, 0, len_a - 1, lrowindex);
                pangulu_int64_t flag2 = binarysearch_idx(d_cscrowidxa, d_csccolptra[thecolidx] + d_csccolptru[thecolidx + 1] - d_csccolptru[thecolidx], d_csccolptra[thecolidx + 1] - 1, lrowindex);

                s_valuea[flag1] -= d_cscvaluea[flag2] * value3;
            }

            if (!threadIdx.x)
            {
                atomicSub(&s_nnzu[0], 1);
            }
            __syncthreads();
        }

        // step two
        pangulu_int64_t flag5 = colu1 - colu - 1;
        if (s_valuea[flag5] > ERROR || s_valuea[flag5] < -ERROR)
        {
        }
        else
        {
            s_valuea[flag5] = ERROR;
        }
        calculate_type value5 = s_valuea[flag5];

        d_cscvaluel[coll] = 1;
        for (pangulu_int64_t i = cola + colu1 - colu + threadIdx.x; i < cola1; i += blockDim.x)
        {
            const pangulu_int64_t lrowindex = s_rowidxa[i - cola];
            pangulu_int64_t flagl = coll + i - (cola + colu1 - colu) + 1;
            s_valuea[i - cola] = s_valuea[i - cola] / value5;
            d_cscvaluel[flagl] = s_valuea[i - cola];
        }
        __syncthreads();

        for (pangulu_int64_t i = threadIdx.x; i < colu1 - colu; i += blockDim.x)
        {
            d_cscvalueu[colu + i] = s_valuea[i];
        }

        for (pangulu_int64_t i = threadIdx.x; i < len_a; i += blockDim.x)
        {
            d_cscvaluea[i + cola] = s_valuea[i];
        }

        if (!threadIdx.x)
        {
            atomicSub(&s_nnzu[0], 1);
            d_nnzu[colidx] = s_nnzu[0];
        }
    }
    // do not use shared memory
    else
    {
        if (!threadIdx.x)
        {
            s_nnzu[0] = d_nnzu[colidx];
        }
        __syncthreads();

        pangulu_int64_t colu = d_csccolptru[colidx];
        pangulu_int64_t colu1 = d_csccolptru[colidx + 1];
        pangulu_int64_t coll = d_csccolptrl[colidx];
        pangulu_int64_t coll1 = d_csccolptrl[colidx + 1];

        // step one
        for (pangulu_int64_t i = cola; i < cola + colu1 - colu - 1; i++)
        {
            const pangulu_int64_t rowidx = d_cscrowidxa[i];

            calculate_type value3 = d_cscvaluea[i];

            // busy-wait until nnzu[rowidx]==0
            if (!threadIdx.x)
            {
                do
                {
                    __threadfence_block();
                } while (d_nnzu[rowidx] != -1);
            }
            __syncthreads();

            pangulu_int64_t rowa = d_csccolptra[rowidx];
            pangulu_int64_t rowa1 = d_csccolptra[rowidx + 1];
            pangulu_int64_t rowu = d_csccolptru[rowidx];
            pangulu_int64_t rowu1 = d_csccolptru[rowidx + 1];

            for (pangulu_int64_t j = rowa + rowu1 - rowu + threadIdx.x; j < rowa1; j += blockDim.x)
            {
                const pangulu_int64_t lrowindex = d_cscrowidxa[j];
                const pangulu_int64_t thecolidx = rowidx;

                pangulu_int64_t flag1 = binarysearch_idx(d_cscrowidxa, cola, cola1 - 1, lrowindex);
                pangulu_int64_t flag2 = binarysearch_idx(d_cscrowidxa, d_csccolptra[thecolidx] + d_csccolptru[thecolidx + 1] - d_csccolptru[thecolidx], d_csccolptra[thecolidx + 1] - 1, lrowindex);

                d_cscvaluea[flag1] -= d_cscvaluea[flag2] * value3;
            }

            if (!threadIdx.x)
            {
                atomicSub(&s_nnzu[0], 1);
            }
            __syncthreads();
        }

        for (pangulu_int64_t i = threadIdx.x; i < colu1 - colu; i += blockDim.x)
        {
            d_cscvalueu[colu + i] = d_cscvaluea[cola + i];
        }

        // step two
        pangulu_int64_t flag5 = cola + colu1 - colu - 1;
        if (d_cscvaluea[flag5] > ERROR || d_cscvaluea[flag5] < -ERROR)
        {
        }
        else
        {
            d_cscvaluea[flag5] = ERROR;
        }
        calculate_type value5 = d_cscvaluea[flag5];

        d_cscvaluel[coll] = 1;
        for (pangulu_int64_t i = cola + colu1 - colu + threadIdx.x; i < cola1; i += blockDim.x)
        {
            const pangulu_int64_t lrowindex = d_cscrowidxa[i];
            pangulu_int64_t flagl = coll + i - (cola + colu1 - colu) + 1;
            d_cscvaluea[i] = d_cscvaluea[i] / value5;
            d_cscvaluel[flagl] = d_cscvaluea[i];
        }

        if (!threadIdx.x)
        {
            atomicSub(&s_nnzu[0], 1);
            d_nnzu[colidx] = s_nnzu[0];
        }
    }
}

__device__ pangulu_int64_t binarysearch_device(pangulu_int64_t *ridx, pangulu_int64_t left, pangulu_int64_t right, pangulu_int64_t target)
{
    pangulu_int64_t mid;
    while (left <= right)
    {
        mid = left + (right - left) / 2;
        if (ridx[mid] == target)
        {
            return mid;
        }
        else if (ridx[mid] > target)
        {
            right = mid - 1;
        }
        else
        {
            left = mid + 1;
        }
    }
    return -1;
}

__device__ pangulu_inblock_idx binarysearch_idx(pangulu_inblock_idx *ridx, pangulu_int64_t left, pangulu_int64_t right, pangulu_inblock_idx target)
{
    pangulu_int64_t mid;
    while (left <= right)
    {
        mid = left + (right - left) / 2;
        if (ridx[mid] == target)
        {
            return mid;
        }
        else if (ridx[mid] > target)
        {
            right = mid - 1;
        }
        else
        {
            left = mid + 1;
        }
    }
    return 0xffff;
}
__global__ void cuda_transform_s_to_d_col(pangulu_int64_t n,
                                          int stride,
                                          pangulu_inblock_ptr *d_rowptra,
                                          pangulu_inblock_idx *d_colidxa,
                                          calculate_type *d_valuea,
                                          calculate_type *temp_value_a)
{
    pangulu_int64_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    pangulu_int64_t lane_id = threadIdx.x % 32;
    pangulu_int64_t warp_local_id = threadIdx.x / 32;

    if (warp_id >= n)
    {
        return;
    }

    calculate_type *value = temp_value_a + warp_id * n;

    int cola1 = d_rowptra[warp_id];
    int cola2 = d_rowptra[warp_id + 1];

    for (int i = cola1 + lane_id; i < cola2; i += warp_size)
    {
        value[d_colidxa[i]] = d_valuea[i];
    }
}

__global__ void cuda_transform_d_to_s_lu_col(pangulu_int64_t n,
                                             pangulu_inblock_ptr *d_rowptra,
                                             pangulu_inblock_idx *d_colidxa,
                                             pangulu_inblock_ptr *d_rowptrl,
                                             pangulu_inblock_idx *d_colidxl,
                                             calculate_type *d_valuel,
                                             pangulu_inblock_ptr *d_rowptru,
                                             pangulu_inblock_idx *d_colidxu,
                                             calculate_type *d_valueu,
                                             calculate_type *temp_value_a)
{
    pangulu_int64_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    pangulu_int64_t lane_id = threadIdx.x % 32;
    pangulu_int64_t warp_local_id = threadIdx.x / 32;

    if (warp_id >= n)
    {
        return;
    }
    calculate_type *value = temp_value_a + warp_id * n;

    pangulu_int64_t coll1 = d_rowptrl[warp_id];
    pangulu_int64_t coll2 = d_rowptrl[warp_id + 1];

    pangulu_int64_t colu1 = d_rowptru[warp_id];
    pangulu_int64_t colu2 = d_rowptru[warp_id + 1];

    for (int i = coll1 + 1 + lane_id; i < coll2; i += warp_size)
    {
        d_valuel[i] = value[d_colidxl[i]];
    }
    d_valuel[coll1] = 1;

    for (int i = colu1 + lane_id; i < colu2; i += warp_size)
    {
        d_valueu[i] = value[d_colidxu[i]];
    }
}
__global__ void lunumeric_cuda_kernel_v2(pangulu_int64_t n,
                                         pangulu_int32_t *d_nnzu,
                                         calculate_type *d_dense_tag_double,
                                         pangulu_inblock_ptr *d_csccolptrl_upperbound,
                                         pangulu_inblock_idx *d_cscrowidxl_upperbound,
                                         pangulu_inblock_ptr *d_csccolptru_upperbound,
                                         pangulu_inblock_idx *d_cscrowidxu_upperbound)
{

    const pangulu_int64_t colidx = blockIdx.x;
    __shared__ pangulu_int32_t s_nnzu[1];

    if (!threadIdx.x)
    {
        s_nnzu[0] = d_nnzu[colidx];
    }
    __syncthreads();

    const pangulu_int64_t baseu_colidx = d_csccolptru_upperbound[colidx];
    const pangulu_int64_t baseu_colidx1 = d_csccolptru_upperbound[colidx + 1];
    const pangulu_int64_t basel_colidx = d_csccolptrl_upperbound[colidx];
    const pangulu_int64_t basel_colidx1 = d_csccolptrl_upperbound[colidx + 1];

    // step one
    for (pangulu_int64_t j = baseu_colidx; j < baseu_colidx1 - 1; j++)
    {
        const pangulu_inblock_idx rowidx = d_cscrowidxu_upperbound[j];
        // busy-wait until nnzu[rowidx] == 0
        if (!threadIdx.x)
        {
            do
            {
                __threadfence_block();
            } while (d_nnzu[rowidx] != -1);
        }
        __syncthreads();

        for (int i = d_csccolptrl_upperbound[rowidx] + 1 + threadIdx.x; i < d_csccolptrl_upperbound[rowidx + 1]; i += blockDim.x)
        {
            const int lrowindex = d_cscrowidxl_upperbound[i];
            const int thecolidx = rowidx;
            d_dense_tag_double[colidx * n + lrowindex] -= d_dense_tag_double[thecolidx * n + lrowindex] * d_dense_tag_double[colidx * n + rowidx];
        }
        if (!threadIdx.x)
        {
            atomicSub(&s_nnzu[0], 1);
        }
        __syncthreads();
    }

    //  step two
    for (int i = basel_colidx + threadIdx.x + 1; i < d_csccolptrl_upperbound[colidx + 1]; i += blockDim.x)
    {
        const int lrowindex = d_cscrowidxl_upperbound[i];
        d_dense_tag_double[colidx * n + lrowindex] = d_dense_tag_double[colidx * n + lrowindex] / d_dense_tag_double[colidx * n + colidx];
    }

    if (!threadIdx.x)
    {
        atomicSub(&s_nnzu[0], 1);
        d_nnzu[colidx] = s_nnzu[0];
    }
}

// __global__ void trans_cuda_csc_to_csr(pangulu_int64_t nnz, calculate_type *d_val_csr, pangulu_inblock_idx *d_idx, calculate_type *d_val_csc)
// {

//     if (blockDim.x * blockIdx.x + threadIdx.x >= nnz)
//         return;

//     pangulu_int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
//     d_val_csr[d_idx[i]] = d_val_csc[i];
// }

// __global__ void trans_cuda_csr_to_csc(pangulu_int64_t nnz, calculate_type *d_val_csc, pangulu_inblock_idx *d_idx, calculate_type *d_val_csr)
// {

//     if (blockDim.x * blockIdx.x + threadIdx.x >= nnz)
//         return;

//     pangulu_int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
//     d_val_csc[i] = d_val_csr[d_idx[i]];
// }

__global__ void vector_add_cuda(pangulu_int64_t nnz, calculate_type *d_now_val_csc, calculate_type *d_old_val_csc)
{

    if (blockDim.x * blockIdx.x + threadIdx.x >= nnz)
        return;

    pangulu_int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    d_now_val_csc[i] += d_old_val_csc[i];
}

void pangulu_cuda_vector_add_kernel(pangulu_int64_t nnz, calculate_type *d_now_val_csc, calculate_type *d_old_val_csc)
{

    pangulu_int64_t num_threads = warp_size * 2;
    pangulu_int64_t num_blocks = ceil((double)nnz / num_threads);
    cudaDeviceSynchronize();
    vector_add_cuda<<<num_blocks, num_threads>>>(nnz, d_now_val_csc, d_old_val_csc);
    cudaDeviceSynchronize();
}

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
                                    calculate_type *d_valuec)
{
    pangulu_int64_t warp_global_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    pangulu_int64_t warp_local_id = threadIdx.x / 32;
    pangulu_int64_t lane_id = threadIdx.x % 32;

    if (warp_global_id >= d_bin_rowpointer[layer + 1] - d_bin_rowpointer[layer])
        return;

    const pangulu_inblock_idx rowidx = d_bin_rowindex[d_bin_rowpointer[layer] + warp_global_id];

    pangulu_int64_t therowc = d_rowptrc[rowidx];
    pangulu_int64_t nextrowc = d_rowptrc[rowidx + 1];

    pangulu_int64_t therowa = d_rowptra[rowidx];
    pangulu_int64_t nextrowa = d_rowptra[rowidx + 1];

    pangulu_int64_t sm_len = 32;

    __shared__ pangulu_inblock_idx s_idxc[32 * warp_per_block_gemm];
    __shared__ calculate_type s_valc[32 * warp_per_block_gemm];

    pangulu_inblock_idx *s_idxc_local = &s_idxc[warp_local_id * sm_len];
    calculate_type *s_valc_local = &s_valc[warp_local_id * sm_len];

    if (lane_id < nextrowc - therowc)
    {
        s_idxc_local[lane_id] = d_colidxc[therowc + lane_id];
        s_valc_local[lane_id] = 0;
    }

    for (pangulu_int64_t i = therowa + lane_id; i < nextrowa; i += warp_size)
    {
        pangulu_int64_t cola = d_colidxa[i];
        calculate_type vala = d_valuea[i];

        pangulu_int64_t therowb = d_rowptrb[cola];
        pangulu_int64_t nextrowb = d_rowptrb[cola + 1];

        for (pangulu_int64_t j = therowb; j < nextrowb; j++)
        {
            pangulu_int64_t colb = d_colidxb[j];
            pangulu_int64_t flag = binarysearch_idx(s_idxc_local, 0, nextrowc - therowc - 1, colb);
            atomicAdd(&s_valc_local[flag], -vala * d_valueb[j]);
        }
    }

    if (lane_id < nextrowc - therowc)
    {
        d_valuec[therowc + lane_id] += s_valc_local[lane_id];
    }
}

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
                                    calculate_type *d_valuec)
{
    pangulu_int64_t warp_global_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    pangulu_int64_t warp_local_id = threadIdx.x / 32;
    pangulu_int64_t lane_id = threadIdx.x % 32;

    if (warp_global_id >= d_bin_rowpointer[layer + 1] - d_bin_rowpointer[layer])
        return;

    const pangulu_inblock_idx rowidx = d_bin_rowindex[d_bin_rowpointer[layer] + warp_global_id];

    pangulu_int64_t therowc = d_rowptrc[rowidx];
    pangulu_int64_t nextrowc = d_rowptrc[rowidx + 1];

    pangulu_int64_t therowa = d_rowptra[rowidx];
    pangulu_int64_t nextrowa = d_rowptra[rowidx + 1];

    pangulu_int64_t sm_len = 64;

    __shared__ pangulu_inblock_idx s_idxc[64 * warp_per_block_gemm];
    __shared__ calculate_type s_valc[64 * warp_per_block_gemm];

    pangulu_inblock_idx *s_idxc_local = &s_idxc[warp_local_id * sm_len];
    calculate_type *s_valc_local = &s_valc[warp_local_id * sm_len];

    for (pangulu_int64_t i = lane_id; i < nextrowc - therowc; i += warp_size)
    {
        s_idxc_local[i] = d_colidxc[therowc + i];
        s_valc_local[i] = 0;
    }
    __syncthreads();

    for (pangulu_int64_t i = therowa + lane_id; i < nextrowa; i += warp_size)
    {
        pangulu_int64_t cola = d_colidxa[i];
        calculate_type vala = d_valuea[i];

        pangulu_int64_t therowb = d_rowptrb[cola];
        pangulu_int64_t nextrowb = d_rowptrb[cola + 1];

        for (pangulu_int64_t j = therowb; j < nextrowb; j++)
        {
            pangulu_int64_t colb = d_colidxb[j];
            pangulu_int64_t flag = binarysearch_idx(s_idxc_local, 0, nextrowc - therowc - 1, colb);
            atomicAdd(&s_valc_local[flag], -vala * d_valueb[j]);
        }
    }
    for (pangulu_int64_t i = lane_id; i < nextrowc - therowc; i += warp_size)
    {
        d_valuec[therowc + i] += s_valc_local[i];
    }
    __syncthreads();
}

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
                                     calculate_type *d_valuec)
{
    pangulu_int64_t warp_global_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    pangulu_int64_t warp_local_id = threadIdx.x / 32;
    pangulu_int64_t lane_id = threadIdx.x % 32;

    if (warp_global_id >= d_bin_rowpointer[layer + 1] - d_bin_rowpointer[layer])
        return;

    const pangulu_inblock_idx rowidx = d_bin_rowindex[d_bin_rowpointer[layer] + warp_global_id];

    pangulu_int64_t therowc = d_rowptrc[rowidx];
    pangulu_int64_t nextrowc = d_rowptrc[rowidx + 1];

    pangulu_int64_t therowa = d_rowptra[rowidx];
    pangulu_int64_t nextrowa = d_rowptra[rowidx + 1];

    pangulu_int64_t sm_len = 128;

    __shared__ pangulu_inblock_idx s_idxc[128 * warp_per_block_gemm];
    __shared__ calculate_type s_valc[128 * warp_per_block_gemm];

    pangulu_inblock_idx *s_idxc_local = &s_idxc[warp_local_id * sm_len];
    calculate_type *s_valc_local = &s_valc[warp_local_id * sm_len];

    for (pangulu_int64_t i = lane_id; i < nextrowc - therowc; i += warp_size)
    {
        s_idxc_local[i] = d_colidxc[therowc + i];
        s_valc_local[i] = 0;
    }
    __syncthreads();

    for (pangulu_int64_t i = therowa; i < nextrowa; i++)
    {
        pangulu_int64_t cola = d_colidxa[i];
        calculate_type vala = d_valuea[i];

        pangulu_int64_t therowb = d_rowptrb[cola];
        pangulu_int64_t nextrowb = d_rowptrb[cola + 1];

        for (pangulu_int64_t j = therowb + lane_id; j < nextrowb; j += warp_size)
        {
            pangulu_int64_t colb = d_colidxb[j];
            pangulu_int64_t flag = binarysearch_idx(s_idxc_local, 0, nextrowc - therowc - 1, colb);
            s_valc_local[flag] -= vala * d_valueb[j];
        }
    }

    for (pangulu_int64_t i = lane_id; i < nextrowc - therowc; i += warp_size)
    {
        d_valuec[therowc + i] += s_valc_local[i];
    }
    __syncthreads();
}

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
                                   calculate_type *d_valuec)
{
    const pangulu_inblock_idx rowidx = d_bin_rowindex[d_bin_rowpointer[layer] + blockDim.x * blockIdx.x + threadIdx.x];

    if (d_bin_rowpointer[layer] + blockDim.x * blockIdx.x + threadIdx.x >= d_bin_rowpointer[layer + 1])
        return;

    pangulu_int64_t therowa = d_rowptra[rowidx];
    pangulu_int64_t nextrowa = d_rowptra[rowidx + 1];

    pangulu_int64_t therowc = d_rowptrc[rowidx];

    for (pangulu_int64_t i = therowa; i < nextrowa; i++)
    {
        pangulu_int64_t cola = d_colidxa[i];
        calculate_type vala = d_valuea[i];

        pangulu_int64_t therowb = d_rowptrb[cola];
        pangulu_int64_t nextrowb = d_rowptrb[cola + 1];

        if (nextrowb - therowb != 0)
            d_valuec[therowc] -= vala * d_valueb[therowb];
    }
    __syncthreads();
}

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
                                      calculate_type *d_valuec)
{
    pangulu_int64_t warp_local_id = threadIdx.x / 32;
    pangulu_int64_t warp_num = blockDim.x / 32;
    pangulu_int64_t lane_id = threadIdx.x % 32;

    const pangulu_inblock_idx rowidx = d_bin_rowindex[d_bin_rowpointer[layer] + blockIdx.x];

    __shared__ pangulu_inblock_idx s_idxc[256];
    __shared__ calculate_type s_valc[256];

    pangulu_int64_t therowc = d_rowptrc[rowidx];
    pangulu_int64_t nextrowc = d_rowptrc[rowidx + 1];

    for (pangulu_int64_t i = threadIdx.x; i < nextrowc - therowc; i += blockDim.x)
    {
        s_idxc[i] = d_colidxc[therowc + i];
        s_valc[i] = 0;
    }
    __syncthreads();

    pangulu_int64_t therow = d_rowptra[rowidx];
    pangulu_int64_t nextrow = d_rowptra[rowidx + 1];

    for (pangulu_int64_t i = therow + warp_local_id; i < nextrow; i += warp_num)
    {
        pangulu_int64_t cola = d_colidxa[i];
        calculate_type vala = d_valuea[i];

        pangulu_int64_t therowb = d_rowptrb[cola];
        pangulu_int64_t nextrowb = d_rowptrb[cola + 1];

        for (pangulu_int64_t j = therowb + lane_id; j < nextrowb; j += warp_size)
        {
            pangulu_int64_t colb = d_colidxb[j];
            pangulu_int64_t flag = binarysearch_idx(s_idxc, 0, nextrowc - therowc - 1, colb);
            atomicAdd(&s_valc[flag], -vala * d_valueb[j]);
        }
    }
    __syncthreads();

    for (pangulu_int64_t i = threadIdx.x; i < nextrowc - therowc; i += blockDim.x)
    {
        d_valuec[therowc + i] += s_valc[i];
    }
}

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
                                      calculate_type *d_valuec)
{
    pangulu_int64_t warp_local_id = threadIdx.x / 32;
    pangulu_int64_t warp_num = blockDim.x / 32;
    pangulu_int64_t lane_id = threadIdx.x % 32;

    const pangulu_inblock_idx rowidx = d_bin_rowindex[d_bin_rowpointer[layer] + blockIdx.x];

    __shared__ pangulu_inblock_idx s_idxc[512];
    __shared__ calculate_type s_valc[512];

    pangulu_int64_t therowc = d_rowptrc[rowidx];
    pangulu_int64_t nextrowc = d_rowptrc[rowidx + 1];

    for (pangulu_int64_t i = threadIdx.x; i < nextrowc - therowc; i += blockDim.x)
    {
        s_idxc[i] = d_colidxc[therowc + i];
        s_valc[i] = 0;
    }
    __syncthreads();

    pangulu_int64_t therow = d_rowptra[rowidx];
    pangulu_int64_t nextrow = d_rowptra[rowidx + 1];

    for (pangulu_int64_t i = therow + warp_local_id; i < nextrow; i += warp_num)
    {
        pangulu_int64_t cola = d_colidxa[i];
        calculate_type vala = d_valuea[i];

        pangulu_int64_t therowb = d_rowptrb[cola];
        pangulu_int64_t nextrowb = d_rowptrb[cola + 1];

        for (pangulu_int64_t j = therowb + lane_id; j < nextrowb; j += warp_size)
        {
            pangulu_int64_t colb = d_colidxb[j];
            pangulu_int64_t flag = binarysearch_idx(s_idxc, 0, nextrowc - therowc - 1, colb);
            atomicAdd(&s_valc[flag], -vala * d_valueb[j]);
        }
    }
    __syncthreads();

    for (pangulu_int64_t i = threadIdx.x; i < nextrowc - therowc; i += blockDim.x)
    {
        d_valuec[therowc + i] += s_valc[i];
    }
}

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
                                       calculate_type *d_valuec)
{
    pangulu_int64_t warp_local_id = threadIdx.x / 32;
    pangulu_int64_t warp_num = blockDim.x / 32;
    pangulu_int64_t lane_id = threadIdx.x % 32;

    const pangulu_inblock_idx rowidx = d_bin_rowindex[d_bin_rowpointer[layer] + blockIdx.x];

    __shared__ pangulu_inblock_idx s_idxc[1024];
    __shared__ calculate_type s_valc[1024];

    pangulu_int64_t therowc = d_rowptrc[rowidx];
    pangulu_int64_t nextrowc = d_rowptrc[rowidx + 1];

    for (pangulu_int64_t i = threadIdx.x; i < nextrowc - therowc; i += blockDim.x)
    {
        s_idxc[i] = d_colidxc[therowc + i];
        s_valc[i] = 0;
    }
    __syncthreads();

    pangulu_int64_t therow = d_rowptra[rowidx];
    pangulu_int64_t nextrow = d_rowptra[rowidx + 1];

    for (pangulu_int64_t i = therow + warp_local_id; i < nextrow; i += warp_num)
    {
        pangulu_int64_t cola = d_colidxa[i];
        calculate_type vala = d_valuea[i];

        pangulu_int64_t therowb = d_rowptrb[cola];
        pangulu_int64_t nextrowb = d_rowptrb[cola + 1];

        for (pangulu_int64_t j = therowb + lane_id; j < nextrowb; j += warp_size)
        {
            pangulu_int64_t colb = d_colidxb[j];
            pangulu_int64_t flag = binarysearch_idx(s_idxc, 0, nextrowc - therowc - 1, colb);
            atomicAdd(&s_valc[flag], -vala * d_valueb[j]);
        }
    }
    __syncthreads();

    for (pangulu_int64_t i = threadIdx.x; i < nextrowc - therowc; i += blockDim.x)
    {
        d_valuec[therowc + i] += s_valc[i];
    }
}

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
                                       calculate_type *d_valuec)
{
    pangulu_int64_t warp_local_id = threadIdx.x / 32;
    pangulu_int64_t warp_num = blockDim.x / 32;
    pangulu_int64_t lane_id = threadIdx.x % 32;

    const pangulu_inblock_idx rowidx = d_bin_rowindex[d_bin_rowpointer[layer] + blockIdx.x];

    __shared__ pangulu_inblock_idx s_idxc[2048];
    __shared__ calculate_type s_valc[2048];

    pangulu_int64_t therowc = d_rowptrc[rowidx];
    pangulu_int64_t nextrowc = d_rowptrc[rowidx + 1];

    for (pangulu_int64_t i = threadIdx.x; i < nextrowc - therowc; i += blockDim.x)
    {
        s_idxc[i] = d_colidxc[therowc + i];
        s_valc[i] = 0;
    }
    __syncthreads();

    pangulu_int64_t therow = d_rowptra[rowidx];
    pangulu_int64_t nextrow = d_rowptra[rowidx + 1];

    for (pangulu_int64_t i = therow + warp_local_id; i < nextrow; i += warp_num)
    {
        pangulu_int64_t cola = d_colidxa[i];
        calculate_type vala = d_valuea[i];

        pangulu_int64_t therowb = d_rowptrb[cola];
        pangulu_int64_t nextrowb = d_rowptrb[cola + 1];

        for (pangulu_int64_t j = therowb + lane_id; j < nextrowb; j += warp_size)
        {
            pangulu_int64_t colb = d_colidxb[j];
            pangulu_int64_t flag = binarysearch_idx(s_idxc, 0, nextrowc - therowc - 1, colb);
            atomicAdd(&s_valc[flag], -vala * d_valueb[j]);
        }
    }
    __syncthreads();

    for (pangulu_int64_t i = threadIdx.x; i < nextrowc - therowc; i += blockDim.x)
    {
        d_valuec[therowc + i] += s_valc[i];
    }
}

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
                                       calculate_type *d_valuec)
{
    pangulu_int64_t warp_local_id = threadIdx.x / 32;
    pangulu_int64_t warp_num = blockDim.x / 32;
    pangulu_int64_t lane_id = threadIdx.x % 32;

    const pangulu_inblock_idx rowidx = d_bin_rowindex[d_bin_rowpointer[layer] + blockIdx.x];

    pangulu_int64_t therow = d_rowptra[rowidx];
    pangulu_int64_t nextrow = d_rowptra[rowidx + 1];

    pangulu_int64_t therowc = d_rowptrc[rowidx];
    pangulu_int64_t nextrowc = d_rowptrc[rowidx + 1];

    for (pangulu_int64_t i = therow + warp_local_id; i < nextrow; i += warp_num)
    {
        pangulu_int64_t cola = d_colidxa[i];
        calculate_type vala = d_valuea[i];

        pangulu_int64_t therowb = d_rowptrb[cola];
        pangulu_int64_t nextrowb = d_rowptrb[cola + 1];

        for (pangulu_int64_t j = therowb + lane_id; j < nextrowb; j += warp_size)
        {
            pangulu_int64_t colb = d_colidxb[j];

            pangulu_int64_t flag = binarysearch_idx(d_colidxc, therowc, nextrowc - 1, colb);

            atomicAdd(&d_valuec[flag], -vala * d_valueb[j]);
        }

        __syncthreads();
    }
}

__forceinline__ __device__ calculate_type sum_32_shfl(calculate_type sum)
{
#pragma unroll
    for (pangulu_int64_t mask = warp_size / 2; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    return sum;
}
__global__ void spmv_warpvec_csr_cuda_executor(const pangulu_inblock_ptr *d_csrrowptr,
                                               const pangulu_inblock_idx *d_csrcolidx,
                                               const calculate_type *d_csrval,
                                               const pangulu_inblock_ptr m,
                                               const calculate_type *d_x,
                                               calculate_type *d_y)
{
    const pangulu_int64_t global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // initialize
    const pangulu_int64_t rowid = global_id / warp_size;
    if (rowid >= m)
        return;

    const pangulu_int64_t lane_id = (warp_size - 1) & threadIdx.x;
    const pangulu_int64_t start = d_csrrowptr[rowid];
    const pangulu_int64_t stop = d_csrrowptr[rowid + 1];
    if (start == stop)
    {
        if (!lane_id)
            d_y[rowid] = (calculate_type)0;
        return;
    }

    calculate_type sum = (calculate_type)0;

    for (pangulu_int64_t j = start + lane_id; j < stop; j += warp_size)
    {
        sum += d_x[d_csrcolidx[j]] * d_csrval[j];
    }
    sum = sum_32_shfl(sum);

    // finish
    if (!lane_id)
        d_y[rowid] = sum;
}
__global__ void cuda_transform_csc_to_coo(pangulu_int64_t n, pangulu_inblock_ptr *d_colptr, pangulu_inblock_idx *d_rowidx, pangulu_int32_t *idx_col)
{
    pangulu_int64_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    pangulu_int64_t lane_id = threadIdx.x % 32;
    pangulu_int64_t warp_local_id = threadIdx.x / 32;

    if (warp_id >= n)
    {
        return;
    }

    int cola1 = d_colptr[warp_id];
    int cola2 = d_colptr[warp_id + 1];

    for (int i = cola1 + lane_id; i < cola2; i += warp_size)
    {
        idx_col[i] = warp_id;
    }
}
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
                                           calculate_type *temp_value_c)
{
    pangulu_int64_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    pangulu_int64_t lane_id = threadIdx.x % 32;
    pangulu_int64_t warp_local_id = threadIdx.x / 32;

    if (warp_id >= nnz)
    {
        return;
    }

    int colb = coo_col_b[warp_id];

    calculate_type *c_value = temp_value_c + n * colb;
    int i = d_colidxb[warp_id];
    int cola1 = d_rowptra[i];
    int cola2 = d_rowptra[i + 1];
    calculate_type value_b = d_valueb[warp_id];
    for (int p_a = cola1 + lane_id; p_a < cola2; p_a += warp_size)
    {
        atomicAdd(&c_value[d_colidxa[p_a]], -value_b * d_valuea[p_a]);
    }
}
__global__ void cuda_transform_d_to_s_col(pangulu_int64_t n,
                                          int stride,
                                          pangulu_inblock_ptr *d_rowptra,
                                          pangulu_inblock_idx *d_colidxa,
                                          calculate_type *d_valuea,
                                          calculate_type *temp_value_a)
{
    pangulu_int64_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    pangulu_int64_t lane_id = threadIdx.x % 32;
    pangulu_int64_t warp_local_id = threadIdx.x / 32;

    if (warp_id >= n)
    {
        return;
    }

    calculate_type *value = temp_value_a + warp_id * n;

    int cola1 = d_rowptra[warp_id];
    int cola2 = d_rowptra[warp_id + 1];

    for (int i = cola1 + lane_id; i < cola2; i += warp_size)
    {
        d_valuea[i] = value[d_colidxa[i]];
    }
}

// void pangulu_cuda_transpose_kernel_csc_to_csr(pangulu_int64_t nnz, calculate_type *d_val_csr, pangulu_inblock_idx *d_idx, calculate_type *d_val_csc)
// {

//     pangulu_int64_t num_threads = warp_size * 2;
//     pangulu_int64_t num_blocks = ceil((double)nnz / num_threads);

//     trans_cuda_csc_to_csr<<<num_blocks, num_threads>>>(nnz, d_val_csr, d_idx, d_val_csc);
//     cudaDeviceSynchronize();
// }

// void pangulu_cuda_transpose_kernel_csr_to_csc(pangulu_int64_t nnz, calculate_type *d_val_csc, pangulu_inblock_idx *d_idx, calculate_type *d_val_csr)
// {

//     pangulu_int64_t num_threads = warp_size * 2;
//     pangulu_int64_t num_blocks = ceil((double)nnz / num_threads);

//     trans_cuda_csr_to_csc<<<num_blocks, num_threads>>>(nnz, d_val_csc, d_idx, d_val_csr);
//     cudaDeviceSynchronize();
// }

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
                               calculate_type *u_cuda_value)
{
    pangulu_int64_t nnz_avrg = nnza / n;

    pangulu_int64_t num_blocks;
    pangulu_int64_t num_threads;

    if (nnz_avrg <= 32)
    {
        num_blocks = ceil((double)n / warp_num_warplu);
        num_threads = warp_num_warplu * warp_size;
        warplevel_sflu<<<num_blocks, num_threads>>>(n, d_nnzu, a_cuda_rowpointer, a_cuda_columnindex, a_cuda_value,
                                                    l_cuda_rowpointer, l_cuda_columnindex, l_cuda_value,
                                                    u_cuda_rowpointer, u_cuda_columnindex, u_cuda_value);
    }
    else if (nnz_avrg > 32 && nnz_avrg <= 96)
    {
        num_blocks = n;
        num_threads = 2 * warp_size;
        blocklevel_sflu_l1<<<num_blocks, num_threads>>>(n, d_nnzu, a_cuda_rowpointer, a_cuda_columnindex, a_cuda_value,
                                                        l_cuda_rowpointer, l_cuda_columnindex, l_cuda_value,
                                                        u_cuda_rowpointer, u_cuda_columnindex, u_cuda_value);
    }
    else if (nnz_avrg > 96 && nnz_avrg <= 192)
    {
        num_blocks = n;
        num_threads = 4 * warp_size;
        blocklevel_sflu_l2<<<num_blocks, num_threads>>>(n, d_nnzu, a_cuda_rowpointer, a_cuda_columnindex, a_cuda_value,
                                                        l_cuda_rowpointer, l_cuda_columnindex, l_cuda_value,
                                                        u_cuda_rowpointer, u_cuda_columnindex, u_cuda_value);
    }
    else if (nnz_avrg > 192 && nnz_avrg <= 384)
    {
        num_blocks = n;
        num_threads = 8 * warp_size;
        blocklevel_sflu_l3<<<num_blocks, num_threads>>>(n, d_nnzu, a_cuda_rowpointer, a_cuda_columnindex, a_cuda_value,
                                                        l_cuda_rowpointer, l_cuda_columnindex, l_cuda_value,
                                                        u_cuda_rowpointer, u_cuda_columnindex, u_cuda_value);
    }
    else
    {
        num_blocks = n;
        num_threads = 16 * warp_size;
        blocklevel_sflu_l4<<<num_blocks, num_threads>>>(n, d_nnzu, a_cuda_rowpointer, a_cuda_columnindex, a_cuda_value,
                                                        l_cuda_rowpointer, l_cuda_columnindex, l_cuda_value,
                                                        u_cuda_rowpointer, u_cuda_columnindex, u_cuda_value);
    }
}

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
                                     calculate_type *u_cuda_value)
{

    int num_threads = warp_size * warp_per_block;
    int num_blocks;

    num_blocks = ceil((double)n / warp_per_block);
    cuda_transform_s_to_d_col<<<num_blocks, num_threads>>>(n, num_blocks, a_cuda_rowpointer, a_cuda_columnindex, a_cuda_value, cuda_temp_value);
    cudaDeviceSynchronize();

    num_blocks = n;

    lunumeric_cuda_kernel_v2<<<num_blocks, num_threads>>>(n, d_nnzu, cuda_temp_value,
                                                          l_cuda_rowpointer, l_cuda_columnindex,
                                                          u_cuda_rowpointer, u_cuda_columnindex);

    cudaDeviceSynchronize();

    num_blocks = ceil((double)n / warp_per_block);

    cuda_transform_d_to_s_lu_col<<<num_blocks, num_threads>>>(n, a_cuda_rowpointer, a_cuda_columnindex, l_cuda_rowpointer, l_cuda_columnindex, l_cuda_value, u_cuda_rowpointer, u_cuda_columnindex, u_cuda_value, cuda_temp_value);
    cudaDeviceSynchronize();
}

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
                                  calculate_type *a_value)
{
    pangulu_int64_t nnzu_avrg = nnzu / n;
    if (nnzu_avrg < 32)
    {
        pangulu_int64_t num_threads = warp_per_block_dtstrf * warp_size;
        pangulu_int64_t num_blocks = ceil((calculate_type)n / warp_per_block_dtstrf);
        tstrf_kernel_v2<<<num_blocks, num_threads>>>(n, u_rowpointer, u_columnindex, u_value,
                                                     x_rowpointer, x_columnindex, x_value,
                                                     a_rowpointer, a_columnindex, a_value);
    }
    else
    {
        pangulu_int64_t num_threads = 8 * warp_size;
        pangulu_int64_t num_blocks = n;
        tstrf_kernel_v3<<<num_blocks, num_threads>>>(n, u_rowpointer, u_columnindex, u_value,
                                                     x_rowpointer, x_columnindex, x_value,
                                                     a_rowpointer, a_columnindex, a_value);
    }
}

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
                                   calculate_type *a_value)
{

    pangulu_int64_t num_threads = warp_size * warp_per_block;
    pangulu_int64_t num_blocks;

    num_blocks = ceil((calculate_type)n / warp_per_block);
    cuda_transform_s_to_d_col<<<num_blocks, num_threads>>>(n, num_blocks, a_rowpointer, a_columnindex, a_value, cuda_temp_value);
    cudaDeviceSynchronize();

    num_threads = 8 * warp_size;
    num_blocks = n;
    tstrf_kernel_dense<<<num_blocks, num_threads>>>(n, u_rowpointer, u_columnindex, u_value,
                                                    x_rowpointer, x_columnindex, x_value,
                                                    a_rowpointer, a_columnindex, a_value, cuda_temp_value);
}

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
                                   calculate_type *a_value)
{
    pangulu_int64_t num_threads = warp_size * warp_per_block;
    pangulu_int64_t num_blocks;

    num_blocks = ceil((calculate_type)n / warp_per_block);
    cuda_transform_s_to_d_col<<<num_blocks, num_threads>>>(n, num_blocks, a_rowpointer, a_columnindex, a_value, cuda_temp_value);
    cudaDeviceSynchronize();

    num_threads = warp_per_block_dgessm * warp_size;
    num_blocks = ceil((calculate_type)n / (calculate_type)warp_per_block_dgessm);
    syncfree_cuda_csr_dense_v11_u<<<num_blocks, num_threads>>>(n, n, degree, u_rowpointer, u_columnindex, u_value,
                                                               a_rowpointer, a_columnindex, x_value, a_value, d_id_extractor, d_left_sum, cuda_temp_value);
    cudaDeviceSynchronize();
}

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
                                  calculate_type *a_value)
{
    pangulu_int64_t nnzl_avrg = nnzl / n;

    if (nnzl_avrg < 32)
    {
        pangulu_int64_t num_threads = warp_per_block_dgessm * warp_size;
        pangulu_int64_t num_blocks = ceil((calculate_type)n / warp_per_block_dgessm);
        gessm_kernel_v2<<<num_blocks, num_threads>>>(n, l_columnpointer, l_rowindex, l_value,
                                                     x_columnpointer, x_rowindex, x_value,
                                                     a_columnpointer, a_rowindex, a_value);
    }
    else
    {
        pangulu_int64_t num_threads = 8 * warp_size;
        pangulu_int64_t num_blocks = n;
        gessm_kernel_v3<<<num_blocks, num_threads>>>(n, l_columnpointer, l_rowindex, l_value,
                                                     x_columnpointer, x_rowindex, x_value,
                                                     a_columnpointer, a_rowindex, a_value);
    }
}

void pangulu_gessm_cuda_kernel_v8(pangulu_int64_t n,
                                  pangulu_int64_t nnzl,
                                  pangulu_int64_t nnzx,
                                  int *degree,
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
                                  calculate_type *a_value)
{

    int num_threads = warp_per_block_dgessm * warp_size;
    int num_blocks = ceil((calculate_type)n / (calculate_type)warp_per_block_dgessm);
    syncfree_cuda_csr<<<num_blocks, num_threads>>>(n, n, degree, l_columnpointer, l_rowindex, l_value,
                                                   a_columnpointer, a_rowindex, x_value, a_value, d_id_extractor, d_left_sum);
    cudaDeviceSynchronize();
}

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
                                   calculate_type *a_value)
{
    pangulu_int64_t num_threads = warp_size * warp_per_block;
    pangulu_int64_t num_blocks;

    num_blocks = ceil((calculate_type)n / warp_per_block);
    cuda_transform_s_to_d_col<<<num_blocks, num_threads>>>(n, num_blocks, a_columnpointer, a_rowindex, a_value, cuda_temp_value);
    cudaDeviceSynchronize();

    num_threads = 8 * warp_size;
    num_blocks = n;
    gessm_kernel_dense<<<num_blocks, num_threads>>>(n, l_columnpointer, l_rowindex, l_value,
                                                    x_columnpointer, x_rowindex, x_value,
                                                    a_columnpointer, a_rowindex, a_value, cuda_temp_value);
    cudaDeviceSynchronize();
}

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
                               calculate_type *d_valuec)
{

    pangulu_int64_t num_blocks;
    pangulu_int64_t num_threads;
    pangulu_int64_t layer;

    // 0:0
    layer = 0;

    // 1:1
    layer = 1;
    num_threads = warp_per_block_gemm * warp_size;
    num_blocks = ceil((double)(h_bin_rowpointer[2] - h_bin_rowpointer[1]) / num_threads);
    if (num_blocks > 0)
    {
        threadlevel_spgemm<<<num_blocks, num_threads>>>(n, layer, d_bin_rowpointer, d_bin_rowindex,
                                                        d_rowptra, d_colidxa, d_valuea,
                                                        d_rowptrb, d_colidxb, d_valueb,
                                                        d_rowptrc, d_colidxc, d_valuec);
    }

    // 2:2-32
    layer = 2;
    num_threads = warp_per_block_gemm * warp_size;
    num_blocks = ceil((double)(h_bin_rowpointer[3] - h_bin_rowpointer[2]) / warp_per_block_gemm);
    if (num_blocks > 0)
    {
        warplevel_spgemm_32<<<num_blocks, num_threads>>>(n, layer, d_bin_rowpointer, d_bin_rowindex,
                                                         d_rowptra, d_colidxa, d_valuea,
                                                         d_rowptrb, d_colidxb, d_valueb,
                                                         d_rowptrc, d_colidxc, d_valuec);
    }

    // 3:33-64
    layer = 3;
    num_threads = warp_per_block_gemm * warp_size;
    num_blocks = ceil((double)(h_bin_rowpointer[4] - h_bin_rowpointer[3]) / warp_per_block_gemm);
    if (num_blocks > 0)
    {
        warplevel_spgemm_64<<<num_blocks, num_threads>>>(n, layer, d_bin_rowpointer, d_bin_rowindex,
                                                         d_rowptra, d_colidxa, d_valuea,
                                                         d_rowptrb, d_colidxb, d_valueb,
                                                         d_rowptrc, d_colidxc, d_valuec);
    }

    // 4:65-128
    layer = 4;
    num_threads = warp_per_block_gemm * warp_size;
    num_blocks = ceil((double)(h_bin_rowpointer[5] - h_bin_rowpointer[4]) / warp_per_block_gemm);
    if (num_blocks > 0)
    {
        warplevel_spgemm_128<<<num_blocks, num_threads>>>(n, layer, d_bin_rowpointer, d_bin_rowindex,
                                                          d_rowptra, d_colidxa, d_valuea,
                                                          d_rowptrb, d_colidxb, d_valueb,
                                                          d_rowptrc, d_colidxc, d_valuec);
    }

    // 5:129-256
    layer = 5;
    num_threads = 64;
    num_blocks = h_bin_rowpointer[6] - h_bin_rowpointer[5];
    if (num_blocks > 0)
    {
        blocklevel_spgemm_256<<<num_blocks, num_threads>>>(n, layer, d_bin_rowpointer, d_bin_rowindex,
                                                           d_rowptra, d_colidxa, d_valuea,
                                                           d_rowptrb, d_colidxb, d_valueb,
                                                           d_rowptrc, d_colidxc, d_valuec);
    }

    // 6:257-512
    layer = 6;
    num_threads = 128;
    num_blocks = h_bin_rowpointer[7] - h_bin_rowpointer[6];
    if (num_blocks > 0)
    {
        blocklevel_spgemm_512<<<num_blocks, num_threads>>>(n, layer, d_bin_rowpointer, d_bin_rowindex,
                                                           d_rowptra, d_colidxa, d_valuea,
                                                           d_rowptrb, d_colidxb, d_valueb,
                                                           d_rowptrc, d_colidxc, d_valuec);
    }

    // 7:513-1024
    layer = 7;
    num_threads = 256;
    num_blocks = h_bin_rowpointer[8] - h_bin_rowpointer[7];
    if (num_blocks > 0)
    {
        blocklevel_spgemm_1024<<<num_blocks, num_threads>>>(n, layer, d_bin_rowpointer, d_bin_rowindex,
                                                            d_rowptra, d_colidxa, d_valuea,
                                                            d_rowptrb, d_colidxb, d_valueb,
                                                            d_rowptrc, d_colidxc, d_valuec);
    }

    // 8:1025-2048
    layer = 8;
    num_threads = 512;
    num_blocks = h_bin_rowpointer[9] - h_bin_rowpointer[8];
    if (num_blocks > 0)
    {
        blocklevel_spgemm_2048<<<num_blocks, num_threads>>>(n, layer, d_bin_rowpointer, d_bin_rowindex,
                                                            d_rowptra, d_colidxa, d_valuea,
                                                            d_rowptrb, d_colidxb, d_valueb,
                                                            d_rowptrc, d_colidxc, d_valuec);
    }

    // 9:2049++
    layer = 9;
    num_threads = 1024;
    num_blocks = h_bin_rowpointer[10] - h_bin_rowpointer[9];
    if (num_blocks > 0)
    {
        blocklevel_spgemm_4097<<<num_blocks, num_threads>>>(n, layer, d_bin_rowpointer, d_bin_rowindex,
                                                            d_rowptra, d_colidxa, d_valuea,
                                                            d_rowptrb, d_colidxb, d_valueb,
                                                            d_rowptrc, d_colidxc, d_valuec);
    }

    // 10:4097+++
    layer = 10;
    num_threads = 1024;
    num_blocks = h_bin_rowpointer[11] - h_bin_rowpointer[10];
    if (num_blocks > 0)
    {
        blocklevel_spgemm_4097<<<num_blocks, num_threads>>>(n, layer, d_bin_rowpointer, d_bin_rowindex,
                                                            d_rowptra, d_colidxa, d_valuea,
                                                            d_rowptrb, d_colidxb, d_valueb,
                                                            d_rowptrc, d_colidxc, d_valuec);
    }
}

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
                                     calculate_type *d_valuec)
{

    pangulu_int64_t num_blocks, num_blocks_b, num_blocks_c;
    pangulu_int64_t num_threads;
    pangulu_int64_t layer;
    num_threads = warp_per_block_gemm * warp_size;

    num_blocks = ceil((double)n / warp_per_block_gemm);

    cuda_transform_csc_to_coo<<<num_blocks, num_threads>>>(n, d_rowptrb, d_colidxb, cuda_b_idx_col);

    num_blocks = ceil((double)n / warp_per_block_gemm);

    cuda_transform_s_to_d_col<<<num_blocks, num_threads>>>(n, num_blocks, d_rowptrc, d_colidxc, d_valuec, cuda_temp_value);

    num_blocks_b = ceil((double)nnzb / warp_per_block_gemm);

    wraplevel_spgemm_dense_nnz<<<num_blocks_b, num_threads>>>(n, nnzb, d_rowptra, d_colidxa, d_valuea, d_rowptrb, d_colidxb, d_valueb, d_rowptrc, d_colidxc, d_valuec, cuda_b_idx_col, cuda_temp_value);

    cuda_transform_d_to_s_col<<<num_blocks, num_threads>>>(n, num_blocks, d_rowptrc, d_colidxc, d_valuec, cuda_temp_value);
}

void pangulu_cudamemcpyasync_host_to_device(void *gpu_address, void *cpu_address, pangulu_int64_t size, cudaStream_t *stream)
{
    cudaMemcpyAsync(gpu_address, cpu_address, size, cudaMemcpyHostToDevice, *stream);
}

void pangulu_cudamemcpyasync_device_to_host(void *cpu_address, void *gpu_address, pangulu_int64_t size, cudaStream_t *stream)
{
    cudaMemcpyAsync(cpu_address, gpu_address, size, cudaMemcpyDeviceToHost, *stream);
}

void pangulu_create_cudastream(cudaStream_t *stream)
{
    cudaStreamCreate(stream);
}

void pangulu_destroy_cudastream(cudaStream_t *stream)
{
    cudaStreamDestroy(*stream);
}

void pangulu_create_cudaevent(cudaEvent_t *event)
{
    cudaEventCreate(event);
}

void pangulu_destroy_cudaevent(cudaEvent_t *event)
{
    cudaEventDestroy(*event);
}

void pangulu_eventrecord(cudaEvent_t *event, cudaStream_t *stream)
{
    cudaEventRecord(*event, *stream);
}

void pangulu_eventsynchronize(cudaEvent_t *event)
{
    cudaEventSynchronize(*event);
}
void pangulu_cudamemcpy_device_to_device(void *gpu_address1, void *gpu_address2, pangulu_int64_t size)
{
    cudaMemcpy(gpu_address1, gpu_address2, size, cudaMemcpyHostToDevice);
}

void pangulu_cudamemcpyasync_device_to_device(void *gpu_address1, void *gpu_address2, pangulu_int64_t size, cudaStream_t *stream)
{
    cudaMemcpyAsync(gpu_address1, gpu_address2, size, cudaMemcpyHostToDevice, *stream);
}
#endif // defined(PANGULU_COMPLEX)