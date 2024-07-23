#include "pangulu_cuda.h"

#ifndef SUBSTITUTION_FORWARD
#define SUBSTITUTION_FORWARD 1
#endif

__global__ void GESSM_Kernel_v2(int_t n,
                                pangulu_inblock_ptr *L_columnpointer,
                                pangulu_inblock_idx *L_rowindex,
                                calculate_type *L_VALUE,
                                pangulu_inblock_ptr *X_columnpointer,
                                pangulu_inblock_idx *X_rowindex,
                                calculate_type *X_VALUE,
                                pangulu_inblock_ptr *A_columnpointer,
                                pangulu_inblock_idx *A_rowindex,
                                calculate_type *A_VALUE)
{
    int_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    int_t lane_id = threadIdx.x % 32;
    int_t warp_local_id = threadIdx.x / 32;
    int_t flagx, flagb;

    if (warp_id >= n || X_columnpointer[warp_id] == X_columnpointer[warp_id + 1])
        return;

    int_t colA1 = A_columnpointer[warp_id];
    int_t colA2 = A_columnpointer[warp_id + 1];
    int_t colX1 = X_columnpointer[warp_id];
    int_t colX2 = X_columnpointer[warp_id + 1];

    if (colA2 - colA1 >= SM_LEN_DGESSM)
    {
        for (int_t i = colX1, t = 0; i < colX2; i++, t++)
        {
            int_t rowX = X_rowindex[i];
            X_VALUE[i] = A_VALUE[i];
            calculate_type valx = (calculate_type)(X_VALUE[i]);

            int_t colL1 = L_columnpointer[rowX];
            int_t colL2 = L_columnpointer[rowX + 1];
            for (int_t j = colL1 + 1 + lane_id, p = lane_id; j < colL2; j += WARP_SIZE, p += WARP_SIZE)
            {
                pangulu_inblock_idx f = binarySearch_idx(A_rowindex, colA1 + 1 + t + p, colA2 - colL2 + j, L_rowindex[j]);
                A_VALUE[f] -= valx * L_VALUE[j];
            }
            __syncthreads();
        }
    }
    else
    {
        __shared__ pangulu_inblock_idx s_rowidxA[SM_LEN_DGESSM * WARP_PER_BLOCK_DGESSM];
        __shared__ calculate_type s_valueA[SM_LEN_DGESSM * WARP_PER_BLOCK_DGESSM];

        pangulu_inblock_idx *idx_local = &s_rowidxA[warp_local_id * SM_LEN_DGESSM];
        calculate_type *val_local = &s_valueA[warp_local_id * SM_LEN_DGESSM];

        for (int_t i = lane_id; i < colA2 - colA1; i += WARP_SIZE)
        {
            idx_local[i] = A_rowindex[colA1 + i];
            val_local[i] = A_VALUE[colA1 + i];
        }

        for (int_t i = colX1, t = 0; i < colX2; i++, t++)
        {
            int_t rowX = X_rowindex[i];
            X_VALUE[i] = val_local[t];
            calculate_type valx = (calculate_type)(X_VALUE[i]);

            int_t colL1 = L_columnpointer[rowX];
            int_t colL2 = L_columnpointer[rowX + 1];
            for (int_t j = colL1 + 1 + lane_id, p = lane_id; j < colL2; j += WARP_SIZE, p += WARP_SIZE)
            {
                // update A's Value;
                pangulu_inblock_idx f = binarySearch_idx(idx_local, 1 + t + p, colA2 - colA1 - colL2 + j, L_rowindex[j]);
                if (f != -1)
                    val_local[f] -= valx * L_VALUE[j];
            }
            __syncthreads();
        }

        for (int_t i = lane_id; i < colA2 - colA1; i += WARP_SIZE)
        {
            A_VALUE[colA1 + i] = val_local[i];
        }
    }
}

__global__ void GESSM_Kernel_v3(int_t n,
                                pangulu_inblock_ptr *L_columnpointer,
                                pangulu_inblock_idx *L_rowindex,
                                calculate_type *L_VALUE,
                                pangulu_inblock_ptr *X_columnpointer,
                                pangulu_inblock_idx *X_rowindex,
                                calculate_type *X_VALUE,
                                pangulu_inblock_ptr *A_columnpointer,
                                pangulu_inblock_idx *A_rowindex,
                                calculate_type *A_VALUE)
{

    int_t colidx = blockIdx.x;
    int_t colX1 = X_columnpointer[colidx];
    int_t colX2 = X_columnpointer[colidx + 1];

    if (colidx >= n || colX2 == colX1)
        return;

    int_t colA1 = A_columnpointer[colidx];
    int_t colA2 = A_columnpointer[colidx + 1];

    if (colA2 - colA1 >= SM_LEN_DGESSM)
    {
        for (int_t i = colX1, t = 0; i < colX2; i++, t++)
        {
            int_t rowX = X_rowindex[i];
            X_VALUE[i] = A_VALUE[i];
            calculate_type valx = X_VALUE[i];

            int_t colL1 = L_columnpointer[rowX];
            int_t colL2 = L_columnpointer[rowX + 1];

            for (int_t j = colL1 + 1 + threadIdx.x, p = threadIdx.x; j < colL2; j += blockDim.x, p += blockDim.x)
            {
                // update A's Value;
                pangulu_inblock_idx f = binarySearch_idx(A_rowindex, colA1 + 1 + t + p, colA2 - colL2 + j, L_rowindex[j]);
                A_VALUE[f] -= valx * L_VALUE[j];
            }
            __syncthreads();
        }
    }
    else
    {
        __shared__ pangulu_inblock_idx s_idxA[SM_LEN_DGESSM];
        __shared__ calculate_type s_valA[SM_LEN_DGESSM];

        for (int_t i = threadIdx.x; i < colA2 - colA1; i += blockDim.x)
        {
            s_idxA[i] = A_rowindex[colA1 + i];
            s_valA[i] = A_VALUE[colA1 + i];
        }
        __syncthreads();

        for (int_t i = colX1, t = 0; i < colX2; i++, t++)
        {
            int_t rowX = X_rowindex[i];
            X_VALUE[i] = s_valA[t];
            calculate_type valx = X_VALUE[i];

            int_t colL1 = L_columnpointer[rowX];
            int_t colL2 = L_columnpointer[rowX + 1];

            for (int_t j = colL1 + 1 + threadIdx.x, p = threadIdx.x; j < colL2; j += blockDim.x, p += blockDim.x)
            {
                // update A's Value;
                pangulu_inblock_idx f = binarySearch_idx(s_idxA, 1 + t + p, colA2 - colA1 - colL2 + j, L_rowindex[j]);
                s_valA[f] -= valx * L_VALUE[j];
            }
            __syncthreads();
        }

        for (int_t i = threadIdx.x; i < colA2 - colA1; i += blockDim.x)
        {
            A_VALUE[colA1 + i] = s_valA[i];
        }
        //__syncthreads();
    }
}

__global__ void TSTRF_Kernel_v2(int_t n,
                                pangulu_inblock_ptr *U_rowpointer,
                                pangulu_inblock_idx *U_columnindex,
                                calculate_type *U_VALUE,
                                pangulu_inblock_ptr *X_rowpointer,
                                pangulu_inblock_idx *X_columnindex,
                                calculate_type *X_VALUE,
                                pangulu_inblock_ptr *A_rowpointer,
                                pangulu_inblock_idx *A_columnindex,
                                calculate_type *A_VALUE)
{
    int_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    int_t lane_id = threadIdx.x % 32;
    int_t warp_local_id = threadIdx.x / 32;

    if (warp_id >= n || X_rowpointer[warp_id] == X_rowpointer[warp_id + 1])
        return;

    int_t colA1 = A_rowpointer[warp_id];
    int_t colA2 = A_rowpointer[warp_id + 1];
    int_t colX1 = X_rowpointer[warp_id];
    int_t colX2 = X_rowpointer[warp_id + 1];

    if (colA2 - colA1 >= SM_LEN_DTSTRF)
    {
        for (int_t i = colX1, t = 0; i < colX2; i++, t++)
        {
            int_t rowX = X_columnindex[i];
            int_t colU1 = U_rowpointer[rowX];
            int_t colU2 = U_rowpointer[rowX + 1];

            X_VALUE[i] = A_VALUE[i] / U_VALUE[colU1];
            calculate_type valx = X_VALUE[i];
            for (int_t j = colU1 + 1 + lane_id, p = lane_id; j < colU2; j += WARP_SIZE, p += WARP_SIZE)
            {
                // update A's Value;
                pangulu_inblock_idx f = binarySearch_idx(A_columnindex, colA1 + t + 1 + p, colA2 - colU2 + j, U_columnindex[j]);
                A_VALUE[f] -= valx * U_VALUE[j];
            }
            __syncthreads();
        }
    }
    else
    {
        __shared__ pangulu_inblock_idx s_rowidxA[SM_LEN_DTSTRF * WARP_PER_BLOCK_DTSTRF];
        __shared__ calculate_type s_valueA[SM_LEN_DTSTRF * WARP_PER_BLOCK_DTSTRF];

        pangulu_inblock_idx *idx_local = &s_rowidxA[warp_local_id * SM_LEN_DTSTRF];
        calculate_type *val_local = &s_valueA[warp_local_id * SM_LEN_DTSTRF];

        for (int_t i = lane_id; i < colA2 - colA1; i += WARP_SIZE)
        {
            idx_local[i] = A_columnindex[colA1 + i];
            val_local[i] = A_VALUE[colA1 + i];
        }

        for (int_t i = colX1, t = 0; i < colX2; i++, t++)
        {
            int_t rowX = X_columnindex[i];
            int_t colU1 = U_rowpointer[rowX];
            int_t colU2 = U_rowpointer[rowX + 1];

            X_VALUE[i] = val_local[t] / U_VALUE[colU1];
            calculate_type valx = X_VALUE[i];
            for (int_t j = colU1 + 1 + lane_id, p = lane_id; j < colU2; j += WARP_SIZE, p += WARP_SIZE)
            {
                // update A's Value;
                pangulu_inblock_idx f = binarySearch_idx(idx_local, 1 + t + p, colA2 - colA1 - colU2 + j, U_columnindex[j]);
                val_local[f] -= valx * U_VALUE[j];
            }
            __syncthreads();
        }

        for (int_t i = lane_id; i < colA2 - colA1; i += WARP_SIZE)
        {
            A_VALUE[colA1 + i] = val_local[i];
        }
    }
}
__global__ void TSTRF_Kernel_v3(int_t n,
                                pangulu_inblock_ptr *U_rowpointer,
                                pangulu_inblock_idx *U_columnindex,
                                calculate_type *U_VALUE,
                                pangulu_inblock_ptr *X_rowpointer,
                                pangulu_inblock_idx *X_columnindex,
                                calculate_type *X_VALUE,
                                pangulu_inblock_ptr *A_rowpointer,
                                pangulu_inblock_idx *A_columnindex,
                                calculate_type *A_VALUE)
{
    int_t colidx = blockIdx.x;
    int_t colX1 = X_rowpointer[colidx];
    int_t colX2 = X_rowpointer[colidx + 1];

    if (colidx >= n || colX1 == colX2)
        return;

    int_t colA1 = A_rowpointer[colidx];
    int_t colA2 = A_rowpointer[colidx + 1];

    if (colA2 - colA1 >= SM_LEN_DTSTRF)
    {
        for (int_t i = colX1, t = 0; i < colX2; i++, t++)
        {
            int_t rowX = X_columnindex[i];
            int_t colU1 = U_rowpointer[rowX];
            int_t colU2 = U_rowpointer[rowX + 1];

            X_VALUE[i] = A_VALUE[i] / U_VALUE[colU1];
            calculate_type valx = X_VALUE[i];
            for (int_t j = colU1 + 1 + threadIdx.x, p = threadIdx.x; j < colU2; j += blockDim.x, p += blockDim.x)
            {
                // update A's Value;
                pangulu_inblock_idx f = binarySearch_idx(A_columnindex, colA1 + t + 1 + p, colA2 - colU2 + j, U_columnindex[j]);
                A_VALUE[f] -= valx * U_VALUE[j];
            }
            __syncthreads();
        }
    }
    else
    {
        __shared__ pangulu_inblock_idx s_rowidxA[SM_LEN_DTSTRF];
        __shared__ calculate_type s_valueA[SM_LEN_DTSTRF];

        for (int_t i = threadIdx.x; i < colA2 - colA1; i += blockDim.x)
        {
            s_rowidxA[i] = A_columnindex[colA1 + i];
            s_valueA[i] = A_VALUE[colA1 + i];
        }
        __syncthreads();

        for (int_t i = colX1, t = 0; i < colX2; i++, t++)
        {
            int_t rowX = X_columnindex[i];
            int_t colU1 = U_rowpointer[rowX];
            int_t colU2 = U_rowpointer[rowX + 1];

            X_VALUE[i] = s_valueA[t] / U_VALUE[colU1];
            calculate_type valx = X_VALUE[i];
            for (int_t j = colU1 + 1 + threadIdx.x, p = threadIdx.x; j < colU2; j += blockDim.x, p += blockDim.x)
            {
                // update A's Value;
                pangulu_inblock_idx f = binarySearch_idx(s_rowidxA, 1 + t + p, colA2 - colA1 - colU2 + j, U_columnindex[j]);
                s_valueA[f] -= valx * U_VALUE[j];
            }
            __syncthreads();
        }

        for (int_t i = threadIdx.x; i < colA2 - colA1; i += blockDim.x)
        {
            A_VALUE[colA1 + i] = s_valueA[i];
        }
    }
}

__global__ void syncfree_cuda_csr(
    int_t n,
    int_t rhs,
    int *degree,
    pangulu_inblock_ptr *L_columnpointer,
    pangulu_inblock_idx *L_rowindex,
    calculate_type *L_VALUE,
    pangulu_inblock_ptr *X_rowpointer,
    pangulu_inblock_idx *X_colindex,
    calculate_type *X_VALUE,
    calculate_type *A_VALUE,
    int *d_id_extractor,
    calculate_type *d_left_sum)
{
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

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

    const int_t loc1 = X_rowpointer[global_x_id];
    const int_t loc2 = X_rowpointer[global_x_id + 1];
    const int_t start_ptr = L_columnpointer[global_x_id] + 1;
    const int_t stop_ptr = L_columnpointer[global_x_id + 1];

    calculate_type res = 0;

    for (int_t m = loc1 + lane_id; m < loc2; m += warpSize)
    {
        X_VALUE[m] = A_VALUE[m] - d_left_sum[m];
    }
    if (loc1 != loc2)
        for (int_t jj = start_ptr + lane_id; jj < stop_ptr; jj += warpSize)
        {
            const pangulu_inblock_idx myidx = L_rowindex[jj];
            const int_t mypos = X_rowpointer[myidx];
            const int_t mypos2 = X_rowpointer[myidx + 1];
            const calculate_type L_id_value = L_VALUE[jj];

            if (mypos2 - mypos > 200)
            {
                for (int_t k = loc1; k < loc2; k++)
                {
                    pangulu_inblock_idx f = binarySearch_idx(X_colindex, mypos, mypos2, X_colindex[k]);
                    res = X_VALUE[k] * L_id_value;
                    // res = 0;
                    atomicAdd(&d_left_sum[f], res);
                }
            }
            else
            {
                for (int_t p = mypos, k = loc1; p < mypos2 && k < loc2; p++, k++)
                {

                    if (X_colindex[p] == X_colindex[k])
                    {
                        res = X_VALUE[k] * L_id_value;
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
        for (int_t jj = start_ptr + lane_id; jj < stop_ptr; jj += warpSize)
        {
            atomicSub(&degree[L_rowindex[jj]], 1);
        }
    }
}

__global__ void syncfree_cuda_csr_U(
    int_t n,
    int_t rhs,
    int *degree,
    pangulu_inblock_ptr *L_columnpointer,
    pangulu_inblock_idx *L_rowindex,
    calculate_type *L_VALUE,
    pangulu_inblock_ptr *X_rowpointer,
    pangulu_inblock_idx *X_colindex,
    calculate_type *X_VALUE,
    calculate_type *A_VALUE,
    int *d_id_extractor,
    calculate_type *d_left_sum)
{
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
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

    const int_t loc1 = X_rowpointer[global_x_id];
    const int_t loc2 = X_rowpointer[global_x_id + 1];
    const int_t start_ptr = L_columnpointer[global_x_id] + 1;
    const int_t stop_ptr = L_columnpointer[global_x_id + 1];

    calculate_type res = 0;

    for (int_t m = loc1 + lane_id; m < loc2; m += warpSize)
    {
        X_VALUE[m] = (A_VALUE[m] - d_left_sum[m]) / L_VALUE[start_ptr - 1];
    }
    if (loc1 != loc2)
        for (int_t jj = start_ptr + lane_id; jj < stop_ptr; jj += warpSize)
        {
            const pangulu_inblock_idx myidx = L_rowindex[jj];
            const int_t mypos = X_rowpointer[myidx];
            const int_t mypos2 = X_rowpointer[myidx + 1];
            const calculate_type L_id_value = L_VALUE[jj];

            if (mypos2 - mypos > 200)
            {
                for (int_t k = loc1; k < loc2; k++)
                {
                    pangulu_inblock_idx f = binarySearch_idx(X_colindex, mypos, mypos2, X_colindex[k]);
                    res = X_VALUE[k] * L_id_value;
                    atomicAdd(&d_left_sum[f], res);
                }
            }
            else
            {
                for (int_t p = mypos, k = loc1; p < mypos2 && k < loc2; p++, k++)
                {
                    if (X_colindex[p] == X_colindex[k])
                    {
                        res = X_VALUE[k] * L_id_value;
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
        for (int_t jj = start_ptr + lane_id; jj < stop_ptr; jj += warpSize)
        {
            atomicSub(&degree[L_rowindex[jj]], 1);
        }
    }
}
void pangulu_tstrf_cuda_kernel_v8(int_t n,
                                  int_t nnzU,
                                  int *degree,
                                  int *d_id_extractor,
                                  calculate_type *d_left_sum,
                                  pangulu_inblock_ptr *U_rowpointer,
                                  pangulu_inblock_idx *U_columnindex,
                                  calculate_type *U_VALUE,
                                  pangulu_inblock_ptr *X_rowpointer,
                                  pangulu_inblock_idx *X_columnindex,
                                  calculate_type *X_VALUE,
                                  pangulu_inblock_ptr *A_rowpointer,
                                  pangulu_inblock_idx *A_columnindex,
                                  calculate_type *A_VALUE)
{

    int num_threads = WARP_PER_BLOCK_DGESSM * WARP_SIZE;
    int num_blocks = ceil((calculate_type)n / (calculate_type)WARP_PER_BLOCK_DGESSM);
    syncfree_cuda_csr_U<<<num_blocks, num_threads>>>(n, n, degree, U_rowpointer, U_columnindex, U_VALUE,
                                                     A_rowpointer, A_columnindex, X_VALUE, A_VALUE, d_id_extractor, d_left_sum);
    cudaDeviceSynchronize();
}

__global__ void sptrsv_syncfree_cuda_executor_update(const pangulu_inblock_ptr *d_cscColPtr,
                                                     const pangulu_inblock_idx *d_cscRowIdx,
                                                     const calculate_type *d_cscVal,
                                                     int *d_graphInDegree,
                                                     calculate_type *d_left_sum,
                                                     const int_t m,
                                                     const int_t substitution,
                                                     const calculate_type *d_b,
                                                     calculate_type *d_x,
                                                     int *d_while_profiler,
                                                     int *d_id_extractor)
{
    const int_t global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize
    const int_t local_warp_id = threadIdx.x / WARP_SIZE;
    const int_t lane_id = (WARP_SIZE - 1) & threadIdx.x;
    int_t global_x_id = 0;
    if (!lane_id)
        global_x_id = atomicAdd(d_id_extractor, 1);
    global_x_id = __shfl(global_x_id, 0);

    if (global_x_id >= m)
        return;

    // substitution is forward or backward
    global_x_id = substitution == SUBSTITUTION_FORWARD ? global_x_id : m - 1 - global_x_id;
    // Prefetch
    const int_t pos = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[global_x_id] : d_cscColPtr[global_x_id + 1] - 1;
    const calculate_type coef = (calculate_type)1 / d_cscVal[pos];
    // Consumer
    do
    {
        __threadfence_block();
    } while (d_graphInDegree[global_x_id] != 1);

    calculate_type xi = d_left_sum[global_x_id];
    xi = (d_b[global_x_id] - xi) * coef;

    // Producer
    const int_t start_ptr = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[global_x_id] + 1 : d_cscColPtr[global_x_id];
    const int_t stop_ptr = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[global_x_id + 1] : d_cscColPtr[global_x_id + 1] - 1;
    for (int_t jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
    {
        const int_t j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
        const pangulu_inblock_idx rowIdx = d_cscRowIdx[j];

        atomicAdd(&d_left_sum[rowIdx], xi * d_cscVal[j]);
        __threadfence();
        atomicSub(&d_graphInDegree[rowIdx], 1);
    }

    // finish
    if (!lane_id)
        d_x[global_x_id] = xi;
}
__global__ void sptrsm_syncfree_cuda_executor_update(const pangulu_inblock_ptr *__restrict__ d_cscColPtr,
                                                     const pangulu_inblock_idx *__restrict__ d_cscRowIdx,
                                                     const calculate_type *__restrict__ d_cscVal,
                                                     int *d_graphInDegree,
                                                     calculate_type *d_left_sum,
                                                     const int_t m,
                                                     const int substitution,
                                                     const int rhs,

                                                     const calculate_type *__restrict__ d_b,
                                                     calculate_type *d_x,
                                                     int *d_while_profiler,
                                                     int *d_id_extractor)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialize
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    int global_x_id = 0;
    if (!lane_id)
        global_x_id = atomicAdd(d_id_extractor, 1);
    global_x_id = __shfl_sync(0xffffffff, global_x_id, 0);

    if (global_x_id >= m)
        return;

    // substitution is forward or backward
    global_x_id = substitution == SUBSTITUTION_FORWARD ? global_x_id : m - 1 - global_x_id;
    // Prefetch
    const int pos = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[global_x_id] : d_cscColPtr[global_x_id + 1] - 1;
    const calculate_type coef = (calculate_type)1 / d_cscVal[pos];
    // Consumer
    do
    {
        __threadfence_block();
    } while (1 != d_graphInDegree[global_x_id]);

    for (int k = lane_id; k < rhs; k += WARP_SIZE)
    {
        const int pos = global_x_id * rhs + k;
        d_x[pos] = (d_b[pos] - d_left_sum[pos]) * coef;
    }

    // Producer
    const int_t start_ptr = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[global_x_id] + 1 : d_cscColPtr[global_x_id];
    const int_t stop_ptr = substitution == SUBSTITUTION_FORWARD ? d_cscColPtr[global_x_id + 1] : d_cscColPtr[global_x_id + 1] - 1;

    for (int_t jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
    {
        const int_t j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
        const pangulu_inblock_idx rowIdx = d_cscRowIdx[j];
        for (int k = 0; k < rhs; k++)
            atomicAdd(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
        __threadfence();
        atomicSub(&d_graphInDegree[rowIdx], 1);
    }
}
__global__ void sptrsv_sparse_to_dense(const pangulu_inblock_ptr *d_cscptrA,
                                       const pangulu_inblock_idx *d_cscidx,
                                       const calculate_type *d_value,
                                       pangulu_inblock_ptr *Spointer,
                                       const int_t rhs,

                                       calculate_type *d_b)
{
    int_t global_x_id = blockIdx.x;

    if (global_x_id >= rhs)
        return;
    else
    {
        for (int_t i = d_cscptrA[Spointer[global_x_id]] + threadIdx.x; i < d_cscptrA[Spointer[global_x_id] + 1]; i += WARP_SIZE)
        {
            d_b[global_x_id + d_cscidx[i] * rhs] = d_value[i];
        }
    }
}
__global__ void sptrsv_dense_to_sparse(const pangulu_inblock_ptr *d_cscptrA,
                                       const pangulu_inblock_idx *d_cscidx,
                                       calculate_type *d_value,
                                       pangulu_inblock_ptr *Spointer,

                                       const int_t rhs,

                                       calculate_type *d_x)
{
    int_t global_x_id = blockIdx.x;

    if (global_x_id >= rhs)
        return;
    else
    {
        int_t index = Spointer[global_x_id];
        for (int_t i = d_cscptrA[index] + threadIdx.x; i < d_cscptrA[index + 1]; i += WARP_SIZE)
        {
            d_value[i] = d_x[global_x_id + d_cscidx[i] * rhs];
        }
    }
}

void pangulu_gessm_cuda_kernel_v9(int_t n,
                                  int_t nnzL,
                                  int_t rhs,
                                  int_t nnzA,
                                  pangulu_inblock_ptr *d_Spointer,
                                  int *d_graphInDegree,
                                  int *d_id_extractor,
                                  int *d_while_profiler,
                                  pangulu_inblock_ptr *L_columnpointer,
                                  pangulu_inblock_idx *L_rowindex,
                                  calculate_type *L_VALUE,
                                  pangulu_inblock_ptr *X_columnpointer,
                                  pangulu_inblock_idx *X_rowindex,
                                  calculate_type *X_VALUE,
                                  pangulu_inblock_ptr *A_columnpointer_cuda,
                                  pangulu_inblock_idx *A_rowindex_cuda,
                                  calculate_type *A_VALUE_cuda,
                                  calculate_type *d_left_sum,
                                  calculate_type *d_x,
                                  calculate_type *d_b)
{

    /*********************************************pre***********************************************/
    int_t num_threads = 8 * WARP_SIZE;
    int_t num_blocks = rhs;
    sptrsv_sparse_to_dense<<<num_blocks, num_threads>>>(A_columnpointer_cuda, A_rowindex_cuda, A_VALUE_cuda, d_Spointer, rhs, d_b);
    cudaDeviceSynchronize();

    int substitution = 1;
    /*********************************************calculate***********************************************/
    if (rhs == 1)
    {
        num_threads = 16 * WARP_SIZE;

        num_blocks = ceil((calculate_type)n / (calculate_type)(num_threads / WARP_SIZE));
        sptrsv_syncfree_cuda_executor_update<<<num_blocks, num_threads>>>(L_columnpointer, L_rowindex, L_VALUE,
                                                                          d_graphInDegree, d_left_sum,
                                                                          n, substitution, d_b, d_x, d_while_profiler, d_id_extractor);
    }
    else
    {
        num_threads = 4 * WARP_SIZE;
        num_blocks = ceil((calculate_type)n / (calculate_type)(num_threads / WARP_SIZE));
        sptrsm_syncfree_cuda_executor_update<<<num_blocks, num_threads>>>(L_columnpointer, L_rowindex, L_VALUE,
                                                                          d_graphInDegree, d_left_sum,
                                                                          n, substitution, rhs,
                                                                          d_b, d_x, d_while_profiler, d_id_extractor);
    }
    cudaDeviceSynchronize();
    /*********************************************calculate***********************************************/

    num_threads = 8 * WARP_SIZE;
    num_blocks = rhs;
    sptrsv_dense_to_sparse<<<num_blocks, num_threads>>>(A_columnpointer_cuda, A_rowindex_cuda, X_VALUE, d_Spointer, rhs, d_x);
    cudaDeviceSynchronize();
}
void pangulu_tstrf_cuda_kernel_v9(int_t n,
                                  int_t nnzL,
                                  int_t rhs,
                                  int_t nnzA,
                                  pangulu_inblock_ptr *d_Spointer,
                                  int *d_graphInDegree,
                                  int *d_id_extractor,
                                  int *d_while_profiler,
                                  pangulu_inblock_ptr *L_columnpointer,
                                  pangulu_inblock_idx *L_rowindex,
                                  calculate_type *L_VALUE,
                                  pangulu_inblock_ptr *X_columnpointer,
                                  pangulu_inblock_idx *X_rowindex,
                                  calculate_type *X_VALUE,
                                  pangulu_inblock_ptr *A_columnpointer_cuda,
                                  pangulu_inblock_idx *A_rowindex_cuda,
                                  calculate_type *A_VALUE_cuda,
                                  calculate_type *d_left_sum,
                                  calculate_type *d_x,
                                  calculate_type *d_b)
{

    /*********************************************pre***********************************************/
    int_t num_threads = 8 * WARP_SIZE;
    int_t num_blocks = rhs;
    sptrsv_sparse_to_dense<<<num_blocks, num_threads>>>(A_columnpointer_cuda, A_rowindex_cuda, A_VALUE_cuda, d_Spointer, rhs, d_b);
    cudaDeviceSynchronize();
    int substitution = 1;
    /*********************************************calculate***********************************************/
    if (rhs == 1)
    {
        num_threads = 16 * WARP_SIZE;

        num_blocks = ceil((calculate_type)n / (calculate_type)(num_threads / WARP_SIZE));
        sptrsv_syncfree_cuda_executor_update<<<num_blocks, num_threads>>>(L_columnpointer, L_rowindex, L_VALUE,
                                                                          d_graphInDegree, d_left_sum,
                                                                          n, substitution, d_b, d_x, d_while_profiler, d_id_extractor);
    }
    else
    {
        num_threads = 4 * WARP_SIZE;
        num_blocks = ceil((calculate_type)n / (calculate_type)(num_threads / WARP_SIZE));
        sptrsm_syncfree_cuda_executor_update<<<num_blocks, num_threads>>>(L_columnpointer, L_rowindex, L_VALUE,
                                                                          d_graphInDegree, d_left_sum,
                                                                          n, substitution, rhs,
                                                                          d_b, d_x, d_while_profiler, d_id_extractor);
    }
    cudaDeviceSynchronize();
    /*********************************************calculate***********************************************/

    num_threads = 8 * WARP_SIZE;
    num_blocks = rhs;
    sptrsv_dense_to_sparse<<<num_blocks, num_threads>>>(A_columnpointer_cuda, A_rowindex_cuda, X_VALUE, d_Spointer, rhs, d_x);
    cudaDeviceSynchronize();
}

__global__ void GESSM_Kernel_dense(int_t n,
                                   pangulu_inblock_ptr *L_columnpointer,
                                   pangulu_inblock_idx *L_rowindex,
                                   calculate_type *L_VALUE,
                                   pangulu_inblock_ptr *X_columnpointer,
                                   pangulu_inblock_idx *X_rowindex,
                                   calculate_type *X_VALUE,
                                   pangulu_inblock_ptr *A_columnpointer,
                                   pangulu_inblock_idx *A_rowindex,
                                   calculate_type *A_VALUE,
                                   calculate_type *dense)
{

    int_t colidx = blockIdx.x;
    int_t colX1 = X_columnpointer[colidx];
    int_t colX2 = X_columnpointer[colidx + 1];

    if (colidx >= n || colX2 == colX1)
        return;

    int_t colA1 = A_columnpointer[colidx];
    int_t colA2 = A_columnpointer[colidx + 1];

    for (int_t i = colX1, t = 0; i < colX2; i++, t++)
    {
        int_t rowX = X_rowindex[i];
        X_VALUE[i] = dense[colidx * n + rowX];
        calculate_type valx = X_VALUE[i];

        int_t colL1 = L_columnpointer[rowX];
        int_t colL2 = L_columnpointer[rowX + 1];

        for (int_t j = colL1 + 1 + threadIdx.x, p = threadIdx.x; j < colL2; j += blockDim.x, p += blockDim.x)
        {
            dense[colidx * n + L_rowindex[j]] -= valx * L_VALUE[j];
        }
        __syncthreads();
    }
}

__global__ void TSTRF_Kernel_dense(int_t n,
                                   pangulu_inblock_ptr *U_rowpointer,
                                   pangulu_inblock_idx *U_columnindex,
                                   calculate_type *U_VALUE,
                                   pangulu_inblock_ptr *X_rowpointer,
                                   pangulu_inblock_idx *X_columnindex,
                                   calculate_type *X_VALUE,
                                   pangulu_inblock_ptr *A_rowpointer,
                                   pangulu_inblock_idx *A_columnindex,
                                   calculate_type *A_VALUE,
                                   calculate_type *dense)
{
    int_t colidx = blockIdx.x;
    int_t colX1 = X_rowpointer[colidx];
    int_t colX2 = X_rowpointer[colidx + 1];

    if (colidx >= n || colX1 == colX2)
        return;

    int_t colA1 = A_rowpointer[colidx];
    int_t colA2 = A_rowpointer[colidx + 1];

    for (int_t i = colX1, t = 0; i < colX2; i++, t++)
    {
        int_t rowX = X_columnindex[i];
        int_t colU1 = U_rowpointer[rowX];
        int_t colU2 = U_rowpointer[rowX + 1];
        dense[colidx * n + rowX] /= U_VALUE[colU1];
        X_VALUE[i] = dense[colidx * n + rowX];

        calculate_type valx = X_VALUE[i];
        for (int_t j = colU1 + 1 + threadIdx.x, p = threadIdx.x; j < colU2; j += blockDim.x, p += blockDim.x)
        {
            dense[colidx * n + U_columnindex[j]] -= valx * U_VALUE[j];
        }
        __syncthreads();
    }
}

__global__ void syncfree_cuda_csr_dense_v11_L(
    int_t n,
    int_t rhs,
    int *degree,
    pangulu_inblock_ptr *L_columnpointer,
    pangulu_inblock_idx *L_rowindex,
    calculate_type *L_VALUE,
    pangulu_inblock_ptr *X_rowpointer,
    pangulu_inblock_idx *X_colindex,
    calculate_type *X_VALUE,
    calculate_type *A_VALUE,
    int *d_id_extractor,
    calculate_type *d_left_sum,
    calculate_type *dense)
{
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

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

    const int_t loc1 = X_rowpointer[global_x_id];
    const int_t loc2 = X_rowpointer[global_x_id + 1];
    const int_t start_ptr = L_columnpointer[global_x_id] + 1;
    const int_t stop_ptr = L_columnpointer[global_x_id + 1];

    calculate_type res = 0;

    for (int_t m = loc1 + lane_id; m < loc2; m += warpSize)
    {
        X_VALUE[m] = dense[global_x_id * n + X_colindex[m]];
    }
    if (loc1 != loc2)
        for (int_t jj = start_ptr + lane_id; jj < stop_ptr; jj += warpSize)
        {
            const pangulu_inblock_idx myidx = L_rowindex[jj];
            const int_t mypos = X_rowpointer[myidx];
            const int_t mypos2 = X_rowpointer[myidx + 1];
            const calculate_type L_id_value = L_VALUE[jj];

            for (int_t k = loc1; k < loc2; k++)
            {
                res = -(X_VALUE[k] * L_id_value);

                atomicAdd(&dense[myidx * n + X_colindex[k]], res);
            }
            atomicSub(&degree[myidx], 1);
        }
    else
    {
        for (int_t jj = start_ptr + lane_id; jj < stop_ptr; jj += warpSize)
        {
            atomicSub(&degree[L_rowindex[jj]], 1);
        }
    }
}

void pangulu_gessm_cuda_kernel_v11(int_t n,
                                   int_t nnzL,
                                   int_t nnzX,
                                   int *degree,
                                   int *d_id_extractor,
                                   calculate_type *d_left_sum,
                                   pangulu_inblock_ptr *L_columnpointer,
                                   pangulu_inblock_idx *L_rowindex,
                                   calculate_type *L_VALUE,
                                   pangulu_inblock_ptr *X_columnpointer,
                                   pangulu_inblock_idx *X_rowindex,
                                   calculate_type *X_VALUE,
                                   pangulu_inblock_ptr *A_columnpointer,
                                   pangulu_inblock_idx *A_rowindex,
                                   calculate_type *A_VALUE)
{
    int_t num_threads = WARP_SIZE * WARP_PER_BLOCK;
    int_t num_blocks;

    num_blocks = ceil((calculate_type)n / WARP_PER_BLOCK);
    cuda_Transform_s_to_d_col<<<num_blocks, num_threads>>>(n, num_blocks, A_columnpointer, A_rowindex, A_VALUE, CUDA_TEMP_value);
    cudaDeviceSynchronize();

    num_threads = WARP_PER_BLOCK_DGESSM * WARP_SIZE;
    num_blocks = ceil((calculate_type)n / (calculate_type)WARP_PER_BLOCK_DGESSM);
    syncfree_cuda_csr_dense_v11_L<<<num_blocks, num_threads>>>(n, n, degree, L_columnpointer, L_rowindex, L_VALUE,
                                                               A_columnpointer, A_rowindex, X_VALUE, A_VALUE, d_id_extractor, d_left_sum, CUDA_TEMP_value);
    cudaDeviceSynchronize();
}

__global__ void syncfree_cuda_csr_dense_v11_U(
    int_t n,
    int_t rhs,
    int *degree,
    pangulu_inblock_ptr *L_columnpointer,
    pangulu_inblock_idx *L_rowindex,
    calculate_type *L_VALUE,
    pangulu_inblock_ptr *X_rowpointer,
    pangulu_inblock_idx *X_colindex,
    calculate_type *X_VALUE,
    calculate_type *A_VALUE,
    int *d_id_extractor,
    calculate_type *d_left_sum,
    calculate_type *dense)
{
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
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

    const int_t loc1 = X_rowpointer[global_x_id];
    const int_t loc2 = X_rowpointer[global_x_id + 1];
    const int_t start_ptr = L_columnpointer[global_x_id] + 1;
    const int_t stop_ptr = L_columnpointer[global_x_id + 1];

    calculate_type res = 0;

    for (int_t m = loc1 + lane_id; m < loc2; m += warpSize)
    {
        dense[global_x_id * n + X_colindex[m]] /= L_VALUE[start_ptr - 1];

        X_VALUE[m] = dense[global_x_id * n + X_colindex[m]];
    }
    if (loc1 != loc2)
        for (int_t jj = start_ptr + lane_id; jj < stop_ptr; jj += warpSize)
        {
            const pangulu_inblock_idx myidx = L_rowindex[jj];
            const int_t mypos = X_rowpointer[myidx];
            const int_t mypos2 = X_rowpointer[myidx + 1];
            const calculate_type L_id_value = L_VALUE[jj];
            for (int_t k = loc1; k < loc2; k++)
            {
                res = -(X_VALUE[k] * L_id_value);

                atomicAdd(&dense[myidx * n + X_colindex[k]], res);
            }
            atomicSub(&degree[myidx], 1);
        }
    else
    {
        for (int_t jj = start_ptr + lane_id; jj < stop_ptr; jj += warpSize)
        {
            atomicSub(&degree[L_rowindex[jj]], 1);
        }
    }
}

void pangulu_cuda_malloc(void **cuda_address,
                         size_t size)
{
    GPU_MEMORY += size;
    if (cudaSuccess != cudaMalloc((cuda_address), size))
    {
        printf(PANGULU_E_CUDA_MALLOC);
        exit(0);
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

void pangulu_cuda_memcpy_host_to_device_int(pangulu_inblock_idx *cuda_address,
                                            pangulu_inblock_idx *cpu_address,
                                            size_t size)
{
    cudaMemcpy(cuda_address,
               cpu_address,
               size * sizeof(pangulu_inblock_idx),
               cudaMemcpyHostToDevice);
}

void pangulu_cuda_memcpy_host_to_device_int32(pangulu_inblock_idx *cuda_address,
                                              pangulu_inblock_idx *cpu_address,
                                              size_t size)
{
    cudaMemcpy(cuda_address,
               cpu_address,
               size * sizeof(pangulu_inblock_idx),
               cudaMemcpyHostToDevice);
}

void pangulu_cuda_memcpy_host_to_device_int32(int_32t *cuda_address,
                                              int_32t *cpu_address,
                                              size_t size)
{
    cudaMemcpy(cuda_address,
               cpu_address,
               size * sizeof(int_32t),
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

void pangulu_cuda_getDevicenum(int_32t *gpu_num)
{
    cudaGetDeviceCount(gpu_num);
}

int_32t pangulu_cuda_setDevice(int_32t gpu_num,
                               int_32t rank)
{
    int_32t usr_id = rank % gpu_num;
    cudaSetDevice(usr_id);
    return usr_id;
}

__global__ void WarpLevel_sflu(int_t n,
                               int_32t *d_nnzU,
                               pangulu_inblock_ptr *d_cscColPtrA,
                               pangulu_inblock_idx *d_cscRowIdxA,
                               calculate_type *d_cscValueA,
                               pangulu_inblock_ptr *d_cscColPtrL,
                               pangulu_inblock_idx *d_cscRowIdxL,
                               calculate_type *d_cscValueL,
                               pangulu_inblock_ptr *d_cscColPtrU,
                               pangulu_inblock_idx *d_cscRowIdxU,
                               calculate_type *d_cscValueU)
{
    int_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    int_t warp_local_id = threadIdx.x / 32;
    int_t lane_id = threadIdx.x % 32;

    if (warp_id >= n)
        return;

    const int_t colidx = warp_id;

    int_t colA = d_cscColPtrA[colidx];
    int_t colA1 = d_cscColPtrA[colidx + 1];
    int_t colU = d_cscColPtrU[colidx];
    int_t colU1 = d_cscColPtrU[colidx + 1];
    int_t colL = d_cscColPtrL[colidx];
    int_t colL1 = d_cscColPtrL[colidx + 1];
    int_t loadlen = colA1 - colA;

    // use shared memory
    if (loadlen <= SM_LEN_WARPLEV)
    {
        __shared__ pangulu_inblock_idx s_idxA[WARP_NUM_WARPLU * SM_LEN_WARPLEV];
        __shared__ calculate_type s_valA[WARP_NUM_WARPLU * SM_LEN_WARPLEV];

        pangulu_inblock_idx *s_idxA_local = &s_idxA[warp_local_id * SM_LEN_WARPLEV];
        calculate_type *s_valA_local = &s_valA[warp_local_id * SM_LEN_WARPLEV];

        for (int_t i = lane_id; i < loadlen; i += WARP_SIZE)
        {
            s_idxA_local[i] = d_cscRowIdxA[colA + i];
            s_valA_local[i] = d_cscValueA[colA + i];
        }

        // step one
        for (int_t i = colA; i < colA + colU1 - colU - 1; i++)
        {
            const int_t rowidx = d_cscRowIdxA[i];
            calculate_type value3 = s_valA_local[i - colA];
            // busy-wait until nnzU[rowidx]==0
            do
            {
                __threadfence();
            } while (d_nnzU[rowidx] != -1);

            int_t rowA = d_cscColPtrA[rowidx];
            int_t rowA1 = d_cscColPtrA[rowidx + 1];
            int_t rowU = d_cscColPtrU[rowidx];
            int_t rowU1 = d_cscColPtrU[rowidx + 1];

            for (int_t j = rowA + rowU1 - rowU + lane_id; j < rowA1; j += WARP_SIZE)
            {
                const int_t Lrowindex = d_cscRowIdxA[j];
                const int_t thecolidx = rowidx;
                int_t flag1 = binarySearch_idx(s_idxA_local, 0, loadlen - 1, Lrowindex);
                int_t flag2 = binarySearch_idx(d_cscRowIdxA, d_cscColPtrA[thecolidx] + d_cscColPtrU[thecolidx + 1] - d_cscColPtrU[thecolidx], d_cscColPtrA[thecolidx + 1] - 1, Lrowindex);

                s_valA_local[flag1] -= d_cscValueA[flag2] * value3;
            }

            if (!lane_id)
            {
                atomicSub(&d_nnzU[colidx], 1);
            }
        }

        // step two
        int_t flag5 = colU1 - colU - 1;
        if (s_valA_local[flag5] > ERROR || s_valA_local[flag5] < -ERROR)
        {
        }
        else
        {
            s_valA_local[flag5] = ERROR;
        }
        calculate_type value5 = s_valA_local[flag5];

        d_cscValueL[colL] = 1;
        for (int_t i = colA + colU1 - colU + lane_id; i < colA1; i += WARP_SIZE)
        {
            const int_t Lrowindex = s_idxA_local[i - colA];
            int_t flagL = colL + i - (colA + colU1 - colU) + 1;
            s_valA_local[i - colA] = s_valA_local[i - colA] / value5;
            d_cscValueL[flagL] = s_valA_local[i - colA];
        }

        for (int_t i = lane_id; i < colU1 - colU; i += WARP_SIZE)
        {
            d_cscValueU[colU + i] = s_valA_local[i];
        }

        for (int_t i = lane_id; i < loadlen; i += WARP_SIZE)
        {
            d_cscValueA[i + colA] = s_valA_local[i];
        }

        if (!lane_id)
        {
            atomicSub(&d_nnzU[colidx], 1);
        }
    }
    // do not use shared memory
    else
    {
        // step one
        for (int_t i = colA; i < colA + colU1 - colU - 1; i++)
        {
            const int_t rowidx = d_cscRowIdxA[i];
            calculate_type value3 = d_cscValueA[i];
            // busy-wait until nnzU[rowidx]==0
            do
            {
                __threadfence();
            } while (d_nnzU[rowidx] != -1);

            int_t rowA = d_cscColPtrA[rowidx];
            int_t rowA1 = d_cscColPtrA[rowidx + 1];
            int_t rowU = d_cscColPtrU[rowidx];
            int_t rowU1 = d_cscColPtrU[rowidx + 1];

            for (int_t j = rowA + rowU1 - rowU + lane_id; j < rowA1; j += WARP_SIZE)
            {
                const int_t Lrowindex = d_cscRowIdxA[j];
                const int_t thecolidx = rowidx;
                int_t flag1 = binarySearch_idx(d_cscRowIdxA, colA, colA1 - 1, Lrowindex);
                int_t flag2 = binarySearch_idx(d_cscRowIdxA, d_cscColPtrA[thecolidx] + d_cscColPtrU[thecolidx + 1] - d_cscColPtrU[thecolidx], d_cscColPtrA[thecolidx + 1] - 1, Lrowindex);

                d_cscValueA[flag1] -= d_cscValueA[flag2] * value3;
            }

            if (!lane_id)
            {
                atomicSub(&d_nnzU[colidx], 1);
            }
        }

        for (int_t i = lane_id; i < colU1 - colU; i += WARP_SIZE)
        {
            d_cscValueU[colU + i] = d_cscValueA[colA + i];
        }

        // step two
        int_t flag5 = colA + colU1 - colU - 1;
        if (d_cscValueA[flag5] > ERROR || d_cscValueA[flag5] < -ERROR)
        {
        }
        else
        {
            d_cscValueA[flag5] = ERROR;
        }

        calculate_type value5 = d_cscValueA[flag5];
        d_cscValueL[colL] = 1;
        for (int_t i = colA + colU1 - colU + lane_id; i < colA1; i += WARP_SIZE)
        {
            const int_t Lrowindex = d_cscRowIdxA[i];
            int_t flagL = colL + i - (colA + colU1 - colU) + 1;
            d_cscValueA[i] = d_cscValueA[i] / value5;
            d_cscValueL[flagL] = d_cscValueA[i];
        }

        if (!lane_id)
        {
            atomicSub(&d_nnzU[colidx], 1);
        }
    }
}

__global__ void BlockLevel_sflu_L1(int_t n,
                                   int_32t *d_nnzU,
                                   pangulu_inblock_ptr *d_cscColPtrA,
                                   pangulu_inblock_idx *d_cscRowIdxA,
                                   calculate_type *d_cscValueA,
                                   pangulu_inblock_ptr *d_cscColPtrL,
                                   pangulu_inblock_idx *d_cscRowIdxL,
                                   calculate_type *d_cscValueL,
                                   pangulu_inblock_ptr *d_cscColPtrU,
                                   pangulu_inblock_idx *d_cscRowIdxU,
                                   calculate_type *d_cscValueU)
{
    const int_t colidx = blockIdx.x;

    __shared__ int_32t s_nnzU[1];

    int_t colA = d_cscColPtrA[colidx];
    int_t colA1 = d_cscColPtrA[colidx + 1];
    int_t smemlen = 256;
    int_t len_a = colA1 - colA;

    // use shared memory
    if (len_a <= smemlen)
    {
        __shared__ pangulu_inblock_idx s_rowidxA[256];
        __shared__ calculate_type s_valueA[256];

        for (int_t i = threadIdx.x; i < len_a; i += blockDim.x)
        {
            s_rowidxA[i] = d_cscRowIdxA[i + colA];
            s_valueA[i] = d_cscValueA[i + colA];
        }

        if (!threadIdx.x)
        {
            s_nnzU[0] = d_nnzU[colidx];
        }
        __syncthreads();

        int_t colU = d_cscColPtrU[colidx];
        int_t colU1 = d_cscColPtrU[colidx + 1];
        int_t colL = d_cscColPtrL[colidx];
        int_t colL1 = d_cscColPtrL[colidx + 1];

        // step one
        for (int_t i = colA; i < colA + colU1 - colU - 1; i++)
        {
            const int_t rowidx = d_cscRowIdxA[i];
            calculate_type value3 = s_valueA[i - colA];

            // busy-wait until nnzU[rowidx]==0
            if (!threadIdx.x)
            {
                do
                {
                    __threadfence_block();
                } while (d_nnzU[rowidx] != -1);
            }
            __syncthreads();

            int_t rowA = d_cscColPtrA[rowidx];
            int_t rowA1 = d_cscColPtrA[rowidx + 1];
            int_t rowU = d_cscColPtrU[rowidx];
            int_t rowU1 = d_cscColPtrU[rowidx + 1];

            for (int_t j = rowA + rowU1 - rowU + threadIdx.x; j < rowA1; j += blockDim.x)
            {
                const int_t Lrowindex = d_cscRowIdxA[j];
                const int_t thecolidx = rowidx;

                int_t flag1 = binarySearch_idx(s_rowidxA, 0, len_a - 1, Lrowindex);
                int_t flag2 = binarySearch_idx(d_cscRowIdxA, d_cscColPtrA[thecolidx] + d_cscColPtrU[thecolidx + 1] - d_cscColPtrU[thecolidx], d_cscColPtrA[thecolidx + 1] - 1, Lrowindex);

                s_valueA[flag1] -= d_cscValueA[flag2] * value3;
            }

            if (!threadIdx.x)
            {
                atomicSub(&s_nnzU[0], 1);
            }
            __syncthreads();
        }

        // step two
        int_t flag5 = colU1 - colU - 1;
        if (s_valueA[flag5] > ERROR || s_valueA[flag5] < -ERROR)
        {
        }
        else
        {
            s_valueA[flag5] = ERROR;
        }
        calculate_type value5 = s_valueA[flag5];

        d_cscValueL[colL] = 1;
        for (int_t i = colA + colU1 - colU + threadIdx.x; i < colA1; i += blockDim.x)
        {
            const int_t Lrowindex = s_rowidxA[i - colA];
            int_t flagL = colL + i - (colA + colU1 - colU) + 1;
            s_valueA[i - colA] = s_valueA[i - colA] / value5;
            d_cscValueL[flagL] = s_valueA[i - colA];
        }
        __syncthreads();

        for (int_t i = threadIdx.x; i < colU1 - colU; i += blockDim.x)
        {
            d_cscValueU[colU + i] = s_valueA[i];
        }

        for (int_t i = threadIdx.x; i < len_a; i += blockDim.x)
        {
            d_cscValueA[i + colA] = s_valueA[i];
        }

        if (!threadIdx.x)
        {
            atomicSub(&s_nnzU[0], 1);
            d_nnzU[colidx] = s_nnzU[0];
        }
    }
    // do not use shared memory
    else
    {
        if (!threadIdx.x)
        {
            s_nnzU[0] = d_nnzU[colidx];
        }
        __syncthreads();

        int_t colU = d_cscColPtrU[colidx];
        int_t colU1 = d_cscColPtrU[colidx + 1];
        int_t colL = d_cscColPtrL[colidx];
        int_t colL1 = d_cscColPtrL[colidx + 1];

        // step one
        for (int_t i = colA; i < colA + colU1 - colU - 1; i++)
        {
            const int_t rowidx = d_cscRowIdxA[i];
            calculate_type value3 = d_cscValueA[i];
            // busy-wait until nnzU[rowidx]==0
            if (!threadIdx.x)
            {
                do
                {
                    __threadfence_block();
                } while (d_nnzU[rowidx] != -1);
            }
            __syncthreads();

            int_t rowA = d_cscColPtrA[rowidx];
            int_t rowA1 = d_cscColPtrA[rowidx + 1];
            int_t rowU = d_cscColPtrU[rowidx];
            int_t rowU1 = d_cscColPtrU[rowidx + 1];

            for (int_t j = rowA + rowU1 - rowU + threadIdx.x; j < rowA1; j += blockDim.x)
            {
                const int_t Lrowindex = d_cscRowIdxA[j];
                const int_t thecolidx = rowidx;
                int_t flag1 = binarySearch_idx(d_cscRowIdxA, colA, colA1 - 1, Lrowindex);
                int_t flag2 = binarySearch_idx(d_cscRowIdxA, d_cscColPtrA[thecolidx] + d_cscColPtrU[thecolidx + 1] - d_cscColPtrU[thecolidx], d_cscColPtrA[thecolidx + 1] - 1, Lrowindex);

                d_cscValueA[flag1] -= d_cscValueA[flag2] * value3;
            }

            if (!threadIdx.x)
            {
                atomicSub(&s_nnzU[0], 1);
            }
            __syncthreads();
        }

        for (int_t i = threadIdx.x; i < colU1 - colU; i += blockDim.x)
        {
            d_cscValueU[colU + i] = d_cscValueA[colA + i];
        }

        // step two
        int_t flag5 = colA + colU1 - colU - 1;
        if (d_cscValueA[flag5] > ERROR || d_cscValueA[flag5] < -ERROR)
        {
        }
        else
        {
            d_cscValueA[flag5] = ERROR;
        }
        calculate_type value5 = d_cscValueA[flag5];
        d_cscValueL[colL] = 1;

        for (int_t i = colA + colU1 - colU + threadIdx.x; i < colA1; i += blockDim.x)
        {
            const int_t Lrowindex = d_cscRowIdxA[i];
            int_t flagL = colL + i - (colA + colU1 - colU) + 1;
            d_cscValueA[i] = d_cscValueA[i] / value5;
            d_cscValueL[flagL] = d_cscValueA[i];
        }

        if (!threadIdx.x)
        {
            atomicSub(&s_nnzU[0], 1);
            d_nnzU[colidx] = s_nnzU[0];
        }
    }
}

__global__ void BlockLevel_sflu_L2(int_t n,
                                   int_32t *d_nnzU,
                                   pangulu_inblock_ptr *d_cscColPtrA,
                                   pangulu_inblock_idx *d_cscRowIdxA,
                                   calculate_type *d_cscValueA,
                                   pangulu_inblock_ptr *d_cscColPtrL,
                                   pangulu_inblock_idx *d_cscRowIdxL,
                                   calculate_type *d_cscValueL,
                                   pangulu_inblock_ptr *d_cscColPtrU,
                                   pangulu_inblock_idx *d_cscRowIdxU,
                                   calculate_type *d_cscValueU)
{
    const int_t colidx = blockIdx.x;

    __shared__ int_32t s_nnzU[1];

    int_t colA = d_cscColPtrA[colidx];
    int_t colA1 = d_cscColPtrA[colidx + 1];
    int_t smemlen = 512;
    int_t len_a = colA1 - colA;

    // use shared memory
    if (len_a <= smemlen)
    {
        __shared__ pangulu_inblock_idx s_rowidxA[512];
        __shared__ calculate_type s_valueA[512];

        for (int_t i = threadIdx.x; i < len_a; i += blockDim.x)
        {
            s_rowidxA[i] = d_cscRowIdxA[i + colA];
            s_valueA[i] = d_cscValueA[i + colA];
        }

        if (!threadIdx.x)
        {
            s_nnzU[0] = d_nnzU[colidx];
        }
        __syncthreads();

        int_t colU = d_cscColPtrU[colidx];
        int_t colU1 = d_cscColPtrU[colidx + 1];
        int_t colL = d_cscColPtrL[colidx];
        int_t colL1 = d_cscColPtrL[colidx + 1];

        // step one
        for (int_t i = colA; i < colA + colU1 - colU - 1; i++)
        {
            const int_t rowidx = d_cscRowIdxA[i];
            calculate_type value3 = s_valueA[i - colA];

            // busy-wait until nnzU[rowidx]==0
            if (!threadIdx.x)
            {
                do
                {
                    __threadfence_block();
                } while (d_nnzU[rowidx] != -1);
            }
            __syncthreads();

            int_t rowA = d_cscColPtrA[rowidx];
            int_t rowA1 = d_cscColPtrA[rowidx + 1];
            int_t rowU = d_cscColPtrU[rowidx];
            int_t rowU1 = d_cscColPtrU[rowidx + 1];

            for (int_t j = rowA + rowU1 - rowU + threadIdx.x; j < rowA1; j += blockDim.x)
            {
                const int_t Lrowindex = d_cscRowIdxA[j];
                const int_t thecolidx = rowidx;

                int_t flag1 = binarySearch_idx(s_rowidxA, 0, len_a - 1, Lrowindex);
                int_t flag2 = binarySearch_idx(d_cscRowIdxA, d_cscColPtrA[thecolidx] + d_cscColPtrU[thecolidx + 1] - d_cscColPtrU[thecolidx], d_cscColPtrA[thecolidx + 1] - 1, Lrowindex);

                s_valueA[flag1] -= d_cscValueA[flag2] * value3;
            }

            if (!threadIdx.x)
            {
                atomicSub(&s_nnzU[0], 1);
            }
            __syncthreads();
        }

        // step two
        int_t flag5 = colU1 - colU - 1;
        if (s_valueA[flag5] > ERROR || s_valueA[flag5] < -ERROR)
        {
        }
        else
        {
            s_valueA[flag5] = ERROR;
        }
        calculate_type value5 = s_valueA[flag5];

        d_cscValueL[colL] = 1;
        for (int_t i = colA + colU1 - colU + threadIdx.x; i < colA1; i += blockDim.x)
        {
            const int_t Lrowindex = s_rowidxA[i - colA];
            int_t flagL = colL + i - (colA + colU1 - colU) + 1;
            s_valueA[i - colA] = s_valueA[i - colA] / value5;
            d_cscValueL[flagL] = s_valueA[i - colA];
        }
        __syncthreads();

        for (int_t i = threadIdx.x; i < colU1 - colU; i += blockDim.x)
        {
            d_cscValueU[colU + i] = s_valueA[i];
        }

        for (int_t i = threadIdx.x; i < len_a; i += blockDim.x)
        {
            d_cscValueA[i + colA] = s_valueA[i];
        }

        if (!threadIdx.x)
        {
            atomicSub(&s_nnzU[0], 1);
            d_nnzU[colidx] = s_nnzU[0];
        }
    }
    // do not use shared memory
    else
    {
        if (!threadIdx.x)
        {
            s_nnzU[0] = d_nnzU[colidx];
        }
        __syncthreads();

        int_t colU = d_cscColPtrU[colidx];
        int_t colU1 = d_cscColPtrU[colidx + 1];
        int_t colL = d_cscColPtrL[colidx];
        int_t colL1 = d_cscColPtrL[colidx + 1];

        // step one
        for (int_t i = colA; i < colA + colU1 - colU - 1; i++)
        {
            const int_t rowidx = d_cscRowIdxA[i];

            calculate_type value3 = d_cscValueA[i];

            // busy-wait until nnzU[rowidx]==0
            if (!threadIdx.x)
            {
                do
                {
                    __threadfence_block();
                } while (d_nnzU[rowidx] != -1);
            }
            __syncthreads();

            int_t rowA = d_cscColPtrA[rowidx];
            int_t rowA1 = d_cscColPtrA[rowidx + 1];
            int_t rowU = d_cscColPtrU[rowidx];
            int_t rowU1 = d_cscColPtrU[rowidx + 1];

            for (int_t j = rowA + rowU1 - rowU + threadIdx.x; j < rowA1; j += blockDim.x)
            {
                const int_t Lrowindex = d_cscRowIdxA[j];
                const int_t thecolidx = rowidx;

                int_t flag1 = binarySearch_idx(d_cscRowIdxA, colA, colA1 - 1, Lrowindex);
                int_t flag2 = binarySearch_idx(d_cscRowIdxA, d_cscColPtrA[thecolidx] + d_cscColPtrU[thecolidx + 1] - d_cscColPtrU[thecolidx], d_cscColPtrA[thecolidx + 1] - 1, Lrowindex);

                d_cscValueA[flag1] -= d_cscValueA[flag2] * value3;
            }

            if (!threadIdx.x)
            {
                atomicSub(&s_nnzU[0], 1);
            }
            __syncthreads();
        }

        for (int_t i = threadIdx.x; i < colU1 - colU; i += blockDim.x)
        {
            d_cscValueU[colU + i] = d_cscValueA[colA + i];
        }

        // step two
        int_t flag5 = colA + colU1 - colU - 1;
        if (d_cscValueA[flag5] > ERROR || d_cscValueA[flag5] < -ERROR)
        {
        }
        else
        {
            d_cscValueA[flag5] = ERROR;
        }
        calculate_type value5 = d_cscValueA[flag5];

        d_cscValueL[colL] = 1;
        for (int_t i = colA + colU1 - colU + threadIdx.x; i < colA1; i += blockDim.x)
        {
            const int_t Lrowindex = d_cscRowIdxA[i];
            int_t flagL = colL + i - (colA + colU1 - colU) + 1;
            d_cscValueA[i] = d_cscValueA[i] / value5;
            d_cscValueL[flagL] = d_cscValueA[i];
        }

        if (!threadIdx.x)
        {
            atomicSub(&s_nnzU[0], 1);
            d_nnzU[colidx] = s_nnzU[0];
        }
    }
}

__global__ void BlockLevel_sflu_L3(int_t n,
                                   int_32t *d_nnzU,
                                   pangulu_inblock_ptr *d_cscColPtrA,
                                   pangulu_inblock_idx *d_cscRowIdxA,
                                   calculate_type *d_cscValueA,
                                   pangulu_inblock_ptr *d_cscColPtrL,
                                   pangulu_inblock_idx *d_cscRowIdxL,
                                   calculate_type *d_cscValueL,
                                   pangulu_inblock_ptr *d_cscColPtrU,
                                   pangulu_inblock_idx *d_cscRowIdxU,
                                   calculate_type *d_cscValueU)
{
    const int_t colidx = blockIdx.x;

    __shared__ int_32t s_nnzU[1];

    int_t colA = d_cscColPtrA[colidx];
    int_t colA1 = d_cscColPtrA[colidx + 1];
    int_t smemlen = 1024;
    int_t len_a = colA1 - colA;

    // use shared memory
    if (len_a <= smemlen)
    {
        __shared__ pangulu_inblock_idx s_rowidxA[1024];
        __shared__ calculate_type s_valueA[1024];

        for (int_t i = threadIdx.x; i < len_a; i += blockDim.x)
        {
            s_rowidxA[i] = d_cscRowIdxA[i + colA];
            s_valueA[i] = d_cscValueA[i + colA];
        }

        if (!threadIdx.x)
        {
            s_nnzU[0] = d_nnzU[colidx];
        }
        __syncthreads();

        int_t colU = d_cscColPtrU[colidx];
        int_t colU1 = d_cscColPtrU[colidx + 1];
        int_t colL = d_cscColPtrL[colidx];
        int_t colL1 = d_cscColPtrL[colidx + 1];

        // step one
        for (int_t i = colA; i < colA + colU1 - colU - 1; i++)
        {
            const int_t rowidx = d_cscRowIdxA[i];
            calculate_type value3 = s_valueA[i - colA];

            // busy-wait until nnzU[rowidx]==0
            if (!threadIdx.x)
            {
                do
                {
                    __threadfence_block();
                } while (d_nnzU[rowidx] != -1);
            }
            __syncthreads();

            int_t rowA = d_cscColPtrA[rowidx];
            int_t rowA1 = d_cscColPtrA[rowidx + 1];
            int_t rowU = d_cscColPtrU[rowidx];
            int_t rowU1 = d_cscColPtrU[rowidx + 1];

            for (int_t j = rowA + rowU1 - rowU + threadIdx.x; j < rowA1; j += blockDim.x)
            {
                const int_t Lrowindex = d_cscRowIdxA[j];
                const int_t thecolidx = rowidx;

                int_t flag1 = binarySearch_idx(s_rowidxA, 0, len_a - 1, Lrowindex);
                int_t flag2 = binarySearch_idx(d_cscRowIdxA, d_cscColPtrA[thecolidx] + d_cscColPtrU[thecolidx + 1] - d_cscColPtrU[thecolidx], d_cscColPtrA[thecolidx + 1] - 1, Lrowindex);

                s_valueA[flag1] -= d_cscValueA[flag2] * value3;
            }

            if (!threadIdx.x)
            {
                atomicSub(&s_nnzU[0], 1);
            }
            __syncthreads();
        }

        // step two
        int_t flag5 = colU1 - colU - 1;
        if (s_valueA[flag5] > ERROR || s_valueA[flag5] < -ERROR)
        {
        }
        else
        {
            s_valueA[flag5] = ERROR;
        }
        calculate_type value5 = s_valueA[flag5];

        d_cscValueL[colL] = 1;
        for (int_t i = colA + colU1 - colU + threadIdx.x; i < colA1; i += blockDim.x)
        {
            const int_t Lrowindex = s_rowidxA[i - colA];
            int_t flagL = colL + i - (colA + colU1 - colU) + 1;
            s_valueA[i - colA] = s_valueA[i - colA] / value5;
            d_cscValueL[flagL] = s_valueA[i - colA];
        }
        __syncthreads();

        for (int_t i = threadIdx.x; i < colU1 - colU; i += blockDim.x)
        {
            d_cscValueU[colU + i] = s_valueA[i];
        }

        for (int_t i = threadIdx.x; i < len_a; i += blockDim.x)
        {
            d_cscValueA[i + colA] = s_valueA[i];
        }

        if (!threadIdx.x)
        {
            atomicSub(&s_nnzU[0], 1);
            d_nnzU[colidx] = s_nnzU[0];
        }
    }
    // do not use shared memory
    else
    {
        if (!threadIdx.x)
        {
            s_nnzU[0] = d_nnzU[colidx];
        }
        __syncthreads();

        int_t colU = d_cscColPtrU[colidx];
        int_t colU1 = d_cscColPtrU[colidx + 1];
        int_t colL = d_cscColPtrL[colidx];
        int_t colL1 = d_cscColPtrL[colidx + 1];

        // step one
        for (int_t i = colA; i < colA + colU1 - colU - 1; i++)
        {
            const int_t rowidx = d_cscRowIdxA[i];

            calculate_type value3 = d_cscValueA[i];

            // busy-wait until nnzU[rowidx]==0
            if (!threadIdx.x)
            {
                do
                {
                    __threadfence_block();
                } while (d_nnzU[rowidx] != -1);
            }
            __syncthreads();

            int_t rowA = d_cscColPtrA[rowidx];
            int_t rowA1 = d_cscColPtrA[rowidx + 1];
            int_t rowU = d_cscColPtrU[rowidx];
            int_t rowU1 = d_cscColPtrU[rowidx + 1];

            for (int_t j = rowA + rowU1 - rowU + threadIdx.x; j < rowA1; j += blockDim.x)
            {
                const int_t Lrowindex = d_cscRowIdxA[j];
                const int_t thecolidx = rowidx;

                int_t flag1 = binarySearch_idx(d_cscRowIdxA, colA, colA1 - 1, Lrowindex);
                int_t flag2 = binarySearch_idx(d_cscRowIdxA, d_cscColPtrA[thecolidx] + d_cscColPtrU[thecolidx + 1] - d_cscColPtrU[thecolidx], d_cscColPtrA[thecolidx + 1] - 1, Lrowindex);

                d_cscValueA[flag1] -= d_cscValueA[flag2] * value3;
            }

            if (!threadIdx.x)
            {
                atomicSub(&s_nnzU[0], 1);
            }
            __syncthreads();
        }

        for (int_t i = threadIdx.x; i < colU1 - colU; i += blockDim.x)
        {
            d_cscValueU[colU + i] = d_cscValueA[colA + i];
        }

        // step two
        int_t flag5 = colA + colU1 - colU - 1;
        if (d_cscValueA[flag5] > ERROR || d_cscValueA[flag5] < -ERROR)
        {
        }
        else
        {
            d_cscValueA[flag5] = ERROR;
        }
        calculate_type value5 = d_cscValueA[flag5];

        d_cscValueL[colL] = 1;
        for (int_t i = colA + colU1 - colU + threadIdx.x; i < colA1; i += blockDim.x)
        {
            const int_t Lrowindex = d_cscRowIdxA[i];
            int_t flagL = colL + i - (colA + colU1 - colU) + 1;
            d_cscValueA[i] = d_cscValueA[i] / value5;
            d_cscValueL[flagL] = d_cscValueA[i];
        }

        if (!threadIdx.x)
        {
            atomicSub(&s_nnzU[0], 1);
            d_nnzU[colidx] = s_nnzU[0];
        }
    }
}

__global__ void BlockLevel_sflu_L4(int_t n,
                                   int_32t *d_nnzU,
                                   pangulu_inblock_ptr *d_cscColPtrA,
                                   pangulu_inblock_idx *d_cscRowIdxA,
                                   calculate_type *d_cscValueA,
                                   pangulu_inblock_ptr *d_cscColPtrL,
                                   pangulu_inblock_idx *d_cscRowIdxL,
                                   calculate_type *d_cscValueL,
                                   pangulu_inblock_ptr *d_cscColPtrU,
                                   pangulu_inblock_idx *d_cscRowIdxU,
                                   calculate_type *d_cscValueU)
{
    const int_t colidx = blockIdx.x;

    __shared__ int_32t s_nnzU[1];

    int_t colA = d_cscColPtrA[colidx];
    int_t colA1 = d_cscColPtrA[colidx + 1];
    int_t smemlen = 2048;
    int_t len_a = colA1 - colA;

    // use shared memory
    if (len_a <= smemlen)
    {
        __shared__ pangulu_inblock_idx s_rowidxA[2048];
        __shared__ calculate_type s_valueA[2048];

        for (int_t i = threadIdx.x; i < len_a; i += blockDim.x)
        {
            s_rowidxA[i] = d_cscRowIdxA[i + colA];
            s_valueA[i] = d_cscValueA[i + colA];
        }

        if (!threadIdx.x)
        {
            s_nnzU[0] = d_nnzU[colidx];
        }
        __syncthreads();

        int_t colU = d_cscColPtrU[colidx];
        int_t colU1 = d_cscColPtrU[colidx + 1];
        int_t colL = d_cscColPtrL[colidx];
        int_t colL1 = d_cscColPtrL[colidx + 1];

        // step one
        for (int_t i = colA; i < colA + colU1 - colU - 1; i++)
        {
            const int_t rowidx = d_cscRowIdxA[i];
            calculate_type value3 = s_valueA[i - colA];

            // busy-wait until nnzU[rowidx]==0
            if (!threadIdx.x)
            {
                do
                {
                    __threadfence_block();
                } while (d_nnzU[rowidx] != -1);
            }
            __syncthreads();

            int_t rowA = d_cscColPtrA[rowidx];
            int_t rowA1 = d_cscColPtrA[rowidx + 1];
            int_t rowU = d_cscColPtrU[rowidx];
            int_t rowU1 = d_cscColPtrU[rowidx + 1];

            for (int_t j = rowA + rowU1 - rowU + threadIdx.x; j < rowA1; j += blockDim.x)
            {
                const int_t Lrowindex = d_cscRowIdxA[j];
                const int_t thecolidx = rowidx;

                int_t flag1 = binarySearch_idx(s_rowidxA, 0, len_a - 1, Lrowindex);
                int_t flag2 = binarySearch_idx(d_cscRowIdxA, d_cscColPtrA[thecolidx] + d_cscColPtrU[thecolidx + 1] - d_cscColPtrU[thecolidx], d_cscColPtrA[thecolidx + 1] - 1, Lrowindex);

                s_valueA[flag1] -= d_cscValueA[flag2] * value3;
            }

            if (!threadIdx.x)
            {
                atomicSub(&s_nnzU[0], 1);
            }
            __syncthreads();
        }

        // step two
        int_t flag5 = colU1 - colU - 1;
        if (s_valueA[flag5] > ERROR || s_valueA[flag5] < -ERROR)
        {
        }
        else
        {
            s_valueA[flag5] = ERROR;
        }
        calculate_type value5 = s_valueA[flag5];

        d_cscValueL[colL] = 1;
        for (int_t i = colA + colU1 - colU + threadIdx.x; i < colA1; i += blockDim.x)
        {
            const int_t Lrowindex = s_rowidxA[i - colA];
            int_t flagL = colL + i - (colA + colU1 - colU) + 1;
            s_valueA[i - colA] = s_valueA[i - colA] / value5;
            d_cscValueL[flagL] = s_valueA[i - colA];
        }
        __syncthreads();

        for (int_t i = threadIdx.x; i < colU1 - colU; i += blockDim.x)
        {
            d_cscValueU[colU + i] = s_valueA[i];
        }

        for (int_t i = threadIdx.x; i < len_a; i += blockDim.x)
        {
            d_cscValueA[i + colA] = s_valueA[i];
        }

        if (!threadIdx.x)
        {
            atomicSub(&s_nnzU[0], 1);
            d_nnzU[colidx] = s_nnzU[0];
        }
    }
    // do not use shared memory
    else
    {
        if (!threadIdx.x)
        {
            s_nnzU[0] = d_nnzU[colidx];
        }
        __syncthreads();

        int_t colU = d_cscColPtrU[colidx];
        int_t colU1 = d_cscColPtrU[colidx + 1];
        int_t colL = d_cscColPtrL[colidx];
        int_t colL1 = d_cscColPtrL[colidx + 1];

        // step one
        for (int_t i = colA; i < colA + colU1 - colU - 1; i++)
        {
            const int_t rowidx = d_cscRowIdxA[i];

            calculate_type value3 = d_cscValueA[i];

            // busy-wait until nnzU[rowidx]==0
            if (!threadIdx.x)
            {
                do
                {
                    __threadfence_block();
                } while (d_nnzU[rowidx] != -1);
            }
            __syncthreads();

            int_t rowA = d_cscColPtrA[rowidx];
            int_t rowA1 = d_cscColPtrA[rowidx + 1];
            int_t rowU = d_cscColPtrU[rowidx];
            int_t rowU1 = d_cscColPtrU[rowidx + 1];

            for (int_t j = rowA + rowU1 - rowU + threadIdx.x; j < rowA1; j += blockDim.x)
            {
                const int_t Lrowindex = d_cscRowIdxA[j];
                const int_t thecolidx = rowidx;

                int_t flag1 = binarySearch_idx(d_cscRowIdxA, colA, colA1 - 1, Lrowindex);
                int_t flag2 = binarySearch_idx(d_cscRowIdxA, d_cscColPtrA[thecolidx] + d_cscColPtrU[thecolidx + 1] - d_cscColPtrU[thecolidx], d_cscColPtrA[thecolidx + 1] - 1, Lrowindex);

                d_cscValueA[flag1] -= d_cscValueA[flag2] * value3;
            }

            if (!threadIdx.x)
            {
                atomicSub(&s_nnzU[0], 1);
            }
            __syncthreads();
        }

        for (int_t i = threadIdx.x; i < colU1 - colU; i += blockDim.x)
        {
            d_cscValueU[colU + i] = d_cscValueA[colA + i];
        }

        // step two
        int_t flag5 = colA + colU1 - colU - 1;
        if (d_cscValueA[flag5] > ERROR || d_cscValueA[flag5] < -ERROR)
        {
        }
        else
        {
            d_cscValueA[flag5] = ERROR;
        }
        calculate_type value5 = d_cscValueA[flag5];

        d_cscValueL[colL] = 1;
        for (int_t i = colA + colU1 - colU + threadIdx.x; i < colA1; i += blockDim.x)
        {
            const int_t Lrowindex = d_cscRowIdxA[i];
            int_t flagL = colL + i - (colA + colU1 - colU) + 1;
            d_cscValueA[i] = d_cscValueA[i] / value5;
            d_cscValueL[flagL] = d_cscValueA[i];
        }

        if (!threadIdx.x)
        {
            atomicSub(&s_nnzU[0], 1);
            d_nnzU[colidx] = s_nnzU[0];
        }
    }
}

__device__ int_t binarySearch(int_t *ridx, int_t left, int_t right, int_t target)
{
    int_t mid;
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

__device__ pangulu_inblock_idx binarySearch_idx(pangulu_inblock_idx *ridx, int_t left, int_t right, pangulu_inblock_idx target)
{
    int_t mid;
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
__global__ void cuda_Transform_s_to_d_col(int_t n,
                                          int stride,
                                          pangulu_inblock_ptr *d_rowPtrA,
                                          pangulu_inblock_idx *d_colIdxA,
                                          calculate_type *d_valueA,
                                          calculate_type *temp_value_A)
{
    int_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    int_t lane_id = threadIdx.x % 32;
    int_t warp_local_id = threadIdx.x / 32;

    if (warp_id >= n)
    {
        return;
    }

    calculate_type *value = temp_value_A + warp_id * n;

    int colA1 = d_rowPtrA[warp_id];
    int colA2 = d_rowPtrA[warp_id + 1];

    for (int i = colA1 + lane_id; i < colA2; i += WARP_SIZE)
    {
        value[d_colIdxA[i]] = d_valueA[i];
    }
}

__global__ void cuda_Transform_d_to_s_lu_col(int_t n,
                                             pangulu_inblock_ptr *d_rowPtrA,
                                             pangulu_inblock_idx *d_colIdxA,
                                             pangulu_inblock_ptr *d_rowPtrL,
                                             pangulu_inblock_idx *d_colIdxL,
                                             calculate_type *d_valueL,
                                             pangulu_inblock_ptr *d_rowPtrU,
                                             pangulu_inblock_idx *d_colIdxU,
                                             calculate_type *d_valueU,
                                             calculate_type *temp_value_A)
{
    int_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    int_t lane_id = threadIdx.x % 32;
    int_t warp_local_id = threadIdx.x / 32;

    if (warp_id >= n)
    {
        return;
    }
    calculate_type *value = temp_value_A + warp_id * n;

    int_t colL1 = d_rowPtrL[warp_id];
    int_t colL2 = d_rowPtrL[warp_id + 1];

    int_t colU1 = d_rowPtrU[warp_id];
    int_t colU2 = d_rowPtrU[warp_id + 1];

    for (int i = colL1 + 1 + lane_id; i < colL2; i += WARP_SIZE)
    {
        d_valueL[i] = value[d_colIdxL[i]];
    }
    d_valueL[colL1] = 1;

    for (int i = colU1 + lane_id; i < colU2; i += WARP_SIZE)
    {
        d_valueU[i] = value[d_colIdxU[i]];
    }
}
__global__ void lunumeric_cuda_kernel_v2(int_t n,
                                         int_32t *d_nnzU,
                                         calculate_type *d_dense_tag_double,
                                         pangulu_inblock_ptr *d_cscColPtrL_upperbound,
                                         pangulu_inblock_idx *d_cscRowIdxL_upperbound,
                                         pangulu_inblock_ptr *d_cscColPtrU_upperbound,
                                         pangulu_inblock_idx *d_cscRowIdxU_upperbound)
{

    const int_t colidx = blockIdx.x;
    __shared__ int_32t s_nnzU[1];

    if (!threadIdx.x)
    {
        s_nnzU[0] = d_nnzU[colidx];
    }
    __syncthreads();

    const int_t baseU_colidx = d_cscColPtrU_upperbound[colidx];
    const int_t baseU_colidx1 = d_cscColPtrU_upperbound[colidx + 1];
    const int_t baseL_colidx = d_cscColPtrL_upperbound[colidx];
    const int_t baseL_colidx1 = d_cscColPtrL_upperbound[colidx + 1];

    // step one
    for (int_t j = baseU_colidx; j < baseU_colidx1 - 1; j++)
    {
        const pangulu_inblock_idx rowidx = d_cscRowIdxU_upperbound[j];
        // busy-wait until nnzU[rowidx] == 0
        if (!threadIdx.x)
        {
            do
            {
                __threadfence_block();
            } while (d_nnzU[rowidx] != -1);
        }
        __syncthreads();

        for (int i = d_cscColPtrL_upperbound[rowidx] + 1 + threadIdx.x; i < d_cscColPtrL_upperbound[rowidx + 1]; i += blockDim.x)
        {
            const int Lrowindex = d_cscRowIdxL_upperbound[i];
            const int thecolidx = rowidx;
            d_dense_tag_double[colidx * n + Lrowindex] -= d_dense_tag_double[thecolidx * n + Lrowindex] * d_dense_tag_double[colidx * n + rowidx];
        }
        if (!threadIdx.x)
        {
            atomicSub(&s_nnzU[0], 1);
        }
        __syncthreads();
    }

    //  step two
    for (int i = baseL_colidx + threadIdx.x + 1; i < d_cscColPtrL_upperbound[colidx + 1]; i += blockDim.x)
    {
        const int Lrowindex = d_cscRowIdxL_upperbound[i];
        d_dense_tag_double[colidx * n + Lrowindex] = d_dense_tag_double[colidx * n + Lrowindex] / d_dense_tag_double[colidx * n + colidx];
    }

    if (!threadIdx.x)
    {
        atomicSub(&s_nnzU[0], 1);
        d_nnzU[colidx] = s_nnzU[0];
    }
}

__global__ void trans_cuda_CSC_to_CSR(int_t nnz, calculate_type *d_val_csr, pangulu_inblock_idx *d_idx, calculate_type *d_val_csc)
{

    if (blockDim.x * blockIdx.x + threadIdx.x >= nnz)
        return;

    int_t i = blockDim.x * blockIdx.x + threadIdx.x;
    d_val_csr[d_idx[i]] = d_val_csc[i];
}

__global__ void trans_cuda_CSR_to_CSC(int_t nnz, calculate_type *d_val_csc, pangulu_inblock_idx *d_idx, calculate_type *d_val_csr)
{

    if (blockDim.x * blockIdx.x + threadIdx.x >= nnz)
        return;

    int_t i = blockDim.x * blockIdx.x + threadIdx.x;
    d_val_csc[i] = d_val_csr[d_idx[i]];
}

__global__ void vector_add_cuda(int_t nnz, calculate_type *d_now_val_csc, calculate_type *d_old_val_csc)
{

    if (blockDim.x * blockIdx.x + threadIdx.x >= nnz)
        return;

    int_t i = blockDim.x * blockIdx.x + threadIdx.x;
    d_now_val_csc[i] += d_old_val_csc[i];
}

void pangulu_cuda_vector_add_kernel(int_t nnz, calculate_type *d_now_val_csc, calculate_type *d_old_val_csc)
{

    int_t num_threads = WARP_SIZE * 2;
    int_t num_blocks = ceil((double)nnz / num_threads);
    cudaDeviceSynchronize();
    vector_add_cuda<<<num_blocks, num_threads>>>(nnz, d_now_val_csc, d_old_val_csc);
    cudaDeviceSynchronize();
}

__global__ void WarpLevel_spgemm_32(int_t n,
                                    int_t layer,
                                    pangulu_inblock_ptr *d_bin_rowpointer,
                                    pangulu_inblock_idx *d_bin_rowindex,
                                    pangulu_inblock_ptr *d_rowPtrA,
                                    pangulu_inblock_idx *d_colIdxA,
                                    calculate_type *d_valueA,
                                    pangulu_inblock_ptr *d_rowPtrB,
                                    pangulu_inblock_idx *d_colIdxB,
                                    calculate_type *d_valueB,
                                    pangulu_inblock_ptr *d_rowPtrC,
                                    pangulu_inblock_idx *d_colIdxC,
                                    calculate_type *d_valueC)
{
    int_t warp_global_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    int_t warp_local_id = threadIdx.x / 32;
    int_t lane_id = threadIdx.x % 32;

    if (warp_global_id >= d_bin_rowpointer[layer + 1] - d_bin_rowpointer[layer])
        return;

    const int_t rowidx = d_bin_rowindex[d_bin_rowpointer[layer] + warp_global_id];

    int_t therowc = d_rowPtrC[rowidx];
    int_t nextrowc = d_rowPtrC[rowidx + 1];

    int_t therowa = d_rowPtrA[rowidx];
    int_t nextrowa = d_rowPtrA[rowidx + 1];

    int_t SM_LEN = 32;

    __shared__ pangulu_inblock_idx s_idxC[32 * WARP_PER_BLOCK_GEMM];
    __shared__ calculate_type s_valC[32 * WARP_PER_BLOCK_GEMM];

    pangulu_inblock_idx *s_idxC_local = &s_idxC[warp_local_id * SM_LEN];
    calculate_type *s_valC_local = &s_valC[warp_local_id * SM_LEN];

    if (lane_id < nextrowc - therowc)
    {
        s_idxC_local[lane_id] = d_colIdxC[therowc + lane_id];
        s_valC_local[lane_id] = 0;
    }

    for (int_t i = therowa + lane_id; i < nextrowa; i += WARP_SIZE)
    {
        int_t colA = d_colIdxA[i];
        calculate_type valA = d_valueA[i];

        int_t therowb = d_rowPtrB[colA];
        int_t nextrowb = d_rowPtrB[colA + 1];

        for (int_t j = therowb; j < nextrowb; j++)
        {
            int_t colB = d_colIdxB[j];
            int_t flag = binarySearch_idx(s_idxC_local, 0, nextrowc - therowc - 1, colB);
            atomicAdd(&s_valC_local[flag], -valA * d_valueB[j]);
        }
    }

    if (lane_id < nextrowc - therowc)
    {
        d_valueC[therowc + lane_id] += s_valC_local[lane_id];
    }
}

__global__ void WarpLevel_spgemm_64(int_t n,
                                    int_t layer,
                                    pangulu_inblock_ptr *d_bin_rowpointer,
                                    pangulu_inblock_idx *d_bin_rowindex,
                                    pangulu_inblock_ptr *d_rowPtrA,
                                    pangulu_inblock_idx *d_colIdxA,
                                    calculate_type *d_valueA,
                                    pangulu_inblock_ptr *d_rowPtrB,
                                    pangulu_inblock_idx *d_colIdxB,
                                    calculate_type *d_valueB,
                                    pangulu_inblock_ptr *d_rowPtrC,
                                    pangulu_inblock_idx *d_colIdxC,
                                    calculate_type *d_valueC)
{
    int_t warp_global_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    int_t warp_local_id = threadIdx.x / 32;
    int_t lane_id = threadIdx.x % 32;

    if (warp_global_id >= d_bin_rowpointer[layer + 1] - d_bin_rowpointer[layer])
        return;

    const int_t rowidx = d_bin_rowindex[d_bin_rowpointer[layer] + warp_global_id];

    int_t therowc = d_rowPtrC[rowidx];
    int_t nextrowc = d_rowPtrC[rowidx + 1];

    int_t therowa = d_rowPtrA[rowidx];
    int_t nextrowa = d_rowPtrA[rowidx + 1];

    int_t SM_LEN = 64;

    __shared__ pangulu_inblock_idx s_idxC[64 * WARP_PER_BLOCK_GEMM];
    __shared__ calculate_type s_valC[64 * WARP_PER_BLOCK_GEMM];

    pangulu_inblock_idx *s_idxC_local = &s_idxC[warp_local_id * SM_LEN];
    calculate_type *s_valC_local = &s_valC[warp_local_id * SM_LEN];

    for (int_t i = lane_id; i < nextrowc - therowc; i += WARP_SIZE)
    {
        s_idxC_local[i] = d_colIdxC[therowc + i];
        s_valC_local[i] = 0;
    }
    __syncthreads();

    for (int_t i = therowa + lane_id; i < nextrowa; i += WARP_SIZE)
    {
        int_t colA = d_colIdxA[i];
        calculate_type valA = d_valueA[i];

        int_t therowb = d_rowPtrB[colA];
        int_t nextrowb = d_rowPtrB[colA + 1];

        for (int_t j = therowb; j < nextrowb; j++)
        {
            int_t colB = d_colIdxB[j];
            int_t flag = binarySearch_idx(s_idxC_local, 0, nextrowc - therowc - 1, colB);
            atomicAdd(&s_valC_local[flag], -valA * d_valueB[j]);
        }
    }
    for (int_t i = lane_id; i < nextrowc - therowc; i += WARP_SIZE)
    {
        d_valueC[therowc + i] += s_valC_local[i];
    }
    __syncthreads();
}

__global__ void WarpLevel_spgemm_128(int_t n,
                                     int_t layer,
                                     pangulu_inblock_ptr *d_bin_rowpointer,
                                     pangulu_inblock_idx *d_bin_rowindex,
                                     pangulu_inblock_ptr *d_rowPtrA,
                                     pangulu_inblock_idx *d_colIdxA,
                                     calculate_type *d_valueA,
                                     pangulu_inblock_ptr *d_rowPtrB,
                                     pangulu_inblock_idx *d_colIdxB,
                                     calculate_type *d_valueB,
                                     pangulu_inblock_ptr *d_rowPtrC,
                                     pangulu_inblock_idx *d_colIdxC,
                                     calculate_type *d_valueC)
{
    int_t warp_global_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    int_t warp_local_id = threadIdx.x / 32;
    int_t lane_id = threadIdx.x % 32;

    if (warp_global_id >= d_bin_rowpointer[layer + 1] - d_bin_rowpointer[layer])
        return;

    const int_t rowidx = d_bin_rowindex[d_bin_rowpointer[layer] + warp_global_id];

    int_t therowc = d_rowPtrC[rowidx];
    int_t nextrowc = d_rowPtrC[rowidx + 1];

    int_t therowa = d_rowPtrA[rowidx];
    int_t nextrowa = d_rowPtrA[rowidx + 1];

    int_t SM_LEN = 128;

    __shared__ pangulu_inblock_idx s_idxC[128 * WARP_PER_BLOCK_GEMM];
    __shared__ calculate_type s_valC[128 * WARP_PER_BLOCK_GEMM];

    pangulu_inblock_idx *s_idxC_local = &s_idxC[warp_local_id * SM_LEN];
    calculate_type *s_valC_local = &s_valC[warp_local_id * SM_LEN];

    for (int_t i = lane_id; i < nextrowc - therowc; i += WARP_SIZE)
    {
        s_idxC_local[i] = d_colIdxC[therowc + i];
        s_valC_local[i] = 0;
    }
    __syncthreads();

    for (int_t i = therowa; i < nextrowa; i++)
    {
        int_t colA = d_colIdxA[i];
        calculate_type valA = d_valueA[i];

        int_t therowb = d_rowPtrB[colA];
        int_t nextrowb = d_rowPtrB[colA + 1];

        for (int_t j = therowb + lane_id; j < nextrowb; j += WARP_SIZE)
        {
            int_t colB = d_colIdxB[j];
            int_t flag = binarySearch_idx(s_idxC_local, 0, nextrowc - therowc - 1, colB);
            s_valC_local[flag] -= valA * d_valueB[j];
        }
    }

    for (int_t i = lane_id; i < nextrowc - therowc; i += WARP_SIZE)
    {
        d_valueC[therowc + i] += s_valC_local[i];
    }
    __syncthreads();
}

__global__ void ThreadLevel_spgemm(int_t n,
                                   int_t layer,
                                   pangulu_inblock_ptr *d_bin_rowpointer,
                                   pangulu_inblock_idx *d_bin_rowindex,
                                   pangulu_inblock_ptr *d_rowPtrA,
                                   pangulu_inblock_idx *d_colIdxA,
                                   calculate_type *d_valueA,
                                   pangulu_inblock_ptr *d_rowPtrB,
                                   pangulu_inblock_idx *d_colIdxB,
                                   calculate_type *d_valueB,
                                   pangulu_inblock_ptr *d_rowPtrC,
                                   pangulu_inblock_idx *d_colIdxC,
                                   calculate_type *d_valueC)
{
    const int_t rowidx = d_bin_rowindex[d_bin_rowpointer[layer] + blockDim.x * blockIdx.x + threadIdx.x];

    if (d_bin_rowpointer[layer] + blockDim.x * blockIdx.x + threadIdx.x >= d_bin_rowpointer[layer + 1])
        return;

    int_t therowa = d_rowPtrA[rowidx];
    int_t nextrowa = d_rowPtrA[rowidx + 1];

    int_t therowc = d_rowPtrC[rowidx];

    for (int_t i = therowa; i < nextrowa; i++)
    {
        int_t colA = d_colIdxA[i];
        calculate_type valA = d_valueA[i];

        int_t therowb = d_rowPtrB[colA];
        int_t nextrowb = d_rowPtrB[colA + 1];

        if (nextrowb - therowb != 0)
            d_valueC[therowc] -= valA * d_valueB[therowb];
    }
    __syncthreads();
}

__global__ void BlockLevel_spgemm_256(int_t n,
                                      int_t layer,
                                      pangulu_inblock_ptr *d_bin_rowpointer,
                                      pangulu_inblock_idx *d_bin_rowindex,
                                      pangulu_inblock_ptr *d_rowPtrA,
                                      pangulu_inblock_idx *d_colIdxA,
                                      calculate_type *d_valueA,
                                      pangulu_inblock_ptr *d_rowPtrB,
                                      pangulu_inblock_idx *d_colIdxB,
                                      calculate_type *d_valueB,
                                      pangulu_inblock_ptr *d_rowPtrC,
                                      pangulu_inblock_idx *d_colIdxC,
                                      calculate_type *d_valueC)
{
    int_t warp_local_id = threadIdx.x / 32;
    int_t warp_num = blockDim.x / 32;
    int_t lane_id = threadIdx.x % 32;

    const int_t rowidx = d_bin_rowindex[d_bin_rowpointer[layer] + blockIdx.x];

    __shared__ pangulu_inblock_idx s_idxC[256];
    __shared__ calculate_type s_valC[256];

    int_t therowc = d_rowPtrC[rowidx];
    int_t nextrowc = d_rowPtrC[rowidx + 1];

    for (int_t i = threadIdx.x; i < nextrowc - therowc; i += blockDim.x)
    {
        s_idxC[i] = d_colIdxC[therowc + i];
        s_valC[i] = 0;
    }
    __syncthreads();

    int_t therow = d_rowPtrA[rowidx];
    int_t nextrow = d_rowPtrA[rowidx + 1];

    for (int_t i = therow + warp_local_id; i < nextrow; i += warp_num)
    {
        int_t colA = d_colIdxA[i];
        calculate_type valA = d_valueA[i];

        int_t therowb = d_rowPtrB[colA];
        int_t nextrowb = d_rowPtrB[colA + 1];

        for (int_t j = therowb + lane_id; j < nextrowb; j += WARP_SIZE)
        {
            int_t colB = d_colIdxB[j];
            int_t flag = binarySearch_idx(s_idxC, 0, nextrowc - therowc - 1, colB);
            atomicAdd(&s_valC[flag], -valA * d_valueB[j]);
        }
    }
    __syncthreads();

    for (int_t i = threadIdx.x; i < nextrowc - therowc; i += blockDim.x)
    {
        d_valueC[therowc + i] += s_valC[i];
    }
}

__global__ void BlockLevel_spgemm_512(int_t n,
                                      int_t layer,
                                      pangulu_inblock_ptr *d_bin_rowpointer,
                                      pangulu_inblock_idx *d_bin_rowindex,
                                      pangulu_inblock_ptr *d_rowPtrA,
                                      pangulu_inblock_idx *d_colIdxA,
                                      calculate_type *d_valueA,
                                      pangulu_inblock_ptr *d_rowPtrB,
                                      pangulu_inblock_idx *d_colIdxB,
                                      calculate_type *d_valueB,
                                      pangulu_inblock_ptr *d_rowPtrC,
                                      pangulu_inblock_idx *d_colIdxC,
                                      calculate_type *d_valueC)
{
    int_t warp_local_id = threadIdx.x / 32;
    int_t warp_num = blockDim.x / 32;
    int_t lane_id = threadIdx.x % 32;

    const int_t rowidx = d_bin_rowindex[d_bin_rowpointer[layer] + blockIdx.x];

    __shared__ pangulu_inblock_idx s_idxC[512];
    __shared__ calculate_type s_valC[512];

    int_t therowc = d_rowPtrC[rowidx];
    int_t nextrowc = d_rowPtrC[rowidx + 1];

    for (int_t i = threadIdx.x; i < nextrowc - therowc; i += blockDim.x)
    {
        s_idxC[i] = d_colIdxC[therowc + i];
        s_valC[i] = 0;
    }
    __syncthreads();

    int_t therow = d_rowPtrA[rowidx];
    int_t nextrow = d_rowPtrA[rowidx + 1];

    for (int_t i = therow + warp_local_id; i < nextrow; i += warp_num)
    {
        int_t colA = d_colIdxA[i];
        calculate_type valA = d_valueA[i];

        int_t therowb = d_rowPtrB[colA];
        int_t nextrowb = d_rowPtrB[colA + 1];

        for (int_t j = therowb + lane_id; j < nextrowb; j += WARP_SIZE)
        {
            int_t colB = d_colIdxB[j];
            int_t flag = binarySearch_idx(s_idxC, 0, nextrowc - therowc - 1, colB);
            atomicAdd(&s_valC[flag], -valA * d_valueB[j]);
        }
    }
    __syncthreads();

    for (int_t i = threadIdx.x; i < nextrowc - therowc; i += blockDim.x)
    {
        d_valueC[therowc + i] += s_valC[i];
    }
}

__global__ void BlockLevel_spgemm_1024(int_t n,
                                       int_t layer,
                                       pangulu_inblock_ptr *d_bin_rowpointer,
                                       pangulu_inblock_idx *d_bin_rowindex,
                                       pangulu_inblock_ptr *d_rowPtrA,
                                       pangulu_inblock_idx *d_colIdxA,
                                       calculate_type *d_valueA,
                                       pangulu_inblock_ptr *d_rowPtrB,
                                       pangulu_inblock_idx *d_colIdxB,
                                       calculate_type *d_valueB,
                                       pangulu_inblock_ptr *d_rowPtrC,
                                       pangulu_inblock_idx *d_colIdxC,
                                       calculate_type *d_valueC)
{
    int_t warp_local_id = threadIdx.x / 32;
    int_t warp_num = blockDim.x / 32;
    int_t lane_id = threadIdx.x % 32;

    const int_t rowidx = d_bin_rowindex[d_bin_rowpointer[layer] + blockIdx.x];

    __shared__ pangulu_inblock_idx s_idxC[1024];
    __shared__ calculate_type s_valC[1024];

    int_t therowc = d_rowPtrC[rowidx];
    int_t nextrowc = d_rowPtrC[rowidx + 1];

    for (int_t i = threadIdx.x; i < nextrowc - therowc; i += blockDim.x)
    {
        s_idxC[i] = d_colIdxC[therowc + i];
        s_valC[i] = 0;
    }
    __syncthreads();

    int_t therow = d_rowPtrA[rowidx];
    int_t nextrow = d_rowPtrA[rowidx + 1];

    for (int_t i = therow + warp_local_id; i < nextrow; i += warp_num)
    {
        int_t colA = d_colIdxA[i];
        calculate_type valA = d_valueA[i];

        int_t therowb = d_rowPtrB[colA];
        int_t nextrowb = d_rowPtrB[colA + 1];

        for (int_t j = therowb + lane_id; j < nextrowb; j += WARP_SIZE)
        {
            int_t colB = d_colIdxB[j];
            int_t flag = binarySearch_idx(s_idxC, 0, nextrowc - therowc - 1, colB);
            atomicAdd(&s_valC[flag], -valA * d_valueB[j]);
        }
    }
    __syncthreads();

    for (int_t i = threadIdx.x; i < nextrowc - therowc; i += blockDim.x)
    {
        d_valueC[therowc + i] += s_valC[i];
    }
}

__global__ void BlockLevel_spgemm_2048(int_t n,
                                       int_t layer,
                                       pangulu_inblock_ptr *d_bin_rowpointer,
                                       pangulu_inblock_idx *d_bin_rowindex,
                                       pangulu_inblock_ptr *d_rowPtrA,
                                       pangulu_inblock_idx *d_colIdxA,
                                       calculate_type *d_valueA,
                                       pangulu_inblock_ptr *d_rowPtrB,
                                       pangulu_inblock_idx *d_colIdxB,
                                       calculate_type *d_valueB,
                                       pangulu_inblock_ptr *d_rowPtrC,
                                       pangulu_inblock_idx *d_colIdxC,
                                       calculate_type *d_valueC)
{
    int_t warp_local_id = threadIdx.x / 32;
    int_t warp_num = blockDim.x / 32;
    int_t lane_id = threadIdx.x % 32;

    const int_t rowidx = d_bin_rowindex[d_bin_rowpointer[layer] + blockIdx.x];

    __shared__ pangulu_inblock_idx s_idxC[2048];
    __shared__ calculate_type s_valC[2048];

    int_t therowc = d_rowPtrC[rowidx];
    int_t nextrowc = d_rowPtrC[rowidx + 1];

    for (int_t i = threadIdx.x; i < nextrowc - therowc; i += blockDim.x)
    {
        s_idxC[i] = d_colIdxC[therowc + i];
        s_valC[i] = 0;
    }
    __syncthreads();

    int_t therow = d_rowPtrA[rowidx];
    int_t nextrow = d_rowPtrA[rowidx + 1];

    for (int_t i = therow + warp_local_id; i < nextrow; i += warp_num)
    {
        int_t colA = d_colIdxA[i];
        calculate_type valA = d_valueA[i];

        int_t therowb = d_rowPtrB[colA];
        int_t nextrowb = d_rowPtrB[colA + 1];

        for (int_t j = therowb + lane_id; j < nextrowb; j += WARP_SIZE)
        {
            int_t colB = d_colIdxB[j];
            int_t flag = binarySearch_idx(s_idxC, 0, nextrowc - therowc - 1, colB);
            atomicAdd(&s_valC[flag], -valA * d_valueB[j]);
        }
    }
    __syncthreads();

    for (int_t i = threadIdx.x; i < nextrowc - therowc; i += blockDim.x)
    {
        d_valueC[therowc + i] += s_valC[i];
    }
}

__global__ void BlockLevel_spgemm_4097(int_t n,
                                       int_t layer,
                                       pangulu_inblock_ptr *d_bin_rowpointer,
                                       pangulu_inblock_idx *d_bin_rowindex,
                                       pangulu_inblock_ptr *d_rowPtrA,
                                       pangulu_inblock_idx *d_colIdxA,
                                       calculate_type *d_valueA,
                                       pangulu_inblock_ptr *d_rowPtrB,
                                       pangulu_inblock_idx *d_colIdxB,
                                       calculate_type *d_valueB,
                                       pangulu_inblock_ptr *d_rowPtrC,
                                       pangulu_inblock_idx *d_colIdxC,
                                       calculate_type *d_valueC)
{
    int_t warp_local_id = threadIdx.x / 32;
    int_t warp_num = blockDim.x / 32;
    int_t lane_id = threadIdx.x % 32;

    const int_t rowidx = d_bin_rowindex[d_bin_rowpointer[layer] + blockIdx.x];

    int_t therow = d_rowPtrA[rowidx];
    int_t nextrow = d_rowPtrA[rowidx + 1];

    int_t therowc = d_rowPtrC[rowidx];
    int_t nextrowc = d_rowPtrC[rowidx + 1];

    for (int_t i = therow + warp_local_id; i < nextrow; i += warp_num)
    {
        int_t colA = d_colIdxA[i];
        calculate_type valA = d_valueA[i];

        int_t therowb = d_rowPtrB[colA];
        int_t nextrowb = d_rowPtrB[colA + 1];

        for (int_t j = therowb + lane_id; j < nextrowb; j += WARP_SIZE)
        {
            int_t colB = d_colIdxB[j];

            int_t flag = binarySearch_idx(d_colIdxC, therowc, nextrowc - 1, colB);

            atomicAdd(&d_valueC[flag], -valA * d_valueB[j]);
        }

        __syncthreads();
    }
}

__forceinline__ __device__ calculate_type sum_32_shfl(calculate_type sum)
{
#pragma unroll
    for (int_t mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    return sum;
}
__global__ void spmv_warpvec_csr_cuda_executor(const pangulu_inblock_ptr *d_csrRowPtr,
                                               const pangulu_inblock_idx *d_csrColIdx,
                                               const calculate_type *d_csrVal,
                                               const pangulu_inblock_ptr m,
                                               const calculate_type *d_x,
                                               calculate_type *d_y)
{
    const int_t global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize
    const int_t rowid = global_id / WARP_SIZE;
    if (rowid >= m)
        return;

    const int_t lane_id = (WARP_SIZE - 1) & threadIdx.x;
    const int_t start = d_csrRowPtr[rowid];
    const int_t stop = d_csrRowPtr[rowid + 1];
    if (start == stop)
    {
        if (!lane_id)
            d_y[rowid] = (calculate_type)0;
        return;
    }

    calculate_type sum = (calculate_type)0;

    for (int_t j = start + lane_id; j < stop; j += WARP_SIZE)
    {
        sum += d_x[d_csrColIdx[j]] * d_csrVal[j];
    }
    sum = sum_32_shfl(sum);

    // finish
    if (!lane_id)
        d_y[rowid] = sum;
}
__global__ void cuda_Transform_csc_to_coo(int_t n, pangulu_inblock_ptr *d_colPtr, pangulu_inblock_idx *d_rowIdx, pangulu_inblock_idx *idx_col)
{
    int_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    int_t lane_id = threadIdx.x % 32;
    int_t warp_local_id = threadIdx.x / 32;

    if (warp_id >= n)
    {
        return;
    }

    int colA1 = d_colPtr[warp_id];
    int colA2 = d_colPtr[warp_id + 1];

    for (int i = colA1 + lane_id; i < colA2; i += WARP_SIZE)
    {
        idx_col[i] = warp_id;
    }
}
__global__ void WrapLevel_spgemm_dense_nnz(int_t n,
                                           int_t nnz,
                                           pangulu_inblock_ptr *d_rowPtrA,
                                           pangulu_inblock_idx *d_colIdxA,
                                           calculate_type *d_valueA,
                                           pangulu_inblock_ptr *d_rowPtrB,
                                           pangulu_inblock_idx *d_colIdxB,
                                           calculate_type *d_valueB,
                                           pangulu_inblock_ptr *d_rowPtrC,
                                           pangulu_inblock_idx *d_colIdxC,
                                           calculate_type *d_valueC,
                                           pangulu_inblock_idx *coo_col_B,
                                           calculate_type *temp_value_C)
{
    int_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    int_t lane_id = threadIdx.x % 32;
    int_t warp_local_id = threadIdx.x / 32;

    if (warp_id >= nnz)
    {
        return;
    }

    int colB = coo_col_B[warp_id];

    calculate_type *C_value = temp_value_C + n * colB;
    int i = d_colIdxB[warp_id];
    int colA1 = d_rowPtrA[i];
    int colA2 = d_rowPtrA[i + 1];
    calculate_type value_B = d_valueB[warp_id];
    for (int p_a = colA1 + lane_id; p_a < colA2; p_a += WARP_SIZE)
    {
        atomicAdd(&C_value[d_colIdxA[p_a]], -value_B * d_valueA[p_a]);
    }
}
__global__ void cuda_Transform_d_to_s_col(int_t n,
                                          int stride,
                                          pangulu_inblock_ptr *d_rowPtrA,
                                          pangulu_inblock_idx *d_colIdxA,
                                          calculate_type *d_valueA,
                                          calculate_type *temp_value_A)
{
    int_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    int_t lane_id = threadIdx.x % 32;
    int_t warp_local_id = threadIdx.x / 32;

    if (warp_id >= n)
    {
        return;
    }

    calculate_type *value = temp_value_A + warp_id * n;

    int colA1 = d_rowPtrA[warp_id];
    int colA2 = d_rowPtrA[warp_id + 1];

    for (int i = colA1 + lane_id; i < colA2; i += WARP_SIZE)
    {
        d_valueA[i] = value[d_colIdxA[i]];
    }
}

void pangulu_cuda_transport_kernel_CSC_to_CSR(int_t nnz, calculate_type *d_val_csr, pangulu_inblock_idx *d_idx, calculate_type *d_val_csc)
{

    int_t num_threads = WARP_SIZE * 2;
    int_t num_blocks = ceil((double)nnz / num_threads);

    trans_cuda_CSC_to_CSR<<<num_blocks, num_threads>>>(nnz, d_val_csr, d_idx, d_val_csc);
    cudaDeviceSynchronize();
}

void pangulu_cuda_transport_kernel_CSR_to_CSC(int_t nnz, calculate_type *d_val_csc, pangulu_inblock_idx *d_idx, calculate_type *d_val_csr)
{

    int_t num_threads = WARP_SIZE * 2;
    int_t num_blocks = ceil((double)nnz / num_threads);

    trans_cuda_CSR_to_CSC<<<num_blocks, num_threads>>>(nnz, d_val_csc, d_idx, d_val_csr);
    cudaDeviceSynchronize();
}

void pangulu_getrf_cuda_kernel(int_t n,
                               int_t nnzA,
                               int_32t *d_nnzU,
                               pangulu_inblock_ptr *A_CUDA_rowpointer,
                               pangulu_inblock_idx *A_CUDA_columnindex,
                               calculate_type *A_CUDA_value,
                               pangulu_inblock_ptr *L_CUDA_rowpointer,
                               pangulu_inblock_idx *L_CUDA_columnindex,
                               calculate_type *L_CUDA_value,
                               pangulu_inblock_ptr *U_CUDA_rowpointer,
                               pangulu_inblock_idx *U_CUDA_columnindex,
                               calculate_type *U_CUDA_value)
{
    int_t nnz_avrg = nnzA / n;

    int_t num_blocks;
    int_t num_threads;

    if (nnz_avrg <= 32)
    {
        num_blocks = ceil((double)n / WARP_NUM_WARPLU);
        num_threads = WARP_NUM_WARPLU * WARP_SIZE;
        WarpLevel_sflu<<<num_blocks, num_threads>>>(n, d_nnzU, A_CUDA_rowpointer, A_CUDA_columnindex, A_CUDA_value,
                                                    L_CUDA_rowpointer, L_CUDA_columnindex, L_CUDA_value,
                                                    U_CUDA_rowpointer, U_CUDA_columnindex, U_CUDA_value);
    }
    else if (nnz_avrg > 32 && nnz_avrg <= 96)
    {
        num_blocks = n;
        num_threads = 2 * WARP_SIZE;
        BlockLevel_sflu_L1<<<num_blocks, num_threads>>>(n, d_nnzU, A_CUDA_rowpointer, A_CUDA_columnindex, A_CUDA_value,
                                                        L_CUDA_rowpointer, L_CUDA_columnindex, L_CUDA_value,
                                                        U_CUDA_rowpointer, U_CUDA_columnindex, U_CUDA_value);
    }
    else if (nnz_avrg > 96 && nnz_avrg <= 192)
    {
        num_blocks = n;
        num_threads = 4 * WARP_SIZE;
        BlockLevel_sflu_L2<<<num_blocks, num_threads>>>(n, d_nnzU, A_CUDA_rowpointer, A_CUDA_columnindex, A_CUDA_value,
                                                        L_CUDA_rowpointer, L_CUDA_columnindex, L_CUDA_value,
                                                        U_CUDA_rowpointer, U_CUDA_columnindex, U_CUDA_value);
    }
    else if (nnz_avrg > 192 && nnz_avrg <= 384)
    {
        num_blocks = n;
        num_threads = 8 * WARP_SIZE;
        BlockLevel_sflu_L3<<<num_blocks, num_threads>>>(n, d_nnzU, A_CUDA_rowpointer, A_CUDA_columnindex, A_CUDA_value,
                                                        L_CUDA_rowpointer, L_CUDA_columnindex, L_CUDA_value,
                                                        U_CUDA_rowpointer, U_CUDA_columnindex, U_CUDA_value);
    }
    else
    {
        num_blocks = n;
        num_threads = 16 * WARP_SIZE;
        BlockLevel_sflu_L4<<<num_blocks, num_threads>>>(n, d_nnzU, A_CUDA_rowpointer, A_CUDA_columnindex, A_CUDA_value,
                                                        L_CUDA_rowpointer, L_CUDA_columnindex, L_CUDA_value,
                                                        U_CUDA_rowpointer, U_CUDA_columnindex, U_CUDA_value);
    }
}

void pangulu_getrf_cuda_dense_kernel(int_t n,
                                     int_t nnzA,
                                     int_32t *d_nnzU,
                                     pangulu_inblock_ptr *A_CUDA_rowpointer,
                                     pangulu_inblock_idx *A_CUDA_columnindex,
                                     calculate_type *A_CUDA_value,
                                     pangulu_inblock_ptr *L_CUDA_rowpointer,
                                     pangulu_inblock_idx *L_CUDA_columnindex,
                                     calculate_type *L_CUDA_value,
                                     pangulu_inblock_ptr *U_CUDA_rowpointer,
                                     pangulu_inblock_idx *U_CUDA_columnindex,
                                     calculate_type *U_CUDA_value)
{

    int num_threads = WARP_SIZE * WARP_PER_BLOCK;
    int num_blocks;

    num_blocks = ceil((double)n / WARP_PER_BLOCK);
    cuda_Transform_s_to_d_col<<<num_blocks, num_threads>>>(n, num_blocks, A_CUDA_rowpointer, A_CUDA_columnindex, A_CUDA_value, CUDA_TEMP_value);
    cudaDeviceSynchronize();

    num_blocks = n;

    lunumeric_cuda_kernel_v2<<<num_blocks, num_threads>>>(n, d_nnzU, CUDA_TEMP_value,
                                                          L_CUDA_rowpointer, L_CUDA_columnindex,
                                                          U_CUDA_rowpointer, U_CUDA_columnindex);

    cudaDeviceSynchronize();

    num_blocks = ceil((double)n / WARP_PER_BLOCK);

    cuda_Transform_d_to_s_lu_col<<<num_blocks, num_threads>>>(n, A_CUDA_rowpointer, A_CUDA_columnindex, L_CUDA_rowpointer, L_CUDA_columnindex, L_CUDA_value, U_CUDA_rowpointer, U_CUDA_columnindex, U_CUDA_value, CUDA_TEMP_value);
    cudaDeviceSynchronize();
}

void pangulu_tstrf_cuda_kernel_v7(int_t n,
                                  int_t nnzU,
                                  pangulu_inblock_ptr *U_rowpointer,
                                  pangulu_inblock_idx *U_columnindex,
                                  calculate_type *U_VALUE,
                                  pangulu_inblock_ptr *X_rowpointer,
                                  pangulu_inblock_idx *X_columnindex,
                                  calculate_type *X_VALUE,
                                  pangulu_inblock_ptr *A_rowpointer,
                                  pangulu_inblock_idx *A_columnindex,
                                  calculate_type *A_VALUE)
{
    int_t nnzU_avrg = nnzU / n;
    if (nnzU_avrg < 32)
    {
        int_t num_threads = WARP_PER_BLOCK_DTSTRF * WARP_SIZE;
        int_t num_blocks = ceil((calculate_type)n / WARP_PER_BLOCK_DTSTRF);
        TSTRF_Kernel_v2<<<num_blocks, num_threads>>>(n, U_rowpointer, U_columnindex, U_VALUE,
                                                     X_rowpointer, X_columnindex, X_VALUE,
                                                     A_rowpointer, A_columnindex, A_VALUE);
    }
    else
    {
        int_t num_threads = 8 * WARP_SIZE;
        int_t num_blocks = n;
        TSTRF_Kernel_v3<<<num_blocks, num_threads>>>(n, U_rowpointer, U_columnindex, U_VALUE,
                                                     X_rowpointer, X_columnindex, X_VALUE,
                                                     A_rowpointer, A_columnindex, A_VALUE);
    }
}

void pangulu_tstrf_cuda_kernel_v10(int_t n,
                                   int_t nnzU,
                                   pangulu_inblock_ptr *U_rowpointer,
                                   pangulu_inblock_idx *U_columnindex,
                                   calculate_type *U_VALUE,
                                   pangulu_inblock_ptr *X_rowpointer,
                                   pangulu_inblock_idx *X_columnindex,
                                   calculate_type *X_VALUE,
                                   pangulu_inblock_ptr *A_rowpointer,
                                   pangulu_inblock_idx *A_columnindex,
                                   calculate_type *A_VALUE)
{

    int_t num_threads = WARP_SIZE * WARP_PER_BLOCK;
    int_t num_blocks;

    num_blocks = ceil((calculate_type)n / WARP_PER_BLOCK);
    cuda_Transform_s_to_d_col<<<num_blocks, num_threads>>>(n, num_blocks, A_rowpointer, A_columnindex, A_VALUE, CUDA_TEMP_value);
    cudaDeviceSynchronize();

    num_threads = 8 * WARP_SIZE;
    num_blocks = n;
    TSTRF_Kernel_dense<<<num_blocks, num_threads>>>(n, U_rowpointer, U_columnindex, U_VALUE,
                                                    X_rowpointer, X_columnindex, X_VALUE,
                                                    A_rowpointer, A_columnindex, A_VALUE, CUDA_TEMP_value);
}

void pangulu_tstrf_cuda_kernel_v11(int_t n,
                                   int_t nnzU,
                                   int *degree,
                                   int *d_id_extractor,
                                   calculate_type *d_left_sum,
                                   pangulu_inblock_ptr *U_rowpointer,
                                   pangulu_inblock_idx *U_columnindex,
                                   calculate_type *U_VALUE,
                                   pangulu_inblock_ptr *X_rowpointer,
                                   pangulu_inblock_idx *X_columnindex,
                                   calculate_type *X_VALUE,
                                   pangulu_inblock_ptr *A_rowpointer,
                                   pangulu_inblock_idx *A_columnindex,
                                   calculate_type *A_VALUE)
{
    int_t num_threads = WARP_SIZE * WARP_PER_BLOCK;
    int_t num_blocks;

    num_blocks = ceil((calculate_type)n / WARP_PER_BLOCK);
    cuda_Transform_s_to_d_col<<<num_blocks, num_threads>>>(n, num_blocks, A_rowpointer, A_columnindex, A_VALUE, CUDA_TEMP_value);
    cudaDeviceSynchronize();

    num_threads = WARP_PER_BLOCK_DGESSM * WARP_SIZE;
    num_blocks = ceil((calculate_type)n / (calculate_type)WARP_PER_BLOCK_DGESSM);
    syncfree_cuda_csr_dense_v11_U<<<num_blocks, num_threads>>>(n, n, degree, U_rowpointer, U_columnindex, U_VALUE,
                                                               A_rowpointer, A_columnindex, X_VALUE, A_VALUE, d_id_extractor, d_left_sum, CUDA_TEMP_value);
    cudaDeviceSynchronize();
}

void pangulu_gessm_cuda_kernel_v7(int_t n,
                                  int_t nnzL,
                                  pangulu_inblock_ptr *L_columnpointer,
                                  pangulu_inblock_idx *L_rowindex,
                                  calculate_type *L_VALUE,
                                  pangulu_inblock_ptr *X_columnpointer,
                                  pangulu_inblock_idx *X_rowindex,
                                  calculate_type *X_VALUE,
                                  pangulu_inblock_ptr *A_columnpointer,
                                  pangulu_inblock_idx *A_rowindex,
                                  calculate_type *A_VALUE)
{
    int_t nnzL_avrg = nnzL / n;

    if (nnzL_avrg < 32)
    {
        int_t num_threads = WARP_PER_BLOCK_DGESSM * WARP_SIZE;
        int_t num_blocks = ceil((calculate_type)n / WARP_PER_BLOCK_DGESSM);
        GESSM_Kernel_v2<<<num_blocks, num_threads>>>(n, L_columnpointer, L_rowindex, L_VALUE,
                                                     X_columnpointer, X_rowindex, X_VALUE,
                                                     A_columnpointer, A_rowindex, A_VALUE);
    }
    else
    {
        int_t num_threads = 8 * WARP_SIZE;
        int_t num_blocks = n;
        GESSM_Kernel_v3<<<num_blocks, num_threads>>>(n, L_columnpointer, L_rowindex, L_VALUE,
                                                     X_columnpointer, X_rowindex, X_VALUE,
                                                     A_columnpointer, A_rowindex, A_VALUE);
    }
}

void pangulu_gessm_cuda_kernel_v8(int_t n,
                                  int_t nnzL,
                                  int_t nnzX,
                                  int *degree,
                                  int *d_id_extractor,
                                  calculate_type *d_left_sum,
                                  pangulu_inblock_ptr *L_columnpointer,
                                  pangulu_inblock_idx *L_rowindex,
                                  calculate_type *L_VALUE,
                                  pangulu_inblock_ptr *X_columnpointer,
                                  pangulu_inblock_idx *X_rowindex,
                                  calculate_type *X_VALUE,
                                  pangulu_inblock_ptr *A_columnpointer,
                                  pangulu_inblock_idx *A_rowindex,
                                  calculate_type *A_VALUE)
{

    int num_threads = WARP_PER_BLOCK_DGESSM * WARP_SIZE;
    int num_blocks = ceil((calculate_type)n / (calculate_type)WARP_PER_BLOCK_DGESSM);
    syncfree_cuda_csr<<<num_blocks, num_threads>>>(n, n, degree, L_columnpointer, L_rowindex, L_VALUE,
                                                   A_columnpointer, A_rowindex, X_VALUE, A_VALUE, d_id_extractor, d_left_sum);
    cudaDeviceSynchronize();
}

void pangulu_gessm_cuda_kernel_v10(int_t n,
                                   int_t nnzL,
                                   pangulu_inblock_ptr *L_columnpointer,
                                   pangulu_inblock_idx *L_rowindex,
                                   calculate_type *L_VALUE,
                                   pangulu_inblock_ptr *X_columnpointer,
                                   pangulu_inblock_idx *X_rowindex,
                                   calculate_type *X_VALUE,
                                   pangulu_inblock_ptr *A_columnpointer,
                                   pangulu_inblock_idx *A_rowindex,
                                   calculate_type *A_VALUE)
{
    int_t num_threads = WARP_SIZE * WARP_PER_BLOCK;
    int_t num_blocks;

    num_blocks = ceil((calculate_type)n / WARP_PER_BLOCK);
    cuda_Transform_s_to_d_col<<<num_blocks, num_threads>>>(n, num_blocks, A_columnpointer, A_rowindex, A_VALUE, CUDA_TEMP_value);
    cudaDeviceSynchronize();

    num_threads = 8 * WARP_SIZE;
    num_blocks = n;
    GESSM_Kernel_dense<<<num_blocks, num_threads>>>(n, L_columnpointer, L_rowindex, L_VALUE,
                                                    X_columnpointer, X_rowindex, X_VALUE,
                                                    A_columnpointer, A_rowindex, A_VALUE, CUDA_TEMP_value);
    cudaDeviceSynchronize();
}

void pangulu_ssssm_cuda_kernel(int_t n,
                               pangulu_inblock_ptr *h_bin_rowpointer,
                               pangulu_inblock_ptr *d_bin_rowpointer,
                               pangulu_inblock_idx *d_bin_rowindex,
                               pangulu_inblock_ptr *d_rowPtrA,
                               pangulu_inblock_idx *d_colIdxA,
                               calculate_type *d_valueA,
                               pangulu_inblock_ptr *d_rowPtrB,
                               pangulu_inblock_idx *d_colIdxB,
                               calculate_type *d_valueB,
                               pangulu_inblock_ptr *d_rowPtrC,
                               pangulu_inblock_idx *d_colIdxC,
                               calculate_type *d_valueC)
{

    int_t num_blocks;
    int_t num_threads;
    int_t layer;

    // 0:0
    layer = 0;

    // 1:1
    layer = 1;
    num_threads = WARP_PER_BLOCK_GEMM * WARP_SIZE;
    num_blocks = ceil((double)(h_bin_rowpointer[2] - h_bin_rowpointer[1]) / num_threads);
    if (num_blocks > 0)
    {
        ThreadLevel_spgemm<<<num_blocks, num_threads>>>(n, layer, d_bin_rowpointer, d_bin_rowindex,
                                                        d_rowPtrA, d_colIdxA, d_valueA,
                                                        d_rowPtrB, d_colIdxB, d_valueB,
                                                        d_rowPtrC, d_colIdxC, d_valueC);
    }

    // 2:2-32
    layer = 2;
    num_threads = WARP_PER_BLOCK_GEMM * WARP_SIZE;
    num_blocks = ceil((double)(h_bin_rowpointer[3] - h_bin_rowpointer[2]) / WARP_PER_BLOCK_GEMM);
    if (num_blocks > 0)
    {
        WarpLevel_spgemm_32<<<num_blocks, num_threads>>>(n, layer, d_bin_rowpointer, d_bin_rowindex,
                                                         d_rowPtrA, d_colIdxA, d_valueA,
                                                         d_rowPtrB, d_colIdxB, d_valueB,
                                                         d_rowPtrC, d_colIdxC, d_valueC);
    }

    // 3:33-64
    layer = 3;
    num_threads = WARP_PER_BLOCK_GEMM * WARP_SIZE;
    num_blocks = ceil((double)(h_bin_rowpointer[4] - h_bin_rowpointer[3]) / WARP_PER_BLOCK_GEMM);
    if (num_blocks > 0)
    {
        WarpLevel_spgemm_64<<<num_blocks, num_threads>>>(n, layer, d_bin_rowpointer, d_bin_rowindex,
                                                         d_rowPtrA, d_colIdxA, d_valueA,
                                                         d_rowPtrB, d_colIdxB, d_valueB,
                                                         d_rowPtrC, d_colIdxC, d_valueC);
    }

    // 4:65-128
    layer = 4;
    num_threads = WARP_PER_BLOCK_GEMM * WARP_SIZE;
    num_blocks = ceil((double)(h_bin_rowpointer[5] - h_bin_rowpointer[4]) / WARP_PER_BLOCK_GEMM);
    if (num_blocks > 0)
    {
        WarpLevel_spgemm_128<<<num_blocks, num_threads>>>(n, layer, d_bin_rowpointer, d_bin_rowindex,
                                                          d_rowPtrA, d_colIdxA, d_valueA,
                                                          d_rowPtrB, d_colIdxB, d_valueB,
                                                          d_rowPtrC, d_colIdxC, d_valueC);
    }

    // 5:129-256
    layer = 5;
    num_threads = 64;
    num_blocks = h_bin_rowpointer[6] - h_bin_rowpointer[5];
    if (num_blocks > 0)
    {
        BlockLevel_spgemm_256<<<num_blocks, num_threads>>>(n, layer, d_bin_rowpointer, d_bin_rowindex,
                                                           d_rowPtrA, d_colIdxA, d_valueA,
                                                           d_rowPtrB, d_colIdxB, d_valueB,
                                                           d_rowPtrC, d_colIdxC, d_valueC);
    }

    // 6:257-512
    layer = 6;
    num_threads = 128;
    num_blocks = h_bin_rowpointer[7] - h_bin_rowpointer[6];
    if (num_blocks > 0)
    {
        BlockLevel_spgemm_512<<<num_blocks, num_threads>>>(n, layer, d_bin_rowpointer, d_bin_rowindex,
                                                           d_rowPtrA, d_colIdxA, d_valueA,
                                                           d_rowPtrB, d_colIdxB, d_valueB,
                                                           d_rowPtrC, d_colIdxC, d_valueC);
    }

    // 7:513-1024
    layer = 7;
    num_threads = 256;
    num_blocks = h_bin_rowpointer[8] - h_bin_rowpointer[7];
    if (num_blocks > 0)
    {
        BlockLevel_spgemm_1024<<<num_blocks, num_threads>>>(n, layer, d_bin_rowpointer, d_bin_rowindex,
                                                            d_rowPtrA, d_colIdxA, d_valueA,
                                                            d_rowPtrB, d_colIdxB, d_valueB,
                                                            d_rowPtrC, d_colIdxC, d_valueC);
    }

    // 8:1025-2048
    layer = 8;
    num_threads = 512;
    num_blocks = h_bin_rowpointer[9] - h_bin_rowpointer[8];
    if (num_blocks > 0)
    {
        BlockLevel_spgemm_2048<<<num_blocks, num_threads>>>(n, layer, d_bin_rowpointer, d_bin_rowindex,
                                                            d_rowPtrA, d_colIdxA, d_valueA,
                                                            d_rowPtrB, d_colIdxB, d_valueB,
                                                            d_rowPtrC, d_colIdxC, d_valueC);
    }

    // 9:2049++
    layer = 9;
    num_threads = 1024;
    num_blocks = h_bin_rowpointer[10] - h_bin_rowpointer[9];
    if (num_blocks > 0)
    {
        BlockLevel_spgemm_4097<<<num_blocks, num_threads>>>(n, layer, d_bin_rowpointer, d_bin_rowindex,
                                                            d_rowPtrA, d_colIdxA, d_valueA,
                                                            d_rowPtrB, d_colIdxB, d_valueB,
                                                            d_rowPtrC, d_colIdxC, d_valueC);
    }

    // 10:4097+++
    layer = 10;
    num_threads = 1024;
    num_blocks = h_bin_rowpointer[11] - h_bin_rowpointer[10];
    if (num_blocks > 0)
    {
        BlockLevel_spgemm_4097<<<num_blocks, num_threads>>>(n, layer, d_bin_rowpointer, d_bin_rowindex,
                                                            d_rowPtrA, d_colIdxA, d_valueA,
                                                            d_rowPtrB, d_colIdxB, d_valueB,
                                                            d_rowPtrC, d_colIdxC, d_valueC);
    }
}

void pangulu_ssssm_dense_cuda_kernel(int_t n,
                                     int_t nnzC,
                                     int_t nnzB,
                                     pangulu_inblock_ptr *d_rowPtrA,
                                     pangulu_inblock_idx *d_colIdxA,
                                     calculate_type *d_valueA,
                                     pangulu_inblock_ptr *d_rowPtrB,
                                     pangulu_inblock_idx *d_colIdxB,
                                     calculate_type *d_valueB,
                                     pangulu_inblock_ptr *d_rowPtrC,
                                     pangulu_inblock_idx *d_colIdxC,
                                     calculate_type *d_valueC)
{

    int_t num_blocks, num_blocks_B, num_blocks_C;
    int_t num_threads;
    int_t layer;
    num_threads = WARP_PER_BLOCK_GEMM * WARP_SIZE;

    num_blocks = ceil((double)n / WARP_PER_BLOCK_GEMM);

    cuda_Transform_csc_to_coo<<<num_blocks, num_threads>>>(n, d_rowPtrB, d_colIdxB, CUDA_B_idx_COL);

    num_blocks = ceil((double)n / WARP_PER_BLOCK_GEMM);

    cuda_Transform_s_to_d_col<<<num_blocks, num_threads>>>(n, num_blocks, d_rowPtrC, d_colIdxC, d_valueC, CUDA_TEMP_value);

    num_blocks_B = ceil((double)nnzB / WARP_PER_BLOCK_GEMM);

    WrapLevel_spgemm_dense_nnz<<<num_blocks_B, num_threads>>>(n, nnzB, d_rowPtrA, d_colIdxA, d_valueA, d_rowPtrB, d_colIdxB, d_valueB, d_rowPtrC, d_colIdxC, d_valueC, CUDA_B_idx_COL, CUDA_TEMP_value);

    cuda_Transform_d_to_s_col<<<num_blocks, num_threads>>>(n, num_blocks, d_rowPtrC, d_colIdxC, d_valueC, CUDA_TEMP_value);
}

void pangulu_cudaMemcpyAsync_host_to_device(void *gpu_address, void *cpu_address, int_t size, cudaStream_t *stream)
{
    cudaMemcpyAsync(gpu_address, cpu_address, size, cudaMemcpyHostToDevice, *stream);
}

void pangulu_cudaMemcpyAsync_device_to_host(void *cpu_address, void *gpu_address, int_t size, cudaStream_t *stream)
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
void pangulu_cudamemcpy_device_to_device(void *gpu_address1, void *gpu_address2, int_t size)
{
    cudaMemcpy(gpu_address1, gpu_address2, size, cudaMemcpyHostToDevice);
}

void pangulu_cudamemcpyAsync_device_to_device(void *gpu_address1, void *gpu_address2, int_t size, cudaStream_t *stream)
{
    cudaMemcpyAsync(gpu_address1, gpu_address2, size, cudaMemcpyHostToDevice, *stream);
}