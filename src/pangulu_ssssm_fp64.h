#ifndef PANGULU_SSSSM_FP64_H
#define PANGULU_SSSSM_FP64_H

#include "pangulu_common.h"

typedef unsigned int sflu_uint;

#define setbit(x, y) x |= (1 << y)  // set the yth bit of x is 1
#define clrbit(x, y) x &= ~(1 << y) // set the yth bit of x is 0
#define getbit(x, y) ((x) >> (y)&1) // get the yth bit of x

#define Bound 5000
#define binbd1 64
#define binbd2 4096

void exclusive_scan(int_t *input, int length)
{
    if (length == 0 || length == 1)
        return;

    calculate_type old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i - 1];
        old_val = new_val;
    }
}

void swap_key(int_t *a, int_t *b)
{
    int_t tmp = *a;
    if (a != NULL && b != NULL)
    {
        *a = *b;
    }
    if (b != NULL)
    {
        *b = tmp;
    }
}

void swap_val(calculate_type *a, calculate_type *b)
{
    calculate_type tmp = *a;
    if (a != NULL && b != NULL)
    {
        *a = *b;
    }
    if (b != NULL)
    {
        *b = tmp;
    }
}

int_t partition_key_val_pair(int_t *key, calculate_type *val, int_t length, int_t pivot_index)
{
    int_t i = 0;
    int_t small_length = pivot_index;

    int_t pivot = key[pivot_index];
    if (val != NULL && key != NULL)
    {
        swap_key(&key[pivot_index], &key[pivot_index + (length - 1)]);
        swap_val(&val[pivot_index], &val[pivot_index + (length - 1)]);
    }

    for (; i < length; i++)
    {
        if (key != NULL)
        {
            if (key[pivot_index + i] < pivot)
            {
                swap_key(&key[pivot_index + i], &key[small_length]);
                if (val != NULL)
                {
                    swap_val(&val[pivot_index + i], &val[small_length]);
                }
                small_length++;
            }
        }
    }

    if (key != NULL)
    {
        swap_key(&key[pivot_index + length - 1], &key[small_length]);
    }
    if (val != NULL)
    {
        swap_val(&val[pivot_index + length - 1], &val[small_length]);
    }

    return small_length;
}

void insert_sort_key_val_pair(int_t *key, calculate_type *val, int_t length)
{
    for (int_t i = 1; i < length; i++)
    {
        int_t tmp_key = key[i];
        calculate_type tmp_val = val[i];
        int_t j = i - 1;
        while ((j >= 0) && (key[j] > tmp_key))
        {
            key[j + 1] = key[j];
            val[j + 1] = val[j];
            j--;
        }
        key[j + 1] = tmp_key;
        val[j + 1] = tmp_val;
    }
}

void quick_sort_key_val_pair1(int_t *key, calculate_type *val, int_t length)
{
    if (length == 0 || length == 1)
        return;

    int_t sorted = 1;

    // get mid of three
    int_t first = 0;
    int_t last = length - 1;
    int_t mid = first + ((last - last) >> 1);
    // int_t pivot = mid;
    if (key[mid] > key[first])
    {
        swap_key(&key[mid], &key[first]);
        swap_val(&val[mid], &val[first]);
    }
    if (key[first] > key[last])
    {
        swap_key(&key[first], &key[last]);
        swap_val(&val[first], &val[last]);
    }
    if (key[mid] > key[last])
    {
        swap_key(&key[mid], &key[last]);
        swap_val(&val[mid], &val[last]);
    }

    for (int_t i = 1; i < length; i++)
    {
        if (key[i] < key[i - 1])
        {
            sorted = 0;
            break;
        }
    }

    if (!sorted)
    {
        if (length > 64)
        {
            int_t small_length = partition_key_val_pair(key, val, length, 0);
            quick_sort_key_val_pair1(key, val, small_length);
            quick_sort_key_val_pair1(&key[small_length + 1], &val[small_length + 1], length - small_length - 1);
        }
        else
        {
            insert_sort_key_val_pair(key, val, length);
        }
    }
}

void segmented_sum(calculate_type *input, int_t *bit_flag, int_t length)
{
    if (length == 0 || length == 1)
        return;

    for (int_t i = 0; i < length; i++)
    {
        if (bit_flag[i])
        {
            int_t j = i + 1;
            while (!bit_flag[j] && j < length)
            {
                input[i] += input[j];
                j++;
            }
        }
    }
}

void thmkl_dcsrmultcsr(const int_t *m, const int_t *n, const int_t *k,
                       calculate_type *a, idx_int *ja, int_t *ia,
                       calculate_type *b, idx_int *jb, int_t *ib,
                       calculate_type *c, idx_int *jc, int_t *ic)
{

    if (*m <= Bound) // if C size < bound, use spa method
    {
#pragma omp parallel for num_threads(1)
        for (int_t iid = 0; iid < *m; iid++)
        {
            int_t len = (*k + 31) / 32;
            sflu_uint *mask = (sflu_uint *)malloc(sizeof(sflu_uint) * len);
            memset(mask, 0, sizeof(sflu_uint) * len);
            calculate_type *d_dense_row_value = (calculate_type *)malloc((*k) * sizeof(calculate_type));
            memset(d_dense_row_value, 0, (*k) * sizeof(calculate_type));

            for (int_t i = ia[iid] - ia[0]; i < ia[iid + 1] - ia[0]; i++)
            {
                int_t col = ja[i] - ia[0];
                for (int_t l = ib[col] - ib[0]; l < ib[col + 1] - ib[0]; l++)
                {
                    const int_t key = jb[l] - ib[0];
                    setbit(mask[key / 32], key % 32);
                    d_dense_row_value[key] += b[l] * a[i];
                }
            }

            int_t nnzr = ic[iid] - ic[0];
            for (int_t cid = 0; cid < *k; cid++)
            {
                if (getbit(mask[cid / 32], cid % 32) == 1)
                {
                    while (jc[nnzr] < cid + ic[0] && nnzr < ic[iid + 1] - ic[0])
                    {
                        nnzr++;
                    }
                    c[nnzr] -= d_dense_row_value[cid];
                }
            }

            free(mask);
            free(d_dense_row_value);
        }
    }
    else // if C size > bound, use bin method
    {

        int_t *iCub = (int_t *)malloc(*m * sizeof(int_t));
        memset(iCub, 0, sizeof(int_t) * *m);

        for (int_t i = 0; i < *m; i++)
        {
            for (int_t j = ia[i] - ia[0]; j < ia[i + 1] - ia[0]; j++)
            {
                int_t rowAtop = ja[j] - ia[0];
                iCub[i] += ib[rowAtop + 1] - ib[rowAtop];
            }
        }
#pragma omp parallel for num_threads(1)
        for (int_t i = 0; i < *m; i++)
        {
            int_t rowid = i;
            int_t rowsize = iCub[rowid];
            if (rowsize == 0)
                continue;

            if (rowsize <= 64)
            {
                // esc
                if (rowsize == 0)
                    continue;
                int_t *jCub = (int_t *)malloc(rowsize * sizeof(int_t));
                calculate_type *valCub = (calculate_type *)malloc(rowsize * sizeof(calculate_type));
                int_t *d_flagCub = (int_t *)malloc(rowsize * sizeof(int_t));
                memset(jCub, 0, rowsize * sizeof(int_t));
                memset(valCub, 0, rowsize * sizeof(calculate_type));
                memset(d_flagCub, 0, rowsize * sizeof(int_t));

                int_t incr = 0;
                for (int_t l = ia[rowid] - ia[0]; l < ia[rowid + 1] - ia[0]; l++)
                {
                    int_t rowB = ja[l] - ia[0];
                    calculate_type val = a[l];
                    for (int_t k = ib[rowB] - ib[0]; k < ib[rowB + 1] - ib[0]; k++)
                    {
                        jCub[incr] = jb[k];
                        valCub[incr] = val * b[k];
                        incr++;
                    }
                }

                // sort
                quick_sort_key_val_pair1(jCub, valCub, rowsize);

                // compress
                d_flagCub[0] = 1;
                for (int_t idx = 0; idx < rowsize - 1; idx++)
                {
                    d_flagCub[idx + 1] = jCub[idx + 1] == jCub[idx] ? 0 : 1;
                }
                segmented_sum(valCub, d_flagCub, rowsize);

                int_t incrn = ic[rowid] - ic[0];
                for (int_t idx = 0; idx < rowsize; idx++)
                {
                    if (d_flagCub[idx] == 1)
                    {
                        while (jc[incrn] < jCub[idx] && incrn < ic[rowid + 1] - ic[0])
                        {
                            incrn++;
                        }
                        c[incrn] -= valCub[idx];
                    }
                }

                free(jCub);
                free(valCub);
                free(d_flagCub);
            }
            else if (rowsize > 64 && rowsize <= 4096)
            {
                // hash
                int_t hashsize_full_reg = (ic[rowid + 1] - ic[rowid]) / 0.75;
                int_t *tmpHashtable = (int_t *)malloc(hashsize_full_reg * sizeof(int_t));
                memset(tmpHashtable, -1, sizeof(int_t) * hashsize_full_reg);
                calculate_type *tmpValue = (calculate_type *)malloc(hashsize_full_reg * sizeof(calculate_type));
                memset(tmpValue, 0, sizeof(calculate_type) * hashsize_full_reg);
                for (int_t blkj = ia[rowid] - ia[0]; blkj < ia[rowid + 1] - ia[0]; blkj++)
                {
                    int_t col = ja[blkj] - ia[0];
                    for (int_t l = ib[col] - ib[0]; l < ib[col + 1] - ib[0]; l++)
                    {
                        const int_t key = jb[l] - ib[0];
                        int_t hashadr = (key * 107) % hashsize_full_reg;
                        // int_t hashadr = key % hashsize_full_reg;
                        while (1)
                        {
                            const int_t keyexist = tmpHashtable[hashadr];
                            if (keyexist == key)
                            {
                                tmpValue[hashadr] += b[l] * a[blkj];
                                break;
                            }
                            else if (keyexist == -1)
                            {
                                tmpHashtable[hashadr] = key;
                                tmpValue[hashadr] = b[l] * a[blkj];
                                break;
                            }
                            else
                            {
                                hashadr = (hashadr + 1) % hashsize_full_reg;
                            }
                        }
                    }
                }
                quick_sort_key_val_pair1(tmpHashtable, tmpValue, hashsize_full_reg);

                int_t cptr = ic[rowid] - ic[0];
                for (int_t k = 0; k < hashsize_full_reg; k++)
                {
                    if (tmpHashtable[k] != -1)
                    {
                        while (jc[cptr] < tmpHashtable[k] + ib[0] && cptr < ic[rowid + 1] - ic[0])
                        {
                            cptr++;
                        }
                        c[cptr] -= tmpValue[k];
                    }
                }
                free(tmpHashtable);
                free(tmpValue);
            }
            else if (rowsize > 4096)
            {
                // spa
                if (rowsize == 0)
                    continue;
                int_t len = (*k + 31) / 32;
                int_t *mask = (int_t *)malloc(sizeof(int_t) * len);
                memset(mask, 0, sizeof(int_t) * len);
                calculate_type *d_dense_row_value = (calculate_type *)malloc((*k) * sizeof(calculate_type));
                memset(d_dense_row_value, 0, (*k) * sizeof(calculate_type));

                for (int_t offsetA = ia[rowid] - ia[0]; offsetA < ia[rowid + 1] - ia[0]; offsetA++)
                {
                    int_t col = ja[offsetA] - ia[0];
                    for (int_t l = ib[col] - ib[0]; l < ib[col + 1] - ib[0]; l++)
                    {
                        const int_t key = jb[l] - ib[0];
                        setbit(mask[key / 32], key % 32);
                        d_dense_row_value[key] += b[l] * a[offsetA];
                    }
                }

                int_t nnzr = ic[rowid] - ic[0];
                for (int_t cid = 0; cid < *k; cid++)
                {
                    if (getbit(mask[cid / 32], cid % 32) == 1)
                    {
                        while (jc[nnzr] < cid + ic[0] && nnzr < ic[rowid + 1] - ic[0])
                        {
                            nnzr++;
                        }
                        c[nnzr] -= d_dense_row_value[cid];
                    }
                }

                free(mask);
                free(d_dense_row_value);
            }
        }
        free(iCub);
    }
}

const idx_int *lower_bound(const idx_int *begins, const idx_int *ends, idx_int key)
{
    int_t curSize = 0;
    curSize = ends - begins;
    int_t half;
    const idx_int *middle;
    while (curSize > 0)
    {
        half = curSize >> 1;
        middle = begins + half;
        if (*middle < key)
        {
            begins = middle;
            ++begins;
            curSize = curSize - half - 1;
        }
        else
            curSize = half;
    }
    return begins;
}

void SpMM(
    int_t m, int_t k, int_t n,
    const int_t *A_csrOffsets,
    const idx_int *A_columns,
    const calculate_type *A_values,
    const int_t *B_csrOffsets,
    const idx_int *B_columns,
    const calculate_type *B_values,
    const int_t *C_csrOffsets,
    const idx_int *C_columns,
    calculate_type *C_values)
{
    for (idx_int rowA = 0; rowA < m; ++rowA)
    {
        for (int_t i = A_csrOffsets[rowA]; i < A_csrOffsets[rowA + 1]; ++i)
        {
            idx_int rowB = A_columns[i];
            for (int_t j = B_csrOffsets[rowB]; j < B_csrOffsets[rowB + 1]; ++j)
            {
                idx_int colB = B_columns[j];
                /// find locate of C at [rowA,colB]
                idx_int loc = (idx_int)(lower_bound(
                                            C_columns + C_csrOffsets[rowA],
                                            C_columns + C_csrOffsets[rowA + 1],
                                            colB) -
                                        C_columns);
                if (C_columns[loc] == colB)
                {
                    // printf("ERROR\n");
                    C_values[loc] -= A_values[i] * B_values[j];
                }
            }
        }
    }
}
int binary_search_right_boundary(const int_t *data,
                                 const int_t key_input,
                                 const int begin,
                                 const int end)
{
    int start = begin;
    int stop = end;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;

        key_median = data[median];

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start;
}
void cscmultcsc_dense(pangulu_Smatrix *A,
                      pangulu_Smatrix *L,
                      pangulu_Smatrix *U)
{
    int n = A->row;
    int omp_threads_num = PANGU_OMP_NUM_THREADS;
    ssssm_col_ops_u[0] = 0;
    ssssm_ops_pointer[0] = 0;

#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
    for (int i = 0; i < n; i++)
    {
        for (int j = A->columnpointer[i]; j < A->columnpointer[i + 1]; j++)
        {
            int row = A->rowindex[j];
            TEMP_A_value[i * n + row] = A->value_CSC[j]; // tranform csc to dense,only value
        }
    }

#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
    for (int i = 0; i < n; i++)
    {
        int ops = 0;
        for (int k = U->columnpointer[i]; k < U->columnpointer[i + 1]; k++)
        {
            int L_col = U->rowindex[k];
            ops += (L->columnpointer[L_col + 1] - L->columnpointer[L_col]);
        }
        ssssm_col_ops_u[i] = ops;
    }
    for (int i = 0; i < n; i++)
    {
        ssssm_col_ops_u[i + 1] = ssssm_col_ops_u[i + 1] + ssssm_col_ops_u[i];
    }
    int_t ops_div = (ssssm_col_ops_u[n] - 1) / omp_threads_num + 1;
    int_t boundary;
#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
    for (int i = 1; i <= omp_threads_num; i++)
    {
        boundary = i * ops_div;
        ssssm_ops_pointer[i] = binary_search_right_boundary(ssssm_col_ops_u, boundary, 0, n) - 1;
    }

#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
    for (int t = 0; t < omp_threads_num; t++)
    {
        for (int j = ssssm_ops_pointer[t]; j < ssssm_ops_pointer[t + 1]; j++)
        {
            for (int pointer_u = U->columnpointer[j]; pointer_u < U->columnpointer[j + 1]; pointer_u++)
            {
                int i = U->rowindex[pointer_u]; // ith row
                calculate_type value_U = U->value_CSC[pointer_u];
                calculate_type *A_value = TEMP_A_value + n * j; // point to jth column
                for (int pointer_l = L->columnpointer[i]; pointer_l < L->columnpointer[i + 1]; pointer_l++)
                {                                                    // ith column Lki Uij Akj
                    int k = L->rowindex[pointer_l];                  // kth row
                    A_value[k] -= value_U * L->value_CSC[pointer_l]; // schur
                }
            }
        }
    }
#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
    for (int i = 0; i < n; i++)
    {
        calculate_type *A_value = TEMP_A_value + i * n;
        for (int j = A->columnpointer[i]; j < A->columnpointer[i + 1]; j++)
        {
            int row = A->rowindex[j];
            A->value_CSC[j] = A_value[row];
        }
    }
}
void pangulu_ssssm_fp64(pangulu_Smatrix *A,
                        pangulu_Smatrix *L,
                        pangulu_Smatrix *U)
{

#ifdef OUTPUT_MATRICES
    char out_name_A[512];
    char out_name_B[512];
    char out_name_C[512];
    sprintf(out_name_A, "%s/%s/%d%s", OUTPUT_FILE, "ssssm", ssssm_number, "_ssssm_A.cbd");
    sprintf(out_name_B, "%s/%s/%d%s", OUTPUT_FILE, "ssssm", ssssm_number, "_ssssm_B.cbd");
    sprintf(out_name_C, "%s/%s/%d%s", OUTPUT_FILE, "ssssm", ssssm_number, "_ssssm_C.cbd");
    pangulu_binary_write_csc_pangulu_Smatrix(L, out_name_A);
    pangulu_binary_write_csc_pangulu_Smatrix(U, out_name_B);
    pangulu_binary_write_csc_pangulu_Smatrix(A, out_name_C);
    ssssm_number++;
#endif
    cscmultcsc_dense(A, L, U);
}

void pangulu_ssssm_interface_C_V1(pangulu_Smatrix *A,
                                  pangulu_Smatrix *L,
                                  pangulu_Smatrix *U)
{
#ifdef GPU_OPEN
    pangulu_Smatrix_CUDA_memcpy_value_CSC(A, A);
    pangulu_Smatrix_CUDA_memcpy_value_CSC(L, L);
    pangulu_Smatrix_CUDA_memcpy_value_CSC(U, U);
#endif
    cscmultcsc_dense(A, L, U);
#ifdef GPU_OPEN
    pangulu_Smatrix_CUDA_memcpy_to_device_value_CSC(A, A);
#endif
}
void pangulu_ssssm_interface_C_V2(pangulu_Smatrix *A,
                                  pangulu_Smatrix *L,
                                  pangulu_Smatrix *U)
{
#ifdef GPU_OPEN
    pangulu_Smatrix_CUDA_memcpy_value_CSC(A, A);
    pangulu_Smatrix_CUDA_memcpy_value_CSC(L, L);
    pangulu_Smatrix_CUDA_memcpy_value_CSC(U, U);
#endif
    int_t *m = &(A->row);
    int_t *n = &(A->row);
    int_t *k = &(A->row);
    thmkl_dcsrmultcsr(m, n, k,
                      U->value_CSC, U->rowindex, U->columnpointer,
                      L->value_CSC, L->rowindex, L->columnpointer,
                      A->value_CSC, A->rowindex, A->columnpointer);
#ifdef GPU_OPEN
    pangulu_Smatrix_CUDA_memcpy_to_device_value_CSC(A, A);
#endif
}

#endif