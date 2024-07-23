#ifndef PANGULU_SSSSM_FP64_H
#define PANGULU_SSSSM_FP64_H

#include "pangulu_common.h"
#include "pangulu_utils.h"
#include "cblas.h"

typedef unsigned int sflu_uint;

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

void thmkl_dcsrmultcsr(const int *m, const int *n, const int *k,
                       calculate_type *a, pangulu_inblock_idx *ja, pangulu_inblock_ptr *ia,
                       calculate_type *b, pangulu_inblock_idx *jb, pangulu_inblock_ptr *ib,
                       calculate_type *c, pangulu_inblock_idx *jc, pangulu_inblock_ptr *ic)
{

    if (*m <= Bound) // if C size < bound, use spa method
    {
#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
        for (int_t iid = 0; iid < *m; iid++)
        {
            int_t len = (*k + 31) / 32;
            sflu_uint *mask = (sflu_uint *)pangulu_malloc(__FILE__, __LINE__, sizeof(sflu_uint) * len);
            memset(mask, 0, sizeof(sflu_uint) * len);
            calculate_type *d_dense_row_value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, (*k) * sizeof(calculate_type));
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

            pangulu_free(__FILE__, __LINE__, mask);
            pangulu_free(__FILE__, __LINE__, d_dense_row_value);
        }
    }
    else // if C size > bound, use bin method
    {

        int_t *iCub = (int_t *)pangulu_malloc(__FILE__, __LINE__, *m * sizeof(int_t));
        memset(iCub, 0, sizeof(int_t) * *m);

        for (int_t i = 0; i < *m; i++)
        {
            for (int_t j = ia[i] - ia[0]; j < ia[i + 1] - ia[0]; j++)
            {
                int_t rowAtop = ja[j] - ia[0];
                iCub[i] += ib[rowAtop + 1] - ib[rowAtop];
            }
        }
#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
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
                int_t *jCub = (int_t *)pangulu_malloc(__FILE__, __LINE__, rowsize * sizeof(int_t));
                calculate_type *valCub = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, rowsize * sizeof(calculate_type));
                int_t *d_flagCub = (int_t *)pangulu_malloc(__FILE__, __LINE__, rowsize * sizeof(int_t));
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

                pangulu_free(__FILE__, __LINE__, jCub);
                pangulu_free(__FILE__, __LINE__, valCub);
                pangulu_free(__FILE__, __LINE__, d_flagCub);
            }
            else if (rowsize > 64 && rowsize <= 4096)
            {
                // hash
                int_t hashsize_full_reg = (ic[rowid + 1] - ic[rowid]) / 0.75;
                int_t *tmpHashtable = (int_t *)pangulu_malloc(__FILE__, __LINE__, hashsize_full_reg * sizeof(int_t));
                memset(tmpHashtable, -1, sizeof(int_t) * hashsize_full_reg);
                calculate_type *tmpValue = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, hashsize_full_reg * sizeof(calculate_type));
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
                pangulu_free(__FILE__, __LINE__, tmpHashtable);
                pangulu_free(__FILE__, __LINE__, tmpValue);
            }
            else if (rowsize > 4096)
            {
                // spa
                if (rowsize == 0)
                    continue;
                int_t len = (*k + 31) / 32;
                int_t *mask = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * len);
                memset(mask, 0, sizeof(int_t) * len);
                calculate_type *d_dense_row_value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, (*k) * sizeof(calculate_type));
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

                pangulu_free(__FILE__, __LINE__, mask);
                pangulu_free(__FILE__, __LINE__, d_dense_row_value);
            }
        }
        pangulu_free(__FILE__, __LINE__, iCub);
    }
}

const pangulu_inblock_idx *lower_bound(const pangulu_inblock_idx *begins, const pangulu_inblock_idx *ends, pangulu_inblock_idx key)
{
    int_t curSize = 0;
    curSize = ends - begins;
    int_t half;
    const pangulu_inblock_idx *middle;
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
    const pangulu_inblock_idx *A_columns,
    const calculate_type *A_values,
    const int_t *B_csrOffsets,
    const pangulu_inblock_idx *B_columns,
    const calculate_type *B_values,
    const int_t *C_csrOffsets,
    const pangulu_inblock_idx *C_columns,
    calculate_type *C_values)
{
    for (pangulu_inblock_idx rowA = 0; rowA < m; ++rowA)
    {
        for (int_t i = A_csrOffsets[rowA]; i < A_csrOffsets[rowA + 1]; ++i)
        {
            pangulu_inblock_idx rowB = A_columns[i];
            for (int_t j = B_csrOffsets[rowB]; j < B_csrOffsets[rowB + 1]; ++j)
            {
                pangulu_inblock_idx colB = B_columns[j];
                /// find locate of C at [rowA,colB]
                pangulu_inblock_idx loc = (pangulu_inblock_idx)(lower_bound(
                                            C_columns + C_csrOffsets[rowA],
                                            C_columns + C_csrOffsets[rowA + 1],
                                            colB) -
                                        C_columns);
                if (C_columns[loc] == colB)
                {
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

void openblas_dgemm_reprecess(pangulu_Smatrix *A,
                              pangulu_Smatrix *L,
                              pangulu_Smatrix *U)
{
    int omp_threads_num = 1;
    #pragma omp parallel num_threads(PANGU_OMP_NUM_THREADS)
    {
        omp_threads_num = omp_get_num_threads() == 0 ? omp_threads_num : omp_get_num_threads();
    }
    int n = A->row;
    int L_row_num = 0, U_col_num = 0, LU_rc_num = 0;
    char *blas_dense_flag_col_LU = NULL;
    int *blas_dense_hash_col_LU = NULL;
    char *blas_dense_flag_row_L = NULL;
    int *blas_dense_hash_row_L = NULL;
    if (L->zip_flag == 0)
    {
        blas_dense_flag_col_LU = SSSSM_flag_LU + zip_cur_id * n;
        blas_dense_hash_col_LU = SSSSM_hash_LU + zip_cur_id * n;
        blas_dense_flag_row_L = SSSSM_flag_L_row + zip_cur_id * n;
        blas_dense_hash_row_L = SSSSM_hash_L_row + zip_cur_id * n;
        // memset(blas_dense_flag_row_L, 0, sizeof(char) * n);
        // memset(blas_dense_flag_col_LU, 0, sizeof(char) * n);
        for (int i = 0; i < n; i++)
        {
            if (L->columnpointer[i + 1] > L->columnpointer[i])
            {
                blas_dense_flag_col_LU[i] = 1;
                blas_dense_hash_col_LU[i] = LU_rc_num;
                LU_rc_num++;
            }
            int col_begin = L->columnpointer[i];
            int col_end = L->columnpointer[i + 1];
            for (int j = col_begin; j < col_end; j++)
            {
                int L_row = L->rowindex[j];
                if (blas_dense_flag_row_L[L_row] != 1) // 如果当前行未标记
                {
                    blas_dense_hash_row_L[L_row] = L_row_num;
                    L_row_num++;
                    blas_dense_flag_row_L[L_row] = 1;
                }
            }
        }
        zip_rows[zip_cur_id] = L_row_num;
        zip_cols[zip_cur_id] = LU_rc_num;
        L->zip_id = zip_cur_id;
        L->zip_flag = 1;
        zip_cur_id++;
        zip_cur_id %= zip_max_id;
    }
    else
    {
        blas_dense_flag_col_LU = SSSSM_flag_LU + L->zip_id * n;
        blas_dense_hash_col_LU = SSSSM_hash_LU + L->zip_id * n;
        blas_dense_flag_row_L = SSSSM_flag_L_row + L->zip_id * n;
        blas_dense_hash_row_L = SSSSM_hash_L_row + L->zip_id * n;
        L_row_num = zip_rows[L->zip_id];
        LU_rc_num = zip_cols[L->zip_id];
    }

    for (int i = 0; i < n; i++)
    {
        int begin = U->columnpointer[i];
        int end = U->columnpointer[i + 1];
        if (end > begin)
        {
            calculate_type *U_temp_value = SSSSM_U_value + U_col_num * LU_rc_num; // U column based

            for (int j = begin; j < end; j++)
            {
                int U_row = U->rowindex[j];
                if (blas_dense_flag_col_LU[U_row] == 1)
                { // only store the remain data
                    U_temp_value[blas_dense_hash_col_LU[U_row]] = U->value_CSC[j];
                }
            }
            SSSSM_hash_U_col[U_col_num] = i;
            U_col_num++;
        }
    }
#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
    for (int i = 0; i < n; i++)
    {
        int col_begin = L->columnpointer[i];
        int col_end = L->columnpointer[i + 1];
        calculate_type *temp_data = SSSSM_L_value + L_row_num * blas_dense_hash_col_LU[i];
        for (int j = col_begin; j < col_end; j++)
        {
            temp_data[blas_dense_hash_row_L[L->rowindex[j]]] = L->value_CSC[j];
        }
    }
    int m = L_row_num;
    int k = LU_rc_num;
    n = U_col_num;
    

    calculate_type alpha = 1.0, beta = 0.0;
    #if defined(CALCULATE_TYPE_CR64)
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, SSSSM_L_value, m, SSSSM_U_value, k, &beta, TEMP_A_value, m);
    #elif defined(CALCULATE_TYPE_R64)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, SSSSM_L_value, m, SSSSM_U_value, k, beta, TEMP_A_value, m);
    #else
    #error [PanguLU Compile Error] Unsupported value type for BLAS library.
    #endif
    // cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, SSSSM_L_value, m, SSSSM_U_value, k, 0.0, TEMP_A_value, m);


    memset(SSSSM_L_value, 0, sizeof(calculate_type) * m * k);
    memset(SSSSM_U_value, 0, sizeof(calculate_type) * k * n);

#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
    for (int i = 0; i < U_col_num; i++)
    {
        int col_num = SSSSM_hash_U_col[i];
        calculate_type *temp_value = TEMP_A_value + i * m;
        int j_begin = A->columnpointer[col_num];
        int j_end = A->columnpointer[col_num + 1];
        for (int j = j_begin; j < j_end; j++)
        {
            int row = A->rowindex[j];
            if (blas_dense_flag_row_L[row] == 1)
            {
                int row_index = blas_dense_hash_row_L[row];
                A->value_CSC[j] -= temp_value[row_index];
            }
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
    openblas_dgemm_reprecess(A, L, U);
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
    int *m = &(A->row);
    int *n = &(A->row);
    int *k = &(A->row);
    thmkl_dcsrmultcsr(m, n, k,
                      U->value_CSC, U->rowindex, U->columnpointer,
                      L->value_CSC, L->rowindex, L->columnpointer,
                      A->value_CSC, A->rowindex, A->columnpointer);
#ifdef GPU_OPEN
    pangulu_Smatrix_CUDA_memcpy_to_device_value_CSC(A, A);
#endif
}

#endif