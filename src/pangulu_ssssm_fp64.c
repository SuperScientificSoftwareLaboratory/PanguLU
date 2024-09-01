#include "pangulu_common.h"

pangulu_int64_t partition_key_val_pair(pangulu_int64_t *key, calculate_type *val, pangulu_int64_t length, pangulu_int64_t pivot_index)
{
    pangulu_int64_t i = 0;
    pangulu_int64_t small_length = pivot_index;

    pangulu_int64_t pivot = key[pivot_index];
    if (val != NULL)
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

void insert_sort_key_val_pair(pangulu_int64_t *key, calculate_type *val, pangulu_int64_t length)
{
    for (pangulu_int64_t i = 1; i < length; i++)
    {
        pangulu_int64_t tmp_key = key[i];
        calculate_type tmp_val = val[i];
        pangulu_int64_t j = i - 1;
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

void quick_sort_key_val_pair1(pangulu_int64_t *key, calculate_type *val, pangulu_int64_t length)
{
    if (length == 0 || length == 1)
        return;

    pangulu_int64_t sorted = 1;

    // get mid of three
    pangulu_int64_t first = 0;
    pangulu_int64_t last = length - 1;
    pangulu_int64_t mid = first + ((last - first) >> 1);
    // pangulu_int64_t pivot = mid;
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

    for (pangulu_int64_t i = 1; i < length; i++)
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
            pangulu_int64_t small_length = partition_key_val_pair(key, val, length, 0);
            quick_sort_key_val_pair1(key, val, small_length);
            quick_sort_key_val_pair1(&key[small_length + 1], &val[small_length + 1], length - small_length - 1);
        }
        else
        {
            insert_sort_key_val_pair(key, val, length);
        }
    }
}

void segmented_sum(calculate_type *input, pangulu_int64_t *bit_flag, pangulu_int64_t length)
{
    if (length == 0 || length == 1)
        return;

    for (pangulu_int64_t i = 0; i < length; i++)
    {
        if (bit_flag[i])
        {
            pangulu_int64_t j = i + 1;
            while (!bit_flag[j] && j < length)
            {
                input[i] += input[j];
                j++;
            }
        }
    }
}

void thmkl_dcsrmultcsr(pangulu_inblock_idx *m, pangulu_inblock_idx *n, pangulu_inblock_idx *k,
                       calculate_type *a, pangulu_inblock_idx *ja, pangulu_inblock_ptr *ia,
                       calculate_type *b, pangulu_inblock_idx *jb, pangulu_inblock_ptr *ib,
                       calculate_type *c, pangulu_inblock_idx *jc, pangulu_inblock_ptr *ic)
{

    if (*m <= Bound) // if C size < bound, use spa method
    {
#pragma omp parallel for num_threads(pangu_omp_num_threads)
        for (pangulu_int64_t iid = 0; iid < *m; iid++)
        {
            pangulu_int64_t len = (*k + 31) / 32;
            pangulu_uint32_t *mask = (pangulu_uint32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_uint32_t) * len);
            memset(mask, 0, sizeof(pangulu_uint32_t) * len);
            calculate_type *d_dense_row_value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, (*k) * sizeof(calculate_type));
            memset(d_dense_row_value, 0, (*k) * sizeof(calculate_type));

            for (pangulu_int64_t i = ia[iid] - ia[0]; i < ia[iid + 1] - ia[0]; i++)
            {
                pangulu_int64_t col = ja[i] - ia[0];
                for (pangulu_int64_t l = ib[col] - ib[0]; l < ib[col + 1] - ib[0]; l++)
                {
                    const pangulu_int64_t key = jb[l] - ib[0];
                    setbit(mask[key / 32], key % 32);
                    d_dense_row_value[key] += b[l] * a[i];
                }
            }

            pangulu_int64_t nnzr = ic[iid] - ic[0];
            for (pangulu_int64_t cid = 0; cid < *k; cid++)
            {
                if (getbit(mask[cid / 32], cid % 32) == 1)
                {
                    while (nnzr < ic[iid + 1] - ic[0] && jc[nnzr] < cid + ic[0])
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

        pangulu_int64_t *iCub = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, *m * sizeof(pangulu_int64_t));
        memset(iCub, 0, sizeof(pangulu_int64_t) * *m);

        for (pangulu_int64_t i = 0; i < *m; i++)
        {
            for (pangulu_int64_t j = ia[i] - ia[0]; j < ia[i + 1] - ia[0]; j++)
            {
                pangulu_int64_t rowAtop = ja[j] - ia[0];
                iCub[i] += ib[rowAtop + 1] - ib[rowAtop];
            }
        }
#pragma omp parallel for num_threads(pangu_omp_num_threads)
        for (pangulu_int64_t i = 0; i < *m; i++)
        {
            pangulu_int64_t rowid = i;
            pangulu_int64_t rowsize = iCub[rowid];
            if (rowsize == 0)
                continue;

            if (rowsize <= 64)
            {
                // esc
                pangulu_int64_t *jCub = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, rowsize * sizeof(pangulu_int64_t));
                calculate_type *valCub = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, rowsize * sizeof(calculate_type));
                pangulu_int64_t *d_flagCub = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, rowsize * sizeof(pangulu_int64_t));
                memset(jCub, 0, rowsize * sizeof(pangulu_int64_t));
                memset(valCub, 0, rowsize * sizeof(calculate_type));
                memset(d_flagCub, 0, rowsize * sizeof(pangulu_int64_t));

                pangulu_int64_t incr = 0;
                for (pangulu_int64_t l = ia[rowid] - ia[0]; l < ia[rowid + 1] - ia[0]; l++)
                {
                    pangulu_int64_t rowB = ja[l] - ia[0];
                    calculate_type val = a[l];
                    for (pangulu_int64_t ik = ib[rowB] - ib[0]; ik < ib[rowB + 1] - ib[0]; ik++)
                    {
                        jCub[incr] = jb[ik];
                        valCub[incr] = val * b[ik];
                        incr++;
                    }
                }

                // sort
                quick_sort_key_val_pair1(jCub, valCub, rowsize);

                // compress
                d_flagCub[0] = 1;
                for (pangulu_int64_t idx = 0; idx < rowsize - 1; idx++)
                {
                    d_flagCub[idx + 1] = jCub[idx + 1] == jCub[idx] ? 0 : 1;
                }
                segmented_sum(valCub, d_flagCub, rowsize);

                pangulu_int64_t incrn = ic[rowid] - ic[0];
                for (pangulu_int64_t idx = 0; idx < rowsize; idx++)
                {
                    if (d_flagCub[idx] == 1)
                    {
                        while (incrn < ic[rowid + 1] - ic[0] && jc[incrn] < jCub[idx])
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
            else if (rowsize <= 4096)
            {
                // hash
                pangulu_int64_t hashsize_full_reg = (ic[rowid + 1] - ic[rowid]) / 0.75;
                pangulu_int64_t *tmpHashtable = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, hashsize_full_reg * sizeof(pangulu_int64_t));
                memset(tmpHashtable, -1, sizeof(pangulu_int64_t) * hashsize_full_reg);
                calculate_type *tmpValue = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, hashsize_full_reg * sizeof(calculate_type));
                memset(tmpValue, 0, sizeof(calculate_type) * hashsize_full_reg);
                for (pangulu_int64_t blkj = ia[rowid] - ia[0]; blkj < ia[rowid + 1] - ia[0]; blkj++)
                {
                    pangulu_int64_t col = ja[blkj] - ia[0];
                    for (pangulu_int64_t l = ib[col] - ib[0]; l < ib[col + 1] - ib[0]; l++)
                    {
                        const pangulu_int64_t key = jb[l] - ib[0];
                        pangulu_int64_t hashadr = (key * 107) % hashsize_full_reg;
                        // pangulu_int64_t hashadr = key % hashsize_full_reg;
                        while (1)
                        {
                            const pangulu_int64_t keyexist = tmpHashtable[hashadr];
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

                pangulu_int64_t cptr = ic[rowid] - ic[0];
                for (pangulu_int64_t ik = 0; ik < hashsize_full_reg; ik++)
                {
                    if (tmpHashtable[ik] != -1)
                    {
                        while (cptr < ic[rowid + 1] - ic[0] && jc[cptr] < tmpHashtable[ik] + ib[0])
                        {
                            cptr++;
                        }
                        c[cptr] -= tmpValue[ik];
                    }
                }
                pangulu_free(__FILE__, __LINE__, tmpHashtable);
                pangulu_free(__FILE__, __LINE__, tmpValue);
            }
            else
            {
                // spa
                pangulu_int64_t len = (*k + 31) / 32;
                pangulu_int64_t *mask = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * len);
                memset(mask, 0, sizeof(pangulu_int64_t) * len);
                calculate_type *d_dense_row_value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, (*k) * sizeof(calculate_type));
                memset(d_dense_row_value, 0, (*k) * sizeof(calculate_type));

                for (pangulu_int64_t offsetA = ia[rowid] - ia[0]; offsetA < ia[rowid + 1] - ia[0]; offsetA++)
                {
                    pangulu_int64_t col = ja[offsetA] - ia[0];
                    for (pangulu_int64_t l = ib[col] - ib[0]; l < ib[col + 1] - ib[0]; l++)
                    {
                        const pangulu_int64_t key = jb[l] - ib[0];
                        setbit(mask[key / 32], key % 32);
                        d_dense_row_value[key] += b[l] * a[offsetA];
                    }
                }

                pangulu_int64_t nnzr = ic[rowid] - ic[0];
                for (pangulu_int64_t cid = 0; cid < *k; cid++)
                {
                    if (getbit(mask[cid / 32], cid % 32) == 1)
                    {
                        while (nnzr < ic[rowid + 1] - ic[0] && jc[nnzr] < cid + ic[0])
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
    pangulu_int64_t curSize = 0;
    curSize = ends - begins;
    pangulu_int64_t half;
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

void spmm(
    pangulu_int64_t m, pangulu_int64_t k, pangulu_int64_t n,
    const pangulu_int64_t *A_csrOffsets,
    const pangulu_inblock_idx *A_columns,
    const calculate_type *A_values,
    const pangulu_int64_t *B_csrOffsets,
    const pangulu_inblock_idx *B_columns,
    const calculate_type *B_values,
    const pangulu_int64_t *C_csrOffsets,
    const pangulu_inblock_idx *C_columns,
    calculate_type *C_values)
{
    for (pangulu_inblock_idx rowA = 0; rowA < m; ++rowA)
    {
        for (pangulu_int64_t i = A_csrOffsets[rowA]; i < A_csrOffsets[rowA + 1]; ++i)
        {
            pangulu_inblock_idx rowB = A_columns[i];
            for (pangulu_int64_t j = B_csrOffsets[rowB]; j < B_csrOffsets[rowB + 1]; ++j)
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
int binary_search_right_boundary(const pangulu_int64_t *data,
                                 const pangulu_int64_t key_input,
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
void cscmultcsc_dense(pangulu_smatrix *a,
                      pangulu_smatrix *l,
                      pangulu_smatrix *u)
{
    int n = a->row;
    int omp_threads_num = pangu_omp_num_threads;
    ssssm_col_ops_u[0] = 0;
    ssssm_ops_pointer[0] = 0;

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int i = 0; i < n; i++)
    {
        for (int j = a->columnpointer[i]; j < a->columnpointer[i + 1]; j++)
        {
            int row = a->rowindex[j];
            temp_a_value[i * n + row] = a->value_csc[j]; // tranform csc to dense,only value
        }
    }

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int i = 0; i < n; i++)
    {
        int ops = 0;
        for (int k = u->columnpointer[i]; k < u->columnpointer[i + 1]; k++)
        {
            int L_col = u->rowindex[k];
            ops += (l->columnpointer[L_col + 1] - l->columnpointer[L_col]);
        }
        ssssm_col_ops_u[i] = ops;
    }
    for (int i = 0; i < n; i++)
    {
        ssssm_col_ops_u[i + 1] = ssssm_col_ops_u[i + 1] + ssssm_col_ops_u[i];
    }
    pangulu_int64_t ops_div = (ssssm_col_ops_u[n] - 1) / omp_threads_num + 1;
    pangulu_int64_t boundary;
#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int i = 1; i <= omp_threads_num; i++)
    {
        boundary = i * ops_div;
        ssssm_ops_pointer[i] = binary_search_right_boundary(ssssm_col_ops_u, boundary, 0, n) - 1;
    }

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int t = 0; t < omp_threads_num; t++)
    {
        for (int j = ssssm_ops_pointer[t]; j < ssssm_ops_pointer[t + 1]; j++)
        {
            for (int pointer_u = u->columnpointer[j]; pointer_u < u->columnpointer[j + 1]; pointer_u++)
            {
                int i = u->rowindex[pointer_u]; // ith row
                calculate_type value_U = u->value_csc[pointer_u];
                calculate_type *A_value = temp_a_value + n * j; // point to jth column
                for (int pointer_l = l->columnpointer[i]; pointer_l < l->columnpointer[i + 1]; pointer_l++)
                {                                                    // ith column Lki Uij Akj
                    int k = l->rowindex[pointer_l];                  // kth row
                    A_value[k] -= value_U * l->value_csc[pointer_l]; // schur
                }
            }
        }
    }
#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int i = 0; i < n; i++)
    {
        calculate_type *A_value = temp_a_value + i * n;
        for (int j = a->columnpointer[i]; j < a->columnpointer[i + 1]; j++)
        {
            int row = a->rowindex[j];
            a->value_csc[j] = A_value[row];
        }
    }
}

void openblas_dgemm_reprecess(pangulu_smatrix *a,
                              pangulu_smatrix *l,
                              pangulu_smatrix *u)
{
    int n = a->row;
    int L_row_num = 0, U_col_num = 0, LU_rc_num = 0;
    int *blas_dense_hash_col_LU = NULL;
    int *blas_dense_hash_row_L = NULL;

    blas_dense_hash_col_LU = ssssm_hash_lu;
    blas_dense_hash_row_L = ssssm_hash_l_row;
    for (int i = 0; i < n; i++)
    {
        blas_dense_hash_row_L[i] = -1;
    }
    for (int i = 0; i < n; i++)
    {
        if (l->columnpointer[i + 1] > l->columnpointer[i])
        {
            blas_dense_hash_col_LU[i] = LU_rc_num;
            LU_rc_num++;
        }
    }
    for (int i = 0; i < n; i++)
    {
        int col_begin = l->columnpointer[i];
        int col_end = l->columnpointer[i + 1];
        for (int j = col_begin; j < col_end; j++)
        {
            int L_row = l->rowindex[j];
            if (blas_dense_hash_row_L[L_row] == -1) // 如果当前行未标记
            {
                blas_dense_hash_row_L[L_row] = L_row_num;
                L_row_num++;
            }
        }
    }

    for (int i = 0; i < n; i++)
    {
        int begin = u->columnpointer[i];
        int end = u->columnpointer[i + 1];
        if (end > begin)
        {
            calculate_type *U_temp_value = ssssm_u_value + U_col_num * LU_rc_num; // u column based

            for (int j = begin; j < end; j++)
            {
                int U_row = u->rowindex[j];
                if (l->columnpointer[U_row + 1] > l->columnpointer[U_row] > 0)
                { // only store the remain data
                    U_temp_value[blas_dense_hash_col_LU[U_row]] = u->value_csc[j];
                }
            }
            ssssm_hash_u_col[U_col_num] = i;
            U_col_num++;
        }
    }
#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int i = 0; i < n; i++)
    {
        int col_begin = l->columnpointer[i];
        int col_end = l->columnpointer[i + 1];
        calculate_type *temp_data = ssssm_l_value + L_row_num * blas_dense_hash_col_LU[i];
        for (int j = col_begin; j < col_end; j++)
        {
            temp_data[blas_dense_hash_row_L[l->rowindex[j]]] = l->value_csc[j];
        }
    }
    int m = L_row_num;
    int k = LU_rc_num;
    n = U_col_num;

    calculate_type alpha = 1.0, beta = 0.0;
#if defined(CALCULATE_TYPE_CR64)
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, ssssm_l_value, m, ssssm_u_value, k, &beta, temp_a_value, m);
#elif defined(CALCULATE_TYPE_R64)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, ssssm_l_value, m, ssssm_u_value, k, beta, temp_a_value, m);
#elif defined(CALCULATE_TYPE_R32)
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, ssssm_l_value, m, ssssm_u_value, k, beta, temp_a_value, m);
#elif defined(CALCULATE_TYPE_CR32)
    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, ssssm_l_value, m, ssssm_u_value, k, &beta, temp_a_value, m);
#else
#error[PanguLU Compile Error] Unsupported value type for BLAS library.
#endif

    memset(ssssm_l_value, 0, sizeof(calculate_type) * m * k);
    memset(ssssm_u_value, 0, sizeof(calculate_type) * k * n);

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int i = 0; i < U_col_num; i++)
    {
        int col_num = ssssm_hash_u_col[i];
        calculate_type *temp_value = temp_a_value + i * m;
        int j_begin = a->columnpointer[col_num];
        int j_end = a->columnpointer[col_num + 1];
        for (int j = j_begin; j < j_end; j++)
        {
            int row = a->rowindex[j];
            if (blas_dense_hash_row_L[row] != -1)
            {
                int row_index = blas_dense_hash_row_L[row];
                a->value_csc[j] -= temp_value[row_index];
            }
        }
    }
}

void pangulu_ssssm_fp64(pangulu_smatrix *a,
                        pangulu_smatrix *l,
                        pangulu_smatrix *u)
{

#ifdef OUTPUT_MATRICES
    char out_name_A[512];
    char out_name_B[512];
    char out_name_C[512];
    sprintf(out_name_A, "%s/%s/%d%s", OUTPUT_FILE, "ssssm", ssssm_number, "_ssssm_A.cbd");
    sprintf(out_name_B, "%s/%s/%d%s", OUTPUT_FILE, "ssssm", ssssm_number, "_ssssm_B.cbd");
    sprintf(out_name_C, "%s/%s/%d%s", OUTPUT_FILE, "ssssm", ssssm_number, "_ssssm_C.cbd");
    pangulu_binary_write_csc_pangulu_smatrix(l, out_name_A);
    pangulu_binary_write_csc_pangulu_smatrix(u, out_name_B);
    pangulu_binary_write_csc_pangulu_smatrix(a, out_name_C);
    ssssm_number++;
#endif
    openblas_dgemm_reprecess(a, l, u);
}

void pangulu_ssssm_interface_c_v1(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *u)
{
#ifdef GPU_OPEN
    pangulu_smatrix_cuda_memcpy_value_csc(a, a);
    pangulu_smatrix_cuda_memcpy_value_csc(l, l);
    pangulu_smatrix_cuda_memcpy_value_csc(u, u);
#endif
    cscmultcsc_dense(a, l, u);
#ifdef GPU_OPEN
    pangulu_smatrix_cuda_memcpy_to_device_value_csc(a, a);
#endif
}
void pangulu_ssssm_interface_c_v2(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *u)
{
#ifdef GPU_OPEN
    pangulu_smatrix_cuda_memcpy_value_csc(a, a);
    pangulu_smatrix_cuda_memcpy_value_csc(l, l);
    pangulu_smatrix_cuda_memcpy_value_csc(u, u);
#endif
    pangulu_inblock_idx *m = &(a->row);
    pangulu_inblock_idx *n = &(a->row);
    pangulu_inblock_idx *k = &(a->row);
    thmkl_dcsrmultcsr(m, n, k,
                      u->value_csc, u->rowindex, u->columnpointer,
                      l->value_csc, l->rowindex, l->columnpointer,
                      a->value_csc, a->rowindex, a->columnpointer);
#ifdef GPU_OPEN
    pangulu_smatrix_cuda_memcpy_to_device_value_csc(a, a);
#endif
}
