#include "pangulu_common.h"

void pangulu_sflu_fp64(pangulu_smatrix *a,
                       pangulu_smatrix *l,
                       pangulu_smatrix *u)
{
    pangulu_int64_t n = a->row;
    pangulu_int64_t index = 0;\
    for (pangulu_int64_t i = 0; i < n; i++) // for each column
    {
        if (a->columnpointer[i] == a->columnpointer[i + 1])
        {
            continue;
        }
        index = binarysearch_inblock_idx(a->columnpointer[i], a->columnpointer[i + 1], i, a->rowindex);
        calculate_type pivot;
        pivot = a->value_csc[index];

        int L_index = l->columnpointer[i]; // restore l value
        l->value_csc[L_index++] = 1.0;

        for (pangulu_int64_t j = index + 1; j < a->columnpointer[i + 1]; j++) // i th colnum j th row  (j,i) in l
        {
            if (fabs(pivot) > ERROR)
            {
            }
            else
            {
                pivot = ERROR;
            }
            index = binarysearch_inblock_idx(a->columnpointer[i], a->columnpointer[i + 1], a->rowindex[j], a->rowindex); // index->(j,i)

            calculate_type scale = a->value_csc[index] / pivot;

            // Binary search return ith colnum jth row 's  index   of  a->value_csc
            a->value_csc[index] = scale;

            l->value_csc[L_index++] = scale; // l value
            for (pangulu_int64_t x = i + 1; x < n; x++)
            {
                pangulu_int64_t nonzero1 = binarysearch_inblock_idx(a->columnpointer[x], a->columnpointer[x + 1], a->rowindex[j], a->rowindex); //(j,x)
                pangulu_int64_t nonzero2 = binarysearch_inblock_idx(a->columnpointer[x], a->columnpointer[x + 1], i, a->rowindex);              //(i,x)

                if (nonzero1 != -1 && nonzero2 != -1)
                {
                    a->value_csc[nonzero1] -= scale * a->value_csc[nonzero2];
                }
            }
        }
    }
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        for (pangulu_int64_t j = u->columnpointer[i], k = a->columnpointer[i]; j < u->columnpointer[i + 1]; j++, k++)
        {
            u->value_csc[j] = a->value_csc[k];
            if (i == u->rowindex[j] && fabs(u->value_csc[j]) < ERROR)
            {
                u->value_csc[j] = 1.0;
            }
        }
    }
}

void pangulu_sflu_fp64_dense_col(pangulu_smatrix *a,
                                 pangulu_smatrix *l,
                                 pangulu_smatrix *u)
{
    // char *data = getenv("OMP_NUM_THREADS");
    // int num = data == NULL ? 1 : atoi(data);
    // int omp_threads_num = num > 0 ? num : 1;

    pangulu_int64_t n = a->row;
    calculate_type *A_value = (calculate_type *)calloc(n * n, sizeof(calculate_type));
    char *A_flag = (char *)calloc(n * n, sizeof(char));

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int i = 0; i < n; i++)
    {
        for (int j = a->columnpointer[i]; j < a->columnpointer[i + 1]; j++)
        {
            int row = a->rowindex[j];
            A_value[i * n + row] = a->value_csc[j]; // tranform csc to dense,only value
            A_flag[i * n + row] = 1;
        }
    }
    for (int i = 0; i < n; i++)
    {
        if (a->columnpointer[i] == a->columnpointer[i + 1])
        {
            continue;
        }
        calculate_type pivot = A_value[i * n + i]; // diag value
        int L_index = l->columnpointer[i];         // restore l value

        l->value_csc[L_index++] = 1.0;

#pragma omp parallel for num_threads(pangu_omp_num_threads)
        for (pangulu_int64_t j = i + 1; j < n; j++) // i th colnum j th row  (j,i) in l
        {
            if (A_flag[i * n + j] == 0)
            {
                continue;
            }
            if (fabs(pivot) > ERROR)
            {
            }
            else
            {
                pivot = ERROR;
            }
            calculate_type scale = A_value[i * n + j] / pivot;

            A_value[i * n + j] = scale;

            // l->value_csc[L_index++] = scale; // l value

            for (pangulu_int64_t x = i + 1; x < n; x++)
            {
                if (A_flag[x * n + j] != 0)
                {
                    A_value[x * n + j] -= scale * A_value[x * n + i];
                }
            }
        }
    }
    // #pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int i = 0; i < n; i++)
    {
        int L_index = l->columnpointer[i] + 1;
        for (int j = i + 1; j < n; j++)
        {
            if (A_flag[i * n + j] != 0)
            {
                l->value_csc[L_index++] = A_value[i * n + j];
            }
        }
        int U_index = u->columnpointer[i]; // restore u value
        for (int k = 0; k <= i; k++)
        {
            if (A_flag[i * n + k] != 0)
            {
                u->value_csc[U_index++] = A_value[i * n + k];
            }
        }
    }

    pangulu_free(__FILE__, __LINE__, A_value);
}

double wtc_get_time(struct timeval time_start, struct timeval time_end)
{
    return ((time_end.tv_sec - time_start.tv_sec) * 1000.0 + (time_end.tv_usec - time_start.tv_usec) / 1000.0);
}

// void pangulu_sflu_fp64_dense_row_purge(pangulu_smatrix *a,
//                                        pangulu_smatrix *l,
//                                        pangulu_smatrix *u)
// {

//     double time_calloc = 0, time_s_to_d = 0, time_lu = 0, time_d_to_s = 0;
//     pangulu_int64_t n = a->row;
//     int nnz = a->columnpointer[n] - a->columnpointer[0];
//     struct timeval tstart, tend;

//     gettimeofday(&tstart, NULL);

//     char *TEMP_A_flag = (char *)calloc(n * n, sizeof(char));
//     int *a_rowPointer = (int *)pangulu_malloc(__FILE__, __LINE__, sizeof(int) * (n + 1));
//     int *a_colIndex = (int *)pangulu_malloc(__FILE__, __LINE__, sizeof(int) * nnz);

//     int *getrf_diagIndex_csr = (int *)pangulu_malloc(__FILE__, __LINE__, sizeof(int) * n);
//     int *getrf_diagIndex_csc = (int *)pangulu_malloc(__FILE__, __LINE__, sizeof(int) * n);
//     gettimeofday(&tend, NULL);
//     time_calloc += wtc_get_time(tstart, tend);

//     gettimeofday(&tstart, NULL);

// #pragma omp parallel for
//     for (int i = 0; i < n; i++)
//     {
//         for (int j = a->columnpointer[i]; j < a->columnpointer[i + 1]; j++)
//         {
//             int row = a->rowindex[j];
//             if (row == i)
//             {
//                 getrf_diagIndex_csc[i] = j;
//             }
//             temp_a_value[row * n + i] = a->value_csc[j]; // tranform csc to dense,only value
//             TEMP_A_flag[row * n + i] = 1;
//         }
//     }
//     // csr data

//     a_rowPointer[0] = 0;
//     int data_index = 0;
//     for (int i = 0; i < n; i++)
//     {
//         int row_num = 0;
//         for (int j = 0; j < n; j++)
//         {
//             if (TEMP_A_flag[i * n + j] == 1)
//             {
//                 if (i == j)
//                 {
//                     getrf_diagIndex_csr[i] = data_index;
//                 }
//                 a_colIndex[data_index++] = j;
//                 row_num++;
//             }
//         }
//         a_rowPointer[i + 1] = a_rowPointer[i] + row_num;
//     }
//     gettimeofday(&tend, NULL);
//     time_s_to_d += wtc_get_time(tstart, tend);
//     gettimeofday(&tstart, NULL);

//     for (int i = 0; i < n; i++)
//     {
//         if (a->columnpointer[i] == a->columnpointer[i + 1])
//         {
//             continue;
//         }
//         calculate_type pivot = temp_a_value[i * n + i]; // diag value
//         int L_index = l->columnpointer[i];              // restore l value

//         l->value_csc[L_index++] = 1.0;
// #pragma omp parallel for
//         for (int p = getrf_diagIndex_csc[i] + 1; p < a->columnpointer[i + 1]; p++)
//         {

//             int j = a->rowindex[p];
//             calculate_type scale = temp_a_value[j * n + i] / pivot;

//             temp_a_value[j * n + i] = scale;

//             calculate_type *temp_value = temp_a_value + j * n;

//             calculate_type *x_value = temp_a_value + i * n;

//             for (int k = getrf_diagIndex_csr[i] + 1; k < a_rowPointer[i + 1]; k++)
//             {
//                 int index_k = a_colIndex[k];
//                 temp_value[index_k] -= scale * x_value[index_k];
//             }
//         }
//     }

//     gettimeofday(&tend, NULL);
//     time_lu += wtc_get_time(tstart, tend);

//     gettimeofday(&tstart, NULL);
// #pragma omp parallel for
//     for (int i = 0; i < n; i++)
//     {
//         for (int j = u->columnpointer[i]; j < u->columnpointer[i + 1]; j++)
//         {
//             int index = u->rowindex[j];
//             u->value_csc[j] = temp_a_value[index * n + i];
//         }
//         for (int j = l->columnpointer[i] + 1; j < l->columnpointer[i + 1]; j++)
//         {
//             int index = l->rowindex[j];
//             l->value_csc[j] = temp_a_value[index * n + i];
//         }
//     }

//     gettimeofday(&tend, NULL);
//     time_d_to_s += wtc_get_time(tstart, tend);

//     pangulu_free(__FILE__, __LINE__, a_rowPointer);
//     pangulu_free(__FILE__, __LINE__, a_colIndex);
//     pangulu_free(__FILE__, __LINE__, getrf_diagIndex_csr);
//     pangulu_free(__FILE__, __LINE__, getrf_diagIndex_csc);
//     pangulu_free(__FILE__, __LINE__, TEMP_A_flag);
// }

void pangulu_sflu_fp64_2(pangulu_smatrix *a,
                         pangulu_smatrix *l,
                         pangulu_smatrix *u)
{
    pangulu_int64_t n = a->row;
    pangulu_int64_t index = 0;
    for (pangulu_int64_t i = 0; i < n; i++) // for each column
    {
        if (a->columnpointer[i] == a->columnpointer[i + 1])
        {
            continue;
        }
        index = binarysearch_inblock_idx(a->columnpointer[i], a->columnpointer[i + 1], i, a->rowindex);
        calculate_type pivot;
        pivot = a->value_csc[index];

        for (pangulu_int64_t j = index + 1; j < a->columnpointer[i + 1]; j++) // i th colnum j th row  (j,i) in l
        {
            // pangulu_int64_t row = a->rowindex[j];
            if (fabs(pivot) > ERROR)
            {
            }
            else
            {
                pivot = ERROR;
            }
            index = binarysearch_inblock_idx(a->columnpointer[i], a->columnpointer[i + 1], a->rowindex[j], a->rowindex); // index->(j,i)
            calculate_type scale = a->value_csc[index] / pivot;

            // Binary search return ith colnum jth row 's  index   of  a->value_csc
            a->value_csc[index] = scale;
            for (pangulu_int64_t x = i + 1; x < n; x++)
            {
                pangulu_int64_t nonzero1 = binarysearch_inblock_idx(a->columnpointer[x], a->columnpointer[x + 1], a->rowindex[j], a->rowindex); //(j,x)
                pangulu_int64_t nonzero2 = binarysearch_inblock_idx(a->columnpointer[x], a->columnpointer[x + 1], i, a->rowindex);              //(i,x)

                if (nonzero1 != -1 && nonzero2 != -1)
                {
                    a->value_csc[nonzero1] -= scale * a->value_csc[nonzero2];
                }
            }
        }
    }

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        index = binarysearch_inblock_idx(a->columnpointer[i], a->columnpointer[i + 1], i, a->rowindex);
        for (pangulu_int64_t j = l->columnpointer[i], k = index; j < l->columnpointer[i + 1]; j++, k++)
        {
            l->value_csc[j] = a->value_csc[k];
            if (i == l->rowindex[j])
            {
                l->value_csc[j] = 1.0;
            }
        }
    }

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        for (pangulu_int64_t j = u->columnpointer[i], k = a->columnpointer[i]; j < u->columnpointer[i + 1]; j++, k++)
        {
            u->value_csc[j] = a->value_csc[k];
            if (i == u->rowindex[j] && fabs(u->value_csc[j]) < ERROR)
            {
                u->value_csc[j] = 1.0;
            }
        }
    }
}
#ifndef GPU_OPEN
void pangulu_sflu_omp_fp64(pangulu_smatrix *a,
                           pangulu_smatrix *l,
                           pangulu_smatrix *u)
{
    pangulu_int64_t n = a->row;
    pangulu_int64_t *level_size = a->level_size;
    pangulu_int64_t *level_idx = a->level_idx;
    pangulu_int64_t num_lev = a->num_lev;
    pangulu_int64_t level_begin = 0;

    pangulu_int64_t perthread_worknum = 200;
    pangulu_int64_t parallelism = n / num_lev;
    pangulu_int64_t use_thread_num = (parallelism / perthread_worknum) + 1;
    pangulu_int64_t threadnum_max = omp_get_max_threads();
    if (use_thread_num > threadnum_max)
        use_thread_num = threadnum_max;
    if (use_thread_num == 1)
    {
        for (pangulu_int64_t level = 0; level < num_lev; ++level)
        {
            pangulu_int64_t level_end = level_begin + level_size[level];
            for (pangulu_int64_t aim = level_begin; aim < level_end; ++aim)
            {
                pangulu_int64_t i = level_idx[aim];
                if (a->columnpointer[i] == a->columnpointer[i + 1])
                {
                    continue;
                }
                pangulu_int64_t index = binarysearch_inblock_idx(a->columnpointer[i], a->columnpointer[i + 1], i, a->rowindex);
                calculate_type pivot;
                pivot = a->value_csc[index];

                for (pangulu_int64_t j = index + 1; j < a->columnpointer[i + 1]; j++) // i th colnum j th row  (j,i)
                {
                    // pangulu_int64_t row = a->rowindex[j];
                    if (fabs(pivot) > ERROR)
                    {
                    }
                    else
                    {
                        pivot = ERROR;
                    }
                    index = binarysearch_inblock_idx(a->columnpointer[i], a->columnpointer[i + 1], a->rowindex[j], a->rowindex);
                    calculate_type scale = a->value_csc[index] / pivot;

                    // Binary search return ith colnum jth row 's  index   of  a->value_csc
                    a->value_csc[index] = scale;
                    for (pangulu_int64_t x = i + 1; x < n; x++)
                    {
                        pangulu_int64_t nonzero1 = binarysearch_inblock_idx(a->columnpointer[x], a->columnpointer[x + 1], a->rowindex[j], a->rowindex); //(j,x)
                        pangulu_int64_t nonzero2 = binarysearch_inblock_idx(a->columnpointer[x], a->columnpointer[x + 1], i, a->rowindex);

                        if (nonzero1 != -1 && nonzero2 != -1)
                        {
                            a->value_csc[nonzero1] -= scale * a->value_csc[nonzero2];
                        }
                    }
                }
            }
            level_begin = level_end;
        }
    }
    else
    {

        for (pangulu_int64_t level = 0; level < num_lev; ++level)
        {
            pangulu_int64_t level_end = level_begin + level_size[level];
#pragma omp parallel for num_threads(use_thread_num)
            for (pangulu_int64_t aim = level_begin; aim < level_end; ++aim)
            {
                pangulu_int64_t i = level_idx[aim];
                if (a->columnpointer[i] == a->columnpointer[i + 1])
                {
                    continue;
                }
                pangulu_int64_t index = binarysearch_inblock_idx(a->columnpointer[i], a->columnpointer[i + 1], i, a->rowindex);
                calculate_type pivot;
                pivot = a->value_csc[index];

                for (pangulu_int64_t j = index + 1; j < a->columnpointer[i + 1]; j++) // i th colnum j th row  (j,i)
                {
                    // pangulu_int64_t row = a->rowindex[j];
                    if (fabs(pivot) > ERROR)
                    {
                    }
                    else
                    {
                        pivot = ERROR;
                    }
                    index = binarysearch_inblock_idx(a->columnpointer[i], a->columnpointer[i + 1], a->rowindex[j], a->rowindex);
                    calculate_type scale = a->value_csc[index] / pivot;

                    // Binary search return ith colnum jth row 's  index   of  a->value_csc
                    a->value_csc[index] = scale;
                    for (pangulu_int64_t x = i + 1; x < n; x++)
                    {
                        pangulu_int64_t nonzero1 = binarysearch_inblock_idx(a->columnpointer[x], a->columnpointer[x + 1], a->rowindex[j], a->rowindex); //(j,x)
                        pangulu_int64_t nonzero2 = binarysearch_inblock_idx(a->columnpointer[x], a->columnpointer[x + 1], i, a->rowindex);

                        if (nonzero1 != -1 && nonzero2 != -1)
                        {
#pragma omp critical
                            a->value_csc[nonzero1] -= scale * a->value_csc[nonzero2];
                        }
                    }
                }
            }
            level_begin = level_end;
        }
    }
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        pangulu_int64_t index = binarysearch_inblock_idx(a->columnpointer[i], a->columnpointer[i + 1], i, a->rowindex);
        for (pangulu_int64_t j = l->columnpointer[i], k = index; j < l->columnpointer[i + 1]; j++, k++)
        {
            l->value_csc[j] = a->value_csc[k];
            if (i == l->rowindex[j])
            {
                l->value_csc[j] = 1.0;
            }
        }
    }
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        for (pangulu_int64_t j = u->columnpointer[i], k = a->columnpointer[i]; j < u->columnpointer[i + 1]; j++, k++)
        {
            u->value_csc[j] = a->value_csc[k];
            if (i == u->rowindex[j] && fabs(u->value_csc[j]) < ERROR)
            {
                u->value_csc[j] = 1.0;
            }
        }
    }
}
#endif

void pangulu_sflu_fp64_dense(pangulu_smatrix *a,
                             pangulu_smatrix *l,
                             pangulu_smatrix *u)
{
    pangulu_int64_t n = a->row;

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int i = 0; i < n; i++)
    {
        for (int j = a->columnpointer[i]; j < a->columnpointer[i + 1]; j++)
        {
            int row = a->rowindex[j];
            if (row == i)
            {
                getrf_diagIndex_csc[i] = j;
            }
            temp_a_value[row * n + i] = a->value_csc[j]; // tranform csc to dense,only value
        }
        for (int j = a->rowpointer[i]; j < a->rowpointer[i + 1]; j++)
        {
            int col = a->columnindex[j];
            if (col == i)
            {
                getrf_diagIndex_csr[i] = j;
            }
        }
    }

    for (int i = 0; i < n; i++)
    {
        if (a->columnpointer[i] == a->columnpointer[i + 1])
        {
            continue;
        }
        calculate_type pivot = temp_a_value[i * n + i]; // diag value
        int L_index = l->columnpointer[i];              // restore l value

        l->value_csc[L_index++] = 1.0;
#pragma omp parallel for num_threads(pangu_omp_num_threads)
        for (int p = getrf_diagIndex_csc[i] + 1; p < a->columnpointer[i + 1]; p++)
        {

            int j = a->rowindex[p];
            calculate_type scale = temp_a_value[j * n + i] / pivot;

            temp_a_value[j * n + i] = scale;

            calculate_type *temp_value = temp_a_value + j * n;

            calculate_type *x_value = temp_a_value + i * n;

            for (int k = getrf_diagIndex_csr[i] + 1; k < a->rowpointer[i + 1]; k++)
            {
                int index_k = a->columnindex[k];
                temp_value[index_k] -= scale * x_value[index_k];
            }
        }
    }

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int i = 0; i < n; i++)
    {
        for (int j = u->columnpointer[i]; j < u->columnpointer[i + 1]; j++)
        {
            int index = u->rowindex[j];
            u->value_csc[j] = temp_a_value[index * n + i];
        }
        for (int j = l->columnpointer[i] + 1; j < l->columnpointer[i + 1]; j++)
        {
            int index = l->rowindex[j];
            l->value_csc[j] = temp_a_value[index * n + i];
        }
    }
}


void pangulu_getrf_fp64(pangulu_smatrix *a,
                        pangulu_smatrix *l,
                        pangulu_smatrix *u)
{

#ifdef OUTPUT_MATRICES
    char out_name_A[512];
    char out_name_L[512];
    char out_name_U[512];
    sprintf(out_name_A, "%s/%s/%d%s", OUTPUT_FILE, "getrf", getrf_number, "_getrf_A.cbd");
    sprintf(out_name_L, "%s/%s/%d%s", OUTPUT_FILE, "getrf", getrf_number, "_getrf_L.cbd");
    sprintf(out_name_U, "%s/%s/%d%s", OUTPUT_FILE, "getrf", getrf_number, "_getrf_U.cbd");
    pangulu_binary_write_csc_value_pangulu_smatrix(a, out_name_A);
    pangulu_binary_write_csc_pangulu_smatrix(l, out_name_L);
    pangulu_binary_write_csc_pangulu_smatrix(u, out_name_U);
    getrf_number++;
#endif
    pangulu_sflu_fp64_dense(a, l, u);
}
void pangulu_getrf_interface_c_v1(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *u)
{
#ifdef GPU_OPEN
    pangulu_smatrix_cuda_memcpy_value_csc(a, a);
#endif
    pangulu_sflu_fp64_dense(a, l, u);
#ifdef GPU_OPEN
    pangulu_smatrix_cuda_memcpy_to_device_value_csc(l, l);
    pangulu_smatrix_cuda_memcpy_to_device_value_csc(u, u);
#endif
}