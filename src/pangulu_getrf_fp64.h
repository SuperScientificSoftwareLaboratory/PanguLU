#ifndef PANGULU_GETRF_FP64_H
#define PANGULU_GETRF_FP64_H
#include "pangulu_common.h"

int_t Binary_search(int_t begin, int_t end, int_t aim, idx_int *array)
{
    end = end - 1;
    int_t middle = (end + begin) / 2;
    int_t left = begin;
    int_t right = end;
    while (left <= right)
    {
        if (array[middle] > aim)
        {
            right = middle - 1;
        }
        else if (array[middle] < aim)
        {
            left = middle + 1;
        }
        else
        {
            return middle;
        }
        middle = (right + left) / 2;
    }
    return -1; // not find
}

void pangulu_sflu_fp64(pangulu_Smatrix *A,
                       pangulu_Smatrix *L,
                       pangulu_Smatrix *U)
{
    int_t n = A->row;
    int_t index = 0;\
    for (int_t i = 0; i < n; i++) // for each column
    {
        if (A->columnpointer[i] == A->columnpointer[i + 1])
        {
            continue;
        }
        index = Binary_search(A->columnpointer[i], A->columnpointer[i + 1], i, A->rowindex);
        calculate_type pivot;
        pivot = A->value_CSC[index];

        int L_index = L->columnpointer[i]; // restore L value
        // int U_index = U->columnpointer[i];//restore U value
        L->value_CSC[L_index++] = 1.0;

        for (int_t j = index + 1; j < A->columnpointer[i + 1]; j++) // i th colnum j th row  (j,i) in L
        {
            // int_t row = A->rowindex[j];
            if (pivot > ERROR || pivot < -ERROR)
            {
            }
            else
            {
                pivot = ERROR;
            }
            index = Binary_search(A->columnpointer[i], A->columnpointer[i + 1], A->rowindex[j], A->rowindex); // index->(j,i)

            calculate_type scale = A->value_CSC[index] / pivot;

            // Binary search return ith colnum jth row 's  index   of  A->value_CSC
            A->value_CSC[index] = scale;

            L->value_CSC[L_index++] = scale; // L value
            for (int_t x = i + 1; x < n; x++)
            {
                int_t nonzero1 = Binary_search(A->columnpointer[x], A->columnpointer[x + 1], A->rowindex[j], A->rowindex); //(j,x)
                int_t nonzero2 = Binary_search(A->columnpointer[x], A->columnpointer[x + 1], i, A->rowindex);              //(i,x)

                if (nonzero1 != -1 && nonzero2 != -1)
                {
                    A->value_CSC[nonzero1] -= scale * A->value_CSC[nonzero2];
                }
            }
        }
    }
    for (int_t i = 0; i < n; i++)
    {
        for (int_t j = U->columnpointer[i], k = A->columnpointer[i]; j < U->columnpointer[i + 1]; j++, k++)
        {
            U->value_CSC[j] = A->value_CSC[k];
            if (i == U->rowindex[j] && U->value_CSC[j] < ERROR && U->value_CSC[j] > -ERROR)
            {
                U->value_CSC[j] = 1.0;
            }
        }
    }
}

void pangulu_sflu_fp64_dense_col(pangulu_Smatrix *A,
                                 pangulu_Smatrix *L,
                                 pangulu_Smatrix *U)
{
    char *data = getenv("OMP_NUM_THREADS");
    int num = data == NULL ? 1 : atoi(data);
    int omp_threads_num = num > 0 ? num : 1;

    int_t n = A->row;
    calculate_type *A_value = (calculate_type *)calloc(n * n, sizeof(calculate_type));
    char *A_flag = (char *)calloc(n * n, sizeof(char));

#pragma omp parallel for num_threads(omp_threads_num)
    for (int i = 0; i < n; i++)
    {
        for (int j = A->columnpointer[i]; j < A->columnpointer[i + 1]; j++)
        {
            int row = A->rowindex[j];
            A_value[i * n + row] = A->value_CSC[j]; // tranform csc to dense,only value
            A_flag[i * n + row] = 1;
        }
    }
    for (int i = 0; i < n; i++)
    {
        if (A->columnpointer[i] == A->columnpointer[i + 1])
        {
            continue;
        }
        calculate_type pivot = A_value[i * n + i]; // diag value
        int L_index = L->columnpointer[i];         // restore L value

        L->value_CSC[L_index++] = 1.0;

#pragma omp parallel for num_threads(omp_threads_num)
        for (int_t j = i + 1; j < n; j++) // i th colnum j th row  (j,i) in L
        {
            if (A_flag[i * n + j] == 0)
            {
                continue;
            }
            if (pivot > ERROR || pivot < -ERROR)
            {
            }
            else
            {
                pivot = ERROR;
            }
            calculate_type scale = A_value[i * n + j] / pivot;

            A_value[i * n + j] = scale;

            // L->value_CSC[L_index++] = scale; // L value

            for (int_t x = i + 1; x < n; x++)
            {
                if (A_flag[x * n + j] != 0)
                {
                    A_value[x * n + j] -= scale * A_value[x * n + i];
                }
            }
        }
    }
    // #pragma omp parallel for num_threads(omp_threads_num)
    for (int i = 0; i < n; i++)
    {
        int L_index = L->columnpointer[i] + 1;
        for (int j = i + 1; j < n; j++)
        {
            if (A_flag[i * n + j] != 0)
            {
                L->value_CSC[L_index++] = A_value[i * n + j];
            }
        }
        int U_index = U->columnpointer[i]; // restore U value
        for (int k = 0; k <= i; k++)
        {
            if (A_flag[i * n + k] != 0)
            {
                U->value_CSC[U_index++] = A_value[i * n + k];
            }
        }
    }

    free(A_value);
}

double wtc_get_time(struct timeval time_start, struct timeval time_end)
{
    return ((time_end.tv_sec - time_start.tv_sec) * 1000.0 + (time_end.tv_usec - time_start.tv_usec) / 1000.0);
}

void pangulu_sflu_fp64_dense_row_purge(pangulu_Smatrix *A,
                                       pangulu_Smatrix *L,
                                       pangulu_Smatrix *U)
{

    double time_calloc = 0, time_s_to_d = 0, time_lu = 0, time_d_to_s = 0;
    int_t n = A->row;
    int nnz = A->columnpointer[n] - A->columnpointer[0];
    struct timeval tstart, tend;

    gettimeofday(&tstart, NULL);

    char *TEMP_A_flag = (char *)calloc(n * n, sizeof(char));
    int *a_rowPointer = (int *)malloc(sizeof(int) * (n + 1));
    int *a_colIndex = (int *)malloc(sizeof(int) * nnz);

    int *getrf_diagIndex_csr = (int *)malloc(sizeof(int) * n);
    int *getrf_diagIndex_csc = (int *)malloc(sizeof(int) * n);
    gettimeofday(&tend, NULL);
    time_calloc += wtc_get_time(tstart, tend);

    gettimeofday(&tstart, NULL);

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int j = A->columnpointer[i]; j < A->columnpointer[i + 1]; j++)
        {
            int row = A->rowindex[j];
            if (row == i)
            {
                getrf_diagIndex_csc[i] = j;
            }
            TEMP_A_value[row * n + i] = A->value_CSC[j]; // tranform csc to dense,only value
            TEMP_A_flag[row * n + i] = 1;
        }
    }
    // csr data

    a_rowPointer[0] = 0;
    int data_index = 0;
    for (int i = 0; i < n; i++)
    {
        int row_num = 0;
        for (int j = 0; j < n; j++)
        {
            if (TEMP_A_flag[i * n + j] == 1)
            {
                if (i == j)
                {
                    getrf_diagIndex_csr[i] = data_index;
                }
                a_colIndex[data_index++] = j;
                row_num++;
            }
        }
        a_rowPointer[i + 1] = a_rowPointer[i] + row_num;
    }
    gettimeofday(&tend, NULL);
    time_s_to_d += wtc_get_time(tstart, tend);
    gettimeofday(&tstart, NULL);

    for (int i = 0; i < n; i++)
    {
        if (A->columnpointer[i] == A->columnpointer[i + 1])
        {
            continue;
        }
        calculate_type pivot = TEMP_A_value[i * n + i]; // diag value
        int L_index = L->columnpointer[i];              // restore L value

        L->value_CSC[L_index++] = 1.0;
#pragma omp parallel for
        for (int p = getrf_diagIndex_csc[i] + 1; p < A->columnpointer[i + 1]; p++)
        {

            int j = A->rowindex[p];
            calculate_type scale = TEMP_A_value[j * n + i] / pivot;

            TEMP_A_value[j * n + i] = scale;

            calculate_type *temp_value = TEMP_A_value + j * n;

            calculate_type *x_value = TEMP_A_value + i * n;

            for (int k = getrf_diagIndex_csr[i] + 1; k < a_rowPointer[i + 1]; k++)
            {
                int index_k = a_colIndex[k];
                temp_value[index_k] -= scale * x_value[index_k];
            }
        }
    }

    gettimeofday(&tend, NULL);
    time_lu += wtc_get_time(tstart, tend);

    gettimeofday(&tstart, NULL);
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int j = U->columnpointer[i]; j < U->columnpointer[i + 1]; j++)
        {
            int index = U->rowindex[j];
            U->value_CSC[j] = TEMP_A_value[index * n + i];
        }
        for (int j = L->columnpointer[i] + 1; j < L->columnpointer[i + 1]; j++)
        {
            int index = L->rowindex[j];
            L->value_CSC[j] = TEMP_A_value[index * n + i];
        }
    }

    gettimeofday(&tend, NULL);
    time_d_to_s += wtc_get_time(tstart, tend);

    free(a_rowPointer);
    free(a_colIndex);
    free(getrf_diagIndex_csr);
    free(getrf_diagIndex_csc);
    free(TEMP_A_flag);
}

void pangulu_sflu_fp64_2(pangulu_Smatrix *A,
                         pangulu_Smatrix *L,
                         pangulu_Smatrix *U)
{
    int_t n = A->row;
    int_t index = 0;
    for (int_t i = 0; i < n; i++) // for each column
    {
        if (A->columnpointer[i] == A->columnpointer[i + 1])
        {
            continue;
        }
        index = Binary_search(A->columnpointer[i], A->columnpointer[i + 1], i, A->rowindex);
        calculate_type pivot;
        pivot = A->value_CSC[index];

        for (int_t j = index + 1; j < A->columnpointer[i + 1]; j++) // i th colnum j th row  (j,i) in L
        {
            // int_t row = A->rowindex[j];
            if (pivot > ERROR || pivot < -ERROR)
            {
            }
            else
            {
                pivot = ERROR;
            }
            index = Binary_search(A->columnpointer[i], A->columnpointer[i + 1], A->rowindex[j], A->rowindex); // index->(j,i)
            calculate_type scale = A->value_CSC[index] / pivot;

            // Binary search return ith colnum jth row 's  index   of  A->value_CSC
            A->value_CSC[index] = scale;
            for (int_t x = i + 1; x < n; x++)
            {
                int_t nonzero1 = Binary_search(A->columnpointer[x], A->columnpointer[x + 1], A->rowindex[j], A->rowindex); //(j,x)
                int_t nonzero2 = Binary_search(A->columnpointer[x], A->columnpointer[x + 1], i, A->rowindex);              //(i,x)

                if (nonzero1 != -1 && nonzero2 != -1)
                {
                    A->value_CSC[nonzero1] -= scale * A->value_CSC[nonzero2];
                }
            }
        }
    }

    for (int_t i = 0; i < n; i++)
    {
        index = Binary_search(A->columnpointer[i], A->columnpointer[i + 1], i, A->rowindex);
        for (int_t j = L->columnpointer[i], k = index; j < L->columnpointer[i + 1]; j++, k++)
        {
            L->value_CSC[j] = A->value_CSC[k];
            if (i == L->rowindex[j])
            {
                L->value_CSC[j] = 1.0;
            }
        }
    }

    for (int_t i = 0; i < n; i++)
    {
        for (int_t j = U->columnpointer[i], k = A->columnpointer[i]; j < U->columnpointer[i + 1]; j++, k++)
        {
            U->value_CSC[j] = A->value_CSC[k];
            if (i == U->rowindex[j] && U->value_CSC[j] < ERROR && U->value_CSC[j] > -ERROR)
            {
                U->value_CSC[j] = 1.0;
            }
        }
    }
}
#ifndef GPU_OPEN
void pangulu_sflu_omp_fp64(pangulu_Smatrix *A,
                           pangulu_Smatrix *L,
                           pangulu_Smatrix *U)
{
    int_t n = A->row;
    int_t *level_size = A->level_size;
    int_t *level_idx = A->level_idx;
    int_t num_lev = A->num_lev;
    int_t level_begin = 0;

    int_t perthread_worknum = 200;
    int_t parallelism = n / num_lev;
    int_t use_thread_num = (parallelism / perthread_worknum) + 1;
    int_t threadnum_max = omp_get_max_threads();
    if (use_thread_num > threadnum_max)
        use_thread_num = threadnum_max;
    if (use_thread_num == 1)
    {
        for (int_t level = 0; level < num_lev; ++level)
        {
            int_t level_end = level_begin + level_size[level];
            for (int_t aim = level_begin; aim < level_end; ++aim)
            {
                int_t i = level_idx[aim];
                if (A->columnpointer[i] == A->columnpointer[i + 1])
                {
                    continue;
                }
                int_t index = Binary_search(A->columnpointer[i], A->columnpointer[i + 1], i, A->rowindex);
                calculate_type pivot;
                pivot = A->value_CSC[index];

                for (int_t j = index + 1; j < A->columnpointer[i + 1]; j++) // i th colnum j th row  (j,i)
                {
                    // int_t row = A->rowindex[j];
                    if (pivot > ERROR || pivot < -ERROR)
                    {
                    }
                    else
                    {
                        pivot = ERROR;
                    }
                    index = Binary_search(A->columnpointer[i], A->columnpointer[i + 1], A->rowindex[j], A->rowindex);
                    calculate_type scale = A->value_CSC[index] / pivot;

                    // Binary search return ith colnum jth row 's  index   of  A->value_CSC
                    A->value_CSC[index] = scale;
                    for (int_t x = i + 1; x < n; x++)
                    {
                        int_t nonzero1 = Binary_search(A->columnpointer[x], A->columnpointer[x + 1], A->rowindex[j], A->rowindex); //(j,x)
                        int_t nonzero2 = Binary_search(A->columnpointer[x], A->columnpointer[x + 1], i, A->rowindex);

                        if (nonzero1 != -1 && nonzero2 != -1)
                        {
                            A->value_CSC[nonzero1] -= scale * A->value_CSC[nonzero2];
                        }
                    }
                }
            }
            level_begin = level_end;
        }
    }
    else
    {

        for (int_t level = 0; level < num_lev; ++level)
        {
            int_t level_end = level_begin + level_size[level];
#pragma omp parallel for num_threads(use_thread_num)
            for (int_t aim = level_begin; aim < level_end; ++aim)
            {
                int_t i = level_idx[aim];
                if (A->columnpointer[i] == A->columnpointer[i + 1])
                {
                    continue;
                }
                int_t index = Binary_search(A->columnpointer[i], A->columnpointer[i + 1], i, A->rowindex);
                calculate_type pivot;
                pivot = A->value_CSC[index];

                for (int_t j = index + 1; j < A->columnpointer[i + 1]; j++) // i th colnum j th row  (j,i)
                {
                    // int_t row = A->rowindex[j];
                    if (pivot > ERROR || pivot < -ERROR)
                    {
                    }
                    else
                    {
                        pivot = ERROR;
                    }
                    index = Binary_search(A->columnpointer[i], A->columnpointer[i + 1], A->rowindex[j], A->rowindex);
                    calculate_type scale = A->value_CSC[index] / pivot;

                    // Binary search return ith colnum jth row 's  index   of  A->value_CSC
                    A->value_CSC[index] = scale;
                    for (int_t x = i + 1; x < n; x++)
                    {
                        int_t nonzero1 = Binary_search(A->columnpointer[x], A->columnpointer[x + 1], A->rowindex[j], A->rowindex); //(j,x)
                        int_t nonzero2 = Binary_search(A->columnpointer[x], A->columnpointer[x + 1], i, A->rowindex);

                        if (nonzero1 != -1 && nonzero2 != -1)
                        {
#pragma omp atomic
                            A->value_CSC[nonzero1] -= scale * A->value_CSC[nonzero2];
                        }
                    }
                }
            }
            level_begin = level_end;
        }
    }
    int_t index = 0;
    for (int_t i = 0; i < n; i++)
    {
        index = Binary_search(A->columnpointer[i], A->columnpointer[i + 1], i, A->rowindex);
        for (int_t j = L->columnpointer[i], k = index; j < L->columnpointer[i + 1]; j++, k++)
        {
            L->value_CSC[j] = A->value_CSC[k];
            if (i == L->rowindex[j])
            {
                L->value_CSC[j] = 1.0;
            }
        }
    }
    for (int_t i = 0; i < n; i++)
    {
        for (int_t j = U->columnpointer[i], k = A->columnpointer[i]; j < U->columnpointer[i + 1]; j++, k++)
        {
            U->value_CSC[j] = A->value_CSC[k];
            if (i == U->rowindex[j] && U->value_CSC[j] < ERROR && U->value_CSC[j] > -ERROR)
            {
                U->value_CSC[j] = 1.0;
            }
        }
    }
}
#endif

void pangulu_sflu_fp64_dense(pangulu_Smatrix *A,
                             pangulu_Smatrix *L,
                             pangulu_Smatrix *U)
{
    int_t n = A->row;

#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
    for (int i = 0; i < n; i++)
    {
        for (int j = A->columnpointer[i]; j < A->columnpointer[i + 1]; j++)
        {
            int row = A->rowindex[j];
            if (row == i)
            {
                getrf_diagIndex_csc[i] = j;
            }
            TEMP_A_value[row * n + i] = A->value_CSC[j]; // tranform csc to dense,only value
        }
        for (int j = A->rowpointer[i]; j < A->rowpointer[i + 1]; j++)
        {
            int col = A->columnindex[j];
            if (col == i)
            {
                getrf_diagIndex_csr[i] = j;
            }
        }
    }

    for (int i = 0; i < n; i++)
    {
        if (A->columnpointer[i] == A->columnpointer[i + 1])
        {
            continue;
        }
        calculate_type pivot = TEMP_A_value[i * n + i]; // diag value
        int L_index = L->columnpointer[i];              // restore L value

        L->value_CSC[L_index++] = 1.0;
#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
        for (int p = getrf_diagIndex_csc[i] + 1; p < A->columnpointer[i + 1]; p++)
        {

            int j = A->rowindex[p];
            calculate_type scale = TEMP_A_value[j * n + i] / pivot;

            TEMP_A_value[j * n + i] = scale;

            calculate_type *temp_value = TEMP_A_value + j * n;

            calculate_type *x_value = TEMP_A_value + i * n;

            for (int k = getrf_diagIndex_csr[i] + 1; k < A->rowpointer[i + 1]; k++)
            {
                int index_k = A->columnindex[k];
                temp_value[index_k] -= scale * x_value[index_k];
            }
        }
    }

#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
    for (int i = 0; i < n; i++)
    {
        for (int j = U->columnpointer[i]; j < U->columnpointer[i + 1]; j++)
        {
            int index = U->rowindex[j];
            U->value_CSC[j] = TEMP_A_value[index * n + i];
        }
        for (int j = L->columnpointer[i] + 1; j < L->columnpointer[i + 1]; j++)
        {
            int index = L->rowindex[j];
            L->value_CSC[j] = TEMP_A_value[index * n + i];
        }
    }
}

void pangulu_getrf_fp64(pangulu_Smatrix *A,
                        pangulu_Smatrix *L,
                        pangulu_Smatrix *U)
{

#ifdef OUTPUT_MATRICES
    char out_name_A[512];
    char out_name_L[512];
    char out_name_U[512];
    sprintf(out_name_A, "%s/%s/%d%s", OUTPUT_FILE, "getrf", getrf_number, "_getrf_A.cbd");
    sprintf(out_name_L, "%s/%s/%d%s", OUTPUT_FILE, "getrf", getrf_number, "_getrf_L.cbd");
    sprintf(out_name_U, "%s/%s/%d%s", OUTPUT_FILE, "getrf", getrf_number, "_getrf_U.cbd");
    pangulu_binary_write_csc_value_pangulu_Smatrix(A, out_name_A);
    pangulu_binary_write_csc_pangulu_Smatrix(L, out_name_L);
    pangulu_binary_write_csc_pangulu_Smatrix(U, out_name_U);
    getrf_number++;
#endif
    pangulu_sflu_fp64_dense(A, L, U);
}
void pangulu_getrf_interface_C_V1(pangulu_Smatrix *A,
                                  pangulu_Smatrix *L,
                                  pangulu_Smatrix *U)
{
#ifdef GPU_OPEN
    pangulu_Smatrix_CUDA_memcpy_value_CSC(A, A);
#endif
    pangulu_sflu_fp64_dense(A, L, U);
#ifdef GPU_OPEN
    pangulu_Smatrix_CUDA_memcpy_to_device_value_CSC(L, L);
    pangulu_Smatrix_CUDA_memcpy_to_device_value_CSC(U, U);
#endif
}
#endif