#ifndef PANGULU_TSTRF_FP64_H
#define PANGULU_TSTRF_FP64_H

#include "pangulu_common.h"
void pangulu_tstrf_fp64_CPU_1(pangulu_Smatrix *A,
                              pangulu_Smatrix *X,
                              pangulu_Smatrix *U)
{
    int_t *X_colpointer = A->columnpointer;
    idx_int *X_rowindex = A->rowindex;
    calculate_type *X_value = A->value_CSC;
    int_t *U_rowpointer = U->rowpointer;
    idx_int *U_columnindex = U->columnindex;
    calculate_type *U_value = U->value;
    int_t *A_colpointer = A->columnpointer;
    idx_int *A_rowindex = A->rowindex;
    calculate_type *A_value = X->value_CSC;
    int_t n = A->row;

    for (int_t i = 0; i < A->nnz; i++)
    {
        X_value[i] = 0.0;
    }
    for (int_t i = 0; i < n; i++)
    {
        calculate_type t = U_value[U_rowpointer[i]];
        if (t < ERROR && t > -ERROR)
        {
            t = ERROR;
        }
        for (int_t k = A_colpointer[i]; k < A_colpointer[i + 1]; k++)
        {
            X_value[k] = A_value[k] / t;
        }
        // update Value
        if (A_colpointer[i] != A_colpointer[i + 1])
        {
#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
            for (int_t k = U_rowpointer[i]; k < U_rowpointer[i + 1]; k++)
            {
                int_t p = X_colpointer[i];
                for (int_t s = A_colpointer[U_columnindex[k]]; s < A_colpointer[U_columnindex[k] + 1]; s++)
                {
                    if (X_rowindex[p] == A_rowindex[s])
                    {
                        A_value[s] -= X_value[p] * U_value[k];
                        p++;
                    }
                    else
                    {
                        continue;
                    }
                }
            }
        }
    }
}

int findlevel(const int_t *cscColPtr,
              const idx_int *cscRowIdx,
              const int_t *csrRowPtr,
              const int_t m,
              int *nlevel,
              int *levelPtr,
              int *levelItem);
void pangulu_tstrf_fp64_CPU_2(pangulu_Smatrix *A,
                              pangulu_Smatrix *X,
                              pangulu_Smatrix *U)
{

    int_t *A_columnpointer = A->rowpointer;
    idx_int *A_rowidx = A->columnindex;

    calculate_type *A_value = A->value;

    int_t *L_rowpointer = U->columnpointer;

    int_t *L_colpointer = U->rowpointer;
    idx_int *L_rowindex = U->columnindex;
    calculate_type *L_value = U->value;

    int_t n = A->row;

    int_t *Spointer = (int_t *)malloc(sizeof(int_t) * (n + 1));
    memset(Spointer, 0, sizeof(int_t) * (n + 1));
    int rhs = 0;
    for (int_t i = 0; i < n; i++)
    {
        if (A_columnpointer[i] != A_columnpointer[i + 1])
        {
            Spointer[rhs] = i;
            rhs++;
        }
    }

    calculate_type *C_b = (calculate_type *)malloc(sizeof(calculate_type) * n * rhs);
    calculate_type *D_x = (calculate_type *)malloc(sizeof(calculate_type) * n * rhs);

    memset(C_b, 0.0, sizeof(calculate_type) * n * rhs);
    memset(D_x, 0.0, sizeof(calculate_type) * n * rhs);

#pragma omp parallel for
    for (int i = 0; i < rhs; i++)
    {
        int index = Spointer[i];
        for (int j = A_columnpointer[index]; j < A_columnpointer[index + 1]; j++)
        {
            C_b[i * n + A_rowidx[j]] = A_value[j];
        }
    }

    int nlevel = 0;
    int *levelPtr = (int *)malloc(sizeof(int) * (n + 1));
    int *levelItem = (int *)malloc(sizeof(int) * n);
    findlevel(L_colpointer, L_rowindex, L_rowpointer, n, &nlevel, levelPtr, levelItem);

#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
    for (int i = 0; i < rhs; i++)
    {
        for (int li = 0; li < nlevel; li++)
        {

            for (int ri = levelPtr[li]; ri < levelPtr[li + 1]; ri++)
            {
                C_b[i * n + levelItem[ri]] /= L_value[L_colpointer[levelItem[ri]]];
                for (int j = L_colpointer[levelItem[ri]] + 1; j < L_colpointer[levelItem[ri] + 1]; j++)
                {
                    C_b[i * n + L_rowindex[j]] -= L_value[j] * C_b[i * n + levelItem[ri]];
                }
            }
        }
    }

#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
    for (int i = 0; i < rhs; i++)
    {
        int index = Spointer[i];
        for (int j = A_columnpointer[index]; j < A_columnpointer[index + 1]; j++)
        {
            A_value[j] = C_b[i * n + A_rowidx[j]];
        }
    }

    free(Spointer);
    free(C_b);
    free(D_x);
}
void pangulu_tstrf_fp64_CPU_3(pangulu_Smatrix *A,
                              pangulu_Smatrix *X,
                              pangulu_Smatrix *U)
{

    int_t *A_columnpointer = A->rowpointer;
    idx_int *A_rowidx = A->columnindex;

    calculate_type *A_value = A->value;

    int_t *L_columnpointer = U->rowpointer;
    idx_int *L_rowidx = U->columnindex;
    calculate_type *L_value = U->value;

    int_t n = A->row;

    calculate_type *C_b = (calculate_type *)malloc(sizeof(calculate_type) * n * n);

#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
    for (int i = 0; i < n; i++) // jth column of U
    {
        for (int j = A_columnpointer[i]; j < A_columnpointer[i + 1]; j++)
        {
            int idx = A_rowidx[j];
            C_b[i * n + idx] = A_value[j]; // tranform csr to dense,only value
        }
    }

#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
    for (int_t i = 0; i < n; i++)
    {
        for (int_t j = A_columnpointer[i]; j < A_columnpointer[i + 1]; j++)
        {
            C_b[i * n + A_rowidx[j]] /= L_value[L_columnpointer[A_rowidx[j]]];
            idx_int idx = A_rowidx[j];
            for (int_t k = L_columnpointer[idx] + 1; k < L_columnpointer[idx + 1]; k++)
            {
                C_b[i * n + L_rowidx[k]] -= L_value[k] * C_b[i * n + A_rowidx[j]];
            }
        }
    }

#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
    for (int i = 0; i < n; i++)
    {
        for (int j = A_columnpointer[i]; j < A_columnpointer[i + 1]; j++)
        {
            int idx = A_rowidx[j];
            A_value[j] = C_b[i * n + idx];
        }
    }
    free(C_b);
}
void pangulu_tstrf_fp64_CPU_4(pangulu_Smatrix *A,
                              pangulu_Smatrix *X,
                              pangulu_Smatrix *U)
{

    int_t *A_columnpointer = A->rowpointer;
    idx_int *A_rowidx = A->columnindex;

    calculate_type *A_value = A->value;

    int_t *L_columnpointer = U->rowpointer;
    idx_int *L_rowidx = U->columnindex;
    calculate_type *L_value = U->value;

    int_t n = A->row;

#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
    for (int_t i = 0; i < n; i++)
    {
        for (int_t j = A_columnpointer[i]; j < A_columnpointer[i + 1]; j++)
        {
            idx_int idx = A_rowidx[j];
            A_value[j] /= L_value[L_columnpointer[idx]];
            for (int_t k = L_columnpointer[idx] + 1, p = j + 1; k < L_columnpointer[idx + 1] && p < A_columnpointer[i + 1]; k++, p++)
            {
                if (L_rowidx[k] == A_rowidx[p])
                {
                    A_value[p] -= L_value[k] * A_value[j];
                }
                else
                {
                    k--;
                }
            }
        }
    }
}
void pangulu_tstrf_fp64_CPU_5(pangulu_Smatrix *A,
                              pangulu_Smatrix *X,
                              pangulu_Smatrix *U)
{

    int_t *A_rowpointer = A->columnpointer;
    idx_int *A_colindex = A->rowindex;
    calculate_type *A_value = X->value_CSC;

    int_t *L_colpointer = U->rowpointer;
    idx_int *L_rowindex = U->columnindex;
    calculate_type *L_value = U->value;

    int_t *X_rowpointer = A->columnpointer;
    idx_int *X_colindex = A->rowindex;
    calculate_type *X_value = A->value_CSC;

    int_t n = A->row;

#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
    for (int i = 0; i < n; i++) // jth column of U
    {
        for (int j = A_rowpointer[i]; j < A_rowpointer[i + 1]; j++)
        {
            idx_int idx = A_colindex[j];
            TEMP_A_value[i * n + idx] = A_value[j]; // tranform csr to dense,only value
        }
    }

    for (int_t i = 0; i < n; i++)
    {
        // X get value from A
        for (int_t k = X_rowpointer[i]; k < X_rowpointer[i + 1]; k++)
        {
            TEMP_A_value[i * n + X_colindex[k]] /= L_value[L_colpointer[i]];
            X_value[k] = TEMP_A_value[i * n + X_colindex[k]];
        }
        // update Value
        if (X_rowpointer[i] != X_rowpointer[i + 1])
        {
#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
            for (int_t j = L_colpointer[i] + 1; j < L_colpointer[i + 1]; j++)
            {
                idx_int idx1 = L_rowindex[j];

                for (int_t p = X_rowpointer[i]; p < X_rowpointer[i + 1]; p++)
                {

                    idx_int idx2 = A_colindex[p];
                    TEMP_A_value[idx1 * n + idx2] -= L_value[j] * TEMP_A_value[i * n + idx2];
                }
            }
        }
    }
}
void pangulu_tstrf_fp64_CPU_6(pangulu_Smatrix *A,
                              pangulu_Smatrix *X,
                              pangulu_Smatrix *U)
{

    int_t *A_columnpointer = A->rowpointer;
    idx_int *A_rowidx = A->columnindex;

    calculate_type *A_value = A->value;

    int_t *L_columnpointer = U->rowpointer;
    idx_int *L_rowidx = U->columnindex;
    calculate_type *L_value = U->value;

    int_t n = A->row;
#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
    for (int i = 0; i < n; i++) // jth column of U
    {
        for (int j = A_columnpointer[i]; j < A_columnpointer[i + 1]; j++)
        {
            int idx = A_rowidx[j];
            TEMP_A_value[i * n + idx] = A_value[j]; // tranform csr to dense,only value
        }
    }

#pragma omp parallel for num_threads(PANGU_OMP_NUM_THREADS)
    for (int_t i = 0; i < n; i++)
    {
        for (int_t j = A_columnpointer[i]; j < A_columnpointer[i + 1]; j++)
        {
            idx_int idx = A_rowidx[j];

            A_value[j] = TEMP_A_value[i * n + idx] / L_value[L_columnpointer[idx]];
            for (int_t k = L_columnpointer[idx] + 1; k < L_columnpointer[idx + 1]; k++)
            {
                TEMP_A_value[i * n + L_rowidx[k]] -= L_value[k] * A_value[j];
            }
        }
    }
}
void pangulu_tstrf_interface_CPU_CSR(pangulu_Smatrix *A,
                                     pangulu_Smatrix *X,
                                     pangulu_Smatrix *U)
{

#ifdef OUTPUT_MATRICES
    char out_name_B[512];
    char out_name_U[512];
    sprintf(out_name_B, "%s/%s/%d%s", OUTPUT_FILE, "tstrf", tstrf_number, "_tstrf_B.cbd");
    sprintf(out_name_U, "%s/%s/%d%s", OUTPUT_FILE, "tstrf", tstrf_number, "_tstrf_U.cbd");
    pangulu_binary_write_csc_pangulu_Smatrix(A, out_name_B);
    pangulu_binary_write_csc_pangulu_Smatrix(U, out_name_U);
    tstrf_number++;
#endif
    pangulu_tstrf_fp64_CPU_1(A, X, U);
}

void pangulu_tstrf_interface_CPU_CSC(pangulu_Smatrix *A,
                                     pangulu_Smatrix *X,
                                     pangulu_Smatrix *U)
{
    pangulu_tstrf_fp64_CPU_6(A, X, U);
}

void pangulu_tstrf_interface_C_V1(pangulu_Smatrix *A,
                                  pangulu_Smatrix *X,
                                  pangulu_Smatrix *U)
{
#ifdef GPU_OPEN
    pangulu_Smatrix_CUDA_memcpy_value_CSC(A, A);
#endif
    pangulu_transport_pangulu_Smatrix_CSC_to_CSR(A);
    pangulu_pangulu_Smatrix_memcpy_value_CSC_copy_length(X, A);
    pangulu_tstrf_fp64_CPU_4(A, X, U);
    pangulu_transport_pangulu_Smatrix_CSR_to_CSC(A);
#ifdef GPU_OPEN
    pangulu_Smatrix_CUDA_memcpy_to_device_value_CSC(A, A);
#endif
}
void pangulu_tstrf_interface_C_V2(pangulu_Smatrix *A,
                                  pangulu_Smatrix *X,
                                  pangulu_Smatrix *U)
{
#ifdef GPU_OPEN
    pangulu_Smatrix_CUDA_memcpy_value_CSC(A, A);
#endif
    pangulu_transport_pangulu_Smatrix_CSC_to_CSR(A);
    pangulu_pangulu_Smatrix_memcpy_value_CSC_copy_length(X, A);
    pangulu_tstrf_fp64_CPU_6(A, X, U);
    pangulu_transport_pangulu_Smatrix_CSR_to_CSC(A);
#ifdef GPU_OPEN
    pangulu_Smatrix_CUDA_memcpy_to_device_value_CSC(A, A);
#endif
}
#endif