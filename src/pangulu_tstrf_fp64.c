#include "pangulu_common.h"
void pangulu_tstrf_fp64_cpu_1(pangulu_smatrix *a,
                              pangulu_smatrix *x,
                              pangulu_smatrix *u)
{
    pangulu_inblock_ptr *x_colpointer = a->columnpointer;
    pangulu_inblock_idx *x_rowindex = a->rowindex;
    calculate_type *x_value = a->value_csc;
    pangulu_inblock_ptr *u_rowpointer = u->rowpointer;
    pangulu_inblock_idx *u_columnindex = u->columnindex;
    calculate_type *u_value = u->value;
    pangulu_inblock_ptr *a_colpointer = a->columnpointer;
    pangulu_inblock_idx *a_rowindex = a->rowindex;
    calculate_type *a_value = x->value_csc;
    pangulu_int64_t n = a->row;

    for (pangulu_int64_t i = 0; i < a->nnz; i++)
    {
        x_value[i] = 0.0;
    }
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        calculate_type t = u_value[u_rowpointer[i]];
        if (fabs(t) < ERROR)
        {
            t = ERROR;
        }
        for (pangulu_int64_t k = a_colpointer[i]; k < a_colpointer[i + 1]; k++)
        {
            x_value[k] = a_value[k] / t;
        }
        // update Value
        if (a_colpointer[i] != a_colpointer[i + 1])
        {
#pragma omp parallel for num_threads(pangu_omp_num_threads)
            for (pangulu_int64_t k = u_rowpointer[i]; k < u_rowpointer[i + 1]; k++)
            {
                pangulu_int64_t p = x_colpointer[i];
                for (pangulu_int64_t s = a_colpointer[u_columnindex[k]]; s < a_colpointer[u_columnindex[k] + 1]; s++)
                {
                    if (x_rowindex[p] == a_rowindex[s])
                    {
                        a_value[s] -= x_value[p] * u_value[k];
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

void pangulu_tstrf_fp64_cpu_2(pangulu_smatrix *a,
                              pangulu_smatrix *x,
                              pangulu_smatrix *u)
{

    pangulu_inblock_ptr *A_columnpointer = a->rowpointer;
    pangulu_inblock_idx *A_rowidx = a->columnindex;

    calculate_type *a_value = a->value;

    pangulu_inblock_ptr *L_rowpointer = u->columnpointer;

    pangulu_inblock_ptr *L_colpointer = u->rowpointer;
    pangulu_inblock_idx *L_rowindex = u->columnindex;
    calculate_type *L_value = u->value;

    pangulu_int64_t n = a->row;

    pangulu_int64_t *Spointer = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (n + 1));
    memset(Spointer, 0, sizeof(pangulu_int64_t) * (n + 1));
    int rhs = 0;
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        if (A_columnpointer[i] != A_columnpointer[i + 1])
        {
            Spointer[rhs] = i;
            rhs++;
        }
    }

    calculate_type *C_b = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n * rhs);
    calculate_type *D_x = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n * rhs);

    memset(C_b, 0.0, sizeof(calculate_type) * n * rhs);
    memset(D_x, 0.0, sizeof(calculate_type) * n * rhs);

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int i = 0; i < rhs; i++)
    {
        int index = Spointer[i];
        for (int j = A_columnpointer[index]; j < A_columnpointer[index + 1]; j++)
        {
            C_b[i * n + A_rowidx[j]] = a_value[j];
        }
    }

    int nlevel = 0;
    int *levelPtr = (int *)pangulu_malloc(__FILE__, __LINE__, sizeof(int) * (n + 1));
    int *levelItem = (int *)pangulu_malloc(__FILE__, __LINE__, sizeof(int) * n);
    findlevel(L_colpointer, L_rowindex, L_rowpointer, n, &nlevel, levelPtr, levelItem);

#pragma omp parallel for num_threads(pangu_omp_num_threads)
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

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int i = 0; i < rhs; i++)
    {
        int index = Spointer[i];
        for (int j = A_columnpointer[index]; j < A_columnpointer[index + 1]; j++)
        {
            a_value[j] = C_b[i * n + A_rowidx[j]];
        }
    }

    pangulu_free(__FILE__, __LINE__, Spointer);
    pangulu_free(__FILE__, __LINE__, C_b);
    pangulu_free(__FILE__, __LINE__, D_x);
}
void pangulu_tstrf_fp64_cpu_3(pangulu_smatrix *a,
                              pangulu_smatrix *x,
                              pangulu_smatrix *u)
{

    pangulu_inblock_ptr *A_columnpointer = a->rowpointer;
    pangulu_inblock_idx *A_rowidx = a->columnindex;

    calculate_type *a_value = a->value;

    pangulu_inblock_ptr *L_columnpointer = u->rowpointer;
    pangulu_inblock_idx *L_rowidx = u->columnindex;
    calculate_type *L_value = u->value;

    pangulu_int64_t n = a->row;

    calculate_type *C_b = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n * n);

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int i = 0; i < n; i++) // jth column of u
    {
        for (int j = A_columnpointer[i]; j < A_columnpointer[i + 1]; j++)
        {
            int idx = A_rowidx[j];
            C_b[i * n + idx] = a_value[j]; // tranform csr to dense,only value
        }
    }

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        for (pangulu_int64_t j = A_columnpointer[i]; j < A_columnpointer[i + 1]; j++)
        {
            C_b[i * n + A_rowidx[j]] /= L_value[L_columnpointer[A_rowidx[j]]];
            pangulu_inblock_idx idx = A_rowidx[j];
            for (pangulu_int64_t k = L_columnpointer[idx] + 1; k < L_columnpointer[idx + 1]; k++)
            {
                C_b[i * n + L_rowidx[k]] -= L_value[k] * C_b[i * n + A_rowidx[j]];
            }
        }
    }

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int i = 0; i < n; i++)
    {
        for (int j = A_columnpointer[i]; j < A_columnpointer[i + 1]; j++)
        {
            int idx = A_rowidx[j];
            a_value[j] = C_b[i * n + idx];
        }
    }
    pangulu_free(__FILE__, __LINE__, C_b);
}
void pangulu_tstrf_fp64_cpu_4(pangulu_smatrix *a,
                              pangulu_smatrix *x,
                              pangulu_smatrix *u)
{

    pangulu_inblock_ptr *A_columnpointer = a->rowpointer;
    pangulu_inblock_idx *A_rowidx = a->columnindex;

    calculate_type *a_value = a->value;

    pangulu_inblock_ptr *L_columnpointer = u->rowpointer;
    pangulu_inblock_idx *L_rowidx = u->columnindex;
    calculate_type *L_value = u->value;

    pangulu_int64_t n = a->row;

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        for (pangulu_int64_t j = A_columnpointer[i]; j < A_columnpointer[i + 1]; j++)
        {
            pangulu_inblock_idx idx = A_rowidx[j];
            a_value[j] /= L_value[L_columnpointer[idx]];
            for (pangulu_int64_t k = L_columnpointer[idx] + 1, p = j + 1; k < L_columnpointer[idx + 1] && p < A_columnpointer[i + 1]; k++, p++)
            {
                if (L_rowidx[k] == A_rowidx[p])
                {
                    a_value[p] -= L_value[k] * a_value[j];
                }
                else
                {
                    k--;
                }
            }
        }
    }
}
void pangulu_tstrf_fp64_cpu_5(pangulu_smatrix *a,
                              pangulu_smatrix *x,
                              pangulu_smatrix *u)
{

    pangulu_inblock_ptr *A_rowpointer = a->columnpointer;
    pangulu_inblock_idx *A_colindex = a->rowindex;
    calculate_type *a_value = x->value_csc;

    pangulu_inblock_ptr *L_colpointer = u->rowpointer;
    pangulu_inblock_idx *L_rowindex = u->columnindex;
    calculate_type *L_value = u->value;

    pangulu_inblock_ptr *X_rowpointer = a->columnpointer;
    pangulu_inblock_idx *X_colindex = a->rowindex;
    calculate_type *x_value = a->value_csc;

    pangulu_int64_t n = a->row;

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int i = 0; i < n; i++) // jth column of u
    {
        for (int j = A_rowpointer[i]; j < A_rowpointer[i + 1]; j++)
        {
            pangulu_inblock_idx idx = A_colindex[j];
            temp_a_value[i * n + idx] = a_value[j]; // tranform csr to dense,only value
        }
    }

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        // x get value from a
        for (pangulu_int64_t k = X_rowpointer[i]; k < X_rowpointer[i + 1]; k++)
        {
            temp_a_value[i * n + X_colindex[k]] /= L_value[L_colpointer[i]];
            x_value[k] = temp_a_value[i * n + X_colindex[k]];
        }
        // update Value
        if (X_rowpointer[i] != X_rowpointer[i + 1])
        {
#pragma omp parallel for num_threads(pangu_omp_num_threads)
            for (pangulu_int64_t j = L_colpointer[i] + 1; j < L_colpointer[i + 1]; j++)
            {
                pangulu_inblock_idx idx1 = L_rowindex[j];

                for (pangulu_int64_t p = X_rowpointer[i]; p < X_rowpointer[i + 1]; p++)
                {

                    pangulu_inblock_idx idx2 = A_colindex[p];
                    temp_a_value[idx1 * n + idx2] -= L_value[j] * temp_a_value[i * n + idx2];
                }
            }
        }
    }
}
void pangulu_tstrf_fp64_cpu_6(pangulu_smatrix *a,
                              pangulu_smatrix *x,
                              pangulu_smatrix *u)
{

    pangulu_inblock_ptr *A_columnpointer = a->rowpointer;
    pangulu_inblock_idx *A_rowidx = a->columnindex;

    calculate_type *a_value = a->value;

    pangulu_inblock_ptr *L_columnpointer = u->rowpointer;
    pangulu_inblock_idx *L_rowidx = u->columnindex;
    calculate_type *L_value = u->value;

    pangulu_inblock_ptr n = a->row;
#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int i = 0; i < n; i++) // jth column of u
    {
        for (int j = A_columnpointer[i]; j < A_columnpointer[i + 1]; j++)
        {
            int idx = A_rowidx[j];
            temp_a_value[i * n + idx] = a_value[j]; // tranform csr to dense,only value
        }
    }

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        for (pangulu_int64_t j = A_columnpointer[i]; j < A_columnpointer[i + 1]; j++)
        {
            pangulu_inblock_idx idx = A_rowidx[j];

            a_value[j] = temp_a_value[i * n + idx] / L_value[L_columnpointer[idx]];
            for (pangulu_int64_t k = L_columnpointer[idx] + 1; k < L_columnpointer[idx + 1]; k++)
            {
                temp_a_value[i * n + L_rowidx[k]] -= L_value[k] * a_value[j];
            }
        }
    }
}
void pangulu_tstrf_interface_cpu_csr(pangulu_smatrix *a,
                                     pangulu_smatrix *x,
                                     pangulu_smatrix *u)
{

#ifdef OUTPUT_MATRICES
    char out_name_B[512];
    char out_name_U[512];
    sprintf(out_name_B, "%s/%s/%d%s", OUTPUT_FILE, "tstrf", tstrf_number, "_tstrf_B.cbd");
    sprintf(out_name_U, "%s/%s/%d%s", OUTPUT_FILE, "tstrf", tstrf_number, "_tstrf_U.cbd");
    pangulu_binary_write_csc_pangulu_smatrix(a, out_name_B);
    pangulu_binary_write_csc_pangulu_smatrix(u, out_name_U);
    tstrf_number++;
#endif
    pangulu_tstrf_fp64_cpu_1(a, x, u);
}

void pangulu_tstrf_interface_cpu_csc(pangulu_smatrix *a,
                                     pangulu_smatrix *x,
                                     pangulu_smatrix *u)
{
    pangulu_tstrf_fp64_cpu_6(a, x, u);
}

void pangulu_tstrf_interface_c_v1(pangulu_smatrix *a,
                                  pangulu_smatrix *x,
                                  pangulu_smatrix *u)
{
#ifdef GPU_OPEN
    pangulu_smatrix_cuda_memcpy_value_csc(a, a);
#endif
    pangulu_transpose_pangulu_smatrix_csc_to_csr(a);
    pangulu_pangulu_smatrix_memcpy_value_csc_copy_length(x, a);
    pangulu_tstrf_fp64_cpu_4(a, x, u);
    pangulu_transpose_pangulu_smatrix_csr_to_csc(a);
#ifdef GPU_OPEN
    pangulu_smatrix_cuda_memcpy_to_device_value_csc(a, a);
#endif
}
void pangulu_tstrf_interface_c_v2(pangulu_smatrix *a,
                                  pangulu_smatrix *x,
                                  pangulu_smatrix *u)
{
#ifdef GPU_OPEN
    pangulu_smatrix_cuda_memcpy_value_csc(a, a);
#endif
    pangulu_transpose_pangulu_smatrix_csc_to_csr(a);
    pangulu_pangulu_smatrix_memcpy_value_csc_copy_length(x, a);
    pangulu_tstrf_fp64_cpu_6(a, x, u);
    pangulu_transpose_pangulu_smatrix_csr_to_csc(a);
#ifdef GPU_OPEN
    pangulu_smatrix_cuda_memcpy_to_device_value_csc(a, a);
#endif
}

pangulu_int64_t TEMP_calculate_type_len = 0;
calculate_type* TEMP_calculate_type = NULL;
pangulu_int64_t TEMP_pangulu_inblock_ptr_len = 0;
pangulu_inblock_ptr* TEMP_pangulu_inblock_ptr = NULL;

int tstrf_csc_csc(
    pangulu_inblock_idx n,
    pangulu_inblock_ptr* U_colptr,
    pangulu_inblock_idx* U_rowidx,
    calculate_type* u_value,
    pangulu_inblock_ptr* A_colptr,
    pangulu_inblock_idx* A_rowidx,
    calculate_type* a_value
){
    if(TEMP_calculate_type_len < n){
        calculate_type* TEMP_calculate_type_old = TEMP_calculate_type;
        TEMP_calculate_type = (calculate_type*)pangulu_realloc(__FILE__, __LINE__, TEMP_calculate_type, n*sizeof(calculate_type));
        if(TEMP_calculate_type == NULL){
            pangulu_free(__FILE__, __LINE__, TEMP_calculate_type_old);
            TEMP_calculate_type_len = 0;
            printf("[ERROR] kernel error : CPU sparse tstrf : realloc TEMP_calculate_type failed.\n");
            return 1;
        }
        TEMP_calculate_type_len = n;
    }
    
    if(TEMP_pangulu_inblock_ptr_len < n){
        pangulu_inblock_ptr* TEMP_int64_old = TEMP_pangulu_inblock_ptr;
        TEMP_pangulu_inblock_ptr = (pangulu_inblock_ptr*)pangulu_realloc(__FILE__, __LINE__, TEMP_pangulu_inblock_ptr, n*sizeof(pangulu_inblock_ptr));
        if(TEMP_pangulu_inblock_ptr == NULL){
            pangulu_free(__FILE__, __LINE__, TEMP_int64_old);
            TEMP_pangulu_inblock_ptr_len = 0;
            printf("[ERROR] kernel error : CPU sparse tstrf : realloc TEMP_int64 failed.\n");
            return 2;
        }
        TEMP_pangulu_inblock_ptr_len = n;
    }

    pangulu_inblock_ptr* U_next_array = TEMP_pangulu_inblock_ptr;
    calculate_type* A_major_column = TEMP_calculate_type;
    memcpy(U_next_array, U_colptr, sizeof(pangulu_inblock_ptr) * n);
    for(pangulu_int64_t i=0;i<n;i++){ // A的每列作为主列
        memset(A_major_column, 0, sizeof(calculate_type)*n);
        calculate_type U_pivot = u_value[U_colptr[i+1]-1]; //这里i本来应是A主列的列号，也是U主元的行号。U的主元在对角线上，因此，i也是U的主元的列号。
        // #pragma omp parallel for
        for(pangulu_int64_t j=A_colptr[i];j<A_colptr[i+1];j++){ 
            A_major_column[A_rowidx[j]] = (a_value[j] /= U_pivot);
        }
        // #pragma omp parallel for
        for(pangulu_int64_t k=i+1;k<n;k++){ // 遍历A的副列
            if(U_next_array[k] >= U_colptr[k+1]/*U_next_array[k]跑到了下一列*/ || U_rowidx[U_next_array[k]] > i/*U的第k列中，下一个要访问的元素的行号大于A当前主列号i*/){
                continue;
            }
            for(pangulu_int64_t j=A_colptr[k];j<A_colptr[k+1];j++){ // 遍历A的副列k中的每个元素
                a_value[j] -= u_value[U_next_array[k]] * A_major_column[A_rowidx[j]];
            }
            U_next_array[k]++;
        }
    }
    return 0;
}