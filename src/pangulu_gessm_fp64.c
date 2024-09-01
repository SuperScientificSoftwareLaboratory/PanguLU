#include "pangulu_common.h"
void pangulu_gessm_fp64_cpu_1(pangulu_smatrix *a,
                              pangulu_smatrix *l,
                              pangulu_smatrix *x)
{

    pangulu_inblock_ptr *a_rowpointer = a->rowpointer;
    pangulu_inblock_idx *a_colindex = a->columnindex;
    calculate_type *a_value = x->value;

    pangulu_inblock_ptr *l_colpointer = l->columnpointer;
    pangulu_inblock_idx *l_rowindex = l->rowindex;
    calculate_type *l_value = l->value_csc;

    pangulu_inblock_ptr *x_rowpointer = a->rowpointer;
    pangulu_inblock_idx *x_colindex = a->columnindex;
    calculate_type *x_value = a->value;

    pangulu_int64_t n = a->row;
#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (pangulu_int64_t i = 0; i < a->nnz; i++)
    {
        x_value[i] = 0.0;
    }

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        // x get value from a
        for (pangulu_int64_t k = x_rowpointer[i]; k < x_rowpointer[i + 1]; k++)
        {
            x_value[k] = a_value[k];
        }
        // update Value
        if (x_rowpointer[i] != x_rowpointer[i + 1])
        {
#pragma omp parallel for num_threads(pangu_omp_num_threads)
            for (pangulu_int64_t j = l_colpointer[i]; j < l_colpointer[i + 1]; j++)
            {

                for (pangulu_int64_t p = a_rowpointer[l_rowindex[j]], k = x_rowpointer[i]; p < a_rowpointer[l_rowindex[j] + 1]; p++, k++)
                {
                    if (a_colindex[p] == x_colindex[k])
                    {
                        a_value[p] -= l_value[j] * x_value[k];
                    }
                    else
                    {
                        k--;
                    }
                }
            }
        }
    }
}

void pangulu_gessm_fp64_cpu_2(pangulu_smatrix *a,
                              pangulu_smatrix *l,
                              pangulu_smatrix *x)
{

    pangulu_inblock_ptr *a_columnpointer = a->columnpointer;
    pangulu_inblock_idx *a_rowidx = a->rowindex;

    calculate_type *a_value = a->value_csc;

    pangulu_inblock_ptr *l_rowpointer = l->rowpointer;

    pangulu_inblock_ptr *l_colpointer = l->columnpointer;
    pangulu_inblock_idx *l_rowindex = l->rowindex;
    calculate_type *l_value = l->value_csc;

    pangulu_int64_t n = a->row;

    pangulu_int64_t *spointer = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (n + 1));
    memset(spointer, 0, sizeof(pangulu_int64_t) * (n + 1));
    int rhs = 0;
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        if (a_columnpointer[i] != a_columnpointer[i + 1])
        {
            spointer[rhs] = i;
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
        int index = spointer[i];
        for (int j = a_columnpointer[index]; j < a_columnpointer[index + 1]; j++)
        {
            C_b[i * n + a_rowidx[j]] = a_value[j];
        }
    }

    int nlevel = 0;
    int *levelPtr = (int *)pangulu_malloc(__FILE__, __LINE__, sizeof(int) * (n + 1));
    int *levelItem = (int *)pangulu_malloc(__FILE__, __LINE__, sizeof(int) * n);
    findlevel(l_colpointer, l_rowindex, l_rowpointer, n, &nlevel, levelPtr, levelItem);

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int i = 0; i < rhs; i++)
    {
        for (int li = 0; li < nlevel; li++)
        {

            for (int ri = levelPtr[li]; ri < levelPtr[li + 1]; ri++)
            {
                for (int j = l_colpointer[levelItem[ri]] + 1; j < l_colpointer[levelItem[ri] + 1]; j++)
                {
                    C_b[i * n + l_rowindex[j]] -= l_value[j] * C_b[i * n + levelItem[ri]];
                }
            }
        }
    }

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int i = 0; i < rhs; i++)
    {
        int index = spointer[i];
        for (int j = a_columnpointer[index]; j < a_columnpointer[index + 1]; j++)
        {
            a_value[j] = C_b[i * n + a_rowidx[j]];
        }
    }

    pangulu_free(__FILE__, __LINE__, spointer);
    pangulu_free(__FILE__, __LINE__, C_b);
    pangulu_free(__FILE__, __LINE__, D_x);
}

void pangulu_gessm_fp64_cpu_3(pangulu_smatrix *a,
                              pangulu_smatrix *l,
                              pangulu_smatrix *x)
{

    pangulu_inblock_ptr *a_columnpointer = a->columnpointer;
    pangulu_inblock_idx *a_rowidx = a->rowindex;

    calculate_type *a_value = a->value_csc;

    pangulu_inblock_ptr *l_columnpointer = l->columnpointer;
    pangulu_inblock_idx *l_rowidx = l->rowindex;
    calculate_type *l_value = l->value_csc;

    pangulu_int64_t n = a->row;

    calculate_type *C_b = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n * n);

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int i = 0; i < n; i++)
    {
        for (int j = a_columnpointer[i]; j < a_columnpointer[i + 1]; j++)
        {
            int idx = a_rowidx[j];
            C_b[i * n + idx] = a_value[j];
        }
    }

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        for (pangulu_int64_t j = a_columnpointer[i]; j < a_columnpointer[i + 1]; j++)
        {
            pangulu_inblock_idx idx = a_rowidx[j];
            for (pangulu_int64_t k = l_columnpointer[idx] + 1; k < l_columnpointer[idx + 1]; k++)
            {
                C_b[i * n + l_rowidx[k]] -= l_value[k] * C_b[i * n + a_rowidx[j]];
            }
        }
    }

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int i = 0; i < n; i++)
    {
        for (int j = a_columnpointer[i]; j < a_columnpointer[i + 1]; j++)
        {
            int idx = a_rowidx[j];
            a_value[j] = C_b[i * n + idx];
        }
    }
    pangulu_free(__FILE__, __LINE__, C_b);
}

void pangulu_gessm_fp64_cpu_4(pangulu_smatrix *a,
                              pangulu_smatrix *l,
                              pangulu_smatrix *x)
{

    pangulu_inblock_ptr *a_columnpointer = a->columnpointer;
    pangulu_inblock_idx *a_rowidx = a->rowindex;

    calculate_type *a_value = a->value_csc;

    pangulu_inblock_ptr *l_columnpointer = l->columnpointer;
    pangulu_inblock_idx *l_rowidx = l->rowindex;
    calculate_type *l_value = l->value_csc;

    pangulu_int64_t n = a->row;

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        for (pangulu_int64_t j = a_columnpointer[i]; j < a_columnpointer[i + 1]; j++)
        {
            pangulu_inblock_idx idx = a_rowidx[j];
            for (pangulu_int64_t k = l_columnpointer[idx] + 1, p = j + 1; k < l_columnpointer[idx + 1] && p < a_columnpointer[i + 1]; k++, p++)
            {
                if (l_rowidx[k] == a_rowidx[p])
                {
                    a_value[p] -= l_value[k] * a_value[j];
                }
                else
                {
                    k--;
                }
            }
        }
    }
}

void pangulu_gessm_fp64_cpu_5(pangulu_smatrix *a,
                              pangulu_smatrix *l,
                              pangulu_smatrix *x)
{

    pangulu_inblock_ptr *a_rowpointer = a->rowpointer;
    pangulu_inblock_idx *a_colindex = a->columnindex;
    calculate_type *a_value = x->value;

    pangulu_inblock_ptr *l_colpointer = l->columnpointer;
    pangulu_inblock_idx *l_rowindex = l->rowindex;
    calculate_type *l_value = l->value_csc;

    pangulu_inblock_ptr *x_rowpointer = a->rowpointer;
    pangulu_inblock_idx *x_colindex = a->columnindex;
    calculate_type *x_value = a->value;

    pangulu_int64_t n = a->row;

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int i = 0; i < n; i++) // jth column of U
    {
        for (int j = a_rowpointer[i]; j < a_rowpointer[i + 1]; j++)
        {
            pangulu_inblock_idx idx = a_colindex[j];
            temp_a_value[i * n + idx] = a_value[j]; // tranform csr to dense,only value
        }
    }

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        // x get value from a
        for (pangulu_int64_t k = x_rowpointer[i]; k < x_rowpointer[i + 1]; k++)
        {
            x_value[k] = temp_a_value[i * n + x_colindex[k]];
        }
        // update Value
        if (x_rowpointer[i] != x_rowpointer[i + 1])
        {
#pragma omp parallel for num_threads(pangu_omp_num_threads)
            for (pangulu_int64_t j = l_colpointer[i] + 1; j < l_colpointer[i + 1]; j++)
            {
                pangulu_inblock_idx idx1 = l_rowindex[j];

                for (pangulu_int64_t p = x_rowpointer[i]; p < x_rowpointer[i + 1]; p++)
                {

                    pangulu_inblock_idx idx2 = a_colindex[p];
                    temp_a_value[idx1 * n + idx2] -= l_value[j] * temp_a_value[i * n + idx2];
                }
            }
        }
    }
}

void pangulu_gessm_fp64_cpu_6(pangulu_smatrix *a,
                              pangulu_smatrix *l,
                              pangulu_smatrix *x)
{

    pangulu_inblock_ptr *a_columnpointer = a->columnpointer;
    pangulu_inblock_idx *a_rowidx = a->rowindex;

    calculate_type *a_value = a->value_csc;

    pangulu_inblock_ptr *l_columnpointer = l->columnpointer;
    pangulu_inblock_idx *l_rowidx = l->rowindex;
    calculate_type *l_value = l->value_csc;

    pangulu_int64_t n = a->row;
#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (int i = 0; i < n; i++) // jth column of U
    {
        for (int j = a_columnpointer[i]; j < a_columnpointer[i + 1]; j++)
        {
            int idx = a_rowidx[j];
            temp_a_value[i * n + idx] = a_value[j]; // tranform csr to dense,only value
        }
    }

#pragma omp parallel for num_threads(pangu_omp_num_threads)
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        for (pangulu_int64_t j = a_columnpointer[i]; j < a_columnpointer[i + 1]; j++)
        {
            pangulu_inblock_idx idx = a_rowidx[j];
            a_value[j] = temp_a_value[i * n + idx];
            for (pangulu_int64_t k = l_columnpointer[idx] + 1; k < l_columnpointer[idx + 1]; k++)
            {
                temp_a_value[i * n + l_rowidx[k]] -= l_value[k] * a_value[j];
            }
        }
    }
}

int findlevel(const pangulu_inblock_ptr *cscColPtr,
              const pangulu_inblock_idx *cscRowIdx,
              const pangulu_inblock_ptr *csrRowPtr,
              const pangulu_int64_t m,
              int *nlevel,
              int *levelPtr,
              int *levelItem)
{
    int *indegree = (int *)pangulu_malloc(__FILE__, __LINE__, m * sizeof(int));

    for (int i = 0; i < m; i++)
    {
        indegree[i] = csrRowPtr[i + 1] - csrRowPtr[i];
    }

    int ptr = 0;

    levelPtr[0] = 0;
    for (int i = 0; i < m; i++)
    {
        if (indegree[i] == 1)
        {
            levelItem[ptr] = i;
            ptr++;
        }
    }

    levelPtr[1] = ptr;

    int lvi = 1;
    while (levelPtr[lvi] != m)
    {
        for (pangulu_int64_t i = levelPtr[lvi - 1]; i < levelPtr[lvi]; i++)
        {
            int node = levelItem[i];
            for (pangulu_int64_t j = cscColPtr[node]; j < cscColPtr[node + 1]; j++)
            {
                pangulu_inblock_idx visit_node = cscRowIdx[j];
                indegree[visit_node]--;
                if (indegree[visit_node] == 1)
                {
                    levelItem[ptr] = visit_node;
                    ptr++;
                }
            }
        }
        lvi++;
        levelPtr[lvi] = ptr;
    }

    *nlevel = lvi;

    pangulu_free(__FILE__, __LINE__, indegree);

    return 0;
}

void pangulu_gessm_interface_cpu_csc(pangulu_smatrix *a,
                                     pangulu_smatrix *l,
                                     pangulu_smatrix *x)
{
    pangulu_gessm_fp64_cpu_4(a, l, x);
}

void pangulu_gessm_interface_cpu_csr(pangulu_smatrix *a,
                                     pangulu_smatrix *l,
                                     pangulu_smatrix *x)
{
#ifdef OUTPUT_MATRICES
    char out_name_B[512];
    char out_name_L[512];
    sprintf(out_name_B, "%s/%s/%d%s", OUTPUT_FILE, "gessm", gessm_number, "_gessm_B.cbd");
    sprintf(out_name_L, "%s/%s/%d%s", OUTPUT_FILE, "gessm", gessm_number, "_gessm_L.cbd");
    pangulu_binary_write_csc_pangulu_smatrix(a, out_name_B);
    pangulu_binary_write_csc_pangulu_smatrix(l, out_name_L);
    gessm_number++;
#endif

    pangulu_gessm_fp64_cpu_1(a, l, x);
}
void pangulu_gessm_interface_c_v1(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *x)
{
#ifdef GPU_OPEN
    pangulu_smatrix_cuda_memcpy_value_csc(a, a);
#endif
    pangulu_pangulu_smatrix_memcpy_value_csr_copy_length(x, a);
    pangulu_gessm_fp64_cpu_4(a, l, x);
#ifdef GPU_OPEN
    pangulu_smatrix_cuda_memcpy_to_device_value_csc(a, a);
#endif
}
void pangulu_gessm_interface_c_v2(pangulu_smatrix *a,
                                  pangulu_smatrix *l,
                                  pangulu_smatrix *x)
{
#ifdef GPU_OPEN
    pangulu_smatrix_cuda_memcpy_value_csc(a, a);
#endif
    pangulu_pangulu_smatrix_memcpy_value_csr_copy_length(x, a);
    pangulu_gessm_fp64_cpu_6(a, l, x);
#ifdef GPU_OPEN
    pangulu_smatrix_cuda_memcpy_to_device_value_csc(a, a);
#endif
}