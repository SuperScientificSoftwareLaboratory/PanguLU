#include "pangulu_common.h"

void pangulu_convert_csr_to_csc_block(
    int free_csrmatrix,
    pangulu_inblock_idx n,
    pangulu_inblock_ptr **csr_pointer,
    pangulu_inblock_idx **csr_index,
    calculate_type **csr_value,
    pangulu_inblock_ptr **csc_pointer,
    pangulu_inblock_idx **csc_index,
    calculate_type **csc_value)
{
    pangulu_inblock_ptr *rowpointer = *csr_pointer;
    pangulu_inblock_idx *columnindex = *csr_index;
    calculate_type *value = NULL;
    if (csr_value)
    {
        value = *csr_value;
    }
    pangulu_inblock_ptr nnz = rowpointer[n];
    pangulu_inblock_ptr *columnpointer = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (n + 1));
    pangulu_inblock_ptr *aid_ptr_arr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (n + 1));
    pangulu_inblock_idx *rowindex = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx) * nnz);
    calculate_type *value_csc = NULL;
    if (value)
    {
        value_csc = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz);
    }
    memset(columnpointer, 0, sizeof(pangulu_inblock_ptr) * (n + 1));
    for (pangulu_inblock_idx row = 0; row < n; row++)
    {
        for (pangulu_inblock_ptr idx = rowpointer[row]; idx < rowpointer[row + 1]; idx++)
        {
            pangulu_inblock_idx col = columnindex[idx];
            columnpointer[col + 1]++;
        }
    }
    for (pangulu_inblock_idx col = 0; col < n; col++)
    {
        columnpointer[col + 1] += columnpointer[col];
    }
    memcpy(aid_ptr_arr, columnpointer, sizeof(pangulu_inblock_ptr) * (n + 1));
    for (pangulu_inblock_idx row = 0; row < n; row++)
    {
        for (pangulu_inblock_ptr idx = rowpointer[row]; idx < rowpointer[row + 1]; idx++)
        {
            pangulu_inblock_idx col = columnindex[idx];
            rowindex[aid_ptr_arr[col]] = row;
            if (value)
            {
                value_csc[aid_ptr_arr[col]] = value[idx];
            }
            aid_ptr_arr[col]++;
        }
    }
    pangulu_free(__FILE__, __LINE__, aid_ptr_arr);
    if (free_csrmatrix)
    {
        pangulu_free(__FILE__, __LINE__, *csr_pointer);
        pangulu_free(__FILE__, __LINE__, *csr_index);
        pangulu_free(__FILE__, __LINE__, *csr_value);
        *csr_pointer = NULL;
        *csr_index = NULL;
        *csr_value = NULL;
    }
    *csc_pointer = columnpointer;
    *csc_index = rowindex;
    if (csc_value)
    {
        *csc_value = value_csc;
    }
}

void pangulu_convert_csr_to_csc_block_with_index(
    pangulu_exblock_ptr n,
    pangulu_exblock_ptr *in_pointer,
    pangulu_exblock_idx *in_index,
    pangulu_exblock_ptr *out_pointer,
    pangulu_exblock_idx *out_index,
    pangulu_exblock_ptr *out_csr_to_csc)
{
    for (pangulu_exblock_ptr i = 0; i < n; i++)
    {
        for (pangulu_exblock_ptr j = in_pointer[i]; j < in_pointer[i + 1]; j++)
        {
            pangulu_exblock_idx index = in_index[j];
            out_pointer[index + 1]++;
        }
    }
    for (pangulu_exblock_ptr i = 0; i < n; i++)
        out_pointer[i + 1] += out_pointer[i];
    pangulu_exblock_ptr *out_pointer_tmp = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1));
    memcpy(out_pointer_tmp, out_pointer, sizeof(pangulu_exblock_ptr) * (n + 1));

    for (pangulu_exblock_ptr i = 0; i < n; i++)
    {
        for (pangulu_exblock_ptr j = in_pointer[i]; j < in_pointer[i + 1]; j++)
        {
            pangulu_exblock_idx index = in_index[j];
            out_index[out_pointer_tmp[index]] = i;
            out_csr_to_csc[out_pointer_tmp[index]] = j;
            out_pointer_tmp[index]++;
        }
    }
    pangulu_free(__FILE__, __LINE__, out_pointer_tmp);
}

void pangulu_convert_csr_to_csc(
    int free_csrmatrix,
    pangulu_exblock_idx n,
    pangulu_exblock_ptr **csr_pointer,
    pangulu_exblock_idx **csr_index,
    calculate_type **csr_value,
    pangulu_exblock_ptr **csc_pointer,
    pangulu_exblock_idx **csc_index,
    calculate_type **csc_value)
{
    pangulu_exblock_ptr *rowpointer = *csr_pointer;
    pangulu_exblock_idx *columnindex = *csr_index;
    calculate_type *value = NULL;
    if (csr_value)
    {
        value = *csr_value;
    }
    pangulu_exblock_ptr nnz = rowpointer[n];
    pangulu_exblock_ptr *columnpointer = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1));
    pangulu_exblock_ptr *aid_ptr_arr = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1));
    pangulu_exblock_idx *rowindex = pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * nnz);
    calculate_type *value_csc = NULL;
    if (value)
    {
        value_csc = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz);
    }
    memset(columnpointer, 0, sizeof(pangulu_exblock_ptr) * (n + 1));
    for (pangulu_exblock_idx row = 0; row < n; row++)
    {
        for (pangulu_exblock_ptr idx = rowpointer[row]; idx < rowpointer[row + 1]; idx++)
        {
            pangulu_exblock_idx col = columnindex[idx];
            columnpointer[col + 1]++;
        }
    }
    for (pangulu_exblock_idx col = 0; col < n; col++)
    {
        columnpointer[col + 1] += columnpointer[col];
    }
    memcpy(aid_ptr_arr, columnpointer, sizeof(pangulu_exblock_ptr) * (n + 1));
    for (pangulu_exblock_idx row = 0; row < n; row++)
    {
        for (pangulu_exblock_ptr idx = rowpointer[row]; idx < rowpointer[row + 1]; idx++)
        {
            pangulu_exblock_idx col = columnindex[idx];
            rowindex[aid_ptr_arr[col]] = row;
            if (value)
            {
                value_csc[aid_ptr_arr[col]] = value[idx];
            }
            aid_ptr_arr[col]++;
        }
    }
    pangulu_free(__FILE__, __LINE__, aid_ptr_arr);
    if (free_csrmatrix)
    {
        pangulu_free(__FILE__, __LINE__, *csr_pointer);
        pangulu_free(__FILE__, __LINE__, *csr_index);
        pangulu_free(__FILE__, __LINE__, *csr_value);
        *csr_pointer = NULL;
        *csr_index = NULL;
        *csr_value = NULL;
    }
    *csc_pointer = columnpointer;
    *csc_index = rowindex;
    if (csc_value)
    {
        *csc_value = value_csc;
    }
}
void pangulu_convert_halfsymcsc_to_csc_struct(
    int free_halfmatrix,
    int if_colsort,
    pangulu_exblock_idx n,
    pangulu_exblock_ptr **half_csc_pointer,
    pangulu_exblock_idx **half_csc_index,
    pangulu_exblock_ptr **csc_pointer,
    pangulu_exblock_idx **csc_index)
{
    pangulu_exblock_ptr *L_colptr = *half_csc_pointer;
    pangulu_exblock_idx *L_rowidx = *half_csc_index;
    pangulu_exblock_ptr nnzL = L_colptr[n];
    pangulu_exblock_idx ptrsize = n + 1;
    pangulu_exblock_ptr *UT_colptr = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * ptrsize);
    pangulu_exblock_idx *UT_rowidx = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * (nnzL));
    memset(UT_colptr, 0, ptrsize * sizeof(pangulu_exblock_ptr));
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        for (pangulu_exblock_ptr j = L_colptr[i]; j < L_colptr[i + 1]; j++)
        {
            pangulu_exblock_idx index = L_rowidx[j];
            if (index != i)
                UT_colptr[index + 1]++;
        }
    }
    pangulu_exblock_ptr *UT_colptr_tmp = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * ptrsize);
    for (pangulu_exblock_idx i = 0; i < n; i++)
        UT_colptr[i + 1] += UT_colptr[i];
    memcpy(UT_colptr_tmp, UT_colptr, (ptrsize) * sizeof(pangulu_exblock_ptr));
    pangulu_exblock_ptr nnzU = 0;
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        for (pangulu_exblock_ptr j = L_colptr[i]; j < L_colptr[i + 1]; j++)
        {
            pangulu_exblock_idx index = L_rowidx[j];
            if (index != i)
            {
                UT_rowidx[UT_colptr_tmp[index]] = i;
                nnzU++;
                UT_colptr_tmp[index]++;
            }
        }
    }
    pangulu_exblock_ptr *full_colptr = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * ptrsize);
    pangulu_exblock_idx *full_rowidx = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * (nnzL * 2 - n));
    full_colptr[0] = 0;
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        pangulu_exblock_ptr lenL = L_colptr[i + 1] - L_colptr[i];
        pangulu_exblock_ptr lenU = UT_colptr[i + 1] - UT_colptr[i];
        full_colptr[i + 1] = full_colptr[i];
        full_colptr[i + 1] += lenL + lenU;

        memcpy(&full_rowidx[full_colptr[i]], &UT_rowidx[UT_colptr[i]], lenU * sizeof(pangulu_exblock_idx));
        memcpy(&full_rowidx[full_colptr[i] + lenU], &L_rowidx[L_colptr[i]], lenL * sizeof(pangulu_exblock_idx));
    }
    *csc_pointer = full_colptr;
    *csc_index = full_rowidx;
    pangulu_free(__FILE__, __LINE__, UT_colptr);
    pangulu_free(__FILE__, __LINE__, UT_rowidx);
    pangulu_free(__FILE__, __LINE__, UT_colptr_tmp);
}

void pangulu_convert_block_fill_value_to_struct(
    pangulu_int32_t p,
    pangulu_int32_t q,
    pangulu_int32_t rank,
    pangulu_exblock_idx n,
    pangulu_inblock_idx nb,
    pangulu_exblock_ptr *value_block_pointer,
    pangulu_exblock_idx *value_block_index,
    pangulu_inblock_ptr *value_block_nnzptr,
    pangulu_inblock_ptr **value_bcsc_inblock_pointers,
    pangulu_inblock_idx **value_bcsc_inblock_indeces,
    calculate_type **value_bcsc_values,
    pangulu_exblock_ptr *nondiag_block_colpointer,
    pangulu_exblock_idx *nondiag_block_rowindex,
    pangulu_inblock_ptr **nondiag_colpointers,
    pangulu_inblock_idx **nondiag_rowindeces,
    calculate_type **nondiag_values,
    pangulu_uint64_t *diag_uniaddr,
    pangulu_inblock_ptr **diag_upper_rowpointers,
    pangulu_inblock_idx **diag_upper_colindeces,
    calculate_type **diag_upper_values,
    pangulu_inblock_ptr **diag_lower_colpointers,
    pangulu_inblock_idx **diag_lower_rowindeces,
    calculate_type **diag_lower_values)
{
    struct timeval start_time;
    pangulu_time_start(&start_time);

    pangulu_exblock_idx block_length = PANGULU_ICEIL(n, nb);

#pragma omp parallel for schedule(dynamic)
    for (pangulu_exblock_idx sp = 0; sp < block_length; sp++)
    {
        pangulu_exblock_ptr ssi = nondiag_block_colpointer[sp];
        for (pangulu_exblock_ptr vsi = value_block_pointer[sp]; vsi < value_block_pointer[sp + 1]; vsi++)
        {
            if (value_block_index[vsi] == sp)
            {
                pangulu_exblock_idx level = sp;
                pangulu_uint64_t dsi = diag_uniaddr[level];
                if (dsi == 0)
                {
                    continue;
                }
                dsi--;
                memset(diag_lower_values[dsi], 0, sizeof(calculate_type) * diag_lower_colpointers[dsi][nb]);
                for (pangulu_exblock_idx ip = 0; ip < nb; ip++)
                {
                    pangulu_inblock_ptr sii = diag_lower_colpointers[dsi][ip];
                    for (pangulu_exblock_ptr vii = value_bcsc_inblock_pointers[vsi][ip]; vii < value_bcsc_inblock_pointers[vsi][ip + 1]; vii++)
                    {
                        if (diag_lower_rowindeces[dsi][sii] > value_bcsc_inblock_indeces[vsi][vii])
                        {
                            continue;
                        }
                        while ((diag_lower_rowindeces[dsi][sii] < value_bcsc_inblock_indeces[vsi][vii]) && (sii < diag_lower_colpointers[dsi][ip + 1]))
                        {
                            sii++;
                        }
                        if (sii >= diag_lower_colpointers[dsi][ip + 1])
                        {
                            break;
                        }
                        diag_lower_values[dsi][sii] = value_bcsc_values[vsi][vii];
                    }
                }
                memset(diag_upper_values[dsi], 0, sizeof(calculate_type) * diag_upper_rowpointers[dsi][nb]);
                for (pangulu_exblock_idx col = 0; col < nb; col++)
                {
                    for (pangulu_exblock_ptr vii = value_bcsc_inblock_pointers[vsi][col]; vii < value_bcsc_inblock_pointers[vsi][col + 1]; vii++)
                    {
                        pangulu_inblock_idx row = value_bcsc_inblock_indeces[vsi][vii];
                        if (row <= col)
                        {
                            pangulu_int32_t sii = pangulu_binarysearch_inblk(diag_upper_colindeces[dsi], diag_upper_rowpointers[dsi][row], diag_upper_rowpointers[dsi][row + 1], col);
                            if (sii != -1)
                            {
                                diag_upper_values[dsi][sii] = value_bcsc_values[vsi][vii];
                            }
                        }
                    }
                }
                continue;
            }
            while ((nondiag_block_rowindex[ssi] != value_block_index[vsi]) && (ssi < nondiag_block_colpointer[sp + 1]))
            {
                memset(nondiag_values[ssi], 0, sizeof(calculate_type) * nondiag_colpointers[ssi][nb]);
                ssi++;
            }
            memset(nondiag_values[ssi], 0, sizeof(calculate_type) * nondiag_colpointers[ssi][nb]);
            for (pangulu_exblock_idx ip = 0; ip < nb; ip++)
            {
                pangulu_inblock_ptr sii = nondiag_colpointers[ssi][ip];
                for (pangulu_exblock_ptr vii = value_bcsc_inblock_pointers[vsi][ip]; vii < value_bcsc_inblock_pointers[vsi][ip + 1]; vii++)
                {
                    while ((nondiag_rowindeces[ssi][sii] != value_bcsc_inblock_indeces[vsi][vii]) && (sii < nondiag_colpointers[ssi][ip + 1]))
                    {
                        sii++;
                    }
                    if (sii >= nondiag_colpointers[ssi][ip + 1])
                    {
                        break;
                    }
                    nondiag_values[ssi][sii] = value_bcsc_values[vsi][vii];
                }
            }
            ssi++;
        }
    }
}

void pangulu_convert_bcsc_to_digestcoo(
    pangulu_exblock_idx block_length,
    const pangulu_exblock_ptr *bcsc_struct_pointer,
    const pangulu_exblock_idx *bcsc_struct_index,
    const pangulu_exblock_ptr *bcsc_struct_nnzptr,
    pangulu_digest_coo_t *digest_info)
{
    for (pangulu_exblock_idx bcol = 0; bcol < block_length; bcol++)
    {
        for (pangulu_exblock_ptr bidx = bcsc_struct_pointer[bcol]; bidx < bcsc_struct_pointer[bcol + 1]; bidx++)
        {
            pangulu_exblock_idx brow = bcsc_struct_index[bidx];
            pangulu_exblock_ptr bnnz = bcsc_struct_nnzptr[bidx + 1] - bcsc_struct_nnzptr[bidx];
            digest_info[bidx].row = brow;
            digest_info[bidx].col = bcol;
            digest_info[bidx].nnz = bnnz;
        }
    }
}
