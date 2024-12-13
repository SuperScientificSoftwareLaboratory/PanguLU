#include "pangulu_common.h"

void *pangulu_malloc(const char* file, pangulu_int64_t line, pangulu_int64_t size)
{
    if(size == 0){
        return NULL;
    }
    cpu_memory += size;
    if (cpu_memory > cpu_peak_memory)
    {
        cpu_peak_memory = cpu_memory;
    }
    void *malloc_address = NULL;
    malloc_address = (void *)malloc(size);
    if (malloc_address == NULL)
    {
        printf(PANGULU_E_CPU_MEM);
        pangulu_exit(1);
    }
    memset(malloc_address, 0, size);
    return malloc_address;
}

void *pangulu_realloc(const char* file, pangulu_int64_t line, void* oldptr, pangulu_int64_t size)
{
    if(size == 0){
        pangulu_free(__FILE__, __LINE__, oldptr);
        return NULL;
    }
    cpu_memory += size;
    if (cpu_memory > cpu_peak_memory)
    {
        cpu_peak_memory = cpu_memory;
    }
    void *realloc_address = NULL;
    realloc_address = (void *)realloc(oldptr, size);
    if (realloc_address == NULL)
    {
        printf(PANGULU_E_CPU_MEM);
        pangulu_exit(1);
    }
    return realloc_address;
}

void pangulu_free(const char* file, pangulu_int64_t line, void* ptr){
    if(ptr==NULL){
        return;
    }
    free(ptr);
}

pangulu_int64_t pangulu_bin_map(pangulu_int64_t nnz)
{
    pangulu_int64_t log_nnz = ceil(log2((double)nnz));
    if (log_nnz >= 14)
    {
        return 11;
    }
    else if (log_nnz > 5)
    {
        return log_nnz - 2;
    }
    else if (nnz > 1)
    {
        return 3;
    }
    else if (nnz == 1)
    {
        return 2;
    }
    else
    {
        return 1;
    }
}

void pangulu_get_pangulu_smatrix_to_u(pangulu_smatrix *s,
                                      pangulu_smatrix *u,
                                      pangulu_int64_t nb)
{
    pangulu_int64_t nnz = 0;
    for (pangulu_int64_t i = 0; i < nb; i++)
    {
        for (pangulu_int64_t j = s->rowpointer[i]; j < s->rowpointer[i + 1]; j++)
        {
            pangulu_int64_t col = s->columnindex[j];
            if (i <= col)
            {
                nnz++;
            }
        }
    }
    char *now_malloc_space = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (nb + 1) + sizeof(pangulu_inblock_idx) * nnz + sizeof(calculate_type) * nnz);
    pangulu_inblock_ptr *rowpointer = (pangulu_inblock_ptr *)now_malloc_space;
    rowpointer[0] = 0;
    for (pangulu_int64_t i = 0; i < nb; i++)
    {
        rowpointer[i + 1] = rowpointer[i];
        for (pangulu_int64_t j = s->rowpointer[i]; j < s->rowpointer[i + 1]; j++)
        {
            pangulu_int64_t col = s->columnindex[j];
            if (i <= col)
            {
                rowpointer[i + 1]++;
            }
        }
    }
    pangulu_inblock_idx *columnindex = (pangulu_inblock_idx *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (nb + 1));
    calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (nb + 1) + sizeof(pangulu_inblock_idx) * nnz);
    for (pangulu_int64_t i = 0; i < rowpointer[nb]; i++)
    {
        value[i] = 0.0;
    }
    pangulu_int64_t nzz = 0;
    for (pangulu_int64_t i = 0; i < nb; i++)
    {
        for (pangulu_int64_t j = s->rowpointer[i]; j < s->rowpointer[i + 1]; j++)
        {
            pangulu_int64_t col = s->columnindex[j];
            if (i <= col)
            {
                columnindex[nzz] = col;
                // value[nzz] = s->value[j];
                nzz++;
            }
        }
    }
    u->nnz = rowpointer[nb];
    u->rowpointer = rowpointer;
    u->columnindex = columnindex;
    u->value = value;
    u->row = nb;
    u->column = nb;
    return;
}

void pangulu_get_pangulu_smatrix_to_l(pangulu_smatrix *s,
                                      pangulu_smatrix *l,
                                      pangulu_int64_t nb)
{
    pangulu_int64_t nnz = 0;
    for (pangulu_int64_t i = 0; i < nb; i++)
    {
        for (pangulu_int64_t j = s->rowpointer[i]; j < s->rowpointer[i + 1]; j++)
        {
            pangulu_int64_t col = s->columnindex[j];
            if (i >= col)
            {
                nnz++;
            }
        }
    }
    char *now_malloc_space = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (nb + 1) + sizeof(pangulu_inblock_idx) * nnz + sizeof(calculate_type) * nnz);
    pangulu_inblock_ptr *rowpointer = (pangulu_inblock_ptr *)now_malloc_space;
    rowpointer[0] = 0;
    for (pangulu_int64_t i = 0; i < nb; i++)
    {
        rowpointer[i + 1] = rowpointer[i];
        for (pangulu_int64_t j = s->rowpointer[i]; j < s->rowpointer[i + 1]; j++)
        {
            pangulu_int64_t col = s->columnindex[j];
            if (i >= col)
            {
                rowpointer[i + 1]++;
            }
        }
    }
    pangulu_inblock_idx *columnindex = (pangulu_inblock_idx *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (nb + 1));
    calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (nb + 1) + sizeof(pangulu_inblock_idx) * nnz);
    for (pangulu_int64_t i = 0; i < rowpointer[nb]; i++)
    {
        value[i] = 0.0;
    }
    pangulu_int64_t nzz = 0;
    for (pangulu_int64_t i = 0; i < nb; i++)
    {
        for (pangulu_int64_t j = s->rowpointer[i]; j < s->rowpointer[i + 1]; j++)
        {
            pangulu_int64_t col = s->columnindex[j];
            if (i >= col)
            {
                columnindex[nzz] = col;
                // value[nzz] = s->value[j];
                nzz++;
            }
        }
    }
    l->nnz = rowpointer[nb];
    l->rowpointer = rowpointer;
    l->columnindex = columnindex;
    l->value = value;
    l->row = nb;
    l->column = nb;
    return;
}

void pangulu_smatrix_add_more_memory(pangulu_smatrix *a)
{
    // add CPU memory
    pangulu_int64_t nnzA = a->rowpointer[a->row];
    pangulu_int64_t n = a->row;
    char *now_malloc_space = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (n + 1) + sizeof(pangulu_inblock_idx) * nnzA + sizeof(calculate_type) * nnzA);

    pangulu_inblock_ptr *rowpointer = (pangulu_inblock_ptr *)now_malloc_space;
    pangulu_inblock_idx *columnindex = (pangulu_inblock_idx *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (n + 1));
    calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (n + 1) + sizeof(pangulu_inblock_idx) * nnzA);
    pangulu_inblock_ptr *csc_to_csr_index = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * nnzA);
    for (pangulu_int64_t i = 0; i < nnzA; i++)
    {
        value[i] = (calculate_type)0.0;
    }

    for (pangulu_int64_t i = 0; i < (n + 1); i++)
    {
        rowpointer[i] = 0;
    }
    for (pangulu_int64_t i = 0; i < nnzA; i++)
    {
        rowpointer[a->columnindex[i] + 1]++;
    }
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        rowpointer[i + 1] += rowpointer[i];
    }
    pangulu_inblock_ptr *index_rowpointer = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (n + 1));
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        index_rowpointer[i] = rowpointer[i];
    }
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        for (pangulu_int64_t j = a->rowpointer[i]; j < a->rowpointer[i + 1]; j++)
        {

            pangulu_int64_t col = a->columnindex[j];
            pangulu_inblock_ptr index = index_rowpointer[col];
            columnindex[index] = i;
            value[index] = a->value[j];
            csc_to_csr_index[index] = j;
            index_rowpointer[col]++;
        }
    }
    pangulu_inblock_ptr *bin_rowpointer = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * BIN_LENGTH);
    pangulu_inblock_idx *bin_rowindex = (pangulu_inblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx) * n);
    for (pangulu_int64_t i = 0; i < BIN_LENGTH; i++)
    {
        bin_rowpointer[i] = 0;
    }
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        pangulu_int64_t row_length = rowpointer[i + 1] - rowpointer[i];
        bin_rowpointer[pangulu_bin_map(row_length)]++;
    }
    pangulu_inblock_ptr *save_bin_rowpointer = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (BIN_LENGTH + 1));
    for (pangulu_int64_t i = 0; i < BIN_LENGTH - 1; i++)
    {
        bin_rowpointer[i + 1] += bin_rowpointer[i];
    }
    for (pangulu_int64_t i = 0; i < BIN_LENGTH; i++)
    {
        save_bin_rowpointer[i + 1] = bin_rowpointer[i];
    }
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        pangulu_int64_t row_length = rowpointer[i + 1] - rowpointer[i];
        pangulu_int64_t index = pangulu_bin_map(row_length);
        bin_rowindex[save_bin_rowpointer[index]] = i;
        save_bin_rowpointer[index]++;
    }
    a->bin_rowpointer = bin_rowpointer;
    a->bin_rowindex = bin_rowindex;
    a->columnpointer = rowpointer;
    a->rowindex = columnindex;
    a->value_csc = value;
    a->csc_to_csr_index = csc_to_csr_index;
    pangulu_free(__FILE__, __LINE__, index_rowpointer);
    pangulu_free(__FILE__, __LINE__, save_bin_rowpointer);

    // add GPU memory
}

void pangulu_smatrix_add_more_memory_csr(pangulu_smatrix *a)
{
    // add CPU memory

    pangulu_int64_t nnzA = a->nnz;
    pangulu_int64_t n = a->row;

    pangulu_inblock_ptr *rowpointer = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (n + 1));
    pangulu_inblock_idx *columnindex = (pangulu_inblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx) * (nnzA));
    calculate_type *value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * (nnzA));

    pangulu_inblock_ptr *csc_to_csr_index = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * nnzA);
    for (pangulu_int64_t i = 0; i < nnzA; i++)
    {
        value[i] = 0.0;
    }

    for (pangulu_int64_t i = 0; i < (n + 1); i++)
    {
        rowpointer[i] = 0;
    }
    for (pangulu_int64_t i = 0; i < nnzA; i++)
    {
        rowpointer[a->rowindex[i] + 1]++;
    }
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        rowpointer[i + 1] += rowpointer[i];
    }
    pangulu_inblock_ptr *index_rowpointer = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (n + 1));
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        index_rowpointer[i] = rowpointer[i];
    }
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        for (pangulu_int64_t j = a->columnpointer[i]; j < a->columnpointer[i + 1]; j++)
        {

            pangulu_int64_t col = a->rowindex[j];
            pangulu_int64_t index = index_rowpointer[col];
            columnindex[index] = i;
            value[index] = a->value_csc[j];
            csc_to_csr_index[j] = index;
            index_rowpointer[col]++;
        }
    }
    pangulu_inblock_ptr *bin_rowpointer = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * BIN_LENGTH);
    pangulu_inblock_idx *bin_rowindex = (pangulu_inblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_idx) * n);
    for (pangulu_int64_t i = 0; i < BIN_LENGTH; i++)
    {
        bin_rowpointer[i] = 0;
    }
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        pangulu_int64_t row_length = a->columnpointer[i + 1] - a->columnpointer[i];
        bin_rowpointer[pangulu_bin_map(row_length)]++;
    }
    pangulu_inblock_ptr *save_bin_rowpointer = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (BIN_LENGTH + 1));
    for (pangulu_int64_t i = 0; i < BIN_LENGTH - 1; i++)
    {
        bin_rowpointer[i + 1] += bin_rowpointer[i];
    }
    for (pangulu_int64_t i = 0; i < BIN_LENGTH; i++)
    {
        save_bin_rowpointer[i + 1] = bin_rowpointer[i];
    }
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        pangulu_int64_t row_length = a->columnpointer[i + 1] - a->columnpointer[i];
        pangulu_int64_t index = pangulu_bin_map(row_length);
        bin_rowindex[save_bin_rowpointer[index]] = i;
        save_bin_rowpointer[index]++;
    }
    a->bin_rowpointer = bin_rowpointer;
    a->bin_rowindex = bin_rowindex;
    a->rowpointer = rowpointer;
    a->columnindex = columnindex;
    a->value = value;
    a->csc_to_csr_index = csc_to_csr_index;
    pangulu_free(__FILE__, __LINE__, index_rowpointer);
    pangulu_free(__FILE__, __LINE__, save_bin_rowpointer);

    // add GPU memory
}

void pangulu_smatrix_add_csc(pangulu_smatrix *a)
{
    // add csc memory
    pangulu_int64_t nnzA = a->rowpointer[a->row];
    pangulu_int64_t n = a->row;
    char *now_malloc_space = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (n + 1) + sizeof(pangulu_inblock_idx) * nnzA + sizeof(calculate_type) * nnzA);

    pangulu_inblock_ptr *rowpointer = (pangulu_inblock_ptr *)now_malloc_space;
    pangulu_inblock_idx *columnindex = (pangulu_inblock_idx *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (n + 1));
    calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (n + 1) + sizeof(pangulu_inblock_idx) * nnzA);
    pangulu_inblock_ptr *csc_to_csr_index = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * nnzA);
    for (pangulu_int64_t i = 0; i < nnzA; i++)
    {
        value[i] = 0.0;
    }

    for (pangulu_int64_t i = 0; i < (n + 1); i++)
    {
        rowpointer[i] = 0;
    }
    for (pangulu_int64_t i = 0; i < nnzA; i++)
    {
        rowpointer[a->columnindex[i] + 1]++;
    }
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        rowpointer[i + 1] += rowpointer[i];
    }
    pangulu_int64_t *index_rowpointer = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (n + 1));
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        index_rowpointer[i] = rowpointer[i];
    }
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        for (pangulu_int64_t j = a->rowpointer[i]; j < a->rowpointer[i + 1]; j++)
        {

            pangulu_int64_t col = a->columnindex[j];
            pangulu_int64_t index = index_rowpointer[col];
            columnindex[index] = i;
            value[index] = a->value[j];
            csc_to_csr_index[index] = j;
            index_rowpointer[col]++;
        }
    }
    a->columnpointer = rowpointer;
    a->rowindex = columnindex;
    a->value_csc = value;
    a->csc_to_csr_index = csc_to_csr_index;
    pangulu_free(__FILE__, __LINE__, index_rowpointer);
}

// void pangulu_origin_smatrix_add_csc(pangulu_origin_smatrix *a)
// {
//     // add csc memory
//     pangulu_exblock_ptr nnzA = a->rowpointer[a->row];
//     pangulu_exblock_idx n = a->row;
//     char *now_malloc_space = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1) + sizeof(pangulu_exblock_idx) * nnzA + sizeof(calculate_type) * nnzA);

//     pangulu_exblock_ptr *rowpointer = (pangulu_exblock_ptr *)now_malloc_space;
//     pangulu_exblock_idx *columnindex = (pangulu_exblock_idx *)(now_malloc_space + sizeof(pangulu_exblock_ptr) * (n + 1));
//     calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(pangulu_exblock_ptr) * (n + 1) + sizeof(pangulu_exblock_idx) * nnzA);
//     pangulu_exblock_ptr *csc_to_csr_index = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * nnzA);
//     for (pangulu_exblock_ptr i = 0; i < nnzA; i++)
//     {
//         value[i] = 0.0;
//     }

//     for (pangulu_exblock_idx i = 0; i < (n + 1); i++)
//     {
//         rowpointer[i] = 0;
//     }
//     for (pangulu_exblock_ptr i = 0; i < nnzA; i++)
//     {
//         rowpointer[a->columnindex[i] + 1]++;
//     }
//     for (pangulu_exblock_idx i = 0; i < n; i++)
//     {
//         rowpointer[i + 1] += rowpointer[i];
//     }
//     pangulu_exblock_ptr *index_rowpointer = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1));
//     for (pangulu_exblock_idx i = 0; i < n; i++)
//     {
//         index_rowpointer[i] = rowpointer[i];
//     }
//     for (pangulu_exblock_idx i = 0; i < n; i++)
//     {
//         for (pangulu_exblock_ptr j = a->rowpointer[i]; j < a->rowpointer[i + 1]; j++)
//         {

//             pangulu_exblock_idx col = a->columnindex[j];
//             pangulu_exblock_ptr index = index_rowpointer[col];
//             columnindex[index] = i;
//             value[index] = a->value[j];
//             csc_to_csr_index[index] = j;
//             index_rowpointer[col]++;
//         }
//     }
//     a->columnpointer = rowpointer;
//     a->rowindex = columnindex;
//     a->value_csc = value;
//     a->csc_to_csr_index = csc_to_csr_index;
//     pangulu_free(__FILE__, __LINE__, index_rowpointer);
// }

void pangulu_origin_smatrix_add_csc(pangulu_origin_smatrix *a)
{
    pangulu_exblock_ptr nnzA = a->rowpointer[a->row];
    pangulu_exblock_idx n = a->row;
    char *now_malloc_space = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1) + sizeof(pangulu_exblock_idx) * nnzA + sizeof(calculate_type) * nnzA);
    pangulu_exblock_ptr *columnpointer = (pangulu_exblock_ptr *)now_malloc_space;
    pangulu_exblock_idx *rowindex = (pangulu_exblock_idx *)(now_malloc_space + sizeof(pangulu_exblock_ptr) * (n + 1));
    calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(pangulu_exblock_ptr) * (n + 1) + sizeof(pangulu_exblock_idx) * nnzA);
    pangulu_exblock_ptr *csc_to_csr_index = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * nnzA);
    pangulu_exblock_ptr *index_columnpointer = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1));

    #pragma omp parallel for
    for (pangulu_exblock_idx i = 0; i < (n + 1); i++)
    {
        columnpointer[i] = 0;
    }
    for (pangulu_exblock_ptr i = 0; i < nnzA; i++)
    {
        columnpointer[a->columnindex[i] + 1]++;
    }
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        columnpointer[i + 1] += columnpointer[i];
    }
    #pragma omp parallel for
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        index_columnpointer[i] = columnpointer[i];
    }
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        for (pangulu_exblock_ptr j = a->rowpointer[i]; j < a->rowpointer[i + 1]; j++)
        {
            pangulu_exblock_idx col = a->columnindex[j];
            pangulu_exblock_ptr index = index_columnpointer[col];
            rowindex[index] = i;
            value[index] = a->value[j];
            csc_to_csr_index[index] = j;
            index_columnpointer[col]++;
        }
    }
    a->columnpointer = columnpointer;
    a->rowindex = rowindex;
    a->value_csc = value;
    a->csc_to_csr_index = csc_to_csr_index;
    pangulu_free(__FILE__, __LINE__, index_columnpointer);
}

void pangulu_malloc_pangulu_smatrix_csc(pangulu_smatrix *s,
                                        pangulu_int64_t nb, pangulu_int64_t *save_columnpointer)
{

    pangulu_int64_t nnzA = save_columnpointer[nb];
    char *now_malloc_space = (char *)pangulu_malloc(__FILE__, __LINE__, (sizeof(pangulu_inblock_ptr) * (nb + 1)) + (sizeof(pangulu_inblock_idx) * nnzA) + (sizeof(calculate_type) * nnzA));
    pangulu_inblock_ptr *columnpointer = (pangulu_inblock_ptr *)now_malloc_space;
    pangulu_inblock_idx *rowindex = (pangulu_inblock_idx *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (nb + 1));
    calculate_type *value_csc = (calculate_type *)(now_malloc_space + (sizeof(pangulu_inblock_ptr) * (nb + 1) + sizeof(pangulu_inblock_idx) * nnzA));

    s->row = nb;
    s->column = nb;
    for (pangulu_int64_t i = 0; i <= nb; i++)
    {
        columnpointer[i] = save_columnpointer[i];
    }
    for (pangulu_int64_t i = 0; i < nnzA; i++)
    {
        rowindex[i] = 0;
    }

    for (pangulu_int64_t i = 0; i < nnzA; i++)
    {
        value_csc[i] = 0.0;
    }
    s->nnz = nnzA;
    s->columnpointer = columnpointer;
    s->rowindex = rowindex;
    s->value_csc = value_csc;
}

void pangulu_malloc_pangulu_smatrix_csc_value(pangulu_smatrix *s,
                                              pangulu_smatrix *b)
{
    b->row = s->row;
    b->column = s->column;
    b->nnz = s->nnz;
    calculate_type *value_csc = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * s->nnz);
    b->value_csc = value_csc;
}

void pangulu_malloc_pangulu_smatrix_csr_value(pangulu_smatrix *s,
                                              pangulu_smatrix *b)
{
    b->row = s->row;
    b->column = s->column;
    b->nnz = s->nnz;
    calculate_type *value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * s->nnz);
    b->value = value;
}

void pangulu_malloc_pangulu_smatrix_nnz_csc(pangulu_smatrix *s,
                                            pangulu_int64_t nb, pangulu_int64_t nnz)
{
    char *now_malloc_space = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (nb + 1) + sizeof(pangulu_inblock_idx) * nnz + sizeof(calculate_type) * nnz);
    pangulu_inblock_ptr *rowpointer = (pangulu_inblock_ptr *)now_malloc_space;
    s->row = nb;
    s->column = nb;
    pangulu_inblock_idx *columnindex = (pangulu_inblock_idx *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (nb + 1));
    calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (nb + 1) + sizeof(pangulu_inblock_idx) * nnz);
    s->columnpointer = rowpointer;
    s->rowindex = columnindex;
    s->value_csc = value;
    s->nnz = nnz;
}

void pangulu_malloc_pangulu_smatrix_nnz_csr(pangulu_smatrix *s,
                                            pangulu_int64_t nb, pangulu_int64_t nnz)
{
    char *now_malloc_space = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * (nb + 1) + sizeof(pangulu_inblock_idx) * nnz + sizeof(calculate_type) * nnz);
    pangulu_inblock_ptr *rowpointer = (pangulu_inblock_ptr *)now_malloc_space;
    s->row = nb;
    s->column = nb;
    pangulu_inblock_idx *columnindex = (pangulu_inblock_idx *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (nb + 1));
    calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(pangulu_inblock_ptr) * (nb + 1) + sizeof(pangulu_inblock_idx) * nnz);
    s->rowpointer = rowpointer;
    s->columnindex = columnindex;
    s->value = value;
    s->nnz = nnz;
}

void pangulu_malloc_pangulu_smatrix_value_csc(pangulu_smatrix *s, pangulu_int64_t nnz)
{
    calculate_type *value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz);
    s->value_csc = value;
    s->nnz = nnz;
}

void pangulu_malloc_pangulu_smatrix_value_csr(pangulu_smatrix *s, pangulu_int64_t nnz)
{
    calculate_type *value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz);
    s->value = value;
    s->nnz = nnz;
}

void pangulu_smatrix_add_memory_u(pangulu_smatrix *u)
{
    pangulu_int32_t *nnzU = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * (u->row));
    for (pangulu_int64_t j = 0; j < u->row; j++)
    {
        nnzU[j] = u->columnpointer[j + 1] - u->columnpointer[j] - 1;
    }
    u->nnzu = nnzU;
}

#ifndef GPU_OPEN

void pangulu_malloc_smatrix_level(pangulu_smatrix *a)
{
    pangulu_int64_t n = a->row;
    pangulu_int64_t *level_size = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (n + 1));
    pangulu_int64_t *level_idx = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * (n + 1));
    a->level_size = level_size;
    a->level_idx = level_idx;
    a->num_lev = 0;
}

#endif
