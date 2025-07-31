#ifndef _MMIO_HIGHLEVEL_
#define _MMIO_HIGHLEVEL_
#include "mmio.h"

// read matrix infomation from mtx file
int mmio_info(sparse_index_t *m, sparse_index_t *n, sparse_pointer_t *nnz, sparse_index_t *isSymmetric, char *filename)
{
    sparse_index_t m_tmp, n_tmp;
    sparse_pointer_t nnz_tmp;

    int ret_code;
    mm_typecode matcode;
    FILE *f;

    sparse_pointer_t nnz_mtx_report;
    int is_integer = 0, is_real = 0, is_pattern = 0, is_symmetric_tmp = 0, is_complex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_pattern(matcode))
    {
        is_pattern = 1; /*printf("type = Pattern\n");*/
    }
    if (mm_is_real(matcode))
    {
        is_real = 1; /*printf("type = real\n");*/
    }
    if (mm_is_complex(matcode))
    {
        is_complex = 1; /*printf("type = real\n");*/
    }
    if (mm_is_integer(matcode))
    {
        is_integer = 1; /*printf("type = integer\n");*/
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        is_symmetric_tmp = 1;
    }
    else
    {
    }

    sparse_pointer_t *csr_row_ptr_counter = (sparse_pointer_t *)malloc(sizeof(sparse_pointer_t) * (m_tmp + 1));
    memset(csr_row_ptr_counter, 0, sizeof(sparse_pointer_t) * (m_tmp + 1));

    sparse_pointer_t *csr_rowIdx_tmp = (sparse_pointer_t *)malloc(sizeof(sparse_pointer_t) * nnz_mtx_report);
    sparse_pointer_t *csr_colIdx_tmp = (sparse_pointer_t *)malloc(sizeof(sparse_pointer_t) * nnz_mtx_report);

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (sparse_pointer_t i = 0; i < nnz_mtx_report; i++)
    {
        sparse_pointer_t idxi = 0, idxj = 0;
        double fval, fval_im;
        int ival;
        int returnvalue;

        if (is_real)
        {
            returnvalue = fscanf(f, FMT_SPARSE_POINTER_T " " FMT_SPARSE_POINTER_T " %lg\n", &idxi, &idxj, &fval);
        }
        else if (is_complex)
        {
            returnvalue = fscanf(f, FMT_SPARSE_POINTER_T " " FMT_SPARSE_POINTER_T" %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (is_integer)
        {
            returnvalue = fscanf(f, FMT_SPARSE_POINTER_T " " FMT_SPARSE_POINTER_T" %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (is_pattern)
        {
            returnvalue = fscanf(f, FMT_SPARSE_POINTER_T " " FMT_SPARSE_POINTER_T"\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csr_row_ptr_counter[idxi]++;
        csr_rowIdx_tmp[i] = idxi;
        csr_colIdx_tmp[i] = idxj;
    }

    if (f != stdin)
        fclose(f);

    if (is_symmetric_tmp)
    {
        for (sparse_pointer_t i = 0; i < nnz_mtx_report; i++)
        {
            if (csr_rowIdx_tmp[i] != csr_colIdx_tmp[i])
                csr_row_ptr_counter[csr_colIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csr_row_ptr_counter
    sparse_pointer_t old_val, new_val;

    old_val = csr_row_ptr_counter[0];
    csr_row_ptr_counter[0] = 0;
    for (sparse_pointer_t i = 1; i <= m_tmp; i++)
    {
        new_val = csr_row_ptr_counter[i];
        csr_row_ptr_counter[i] = old_val + csr_row_ptr_counter[i - 1];
        old_val = new_val;
    }

    nnz_tmp = csr_row_ptr_counter[m_tmp];

    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_tmp;
    *isSymmetric = is_symmetric_tmp;

    // free tmp space
    free(csr_colIdx_tmp);
    free(csr_rowIdx_tmp);

    return 0;
}

// read matrix infomation from mtx file
int mmio_data_csr(sparse_pointer_t *csr_row_ptr, sparse_index_t *csr_colIdx, sparse_value_t *csr_val, char *filename)
{
    sparse_index_t m_tmp, n_tmp;

    int ret_code;
    mm_typecode matcode;
    FILE *f;

    sparse_pointer_t nnz_mtx_report;
    int is_integer = 0, is_real = 0, is_pattern = 0, is_symmetric_tmp = 0, is_complex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_pattern(matcode))
    {
        is_pattern = 1; /*printf("type = Pattern\n");*/
    }
    if (mm_is_real(matcode))
    {
        is_real = 1; /*printf("type = real\n");*/
    }
    if (mm_is_complex(matcode))
    {
        is_complex = 1; /*printf("type = real\n");*/
    }
    if (mm_is_integer(matcode))
    {
        is_integer = 1; /*printf("type = integer\n");*/
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        is_symmetric_tmp = 1;
    }
    else
    {
    }

    sparse_pointer_t *csr_row_ptr_counter = (sparse_pointer_t *)malloc((m_tmp + 1) * sizeof(sparse_pointer_t));
    memset(csr_row_ptr_counter, 0, (m_tmp + 1) * sizeof(sparse_pointer_t));

    sparse_pointer_t *csr_rowIdx_tmp = (sparse_pointer_t *)malloc(nnz_mtx_report * sizeof(sparse_pointer_t));
    sparse_pointer_t *csr_colIdx_tmp = (sparse_pointer_t *)malloc(nnz_mtx_report * sizeof(sparse_pointer_t));
    sparse_value_t *csr_val_tmp = (sparse_value_t *)malloc(nnz_mtx_report * sizeof(sparse_value_t));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (sparse_pointer_t i = 0; i < nnz_mtx_report; i++)
    {
        long long idxi = 0, idxj = 0;
        double fval = 0, fval_im;
        int ival;
        int returnvalue;

        if (is_real)
        {
            returnvalue = fscanf(f, "%lld %lld %lg\n", &idxi, &idxj, &fval);
        }
        else if (is_complex)
        {
            returnvalue = fscanf(f, "%lld %lld %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (is_integer)
        {
            returnvalue = fscanf(f, "%lld %lld %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (is_pattern)
        {
            returnvalue = fscanf(f, "%lld %lld\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csr_row_ptr_counter[idxi]++;
        csr_rowIdx_tmp[i] = idxi;
        csr_colIdx_tmp[i] = idxj;
#ifdef COMPLEX_MTX
        if (is_complex)
        {
            __real__(csr_val_tmp[i]) = fval;
            __imag__(csr_val_tmp[i]) = fval_im;
        }
        else
        {
            csr_val_tmp[i] = fval;
        }
#else
        csr_val_tmp[i] = fval;
#endif
    }

    if (f != stdin)
        fclose(f);

    if (is_symmetric_tmp)
    {
        for (sparse_pointer_t i = 0; i < nnz_mtx_report; i++)
        {
            if (csr_rowIdx_tmp[i] != csr_colIdx_tmp[i])
                csr_row_ptr_counter[csr_colIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csr_row_ptr_counter
    sparse_pointer_t old_val, new_val;

    old_val = csr_row_ptr_counter[0];
    csr_row_ptr_counter[0] = 0;
    for (long i = 1; i <= m_tmp; i++)
    {
        new_val = csr_row_ptr_counter[i];
        csr_row_ptr_counter[i] = old_val + csr_row_ptr_counter[i - 1];
        old_val = new_val;
    }

    memcpy(csr_row_ptr, csr_row_ptr_counter, (m_tmp + 1) * sizeof(sparse_pointer_t));
    memset(csr_row_ptr_counter, 0, (m_tmp + 1) * sizeof(sparse_pointer_t));

    if (is_symmetric_tmp)
    {
        for (long i = 0; i < nnz_mtx_report; i++)
        {
            if (csr_rowIdx_tmp[i] != csr_colIdx_tmp[i])
            {
                long offset = csr_row_ptr[csr_rowIdx_tmp[i]] + csr_row_ptr_counter[csr_rowIdx_tmp[i]];
                csr_colIdx[offset] = csr_colIdx_tmp[i];
                csr_val[offset] = csr_val_tmp[i];
                csr_row_ptr_counter[csr_rowIdx_tmp[i]]++;

                offset = csr_row_ptr[csr_colIdx_tmp[i]] + csr_row_ptr_counter[csr_colIdx_tmp[i]];
                csr_colIdx[offset] = csr_rowIdx_tmp[i];
                csr_val[offset] = csr_val_tmp[i];
                csr_row_ptr_counter[csr_colIdx_tmp[i]]++;
            }
            else
            {
                long offset = csr_row_ptr[csr_rowIdx_tmp[i]] + csr_row_ptr_counter[csr_rowIdx_tmp[i]];
                csr_colIdx[offset] = csr_colIdx_tmp[i];
                csr_val[offset] = csr_val_tmp[i];
                csr_row_ptr_counter[csr_rowIdx_tmp[i]]++;
            }
        }
    }
    else
    {
        for (sparse_pointer_t i = 0; i < nnz_mtx_report; i++)
        {
            sparse_pointer_t offset = csr_row_ptr[csr_rowIdx_tmp[i]] + csr_row_ptr_counter[csr_rowIdx_tmp[i]];
            csr_colIdx[offset] = csr_colIdx_tmp[i];
            csr_val[offset] = csr_val_tmp[i];
            csr_row_ptr_counter[csr_rowIdx_tmp[i]]++;
        }
    }

    // free tmp space
    free(csr_colIdx_tmp);
    free(csr_val_tmp);
    free(csr_rowIdx_tmp);
    free(csr_row_ptr_counter);

    return 0;
}


// read matrix infomation from mtx file
int mmio_data_csc(sparse_pointer_t *csc_col_ptr, sparse_index_t *csc_rowIdx, sparse_value_t *csc_val, char *filename)
{
    sparse_index_t m_tmp, n_tmp;

    int ret_code;
    mm_typecode matcode;
    FILE *f;

    sparse_pointer_t nnz_mtx_report;
    int is_integer = 0, is_real = 0, is_pattern = 0, is_symmetric_tmp = 0, is_complex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_pattern(matcode))
    {
        is_pattern = 1; /*printf("type = Pattern\n");*/
    }
    if (mm_is_real(matcode))
    {
        is_real = 1; /*printf("type = real\n");*/
    }
    if (mm_is_complex(matcode))
    {
        is_complex = 1; /*printf("type = real\n");*/
    }
    if (mm_is_integer(matcode))
    {
        is_integer = 1; /*printf("type = integer\n");*/
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        is_symmetric_tmp = 1;
    }
    else
    {
    }

    sparse_pointer_t *csc_col_ptr_counter = (sparse_pointer_t *)malloc((n_tmp + 1) * sizeof(sparse_pointer_t));
    memset(csc_col_ptr_counter, 0, (n_tmp + 1) * sizeof(sparse_pointer_t));

    sparse_pointer_t *csc_rowIdx_tmp = (sparse_pointer_t *)malloc(nnz_mtx_report * sizeof(sparse_pointer_t));
    sparse_pointer_t *csc_colIdx_tmp = (sparse_pointer_t *)malloc(nnz_mtx_report * sizeof(sparse_pointer_t));
    sparse_value_t *csc_val_tmp = (sparse_value_t *)malloc(nnz_mtx_report * sizeof(sparse_value_t));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (sparse_pointer_t i = 0; i < nnz_mtx_report; i++)
    {
        long long idxi = 0, idxj = 0;
        double fval = 0, fval_im;
        int ival;
        int returnvalue;

        if (is_real)
        {
            returnvalue = fscanf(f, "%lld %lld %lg\n", &idxi, &idxj, &fval);
        }
        else if (is_complex)
        {
            returnvalue = fscanf(f, "%lld %lld %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (is_integer)
        {
            returnvalue = fscanf(f, "%lld %lld %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (is_pattern)
        {
            returnvalue = fscanf(f, "%lld %lld\n", &idxi, &idxj);
            fval = 1.0;
        }

        if ((is_real && returnvalue != 3) ||
            (is_complex && returnvalue != 4) ||
            (is_integer && returnvalue != 3) ||
            (is_pattern && returnvalue != 2)) {
            fprintf(stderr, "Failed to parse matrix entry at line %lld\n", i);
            return -5;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csc_col_ptr_counter[idxj]++;
        csc_rowIdx_tmp[i] = idxi;
        csc_colIdx_tmp[i] = idxj;
#ifdef COMPLEX_MTX
        if (is_complex)
        {
            __real__(csc_val_tmp[i]) = fval;
            __imag__(csc_val_tmp[i]) = fval_im;
        }
        else
        {
            __real__(csc_val_tmp[i]) = fval;
            __imag__(csc_val_tmp[i]) = 0;
        }
#else
        csc_val_tmp[i] = fval;
#endif
    }

    if (f != stdin)
        fclose(f);

    if (is_symmetric_tmp)
    {
        for (sparse_pointer_t i = 0; i < nnz_mtx_report; i++)
        {
            if (csc_rowIdx_tmp[i] != csc_colIdx_tmp[i])
                csc_col_ptr_counter[csc_rowIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csc_row_ptr_counter
    sparse_pointer_t old_val, new_val;

    old_val = csc_col_ptr_counter[0];
    csc_col_ptr_counter[0] = 0;
    for (long i = 1; i <= n_tmp; i++)
    {
        new_val = csc_col_ptr_counter[i];
        csc_col_ptr_counter[i] = old_val + csc_col_ptr_counter[i - 1];
        old_val = new_val;
    }

    memcpy(csc_col_ptr, csc_col_ptr_counter, (n_tmp + 1) * sizeof(sparse_pointer_t));
    memset(csc_col_ptr_counter, 0, (n_tmp + 1) * sizeof(sparse_pointer_t));

    if (is_symmetric_tmp)
    {
        for (long i = 0; i < nnz_mtx_report; i++)
        {
            if (csc_colIdx_tmp[i] != csc_rowIdx_tmp[i])
            {
                long offset = csc_col_ptr[csc_colIdx_tmp[i]] + csc_col_ptr_counter[csc_colIdx_tmp[i]];
                csc_rowIdx[offset] = csc_rowIdx_tmp[i];
                csc_val[offset] = csc_val_tmp[i];
                csc_col_ptr_counter[csc_colIdx_tmp[i]]++;

                offset = csc_col_ptr[csc_rowIdx_tmp[i]] + csc_col_ptr_counter[csc_rowIdx_tmp[i]];
                csc_rowIdx[offset] = csc_colIdx_tmp[i];
                csc_val[offset] = csc_val_tmp[i];
                csc_col_ptr_counter[csc_rowIdx_tmp[i]]++;
            }
            else
            {
                long offset = csc_col_ptr[csc_colIdx_tmp[i]] + csc_col_ptr_counter[csc_colIdx_tmp[i]];
                csc_rowIdx[offset] = csc_rowIdx_tmp[i];
                csc_val[offset] = csc_val_tmp[i];
                csc_col_ptr_counter[csc_colIdx_tmp[i]]++;
            }
        }
    }
    else
    {
        for (sparse_pointer_t i = 0; i < nnz_mtx_report; i++)
        {
            sparse_pointer_t offset = csc_col_ptr[csc_colIdx_tmp[i]] + csc_col_ptr_counter[csc_colIdx_tmp[i]];
            csc_rowIdx[offset] = csc_rowIdx_tmp[i];
            csc_val[offset] = csc_val_tmp[i];
            csc_col_ptr_counter[csc_colIdx_tmp[i]]++;
        }
    }

    // free tmp space
    free(csc_rowIdx_tmp);
    free(csc_val_tmp);
    free(csc_colIdx_tmp);
    free(csc_col_ptr_counter);

    return 0;
}
#endif
