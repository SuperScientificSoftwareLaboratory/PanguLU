#ifndef _MMIO_HIGHLEVEL_
#define _MMIO_HIGHLEVEL_

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif

#include "mmio.h"
// #include "common.h"

// read matrix infomation from mtx file
long mmio_info(long *m, long *n, long *nnz, long *isSymmetric, char *filename)
{
    long m_tmp, n_tmp, nnz_tmp;

    long ret_code;
    MM_typecode matcode;
    FILE *f;

    long nnz_mtx_report;
    long isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

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
        isPattern = 1; /*printf("type = Pattern\n");*/
    }
    if (mm_is_real(matcode))
    {
        isReal = 1; /*printf("type = real\n");*/
    }
    if (mm_is_complex(matcode))
    {
        isComplex = 1; /*printf("type = real\n");*/
    }
    if (mm_is_integer(matcode))
    {
        isInteger = 1; /*printf("type = integer\n");*/
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        isSymmetric_tmp = 1;
        // printf("input matrix is symmetric = true\n");
    }
    else
    {
        // printf("input matrix is symmetric = false\n");
    }

    long *csrRowPtr_counter = (long *)malloc((m_tmp + 1) * sizeof(long));
    memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(long));

    long *csrRowIdx_tmp = (long *)malloc(nnz_mtx_report * sizeof(long));
    long *csrColIdx_tmp = (long *)malloc(nnz_mtx_report * sizeof(long));
    // VALUE_TYPE *csrVal_tmp    = (VALUE_TYPE *)malloc(nnz_mtx_report * sizeof(VALUE_TYPE));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (long i = 0; i < nnz_mtx_report; i++)
    {
        long idxi, idxj;
        double fval, fval_im;
        long ival;
        long returnvalue;

        if (isReal)
        {
            returnvalue = fscanf(f, "%ld %ld %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%ld %ld %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%ld %ld %ld\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%ld %ld\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        // csrVal_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (long i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    long old_val, new_val;

    old_val = csrRowPtr_counter[0];
    csrRowPtr_counter[0] = 0;
    for (long i = 1; i <= m_tmp; i++)
    {
        new_val = csrRowPtr_counter[i];
        csrRowPtr_counter[i] = old_val + csrRowPtr_counter[i - 1];
        old_val = new_val;
    }

    nnz_tmp = csrRowPtr_counter[m_tmp];

    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_tmp;
    *isSymmetric = isSymmetric_tmp;

    // free tmp space
    free(csrColIdx_tmp);
    // free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}

// read matrix infomation from mtx file
long mmio_data_csr(long *csrRowPtr, int *csrColIdx, VALUE_TYPE *csrVal, char *filename)
{
    long m_tmp, n_tmp;

    long ret_code;
    MM_typecode matcode;
    FILE *f;

    long nnz_mtx_report;
    long isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

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
        isPattern = 1; /*printf("type = Pattern\n");*/
    }
    if (mm_is_real(matcode))
    {
        isReal = 1; /*printf("type = real\n");*/
    }
    if (mm_is_complex(matcode))
    {
        isComplex = 1; /*printf("type = real\n");*/
    }
    if (mm_is_integer(matcode))
    {
        isInteger = 1; /*printf("type = integer\n");*/
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        isSymmetric_tmp = 1;
        // printf("input matrix is symmetric = true\n");
    }
    else
    {
        // printf("input matrix is symmetric = false\n");
    }

    long *csrRowPtr_counter = (long *)malloc((m_tmp + 1) * sizeof(long));
    memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(long));

    long *csrRowIdx_tmp = (long *)malloc(nnz_mtx_report * sizeof(long));
    long *csrColIdx_tmp = (long *)malloc(nnz_mtx_report * sizeof(long));
    VALUE_TYPE *csrVal_tmp = (VALUE_TYPE *)malloc(nnz_mtx_report * sizeof(VALUE_TYPE));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (long i = 0; i < nnz_mtx_report; i++)
    {
        long idxi, idxj;
        double fval, fval_im;
        long ival;
        long returnvalue;

        if (isReal)
        {
            returnvalue = fscanf(f, "%ld %ld %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%ld %ld %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%ld %ld %ld\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%ld %ld\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (long i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    long old_val, new_val;

    old_val = csrRowPtr_counter[0];
    csrRowPtr_counter[0] = 0;
    for (long i = 1; i <= m_tmp; i++)
    {
        new_val = csrRowPtr_counter[i];
        csrRowPtr_counter[i] = old_val + csrRowPtr_counter[i - 1];
        old_val = new_val;
    }

    memcpy(csrRowPtr, csrRowPtr_counter, (m_tmp + 1) * sizeof(long));
    memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(long));

    if (isSymmetric_tmp)
    {
        for (long i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
            {
                long offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx[offset] = csrColIdx_tmp[i];
                csrVal[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;

                offset = csrRowPtr[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];
                csrColIdx[offset] = csrRowIdx_tmp[i];
                csrVal[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
            }
            else
            {
                long offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx[offset] = csrColIdx_tmp[i];
                csrVal[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }
    }
    else
    {
        for (long i = 0; i < nnz_mtx_report; i++)
        {
            long offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
            csrColIdx[offset] = csrColIdx_tmp[i];
            csrVal[offset] = csrVal_tmp[i];
            csrRowPtr_counter[csrRowIdx_tmp[i]]++;
        }
    }

    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}

long mmio_data_csr_complex(long *csrRowPtr, int *csrColIdx, VALUE_TYPE _Complex *csrVal, char *filename)
{
    long m_tmp, n_tmp;

    long ret_code;
    MM_typecode matcode;
    FILE *f;

    long nnz_mtx_report;
    long isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

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
        isPattern = 1; /*printf("type = Pattern\n");*/
    }
    if (mm_is_real(matcode))
    {
        isReal = 1; /*printf("type = real\n");*/
    }
    if (mm_is_complex(matcode))
    {
        isComplex = 1; /*printf("type = real\n");*/
    }
    if (mm_is_integer(matcode))
    {
        isInteger = 1; /*printf("type = integer\n");*/
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        isSymmetric_tmp = 1;
        // printf("input matrix is symmetric = true\n");
    }
    else
    {
        // printf("input matrix is symmetric = false\n");
    }

    long *csrRowPtr_counter = (long *)malloc((m_tmp + 1) * sizeof(long));
    memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(long));

    long *csrRowIdx_tmp = (long *)malloc(nnz_mtx_report * sizeof(long));
    long *csrColIdx_tmp = (long *)malloc(nnz_mtx_report * sizeof(long));
    VALUE_TYPE _Complex *csrVal_tmp = (VALUE_TYPE _Complex *)malloc(nnz_mtx_report * sizeof(VALUE_TYPE _Complex));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (long i = 0; i < nnz_mtx_report; i++)
    {
        long idxi, idxj;
        double fval = 0.0, fval_im = 0.0;
        long ival;
        long returnvalue;

        if (isReal)
        {
            returnvalue = fscanf(f, "%ld %ld %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%ld %ld %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%ld %ld %ld\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%ld %ld\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        __real__(csrVal_tmp[i]) = fval;
        __imag__(csrVal_tmp[i]) = fval_im;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (long i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    long old_val, new_val;

    old_val = csrRowPtr_counter[0];
    csrRowPtr_counter[0] = 0;
    for (long i = 1; i <= m_tmp; i++)
    {
        new_val = csrRowPtr_counter[i];
        csrRowPtr_counter[i] = old_val + csrRowPtr_counter[i - 1];
        old_val = new_val;
    }

    memcpy(csrRowPtr, csrRowPtr_counter, (m_tmp + 1) * sizeof(long));
    memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(long));

    if (isSymmetric_tmp)
    {
        for (long i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
            {
                long offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx[offset] = csrColIdx_tmp[i];
                csrVal[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;

                offset = csrRowPtr[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];
                csrColIdx[offset] = csrRowIdx_tmp[i];
                csrVal[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
            }
            else
            {
                long offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx[offset] = csrColIdx_tmp[i];
                csrVal[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }
    }
    else
    {
        for (long i = 0; i < nnz_mtx_report; i++)
        {
            long offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
            csrColIdx[offset] = csrColIdx_tmp[i];
            csrVal[offset] = csrVal_tmp[i];
            csrRowPtr_counter[csrRowIdx_tmp[i]]++;
        }
    }

    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}


// read matrix infomation from mtx file
long mmio_data_csc(long *cscColPtr, int *cscRowIdx, VALUE_TYPE *cscVal, char *filename)
{
    long m_tmp, n_tmp;

    long ret_code;
    MM_typecode matcode;
    FILE *f;

    long nnz_mtx_report;
    long isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

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
        isPattern = 1; /*printf("type = Pattern\n");*/
    }
    if (mm_is_real(matcode))
    {
        isReal = 1; /*printf("type = real\n");*/
    }
    if (mm_is_complex(matcode))
    {
        isComplex = 1; /*printf("type = real\n");*/
    }
    if (mm_is_integer(matcode))
    {
        isInteger = 1; /*printf("type = integer\n");*/
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        isSymmetric_tmp = 1;
        // printf("input matrix is symmetric = true\n");
    }
    else
    {
        // printf("input matrix is symmetric = false\n");
    }

    long *cscColPtr_counter = (long *)malloc((m_tmp + 1) * sizeof(long));
    memset(cscColPtr_counter, 0, (m_tmp + 1) * sizeof(long));

    long *cscColIdx_tmp = (long *)malloc(nnz_mtx_report * sizeof(long));
    long *cscRowIdx_tmp = (long *)malloc(nnz_mtx_report * sizeof(long));
    VALUE_TYPE *cscVal_tmp = (VALUE_TYPE *)malloc(nnz_mtx_report * sizeof(VALUE_TYPE));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (long i = 0; i < nnz_mtx_report; i++)
    {
        long idxi, idxj;
        double fval, fval_im;
        long ival;
        long returnvalue;

        if (isReal)
        {
            returnvalue = fscanf(f, "%ld %ld %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%ld %ld %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%ld %ld %ld\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%ld %ld\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        cscColPtr_counter[idxj]++;
        cscColIdx_tmp[i] = idxj;
        cscRowIdx_tmp[i] = idxi;
        cscVal_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (long i = 0; i < nnz_mtx_report; i++)
        {
            if (cscColIdx_tmp[i] != cscRowIdx_tmp[i])
                cscColPtr_counter[cscRowIdx_tmp[i]]++;
        }
    }

    // exclusive scan for cscColPtr_counter
    long old_val, new_val;

    old_val = cscColPtr_counter[0];
    cscColPtr_counter[0] = 0;
    for (long i = 1; i <= m_tmp; i++)
    {
        new_val = cscColPtr_counter[i];
        cscColPtr_counter[i] = old_val + cscColPtr_counter[i - 1];
        old_val = new_val;
    }

    memcpy(cscColPtr, cscColPtr_counter, (m_tmp + 1) * sizeof(long));
    memset(cscColPtr_counter, 0, (m_tmp + 1) * sizeof(long));

    if (isSymmetric_tmp)
    {
        for (long i = 0; i < nnz_mtx_report; i++)
        {
            if (cscColIdx_tmp[i] != cscRowIdx_tmp[i])
            {
                long offset = cscColPtr[cscColIdx_tmp[i]] + cscColPtr_counter[cscColIdx_tmp[i]];
                cscRowIdx[offset] = cscRowIdx_tmp[i];
                cscVal[offset] = cscVal_tmp[i];
                cscColPtr_counter[cscColIdx_tmp[i]]++;

                offset = cscColPtr[cscRowIdx_tmp[i]] + cscColPtr_counter[cscRowIdx_tmp[i]];
                cscRowIdx[offset] = cscColIdx_tmp[i];
                cscVal[offset] = cscVal_tmp[i];
                cscColPtr_counter[cscRowIdx_tmp[i]]++;
            }
            else
            {
                long offset = cscColPtr[cscColIdx_tmp[i]] + cscColPtr_counter[cscColIdx_tmp[i]];
                cscRowIdx[offset] = cscRowIdx_tmp[i];
                cscVal[offset] = cscVal_tmp[i];
                cscColPtr_counter[cscColIdx_tmp[i]]++;
            }
        }
    }
    else
    {
        for (long i = 0; i < nnz_mtx_report; i++)
        {
            long offset = cscColPtr[cscColIdx_tmp[i]] + cscColPtr_counter[cscColIdx_tmp[i]];
            cscRowIdx[offset] = cscRowIdx_tmp[i];
            cscVal[offset] = cscVal_tmp[i];
            cscColPtr_counter[cscColIdx_tmp[i]]++;
        }
    }

    // free tmp space
    free(cscRowIdx_tmp);
    free(cscVal_tmp);
    free(cscColIdx_tmp);
    free(cscColPtr_counter);

    return 0;
}

#endif
