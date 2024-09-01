typedef unsigned long long int sparse_pointer_t;
#define MPI_SPARSE_POINTER_T MPI_UNSIGNED_LONG_LONG
#define FMT_SPARSE_POINTER_T "%llu"

typedef unsigned int sparse_index_t;
#define MPI_SPARSE_INDEX_T MPI_UNSIGNED
#define FMT_SPARSE_INDEX_T "%u"

#if defined(CALCULATE_TYPE_R64)
typedef double sparse_value_t;
#elif defined(CALCULATE_TYPE_R32)
typedef float sparse_value_t;
#elif defined(CALCULATE_TYPE_CR64)
typedef double _Complex sparse_value_t;
typedef double sparse_value_real_t;
#define COMPLEX_MTX
#elif defined(CALCULATE_TYPE_CR32)
typedef float _Complex sparse_value_t;
typedef float sparse_value_real_t;
#define COMPLEX_MTX
#else
#error[PanguLU Compile Error] Unknown value type. Set -DCALCULATE_TYPE_CR64 or -DCALCULATE_TYPE_R64 or -DCALCULATE_TYPE_CR32 or -DCALCULATE_TYPE_R32 in compile command line.
#endif

#include "../include/pangulu.h"
#include <sys/resource.h>
#include <getopt.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include "mmio_highlevel.h"

#ifdef COMPLEX_MTX
sparse_value_real_t complex_fabs(sparse_value_t x)
{
    return sqrt(__real__(x) * __real__(x) + __imag__(x) * __imag__(x));
}

sparse_value_t complex_sqrt(sparse_value_t x)
{
    sparse_value_t y;
    __real__(y) = sqrt(complex_fabs(x) + __real__(x)) / sqrt(2);
    __imag__(y) = (sqrt(complex_fabs(x) - __real__(x)) / sqrt(2)) * (__imag__(x) > 0 ? 1 : __imag__(x) == 0 ? 0
                                                                                                            : -1);
    return y;
}
#endif

void read_command_params(int argc, char **argv, char *mtx_name, char *rhs_name, int *nb)
{
    int c;
    extern char *optarg;
    while ((c = getopt(argc, argv, "nb:f:r:")) != EOF)
    {
        switch (c)
        {
        case 'b':
            *nb = atoi(optarg);
            continue;
        case 'f':
            strcpy(mtx_name, optarg);
            continue;
        case 'r':
            strcpy(rhs_name, optarg);
            continue;
        }
    }
    if ((nb) == 0)
    {
        printf("Error : nb is 0\n");
        exit(1);
    }
}

int main(int ARGC, char **ARGV)
{
    // Step 1: Create varibles, initialize MPI environment.
    int provided = 0;
    int rank = 0, size = 0;
    int nb = 0;
    MPI_Init_thread(&ARGC, &ARGV, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    sparse_index_t m = 0, n = 0, is_sym = 0;
    sparse_pointer_t nnz;
    sparse_pointer_t *rowptr = NULL;
    sparse_index_t *colidx = NULL;
    sparse_value_t *value = NULL;
    sparse_value_t *sol = NULL;
    sparse_value_t *rhs = NULL;

    // Step 2: Read matrix and rhs vectors.
    if (rank == 0)
    {
        char mtx_name[200] = {'\0'};
        char rhs_name[200] = {'\0'};
        read_command_params(ARGC, ARGV, mtx_name, rhs_name, &nb);

        printf("Reading matrix %s\n", mtx_name);
        mmio_info(&m, &n, &nnz, &is_sym, mtx_name);
        rowptr = (sparse_pointer_t *)malloc(sizeof(sparse_pointer_t) * (n + 1));
        colidx = (sparse_index_t *)malloc(sizeof(sparse_index_t) * nnz);
        value = (sparse_value_t *)malloc(sizeof(sparse_value_t) * nnz);
        mmio_data_csr(rowptr, colidx, value, mtx_name);
        printf("Read mtx done.\n");

        sol = (sparse_value_t *)malloc(sizeof(sparse_value_t) * n);
        rhs = (sparse_value_t *)malloc(sizeof(sparse_value_t) * n);
        for (int i = 0; i < n; i++)
        {
            rhs[i] = 0;
            for (sparse_pointer_t j = rowptr[i]; j < rowptr[i + 1]; j++)
            {
                rhs[i] += value[j];
            }
            sol[i] = rhs[i];
        }
        printf("Generate rhs done.\n");
    }
    MPI_Bcast(&n, 1, MPI_SPARSE_INDEX_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nb, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Step 3: Initialize PanguLU solver.
    pangulu_init_options init_options;
    init_options.nb = nb;
    init_options.nthread = 20;
    void *pangulu_handle;
    pangulu_init(n, nnz, rowptr, colidx, value, &init_options, &pangulu_handle);

    // Step 4: Execute LU factorization.
    pangulu_gstrf_options gstrf_options;
    pangulu_gstrf(&gstrf_options, &pangulu_handle);

    // Step 5: Execute triangle solve using factorize results.
    pangulu_gstrs_options gstrs_options;
    pangulu_gstrs(sol, &gstrs_options, &pangulu_handle);
    MPI_Barrier(MPI_COMM_WORLD);

    // Step 6: Check the answer.
    sparse_value_t *rhs_computed;
    if (rank == 0)
    {
        // Step 6.1: Calculate rhs_computed = A * x.
        rhs_computed = (sparse_value_t *)malloc(sizeof(sparse_value_t) * n);
        for (int i = 0; i < n; i++)
        {
            rhs_computed[i] = 0.0;
            sparse_value_t c = 0.0;
            for (sparse_pointer_t j = rowptr[i]; j < rowptr[i + 1]; j++)
            {
                sparse_value_t num = value[j] * sol[colidx[j]];
                sparse_value_t z = num - c;
                sparse_value_t t = rhs_computed[i] + z;
                c = (t - rhs_computed[i]) - z;
                rhs_computed[i] = t;
            }
        }

        // Step 6.2: Calculate residual residual = rhs_comuted - rhs.
        sparse_value_t *residual = rhs_computed;
        for (int i = 0; i < n; i++)
        {
            residual[i] = rhs_computed[i] - rhs[i];
        }

        sparse_value_t sum, c;
        // Step 6.3: Calculte norm2 of residual.
        sum = 0.0;
        c = 0.0;
        for (int i = 0; i < n; i++)
        {
            sparse_value_t num = residual[i] * residual[i];
            sparse_value_t z = num - c;
            sparse_value_t t = sum + z;
            c = (t - sum) - z;
            sum = t;
        }
#ifdef COMPLEX_MTX
        sparse_value_real_t residual_norm2 = complex_fabs(complex_sqrt(sum));
#else
        sparse_value_t residual_norm2 = sqrt(sum);
#endif

        // Step 6.4: Calculte norm2 of original rhs.
        sum = 0.0;
        c = 0.0;
        for (int i = 0; i < n; i++)
        {
            sparse_value_t num = rhs[i] * rhs[i];
            sparse_value_t z = num - c;
            sparse_value_t t = sum + z;
            c = (t - sum) - z;
            sum = t;
        }
#ifdef COMPLEX_MTX
        sparse_value_real_t rhs_norm2 = complex_fabs(complex_sqrt(sum));
#else
        sparse_value_t rhs_norm2 = sqrt(sum);
#endif

        // Step 6.5: Calculate relative residual.
        double relative_residual = residual_norm2 / rhs_norm2;
        printf("|| Ax - b || / || b || = %le\n", relative_residual);
    }

    // Step 7: Clean and finalize.
    pangulu_finalize(&pangulu_handle);
    if (rank == 0)
    {
        free(rowptr);
        free(colidx);
        free(value);
        free(sol);
        free(rhs);
        free(rhs_computed);
    }
    MPI_Finalize();
}
