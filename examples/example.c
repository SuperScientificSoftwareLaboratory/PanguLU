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
typedef double sparse_value_t;
#error [example.c Compile Error] Unknown value type. Set -DCALCULATE_TYPE_CR64 or -DCALCULATE_TYPE_R64 or -DCALCULATE_TYPE_CR32 or -DCALCULATE_TYPE_R32 in compile command line.
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
    sparse_pointer_t *colptr = NULL;
    sparse_index_t *rowidx = NULL;
    sparse_value_t *value = NULL;
    sparse_value_t *sol = NULL;
    sparse_value_t *rhs = NULL;

    // Step 2: Read matrix and rhs vectors.
    if (rank == 0)
    {
        char mtx_name[200] = {'\0'};
        char rhs_name[200] = {'\0'};
        read_command_params(ARGC, ARGV, mtx_name, rhs_name, &nb);

        switch (mtx_name[strlen(mtx_name) - 1])
        {
        case 'x':
            // mtx read (csc)
            printf("Reading mtx matrix %s\n", mtx_name);
            mmio_info(&m, &n, &nnz, &is_sym, mtx_name);
            colptr = (sparse_pointer_t *)malloc(sizeof(sparse_pointer_t) * (n + 1));
            rowidx = (sparse_index_t *)malloc(sizeof(sparse_index_t) * nnz);
            value = (sparse_value_t *)malloc(sizeof(sparse_value_t) * nnz);
            mmio_data_csc(colptr, rowidx, value, mtx_name);
            printf("Read mtx done.\n");
            break;
        case 'd':

            // lid read
            printf("Reading lid matrix %s\n", mtx_name);
            FILE *lid_file = fopen(mtx_name, "r");
            fread(&m, sizeof(sparse_index_t), 1, lid_file);
            fread(&n, sizeof(sparse_index_t), 1, lid_file);
            fread(&nnz, sizeof(sparse_pointer_t), 1, lid_file);
            sparse_pointer_t *rowptr = (sparse_pointer_t *)malloc(sizeof(sparse_pointer_t) * (n + 1));
            sparse_index_t *colidx = (sparse_index_t *)malloc(sizeof(sparse_index_t) * nnz);
            sparse_value_t *value_csr = (sparse_value_t *)malloc(sizeof(sparse_value_t) * nnz);
            fread(rowptr, sizeof(sparse_pointer_t), n + 1, lid_file);
            fread(colidx, sizeof(sparse_index_t), nnz, lid_file);
            fread(value_csr, sizeof(sparse_value_t), nnz, lid_file);
            fclose(lid_file);

            colptr = (sparse_pointer_t *)malloc(sizeof(sparse_pointer_t) * (n + 1));
            rowidx = (sparse_index_t *)malloc(sizeof(sparse_index_t) * nnz);
            value = (sparse_value_t *)malloc(sizeof(sparse_value_t) * nnz);
            memset(colptr, 0, sizeof(sparse_pointer_t) * (n + 1));
            sparse_pointer_t *trans_aid = (sparse_pointer_t *)malloc(sizeof(sparse_pointer_t) * n);
            memset(trans_aid, 0, sizeof(sparse_pointer_t) * n);
            for (sparse_index_t row = 0; row < n; row++)
            {
                for (sparse_pointer_t idx = rowptr[row]; idx < rowptr[row + 1]; idx++)
                {
                    sparse_index_t col = colidx[idx];
                    colptr[col + 1]++;
                }
            }
            for (sparse_index_t row = 0; row < n; row++)
            {
                colptr[row + 1] += colptr[row];
            }
            memcpy(trans_aid, colptr, sizeof(sparse_pointer_t) * n);
            for (sparse_index_t row = 0; row < n; row++)
            {
                for (sparse_pointer_t idx = rowptr[row]; idx < rowptr[row + 1]; idx++)
                {
                    sparse_index_t col = colidx[idx];
                    rowidx[trans_aid[col]] = row;
                    value[trans_aid[col]] = value_csr[idx];
                    trans_aid[col]++;
                }
            }
            free(rowptr);
            free(colidx);
            free(value_csr);
            free(trans_aid);
            printf("Read lid done.\n");

            break;
        }

        sol = (sparse_value_t *)malloc(sizeof(sparse_value_t) * n);
        rhs = (sparse_value_t *)malloc(sizeof(sparse_value_t) * n);
        memset(rhs, 0, sizeof(sparse_value_t) * n);
        if (rhs_name[0])
        {
            FILE *rhs_file = fopen(rhs_name, "r");
            if (!rhs_file)
            {
                fprintf(stderr, "Error: Failed to open rhs file '%s'\n",
                        rhs_name);
                exit(1);
            }

            char line[512];
            int found_data = 0;
            while (fgets(line, sizeof(line), rhs_file))
            {
                if (line[0] != '%')
                {
                    found_data = 1;
                    break;
                }
            }

            if (!found_data)
            {
                fprintf(stderr, "Error: File '%s' contains only comments or is empty\n", rhs_name);
                fclose(rhs_file);
                exit(1);
            }

            int rhs_len;
            if (sscanf(line, "%d", &rhs_len) != 1)
            {
                fprintf(stderr, "Error: Failed to read vector dimension from '%s'\n", rhs_name);
                fclose(rhs_file);
                exit(1);
            }

            if (rhs_len != n)
            {
                fprintf(stderr, "Error: Vector dimension mismatch - expected %d, got %d\n", n, rhs_len);
                fclose(rhs_file);
                exit(1);
            }

            for (int i = 0; i < n; i++)
            {
                int read_success = 0;

#ifdef COMPLEX_MTX
                double real_part, imag_part;
                if (fscanf(rhs_file, "%le %le", &real_part, &imag_part) == 2)
                {
                    __real__(rhs[i]) = real_part;
                    __imag__(rhs[i]) = imag_part;
                    read_success = 1;
                }
#else
                if (fscanf(rhs_file, "%lf", &rhs[i]) == 1)
                {
                    read_success = 1;
                }
#endif

                if (!read_success)
                {
                    fprintf(stderr, "Error: Failed to read vector element %d from '%s'\n", i, rhs_name);
                    fclose(rhs_file);
                    exit(1);
                }

                sol[i] = rhs[i];
            }

            fclose(rhs_file);
            printf("Successfully read rhs from '%s'\n", rhs_name);
        }
        else
        {
            if (!colptr || !value)
            {
                fprintf(stderr, "Error: Invalid matrix data for rhs generation\n");
                exit(1);
            }

            for(int i=0; i < n; i++){
                rhs[i] = 0;
            }
            for (int i = 0; i < n; i++)
            {
                for (sparse_pointer_t j = colptr[i]; j < colptr[i + 1]; j++)
                {
                    rhs[rowidx[j]] += value[j];
                }
            }
            for(int i=0; i < n; i++){
                sol[i] = rhs[i];
            }

            printf("Successfully generated rhs from matrix\n");
        }

        if((m != n) || m == 0){
            printf("Matrix A is %d * %d. Exit.\n", m, n);
            exit(1);
        }
    }
    MPI_Bcast(&n, 1, MPI_SPARSE_INDEX_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nb, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnz, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Step 3: Initialize PanguLU solver.
    pangulu_init_options init_options;
    init_options.nb = nb;
    init_options.gpu_kernel_warp_per_block = 4;
    init_options.gpu_data_move_warp_per_block = 4;
    init_options.nthread = 1;
    init_options.hunyuan_nthread = 4;
    init_options.sizeof_value = sizeof(sparse_value_t);
    #ifdef COMPLEX_MTX
    init_options.is_complex_matrix = 1;
    #else
    init_options.is_complex_matrix = 0;
    #endif
    init_options.mpi_recv_buffer_level = 0.5;
    void *pangulu_handle;
    pangulu_init(n, nnz, colptr, rowidx, value, &init_options, &pangulu_handle);

    // Step 4: Execute LU factorisation.
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
        memset(rhs_computed, 0, sizeof(sparse_value_t) * n);
        for (int i = 0; i < n; i++)
        {
            for (sparse_pointer_t j = colptr[i]; j < colptr[i + 1]; j++)
            {
                rhs_computed[rowidx[j]] += value[j] * sol[i];
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
        free(colptr);
        free(rowidx);
        free(value);
        free(sol);
        free(rhs);
        free(rhs_computed);
    }
    MPI_Finalize();
}
