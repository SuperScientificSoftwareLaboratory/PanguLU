#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "test_scalar.h"

static sparse_value_t exact(unsigned i, int rhs_id)
{
    sparse_value_t x = 0.5 + 0.01 * (i % 19) + 0.125 * rhs_id;
#if COMPLEX_VALUES
    x += I * (0.1 + 0.02 * (i % 7));
#endif
    return x;
}

static int solve_case(unsigned n, double scale, int phase_id, int block_size, int rank)
{
    const sparse_pointer_t nnz = 3 * n - 2;
    sparse_pointer_t *ptr = NULL;
    sparse_index_t *idx = NULL;
    sparse_value_t *val = NULL, *rhs = NULL, *saved_rhs = NULL;
    sparse_value_t phase = phase_id ? -1 : 1;
#if COMPLEX_VALUES
    if (phase_id == 2) phase = I;
    if (phase_id == 3) phase = -I;
    if (phase_id == 4) phase = 0.6 + 0.8 * I;
#endif
    if (rank == 0) {
        ptr = calloc(n + 1, sizeof(*ptr));
        idx = calloc(nnz, sizeof(*idx));
        val = calloc(nnz, sizeof(*val));
        rhs = calloc(n, sizeof(*rhs));
        saved_rhs = calloc(n, sizeof(*saved_rhs));
        sparse_pointer_t k = 0;
        for (unsigned j = 0; j < n; ++j) {
            ptr[j] = k;
            if (j) {idx[k] = j - 1; val[k++] = phase * (0.25 * scale);}
            idx[k] = j; val[k++] = phase * (4 * scale);
            if (j + 1 < n) {idx[k] = j + 1; val[k++] = phase * (-0.5 * scale);}
        }
        ptr[n] = k;
    }
    pangulu_init_options options = {0};
    options.nb = block_size;
    options.nthread = 1;
    options.sizeof_value = sizeof(sparse_value_t);
    options.is_complex_matrix = COMPLEX_VALUES;
    options.mpi_recv_buffer_level = 0.5;
    void *handle = NULL;
    pangulu_gstrf_options factor_options = {};
    pangulu_gstrs_options solve_options = {};
    int failed = 0;
#ifdef PANGULU_TEST_LEGACY
    pangulu_init(n, nnz, ptr, idx, val, &options, &handle);
    pangulu_gstrf(&factor_options, &handle);
#else
    if (pangulu_init_checked(n, nnz, ptr, idx, val, &options, &handle) != PANGULU_SUCCESS) failed = 1;
    if (!failed && pangulu_gstrf_checked(&factor_options, &handle) != PANGULU_SUCCESS) failed = 1;
#endif
    for (int rhs_id = 0; rhs_id < 3 && !failed; ++rhs_id) {
        if (rank == 0) {
            memset(rhs, 0, n * sizeof(*rhs));
            for (unsigned j = 0; j < n; ++j)
                for (sparse_pointer_t k = ptr[j]; k < ptr[j + 1]; ++k)
                    rhs[idx[k]] += val[k] * exact(j, rhs_id);
            memcpy(saved_rhs, rhs, n * sizeof(*rhs));
        }
#ifdef PANGULU_TEST_LEGACY
        pangulu_gstrs(rhs, &solve_options, &handle);
#else
        if (pangulu_gstrs_checked(rhs, &solve_options, &handle) != PANGULU_SUCCESS) failed = 1;
#endif
        if (rank == 0 && !failed) {
            long double err = 0, residual = 0, anorm = 0, xnorm = 0, bnorm = 0;
            long double _Complex *ax = calloc(n, sizeof(*ax));
            long double *row_sum = calloc(n, sizeof(*row_sum));
            for (unsigned j = 0; j < n; ++j) {
                if (!isfinite(cabs(rhs[j]))) failed = 1;
                err = fmaxl(err, cabsl((long double _Complex)rhs[j] - exact(j, rhs_id)));
                xnorm = fmaxl(xnorm, cabs(rhs[j]));
                bnorm = fmaxl(bnorm, cabs(saved_rhs[j]));
                for (sparse_pointer_t k = ptr[j]; k < ptr[j + 1]; ++k) {
                    ax[idx[k]] += (long double _Complex)val[k] * rhs[j];
                    row_sum[idx[k]] += cabs(val[k]);
                }
            }
            for (unsigned i = 0; i < n; ++i) {
                residual = fmaxl(residual, cabsl(ax[i] - saved_rhs[i]));
                anorm = fmaxl(anorm, row_sum[i]);
            }
            long double backward = residual / (anorm * xnorm + bnorm);
            if (!(err < FORWARD_TOL && backward < BACKWARD_TOL)) failed = 1;
            if (failed) fprintf(stderr, "FAIL n=%u scale=%g phase=%d rhs=%d error=%Lg backward=%Lg\n", n, scale, phase_id, rhs_id, err, backward);
            free(ax); free(row_sum);
        }
        MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (failed && rank == 0)
        fprintf(stderr, "FAIL case n=%u nb=%d scale=%g phase=%d\n", n, block_size, scale, phase_id);
    pangulu_finalize(&handle);
#ifndef PANGULU_TEST_LEGACY
    if (handle != NULL) failed = 1;
    pangulu_finalize(&handle); /* Finalize must also be safe after failure/cleanup. */
#endif
    free(ptr); free(idx); free(val); free(rhs); free(saved_rhs);
    return failed;
}

int main(int argc, char **argv)
{
    int provided, rank;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (provided < MPI_THREAD_MULTIPLE) MPI_Abort(MPI_COMM_WORLD, 2);
    if (argc == 6 && !strcmp(argv[1], "--case")) {
        unsigned n = strtoul(argv[2], NULL, 10);
        int block = atoi(argv[3]), phase = atoi(argv[5]);
        double scale = strtod(argv[4], NULL);
        if (n < 2 || n > 4096 || block < 1 || block > 1024 ||
            !isfinite(scale) || scale <= 0 || phase < 0 || phase >= (COMPLEX_VALUES ? 5 : 2))
            MPI_Abort(MPI_COMM_WORLD, 2);
        int failed = solve_case(n, scale, phase, block, rank);
        if (!rank) printf("numerical contract: %s\n", failed ? "FAIL" : "PASS");
        MPI_Finalize();
        return failed ? 1 : 0;
    }
    int failed = 0, cases = 0;
    const double scales[] = {1, SMALL_SCALE, LARGE_SCALE};
    const int quick = argc > 1 && !strcmp(argv[1], "--quick");
    for (int block_id = 0; block_id < (quick ? 1 : 2); ++block_id)
    for (int size_id = 0; size_id < (quick ? 1 : 2); ++size_id)
        for (int scale_id = 0; scale_id < (quick ? 1 : 3); ++scale_id)
            for (int phase = 0; phase < (quick ? 1 : (COMPLEX_VALUES ? 5 : 2)); ++phase) {
                if (!rank && getenv("PANGULU_TEST_VERBOSE")) {
                    fprintf(stderr, "case n=%d scale=%g phase=%d\n", size_id ? 65 : 2, scales[scale_id], phase);
                    fflush(stderr);
                }
                failed += solve_case(size_id ? 65 : 2, scales[scale_id], phase, block_id ? 64 : 8, rank);
                ++cases;
            }
    if (rank == 0) printf("numerical contracts: %d cases, %d failures\n", cases, failed);
    MPI_Finalize();
    return failed ? 1 : 0;
}
