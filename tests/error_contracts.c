#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "test_scalar.h"

static int failures;
static void expect(pangulu_status_t actual, pangulu_status_t expected, const char *name, int rank)
{
    int low = actual, high = actual;
    MPI_Allreduce(MPI_IN_PLACE, &low, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &high, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (low != high || actual != expected) {
        ++failures;
        if (!rank) fprintf(stderr, "FAIL %s: status range [%d,%d], expected %d\n", name, low, high, expected);
    }
}

int main(int argc, char **argv)
{
    int rank, provided, size;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (provided < MPI_THREAD_MULTIPLE) MPI_Abort(MPI_COMM_WORLD, 2);
    sparse_pointer_t ptr[] = {0, 2, 4};
    sparse_index_t idx[] = {0, 1, 0, 1};
    sparse_value_t values[] = {4, 1, 1, 3};
    sparse_value_t rhs[] = {5, 4};
    pangulu_init_options options = {0};
    options.nb = 8; options.nthread = 1;
    options.sizeof_value = sizeof(sparse_value_t);
    options.is_complex_matrix = COMPLEX_VALUES;
    options.mpi_recv_buffer_level = 0.5;
    pangulu_gstrf_options factor_options = {};
    pangulu_gstrs_options solve_options = {};
    void *handle = NULL;
#define INIT(opt) pangulu_init_checked(2, 4, rank ? NULL : ptr, rank ? NULL : idx, rank ? NULL : values, (opt), &handle)
    expect(INIT(rank == size - 1 ? NULL : &options), PANGULU_INVALID_ARGUMENT, "collective null options", rank);
    options.nb = 0;
    expect(INIT(&options), PANGULU_INVALID_ARGUMENT, "zero block size", rank);
    options.nb = 8;
    idx[1] = 0;
    expect(INIT(&options), PANGULU_INVALID_ARGUMENT, "duplicate row", rank);
    idx[1] = 1;
    values[0] = NAN;
    expect(INIT(&options), PANGULU_INVALID_ARGUMENT, "nonfinite matrix", rank);
    values[0] = 4;
    if (size > 1) {
        options.nb = rank ? 16 : 8;
        expect(INIT(&options), PANGULU_INVALID_ARGUMENT, "inconsistent rank options", rank);
        options.nb = 8;
    }
    /* Row order is not part of the CSC contract; duplicates still are invalid. */
    idx[0] = 1; idx[1] = 0; values[0] = 1; values[1] = 4;
    expect(INIT(&options), PANGULU_SUCCESS, "unsorted rows", rank);
    pangulu_finalize(&handle);
    idx[0] = 0; idx[1] = 1; values[0] = 4; values[1] = 1;
    expect(INIT(&options), PANGULU_SUCCESS, "valid init", rank);
    void *first = handle;
    handle = NULL;
    options.nb = 16; options.nthread = 2;
    expect(INIT(&options), PANGULU_SUCCESS, "second live handle", rank);
    expect(pangulu_gstrf_checked(&factor_options, &handle), PANGULU_SUCCESS, "factor second handle", rank);
    pangulu_finalize(&handle);
    handle = first;
    options.nb = 8; options.nthread = 1;
    expect(pangulu_gstrs_checked(rhs, &solve_options, &handle), PANGULU_INVALID_STATE, "solve before factor", rank);
    expect(pangulu_gstrf_checked(rank == size - 1 ? NULL : &factor_options, &handle), PANGULU_INVALID_ARGUMENT, "null factor options", rank);
    expect(pangulu_gstrf_checked(&factor_options, &handle), PANGULU_SUCCESS, "factor", rank);
    expect(pangulu_gstrs_checked(NULL, &solve_options, &handle), PANGULU_INVALID_ARGUMENT, "null root rhs", rank);
    rhs[0] = NAN;
    expect(pangulu_gstrs_checked(rhs, &solve_options, &handle), PANGULU_INVALID_ARGUMENT, "nonfinite rhs", rank);
    rhs[0] = 5;
    expect(pangulu_gstrs_checked(rhs, &solve_options, &handle), PANGULU_SUCCESS, "recover after bad rhs", rank);
    expect(pangulu_get_status(handle), PANGULU_SUCCESS, "query status", rank);
    pangulu_finalize(&handle);
    pangulu_finalize(&handle);
    if (handle) ++failures;
    for (int k = 0; k < 4; ++k) values[k] = 1;
    expect(INIT(&options), PANGULU_SUCCESS, "rank-deficient init", rank);
    expect(pangulu_gstrf_checked(&factor_options, &handle), PANGULU_ZERO_PIVOT, "zero pivot", rank);
    rhs[0] = 2; rhs[1] = 2;
    expect(pangulu_gstrs_checked(rhs, &solve_options, &handle), PANGULU_ZERO_PIVOT, "failed factor blocks solve", rank);
    if (!rank && (rhs[0] != 2 || rhs[1] != 2)) ++failures;
    pangulu_finalize(&handle);
    expect(pangulu_get_status(handle), PANGULU_INVALID_STATE, "null handle status", rank);
    int global;
    MPI_Allreduce(&failures, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (!rank) printf("error contracts: %d failures\n", global);
    MPI_Finalize();
    return global ? 1 : 0;
}
