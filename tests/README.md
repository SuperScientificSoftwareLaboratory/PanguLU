# Numerical and lifetime regressions

Build PanguLU first with `CPU_NAIVE` and the desired `CALCULATE_TYPE_` in
`make.inc`, without METIS or MC64. Rebuild the library cleanly when changing
precision or backend. Then, from the repository root:

```sh
make -C tests PRECISION=CR64 BLAS_LIBS=-lopenblas
mpiexec -n 1 .build/tests/numerical-CR64
mpiexec -n 2 .build/tests/numerical-CR64
mpiexec -n 2 .build/tests/errors-CR64
```

Repeat for `R64`, `CR32`, and `R32`. `PRECISION` must match the library.
For a `CPU_NAIVE` + `GPU_CUDA` build, also pass the CUDA link libraries, e.g.
`CUDA_LIBS='-L/path/to/cuda/lib64 -lcudart -lcusolver -lcublas -lstdc++'`.
Use `--case N BLOCK SCALE PHASE` to isolate one generated numerical case.
Phase IDs are 0 (+1), 1 (-1), 2 (+i), 3 (-i), and 4 (0.6+0.8i); the last
three apply only to complex builds.

MPI must provide `MPI_THREAD_MULTIPLE`. The numerical tests use one OpenMP thread per
rank; the error tests also switch between handles with different thread counts.
All tests use generated matrices only; no external datasets are required.

The numerical test uses nonsymmetric, diagonally dominant tridiagonal matrices
of orders 2 and 65, block sizes 8 and 64, and three right-hand sides per
factorization. It checks known-solution error and the infinity-norm backward
error `||Ax-b|| / (||A|| ||x|| + ||b||)` using long-double accumulation.
FP64 scales are 1, 1e-80, and 1e80; FP32 scales are 1, 1e-20, and 1e20.
Complex tests include purely imaginary pivots and rotations of the matrix
phase. There are 60 complex or 24 real cases per invocation.

The error test checks collective validation, duplicate and unsorted CSC rows,
nonfinite inputs, invalid solver state, zero pivots, repeated right-hand sides,
overlapping handle lifetimes, and repeated finalization. Run memory checking on
the full numerical test, since a single diagonal tile does not exercise the
temporary task buffer used by repeated multiblock factorizations:

```sh
valgrind --leak-check=full --show-leak-kinds=definite \
  .build/tests/numerical-CR64
```

MPI implementations can retain allocations at process exit. Inspect allocation
stacks separately from solver allocations; a successful numerical exit alone
is not evidence of clean memory usage. GPU execution and device memory checking
must be performed on a CUDA-capable host.

## Checked API

The existing void entry points remain available and report failures to stderr.
For recoverable argument/state and numerical errors, use the additive
`pangulu_init_checked`, `pangulu_gstrf_checked`, `pangulu_gstrs_checked`, and
`pangulu_gssv_checked` functions. All are collective on `MPI_COMM_WORLD` and
return the same status on every participating rank. Initialize the handle slot
to NULL before checked initialization. CSC arrays remain caller-owned.

A handle is initialized once and factored once, then can solve multiple RHSs.
A failed factorization blocks solving with its partial factors. Failed checked
solves preserve the root's RHS. `pangulu_get_status` reports the last numerical
operation's status; argument validation failures are returned directly.
`pangulu_finalize` clears the caller's handle slot and accepts NULL/repeated
calls. Operations on different handles must still be serialized: the backend
scratch state is not reentrant.

Pivot checks use the complex magnitude and a precision-dependent relative row
scale. They never replace a pivot with a fixed real constant. A zero or unsafe
pivot reports breakdown; this does not establish singularity or add dynamic
pivoting. Allocation, MPI transport, CUDA runtime, and existing optional
preprocessing fatal errors are outside the recoverable API's scope. These tests
do not validate optional MC64/METIS configurations or arbitrary ill-conditioned
matrices, and are not performance benchmarks.
