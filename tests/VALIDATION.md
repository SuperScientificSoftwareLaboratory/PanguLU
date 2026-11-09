# Validation record

Validated against upstream `5540d74ec744458f9c54e8a4c53017be7e589f63` on
2026-11-09, with independently compiled CPU and CUDA variants on two hosts:

| GPU | CUDA toolkit | Compiler | MPI |
| --- | --- | --- | --- |
| GeForce RTX 4090 | 12.6.85 | GCC 9.4.0 | Open MPI 4.1.7a1 |
| GeForce RTX 4090 D | 12.9.86 | GCC 9.4.0 | Open MPI 4.1.7a1 |

Both used OpenBLAS 0.3.34 LP64, `-O3`, CUDA `sm_89`, no fast math, and no
MC64/METIS. Each host ran every combination below with one and two MPI ranks;
the two-rank CUDA runs used a separate GPU for each rank. These are independent
host repeats, not a cross-host distributed run.

| Precision | Numerical cases per invocation | CPU, 1/2 ranks | CUDA, 1/2 ranks | Error/lifetime suite |
| --- | ---: | --- | --- | --- |
| R64 | 24 | PASS | PASS | PASS |
| CR64 | 60 | PASS | PASS | PASS |
| R32 | 24 | PASS | PASS | PASS |
| CR32 | 60 | PASS | PASS | PASS |

Across both hosts this covers 1,344 generated numerical cases and 4,032 solves
(three RHSs per factorization), excluding the additional memory-check repeats.
The tests and exact forward/backward error thresholds are described in
[README.md](README.md). On each host all 32 ordinary suite invocations passed.

Compute Sanitizer memcheck ran the full CR64 CUDA numerical and error/lifetime
suites on both hosts with `--leak-check full --error-exitcode 99`. Each of the
four runs reported **0 errors and 0 bytes leaked in 0 allocations**. MPI used
host transports (`OMPI_MCA_pml=ob1`, `OMPI_MCA_osc=pt2pt`,
`OMPI_MCA_coll=^hcoll,ucc`, `OMPI_MCA_btl=self,tcp`) to avoid UCX device-context
probing during MPI initialization. No sanitizer errors were suppressed.

A local CPU Valgrind run of all 60 CR64 numerical cases found no invalid
accesses or definite leaks with solver allocation stacks. MPI initialization
retained allocations reported separately by Valgrind; this is not a claim of
zero whole-process host leaks or validation of every MPI implementation.

Isolated original-version R64/CR64 runs passed the ordinary scale-1 case, but
failed the 1e-80 known-solution test on CPU and CUDA; CR64 also failed the pure
imaginary-pivot case on both backends. These reproductions used only generated
matrices of order 64 and block size 64. An intermediate GPU fix also exposed
large-scale CR32 failures, addressed by guarded complex division and by
normalizing dense diagonal blocks before vendor factorization, then restoring
only U's scale.

This record covers these generated contracts, not arbitrary matrix families,
optional preprocessing configurations, performance improvements, or dynamic
pivoting. The extra CUDA scaling work has not been performance-benchmarked.
