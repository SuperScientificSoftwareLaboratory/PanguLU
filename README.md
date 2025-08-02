# PanguLU

-------------------

## Introduction

PanguLU is an open source software package for solving a linear system *Ax = b* on heterogeneous distributed platforms. The library is written in C, and exploits parallelism from MPI, OpenMP and CUDA. The sparse LU factorisation algorithm used in PanguLU splits the sparse matrix into multiple equally-sized sparse matrix blocks and computes them by using sparse BLAS. The latest version of PanguLU uses a synchronisation-free communication strategy to reduce the overall latency overhead, and a variety of block-wise sparse BLAS methods have been adaptively called to improve efficiency on CPUs and GPUs. Currently, PanguLU supports both single and double precision, both real and complex values. In addition, our team at the SSSLab is constantly optimising and updating PanguLU.

## Structure of code

```
PanguLU/README         instructions on installation
PanguLU/src            C and CUDA source code, to be compiled into libpangulu.a and libpangulu.so
PanguLU/examples       example code
PanguLU/include        contains headers archieve pangulu.h
PanguLU/lib            contains library archieve libpangulu.a and libpangulu.so
PanguLU/reordering_omp Parallel graph partitioning and fill-reducing matrix ordering
PanguLU/Makefile       top-level Makefile that does installation and testing
PanguLU/make.inc       compiler, compiler flags included in all Makefiles (excepts examples/Makefile)
```

## Installation
#### Step 1 : Assert "make" is available.
"make" is an automatic build tool, it is required to build PanguLU. "make" is available in most GNU/Linux. You can install it using package managers like `apt` or `yum`.

#### Step 2 : Assert MPI library is available.
PanguLU requires MPI library. you need to install MPI library with header files. Tested MPI libraries : MPICH 4.3.0.

#### Step 3 : Assert CUDA is available. (optimal, required if GPU is used)
If GPUs are used, CUDA is required. Tested version : CUDA 12.9.

#### Step 4 : Assert BLAS library is available. (optimal, required if GPU is not used)
A BLAS library is required if CPU takes part in algebra computing of numeric factorisation. Tested version : OpenBLAS 0.3.26.

#### Step 5 : Edit `make.inc`.
Search `/path/to` in `make.inc`. Replace them to the path actually on your computer.

#### Step 6 : Edit `examples/Makefile`
The Makefile of example code doesn't include `make.inc`. Search `/path/to` in `examples/Makefile`. Replace them to the path actually on your computer.

#### Step 7 : Decide if you want to use GPU.
GPU is enabled by default. If you want to disable GPU, you should : 
 - Remove `GPU_CUDA` in `build_list.csv`;
 - Remove `-DGPU_OPEN` in `PANGULU_FLAGS`. You can find `PANGULU_FLAGS` in `make.inc`;
 - Comment `LINK_CUDA` in `examples/Makefile`.
 
Vise versa.

#### Step 8 : Run `make -j` in your terminal.
Make sure the working directory of your terminal is the root directory of PanguLU. If PanguLU was successfully built, you will find `libpangulu.a` and `libpangulu.so` in `lib` directory, and `pangulu_example.elf` in `exampls` directory.

## Build flags
`PANGULU_FLAGS` influences build behaviors. You can edit `PANGULU_FLAGS` in `make.inc` to implement different features of PanguLU. Here are available flags :

#### Decide if or not using GPU.
Use `-DGPU_OPEN` to use GPU, vice versa. Please notice that using this flag is not the only thing to do if you want to use GPU. Please check Step 7 in the Installation part.

#### Decide the value type of matrix and vector entries.
Use `-DCALCULATE_TYPE_R64` (double real) or `-DCALCULATE_TYPE_CR64` (double complex) or `-DCALCULATE_TYPE_R32` (float real) or `-DCALCULATE_TYPE_CR32` (float complex). Note to also add this option to the compilation command in `example/Makefile`.

#### Decide if or not using MC64 reordering algorithm.
Use `-DPANGULU_MC64` to enable MC64 algorithm. Please notice that MC64 is not supported when matrix entries are complex numbers. If complex values are selected and `-DPANGULU_MC64` flag is used, MC64 would not enable.

#### Decide using our parallel reordering algorithm or METIS reordering tool.
Use `-DMETIS` if you want to use METIS; otherwise, our parallel reordering algorithm will be used by default.

#### Decide log level.
Please select zero or one of these flags : `-DPANGULU_LOG_INFO`, `-DPANGULU_LOG_WARNING` or `-DPANGULU_LOG_ERROR`. Log level "INFO" prints all messages to standard output (including warnings and errors). Log level "WANRING" only prints warnings and errors. Log level "ERROR" only prints fatal errors causing PanguLU to terminate abnormally.

#### Decide whether additional performance information is needed:
Use `-DPANGULU_PERF` to output additional performance information, such as kernel time per gpu and gflops of numeric factorisation. Note that this will slow down the speed of numeric factorisation.

#### Decide core binding strategy.
Hyper-threading is not recommended. If you can't turn off the hyper-threading and each core of your CPU has 2 threads, using `-DHT_IS_OPEN` may reaps performance gain.

## Function interfaces
To make it easier to call PanguLU in your software, PanguLU provides the following function interfaces:

#### 1. pangulu_init()
```
void pangulu_init(
  sparse_index_t pangulu_n, // Specifies the number of rows in the CSC matrix.
  sparse_pointer_t pangulu_nnz, // Specifies the total number of non-zero elements in the CSC matrix.
  sparse_pointer_t *csc_colptr, // Points to an array that stores pointers to columns of the CSC matrix.
  sparse_index_t *csc_rowidx, // Points to an array that stores indices to rows of the CSC matrix.
  sparse_value_t *csc_value, // Points to an array that stores the values of non-zero elements of the CSC matrix.
  pangulu_init_options *init_options, // Pointer to a pangulu_init_options structure containing initialisation parameters for the solver.
  void **pangulu_handle // On return, contains a handle pointer to the library's internal state.
);
```

#### 2. pangulu_gstrf()
```
void pangulu_gstrf(
  pangulu_gstrf_options *gstrf_options, // Pointer to pangulu_gstrf_options structure.
  void **pangulu_handle // Pointer to the solver handle returned on initialisation.
);
```

#### 3. pangulu_gstrs()
```
void pangulu_gstrs(
  sparse_value_t *rhs, // Pointer to the right-hand side vector.
  pangulu_gstrs_options *gstrs_options, // Pointer to the pangulu_gstrs_options structure.
  void** pangulu_handle // Pointer to the library internal state handle returned on initialisation.
);
```

#### 4. pangulu_gssv()
```
void pangulu_gssv(
  sparse_value_t *rhs, // Pointer to the right-hand side vector.
  pangulu_gstrf_options *gstrf_options, // Pointer to a pangulu_gstrf_options structure.
  pangulu_gstrs_options *gstrs_options, // Pointer to a pangulu_gstrs_options structure.
  void **pangulu_handle // Pointer to the library internal status handle returned on initialisation.
);
```

#### 5. pangulu_finalize()
```
void pangulu_finalize(
  void **pangulu_handle // Pointer to the library internal state handle returned on initialisation.
);
```

`example.c` is a sample program to call PanguLU. You can refer to this file to complete the call to PanguLU. You should first create the distributed matrix using `pangulu_init()`. If you need to solve multiple right-hand side vectors while the matrix is unchanged, you can call `pangulu_gstrs()` multiple times after calling `pangulu_gstrf()`. If you need to factorise a number of different matrices, call `pangulu_finalize()` after completing the solution of one matrix, and then use `pangulu_init()` to to initialise the next matrix.

## Executing the example code of PanguLU
The test routines are placed in the `examples` directory. The routine in `examples/example.c` firstly call `pangulu_gstrf()` to perform LU factorisation, and then call `pangulu_gstrs()` to solve linear equation.
#### run command

> **mpirun -np process_count ./pangulu_example.elf -nb block_size -f path_to_mtx -r path_to_rhs**
 
process_count : MPI process number to launch PanguLU;

block_size : Rank of each non-zero block;

path_to_mtx : The matrix name in mtx format;

path_to_rhs : The path of the right-hand side vector. (optional)

You can also use the run.sh, for example:

> **bash run path_to_mtx block_size process_count**

#### test sample

> **mpirun -np 4 ./pangulu_example.elf -nb 10 -f Trefethen_20b.mtx**

or use the run.sh:
> **bash run.sh Trefethen_20b.mtx 10 4**


In this example, 4 processes are used to test, the block_size is 10, matrix name is Trefethen_20b.mtx.


## Release versions

#### <p align='left'>Version 5.0.0 (Jul. 31, 2025) </p>

* Added a task aggregator to increase numeric factorisation performance;
* Optimised performance of preprocessing phase;
* Added parallel reordering algorithm on CPU as the default reordering algorithm;
* Optimised GPU memory layout to reduce GPU memory usage.

#### <p align='left'>Version 4.1.0 (Sep. 1, 2024) </p>

* Optimised memory usage of numeric factorisation and solving;
* Added parallel building support.

#### <p align='left'>Version 4.0.0 (Jul. 24, 2024) </p>

* Optimised user interfaces of solver routines;
* Optimised performamce of numeric factorisation phase on CPU platform;
* Added support on complex matrix solving;
* Optimised preprocessing performance;

#### <p align='left'>Version 3.5.0 (Aug. 06, 2023) </p>

* Updated the pre-processing phase with OpenMP.
* Updated the compilation method of PanguLU, compile libpangulu.so and libpangulu.a at the same time.
* Updated timing for the reorder phase, the symbolic factorisation phase, the pre-processing phase.
* Added GFLOPS for the numeric factorisation phase.
 
#### <p align='left'>Version 3.0.0 (Apr. 02, 2023) </p>

* Used adaptive selection sparse BLAS in the numeric factorisation phase.
* Added the reorder phase.
* Added the symbolic factorisation phase. 
* Added mc64 sorting algorithm in the reorder phase.
* Added interface for 64-bit metis package in the reorder phase.

#### <p align='left'> Version 2.0.0 (Jul. &thinsp;22, 2022) </p>

* Used a synchronisation-free scheduling strategy in the numeric factorisation phase.
* Updated the MPI communication method in the numeric factorisation phase.
* Added single precision in the numeric factorisation phase.

#### <p align='left'>Version 1.0.0 (Oct. 19, 2021) </p>

* Used a rule-based 2D LU factorisation scheduling strategy.
* Used Sparse BLAS for floating point calculations on GPUs.
* Added the pre-processing phase.
* Added the numeric factorisation phase.
* Added the triangular solve phase.

## Reference

* [1] Xu Fu, Bingbin Zhang, Tengcheng Wang, Wenhao Li, Yuechen Lu, Enxin Yi, Jianqi Zhao, Xiaohan Geng, Fangying Li, Jingwen Zhang, Zhou Jin, Weifeng Liu. PanguLU: A Scalable Regular Two-Dimensional Block-Cyclic Sparse Direct Solver on Distributed Heterogeneous Systems. 36th ACM/IEEE International Conference for High Performance Computing, Networking, Storage, and Analysis (SC '23). 2023.


