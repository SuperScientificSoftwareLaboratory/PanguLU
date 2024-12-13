# PanguLU

-------------------

## Introduction

PanguLU is an open source software package for solving a linear system *Ax = b* on heterogeneous distributed platforms. The library is written in C, and exploits parallelism from MPI, OpenMP and CUDA. The sparse LU factorisation algorithm used in PanguLU splits the sparse matrix into multiple equally-sized sparse matrix blocks and computes them by using sparse BLAS. The latest version of PanguLU uses a synchronisation-free communication strategy to reduce the overall latency overhead, and a variety of block-wise sparse BLAS methods have been adaptively called to improve efficiency on CPUs and GPUs. Currently, PanguLU supports both single and double precision, both real and complex values. In addition, our team at the SSSLab is constantly optimising and updating PanguLU.

## Structure of code

```
PanguLU/README      instructions on installation
PanguLU/src         C and CUDA source code, to be compiled into libpangulu.a and libpangulu.so
PanguLU/examples    example code
PanguLU/include     contains headers archieve libpangulu.a and libpangulu.so
PanguLU/lib         contains library archieve libpangulu.a and libpangulu.so
PanguLU/Makefile    top-level Makefile that does installation and testing
PanguLU/make.inc    compiler, compiler flags included in all Makefiles (excepts examples/Makefile)
```

## Installation
#### Step 1 : Assert "make" is available.
"make" is an automatic build tool, it is required to build PanguLU. "make" is available in most GNU/Linux. You can install it using package managers like `apt` or `yum`.

#### Step 2 : Assert MPI library is available.
PanguLU requires MPI library. you need to install MPI library with header files. Tested MPI libraries : OpenMPI 4.1.2, Intel MPI 2021.12.

#### Step 3 : Assert CUDA is available. (optimal, required if GPU is used)
If GPUs are used, CUDA is required. Tested version : CUDA 12.2.

#### Step 4 : Assert BLAS library is available. (optimal, required if GPU is not used)
A BLAS library is required if CPU takes part in algebra computing of numeric factorization. Tested version : OpenBLAS 0.3.26.

#### Step 5 : Assert METIS is available. (optimal but recommended)
The github page of METIS library is : https://github.com/KarypisLab/METIS

#### Step 6 : Edit `make.inc`.
Search `/path/to` in `make.inc`. Replace them to the path actually on your computer.

#### Step 7 : Edit `examples/Makefile`
The Makefile of example code doesn't include `make.inc`. Search `/path/to` in `examples/Makefile`. Replace them to the path actually on your computer.

#### Step 8 : Decide if you want to use GPU.
If you want to use GPU, you should : 
 - Append `GPU_CUDA` in build_list.csv;
 - Add `-DGPU_OPEN` in `PANGULU_FLAGS`. You can find `PANGULU_FLAGS` in `make.inc`;
 - Uncomment `LINK_CUDA` in `examples/Makefile`.

Vise versa.

#### Step 9 : Run `make -j` in your terminal.
Make sure the working directory of your terminal is the root directory of PanguLU. If PanguLU was successfully built, you will find `libpangulu.a` and `libpangulu.so` in `lib` directory, and `pangulu_example.elf` in `exampls` directory.

## Build flags
`PANGULU_FLAGS` influences build behaviors. You can edit `PANGULU_FLAGS` in `make.inc` to implement different features of PanguLU. Here are available flags :

#### Decide if or not using GPU.
Use `-DGPU_OPEN` to use GPU, vice versa. Please notice that using this flag is not the only thing to do if you want to use GPU. Please check Step 8 in the Installation part.

#### Decide the value type of matrix and vector entries.
Use `-DCALCULATE_TYPE_R64` (double real) or `-DCALCULATE_TYPE_CR64` (double complex) or `-DCALCULATE_TYPE_R32` (float real) or `-DCALCULATE_TYPE_CR32` (float complex).

#### Decide if or not using MC64 reordering algorithm.
Use `-DPANGULU_MC64` to enable MC64 algorithm. Please notice that MC64 is not supported when matrix entries are complex numbers. If complex values are selected and `-DPANGULU_MC64` flag is used, MC64 would not enable.

#### Decide if or not using METIS reordering tool.
Use `-DMETIS` to enable METIS.

#### Decide log level.
Please select zero or one of these flags : `-DPANGULU_LOG_INFO`, `-DPANGULU_LOG_WARNING` or `-DPANGULU_LOG_ERROR`. Log level "INFO" prints all messages to standard output (including warnings and errors). Log level "WANRING" only prints warnings and errors. Log level "ERROR" only prints fatal errors causing PanguLU to terminate abnormally.

#### Decide core binding strategy.
Hyper-threading is not recommended. If you can't turn off the hyper-threading and each core of your CPU has 2 threads, using `-DHT_IS_OPEN`
may reaps performance gain.

## Function interfaces
To make it easier to call PanguLU in your software, PanguLU provides the following function interfaces:

#### 1. pangulu_init()
```
void pangulu_init(
  int pangulu_n, // Specifies the number of rows in the CSR format matrix.
  long long pangulu_nnz, // Specifies the total number of non-zero elements in the CSR format matrix.
  long *csr_rowptr, // Points to an array that stores pointers to rows of the CSR format matrix.
  int *csr_colidx, // Points to an array that stores indices to columns of the CSR format matrix.
  pangulu_calculate_type *csr_value, // Points to an array that stores the values of the CSR format matrix.
  pangulu_init_options *init_options, // Pointer to a pangulu_init_options structure.
  void **pangulu_handle // On return, contains a handle pointer to the library’s internal state.
);
```

#### 2. pangulu_gstrf()
```
void pangulu_gstrf(
  pangulu_gstrf_options *gstrf_options, // Pointer to pangulu_gstrf_options structure.
  void **pangulu_handle // Pointer to the solver handle returned on initialization.
);
```

#### 3. pangulu_gstrs()
```
void pangulu_gstrs(
  pangulu_calculate_type *rhs, // Pointer to the right-hand side vector.
  pangulu_gstrs_options *gstrs_options, // Pointer to the pangulu_gstrs_options structure.
  void** pangulu_handle // Pointer to the library internal state handle returned on initialization.
);
```

#### 4. pangulu_gssv()
```
void pangulu_gssv(
  pangulu_calculate_type *rhs, // Pointer to the right-hand side vector.
  pangulu_gstrf_options *gstrf_options, // Pointer to a pangulu_gstrf_options structure.
  pangulu_gstrs_options *gstrs_options, // Pointer to a pangulu_gstrs_options structure.
  void **pangulu_handle // Pointer to the library internal status handle returned on initialization.
);
```

#### 5. pangulu_finalize()
```
void pangulu_finalize(
  void **pangulu_handle // Pointer to the library internal state handle returned on initialization.
);
```

`example.c` is a sample program to call PanguLU. You can refer to this file to complete the call to PanguLU. You should first create the distributed matrix using `pangulu_init()`. If you need to solve multiple right-hand side vectors while the matrix is unchanged, you can call `pangulu_gstrs()` multiple times after calling `pangulu_gstrf()`. If you need to factorize a number of different matrices, call `pangulu_finalize()` after completing the solution of one matrix, and then use `pangulu_init()` to to initialize the next matrix.

## Executing the example code of PanguLU
The test routines are placed in the `examples` directory. The routine in `examples/example.c` firstly call `pangulu_gstrf()` to perform LU factorization, and then call `pangulu_gstrs()` to solve linear equation.
#### run command

> **mpirun -np process_count ./pangulu_example.elf -nb block_size -f path_to_mtx**
 
process_count : MPI process number to launch PanguLU;

block_size : Rank of each non-zero block;

path_to_mtx : The matrix name in mtx format.

You can also use the run.sh, for example:

> **bash run path_to_mtx block_size process_count**

#### test sample

> **mpirun -np 6 ./pangulu_example.elf -nb 4 -f Trefethen_20b.mtx**

or use the run.sh:
> **bash run.sh Trefethen_20b.mtx 4 6**


In this example, 6 processes are used to test, the block_size is 4, matrix name is Trefethen_20b.mtx.


## Release versions

#### <p align='left'>Version 4.2.0 (Dec. 13, 2024) </p>

* Updated preprocessing phase to distributed data structure.

#### <p align='left'>Version 4.1.0 (Sep. 1, 2024) </p>

* Optimized memory usage of numeric factorisation and solving;
* Added parallel building support.

#### <p align='left'>Version 4.0.0 (Jul. 24, 2024) </p>

* Optimized user interfaces of solver routines;
* Optimized performamce of numeric factorisation phase on CPU platform;
* Added support on complex matrix solving;
* Optimized preprocessing performance;

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

* [1] Xu Fu, Bingbin Zhang, Tengcheng Wang, Wenhao Li, Yuechen Lu, Enxin Yi, Jianqi Zhao, Xiaohan Geng, Fangying Li, Jingwen Zhang, Zhou Jin, Weifeng Liu. PanguLU: A Scalable Regular Two-Dimensional Block-Cyclic Sparse Direct Solver on Distributed Heterogeneous Systems. 36th ACM/IEEE International Conference for High Performance Computing, Networking, Storage, and Analysis (SC ’23). 2023.


