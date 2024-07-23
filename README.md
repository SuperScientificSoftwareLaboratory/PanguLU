# PanguLU

-------------------

## Introduction

PanguLU is an open source software package for solving a linear system *Ax = b* on heterogeneous distributed platforms. The library is written in pure C, and exploits parallelism from MPI, OpenMP and CUDA. The sparse LU factorisation algorithm used in PanguLU splits the sparse matrix into multiple equally-sized sparse matrix blocks and computes them by using sparse BLAS. The latest version of PanguLU uses a synchronisation-free communication strategy to reduce the overall latency overhead, and a variety of block-wise sparse BLAS methods have been adaptively called to improve efficiency on CPUs and GPUs. Currently, PanguLU supports both single and double precision. In addition, our team at the SSSLab is constantly optimising and updating PanguLU.

## Structure of code

```
PanguLU/README      instructions on installation
PanguLU/src         C and CUDA source code, to be compiled into libpangulu.a and libpangulu.so
PanguLU/examples    example code
PanguLU/include     contains headers archieve libpangulu.a and libpangulu.so
PanguLU/lib         contains library archieve libpangulu.a and libpangulu.so
PanguLU/Makefile    top-level Makefile that does installation and testing
PanguLU/make.inc    compiler, compiler flags included in all Makefiles
```

## Installation
we use the method is to use make automatic build system.
installation method:
You will need install make.
Frist, in order to use MPI, you need to install mpich (recommended version: OpenMPI-4.1.2).
Second, if GPUs are used, NVCC is required. in order to use NVCC, you need to install CUDA (recommended version: CUDA-12.2).
Third, Specify the installation path to be used in make.inc.
Fianlly, use make for automatic installation.
> **make**

## Compilation options
Three compilation options are provided.


1 If you want to disable GPU:

1.1 Remove **-DGPU_OPEN** of variable PANGULU_FLAGS in **make.inc**;

1.2 Remove **GPU_CUDA** in file **build_list.csv**.

2 If you want to solve complex matrices:

2.1 Append **-DCALCULATE_TYPE_R64** after variable PANGULU_FLAGS in **make.inc**;

2.2 Use driver routine **driver_cr64.cpp** in directory **examples**.

Note : Solving complex matrices on GPU is not supported in this version.

## Preprocess methods
Now offering two types of preprocessing options.

### MC64
You need to open the mc64 preprocessing option in pangulu_common.h: 
> **#define   PANGULU_MC64**

### Metis
You will need the following two actions:    
1. You need to open the Metis compilation option in make.inc: 
> **METISFLAGS  =  -lmetis**

2. You need to open the Metis preprocessing option in pangulu_common.h: 
> **#define   METIS**

note: METIS needs to be 64-bit.

## Execution of PanguLU
PanguLU is to complete the operation of solving *Ax = b*, and the test files are placed in the **examples** folder. The driver_r64.cpp file is to first perform the LU numeric decomposition of the matrix test.mtx, and use *Ly = b* to complete the lower triangular solution and *Ux = y* to complete the upper triangular solution test method.
### run command

> **mpirun -np process_count ./pangulu_driver.elf -NB NB_number -F Smatrix_name**
 
process_count : MPI process number to launch PanguLU;

NB_number : Block size of each non-zero block;

Smatrix_name : the Matrix name in mtx format.(This matrix needs to be decomposed symbolically)

You can also use the run.sh, for example:

> **bash run Smatrix_name NB_number process_number**

### test sample

> **mpirun -np 6 ./pangulu_driver.elf -NB 2 -F test.mtx**

or use the run.sh:
> **bash run.sh test.mtx 2 6**


In this example,six processes are used to test, the  NB_number is 2, matrix name is test.mtx.


## Release versions

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


