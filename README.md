# PanguLU

 
 **PanguLU** is an open source software package that uses a block sparse structure to solve linear systems A*X=B. It uses a sparse matrix block LU factorisation algorithm. The sparse matrix block LU factorisation algorithm is an algorithm that splits the sparse matrix into multiple sparse matrix blocks, and uses sparse BLAS kernels among the sparse matrix blocks. This solver can be used in heterogeneous The distributed platform operates accurately and efficiently.

-------------------


## Introduction

PanguLU is mainly used on distributed multi-GPU parallel clusters. PanguLU is implemented using MPI and CUDA, MPI is used for communication and CUDA is used for GPU computing. We have implemented multi-GPU acceleration and are still actively accelerating.

PanguLU reads in a CSR matrix that has undergone symbolic decomposition, then performs matrix division, distributes the matrix to each node for numerical decomposition, and then solves the vector by distributed triangular solution. 

PanguLU uses a synchronisation-free distributed communication strategy and uses a global variable to improve the degree of parallelism as much as possible under the premise of ensuring the correctness of the computation.


## Structure of code

```
PanguLU/README      instructions on installation
PanguLU/src         C and CUDA source code, to be compiled into libpangulu.so
PanguLU/test        testing code
PanguLU/icnlude     contains headers archieve libpangulu.so
PanguLU/lib         contains library archieve libpangulu.so
PanguLU/Makefile    top-level Makefile that does installation and testing
PanguLU/make.inc    compiler, compiler flags included in all Makefiles
```

## Installation
we use the method is to use make automatic build system.
installation method:
You will need install make.
Frist, in order to use MPI, you need to install mpich (recommended version: OpenMPI-4.1.2).
Second, in order to use NVCC, you need to install CUDA (recommended version: CUDA-12.1).
Third, Specify the installation path to be used in make.inc.
Fianlly, use make for automatic installation.
> **make**

## Compilation options
One type of compilation are provided.


1.You need to open the GPU run option in pangulu_common.h:
> **#define   GPU_OPEN**

## Preprocess methods
Now offering two types of preprocessing options.

### MC64
You need to open the mc64 preprocessing option in pangulu_common.h: 
> **#define   PANGULU_MC64**

### Metis
You will need the following two actions:    \
1. You need to open the Metis compilation option in make.inc: 
> **METISFLAGS  =  -lmetis**

2. You need to open the Metis preprocessing option in pangulu_common.h: 
> **#define   METIS**

note: METIS needs to be 64-bit.

## Calculation Type
PanguLU currently offer two types of accuracy.

### Double
If you want use double in calculation, You need to change the calculation type in pangulu_common.h:
>**#define calculate_type double**

Then you need to open the MPI_double option in pangulu_common.h:
>**#define MPI_VAL_TYPE MPI_DOUBLE**


### Float
If you want use float in calculation, You need to change the calculation type in pangulu_common.h:
>**#define calculate_type float**

Then you need to open the MPI_float option in pangulu_common.h:
>**#define MPI_VAL_TYPE MPI_FLOAT**


## Execution of PanguLU
PanguLU is to complete the operation of solving AX=b, and the test file is placed in the test folder. The test is to first perform the LU numerical decomposition of the matrix test.mtx, and use Ly=b to complete the lower triangular solution and Ux=y to complete the upper triangular solution test method.
### run command

> **mpirun -np process_number ./PanguLU -NB NB_number -F Smatrix_name**
 
process_number : this process number 
NB_number : the number of processes required is equal to the product of P and Q; 
Smatrix_name : the Matrix name in csr format.(This matrix needs to be decomposed symbolically)

You can also use the run.sh, for example:

> **bash run Smatrix_name NB_number process_number**

### test sample

> **mpirun -np 6 ./PanguLU -NB 2 -F test.mtx**

or use the run.sh:
> **bash run.sh test.mtx 2 6**


In this example,six processes are used to test, the  NB_number is 2 ,P_number is 2,Q_number is 3, matrix name is test.mtx

## Release version
<p align='left'>Oct 19,2021 Version 1.0</p>
<p align='left'>Jul &nbsp;&thinsp;22,2022 Version 2.0</p>
<p align='left'>Nov 28,2022 Version 2.1</p>
<p align='left'>Apr 02,2023 Version 3.0</p>
<p align='left'>Jun 17,2023 Version 3.1</p>

 





