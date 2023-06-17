#ifndef PANGULU_KERNEL_INTERFACE_H
#define PANGULU_KERNEL_INTERFACE_H

#include "pangulu_common.h"

#include "pangulu_time.h"

#ifdef GPU_OPEN
#include "pangulu_getrf_fp64_cuda.h"
#include "pangulu_tstrf_fp64_cuda.h"
#include "pangulu_gessm_fp64_cuda.h"
#include "pangulu_ssssm_fp64_cuda.h"
#include "pangulu_addmatrix_cuda.h"
#endif

#include "pangulu_getrf_fp64.h"
#include "pangulu_tstrf_fp64.h"
#include "pangulu_gessm_fp64.h"
#include "pangulu_ssssm_fp64.h"
#include "pangulu_sptrsv_fp64.h"
#include "pangulu_spmv_fp64.h"
#include "pangulu_addmatrix.h"

void pangulu_getrf_interface(pangulu_Smatrix *A, pangulu_Smatrix *L, pangulu_Smatrix *U,
                             pangulu_Smatrix *calculate_L, pangulu_Smatrix *calculate_U)
{

#ifdef CHECK_TIME
    struct timeval GET_TIME_START;
    pangulu_time_check_begin(&GET_TIME_START);
#endif

#ifdef GPU_OPEN

#ifdef ADD_GPU_MEMORY

#ifdef ADAPTIVE_KERNEL_SELECTION
    int nnzA = A->nnz;
    if (nnzA < 6309)
    { // 6309≈1e3.8
        pangulu_getrf_interface_C_V1(A, L, U);
    }
    else if (nnzA < 1e4)
    {
        pangulu_getrf_interface_G_V1(A, L, U);
    }
    else
    {
        pangulu_getrf_interface_G_V2(A, L, U);
    }
#else
    pangulu_getrf_interface_G_V1(A, L, U);
#endif
    cudaDeviceSynchronize();

#else
    pangulu_Smatrix_CUDA_memcpy_struct_CSC(calculate_L, L);
    pangulu_Smatrix_CUDA_memcpy_struct_CSC(calculate_U, U);
    pangulu_Smatrix_CUDA_memcpy_nnzU(calculate_U, U);
    pangulu_getrf_fp64_cuda(A, calculate_L, calculate_U);
    pangulu_Smatrix_CUDA_memcpy_value_CSC(L, calculate_L);
    pangulu_Smatrix_CUDA_memcpy_value_CSC(U, calculate_U);

#endif
#else
    pangulu_getrf_fp64(A, L, U);

#endif

#ifdef CHECK_TIME
    TIME_getrf += pangulu_time_check_end(&GET_TIME_START);
#endif
}

void pangulu_tstrf_interface(pangulu_Smatrix *A, pangulu_Smatrix *save_X, pangulu_Smatrix *U,
                             pangulu_Smatrix *calculate_X, pangulu_Smatrix *calculate_U)
{

#ifdef CHECK_TIME
    struct timeval GET_TIME_START;
    pangulu_time_check_begin(&GET_TIME_START);
#endif

#ifdef GPU_OPEN

#ifndef GPU_TSTRF

#ifndef CPU_OPTION
    pangulu_Smatrix_CUDA_memcpy_value_CSC_cal_length(calculate_X, A);

    pangulu_tstrf_interface_CPU(A, calculate_X, U);
#else

    int_t CPU_choice2 = A->nnz;
    calculate_type CPU_choice3 = CPU_choice2 / ((calculate_type)nrecord * (calculate_type)CPU_choice1);
    int_t TSTRF_choice_CPU = Select_Function_CPU(CPU_choice1, CPU_choice3, nrecord);
    pangulu_tstrf_kernel_choice_CPU(A, calculate_X, U, TSTRF_choice_CPU);
#endif

#else

#ifdef ADD_GPU_MEMORY
#ifdef ADAPTIVE_KERNEL_SELECTION
    int nnzB = A->nnz;
    if (nnzB < 6309)
    {
        // 6309≈1e3.8
        if (nnzB < 3981)
        { // 3981≈1e3.6
            pangulu_tstrf_interface_C_V1(A, calculate_X, U);
        }
        else
        {
            pangulu_tstrf_interface_C_V2(A, calculate_X, U);
        }
    }
    else
    {
        if (nnzB < 1e4)
        {
            pangulu_tstrf_interface_G_V2(A, calculate_X, U);
        }
        else if (nnzB < 19952)
        { // 19952≈1e4.3
            pangulu_tstrf_interface_G_V1(A, calculate_X, U);
        }
        else
        {
            pangulu_tstrf_interface_G_V3(A, calculate_X, U);
        }
    }
#else
    pangulu_tstrf_interface_G_V1(A, calculate_X, U);
#endif
    cudaDeviceSynchronize();

#else

    pangulu_Smatrix_CUDA_memcpy_complete_CSR(calculate_U, U);
    pangulu_tstrf_interface(A, calculate_X, calculate_U);
    pangulu_Smatrix_CUDA_memcpy_value_CSC(A, calculate_X);
#endif

#endif

#else
    pangulu_pangulu_Smatrix_memcpy_value_CSC_copy_length(calculate_X, A);
    pangulu_tstrf_interface_CPU_CSR(A, calculate_X, U);
    pangulu_pangulu_Smatrix_memcpy_columnpointer_CSC(save_X, A);
#endif

#ifdef CHECK_TIME
    TIME_tstrf += pangulu_time_check_end(&GET_TIME_START);
#endif
}

void pangulu_gessm_interface(pangulu_Smatrix *A, pangulu_Smatrix *save_X, pangulu_Smatrix *L,
                             pangulu_Smatrix *calculate_X, pangulu_Smatrix *calculate_L)
{
#ifdef CHECK_TIME
    struct timeval GET_TIME_START;
    pangulu_time_check_begin(&GET_TIME_START);
#endif

#ifdef GPU_OPEN

#ifndef GPU_GESSM

#ifndef CPU_OPTION
    pangulu_Smatrix_CUDA_memcpy_value_CSC(A, A);
    pangulu_transport_pangulu_Smatrix_CSC_to_CSR(A);
    pangulu_pangulu_Smatrix_memcpy_value_CSR_copy_length(calculate_X, A);
    pangulu_transport_pangulu_Smatrix_CSC_to_CSR(L);
    pangulu_gessm_interface_CPU(A, L, calculate_X);
    pangulu_transport_pangulu_Smatrix_CSR_to_CSC(A);
#else

    /*******************Choose the best performance（性能择优）*************************/
    int_t CPU_choice2 = A->nnz;
    calculate_type CPU_choice3 = CPU_choice2 / ((calculate_type)nrecord * (calculate_type)CPU_choice1);
    int_t GESSM_choice_CPU = Select_Function_CPU(CPU_choice1, CPU_choice3, nrecord);
    pangulu_gessm_kernel_choice_CPU(A, L, calculate_X, GESSM_choice_CPU);
#endif
#else

#ifdef ADD_GPU_MEMORY
#ifdef ADAPTIVE_KERNEL_SELECTION
    int nnzL = L->nnz;
    if (nnzL < 7943)
    {
        // 7943≈1e3.9
        if (nnzL < 3981)
        { // 3981≈1e3.6
            pangulu_gessm_interface_C_V1(A, L, calculate_X);
        }
        else
        {
            pangulu_gessm_interface_C_V2(A, L, calculate_X);
        }
    }
    else
    {
        if (nnzL < 12589)
        {
            // 12589≈1e4.1
            pangulu_gessm_interface_G_V2(A, L, calculate_X);
        }
        else if (nnzL < 19952)
        { // 19952≈1e4.3
            pangulu_gessm_interface_G_V1(A, L, calculate_X);
        }
        else
        {
            pangulu_gessm_interface_G_V3(A, L, calculate_X);
        }
    }
#else
    pangulu_gessm_interface_G_V1(A, L, calculate_X);
#endif
    cudaDeviceSynchronize();

#else

    pangulu_Smatrix_CUDA_memcpy_complete_CSC(calculate_L, L);
    pangulu_gessm_interface(A, calculate_L, calculate_X);
#endif

#endif

#else
    pangulu_transport_pangulu_Smatrix_CSC_to_CSR(A);
    pangulu_pangulu_Smatrix_memcpy_value_CSR_copy_length(calculate_X, A);
    pangulu_gessm_interface_CPU_CSR(A, L, calculate_X);
    pangulu_transport_pangulu_Smatrix_CSR_to_CSC(A);
    pangulu_pangulu_Smatrix_memcpy_columnpointer_CSC(save_X, A);
#endif

#ifdef CHECK_TIME
    TIME_gessm += pangulu_time_check_end(&GET_TIME_START);
#endif
}

void pangulu_ssssm_interface(pangulu_Smatrix *A, pangulu_Smatrix *L, pangulu_Smatrix *U,
                             pangulu_Smatrix *calculate_L, pangulu_Smatrix *calculate_U)
{

#ifdef CHECK_TIME
    struct timeval GET_TIME_START;
    pangulu_time_check_begin(&GET_TIME_START);
#endif

#ifdef GPU_OPEN

#ifndef ADD_GPU_MEMORY
    pangulu_Smatrix_CUDA_memcpy_complete_CSC(calculate_L, L);
    pangulu_Smatrix_CUDA_memcpy_complete_CSC(calculate_U, U);
    pangulu_ssssm_fp64_cuda(A, calculate_L, calculate_U);
#else

#ifdef ADAPTIVE_KERNEL_SELECTION
    long long flops = 0;
    int n = A->row;
    for (int i = 0; i < n; i++)
    {
        for (int j = U->columnpointer[i]; j < U->columnpointer[i + 1]; j++)
        {
            int col_L = U->rowindex[j];
            flops += L->columnpointer[col_L + 1] - L->columnpointer[col_L];
        }
    }
    if (flops < 1e7)
    {
        if (flops < 3981071705)
        {
            // 3981071705≈1e9.6
            pangulu_ssssm_interface_G_V2(A, L, U);
        }
        else
        {
            pangulu_ssssm_interface_G_V1(A, L, U);
        }
    }
    else
    {
        if (flops < 63095)
        {
            // 63095≈1e4.8
            pangulu_ssssm_interface_C_V1(A, L, U);
        }
        else
        {
            pangulu_ssssm_interface_C_V2(A, L, U);
        }
    }
#else
    pangulu_ssssm_interface_G_V1(A, L, U);
#endif
    cudaDeviceSynchronize();
#endif
#else

    pangulu_ssssm_fp64(A, L, U);
#endif

#ifdef CHECK_TIME
    TIME_ssssm += pangulu_time_check_end(&GET_TIME_START);
#endif
}

#ifdef GPU_OPEN

void pangulu_addmatrix_interface(pangulu_Smatrix *A,
                                 pangulu_Smatrix *B)
{
    pangulu_add_pangulu_Smatrix_cuda(A, B);
}

#endif

void pangulu_addmatrix_interface_CPU(pangulu_Smatrix *A,
                                     pangulu_Smatrix *B)
{
    pangulu_add_pangulu_Smatrix_cpu(A, B);
}

void pangulu_spmv(pangulu_Smatrix *S, pangulu_vector *Z, pangulu_vector *answer, int vector_number)
{
    pangulu_spmv_cpu_xishu_csc(S, Z, answer, vector_number);
}

void pangulu_sptrsv(pangulu_Smatrix *S, pangulu_vector *answer, pangulu_vector *Z, int vector_number, int32_t tag)
{
    pangulu_sptrsv_cpu_xishu_csc(S, answer, Z, vector_number, tag);
}

void pangulu_vector_add(pangulu_vector *answer, pangulu_vector *Z)
{
    pangulu_vector_add_cpu(answer, Z);
}

void pangulu_vector_sub(pangulu_vector *answer, pangulu_vector *Z)
{
    pangulu_vector_sub_cpu(answer, Z);
}

void pangulu_vector_copy(pangulu_vector *answer, pangulu_vector *Z)
{
    pangulu_vector_copy_cpu(answer, Z);
}

#endif