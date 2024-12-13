#include "pangulu_common.h"

void pangulu_getrf_interface(pangulu_smatrix *a, pangulu_smatrix *l, pangulu_smatrix *u,
                             pangulu_smatrix *calculate_L, pangulu_smatrix *calculate_U)
{
    for(pangulu_int64_t i=0;i<u->nnz;i++){
        pangulu_int64_t now_row=u->rowindex[i];
        calculate_time+=(l->columnpointer[now_row+1]-l->columnpointer[now_row]);
    }
#ifdef CHECK_TIME
    struct timeval GET_TIME_START;
    pangulu_time_check_begin(&GET_TIME_START);
#endif

#ifdef GPU_OPEN

#ifdef ADD_GPU_MEMORY

#ifdef ADAPTIVE_KERNEL_SELECTION
    int nnzA = a->nnz;
    if (nnzA < 6309)
    { // 6309≈1e3.8
        pangulu_getrf_interface_C_V1(a, l, u);
    }
    else if (nnzA < 1e4)
    {
        pangulu_getrf_interface_G_V1(a, l, u);
    }
    else
    {
        pangulu_getrf_interface_G_V2(a, l, u);
    }
#else // ADAPTIVE_KERNEL_SELECTION
    pangulu_getrf_interface_G_V1(a, l, u);
#endif // ADAPTIVE_KERNEL_SELECTION
    cudaDeviceSynchronize();

#else // ADD_GPU_MEMORY
    pangulu_smatrix_cuda_memcpy_struct_csc(calculate_L, l);
    pangulu_smatrix_cuda_memcpy_struct_csc(calculate_U, u);
    pangulu_smatrix_cuda_memcpy_nnzu(calculate_U, u);
    pangulu_getrf_fp64_cuda(a, calculate_L, calculate_U);
    pangulu_smatrix_cuda_memcpy_value_csc(l, calculate_L);
    pangulu_smatrix_cuda_memcpy_value_csc(u, calculate_U);

#endif // ADD_GPU_MEMORY
#else // GPU_OPEN

    pangulu_getrf_fp64(a, l, u);

#endif // GPU_OPEN

#ifdef CHECK_TIME
    time_getrf += pangulu_time_check_end(&GET_TIME_START);
#endif
}

void pangulu_tstrf_interface(pangulu_smatrix *a, pangulu_smatrix *save_X, pangulu_smatrix *u,
                             pangulu_smatrix *calculate_X, pangulu_smatrix *calculate_U)
{
    // for(int_t i=0;i<a->nnz;i++){
    //     int_t now_col=a->columnindex[i];
    //     calculate_time+=(u->rowpointer[now_col+1]-u->rowpointer[now_col]);
    // }
#ifdef CHECK_TIME
    struct timeval GET_TIME_START;
    pangulu_time_check_begin(&GET_TIME_START);
#endif

#ifdef GPU_OPEN

#ifndef GPU_TSTRF

#ifndef CPU_OPTION
    pangulu_smatrix_cuda_memcpy_value_csc_cal_length(calculate_X, a);

    pangulu_tstrf_interface_cpu(a, calculate_X, u);
#else // CPU_OPTION

    pangulu_int64_t cpu_choice2 = a->nnz;
    calculate_type cpu_choice3 = cpu_choice2 / ((calculate_type)nrecord * (calculate_type)cpu_choice1);
    pangulu_int64_t TSTRF_choice_cpu = Select_Function_CPU(cpu_choice1, cpu_choice3, nrecord);
    pangulu_tstrf_kernel_choice_cpu(a, calculate_X, u, TSTRF_choice_cpu);
#endif // CPU_OPTION

#else // GPU_TSTRF

#ifdef ADD_GPU_MEMORY
#ifdef ADAPTIVE_KERNEL_SELECTION
    pangulu_int64_t nnzB = a->nnz;
    if (nnzB < 6309)
    {
        // 6309≈1e3.8
        if (nnzB < 3981)
        { // 3981≈1e3.6
            pangulu_tstrf_interface_C_V1(a, calculate_X, u);
        }
        else
        {
            pangulu_tstrf_interface_C_V2(a, calculate_X, u);
        }
    }
    else
    {
        if (nnzB < 1e4)
        {
            pangulu_tstrf_interface_G_V2(a, calculate_X, u);
        }
        else if (nnzB < 19952)
        { // 19952≈1e4.3
            pangulu_tstrf_interface_G_V1(a, calculate_X, u);
        }
        else
        {
            pangulu_tstrf_interface_G_V3(a, calculate_X, u);
        }
    }
#else // ADAPTIVE_KERNEL_SELECTION
    pangulu_tstrf_interface_G_V1(a, calculate_X, u);
#endif // ADAPTIVE_KERNEL_SELECTION
    cudaDeviceSynchronize();

#else // ADD_GPU_MEMORY

    pangulu_smatrix_cuda_memcpy_complete_csr(calculate_U, u);
    pangulu_tstrf_interface(a, calculate_X, calculate_U);
    pangulu_smatrix_cuda_memcpy_value_csc(a, calculate_X);
#endif // ADD_GPU_MEMORY

#endif // ADAPTIVE_KERNEL_SELECTION

#else // GPU_OPEN

    // csc
    tstrf_csc_csc(a->row, u->columnpointer, u->rowindex, u->value_csc, a->columnpointer, a->rowindex, a->value_csc);
    pangulu_pangulu_smatrix_memcpy_columnpointer_csc(save_X, a);

    // // csr
    // pangulu_transpose_pangulu_smatrix_csc_to_csr(a);
    // pangulu_pangulu_smatrix_memcpy_value_csc_copy_length(calculate_X, a);
    // pangulu_tstrf_fp64_CPU_6(a, calculate_X, u);
    // pangulu_transpose_pangulu_smatrix_csr_to_csc(a);
    // pangulu_pangulu_smatrix_memcpy_columnpointer_csc(save_X, a);

#endif // GPU_OPEN

#ifdef CHECK_TIME
    time_tstrf += pangulu_time_check_end(&GET_TIME_START);
#endif
}

void pangulu_gessm_interface(pangulu_smatrix *a, pangulu_smatrix *save_X, pangulu_smatrix *l,
                             pangulu_smatrix *calculate_X, pangulu_smatrix *calculate_L)
{
    for(pangulu_int64_t i=0;i<a->nnz;i++){
        pangulu_int64_t now_row=a->rowindex[i];
        calculate_time+=(l->columnpointer[now_row+1]-l->columnpointer[now_row]);
    }
#ifdef CHECK_TIME
    struct timeval GET_TIME_START;
    pangulu_time_check_begin(&GET_TIME_START);
#endif

#ifdef GPU_OPEN

#ifndef GPU_GESSM

#ifndef CPU_OPTION
    pangulu_smatrix_cuda_memcpy_value_csc(a, a);
    pangulu_transpose_pangulu_smatrix_csc_to_csr(a);
    pangulu_pangulu_smatrix_memcpy_value_csr_copy_length(calculate_X, a);
    pangulu_transpose_pangulu_smatrix_csc_to_csr(l);
    pangulu_gessm_interface_cpu(a, l, calculate_X);
    pangulu_transpose_pangulu_smatrix_csr_to_csc(a);
#else

    /*******************Choose the best performance（性能择优）*************************/
    pangulu_int64_t cpu_choice2 = a->nnz;
    calculate_type cpu_choice3 = cpu_choice2 / ((calculate_type)nrecord * (calculate_type)cpu_choice1);
    pangulu_int64_t GESSM_choice_cpu = Select_Function_CPU(cpu_choice1, cpu_choice3, nrecord);
    pangulu_gessm_kernel_choice_cpu(a, l, calculate_X, GESSM_choice_cpu);
#endif
#else

#ifdef ADD_GPU_MEMORY
#ifdef ADAPTIVE_KERNEL_SELECTION
    int nnzL = l->nnz;
    if (nnzL < 7943)
    {
        // 7943≈1e3.9
        if (nnzL < 3981)
        { // 3981≈1e3.6
            pangulu_gessm_interface_C_V1(a, l, calculate_X);
        }
        else
        {
            pangulu_gessm_interface_C_V2(a, l, calculate_X);
        }
    }
    else
    {
        if (nnzL < 12589)
        {
            // 12589≈1e4.1
            pangulu_gessm_interface_G_V2(a, l, calculate_X);
        }
        else if (nnzL < 19952)
        { // 19952≈1e4.3
            pangulu_gessm_interface_g_v1(a, l, calculate_X);
        }
        else
        {
            pangulu_gessm_interface_G_V3(a, l, calculate_X);
        }
    }
#else
    pangulu_gessm_interface_g_v1(a, l, calculate_X);
#endif
    cudaDeviceSynchronize();

#else

    pangulu_smatrix_cuda_memcpy_complete_csc(calculate_L, l);
    pangulu_gessm_interface(a, calculate_L, calculate_X);
#endif

#endif

#else
    pangulu_pangulu_smatrix_memcpy_value_csr_copy_length(calculate_X, a);
    pangulu_gessm_fp64_cpu_6(a, l, calculate_X);
    pangulu_pangulu_smatrix_memcpy_columnpointer_csc(save_X, a);
    // pangulu_transpose_pangulu_smatrix_csc_to_csr(a);
    // pangulu_pangulu_smatrix_memcpy_value_csr_copy_length(calculate_X, a);
    // pangulu_gessm_interface_CPU_csr(a, l, calculate_X);
    // pangulu_transpose_pangulu_smatrix_csr_to_csc(a);
    // pangulu_pangulu_smatrix_memcpy_columnpointer_csc(save_X, a);
#endif

#ifdef CHECK_TIME
    time_gessm += pangulu_time_check_end(&GET_TIME_START);
#endif
}

void pangulu_ssssm_interface(pangulu_smatrix *a, pangulu_smatrix *l, pangulu_smatrix *u,
                             pangulu_smatrix *calculate_L, pangulu_smatrix *calculate_U)
{
    for(pangulu_int64_t i=0;i<u->nnz;i++){
        pangulu_int64_t now_row=u->rowindex[i];
        calculate_time+=(l->columnpointer[now_row+1]-l->columnpointer[now_row]);
    }
#ifdef CHECK_TIME
    struct timeval GET_TIME_START;
    pangulu_time_check_begin(&GET_TIME_START);
#endif

#ifdef GPU_OPEN

#ifndef ADD_GPU_MEMORY
    pangulu_smatrix_cuda_memcpy_complete_csc(calculate_L, l);
    pangulu_smatrix_cuda_memcpy_complete_csc(calculate_U, u);
    pangulu_ssssm_fp64_cuda(a, calculate_L, calculate_U);
#else

#ifdef ADAPTIVE_KERNEL_SELECTION
    long long flops = 0;
    int n = a->row;
    for (int i = 0; i < n; i++)
    {
        for (int j = u->columnpointer[i]; j < u->columnpointer[i + 1]; j++)
        {
            int col_L = u->rowindex[j];
            flops += l->columnpointer[col_L + 1] - l->columnpointer[col_L];
        }
    }
    if (flops < 1e7)
    {
        if (flops < 3981071705)
        {
            // 3981071705≈1e9.6
            pangulu_ssssm_interface_G_V2(a, l, u);
        }
        else
        {
            pangulu_ssssm_interface_G_V1(a, l, u);
        }
    }
    else
    {
        if (flops < 63095)
        {
            // 63095≈1e4.8
            pangulu_ssssm_interface_C_V1(a, l, u);
        }
        else
        {
            pangulu_ssssm_interface_C_V2(a, l, u);
        }
    }
#else
    pangulu_ssssm_interface_G_V1(a, l, u);
#endif
    cudaDeviceSynchronize();
#endif
#else

    pangulu_ssssm_fp64(a, l, u);
#endif

#ifdef CHECK_TIME
    time_ssssm += pangulu_time_check_end(&GET_TIME_START);
#endif
}

#ifdef GPU_OPEN

void pangulu_addmatrix_interface(pangulu_smatrix *a,
                                 pangulu_smatrix *b)
{
    pangulu_add_pangulu_smatrix_cuda(a, b);
}

#endif

void pangulu_addmatrix_interface_cpu(pangulu_smatrix *a,
                                     pangulu_smatrix *b)
{
    pangulu_add_pangulu_smatrix_cpu(a, b);
}

void pangulu_spmv(pangulu_smatrix *s, pangulu_vector *z, pangulu_vector *answer, int vector_number)
{
    pangulu_spmv_cpu_xishu_csc(s, z, answer, vector_number);
}

void pangulu_sptrsv(pangulu_smatrix *s, pangulu_vector *answer, pangulu_vector *z, int vector_number, int32_t tag)
{
    pangulu_sptrsv_cpu_xishu_csc(s, answer, z, vector_number, tag);
}

void pangulu_vector_add(pangulu_vector *answer, pangulu_vector *z)
{
    pangulu_vector_add_cpu(answer, z);
}

void pangulu_vector_sub(pangulu_vector *answer, pangulu_vector *z)
{
    pangulu_vector_sub_cpu(answer, z);
}

void pangulu_vector_copy(pangulu_vector *answer, pangulu_vector *z)
{
    pangulu_vector_copy_cpu(answer, z);
}