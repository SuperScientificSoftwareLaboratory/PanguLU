#ifdef PANGULU_LOG_INFO
#ifndef PANGULU_LOG_WARNING
#define PANGULU_LOG_WARNING
#endif
#endif

#ifdef PANGULU_LOG_WARNING
#ifndef PANGULU_LOG_ERROR
#define PANGULU_LOG_ERROR
#endif
#endif

#ifdef PANGULU_LOG_ERROR
#define PANGULU_E_NB_IS_ZERO "[PanguLU Error] Block size 'nb' is zero.\n"
#define PANGULU_E_INVALID_HEAP_SELECT "[PanguLU Error] Invalid heap comparison strategy selected.\n"
#define PANGULU_E_HEAP_FULL "[PanguLU Error] Task heap is full. Cannot insert new task.\n"
#define PANGULU_E_HEAP_EMPTY "[PanguLU Error] Task heap is empty on this MPI rank.\n"
#define PANGULU_E_CPU_MEM "[PanguLU Error] Failed to allocate " FMT_PANGULU_INT64_T " bytes. CPU memory exhausted. (%s:%lld)\n", size, file, line
#define PANGULU_E_ROW_IS_NULL "[PanguLU Error] Column %d (0-based) contains no non-zero elements. Exiting.\n", i
#define PANGULU_E_K_ID "[PanguLU Error] Invalid kernel ID " FMT_PANGULU_INT64_T " in numeric factorisation.\n", kernel_id
#define PANGULU_E_ASYM "[PanguLU Error] MPI_Barrier_asym failed.\n"
#define PANGULU_E_ADD_DIA "[PanguLU Error] Failed to add diagonal element in pangulu_add_diagonal_element.\n"
#define PANGULU_E_ROW_IS_ZERO "[PanguLU Error] Input matrix is invalid: order is zero.\n"
#define PANGULU_E_MAX_NULL "[PanguLU Error] All elements in column %d (0-based) are zero. Exiting.\n", i
#define PANGULU_E_OPTION_IS_NULLPTR "[PanguLU Error] initialisation options pointer is NULL. (pangulu_init)\n"
#define PANGULU_E_GSTRF_OPTION_IS_NULLPTR "[PanguLU Error] Options pointer is NULL. (pangulu_gstrf)\n"
#define PANGULU_E_GSTRS_OPTION_IS_NULLPTR "[PanguLU Error] Options pointer is NULL. (pangulu_gstrs)\n"
#define PANGULU_E_BIN_FULL "[PanguLU Error] No available slot in bin (slot_capacity = %d). Allocation failed.\n", bin->slot_capacity
#define PANGULU_E_RECYCLE_QUEUE_FULL "[PanguLU Error] Recycle queue is full. Cannot recycle slot.\n"
#define PANGULU_E_TASK_QUEUE_FULL "[PanguLU Error] No available task slot. Allocation failed.\n"
#define PANGULU_E_COMPLEX_MISMATCH "[PanguLU Error] Input matrix type (complex/real) does not match PanguLU configuration. Exiting.\n"
#define PANGULU_E_ELEM_SIZE_MISMATCH "[PanguLU Error] Input element size is %lld B, but expected %lld B in PanguLU. Exiting.\n", sizeof(sparse_value_t), sizeof(calculate_type)
#define PANGULU_E_MPI_BUF_WAIT_EXCEED "[PanguLU Error] MPI receive buffer for other ranks is too small, PanguLU can not continue.\n[PanguLU Error] Please increase 'init_options.mpi_recv_buffer_level' if memory allows.\n"
#else 
#define PANGULU_E_NB_IS_ZERO ""
#define PANGULU_E_INVALID_HEAP_SELECT ""
#define PANGULU_E_HEAP_FULL ""
#define PANGULU_E_HEAP_EMPTY ""
#define PANGULU_E_CPU_MEM ""
#define PANGULU_E_ROW_IS_NULL ""
#define PANGULU_E_K_ID ""
#define PANGULU_E_ASYM ""
#define PANGULU_E_ADD_DIA ""
#define PANGULU_E_ROW_IS_ZERO ""
#define PANGULU_E_MAX_NULL ""
#define PANGULU_E_OPTION_IS_NULLPTR ""
#define PANGULU_E_GSTRF_OPTION_IS_NULLPTR ""
#define PANGULU_E_GSTRS_OPTION_IS_NULLPTR ""
#define PANGULU_E_BIN_FULL ""
#define PANGULU_E_RECYCLE_QUEUE_FULL ""
#define PANGULU_E_TASK_QUEUE_FULL ""
#define PANGULU_E_COMPLEX_MISMATCH ""
#define PANGULU_E_ELEM_SIZE_MISMATCH ""
#define PANGULU_E_MPI_BUF_WAIT_EXCEED ""
#endif

#ifdef PANGULU_LOG_WARNING
#define PANGULU_W_INSUFFICIENT_MPI_BUF "[PanguLU Warning] MPI receive buffer is too small. Consider increasing 'init_options.mpi_recv_buffer_level'.\n"
#define PANGULU_W_MPI_BUF_FULL "[PanguLU Warning] MPI receive buffer for other ranks is full. PanguLU may run slowly.\n[PanguLU Warning] Consider increasing 'init_options.mpi_recv_buffer_level' if memory allows.\n"
#define PANGULU_W_MC64_FAIL "[PanguLU Warning] MC64 reordering failed. Proceeding with original matrix ordering.\n"
#define PANGULU_W_BIND_CORE_FAIL "[PanguLU Warning] Failed to bind thread to core %d: %s\n", core, strerror(errno)
#define PANGULU_W_PERF_MODE_ON "[PanguLU Warning] Macro PANGULU_PERF is defined. Numeric factorisation may run in performance mode and be slower.\n"
#else
#define PANGULU_W_INSUFFICIENT_MPI_BUF ""
#define PANGULU_W_MPI_BUF_FULL ""
#define PANGULU_W_MC64_FAIL ""
#define PANGULU_W_BIND_CORE_FAIL ""
#define PANGULU_W_PERF_MODE_ON ""
#endif

#ifdef PANGULU_LOG_INFO
#define PANGULU_I_NUMERIC_CHECK "[PanguLU Info] Numerical check: || LUx - Ax || / || Ax || = %le\n", residual_norm2 / rhs_norm2
#define PANGULU_I_TIME_REORDER "[PanguLU Info] Reordering time: %lf s\n", elapsed_time
#define PANGULU_I_TIME_SYMBOLIC "[PanguLU Info] Symbolic factorisation time: %lf s\n", elapsed_time
#define PANGULU_I_TIME_PRE "[PanguLU Info] Preprocessing time: %lf s\n", elapsed_time
#define PANGULU_I_TIME_SPTRSV "[PanguLU Info] Solving time (SpTRSV): %lf s\n", elapsed_time
#define PANGULU_I_SYMBOLIC_NONZERO "[PanguLU Info] Symbolic nonzero count: " FMT_PANGULU_EXBLOCK_PTR "\n", *symbolic_nnz
#define PANGULU_I_MEMUSAGE_HOST "[PanguLU Info] Host memory usage: %.0f MiB (bytes = %llu)\n", (double)total / 1024.0, total * 1024
#define PANGULU_I_MEMUSAGE_DEVICE "[PanguLU Info] GPU memory usage: %.0f MiB (bytes = %llu)\n", (double)total_gpu_mem_byte / (1024.0 * 1024.0), total_gpu_mem_byte
#define PANGULU_I_PERF_TABLE_HEAD "[PanguLU Info] rank\tkernel_time\ttime_recv\trecv_count\n"
#define PANGULU_I_PERF_TABLE_ROW "[PanguLU Info] #%d\t%lf\t%lf\t%lld\t\t\n", rank, global_stat.time_outer_kernel, global_stat.time_recv, global_stat.recv_cnt

#ifdef PANGULU_PERF
#define PANGULU_I_TIME_NUMERICAL "[PanguLU Info] Numeric factorisation time: %lf s. %lld flop, %lf GFlop/s\n", elapsed_time, global_stat.flop, ((double)global_stat.flop) / elapsed_time / 1e9
#else
#define PANGULU_I_TIME_NUMERICAL "[PanguLU Info] Numeric factorisation time: %lf s\n", elapsed_time
#endif

#ifdef GPU_OPEN
#ifdef METIS
#define PANGULU_I_BASIC_INFO "[PanguLU Info]\n\
[PanguLU Info] --- PanguLU Configuration & Matrix Info ---\n\
[PanguLU Info]       Matrix Order:             " FMT_PANGULU_INT64_T "\n\
[PanguLU Info]       #NNZ:                     " FMT_PANGULU_EXBLOCK_PTR "\n\
[PanguLU Info]       Matrix Block Order (nb):  " FMT_PANGULU_INT32_T "\n\
[PanguLU Info]       MPI Processes Count:      " FMT_PANGULU_INT32_T "\n\
[PanguLU Info]       MPI Recv Buffer Level:    " "%.2f" "\n\
[PanguLU Info]       METIS Index Type:         %s\n\
[PanguLU Info]       GPU:                      Enabled\n\
[PanguLU Info]       GPU Kernel Warp/Block:    %d\n\
[PanguLU Info]       GPU Data Move Warp/Block: %d\n\
[PanguLU Info] -------------------------------------------\n[PanguLU Info]\n\
", n, origin_smatrix->columnpointer[n], nb, size, common->basic_param, (sizeof(reordering_int_t) == 4) ? ("i32") : ((sizeof(reordering_int_t) == 8) ? ("i64") : ("unknown")), \
init_options->gpu_kernel_warp_per_block, init_options->gpu_data_move_warp_per_block
#else
#define PANGULU_I_BASIC_INFO "[PanguLU Info]\n\
[PanguLU Info] --- PanguLU Configuration & Matrix Info ---\n\
[PanguLU Info]       Matrix Order:             " FMT_PANGULU_INT64_T "\n\
[PanguLU Info]       #NNZ:                     " FMT_PANGULU_EXBLOCK_PTR "\n\
[PanguLU Info]       Matrix Block Order (nb):  " FMT_PANGULU_INT32_T "\n\
[PanguLU Info]       MPI Processes Count:      " FMT_PANGULU_INT32_T "\n\
[PanguLU Info]       MPI Recv Buffer Level:    " "%.2f" "\n\
[PanguLU Info]       Reordering Index Type:    %s\n\
[PanguLU Info]       Reordering Thread Count:  %d\n\
[PanguLU Info]       GPU:                      Enabled\n\
[PanguLU Info]       GPU Kernel Warp/Block:    %d\n\
[PanguLU Info]       GPU Data Move Warp/Block: %d\n\
[PanguLU Info] -------------------------------------------\n[PanguLU Info]\n\
", n, origin_smatrix->columnpointer[n], nb, size, common->basic_param, (sizeof(reordering_int_t) == 4) ? ("i32") : ((sizeof(reordering_int_t) == 8) ? ("i64") : ("unknown")), init_options->reordering_nthread, \
init_options->gpu_kernel_warp_per_block, init_options->gpu_data_move_warp_per_block
#endif
#else
#ifdef METIS
#define PANGULU_I_BASIC_INFO "[PanguLU Info]\n\
[PanguLU Info] --- PanguLU Configuration & Matrix Info ---\n\
[PanguLU Info]       Matrix Order:             " FMT_PANGULU_INT64_T "\n\
[PanguLU Info]       #NNZ:                     " FMT_PANGULU_EXBLOCK_PTR "\n\
[PanguLU Info]       Matrix Block Order (nb):  " FMT_PANGULU_INT32_T "\n\
[PanguLU Info]       MPI Processes Count:      " FMT_PANGULU_INT32_T "\n\
[PanguLU Info]       MPI Recv Buffer Level:    " "%.2f" "\n\
[PanguLU Info]       METIS Index Type:         %s\n\
[PanguLU Info]       GPU:                      Disabled\n\
[PanguLU Info] -------------------------------------------\n[PanguLU Info]\n\
", n, origin_smatrix->columnpointer[n], nb, size, common->basic_param, (sizeof(reordering_int_t) == 4) ? ("i32") : ((sizeof(reordering_int_t) == 8) ? ("i64") : ("unknown"))
#else
#define PANGULU_I_BASIC_INFO "[PanguLU Info]\n\
[PanguLU Info] --- PanguLU Configuration & Matrix Info ---\n\
[PanguLU Info]       Matrix Order:             " FMT_PANGULU_INT64_T "\n\
[PanguLU Info]       #NNZ:                     " FMT_PANGULU_EXBLOCK_PTR "\n\
[PanguLU Info]       Matrix Block Order (nb):  " FMT_PANGULU_INT32_T "\n\
[PanguLU Info]       MPI Processes Count:      " FMT_PANGULU_INT32_T "\n\
[PanguLU Info]       MPI Recv Buffer Level:    " "%.2f" "\n\
[PanguLU Info]       Reordering Index Type:    %s\n\
[PanguLU Info]       Reordering Thread Count:  %d\n\
[PanguLU Info]       GPU:                      Disabled\n\
[PanguLU Info] -------------------------------------------\n[PanguLU Info]\n\
", n, origin_smatrix->columnpointer[n], nb, size, common->basic_param, (sizeof(reordering_int_t) == 4) ? ("i32") : ((sizeof(reordering_int_t) == 8) ? ("i64") : ("unknown")), init_options->reordering_nthread
#endif
#endif

#else
#define PANGULU_I_NUMERIC_CHECK ""
#define PANGULU_I_TIME_REORDER ""
#define PANGULU_I_TIME_SYMBOLIC ""
#define PANGULU_I_TIME_PRE ""
#define PANGULU_I_TIME_SPTRSV ""
#define PANGULU_I_SYMBOLIC_NONZERO ""
#define PANGULU_I_MEMUSAGE_HOST ""
#define PANGULU_I_MEMUSAGE_DEVICE ""
#define PANGULU_I_PERF_TABLE_HEAD ""
#define PANGULU_I_PERF_TABLE_ROW ""
#define PANGULU_I_TIME_NUMERICAL ""
#define PANGULU_I_BASIC_INFO ""
#endif