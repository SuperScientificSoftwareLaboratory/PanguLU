#ifdef PANGULU_EN

#ifdef PANGULU_LOG_ERROR
#define PANGULU_E_NB_IS_ZERO "[PanguLU Error] nb is zero.\n"
#define PANGULU_E_INVALID_HEAP_SELECT "[PanguLU Error] Invalid heap comparing strategy.\n"
#define PANGULU_E_HEAP_FULL "[PanguLU Error] The heap is full on rank " FMT_PANGULU_INT32_T ".\n", rank
#define PANGULU_E_HEAP_EMPTY "[PanguLU Error] The heap is empty on rank " FMT_PANGULU_INT32_T ".\n", rank
#define PANGULU_E_CPU_MEM "[PanguLU Error] Failed to allocate " FMT_PANGULU_INT64_T " byte(s). CPU memory is not enough. %s:" FMT_PANGULU_INT64_T "\n", size, file, line
#define PANGULU_E_ISEND_CSR "[PanguLU Error] pangulu_isend_whole_pangulu_smatrix_csr error. value != s->value.\n"
#define PANGULU_E_ISEND_CSC "[PanguLU Error] pangulu_isend_whole_pangulu_smatrix_csc error. value != s->value_csc.\n"
#define PANGULU_E_ROW_IS_NULL "[PanguLU Error] The matrix has zero row(s).\n"
#define PANGULU_E_ROW_DONT_HAVE_DIA "[PanguLU Error] Row[" FMT_PANGULU_EXBLOCK_IDX "] don't have diagonal element.\n", i
#define PANGULU_E_ERR_IN_RRCL "[PanguLU Error] Invalid numeric factorization task on rank " FMT_PANGULU_INT32_T ". row=" FMT_PANGULU_INT64_T " col=" FMT_PANGULU_INT64_T " level=" FMT_PANGULU_INT64_T "\n", rank, row, col, level
#define PANGULU_E_K_ID "[PanguLU Error] Invalid kernel id " FMT_PANGULU_INT64_T " for numeric factorization.\n", kernel_id
#define PANGULU_E_ASYM "[PanguLU Error] MPI_Barrier_asym error.\n"
#define PANGULU_E_ADD_DIA "[PanguLU Error] pangulu_add_diagonal_element error\n"
#define PANGULU_E_CUDA_MALLOC "[PanguLU Error] Failed to cudaMalloc %lu byte(s). GPU memory is not enough.\n", size
#define PANGULU_E_ROW_IS_ZERO "[PanguLU Error] Invalid input matrix.\n"
#define PANGULU_E_MAX_NULL "[PanguLU Error] pangulu_mc64 internal error. (now_row_max==0)\n"
#define PANGULU_E_WORK_ERR "[PanguLU Error]  Invalid kernel id " FMT_PANGULU_INT64_T " for sptrsv.\n", kernel_id
#define PANGULU_E_BIP_PTR_INVALID "[PanguLU Error] Invalid pangulu_block_info pointer.\n"
#define PANGULU_E_BIP_INVALID "[PanguLU Error] Invalid pangulu_block_info.\n"
#define PANGULU_E_BIP_NOT_EMPTY "[PanguLU Error] Block info pool is not empty.\n"
#define PANGULU_E_BIP_OUT_OF_RANGE "[PanguLU Error] PANGULU_BIP index out of range.\n"
#define PANGULU_E_OPTION_IS_NULLPTR "[PanguLU Error] Option struct pointer is NULL. (pangulu_init)\n"
#define PANGULU_E_GSTRF_OPTION_IS_NULLPTR "[PanguLU Error] Option struct pointer is NULL. (pangulu_gstrf)\n"
#define PANGULU_E_GSTRS_OPTION_IS_NULLPTR "[PanguLU Error] Option struct pointer is NULL. (pangulu_gstrs)\n"
#endif // PANGULU_LOG_ERROR

#ifdef PANGULU_LOG_WARNING
#define PANGULU_W_RANK_HEAP_DONT_NULL "[PanguLU Warning] " FMT_PANGULU_INT64_T " task remaining on rank " FMT_PANGULU_INT32_T ".\n", heap->length, rank
#define PANGULU_W_ERR_RANK "[PanguLU Warning] Receiving message error on rank " FMT_PANGULU_INT32_T ".\n", rank
#define PANGULU_W_BIP_INCREASE_SPEED_TOO_SMALL "[PanguLU Warning] PANGULU_BIP_INCREASE_SPEED too small.\n"
#define PANGULU_W_GPU_BIG_BLOCK "[PanguLU Warning] When GPU is open, init_options->nb > 256 and pangulu_inblock_idx isn't pangulu_uint32_t, performance will be limited.\n"
#define PANGULU_W_COMPLEX_FALLBACK "[PanguLU Warning] Calculating complex value on GPU is not supported. Fallback to CPU.\n"
#endif // PANGULU_LOG_WARNING

#ifdef PANGULU_LOG_INFO
#define PANGULU_I_VECT2NORM_ERR "[PanguLU Info] || Ax - B || / || Ax || = %12.4le.\n", error
#define PANGULU_I_CHECK_PASS  "[PanguLU Info] Check ------------------------------------- pass\n"
#define PANGULU_I_CHECK_ERROR "[PanguLU Info] Check ------------------------------------ error\n"
#define PANGULU_I_DEV_IS "[PanguLU Info] Device is %s.\n", prop.name
#define PANGULU_I_TASK_INFO "[PanguLU Info] Info of inserting task is: row=" FMT_PANGULU_INT64_T " col=" FMT_PANGULU_INT64_T " level=" FMT_PANGULU_INT64_T " kernel=" FMT_PANGULU_INT64_T ".\n", row, col, task_level, kernel_id
#define PANGULU_I_HEAP_LEN "[PanguLU Info] heap.length=" FMT_PANGULU_INT64_T " heap.capacity=" FMT_PANGULU_INT64_T "\n", heap->length, heap->max_length
#define PANGULU_I_ADAPTIVE_KERNEL_SELECTION_ON  "[PanguLU Info] ADAPTIVE_KERNEL_SELECTION ------------- ON\n"
#define PANGULU_I_ADAPTIVE_KERNEL_SELECTION_OFF "[PanguLU Info] ADAPTIVE_KERNEL_SELECTION ------------- OFF\n"
#define PANGULU_I_SYNCHRONIZE_FREE_ON           "[PanguLU Info] SYNCHRONIZE_FREE ---------------------- ON\n"
#define PANGULU_I_SYNCHRONIZE_FREE_OFF          "[PanguLU Info] SYNCHRONIZE_FREE ---------------------- OFF\n"
#ifdef METIS
#define PANGULU_I_BASIC_INFO "[PanguLU Info] n=" FMT_PANGULU_INT64_T " nnz=" FMT_PANGULU_EXBLOCK_PTR " nb=" FMT_PANGULU_INT32_T " mpi_process=" FMT_PANGULU_INT32_T " preprocessing_thread=%d METIS:%s\n", n, origin_smatrix->rowpointer[n], nb, size, init_options->nthread, (sizeof(idx_t) == 4) ? ("i32") : ((sizeof(idx_t) == 8) ? ("i64") : ("?"))
#else
#define PANGULU_I_BASIC_INFO "[PanguLU Info] n=" FMT_PANGULU_INT64_T " nnz=" FMT_PANGULU_EXBLOCK_PTR " nb=" FMT_PANGULU_INT32_T " mpi_process=" FMT_PANGULU_INT32_T " preprocessing_thread=%d\n", n, origin_smatrix->rowpointer[n], nb, size, init_options->nthread
#endif
#define PANGULU_I_TIME_REORDER "[PanguLU Info] Reordering time is %lf s.\n", elapsed_time
#define PANGULU_I_TIME_SYMBOLIC "[PanguLU Info] Symbolic factorization time is %lf s.\n", elapsed_time
#define PANGULU_I_TIME_PRE "[PanguLU Info] Preprocessing time is %lf s.\n", elapsed_time
#define PANGULU_I_TIME_NUMERICAL "[PanguLU Info] Numeric factorization time is %lf s.\n", elapsed_time //, flop / pangulu_get_spend_time(common) / 1000000000.0
#define PANGULU_I_TIME_SPTRSV "[PanguLU Info] Solving time is %lf s.\n", elapsed_time
#define PANGULU_I_SYMBOLIC_NONZERO "[PanguLU Info] Symbolic nonzero count is " FMT_PANGULU_EXBLOCK_PTR ".\n",*symbolic_nnz
#endif // PANGULU_LOG_INFO

#endif // #ifdef PANGULU_EN