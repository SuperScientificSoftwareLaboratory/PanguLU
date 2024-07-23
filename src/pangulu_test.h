#ifndef PANGULU_TEST_H
#define PANGULU_TEST_H

#include "pangulu_common.h"
#include "pangulu_utils.h"
#include "pangulu_preprocessing.h"

#include "pangulu_numeric.h"


#include "pangulu_check.h"
#include "pangulu_destroy.h"
#include "pangulu_sptrsv.h"

void pangulu_init(int pangulu_n, long long pangulu_nnz, long *csr_rowptr, int *csr_colidx, calculate_type *csr_value, pangulu_init_options *init_options, void **pangulu_handle)
{
    

    int_32t rank;
    int_32t size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    pangulu_common *common = (pangulu_common *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_common));
    common->rank = rank;
    common->size = size;
    common->N = pangulu_n;
    common->NB = init_options->nb;
    common->sum_rank_size = size;
    common->omp_thread = init_options->nthread;
    RANK = rank;
    MPI_Bcast(&common->N, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&common->NB, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(RANK==0){
        if(init_options==NULL){
            printf(PANGULUSTR_W_OPTION_IS_NULLPTR);
        }
        if(init_options->nb==0){
            printf(PANGULUSTR_E_NB_IS_ZERO);
            return;
        }
    }
    

    // pangulu_get_common(common, init_options, size);
    int_t tmp_p = sqrt(common->sum_rank_size);
    while (((common->sum_rank_size) % tmp_p) != 0)
    {
        tmp_p--;
    }

    common->P = tmp_p;
    common->Q = common->sum_rank_size / tmp_p;
    pangulu_origin_Smatrix *origin_Smatrix = (pangulu_origin_Smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_origin_Smatrix));
    pangulu_init_pangulu_origin_Smatrix(origin_Smatrix);

    if (rank == 0)
    {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        pangulu_read_pangulu_origin_Smatrix(origin_Smatrix, pangulu_n, pangulu_nnz, csr_rowptr, csr_colidx, csr_value);
        gettimeofday(&end, NULL);
        if (origin_Smatrix->row == 0)
        {
            printf(PANGULU_E_ROW_IS_ZERO);
            return;
        }
    }

    // int_32t rank = common->rank;
    int_32t P = common->P;
    int_32t Q = common->Q;
    int_32t NB = common->NB;
    MPI_Barrier(MPI_COMM_WORLD);
    pangulu_time_start(common);
    common->N = pangulu_Bcast_N(origin_Smatrix->row, 0);
    int_t N = common->N;
    omp_set_num_threads(init_options->nthread);
    openblas_set_num_threads(1);
    if (rank == 0)
    {
#ifdef ADAPTIVE_KERNEL_SELECTION
        printf(PANGULU_I_ADAPTIVE_KERNEL_SELECTION_ON);
#else
        printf(PANGULU_I_ADAPTIVE_KERNEL_SELECTION_OFF);
#endif
#ifdef SYNCHRONIZE_FREE
        printf(PANGULU_I_SYNCHRONIZE_FREE_ON);
#else
        printf(PANGULU_I_SYNCHRONIZE_FREE_OFF);
#endif
        OMP_THREAD = PANGU_OMP_NUM_THREADS;
        int openblas_nthreads = openblas_get_num_threads();
        int omp_nthreads = omp_get_max_threads();
        printf(PANGULU_I_BASIC_INFO);
    }


#ifdef GPU_OPEN
    pangulu_cuda_device_init(rank);
#endif

    pangulu_block_Smatrix *block_Smatrix = (pangulu_block_Smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_block_Smatrix));
    pangulu_init_pangulu_block_Smatrix(block_Smatrix);
    pangulu_block_common *block_common = (pangulu_block_common *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_block_common));
    block_common->rank = rank;
    block_common->P = P;
    block_common->Q = Q;
    block_common->NB = NB;
    block_common->N = N;
    block_common->block_length = pangulu_Calculate_Block(N, NB);
    block_common->sum_rank_size = common->sum_rank_size;
    block_common->max_PQ = PANGULU_MAX(P, Q);
    block_common->every_level_length = block_common->block_length;


#ifdef SYNCHRONIZE_FREE
    block_common->every_level_length = 200;
#else
    block_common->every_level_length = 1;
#endif

    pangulu_origin_Smatrix *reorder_matrix = (pangulu_origin_Smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_origin_Smatrix));
    pangulu_init_pangulu_origin_Smatrix(reorder_matrix);


    block_common->rank_row_length = (block_common->block_length / P + (((block_common->block_length % P) > (rank / Q)) ? 1 : 0));
    block_common->rank_col_length = (block_common->block_length / Q + (((block_common->block_length % Q) > (rank % Q)) ? 1 : 0));
    block_common->every_level_length = PANGULU_MIN(block_common->every_level_length, block_common->block_length);
    MPI_Barrier(MPI_COMM_WORLD);
    pangulu_time_start(common);

    pangulu_reorder(block_Smatrix,
                    origin_Smatrix,
                    reorder_matrix);

    MPI_Barrier(MPI_COMM_WORLD);
    pangulu_time_stop(common);
    if (rank == 0)
    {
        printf(PANGULU_I_TIME_REORDER);
    }

    calculate_time = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    pangulu_time_start(common);

    if (rank == 0)
    {
        pangulu_symbolic(block_common,
                         block_Smatrix,
                         reorder_matrix);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    pangulu_time_stop(common);
    int_t symbolic_csr_need_memory = 0;
    if (rank == 0)
    {
        printf(PANGULU_I_TIME_SYMBOLIC);
        symbolic_csr_need_memory = sizeof(int_t) * (N + 1) + (sizeof(calculate_type) + sizeof(idx_int)) * (block_Smatrix->symbolic_nnz);
    }

    pangulu_init_heap_select(0);

    MPI_Barrier(MPI_COMM_WORLD);
    pangulu_time_start(common);

    pangulu_preprocessing(
        block_common,
        block_Smatrix,
        reorder_matrix,
        init_options->nthread
    );

#ifdef PANGULU_SPTRSV
    
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    pangulu_time_stop(common);
    if (RANK == 0)
    {
        printf(PANGULU_I_TIME_PRE);
    }

    (*pangulu_handle) = malloc(sizeof(pangulu_handle_t));
    (*(pangulu_handle_t**)pangulu_handle)->block_common = block_common;
    (*(pangulu_handle_t**)pangulu_handle)->block_Smatrix = block_Smatrix;
    (*(pangulu_handle_t**)pangulu_handle)->commmon = common;
    
}

void pangulu_gstrf(pangulu_gstrf_options *gstrf_options, void **pangulu_handle)
{
    pangulu_block_common* block_common = (*(pangulu_handle_t**)pangulu_handle)->block_common;
    pangulu_block_Smatrix* block_Smatrix = (*(pangulu_handle_t**)pangulu_handle)->block_Smatrix;
    pangulu_common* common = (*(pangulu_handle_t**)pangulu_handle)->commmon;

    #ifdef CHECK_TIME
    pangulu_time_init();
#endif
    MPI_Barrier(MPI_COMM_WORLD);

#ifdef OVERLAP
    pangulu_create_pthread(block_common,
                           block_Smatrix);
#endif

    pangulu_time_init();
    MPI_Barrier(MPI_COMM_WORLD);
    pangulu_time_start(common);

    pangulu_numeric(block_common,
                      block_Smatrix);

    MPI_Barrier(MPI_COMM_WORLD);
    pangulu_time_stop(common);

    if (RANK == 0)
    {

        int_t another_calculate_time = 0;
        for (int_t i = 1; i < block_common->sum_rank_size; i++)
        {
            pangulu_recv_vector_int(&another_calculate_time, 1, i, 0);
            calculate_time += another_calculate_time;
        }
        FLOP = calculate_time * 2;
    }
    else
    {
        pangulu_send_vector_int(&calculate_time, 1, 0, 0);
    }

    if (RANK == 0)
    {
        printf(PANGULU_I_TIME_NUMERICAL);
    }
}

void pangulu_gstrs(calculate_type *rhs, pangulu_gstrs_options *gstrs_options, void** pangulu_handle)
{
    pangulu_block_common* block_common = (*(pangulu_handle_t**)pangulu_handle)->block_common;
    pangulu_block_Smatrix* block_Smatrix = (*(pangulu_handle_t**)pangulu_handle)->block_Smatrix;
    pangulu_common* common = (*(pangulu_handle_t**)pangulu_handle)->commmon;

    int_t vector_length = common->N;
    pangulu_vector *X_vector = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));
    pangulu_vector *B_vector = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));
    pangulu_vector *answer_vector = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));

    if (RANK == 0)
    {
        X_vector = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));
        B_vector = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));
        answer_vector = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));

        // read B vector
        // char *B_vector_name = NULL;s

        // pangulu_read_pangulu_vector(B_vector, vector_length, common->rhs_name);
        B_vector->row = common->N;
        B_vector->value = rhs;
        pangulu_init_pangulu_vector(X_vector, vector_length);
        pangulu_init_pangulu_vector(answer_vector, vector_length);
        pangulu_reorder_vector_B_tran(block_Smatrix, B_vector, answer_vector);

        // pangulu_display_pangulu_Smatrix(reorder_matrix);
        // pangulu_display_pangulu_vector(answer_vector);
        // pangulu_display_pangulu_vector(B_vector);
    }

    pangulu_sptrsv_preprocessing(
        block_common,
        block_Smatrix,
        answer_vector
    );

    #ifdef PANGULU_SPTRSV

    MPI_Barrier(MPI_COMM_WORLD);
    pangulu_time_start(common);

    pangulu_sptrsv_L(block_common, block_Smatrix);
    pangulu_init_heap_select(4);
    pangulu_sptrsv_U(block_common, block_Smatrix);

    MPI_Barrier(MPI_COMM_WORLD);
    pangulu_time_stop(common);

    if (RANK == 0)
    {
        printf(PANGULU_I_TIME_SPTRSV);
    }

#endif

    // check sptrsv answer
    pangulu_sptrsv_vector_gather(block_common, block_Smatrix, answer_vector);

    int N = common->N;

    if (RANK == 0)
    {
        pangulu_reorder_vector_X_tran(block_Smatrix, answer_vector, X_vector);

        for(int i = 0; i < N; i ++){
            rhs[i] = X_vector->value[i];
        }

        // pangulu_refinement_hp *X_hp_reorder_back = (pangulu_refinement_hp *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_refinement_hp) * N);
        // pangulu_refinement_hp *B_hp = (pangulu_refinement_hp *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_refinement_hp) * N);
        // pangulu_refinement_hp *origin_Smatrix_value_hp = (pangulu_refinement_hp *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_refinement_hp) * origin_Smatrix->nnz);
        // for (int_t i = 0; i < origin_Smatrix->nnz; i++)
        // {
        //     origin_Smatrix_value_hp[i] = origin_Smatrix->value[i];
        // }
        // for (int_t i = 0; i < N; i++)
        // {
        //     X_hp_reorder_back[i] = X_vector->value[i];
        // }
        // for (int i = 0; i < N; i++)
        // {
        //     B_hp[i] = B_vector->value[i];
        // }
        // // pangulu_origin_Smatrix_multiple_pangulu_vector_CSR(origin_Smatrix, X_vector, answer_vector);
        // check_correctness_ld(block_common->N, origin_Smatrix->rowpointer, origin_Smatrix->columnindex, origin_Smatrix_value_hp, X_hp_reorder_back, B_hp);

        // // pangulu_origin_Smatrix_multiple_pangulu_vector_CSR(origin_Smatrix, X_vector, answer_vector);
        // // pangulu_check_answer_vec2norm(B_vector, answer_vector, N);
        // // pangulu_diff_pangulu_vector(block_Smatrix, answer_vector, B_vector);
        // // pangulu_display_pangulu_vector(answer_vector);
        // // pangulu_display_pangulu_vector(X_vector);
    }
}

void pangulu_gssv(calculate_type *rhs, pangulu_gstrf_options *gstrf_options, pangulu_gstrs_options *gstrs_options, void **pangulu_handle)
{
    pangulu_gstrf(gstrf_options, pangulu_handle);
    pangulu_gstrs(rhs, gstrs_options, pangulu_handle);
}

#endif
