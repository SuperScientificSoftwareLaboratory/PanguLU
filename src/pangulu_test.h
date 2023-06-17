#ifndef PANGULU_TEST_H
#define PANGULU_TEST_H

#include "pangulu_common.h"
#include "pangulu_utils.h"
#include "pangulu_preprocess.h"

#include "pangulu_numerical.h"

#include "pangulu_check.h"
#include "pangulu_destroy.h"
#include "pangulu_sptrsv.h"

void pangulu_test(pangulu_common *common,
                  pangulu_Smatrix *origin_Smatrix)
{
        int_32t rank = common->rank;
        int_32t P = common->P;
        int_32t Q = common->Q;
        int_32t NB = common->NB;
        MPI_Barrier(MPI_COMM_WORLD);
        pangulu_time_start(common);
        common->N = pangulu_Bcast_N(origin_Smatrix->row, 0);
        int_t N = common->N;
        if (rank == 0)
        {
#ifdef ADAPTIVE_KERNEL_SELECTION
                printf("ADAPTIVE_KERNEL_SELECTION -------------ON\n");
#else
                printf("ADAPTIVE_KERNEL_SELECTION ------------OFF\n");
#endif
#ifdef SYNCHRONIZE_FREE
                printf("SYNCHRONIZE_FREE ----------------------ON\n");
#else
                printf("SYNCHRONIZE_FREE ---------------------OFF\n");
#endif
                printf("N is %ld ,NNZ is %ld\n", N, origin_Smatrix->rowpointer[N]);
                OMP_THREAD = common->omp_thread;
        }

#ifdef GPU_OPEN
        pangulu_cuda_device_init(rank);
#endif

        pangulu_block_Smatrix *block_Smatrix = (pangulu_block_Smatrix *)pangulu_malloc(sizeof(pangulu_block_Smatrix));
        pangulu_init_pangulu_block_Smatrix(block_Smatrix);
        pangulu_block_common *block_common = (pangulu_block_common *)pangulu_malloc(sizeof(pangulu_block_common));
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
        block_common->every_level_length = 20;
#else
        block_common->every_level_length = 1;
#endif

        pangulu_Smatrix *reorder_matrix = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
        pangulu_init_pangulu_Smatrix(reorder_matrix);

        block_common->rank_row_length = (block_common->block_length / P + (((block_common->block_length % P) > (rank / Q)) ? 1 : 0));
        block_common->rank_col_length = (block_common->block_length / Q + (((block_common->block_length % Q) > (rank % Q)) ? 1 : 0));
        block_common->every_level_length = PANGULU_MIN(block_common->every_level_length, block_common->block_length);

        MPI_Barrier(MPI_COMM_WORLD);
        pangulu_time_start(common);

        if (rank == 0)
        {
                pangulu_reorder(block_Smatrix,
                                origin_Smatrix,
                                reorder_matrix);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        pangulu_time_stop(common);

        if (rank == 0)
        {
                printf("PanguLU the reorder time is %lf s\n", pangulu_get_spend_time(common));
        }

        int_t vector_length = N;
        pangulu_vector *X_vector = (pangulu_vector *)pangulu_malloc(sizeof(pangulu_vector));
        pangulu_vector *B_vector = (pangulu_vector *)pangulu_malloc(sizeof(pangulu_vector));
        pangulu_vector *answer_vector = (pangulu_vector *)pangulu_malloc(sizeof(pangulu_vector));

        if (rank == 0)
        {
                X_vector = (pangulu_vector *)pangulu_malloc(sizeof(pangulu_vector));
                B_vector = (pangulu_vector *)pangulu_malloc(sizeof(pangulu_vector));
                answer_vector = (pangulu_vector *)pangulu_malloc(sizeof(pangulu_vector));

                // read B vector
                char *B_vector_name = NULL;

                pangulu_read_pangulu_vector(B_vector, vector_length, B_vector_name);
                pangulu_init_pangulu_vector(X_vector, vector_length);
                pangulu_init_pangulu_vector(answer_vector, vector_length);
                pangulu_reorder_vector_B_tran(block_Smatrix, B_vector, answer_vector);

                // pangulu_display_pangulu_Smatrix(reorder_matrix);
                // pangulu_display_pangulu_vector(answer_vector);
                // pangulu_display_pangulu_vector(B_vector);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        pangulu_time_start(common);

        if (rank == 0)
        {
                pangulu_symbolic(block_Smatrix,
                                 reorder_matrix);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        pangulu_time_stop(common);
        if (rank == 0)
        {
                printf("PanguLU the symbolic time is %lf s\n", pangulu_get_spend_time(common));
        }

        pangulu_init_heap_select(0);

        MPI_Barrier(MPI_COMM_WORLD);
        pangulu_time_start(common);

        pangulu_preprocess(block_common,
                           block_Smatrix,
                           reorder_matrix);

#ifdef PANGULU_SPTRSV
        pangulu_sptrsv_preprocess(block_common,
                                  block_Smatrix,
                                  answer_vector);
#endif

        MPI_Barrier(MPI_COMM_WORLD);
        pangulu_time_stop(common);
        if (rank == 0)
        {
                printf("PanguLU the preprocess time is %lf s\n", pangulu_get_spend_time(common));
        }

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

        pangulu_numerical(block_common,
                          block_Smatrix);

        MPI_Barrier(MPI_COMM_WORLD);
        pangulu_time_stop(common);
        if (rank == 0)
        {
                printf("PanguLU the numerical time is %lf s %lf GFLOPs\n", pangulu_get_spend_time(common), FLOP / pangulu_get_spend_time(common) / 1000000000.0);
        }

#ifdef PANGULU_SPTRSV

        MPI_Barrier(MPI_COMM_WORLD);
        pangulu_time_start(common);

        pangulu_sptrsv_L(block_common, block_Smatrix);
        pangulu_init_heap_select(4);
        pangulu_sptrsv_U(block_common, block_Smatrix);

        MPI_Barrier(MPI_COMM_WORLD);
        pangulu_time_stop(common);

        if (rank == 0)
        {
                printf("PanguLU the sptrsv time is %lf s\n", pangulu_get_spend_time(common));
        }

#endif

#ifdef PANGULU_SPTRSV

        // check sptrsv answer
        pangulu_sptrsv_vector_gather(block_common, block_Smatrix, answer_vector);

        if (rank == 0)
        {
                pangulu_reorder_vector_X_tran(block_Smatrix, answer_vector, X_vector);
                pangulu_pangulu_Smatrix_multiple_pangulu_vector_CSR(origin_Smatrix, X_vector, answer_vector);
                pangulu_check_answer_vec2norm(B_vector, answer_vector, N);
                // pangulu_display_pangulu_vector(X_vector);
                // pangulu_display_pangulu_vector(answer_vector);
        }
        MPI_Barrier(MPI_COMM_WORLD);

#endif
#ifdef CHECK_LU

        // check numeric answer
        pangulu_check(block_common, block_Smatrix, reorder_matrix);
        pangulu_destroy_part_pangulu_Smatrix(reorder_matrix);
        MPI_Barrier(MPI_COMM_WORLD);

#endif
        reorder_matrix = NULL;
#ifdef CHECK_TIME
        pangulu_time_simple_output(rank);
        pangulu_time_output(rank);
#endif
        pangulu_destroy(block_common, block_Smatrix);
        if (rank == 0)
        {
                printf("pangulu_test----------------------------- finish\n");
        }
        free(block_Smatrix);
        block_Smatrix = NULL;
        return;
}

#endif
