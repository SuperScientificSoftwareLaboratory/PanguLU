#ifndef PANGULU_CHECK_H
#define PANGULU_CHECK_H

#include "pangulu_common.h"
#include "pangulu_utils.h"
#include "pangulu_destroy.h"

void pangulu_multiply_upper_upper_U(pangulu_block_common *block_common,
                                    pangulu_block_Smatrix *block_Smatrix,
                                    pangulu_vector *X, pangulu_vector *B)
{
    int_t block_length = block_common->block_length;
    int_t NB = block_common->NB;
    int_t *mapper_A = block_Smatrix->mapper_Big_pangulu_Smatrix;
    pangulu_Smatrix **Big_Smatrix_value = block_Smatrix->Big_pangulu_Smatrix_value;
    pangulu_Smatrix **diagonal_U = block_Smatrix->diagonal_Smatrix_U;
    int_t *mapper_diagonal = block_Smatrix->mapper_diagonal;
    int_t *real_matrix_flag = block_Smatrix->real_matrix_flag;
    for (int_t row = 0; row < block_length; row++)
    {
        int_t row_offset = row * NB;
        for (int_t col = row; col < block_length; col++)
        {
            int_t mapper_index = mapper_A[row * block_length + col];
            int_t real_flag = real_matrix_flag[mapper_index];
            if (real_flag == 1)
            {
                int_t col_offset = col * NB;
                if (row == col)
                {
                    int_t diagonal_index = mapper_diagonal[row];
                    pangulu_pangulu_Smatrix_multiply_block_pangulu_vector_CSC(diagonal_U[diagonal_index],
                                                                              X->value + col_offset,
                                                                              B->value + row_offset);
                    if (RANK == -1)
                    {
                        printf("\nrow %ld col %ld\n", row, col);
                        pangulu_display_pangulu_Smatrix_CSC(diagonal_U[diagonal_index]);
                    }
                }
                else
                {
                    pangulu_pangulu_Smatrix_multiply_block_pangulu_vector_CSC(Big_Smatrix_value[mapper_index],
                                                                              X->value + col_offset,
                                                                              B->value + row_offset);
                    if (RANK == -1)
                    {
                        printf("\nrow %ld col %ld\n", row, col);
                        pangulu_display_pangulu_Smatrix_CSC(Big_Smatrix_value[mapper_index]);
                    }
                }
            }
        }
    }
}

void pangulu_multiply_triggle_L(pangulu_block_common *block_common,
                                pangulu_block_Smatrix *block_Smatrix,
                                pangulu_vector *X, pangulu_vector *B)
{
    int_t block_length = block_common->block_length;
    int_t NB = block_common->NB;
    int_t *mapper_A = block_Smatrix->mapper_Big_pangulu_Smatrix;
    pangulu_Smatrix **Big_Smatrix_value = block_Smatrix->Big_pangulu_Smatrix_value;
    pangulu_Smatrix **diagonal_L = block_Smatrix->diagonal_Smatrix_L;
    int_t *mapper_diagonal = block_Smatrix->mapper_diagonal;
    int_t *real_matrix_flag = block_Smatrix->real_matrix_flag;
    for (int_t row = 0; row < block_length; row++)
    {
        int_t row_offset = row * NB;
        for (int_t col = 0; col <= row; col++)
        {
            int_t mapper_index = mapper_A[row * block_length + col];
            int_t real_flag = real_matrix_flag[mapper_index];
            if (real_flag == 1)
            {
                int_t col_offset = col * NB;
                if (row == col)
                {
                    int_t diagonal_index = mapper_diagonal[col];
                    pangulu_pangulu_Smatrix_multiply_block_pangulu_vector_CSC(diagonal_L[diagonal_index],
                                                                              X->value + col_offset,
                                                                              B->value + row_offset);
                }
                else
                {
                    pangulu_pangulu_Smatrix_multiply_block_pangulu_vector_CSC(Big_Smatrix_value[mapper_index],
                                                                              X->value + col_offset,
                                                                              B->value + row_offset);
                }
            }
        }
    }
}

void pangulu_gather_pangulu_vector_to_rank_0(int_t rank,
                                             pangulu_vector *gather_v,
                                             int_t vector_length,
                                             int_t sum_rank_size)
{
    if (rank == 0)
    {
        pangulu_vector *save_vector = (pangulu_vector *)pangulu_malloc(sizeof(pangulu_vector));
        pangulu_init_pangulu_vector(save_vector, vector_length);

        for (int_t i = 1; i < sum_rank_size; i++)
        {
            pangulu_recv_pangulu_vector_value(save_vector, i, i, vector_length);
            for (int_t i = 0; i < vector_length; i++)
            {
                gather_v->value[i] += save_vector->value[i];
            }
        }
        for (int_t i = 1; i < sum_rank_size; i++)
        {
            pangulu_send_pangulu_vector_value(gather_v, i, i, vector_length);
        }
        free(save_vector->value);
        free(save_vector);
    }
    else
    {
        pangulu_send_pangulu_vector_value(gather_v, 0, rank, vector_length);
        pangulu_recv_pangulu_vector_value(gather_v, 0, rank, vector_length);
    }
}

void pangulu_check_answer(pangulu_vector *X1, pangulu_vector *X2, int_t n)
{
    double error = 0.0;
    for (int_t i = 0; i < n; i++)
    {
        error += (double)fabs(X1->value[i] - X2->value[i]);
    }
    // printf("the error is %.15f\n", error);
}

calculate_type vec2norm(calculate_type *x, int_t n)
{
    double sum = 0.0;
    for (int_t i = 0; i < n; i++)
        sum += x[i] * x[i];
    return sqrt(sum);
}

calculate_type sub_vec2norm(calculate_type *x1, calculate_type *x2, int_t n)
{
    double sum = 0.0;
    for (int_t i = 0; i < n; i++)
        sum += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    return sqrt(sum);
}

void pangulu_check_answer_vec2norm(pangulu_vector *X1, pangulu_vector *X2, int_t n)
{
    calculate_type vec2 = vec2norm(X1->value, n);
    calculate_type error = sub_vec2norm(X1->value, X2->value, n) / vec2;

    printf("the vec2norm |Ax - B | / AX -------error is %12.4e \n", error);
    if (error < 1e-10)
    {
        printf("Check ------------------------------------- pass\n");
    }
    else
    {
        printf("Check ------------------------------------ error\n");
    }
}

void pangulu_check(pangulu_block_common *block_common,
                   pangulu_block_Smatrix *block_Smatrix,
                   pangulu_Smatrix *origin_Smatrix)
{
    int_32t rank = block_common->rank;
    int_t N = block_common->N;
    int_32t NB = block_common->NB;
    int_t vector_length = ((N + NB - 1) / NB) * NB;
    int_32t sum_rank_size = block_common->sum_rank_size;

    pangulu_vector *X = (pangulu_vector *)pangulu_malloc(sizeof(pangulu_vector));
    pangulu_get_init_value_pangulu_vector(X, vector_length);
    pangulu_vector *b1 = (pangulu_vector *)pangulu_malloc(sizeof(pangulu_vector));
    pangulu_init_pangulu_vector(b1, vector_length);

    if (rank == 0)
    {
        pangulu_pangulu_Smatrix_multiple_pangulu_vector_CSR(origin_Smatrix, X, b1);
    }

    pangulu_vector *b2 = (pangulu_vector *)pangulu_malloc(sizeof(pangulu_vector));
    pangulu_get_init_value_pangulu_vector(b2, vector_length);

    pangulu_vector *b3 = (pangulu_vector *)pangulu_malloc(sizeof(pangulu_vector));
    pangulu_init_pangulu_vector(b3, vector_length);

    pangulu_vector *b4 = (pangulu_vector *)pangulu_malloc(sizeof(pangulu_vector));
    pangulu_init_pangulu_vector(b4, vector_length);
    pangulu_multiply_upper_upper_U(block_common, block_Smatrix, b2, b3);
    pangulu_gather_pangulu_vector_to_rank_0(rank, b3, vector_length, sum_rank_size);
    //pangulu_display_pangulu_vector(b3);
    pangulu_multiply_triggle_L(block_common, block_Smatrix, b3, b4);
    pangulu_gather_pangulu_vector_to_rank_0(rank, b4, vector_length, sum_rank_size);
    if (rank == 0)
    {
        //pangulu_display_pangulu_vector(b1);
        //pangulu_display_pangulu_vector(b4);
        pangulu_check_answer(b1, b4, N);
        pangulu_check_answer_vec2norm(b1, b4, N);
    }

    X = pangulu_destroy_pangulu_vector(X);
    b1 = pangulu_destroy_pangulu_vector(b1);
    b2 = pangulu_destroy_pangulu_vector(b2);
    b3 = pangulu_destroy_pangulu_vector(b3);
    b4 = pangulu_destroy_pangulu_vector(b4);
}

#endif