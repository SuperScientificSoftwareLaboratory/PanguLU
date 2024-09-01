#include "pangulu_common.h"

void pangulu_multiply_upper_upper_u(pangulu_block_common *block_common,
                                    pangulu_block_smatrix *block_smatrix,
                                    pangulu_vector *x, pangulu_vector *b)
{
    pangulu_int64_t block_length = block_common->block_length;
    pangulu_int64_t nb = block_common->nb;
    pangulu_block_info_pool* BIP = block_smatrix->BIP;
    pangulu_smatrix *big_smatrix_value = block_smatrix->big_pangulu_smatrix_value;
    pangulu_smatrix **diagonal_U = block_smatrix->diagonal_smatrix_u;
    pangulu_int64_t *mapper_diagonal = block_smatrix->mapper_diagonal;
    if(block_smatrix->current_rank_block_count == 0){
        return;
    }
    for (pangulu_int64_t row = 0; row < block_length; row++)
    {
        pangulu_int64_t row_offset = row * nb;
        for (pangulu_int64_t col = row; col < block_length; col++)
        {
            pangulu_int64_t mapper_index = pangulu_bip_get(row * block_length + col, BIP)->mapper_a;
            pangulu_int64_t col_offset = col * nb;
            if (row == col)
            {
                pangulu_int64_t diagonal_index = mapper_diagonal[row];
                pangulu_pangulu_smatrix_multiply_block_pangulu_vector_csc(diagonal_U[diagonal_index],
                                                                            x->value + col_offset,
                                                                            b->value + row_offset);
                if (rank == -1)
                {
                    pangulu_display_pangulu_smatrix_csc(diagonal_U[diagonal_index]);
                }
            }
            else
            {
                pangulu_pangulu_smatrix_multiply_block_pangulu_vector_csc(&big_smatrix_value[mapper_index],
                                                                            x->value + col_offset,
                                                                            b->value + row_offset);
                
            }
        }
    }
}

void pangulu_multiply_triggle_l(pangulu_block_common *block_common,
                                pangulu_block_smatrix *block_smatrix,
                                pangulu_vector *x, pangulu_vector *b)
{
    pangulu_int64_t block_length = block_common->block_length;
    pangulu_int64_t nb = block_common->nb;
    pangulu_block_info_pool* BIP = block_smatrix->BIP;
    pangulu_smatrix *big_smatrix_value = block_smatrix->big_pangulu_smatrix_value;
    pangulu_smatrix **diagonal_L = block_smatrix->diagonal_smatrix_l;
    pangulu_int64_t *mapper_diagonal = block_smatrix->mapper_diagonal;
    if(block_smatrix->current_rank_block_count == 0){
        return;
    }
    for (pangulu_int64_t row = 0; row < block_length; row++)
    {
        pangulu_int64_t row_offset = row * nb;
        for (pangulu_int64_t col = 0; col <= row; col++)
        {
            pangulu_int64_t mapper_index = pangulu_bip_get(row * block_length + col, BIP)->mapper_a;
            pangulu_int64_t col_offset = col * nb;
            if (row == col)
            {
                pangulu_int64_t diagonal_index = mapper_diagonal[col];
                pangulu_pangulu_smatrix_multiply_block_pangulu_vector_csc(diagonal_L[diagonal_index],
                                                                            x->value + col_offset,
                                                                            b->value + row_offset);
            }
            else
            {
                pangulu_pangulu_smatrix_multiply_block_pangulu_vector_csc(&big_smatrix_value[mapper_index],
                                                                            x->value + col_offset,
                                                                            b->value + row_offset);
            }
        }
    }
}

void pangulu_gather_pangulu_vector_to_rank_0(pangulu_int64_t rank,
                                             pangulu_vector *gather_v,
                                             pangulu_int64_t vector_length,
                                             pangulu_int64_t sum_rank_size)
{
    if (rank == 0)
    {
        pangulu_vector *save_vector = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));
        pangulu_init_pangulu_vector(save_vector, vector_length);

        for (pangulu_int64_t i = 1; i < sum_rank_size; i++)
        {
            pangulu_recv_pangulu_vector_value(save_vector, i, i, vector_length);
            for (pangulu_int64_t j = 0; j < vector_length; j++)
            {
                gather_v->value[j] += save_vector->value[j];
            }
        }
        for (pangulu_int64_t i = 1; i < sum_rank_size; i++)
        {
            pangulu_send_pangulu_vector_value(gather_v, i, i, vector_length);
        }
        pangulu_free(__FILE__, __LINE__, save_vector->value);
        pangulu_free(__FILE__, __LINE__, save_vector);
    }
    else
    {
        pangulu_send_pangulu_vector_value(gather_v, 0, rank, vector_length);
        pangulu_recv_pangulu_vector_value(gather_v, 0, rank, vector_length);
    }
}

calculate_type vec2norm(const calculate_type *x, pangulu_int64_t n)
{
    calculate_type sum = 0.0;
    for (pangulu_int64_t i = 0; i < n; i++)
        sum += x[i] * x[i];
    return sqrt(sum);
}

calculate_type sub_vec2norm(const calculate_type *x1, const calculate_type *x2, pangulu_int64_t n)
{
    calculate_type sum = 0.0;
    for (pangulu_int64_t i = 0; i < n; i++)
        sum += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    return sqrt(sum);
}

void pangulu_check_answer_vec2norm(pangulu_vector *X1, pangulu_vector *X2, pangulu_int64_t n)
{
    calculate_type vec2 = vec2norm(X1->value, n);
    double error = sub_vec2norm(X1->value, X2->value, n) / vec2;

    printf(PANGULU_I_VECT2NORM_ERR);
    if (fabs(error) < 1e-10)
    {
        printf(PANGULU_I_CHECK_PASS);
    }
    else
    {
        printf(PANGULU_I_CHECK_ERROR);
    }
}

void pangulu_check(pangulu_block_common *block_common,
                   pangulu_block_smatrix *block_smatrix,
                   pangulu_origin_smatrix *origin_smatrix)
{
    pangulu_exblock_idx n = block_common->n;
    pangulu_inblock_idx nb = block_common->nb;
    pangulu_exblock_idx vector_length = ((n + nb - 1) / nb) * nb;
    pangulu_int32_t sum_rank_size = block_common->sum_rank_size;

    pangulu_vector *x = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));
    pangulu_get_init_value_pangulu_vector(x, vector_length);
    pangulu_vector *b1 = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));
    pangulu_init_pangulu_vector(b1, vector_length);

    if (rank == 0)
    {
        pangulu_origin_smatrix_multiple_pangulu_vector_csr(origin_smatrix, x, b1);
    }

    pangulu_vector *b2 = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));
    pangulu_get_init_value_pangulu_vector(b2, vector_length);

    pangulu_vector *b3 = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));
    pangulu_init_pangulu_vector(b3, vector_length);

    pangulu_vector *b4 = (pangulu_vector *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_vector));
    pangulu_init_pangulu_vector(b4, vector_length);
    pangulu_multiply_upper_upper_u(block_common, block_smatrix, b2, b3);
    pangulu_gather_pangulu_vector_to_rank_0(rank, b3, vector_length, sum_rank_size);
    pangulu_multiply_triggle_l(block_common, block_smatrix, b3, b4);
    pangulu_gather_pangulu_vector_to_rank_0(rank, b4, vector_length, sum_rank_size);
    if (rank == 0)
    {
        // pangulu_check_answer(b1, b4, n);
        pangulu_check_answer_vec2norm(b1, b4, n);
    }

    pangulu_destroy_pangulu_vector(x);
    pangulu_destroy_pangulu_vector(b1);
    pangulu_destroy_pangulu_vector(b2);
    pangulu_destroy_pangulu_vector(b3);
    pangulu_destroy_pangulu_vector(b4);
}

long double max_check_ld(long double* x, int n)
{
    long double max = __DBL_MIN__;
    for (int i = 0; i < n; i++) {
        long double x_fabs = fabsl(x[i]);
        max                = max > x_fabs ? max : x_fabs;
    }
    return max;
}


// Multiply a csr matrix with a vector x, and get the resulting vector y ,sum use kekan
// sum
void spmv_ld(int n, const pangulu_int64_t* row_ptr, const pangulu_int32_t* col_idx, const long double* val, const long double* x, long double* y)
{
    for (int i = 0; i < n; i++) {
        y[i]          = 0.0;
        long double c = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            long double num = val[j] * x[col_idx[j]];
            long double z   = num - c;
            long double t   = y[i] + z;
            c               = (t - y[i]) - z;
            y[i]            = t;
        }
    }
}
