#include "pangulu_common.h"

#ifdef PANGULU_MC64
void pangulu_mc64dd(
    pangulu_int64_t col,
    pangulu_int64_t n,
    pangulu_int64_t *queue,
    const calculate_type *row_scale_value,
    pangulu_int64_t *save_tmp)
{
    pangulu_int64_t loc = save_tmp[col];
    calculate_type rsv_min = row_scale_value[col];

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        if (loc <= 0)
        {
            break;
        }
        pangulu_int64_t tmp_loc = (loc - 1) / 2;
        pangulu_int64_t index_loc = queue[tmp_loc];
        if (rsv_min >= row_scale_value[index_loc])
        {
            break;
        }
        queue[loc] = index_loc;
        save_tmp[index_loc] = loc;
        loc = tmp_loc;
    }

    queue[loc] = col;
    save_tmp[col] = loc;
    return;
}

void pangulu_mc64ed(
    pangulu_int64_t *queue_length,
    pangulu_int64_t n,
    pangulu_int64_t *queue,
    const calculate_type *row_scale_value,
    pangulu_int64_t *save_tmp)
{
    pangulu_int64_t loc = 0;
    (*queue_length)--;
    pangulu_int64_t now_queue_length = *queue_length;
    calculate_type rsv_min = row_scale_value[queue[now_queue_length]];

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        pangulu_int64_t tmp_loc = (loc + 1) * 2 - 1;
        if (tmp_loc > now_queue_length)
        {
            break;
        }
        calculate_type rsv_now = row_scale_value[queue[tmp_loc]];
        if (tmp_loc < now_queue_length)
        {
            calculate_type rsv_after = row_scale_value[queue[tmp_loc + 1]];
            if (rsv_now > rsv_after)
            {
                rsv_now = rsv_after;
                tmp_loc++;
            }
        }
        if (rsv_min <= rsv_now)
        {
            break;
        }
        queue[loc] = queue[tmp_loc];
        save_tmp[queue[loc]] = loc;
        loc = tmp_loc;
    }

    queue[loc] = queue[now_queue_length];
    save_tmp[queue[now_queue_length]] = loc;
    return;
}

void pangulu_mc64fd(
    pangulu_int64_t loc_origin,
    pangulu_int64_t *queue_length,
    pangulu_int64_t n,
    pangulu_int64_t *queue,
    const calculate_type *row_scale_value,
    pangulu_int64_t *save_tmp)
{
    (*queue_length)--;
    pangulu_int64_t now_queue_length = *queue_length;

    if (loc_origin == now_queue_length)
    {
        return;
    }

    pangulu_int64_t loc = loc_origin;
    calculate_type rsv_min = row_scale_value[queue[now_queue_length]];

    for (pangulu_int64_t i = 0; i < n; ++i)
    {
        if (loc <= 0)
        {
            break;
        }
        pangulu_int64_t tmp_loc = (loc - 1) / 2;
        pangulu_int64_t index_loc = queue[tmp_loc];
        if (rsv_min >= row_scale_value[index_loc])
        {
            break;
        }
        queue[loc] = index_loc;
        save_tmp[index_loc] = loc;
        loc = tmp_loc;
    }

    queue[loc] = queue[now_queue_length];
    save_tmp[queue[now_queue_length]] = loc;

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        pangulu_int64_t tmp_loc = (loc + 1) * 2 - 1;
        if (tmp_loc > now_queue_length)
        {
            break;
        }
        calculate_type rsv_now = row_scale_value[queue[tmp_loc]];
        if (tmp_loc < now_queue_length)
        {
            calculate_type rsv_after = row_scale_value[queue[tmp_loc + 1]];
            if (rsv_now > rsv_after)
            {
                rsv_now = rsv_after;
                tmp_loc++;
            }
        }
        if (rsv_min <= rsv_now)
        {
            break;
        }
        queue[loc] = queue[tmp_loc];
        save_tmp[queue[loc]] = loc;
        loc = tmp_loc;
    }

    queue[loc] = queue[now_queue_length];
    save_tmp[queue[now_queue_length]] = loc;
    return;
}

void pangulu_mc64(
    pangulu_origin_smatrix *s,
    pangulu_exblock_idx **perm,
    pangulu_exblock_idx **iperm,
    calculate_type **row_scale,
    calculate_type **col_scale)
{
    #define PANGULU_MC64_FLAG 4294967295

    pangulu_exblock_idx n = s->row;
    pangulu_exblock_ptr nnz = s->nnz;

    pangulu_int64_t finish_flag = 0;

    pangulu_exblock_ptr *rowptr = s->columnpointer;
    pangulu_exblock_idx *colidx = s->rowindex;
    calculate_type *val = s->value_csc;

    pangulu_exblock_idx *col_perm = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * n);
    pangulu_exblock_idx *row_perm = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * n);
    pangulu_int64_t *rowptr_tmp = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * n);
    pangulu_int64_t *queue = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * n);
    pangulu_int64_t *save_tmp = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * n);
    pangulu_int64_t *ans_queue = (pangulu_int64_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int64_t) * n);

    calculate_type *fabs_value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz);
    calculate_type *max_value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);
    calculate_type *col_scale_value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);
    calculate_type *row_scale_value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        if (rowptr[i] >= rowptr[i + 1])
        {
            printf(PANGULU_E_ROW_IS_NULL);
            exit(1);
        }
    }

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        col_perm[i] = PANGULU_MC64_FLAG;
    }

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        row_perm[i] = PANGULU_MC64_FLAG;
    }
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        col_scale_value[i] = FLT_MAX;
    }
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        row_scale_value[i] = 0.0;
    }

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        rowptr_tmp[i] = rowptr[i];
    }

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        save_tmp[i] = -1;
    }

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        if (rowptr[i] >= rowptr[i + 1])
        {
            printf(PANGULU_E_ROW_IS_NULL);
            exit(1);
        }
    }

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        max_value[i] = 0.0;
        for (pangulu_int64_t j = rowptr[i]; j < rowptr[i + 1]; j++)
        {

            fabs_value[j] = fabs(val[j]);
            max_value[i] = PANGULU_MAX(fabs_value[j], max_value[i]);
        }

        calculate_type now_row_max = max_value[i];

        if ((now_row_max != 0.0))
        {
            now_row_max = log(now_row_max);
        }
        else
        {
            printf(PANGULU_E_MAX_NULL);
            exit(1);
        }

        for (pangulu_int64_t j = rowptr[i]; j < rowptr[i + 1]; j++)
        {

            if ((fabs_value[j] != 0.0))
            {
                fabs_value[j] = now_row_max - log(fabs_value[j]);
            }
            else
            {
                fabs_value[j] = FLT_MAX / 5.0;
            }
        }
    }

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        for (pangulu_int64_t j = rowptr[i]; j < rowptr[i + 1]; j++)
        {
            pangulu_int64_t col = colidx[j];
            if (fabs_value[j] <= col_scale_value[col])
            {
                col_scale_value[col] = fabs_value[j];
                col_perm[col] = i;
                save_tmp[col] = j;
            }
        }
    }
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        if (col_perm[i] != PANGULU_MC64_FLAG)
        {
            if (row_perm[col_perm[i]] == PANGULU_MC64_FLAG)
            {
                finish_flag++;
                row_perm[col_perm[i]] = save_tmp[i];
            }
            else
            {
                col_perm[i] = PANGULU_MC64_FLAG;
            }
        }
    }

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        if (finish_flag == n)
        {
            break;
        }
        if (row_perm[i] == PANGULU_MC64_FLAG)
        {
            calculate_type col_max = FLT_MAX;
            pangulu_int64_t save_col = -1;
            pangulu_int64_t save_index = -1;
            for (pangulu_int64_t j = rowptr[i]; j < rowptr[i + 1]; j++)
            {
                pangulu_int64_t col = colidx[j];
                calculate_type now_value = fabs_value[j] - col_scale_value[col];
                if (now_value > col_max)
                {
                }
                else if ((now_value >= col_max) && (now_value != FLT_MAX))
                {
                    if ((col_perm[col] == PANGULU_MC64_FLAG) && (col_perm[save_col] != PANGULU_MC64_FLAG))
                    {
                        col_max = now_value;
                        save_col = col;
                        save_index = j;
                    }
                }
                else
                {
                    col_max = now_value;
                    save_col = col;
                    save_index = j;
                }
            }

            row_scale_value[i] = col_max;
            if (col_perm[save_col] == PANGULU_MC64_FLAG)
            {
                row_perm[i] = save_index;
                col_perm[save_col] = i;
                rowptr_tmp[i] = save_index + 1;
                finish_flag++;
            }
            else
            {
                pangulu_int64_t break_flag = 0;
                for (pangulu_int64_t j = save_index; j < rowptr[i + 1]; j++)
                {
                    pangulu_int64_t col = colidx[j];
                    if ((fabs_value[j] - col_scale_value[col]) <= col_max)
                    {
                        pangulu_int64_t now_col = col_perm[col];
                        if (rowptr_tmp[now_col] < rowptr[now_col + 1])
                        {
                            for (pangulu_int64_t k = rowptr_tmp[now_col]; k < rowptr[now_col + 1]; k++)
                            {
                                pangulu_int64_t tmp_col = colidx[k];
                                if (col_perm[tmp_col] == PANGULU_MC64_FLAG)
                                {
                                    if ((fabs_value[k] - col_scale_value[tmp_col]) <= row_scale_value[now_col])
                                    {
                                        row_perm[now_col] = k;
                                        col_perm[tmp_col] = now_col;
                                        rowptr_tmp[now_col] = k + 1;
                                        break_flag = 1;
                                        break;
                                    }
                                }
                            }
                            if (break_flag == 1)
                            {
                                row_perm[i] = j;
                                col_perm[col] = i;
                                rowptr_tmp[i] = j + 1;
                                finish_flag++;
                                break;
                            }
                            rowptr_tmp[now_col] = rowptr[now_col + 1];
                        }
                    }
                }
            }
        }
    }
    if (finish_flag != n)
    {
        for (pangulu_int64_t i = 0; i < n; i++)
        {
            row_scale_value[i] = FLT_MAX;
        }

        for (pangulu_int64_t i = 0; i < n; i++)
        {
            save_tmp[i] = -1;
        }
    }

    for (pangulu_int64_t now_row = 0; now_row < n; now_row++)
    {
        if (finish_flag == n)
        {
            break;
        }
        if (row_perm[now_row] == PANGULU_MC64_FLAG)
        {
            pangulu_int64_t row = now_row;
            pangulu_int64_t queue_length = 0;
            pangulu_int64_t low = n;
            pangulu_int64_t top = n;
            pangulu_int64_t save_index = -1;
            pangulu_int64_t save_row = -1;

            rowptr_tmp[row] = PANGULU_MC64_FLAG;
            calculate_type min_cost = FLT_MAX;
            calculate_type sum_cost = FLT_MAX;

            for (pangulu_int64_t k = rowptr[row]; k < rowptr[row + 1]; k++)
            {
                pangulu_int64_t col = colidx[k];
                calculate_type now_value = fabs_value[k] - col_scale_value[col];
                if (now_value < sum_cost)
                {
                    if (col_perm[col] == PANGULU_MC64_FLAG)
                    {
                        sum_cost = now_value;
                        save_index = k;
                        save_row = row;
                    }
                    else
                    {
                        min_cost = PANGULU_MIN(now_value, min_cost);
                        row_scale_value[col] = now_value;
                        queue[queue_length++] = k;
                    }
                }
            }

            pangulu_int64_t now_queue_length = queue_length;
            queue_length = 0;

            for (pangulu_int64_t k = 0; k < now_queue_length; k++)
            {
                pangulu_int64_t queue_index = queue[k];
                pangulu_int64_t col = colidx[queue_index];
                if (row_scale_value[col] >= sum_cost)
                {
                    row_scale_value[col] = FLT_MAX;
                }
                else
                {
                    if (row_scale_value[col] <= min_cost)
                    {
                        low--;
                        queue[low] = col;
                        save_tmp[col] = low;
                    }
                    else
                    {
                        save_tmp[col] = queue_length++;
                        pangulu_mc64dd(col, n, queue, row_scale_value, save_tmp);
                    }
                    pangulu_exblock_idx now_col = col_perm[col];
                    if(now_col != PANGULU_MC64_FLAG){
                        ans_queue[now_col] = queue_index;
                        rowptr_tmp[now_col] = row;
                    }
                }
            }

            for (pangulu_int64_t k = 0; k < finish_flag; k++)
            {
                if (low == top)
                {

                    if ((queue_length == 0) || (row_scale_value[queue[0]] >= sum_cost))
                    {
                        break;
                    }
                    pangulu_int64_t col = queue[0];
                    min_cost = row_scale_value[col];
                    do
                    {
                        pangulu_mc64ed(&queue_length, n, queue, row_scale_value, save_tmp);
                        queue[--low] = col;
                        save_tmp[col] = low;
                        if (queue_length == 0)
                        {
                            break;
                        }
                        col = queue[0];
                    } while (row_scale_value[col] <= min_cost);
                }
                pangulu_int64_t now_queue_length = queue[top - 1];
                if (row_scale_value[now_queue_length] >= sum_cost)
                {
                    break;
                }
                top--;
                row = col_perm[now_queue_length];
                if(row == PANGULU_MC64_FLAG){
                    return;
                }
                calculate_type row_sum_max = row_scale_value[now_queue_length] - fabs_value[row_perm[row]] + col_scale_value[now_queue_length];

                for (pangulu_int64_t k = rowptr[row]; k < rowptr[row + 1]; k++)
                {
                    pangulu_int64_t col = colidx[k];
                    if (save_tmp[col] < top)
                    {
                        calculate_type now_value = row_sum_max + fabs_value[k] - col_scale_value[col];
                        if (now_value < sum_cost)
                        {
                            if (col_perm[col] == PANGULU_MC64_FLAG)
                            {
                                sum_cost = now_value;
                                save_index = k;
                                save_row = row;
                            }
                            else
                            {
                                if ((row_scale_value[col] > now_value) && (save_tmp[col] < low))
                                {
                                    row_scale_value[col] = now_value;
                                    if (now_value <= min_cost)
                                    {
                                        if (save_tmp[col] >= 0)
                                        {
                                            pangulu_mc64fd(save_tmp[col], &queue_length, n, queue, row_scale_value, save_tmp);
                                        }
                                        low--;
                                        queue[low] = col;
                                        save_tmp[col] = low;
                                    }
                                    else
                                    {
                                        if (save_tmp[col] < 0)
                                        {
                                            save_tmp[col] = queue_length++;
                                        }
                                        pangulu_mc64dd(col, n, queue, row_scale_value, save_tmp);
                                    }

                                    pangulu_int64_t now_col = col_perm[col];
                                    ans_queue[now_col] = k;
                                    rowptr_tmp[now_col] = row;
                                }
                            }
                        }
                    }
                }
            }

            if (sum_cost != FLT_MAX)
            {
                finish_flag++;
                col_perm[colidx[save_index]] = save_row;
                row_perm[save_row] = save_index;
                row = save_row;

                for (pangulu_int64_t k = 0; k < finish_flag; k++)
                {
                    pangulu_int64_t now_rowptr_tmp = rowptr_tmp[row];
                    if (now_rowptr_tmp == PANGULU_MC64_FLAG)
                    {
                        break;
                    }
                    pangulu_int64_t col = colidx[ans_queue[row]];
                    if (col == PANGULU_MC64_FLAG)
                    {
                        break;
                    }
                    col_perm[col] = now_rowptr_tmp;
                    row_perm[now_rowptr_tmp] = ans_queue[row];
                    row = now_rowptr_tmp;
                }

                for (pangulu_int64_t k = top; k < n; k++)
                {
                    pangulu_int64_t col = queue[k];
                    col_scale_value[col] = col_scale_value[col] + row_scale_value[col] - sum_cost;
                }
            }

            for (pangulu_int64_t k = low; k < n; k++)
            {
                pangulu_int64_t col = queue[k];
                row_scale_value[col] = FLT_MAX;
                save_tmp[col] = -1;
            }

            for (pangulu_int64_t k = 0; k < queue_length; k++)
            {
                pangulu_int64_t col = queue[k];
                row_scale_value[col] = FLT_MAX;
                save_tmp[col] = -1;
            }
        }
    }

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        pangulu_exblock_idx now_col = row_perm[i];
        if (now_col < PANGULU_MC64_FLAG)
        {
            row_scale_value[i] = fabs_value[now_col] - col_scale_value[colidx[now_col]];
        }
        else
        {
            row_scale_value[i] = 0.0;
        }
        if (col_perm[i] == PANGULU_MC64_FLAG)
        {
            col_scale_value[i] = 0.0;
        }
    }

    if (finish_flag == n)
    {
        for (pangulu_int64_t i = 0; i < n; i++)
        {
            if(col_perm[i] == PANGULU_MC64_FLAG){
                continue;
            }
            row_perm[col_perm[i]] = i;

            calculate_type a = max_value[i];
            if (a != 0.0)
            {
                row_scale_value[i] -= log(a);
            }
            else
            {
                row_scale_value[i] = 0.0;
            }
        }
    }
    else
    {
        for (pangulu_int64_t i = 0; i < n; i++)
        {
            row_perm[i] = PANGULU_MC64_FLAG;
        }

        pangulu_int64_t ans_queue_length = 0;
        for (pangulu_int64_t i = 0; i < n; i++)
        {
            if (col_perm[i] == PANGULU_MC64_FLAG)
            {
                ans_queue[ans_queue_length++] = i;
            }
            else
            {
                row_perm[col_perm[i]] = i;
            }
        }
        ans_queue_length = 0;
        for (pangulu_int64_t i = 0; i < n; i++)
        {
            if (row_perm[i] == PANGULU_MC64_FLAG)
            {
                col_perm[ans_queue[ans_queue_length++]] = i;
            }
        }
    }

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        col_scale_value[i] = exp(col_scale_value[i]);
    }

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        row_scale_value[i] = exp(row_scale_value[i]);
    }

    pangulu_free(__FILE__, __LINE__, rowptr_tmp);
    pangulu_free(__FILE__, __LINE__, queue);
    pangulu_free(__FILE__, __LINE__, save_tmp);
    pangulu_free(__FILE__, __LINE__, ans_queue);
    pangulu_free(__FILE__, __LINE__, fabs_value);
    pangulu_free(__FILE__, __LINE__, max_value);

    *perm = row_perm;
    *iperm = col_perm;

    *col_scale = col_scale_value;
    *row_scale = row_scale_value;

    return;
#undef PANGULU_MC64_FLAG
}
#endif

void pangulu_reorder_vector_b_tran(
    pangulu_exblock_idx *row_perm,
    pangulu_exblock_idx *metis_perm,
    calculate_type *row_scale,
    pangulu_vector *B_origin,
    pangulu_vector *B_trans)
{
    pangulu_int64_t n = B_origin->row;
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        pangulu_int64_t index_row = row_perm[i];
        pangulu_int64_t now_row = metis_perm[index_row];
        B_trans->value[now_row] = B_origin->value[i] * row_scale[i];
    }
}

void pangulu_reorder_vector_x_tran(
    pangulu_block_smatrix *block_smatrix,
    pangulu_vector *X_origin,
    pangulu_vector *X_trans)
{
    pangulu_int64_t n = X_origin->row;
    pangulu_exblock_idx *metis_perm = block_smatrix->metis_perm;
    calculate_type *col_scale = block_smatrix->col_scale;

    for (Hunyuan_int_t i = 0; i < n; i++)
    {
        pangulu_int64_t now_col = metis_perm[i];
        X_trans->value[i] = X_origin->value[now_col] * col_scale[i];
    }
}

void pangulu_add_diagonal_element_csc(pangulu_origin_smatrix *s)
{
    calculate_type ZERO_ELEMENT = 1e-8;
    pangulu_int64_t diagonal_add = 0;
    pangulu_exblock_idx n = s->row;
    pangulu_exblock_ptr *new_columnpointer = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 5));
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        char flag = 0;
        for (pangulu_exblock_ptr j = s->columnpointer[i]; j < s->columnpointer[i + 1]; j++)
        {
            if (s->rowindex[j] == i)
            {
                flag = 1;
                break;
            }
        }
        new_columnpointer[i] = s->columnpointer[i] + diagonal_add;
        diagonal_add += (!flag);
    }
    new_columnpointer[n] = s->columnpointer[n] + diagonal_add;

    pangulu_exblock_idx *new_rowindex = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * new_columnpointer[n]);
    calculate_type *new_value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * new_columnpointer[n]);

    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        if ((new_columnpointer[i + 1] - new_columnpointer[i]) == (s->columnpointer[i + 1] - s->columnpointer[i]))
        {
            for (pangulu_exblock_ptr j = new_columnpointer[i], k = s->columnpointer[i]; j < new_columnpointer[i + 1]; j++, k++)
            {
                new_rowindex[j] = s->rowindex[k];
                new_value[j] = s->value_csc[k];
            }
        }
        else
        {
            char flag = 0;
            for (pangulu_exblock_ptr j = new_columnpointer[i], k = s->columnpointer[i]; k < s->columnpointer[i + 1]; j++, k++)
            {
                if (s->rowindex[k] < i)
                {
                    new_rowindex[j] = s->rowindex[k];
                    new_value[j] = s->value_csc[k];
                }
                else if (s->rowindex[k] > i)
                {
                    if (flag == 0)
                    {
                        new_rowindex[j] = i;
                        new_value[j] = ZERO_ELEMENT;
                        k--;
                        flag = 1;
                    }
                    else
                    {
                        new_rowindex[j] = s->rowindex[k];
                        new_value[j] = s->value_csc[k];
                    }
                }
                else
                {
                    printf(PANGULU_E_ADD_DIA);
                    exit(1);
                }
            }
            if (flag == 0)
            {
                new_rowindex[new_columnpointer[i + 1] - 1] = i;
                new_value[new_columnpointer[i + 1] - 1] = ZERO_ELEMENT;
            }
        }
    }

    pangulu_free(__FILE__, __LINE__, s->columnpointer);
    pangulu_free(__FILE__, __LINE__, s->rowindex);
    pangulu_free(__FILE__, __LINE__, s->value_csc);
    s->columnpointer = new_columnpointer;
    s->rowindex = new_rowindex;
    s->value_csc = new_value;
    s->nnz = new_columnpointer[n];
}

void pangulu_origin_smatrix_add_csr(pangulu_origin_smatrix *a)
{
    pangulu_exblock_ptr nnzA = a->columnpointer[a->row];
    pangulu_exblock_idx n = a->row;
    char *now_malloc_space = (char *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1) + sizeof(pangulu_exblock_idx) * nnzA + sizeof(calculate_type) * nnzA);

    pangulu_exblock_ptr *columnpointer = (pangulu_exblock_ptr *)now_malloc_space;
    pangulu_exblock_idx *rowindex = (pangulu_exblock_idx *)(now_malloc_space + sizeof(pangulu_exblock_ptr) * (n + 1));
    calculate_type *value = (calculate_type *)(now_malloc_space + sizeof(pangulu_exblock_ptr) * (n + 1) + sizeof(pangulu_exblock_idx) * nnzA);
    pangulu_exblock_ptr *csr_to_csc_index = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * nnzA);
    for (pangulu_exblock_ptr i = 0; i < nnzA; i++)
    {
        value[i] = 0.0;
    }

    for (pangulu_exblock_idx i = 0; i < (n + 1); i++)
    {
        columnpointer[i] = 0;
    }
    for (pangulu_exblock_ptr i = 0; i < nnzA; i++)
    {
        columnpointer[a->rowindex[i] + 1]++;
    }
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        columnpointer[i + 1] += columnpointer[i];
    }
    pangulu_exblock_ptr *index_columnpointer = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1));
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        index_columnpointer[i] = columnpointer[i];
    }
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        for (pangulu_exblock_ptr j = a->columnpointer[i]; j < a->columnpointer[i + 1]; j++)
        {

            pangulu_exblock_idx row = a->rowindex[j];
            pangulu_exblock_ptr index = index_columnpointer[row];
            rowindex[index] = i;
            value[index] = a->value_csc[j];
            csr_to_csc_index[index] = j;
            index_columnpointer[row]++;
        }
    }
    a->rowpointer = columnpointer;
    a->columnindex = rowindex;
    a->value = value;
    pangulu_free(__FILE__, __LINE__, index_columnpointer);
}

void pangulu_get_graph_struct(pangulu_origin_smatrix *s, Hunyuan_int_t **xadj_address, Hunyuan_int_t **adjncy_address)
{
    pangulu_add_diagonal_element(s);
    pangulu_origin_smatrix_add_csc(s);

    Hunyuan_int_t *xadj = (Hunyuan_int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(Hunyuan_int_t) * (s->row + 1));
    Hunyuan_int_t *adjncy = pangulu_malloc(__FILE__, __LINE__, sizeof(Hunyuan_int_t) * s->nnz * 2);

    xadj[0] = 0;
    for (pangulu_exblock_idx rc = 0; rc < s->row; rc++)
    {
        pangulu_exblock_idx nnz_count = 0;
        pangulu_exblock_ptr start1 = s->rowpointer[rc];
        pangulu_exblock_ptr end1 = s->rowpointer[rc + 1];
        pangulu_exblock_ptr start2 = s->columnpointer[rc];
        pangulu_exblock_ptr end2 = s->columnpointer[rc + 1];
        pangulu_exblock_ptr idx1 = start1;
        pangulu_exblock_ptr idx2 = start2;

        while ((idx1 < end1) && (idx2 < end2))
        {
            if (s->columnindex[idx1] < s->rowindex[idx2])
            {
                idx1++;
            }
            else if (s->columnindex[idx1] > s->rowindex[idx2])
            {
                idx2++;
            }
            else
            {
                idx1++;
                idx2++;
            }
            nnz_count++;
        }
        while (idx1 < end1)
        {
            idx1++;
            nnz_count++;
        }
        while (idx2 < end2)
        {
            idx2++;
            nnz_count++;
        }
        nnz_count--;
        xadj[rc + 1] = xadj[rc] + nnz_count;

        nnz_count = 0;
        idx1 = start1;
        idx2 = start2;
        while ((idx1 < end1) && (idx2 < end2))
        {
            if (s->columnindex[idx1] < s->rowindex[idx2])
            {
                if (s->columnindex[idx1] != rc)
                {
                    adjncy[xadj[rc] + nnz_count] = s->columnindex[idx1];
                    nnz_count++;
                }
                idx1++;
            }
            else if (s->columnindex[idx1] > s->rowindex[idx2])
            {
                if (s->rowindex[idx2] != rc)
                {
                    adjncy[xadj[rc] + nnz_count] = s->rowindex[idx2];
                    nnz_count++;
                }
                idx2++;
            }
            else
            {
                if (s->columnindex[idx1] != rc)
                {
                    adjncy[xadj[rc] + nnz_count] = s->columnindex[idx1];
                    nnz_count++;
                }
                idx1++;
                idx2++;
            }
        }
        while (idx1 < end1)
        {
            if (s->columnindex[idx1] != rc)
            {
                adjncy[xadj[rc] + nnz_count] = s->columnindex[idx1];
                nnz_count++;
            }
            idx1++;
        }
        while (idx2 < end2)
        {
            if (s->rowindex[idx2] != rc)
            {
                adjncy[xadj[rc] + nnz_count] = s->rowindex[idx2];
                nnz_count++;
            }
            idx2++;
        }
    }

    adjncy = pangulu_realloc(__FILE__, __LINE__, adjncy, sizeof(Hunyuan_int_t) * xadj[s->row]);
    *xadj_address = xadj;
    *adjncy_address = adjncy;
}

void pangulu_get_graph_struct_csc(pangulu_origin_smatrix *s, Hunyuan_int_t **xadj_address, Hunyuan_int_t **adjncy_address)
{
    pangulu_add_diagonal_element_csc(s);
    pangulu_origin_smatrix_add_csr(s);

    Hunyuan_int_t *xadj = (Hunyuan_int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(Hunyuan_int_t) * (s->row + 1));
    Hunyuan_int_t *adjncy = pangulu_malloc(__FILE__, __LINE__, sizeof(Hunyuan_int_t) * s->nnz * 2);

    xadj[0] = 0;
    for (pangulu_exblock_idx rc = 0; rc < s->row; rc++)
    {
        pangulu_exblock_idx nnz_count = 0;
        pangulu_exblock_ptr start1 = s->rowpointer[rc];
        pangulu_exblock_ptr end1 = s->rowpointer[rc + 1];
        pangulu_exblock_ptr start2 = s->columnpointer[rc];
        pangulu_exblock_ptr end2 = s->columnpointer[rc + 1];
        pangulu_exblock_ptr idx1 = start1;
        pangulu_exblock_ptr idx2 = start2;

        while ((idx1 < end1) && (idx2 < end2))
        {
            if (s->columnindex[idx1] < s->rowindex[idx2])
            {
                idx1++;
            }
            else if (s->columnindex[idx1] > s->rowindex[idx2])
            {
                idx2++;
            }
            else
            {
                idx1++;
                idx2++;
            }
            nnz_count++;
        }
        while (idx1 < end1)
        {
            idx1++;
            nnz_count++;
        }
        while (idx2 < end2)
        {
            idx2++;
            nnz_count++;
        }
        nnz_count--;
        xadj[rc + 1] = xadj[rc] + nnz_count;

        nnz_count = 0;
        idx1 = start1;
        idx2 = start2;
        while ((idx1 < end1) && (idx2 < end2))
        {
            if (s->columnindex[idx1] < s->rowindex[idx2])
            {
                if (s->columnindex[idx1] != rc)
                {
                    adjncy[xadj[rc] + nnz_count] = s->columnindex[idx1];
                    nnz_count++;
                }
                idx1++;
            }
            else if (s->columnindex[idx1] > s->rowindex[idx2])
            {
                if (s->rowindex[idx2] != rc)
                {
                    adjncy[xadj[rc] + nnz_count] = s->rowindex[idx2];
                    nnz_count++;
                }
                idx2++;
            }
            else
            {
                if (s->columnindex[idx1] != rc)
                {
                    adjncy[xadj[rc] + nnz_count] = s->columnindex[idx1];
                    nnz_count++;
                }
                idx1++;
                idx2++;
            }
        }
        while (idx1 < end1)
        {
            if (s->columnindex[idx1] != rc)
            {
                adjncy[xadj[rc] + nnz_count] = s->columnindex[idx1];
                nnz_count++;
            }
            idx1++;
        }
        while (idx2 < end2)
        {
            if (s->rowindex[idx2] != rc)
            {
                adjncy[xadj[rc] + nnz_count] = s->rowindex[idx2];
                nnz_count++;
            }
            idx2++;
        }
    }

    adjncy = pangulu_realloc(__FILE__, __LINE__, adjncy, sizeof(Hunyuan_int_t) * xadj[s->row]);
    *xadj_address = xadj;
    *adjncy_address = adjncy;
}

#ifdef METIS
void pangulu_metis(pangulu_origin_smatrix *a, Hunyuan_int_t **metis_perm)
{
    Hunyuan_int_t nvtxs, nedges, compress, control, is_memery_manage_before;
    Hunyuan_int_t *xadj, *adjncy, *vwgt, *adjwgt, *perm, *iperm;

    nvtxs = a->row;

    iperm = (Hunyuan_int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(Hunyuan_int_t) * nvtxs);
    perm = (Hunyuan_int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(Hunyuan_int_t) * nvtxs);
    memset(iperm, 0, sizeof(Hunyuan_int_t) * nvtxs);
    memset(perm, 0, sizeof(Hunyuan_int_t) * nvtxs);

    pangulu_get_graph_struct_csc(a, &xadj, &adjncy);

    METIS_NodeND(&nvtxs, xadj, adjncy, NULL, NULL, perm, iperm);

    pangulu_free(__FILE__, __LINE__, perm);

    *metis_perm = iperm;
    pangulu_free(__FILE__, __LINE__, a->rowpointer);
    a->rowpointer = NULL;
    a->columnindex = NULL;
}
#else
void pangulu_hunyuan_mt(pangulu_origin_smatrix *a, Hunyuan_int_t **metis_perm, Hunyuan_int_t nthreads)
{
    Hunyuan_int_t nvtxs, nedges, compress, control, is_memery_manage_before;
    Hunyuan_int_t *xadj, *adjncy, *vwgt, *adjwgt, *perm, *iperm;

    nvtxs = a->row;

    iperm = (Hunyuan_int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(Hunyuan_int_t) * nvtxs);
    perm = (Hunyuan_int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(Hunyuan_int_t) * nvtxs);
    vwgt = (Hunyuan_int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(Hunyuan_int_t) * nvtxs);

    pangulu_get_graph_struct_csc(a, &xadj, &adjncy);

    nedges = xadj[nvtxs];
    adjwgt = (Hunyuan_int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(Hunyuan_int_t) * nedges);

    for (Hunyuan_int_t i = 0; i < nvtxs; i++)
        vwgt[i] = 1;
    for (Hunyuan_int_t i = 0; i < nedges; i++)
        adjwgt[i] = 1;

    compress = 1;
    control = 1;
    is_memery_manage_before = 0;
    if (nthreads == 0)
    {
        nthreads = 4;
    }

    mynd_ReorderGraph(&nvtxs, &nedges, xadj, vwgt, adjncy, adjwgt, perm, iperm, &compress, &control, &is_memery_manage_before, nthreads);

    pangulu_free(__FILE__, __LINE__, perm);

    *metis_perm = iperm;
    pangulu_free(__FILE__, __LINE__, a->rowpointer);
    a->rowpointer = NULL;
    a->columnindex = NULL;
}

#endif

void pangulu_origin_smatrix_transport_transport_iperm(
    pangulu_origin_smatrix *s,
    pangulu_origin_smatrix *new_S,
    const pangulu_exblock_idx *metis_perm)
{
    pangulu_exblock_idx n = s->row;
    pangulu_exblock_ptr nnz = s->nnz;
    pangulu_exblock_ptr *columnpointer = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1));
    pangulu_exblock_idx *rowindex = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * nnz);
    calculate_type *value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz);
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        pangulu_exblock_idx column_len = s->columnpointer[i + 1] - s->columnpointer[i];
        pangulu_exblock_idx index = metis_perm[i];
        columnpointer[index + 1] = column_len;
    }
    columnpointer[0] = 0;
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        columnpointer[i + 1] += columnpointer[i];
    }
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        pangulu_exblock_idx index = metis_perm[i];
        pangulu_exblock_ptr before_column_begin = s->columnpointer[i];
        for (pangulu_exblock_ptr column_begin = columnpointer[index]; column_begin < columnpointer[index + 1]; column_begin++, before_column_begin++)
        {
            rowindex[column_begin] = metis_perm[s->rowindex[before_column_begin]];
            value[column_begin] = s->value_csc[before_column_begin];
        }
    }
    new_S->row = n;
    new_S->column = n;
    new_S->columnpointer = columnpointer;
    new_S->rowindex = rowindex;
    new_S->value_csc = value;
    new_S->nnz = nnz;
}

void pangulu_reordering(
    pangulu_block_smatrix *block_smatrix,
    pangulu_origin_smatrix *origin_matrix,
    pangulu_origin_smatrix *reorder_matrix,
    pangulu_int32_t hunyuan_nthread)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        pangulu_exblock_idx n = origin_matrix->row;
        pangulu_exblock_ptr nnz = origin_matrix->nnz;
        pangulu_exblock_idx *perm = NULL;
        pangulu_exblock_idx *iperm = NULL;
        pangulu_exblock_idx *metis_perm = NULL;
        Hunyuan_int_t *metis_perm_tmp = NULL;
        calculate_type *row_scale = NULL;
        calculate_type *col_scale = NULL;
        pangulu_origin_smatrix *tmp;
#if defined(PANGULU_MC64)
        pangulu_mc64(origin_matrix, &perm, &iperm, &row_scale, &col_scale);
        int mc64_fail_flag = 0;
        for (int i = 0; i < n; i++)
        {
            if ((~perm[i]) == 0 || (~iperm[i]) == 0)
            {
                mc64_fail_flag = 1;
                printf(PANGULU_W_MC64_FAIL);
                break;
            }
        }
        if (mc64_fail_flag)
        {
            for (int i = 0; i < n; i++)
            {
                perm[i] = i;
                iperm[i] = i;
                row_scale[i] = 1.0;
                col_scale[i] = 1.0;
            }
        }
#else
        perm = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * n);
        iperm = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * n);

        row_scale = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);
        col_scale = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);

        for (pangulu_int64_t i = 0; i < n; i++)
        {
            perm[i] = i;
        }

        for (pangulu_int64_t i = 0; i < n; i++)
        {
            iperm[i] = i;
        }

        for (pangulu_int64_t i = 0; i < n; i++)
        {
            row_scale[i] = 1.0;
        }

        for (pangulu_int64_t i = 0; i < n; i++)
        {
            col_scale[i] = 1.0;
        }
#endif
        tmp = (pangulu_origin_smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_origin_smatrix));
        pangulu_init_pangulu_origin_smatrix(tmp);
        tmp->row = n;
        tmp->column = n;
        tmp->nnz = nnz;
        tmp->columnpointer = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1));
        tmp->rowindex = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * nnz);
        tmp->value_csc = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz);

        tmp->columnpointer[0] = 0;

        for (pangulu_int64_t i = 0; i < n; i++)
        {
            pangulu_int64_t col = iperm[i];
            tmp->columnpointer[col + 1] = origin_matrix->columnpointer[i + 1] - origin_matrix->columnpointer[i];
        }

        for (pangulu_int64_t i = 0; i < n; i++)
        {
            tmp->columnpointer[i + 1] += tmp->columnpointer[i];
        }

        for (pangulu_int64_t i = 0; i < n; i++)
        {
            pangulu_exblock_idx col = iperm[i];
            calculate_type cs = col_scale[i];
            pangulu_int64_t tmp_index = tmp->columnpointer[col];
            for (pangulu_exblock_ptr j = origin_matrix->columnpointer[i]; j < origin_matrix->columnpointer[i + 1]; j++)
            {
                pangulu_exblock_idx row = origin_matrix->rowindex[j];
                tmp->rowindex[tmp_index] = row;
                tmp->value_csc[tmp_index] = (origin_matrix->value_csc[j] * cs * row_scale[row]);
                tmp_index++;
            }
        }
#ifdef METIS
        pangulu_metis(tmp, &metis_perm_tmp);
#else
        pangulu_hunyuan_mt(tmp, &metis_perm_tmp, hunyuan_nthread);
#endif
        if (sizeof(pangulu_exblock_idx) == sizeof(Hunyuan_int_t))
        {
            metis_perm = metis_perm_tmp;
        }
        else
        {
            metis_perm = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * n);
            for (int i = 0; i < n; i++)
            {
                metis_perm[i] = metis_perm_tmp[i];
            }
            pangulu_free(__FILE__, __LINE__, metis_perm_tmp);
        }

        pangulu_origin_smatrix_transport_transport_iperm(tmp, reorder_matrix, metis_perm);
        pangulu_sort_pangulu_origin_smatrix_csc(reorder_matrix);

        block_smatrix->row_perm = perm;
        block_smatrix->col_perm = iperm;
        block_smatrix->row_scale = row_scale;
        block_smatrix->col_scale = col_scale;
        block_smatrix->metis_perm = metis_perm;

        pangulu_free(__FILE__, __LINE__, tmp->columnpointer);
        pangulu_free(__FILE__, __LINE__, tmp->rowindex);
        pangulu_free(__FILE__, __LINE__, tmp->value_csc);
        pangulu_free(__FILE__, __LINE__, tmp);
    }

    pangulu_cm_sync_asym(0);
}