#include "pangulu_common.h"

#ifdef PANGULU_MC64
void pangulu_mc64dd(pangulu_int64_t col, pangulu_int64_t n, pangulu_int64_t *queue, calculate_type *row_scale_value, pangulu_int64_t *save_tmp)
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

void pangulu_mc64ed(pangulu_int64_t *queue_length, pangulu_int64_t n, pangulu_int64_t *queue, calculate_type *row_scale_value, pangulu_int64_t *save_tmp)
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

void pangulu_mc64fd(pangulu_int64_t loc_origin, pangulu_int64_t *queue_length, pangulu_int64_t n, pangulu_int64_t *queue, calculate_type *row_scale_value, pangulu_int64_t *save_tmp)
{
    (*queue_length)--;
    pangulu_int64_t now_queue_length = *queue_length;

    if (loc_origin == now_queue_length)
    {
        return;
    }

    pangulu_int64_t loc = loc_origin;
    calculate_type rsv_min = row_scale_value[queue[now_queue_length]];

    // begin mc64dd
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

    // begin mc64ed
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

void pangulu_mc64(pangulu_origin_smatrix *s, pangulu_exblock_idx **perm, pangulu_exblock_idx **iperm,
                  calculate_type **row_scale, calculate_type **col_scale)
{

    pangulu_exblock_idx n = s->row;
    pangulu_exblock_ptr nnz = s->nnz;

    pangulu_int64_t finish_flag = 0;

    pangulu_exblock_ptr *rowptr = s->rowpointer;
    pangulu_exblock_idx *colidx = s->columnindex;
    calculate_type *val = s->value;

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
            pangulu_exit(1);
        }
    }

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        col_perm[i] = 4294967295;
    }

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        row_perm[i] = 4294967295;
    }
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        col_scale_value[i] = DBL_MAX;
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
            pangulu_exit(1);
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
            pangulu_exit(1);
        }

        for (pangulu_int64_t j = rowptr[i]; j < rowptr[i + 1]; j++)
        {

            if ((fabs_value[j] != 0.0))
            {
                fabs_value[j] = now_row_max - log(fabs_value[j]);
            }
            else
            {
                fabs_value[j] = DBL_MAX / 5.0;
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
        if (col_perm[i] < 4294967295)
        {
            if (row_perm[col_perm[i]] == 4294967295)
            {
                finish_flag++;
                col_perm[i] = col_perm[i];
                row_perm[col_perm[i]] = save_tmp[i];
            }
            else
            {
                col_perm[i] = 4294967295;
            }
        }
    }

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        if (finish_flag == n)
        {
            break;
        }
        if (row_perm[i] == 4294967295)
        {
            calculate_type col_max = DBL_MAX;
            pangulu_int64_t save_col = -1;
            pangulu_int64_t save_index = -1;
            for (pangulu_int64_t j = rowptr[i]; j < rowptr[i + 1]; j++)
            {
                pangulu_int64_t col = colidx[j];
                calculate_type now_value = fabs_value[j] - col_scale_value[col];
                if (now_value > col_max)
                {
                    // nothing to do
                }
                else if ((now_value >= col_max) && (now_value != DBL_MAX))
                {
                    if ((col_perm[col] == 4294967295) && (col_perm[save_col] < 4294967295))
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
            if (col_perm[save_col] == 4294967295)
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
                                if (col_perm[tmp_col] == 4294967295)
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
            row_scale_value[i] = DBL_MAX;
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
        if (row_perm[now_row] == 4294967295)
        {
            pangulu_int64_t row = now_row;
            pangulu_int64_t queue_length = 0;
            pangulu_int64_t low = n;
            pangulu_int64_t top = n;
            pangulu_int64_t save_index = -1;
            pangulu_int64_t save_row = -1;

            rowptr_tmp[row] = PANGULU_MC64_FLAG;
            calculate_type min_cost = DBL_MAX;
            calculate_type sum_cost = DBL_MAX;

            for (pangulu_int64_t k = rowptr[row]; k < rowptr[row + 1]; k++)
            {
                pangulu_int64_t col = colidx[k];
                calculate_type now_value = fabs_value[k] - col_scale_value[col];
                if (now_value < sum_cost)
                {
                    if (col_perm[col] == 4294967295)
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
                    row_scale_value[col] = DBL_MAX;
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
                    pangulu_int64_t now_col = col_perm[col];
                    ans_queue[now_col] = queue_index;
                    rowptr_tmp[now_col] = row;
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
                calculate_type row_sum_max = row_scale_value[now_queue_length] - fabs_value[row_perm[row]] + col_scale_value[now_queue_length];

                for (pangulu_int64_t k = rowptr[row]; k < rowptr[row + 1]; k++)
                {
                    pangulu_int64_t col = colidx[k];
                    if (save_tmp[col] < top)
                    {
                        calculate_type now_value = row_sum_max + fabs_value[k] - col_scale_value[col];
                        if (now_value < sum_cost)
                        {
                            if (col_perm[col] == 4294967295)
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

            if (sum_cost != DBL_MAX)
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
                row_scale_value[col] = DBL_MAX;
                save_tmp[col] = -1;
            }

            for (pangulu_int64_t k = 0; k < queue_length; k++)
            {
                pangulu_int64_t col = queue[k];
                row_scale_value[col] = DBL_MAX;
                save_tmp[col] = -1;
            }
        }
    }

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        pangulu_exblock_idx now_col = row_perm[i];
        if (now_col < 4294967295)
        {
            pangulu_exblock_idx wcs = colidx[now_col];
            row_scale_value[i] = fabs_value[now_col] - col_scale_value[wcs];
        }
        else
        {
            row_scale_value[i] = 0.0;
        }
        if (col_perm[i] == 4294967295)
        {
            col_scale_value[i] = 0.0;
        }
    }

    if (finish_flag == n)
    {
        for (pangulu_int64_t i = 0; i < n; i++)
        {
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
            row_perm[i] = 4294967295;
        }

        pangulu_int64_t ans_queue_length = 0;
        for (pangulu_int64_t i = 0; i < n; i++)
        {
            if (col_perm[i] == 4294967295)
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
            if (row_perm[i] == 4294967295)
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
}
#endif

void pangulu_origin_smatrix_transport_transport_iperm(pangulu_origin_smatrix *s, pangulu_origin_smatrix *new_S, pangulu_exblock_idx *metis_perm)
{
    pangulu_exblock_idx n = s->row;
    pangulu_exblock_ptr nnz = s->nnz;
    pangulu_exblock_ptr *rowpointer = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1));
    pangulu_exblock_idx *columnindex = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * nnz);
    calculate_type *value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz);
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        pangulu_exblock_idx row_num = s->rowpointer[i + 1] - s->rowpointer[i];
        pangulu_exblock_idx index = metis_perm[i];
        rowpointer[index + 1] = row_num;
    }
    rowpointer[0] = 0;
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        rowpointer[i + 1] += rowpointer[i];
    }
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {
        pangulu_exblock_idx index = metis_perm[i];
        pangulu_exblock_ptr before_row_begin = s->rowpointer[i];
        for (pangulu_exblock_ptr row_begin = rowpointer[index]; row_begin < rowpointer[index + 1]; row_begin++, before_row_begin++)
        {
            columnindex[row_begin] = metis_perm[s->columnindex[before_row_begin]];
            value[row_begin] = s->value[before_row_begin];
        }
    }
    new_S->row = n;
    new_S->column = n;
    new_S->rowpointer = rowpointer;
    new_S->columnindex = columnindex;
    new_S->value = value;
    new_S->nnz = nnz;
}

#ifdef METIS
void pangulu_get_graph_struct(pangulu_origin_smatrix *s, idx_t **xadj_adress, idx_t **adjincy_adress)
{
    pangulu_add_diagonal_element(s);
    pangulu_origin_smatrix_add_csc(s);
    idx_t *xadj = (idx_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(idx_t) * (s->row + 1));
    xadj[0] = 0;

    for (pangulu_exblock_idx i = 0; i < s->row; i++)
    {
        pangulu_exblock_ptr index1 = s->rowpointer[i];
        pangulu_exblock_ptr index2 = s->columnpointer[i];
        pangulu_exblock_ptr end1 = s->rowpointer[i + 1];
        pangulu_exblock_ptr end2 = s->columnpointer[i + 1];
        char diagonal_flag = 0;
        pangulu_int64_t sum_num = 0;
        pangulu_exblock_idx col1 = s->columnindex[index1];
        pangulu_exblock_idx col2 = s->rowindex[index2];
        while ((index1 < end1) && (index2 < end2))
        {
            if (col1 == col2)
            {
                if ((diagonal_flag == 0) && (col1 == i))
                {
                    diagonal_flag = 1;
                }
                index1++;
                index2++;
                col1 = s->columnindex[index1];
                col2 = s->rowindex[index2];
            }
            else if (col1 < col2)
            {
                if ((diagonal_flag == 0) && (col1 == i))
                {
                    diagonal_flag = 1;
                }
                index1++;
                col1 = s->columnindex[index1];
            }
            else
            {
                if ((diagonal_flag == 0) && (col2 == i))
                {
                    diagonal_flag = 1;
                }
                index2++;
                col2 = s->rowindex[index2];
            }
            sum_num++;
        }
        while (index1 < end1)
        {
            sum_num++;
            if ((diagonal_flag == 0) && (col1 == i))
            {
                diagonal_flag = 1;
            }
            index1++;
            col1 = s->columnindex[index1];
        }

        while (index2 < end2)
        {
            sum_num++;
            if ((diagonal_flag == 0) && (col2 == i))
            {
                diagonal_flag = 1;
            }
            index2++;
            col2 = s->rowindex[index2];
        }
        if (diagonal_flag == 0)
        {
            printf(PANGULU_E_ROW_DONT_HAVE_DIA);
            pangulu_exit(1);
        }
        xadj[i + 1] = sum_num - diagonal_flag;
    }
    for (pangulu_exblock_idx i = 0; i < s->row; i++)
    {
        xadj[i + 1] += xadj[i];
    }

    idx_t *adjincy = (idx_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(idx_t) * (xadj[s->row]));

    for (pangulu_exblock_ptr i = 0; i < s->row; i++)
    {
        idx_t now_adjincy_index = xadj[i];
        pangulu_exblock_ptr index1 = s->rowpointer[i];
        pangulu_exblock_ptr index2 = s->columnpointer[i];
        pangulu_exblock_ptr end1 = s->rowpointer[i + 1];
        pangulu_exblock_ptr end2 = s->columnpointer[i + 1];
        char diagonal_flag = 0;
        pangulu_exblock_idx col1 = s->columnindex[index1];
        pangulu_exblock_idx col2 = s->rowindex[index2];
        while ((index1 < end1) && (index2 < end2))
        {
            if (col1 == col2)
            {
                if ((diagonal_flag == 0) && (col1 == i))
                {
                    diagonal_flag = 1;
                }
                else
                {
                    adjincy[now_adjincy_index] = col1;
                    now_adjincy_index++;
                }
                index1++;
                index2++;
                col1 = s->columnindex[index1];
                col2 = s->rowindex[index2];
            }
            else if (col1 < col2)
            {
                if ((diagonal_flag == 0) && (col1 == i))
                {
                    diagonal_flag = 1;
                }
                else
                {
                    adjincy[now_adjincy_index] = col1;
                    now_adjincy_index++;
                }
                index1++;
                col1 = s->columnindex[index1];
            }
            else
            {
                if ((diagonal_flag == 0) && (col2 == i))
                {
                    diagonal_flag = 1;
                }
                else
                {
                    adjincy[now_adjincy_index] = col2;
                    now_adjincy_index++;
                }
                index2++;
                col2 = s->rowindex[index2];
            }
        }
        while (index1 < end1)
        {
            if ((diagonal_flag == 0) && (col1 == i))
            {
                diagonal_flag = 1;
            }
            else
            {
                adjincy[now_adjincy_index] = col1;
                now_adjincy_index++;
            }
            index1++;
            col1 = s->columnindex[index1];
        }

        while (index2 < end2)
        {
            if ((diagonal_flag == 0) && (col2 == i))
            {
                diagonal_flag = 1;
            }
            else
            {
                adjincy[now_adjincy_index] = col2;
                now_adjincy_index++;
            }
            index2++;
            col2 = s->rowindex[index2];
        }
    }
    *xadj_adress = xadj;
    *adjincy_adress = adjincy;
}

void pangulu_metis(pangulu_origin_smatrix *a, idx_t **metis_perm)
{

    idx_t n = a->row;

    idx_t *iperm = (idx_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(idx_t) * n);
    idx_t *perm = (idx_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(idx_t) * n);

    idx_t *xadj = NULL;
    idx_t *adjincy = NULL;

    pangulu_get_graph_struct(a, &xadj, &adjincy);
    METIS_NodeND(&n, xadj, adjincy, NULL, NULL, perm, iperm);

    pangulu_free(__FILE__, __LINE__, perm);
    pangulu_free(__FILE__, __LINE__, xadj);
    pangulu_free(__FILE__, __LINE__, adjincy);

    *metis_perm = iperm;
    pangulu_free(__FILE__, __LINE__, a->columnpointer);
    pangulu_free(__FILE__, __LINE__, a->csc_to_csr_index);
    a->columnpointer = NULL;
    a->rowindex = NULL;
    a->value_csc = NULL;
}

#endif
void pangulu_reorder_vector_x_tran(pangulu_block_smatrix *block_smatrix,
                                   pangulu_vector *X_origin,
                                   pangulu_vector *X_trans)
{
    pangulu_int64_t n = X_origin->row;
    // tran X col
    pangulu_exblock_idx *metis_perm = block_smatrix->metis_perm;
    calculate_type *col_scale = block_smatrix->col_scale;

    for (idx_t i = 0; i < n; i++)
    {
        pangulu_int64_t now_col = metis_perm[i];
        X_trans->value[i] = X_origin->value[now_col] * col_scale[i];
    }
}

void pangulu_reorder_vector_b_tran(pangulu_block_smatrix *block_smatrix,
                                   pangulu_vector *B_origin,
                                   pangulu_vector *B_trans)
{
    pangulu_int64_t n = B_origin->row;
    // tran B row
    pangulu_exblock_idx *row_perm = block_smatrix->row_perm;
    pangulu_exblock_idx *metis_perm = block_smatrix->metis_perm;
    calculate_type *row_scale = block_smatrix->row_scale;

    for (pangulu_int64_t i = 0; i < n; i++)
    {
        pangulu_int64_t index_row = row_perm[i];
        pangulu_int64_t now_row = metis_perm[index_row];
        B_trans->value[now_row] = B_origin->value[i] * row_scale[i];
    }
}

void pangulu_reorder(pangulu_block_smatrix *block_smatrix,
                     pangulu_origin_smatrix *origin_matrix,
                     pangulu_origin_smatrix *reorder_matrix)
{
    pangulu_exblock_idx n = origin_matrix->row;
    pangulu_exblock_ptr nnz = origin_matrix->nnz;
    pangulu_exblock_idx *perm = NULL;
    pangulu_exblock_idx *iperm = NULL;
    pangulu_exblock_idx *metis_perm = NULL;
    idx_t *metis_perm_tmp = NULL;
    calculate_type *row_scale = NULL;
    calculate_type *col_scale = NULL;
    pangulu_origin_smatrix *tmp;
    if (rank == 0)
    {
#if defined(PANGULU_MC64)
        pangulu_mc64(origin_matrix, &perm, &iperm, &row_scale, &col_scale);
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
        tmp->rowpointer = (pangulu_exblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_ptr) * (n + 1));
        tmp->columnindex = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * nnz);
        tmp->value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz);

        tmp->rowpointer[0] = 0;

        for (pangulu_int64_t i = 0; i < n; i++)
        {
            pangulu_int64_t row = perm[i];
            tmp->rowpointer[row + 1] = origin_matrix->rowpointer[i + 1] - origin_matrix->rowpointer[i];
        }

        for (pangulu_int64_t i = 0; i < n; i++)
        {
            tmp->rowpointer[i + 1] += tmp->rowpointer[i];
        }

        for (pangulu_int64_t i = 0; i < n; i++)
        {
            pangulu_exblock_idx row = perm[i];
            calculate_type rs = row_scale[i];
            pangulu_int64_t tmp_index = tmp->rowpointer[row];
            for (pangulu_exblock_ptr j = origin_matrix->rowpointer[i]; j < origin_matrix->rowpointer[i + 1]; j++)
            {
                pangulu_exblock_idx col = origin_matrix->columnindex[j];
                tmp->columnindex[tmp_index] = col;
                tmp->value[tmp_index] = (origin_matrix->value[j] * rs * col_scale[col]);
                // if (col == row)
                // {
                //     if (fabs(fabs(tmp->value[tmp_index]) - 1.0) > 1e-10)
                //     {
                //     }
                // }
                tmp_index++;
            }
        }
    }

#ifdef METIS
    if (rank == 0)
    {
        pangulu_metis(tmp, &metis_perm_tmp);
        metis_perm = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * n);
        for (int i = 0; i < n; i++)
        {
            metis_perm[i] = metis_perm_tmp[i];
        }
        pangulu_free(__FILE__, __LINE__, metis_perm_tmp);
    }
#else
    if (rank == 0)
    {
        pangulu_add_diagonal_element(tmp);
        metis_perm = (pangulu_exblock_idx *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_exblock_idx) * n);
        for (int i = 0; i < n; i++)
        {
            metis_perm[i] = i;
        }
    }
#endif
    if (rank == 0)
    {
        pangulu_origin_smatrix_transport_transport_iperm(tmp, reorder_matrix, metis_perm);
        pangulu_sort_pangulu_origin_smatrix(reorder_matrix);

        block_smatrix->row_perm = perm;
        block_smatrix->col_perm = iperm;
        block_smatrix->row_scale = row_scale;
        block_smatrix->col_scale = col_scale;
        block_smatrix->metis_perm = metis_perm;

        pangulu_free(__FILE__, __LINE__, tmp->rowpointer);
        pangulu_free(__FILE__, __LINE__, tmp->columnindex);
        pangulu_free(__FILE__, __LINE__, tmp->value);
        pangulu_free(__FILE__, __LINE__, tmp);
    }
}