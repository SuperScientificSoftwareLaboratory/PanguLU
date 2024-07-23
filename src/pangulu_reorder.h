#ifndef PANGULU_REORDER_H

#define PANGULU_REORDER_H

#include "pangulu_common.h"
#include "pangulu_utils.h"
#include "pangulu_malloc.h"
#include "math.h"
#include "float.h"


#ifdef PANGULU_MC64
void pangulu_MC64dd(int_t col, int_t n, int_t *queue, calculate_type *row_scale_value, int_t *save_tmp)
{
    int_t loc = save_tmp[col];
    calculate_type rsv_min = row_scale_value[col];

    for (int_t i = 0; i < n; i++)
    {
        if (loc <= 0)
        {
            break;
        }
        int_t tmp_loc = (loc - 1) / 2;
        int_t index_loc = queue[tmp_loc];
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

void pangulu_MC64ed(int_t *queue_length, int_t n, int_t *queue, calculate_type *row_scale_value, int_t *save_tmp)
{
    int_t loc = 0;
    (*queue_length)--;
    int_t now_queue_length = *queue_length;
    calculate_type rsv_min = row_scale_value[queue[now_queue_length]];

    for (int_t i = 0; i < n; i++)
    {
        int_t tmp_loc = (loc + 1) * 2 - 1;
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

void pangulu_MC64fd(int_t loc_origin, int_t *queue_length, int_t n, int_t *queue, calculate_type *row_scale_value, int_t *save_tmp)
{
    (*queue_length)--;
    int_t now_queue_length = *queue_length;

    if (loc_origin == now_queue_length)
    {
        return;
    }

    int_t loc = loc_origin;
    calculate_type rsv_min = row_scale_value[queue[now_queue_length]];

    // begin mc64dd
    for (int_t i = 0; i < n; ++i)
    {
        if (loc <= 0)
        {
            break;
        }
        int_t tmp_loc = (loc - 1) / 2;
        int_t index_loc = queue[tmp_loc];
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
    for (int_t i = 0; i < n; i++)
    {
        int_t tmp_loc = (loc + 1) * 2 - 1;
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

void pangulu_mc64(pangulu_origin_Smatrix *S, int_t **perm, int_t **iperm,
                  calculate_type **row_scale, calculate_type **col_scale)
{

    int_t n = S->row;
    int_t nnz = S->nnz;

    int_t finish_flag = 0;

    int_t *rowptr = S->rowpointer;
    int_32t *colidx = S->columnindex;
    calculate_type *val = S->value;

    int_t *col_perm = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * n);
    int_t *row_perm = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * n);
    int_t *rowptr_tmp = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * n);
    int_t *queue = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * n);
    int_t *save_tmp = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * n);
    int_t *ans_queue = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * n);

    calculate_type *fabs_value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz);
    calculate_type *max_value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);
    calculate_type *col_scale_value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);
    calculate_type *row_scale_value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);

    for (int_t i = 0; i < n; i++)
    {
        if (rowptr[i] >= rowptr[i + 1])
        {
            printf(PANGULU_E_ROW_IS_NULL);
            fflush(NULL);
            exit(-1);
        }
    }

    for (int_t i = 0; i < n; i++)
    {
        col_perm[i] = -1;
    }

    for (int_t i = 0; i < n; i++)
    {
        row_perm[i] = -1;
    }
    for (int_t i = 0; i < n; i++)
    {
        col_scale_value[i] = DBL_MAX;
    }
    for (int_t i = 0; i < n; i++)
    {
        row_scale_value[i] = 0.0;
    }

    for (int_t i = 0; i < n; i++)
    {
        rowptr_tmp[i] = rowptr[i];
    }

    for (int_t i = 0; i < n; i++)
    {
        save_tmp[i] = -1;
    }

    for (int_t i = 0; i < n; i++)
    {
        if (rowptr[i] >= rowptr[i + 1])
        {
            printf(PANGULU_E_ROW_IS_NULL);
            fflush(NULL);
            exit(-1);
        }
    }

    for (int_t i = 0; i < n; i++)
    {
        max_value[i] = 0.0;
        for (int_t j = rowptr[i]; j < rowptr[i + 1]; j++)
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
            fflush(NULL);
            exit(0);
        }

        for (int_t j = rowptr[i]; j < rowptr[i + 1]; j++)
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

    for (int_t i = 0; i < n; i++)
    {
        for (int_t j = rowptr[i]; j < rowptr[i + 1]; j++)
        {
            int_t col = colidx[j];
            if (fabs_value[j] <= col_scale_value[col])
            {
                col_scale_value[col] = fabs_value[j];
                col_perm[col] = i;
                save_tmp[col] = j;
            }
        }
    }
    for (int_t i = 0; i < n; i++)
    {
        if (col_perm[i] >= 0)
        {
            if (row_perm[col_perm[i]] < 0)
            {
                finish_flag++;
                col_perm[i] = col_perm[i];
                row_perm[col_perm[i]] = save_tmp[i];
            }
            else
            {
                col_perm[i] = -1;
            }
        }
    }

    for (int_t i = 0; i < n; i++)
    {
        if (finish_flag == n)
        {
            break;
        }
        if (row_perm[i] < 0)
        {
            calculate_type col_max = DBL_MAX;
            int_t save_col = -1;
            int_t save_index = -1;
            for (int_t j = rowptr[i]; j < rowptr[i + 1]; j++)
            {
                int_t col = colidx[j];
                calculate_type now_value = fabs_value[j] - col_scale_value[col];
                if (now_value > col_max)
                {
                    // nothing to do
                }
                else if ((now_value >= col_max) && (now_value != DBL_MAX))
                {
                    if ((col_perm[col] < 0) && (col_perm[save_col] >= 0))
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
            if (col_perm[save_col] < 0)
            {
                row_perm[i] = save_index;
                col_perm[save_col] = i;
                rowptr_tmp[i] = save_index + 1;
                finish_flag++;
            }
            else
            {
                int_t break_flag = 0;
                for (int_t j = save_index; j < rowptr[i + 1]; j++)
                {
                    int_t col = colidx[j];
                    if ((fabs_value[j] - col_scale_value[col]) <= col_max)
                    {
                        int_t now_col = col_perm[col];
                        if (rowptr_tmp[now_col] < rowptr[now_col + 1])
                        {
                            for (int_t k = rowptr_tmp[now_col]; k < rowptr[now_col + 1]; k++)
                            {
                                int_t tmp_col = colidx[k];
                                if (col_perm[tmp_col] < 0)
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
        for (int_t i = 0; i < n; i++)
        {
            row_scale_value[i] = DBL_MAX;
        }

        for (int_t i = 0; i < n; i++)
        {
            save_tmp[i] = -1;
        }
    }

    for (int_t now_row = 0; now_row < n; now_row++)
    {
        if (finish_flag == n)
        {
            break;
        }
        if (row_perm[now_row] < 0)
        {
            int_t row = now_row;
            int_t queue_length = 0;
            int_t low = n;
            int_t top = n;
            int_t save_index = -1;
            int_t save_row = -1;

            rowptr_tmp[row] = PANGULU_MC64_FLAG;
            calculate_type min_cost = DBL_MAX;
            calculate_type sum_cost = DBL_MAX;

            for (int_t k = rowptr[row]; k < rowptr[row + 1]; k++)
            {
                int_t col = colidx[k];
                calculate_type now_value = fabs_value[k] - col_scale_value[col];
                if (now_value < sum_cost)
                {
                    if (col_perm[col] < 0)
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

            int_t now_queue_length = queue_length;
            queue_length = 0;

            for (int_t k = 0; k < now_queue_length; k++)
            {
                int_t queue_index = queue[k];
                int_t col = colidx[queue_index];
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
                        pangulu_MC64dd(col, n, queue, row_scale_value, save_tmp);
                    }
                    int_t now_col = col_perm[col];
                    ans_queue[now_col] = queue_index;
                    rowptr_tmp[now_col] = row;
                }
            }

            for (int_t k = 0; k < finish_flag; k++)
            {
                if (low == top)
                {

                    if ((queue_length == 0) || (row_scale_value[queue[0]] >= sum_cost))
                    {
                        break;
                    }
                    int_t col = queue[0];
                    min_cost = row_scale_value[col];
                    do
                    {
                        pangulu_MC64ed(&queue_length, n, queue, row_scale_value, save_tmp);
                        queue[--low] = col;
                        save_tmp[col] = low;
                        if (queue_length == 0)
                        {
                            break;
                        }
                        col = queue[0];
                    } while (row_scale_value[col] <= min_cost);
                }
                int_t now_queue_length = queue[top - 1];
                if (row_scale_value[now_queue_length] >= sum_cost)
                {
                    break;
                }
                top--;
                row = col_perm[now_queue_length];
                calculate_type row_sum_max = row_scale_value[now_queue_length] - fabs_value[row_perm[row]] + col_scale_value[now_queue_length];

                for (int_t k = rowptr[row]; k < rowptr[row + 1]; k++)
                {
                    int_t col = colidx[k];
                    if (save_tmp[col] < top)
                    {
                        calculate_type now_value = row_sum_max + fabs_value[k] - col_scale_value[col];
                        if (now_value < sum_cost)
                        {
                            if (col_perm[col] < 0)
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
                                            pangulu_MC64fd(save_tmp[col], &queue_length, n, queue, row_scale_value, save_tmp);
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
                                        pangulu_MC64dd(col, n, queue, row_scale_value, save_tmp);
                                    }

                                    int_t now_col = col_perm[col];
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

                for (int_t k = 0; k < finish_flag; k++)
                {
                    int_t now_rowptr_tmp = rowptr_tmp[row];
                    if (now_rowptr_tmp == PANGULU_MC64_FLAG)
                    {
                        break;
                    }
                    int_t col = colidx[ans_queue[row]];
                    col_perm[col] = now_rowptr_tmp;
                    row_perm[now_rowptr_tmp] = ans_queue[row];
                    row = now_rowptr_tmp;
                }

                for (int_t k = top; k < n; k++)
                {
                    int_t col = queue[k];
                    col_scale_value[col] = col_scale_value[col] + row_scale_value[col] - sum_cost;
                }
            }

            for (int_t k = low; k < n; k++)
            {
                int_t col = queue[k];
                row_scale_value[col] = DBL_MAX;
                save_tmp[col] = -1;
            }

            for (int_t k = 0; k < queue_length; k++)
            {
                int_t col = queue[k];
                row_scale_value[col] = DBL_MAX;
                save_tmp[col] = -1;
            }
        }
    }

    for (int_t i = 0; i < n; i++)
    {
        int_t now_col = row_perm[i];
        if (now_col >= 0)
        {
            row_scale_value[i] = fabs_value[now_col] - col_scale_value[colidx[now_col]];
        }
        else
        {
            row_scale_value[i] = 0.0;
        }
        if (col_perm[i] < 0)
        {
            col_scale_value[i] = 0.0;
        }
    }

    if (finish_flag == n)
    {
        for (int_t i = 0; i < n; i++)
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
        for (int_t i = 0; i < n; i++)
        {
            row_perm[i] = -1;
        }

        int_t ans_queue_length = 0;
        for (int_t i = 0; i < n; i++)
        {
            if (col_perm[i] < 0)
            {
                ans_queue[ans_queue_length++] = i;
            }
            else
            {
                row_perm[col_perm[i]] = i;
            }
        }
        ans_queue_length = 0;
        for (int_t i = 0; i < n; i++)
        {
            if (row_perm[i] < 0)
            {
                col_perm[ans_queue[ans_queue_length++]] = i;
            }
        }
    }

    for (int_t i = 0; i < n; i++)
    {
        col_scale_value[i] = exp(col_scale_value[i]);
    }

    for (int_t i = 0; i < n; i++)
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

// void pangulu_Smatrix_transport_transport_iperm(pangulu_Smatrix *S, pangulu_Smatrix *new_S, int_t *metis_perm)
// {
//     int_t n = S->row;
//     int_t nnz = S->nnz;
//     int_t *rowpointer = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (n + 1));
//     int_32t *columnindex = (int_32t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_32t) * nnz);
//     calculate_type *value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz);
//     for (int_t i = 0; i < n; i++)
//     {
//         int_t row_num = S->rowpointer[i + 1] - S->rowpointer[i];
//         int_t index = metis_perm[i];
//         rowpointer[index + 1] = row_num;
//     }
//     rowpointer[0] = 0;
//     for (int_t i = 0; i < n; i++)
//     {
//         rowpointer[i + 1] += rowpointer[i];
//     }
//     for (int_t i = 0; i < n; i++)
//     {
//         int_t index = metis_perm[i];
//         int_t before_row_begin = S->rowpointer[i];
//         for (int_t row_begin = rowpointer[index]; row_begin < rowpointer[index + 1]; row_begin++, before_row_begin++)
//         {
//             columnindex[row_begin] = metis_perm[S->columnindex[before_row_begin]];
//             value[row_begin] = S->value[before_row_begin];
//         }
//     }
//     new_S->row = n;
//     new_S->column = n;
//     new_S->rowpointer = rowpointer;
//     new_S->columnindex = columnindex;
//     new_S->value = value;
//     new_S->nnz = nnz;
// }

void pangulu_origin_Smatrix_transport_transport_iperm(pangulu_origin_Smatrix *S, pangulu_origin_Smatrix *new_S, int_t *metis_perm)
{
    int_t n = S->row;
    int_t nnz = S->nnz;
    int_t *rowpointer = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (n + 1));
    int_32t *columnindex = (int_32t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_32t) * nnz);
    calculate_type *value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz);
    for (int_t i = 0; i < n; i++)
    {
        int_t row_num = S->rowpointer[i + 1] - S->rowpointer[i];
        int_t index = metis_perm[i];
        rowpointer[index + 1] = row_num;
    }
    rowpointer[0] = 0;
    for (int_t i = 0; i < n; i++)
    {
        rowpointer[i + 1] += rowpointer[i];
    }
    for (int_t i = 0; i < n; i++)
    {
        int_t index = metis_perm[i];
        int_t before_row_begin = S->rowpointer[i];
        for (int_t row_begin = rowpointer[index]; row_begin < rowpointer[index + 1]; row_begin++, before_row_begin++)
        {
            columnindex[row_begin] = metis_perm[S->columnindex[before_row_begin]];
            value[row_begin] = S->value[before_row_begin];
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
#include <parmetis.h>
void pangulu_get_graph_struct(pangulu_origin_Smatrix *S, int_t **xadj_adress, int_t **adjincy_adress)
{
    pangulu_add_diagonal_element(S);
    pangulu_origin_Smatrix_add_CSC(S);
    int_t *xadj = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (S->row + 1));
    xadj[0] = 0;

    for (int_t i = 0; i < S->row; i++)
    {
        int_t index1 = S->rowpointer[i];
        int_t index2 = S->columnpointer[i];
        int_t end1 = S->rowpointer[i + 1];
        int_t end2 = S->columnpointer[i + 1];
        int_t diagonal_flag = 0;
        int_t sum_num = 0;
        int_t col1 = S->columnindex[index1];
        int_t col2 = S->rowindex[index2];
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
                col1 = S->columnindex[index1];
                col2 = S->rowindex[index2];
            }
            else if (col1 < col2)
            {
                if ((diagonal_flag == 0) && (col1 == i))
                {
                    diagonal_flag = 1;
                }
                index1++;
                col1 = S->columnindex[index1];
            }
            else
            {
                if ((diagonal_flag == 0) && (col2 == i))
                {
                    diagonal_flag = 1;
                }
                index2++;
                col2 = S->rowindex[index2];
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
            col1 = S->columnindex[index1];
        }

        while (index2 < end2)
        {
            sum_num++;
            if ((diagonal_flag == 0) && (col2 == i))
            {
                diagonal_flag = 1;
            }
            index2++;
            col2 = S->rowindex[index2];
        }
        if (diagonal_flag == 0)
        {
            printf(PANGULU_E_ROW_DONT_HAVE_DIA);
        }
        xadj[i + 1] = sum_num - diagonal_flag;
    }
    for (int_t i = 0; i < S->row; i++)
    {
        xadj[i + 1] += xadj[i];
    }

    int_t *adjincy = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (xadj[S->row]));

    for (int_t i = 0; i < S->row; i++)
    {
        int_t now_adjincy_index = xadj[i];
        int_t index1 = S->rowpointer[i];
        int_t index2 = S->columnpointer[i];
        int_t end1 = S->rowpointer[i + 1];
        int_t end2 = S->columnpointer[i + 1];
        int_t diagonal_flag = 0;
        int_t col1 = S->columnindex[index1];
        int_t col2 = S->rowindex[index2];
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
                col1 = S->columnindex[index1];
                col2 = S->rowindex[index2];
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
                col1 = S->columnindex[index1];
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
                col2 = S->rowindex[index2];
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
            col1 = S->columnindex[index1];
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
            col2 = S->rowindex[index2];
        }
    }
    *xadj_adress = xadj;
    *adjincy_adress = adjincy;
}

void pangulu_get_graph_struct_MPI(pangulu_origin_Smatrix *S, int_t **xadj_adress, int_t **adjncy_address, int_t **vtxdist_address)
{
    if(RANK==0){
        pangulu_add_diagonal_element(S);
        pangulu_origin_Smatrix_add_CSC(S);
    }


    int_t *xadj;
    int_t *adjncy;
    int sum_rank_size;
    MPI_Comm_size(MPI_COMM_WORLD, &sum_rank_size);
    int_t* vtxdist = (int_t*)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (sum_rank_size+1));

    if(RANK==0){
        vtxdist[0] = 0;
        for(int i=0;i<sum_rank_size;i++){
            vtxdist[i+1] = S->row / sum_rank_size;
            if(i < (S->row % sum_rank_size)){
                vtxdist[i+1]++;
            }
        }
        for(int i=1;i<sum_rank_size;i++){
            vtxdist[i+1] += vtxdist[i];
        }
    }

    MPI_Bcast(vtxdist, sum_rank_size+1, MPI_INT64_T, 0, MPI_COMM_WORLD);

    if(RANK==0){
        xadj = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (S->row + 1));
        xadj[0] = 0;
        for (int_t i = 0; i < S->row; i++)
        {
            int_t index1 = S->rowpointer[i];
            int_t index2 = S->columnpointer[i];
            int_t end1 = S->rowpointer[i + 1];
            int_t end2 = S->columnpointer[i + 1];
            int_t diagonal_flag = 0;
            int_t sum_num = 0;
            int_t col1 = S->columnindex[index1];
            int_t col2 = S->rowindex[index2];
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
                    col1 = S->columnindex[index1];
                    col2 = S->rowindex[index2];
                }
                else if (col1 < col2)
                {
                    if ((diagonal_flag == 0) && (col1 == i))
                    {
                        diagonal_flag = 1;
                    }
                    index1++;
                    col1 = S->columnindex[index1];
                }
                else
                {
                    if ((diagonal_flag == 0) && (col2 == i))
                    {
                        diagonal_flag = 1;
                    }
                    index2++;
                    col2 = S->rowindex[index2];
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
                col1 = S->columnindex[index1];
            }

            while (index2 < end2)
            {
                sum_num++;
                if ((diagonal_flag == 0) && (col2 == i))
                {
                    diagonal_flag = 1;
                }
                index2++;
                col2 = S->rowindex[index2];
            }
            if (diagonal_flag == 0)
            {
                printf(PANGULU_E_ROW_DONT_HAVE_DIA);
            }
            xadj[i + 1] = sum_num - diagonal_flag;
        }
        for (int_t i = 0; i < S->row; i++)
        {
            xadj[i + 1] += xadj[i];
        }

        adjncy = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (xadj[S->row]));

        for (int_t i = 0; i < S->row; i++)
        {
            int_t now_adjncy_index = xadj[i];
            int_t index1 = S->rowpointer[i];
            int_t index2 = S->columnpointer[i];
            int_t end1 = S->rowpointer[i + 1];
            int_t end2 = S->columnpointer[i + 1];
            int_t diagonal_flag = 0;
            int_t col1 = S->columnindex[index1];
            int_t col2 = S->rowindex[index2];
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
                        adjncy[now_adjncy_index] = col1;
                        now_adjncy_index++;
                    }
                    index1++;
                    index2++;
                    col1 = S->columnindex[index1];
                    col2 = S->rowindex[index2];
                }
                else if (col1 < col2)
                {
                    if ((diagonal_flag == 0) && (col1 == i))
                    {
                        diagonal_flag = 1;
                    }
                    else
                    {
                        adjncy[now_adjncy_index] = col1;
                        now_adjncy_index++;
                    }
                    index1++;
                    col1 = S->columnindex[index1];
                }
                else
                {
                    if ((diagonal_flag == 0) && (col2 == i))
                    {
                        diagonal_flag = 1;
                    }
                    else
                    {
                        adjncy[now_adjncy_index] = col2;
                        now_adjncy_index++;
                    }
                    index2++;
                    col2 = S->rowindex[index2];
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
                    adjncy[now_adjncy_index] = col1;
                    now_adjncy_index++;
                }
                index1++;
                col1 = S->columnindex[index1];
            }

            while (index2 < end2)
            {
                if ((diagonal_flag == 0) && (col2 == i))
                {
                    diagonal_flag = 1;
                }
                else
                {
                    adjncy[now_adjncy_index] = col2;
                    now_adjncy_index++;
                }
                index2++;
                col2 = S->rowindex[index2];
            }
        }

        for(int i=1;i<sum_rank_size;i++){
            MPI_Send(xadj + vtxdist[i], vtxdist[i+1]-vtxdist[i] + 1, MPI_INT64_T, i, 1, MPI_COMM_WORLD);
            MPI_Send(
                adjncy + xadj[vtxdist[i]],
                xadj[vtxdist[i+1]] - xadj[vtxdist[i]],
                MPI_INT64_T, i, 2, MPI_COMM_WORLD);
        }
    }else{
        MPI_Status stat;
        xadj = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (vtxdist[RANK+1] - vtxdist[RANK] + 1));
        MPI_Recv(xadj, vtxdist[RANK+1] - vtxdist[RANK] + 1, MPI_INT64_T, 0, 1, MPI_COMM_WORLD, &stat);

        adjncy = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (xadj[vtxdist[RANK+1] - vtxdist[RANK]] - xadj[0]));
        MPI_Recv(adjncy, xadj[vtxdist[RANK+1] - vtxdist[RANK]] - xadj[0], MPI_INT64_T, 0, 2, MPI_COMM_WORLD, &stat);

        int omp_threads_num = 20;
        int_t xadj_first = xadj[0];
        #pragma omp parallel for num_threads(omp_threads_num)
        for(int i=0;i<vtxdist[RANK+1]-vtxdist[RANK]+1;i++){
            xadj[i] -= xadj_first;
        }
    }
    
    *xadj_adress = xadj;
    *adjncy_address = adjncy;
    *vtxdist_address = vtxdist;
}

void pangulu_metis(pangulu_origin_Smatrix *A, int_t **metis_perm)
{

    int_t n = A->row;

    int_t *iperm = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * n);
    int_t *perm = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * n);

    int_t *xadj = NULL;
    int_t *adjincy = NULL;

    pangulu_get_graph_struct(A, &xadj, &adjincy);
    METIS_NodeND(&n, xadj, adjincy, NULL, NULL, perm, iperm);

    pangulu_free(__FILE__, __LINE__, perm);
    pangulu_free(__FILE__, __LINE__, xadj);
    pangulu_free(__FILE__, __LINE__, adjincy);

    *metis_perm = iperm;
    pangulu_free(__FILE__, __LINE__, A->columnpointer);
    pangulu_free(__FILE__, __LINE__, A->CSC_to_CSR_index);
    A->columnpointer = NULL;
    A->rowindex = NULL;
    A->value_CSC = NULL;
}

void pangulu_parmetis(pangulu_origin_Smatrix *A, int_t **metis_perm){
    int_t n = A->row;
    MPI_Bcast(&n, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    int sum_rank_size;
    MPI_Comm_size(MPI_COMM_WORLD, &sum_rank_size);

    // int_t *iperm = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * n);
    int_t *perm = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * n);
    int_t *sizes = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (n + 5));
    int_t *optionsnodeND = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (4));
    optionsnodeND[0] = 0;
    optionsnodeND[1] = 1;
    optionsnodeND[2] = 2;

    int_t *vtxdist = NULL;
    int_t *xadj = NULL;
    int_t *adjncy = NULL;
    int_t numflag = 0;
    MPI_Comm mpi_comm_world = MPI_COMM_WORLD;

    pangulu_get_graph_struct_MPI(A, &xadj, &adjncy, &vtxdist);
    MPI_Barrier(MPI_COMM_WORLD);

    // for(int i=0;i<sum_rank_size;i++){
    //     if(i==RANK){
    //         printf("rank %d\n", RANK);
    //         for(int j=0;j<sum_rank_size+1;j++){
    //             printf("%ld ", vtxdist[j]);
    //         }
    //         printf("\n");
    //         for(int j=0;j<vtxdist[RANK+1]-vtxdist[RANK]+1;j++){
    //             printf("%ld ", xadj[j]);
    //         }
    //         printf("\n");
    //         for(int j=0;j<xadj[vtxdist[RANK+1]-vtxdist[RANK]];j++){
    //             printf("%ld ", adjncy[j]);
    //         }
    //         printf("\n\n");
    //     }
    //     sleep(1);
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }

    ParMETIS_V3_NodeND(
        vtxdist, xadj, adjncy, &numflag, optionsnodeND, perm,
        sizes, &mpi_comm_world);
    MPI_Abort(MPI_COMM_WORLD, 0);
    

    
    if(RANK==0){
        MPI_Status stat;
        for(int i=1;i<sum_rank_size;i++){
            int_t offset = perm[vtxdist[i]];
            MPI_Recv(perm + vtxdist[i], vtxdist[i+1] - vtxdist[i] + 1, MPI_INT64_T, i, 3, MPI_COMM_WORLD, &stat);
            for(int j=vtxdist[i];j<=vtxdist[i+1];j++){
                vtxdist[i]+=offset;
            }
        }
    }else{
        MPI_Send(perm, vtxdist[RANK+1] - vtxdist[RANK] + 1, MPI_INT64_T, 0, 3, MPI_COMM_WORLD);
    }


    if(RANK==0){
        *metis_perm = perm;
    }else{
        *metis_perm = NULL;
    }
}

#endif
void pangulu_reorder_vector_X_tran(pangulu_block_Smatrix *block_Smatrix,
                                   pangulu_vector *X_origin,
                                   pangulu_vector *X_trans)
{
    int_t n = X_origin->row;
    // tran X col
    int_t *metis_perm = block_Smatrix->metis_perm;
    calculate_type *col_scale = block_Smatrix->col_scale;

    for (int_t i = 0; i < n; i++)
    {
        int_t now_col = metis_perm[i];
        X_trans->value[i] = X_origin->value[now_col] * col_scale[i];
    }
}

void pangulu_reorder_vector_B_tran(pangulu_block_Smatrix *block_Smatrix,
                                   pangulu_vector *B_origin,
                                   pangulu_vector *B_trans)
{
    int_t n = B_origin->row;
    // tran B row
    int_t *row_perm = block_Smatrix->row_perm;
    int_t *metis_perm = block_Smatrix->metis_perm;
    calculate_type *row_scale = block_Smatrix->row_scale;

    for (int_t i = 0; i < n; i++)
    {
        int_t index_row = row_perm[i];
        int_t now_row = metis_perm[index_row];
        B_trans->value[now_row] = B_origin->value[i] * row_scale[i];
    }
}

void pangulu_reorder(pangulu_block_Smatrix *block_Smatrix,
                     pangulu_origin_Smatrix *origin_matrix,
                     pangulu_origin_Smatrix *reorder_matrix)
{
    int_t n = origin_matrix->row;
    int_t nnz = origin_matrix->nnz;
    int_t *perm = NULL;
    int_t *iperm = NULL;
    int_t *metis_perm = NULL;
    calculate_type *row_scale = NULL;
    calculate_type *col_scale = NULL;
    pangulu_origin_Smatrix *tmp;
    if(RANK==0){
#if defined(PANGULU_MC64)
        pangulu_mc64(origin_matrix, &perm, &iperm, &row_scale, &col_scale);
#else
        perm = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * n);
        iperm = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * n);
        row_scale = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);
        col_scale = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * n);

        for (int_t i = 0; i < n; i++)
        {
            perm[i] = i;
        }

        for (int_t i = 0; i < n; i++)
        {
            iperm[i] = i;
        }

        for (int_t i = 0; i < n; i++)
        {
            row_scale[i] = 1.0;
        }

        for (int_t i = 0; i < n; i++)
        {
            col_scale[i] = 1.0;
        }

#endif
        tmp = (pangulu_origin_Smatrix *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_origin_Smatrix));
        pangulu_init_pangulu_origin_Smatrix(tmp);
        tmp->row = n;
        tmp->column = n;
        tmp->nnz = nnz;
        tmp->rowpointer = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * (n + 1));
        tmp->columnindex = (idx_int *)pangulu_malloc(__FILE__, __LINE__, sizeof(idx_int) * nnz);
        tmp->value = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nnz);

        tmp->rowpointer[0] = 0;

        for (int_t i = 0; i < n; i++)
        {
            int_t row = perm[i];
            tmp->rowpointer[row + 1] = origin_matrix->rowpointer[i + 1] - origin_matrix->rowpointer[i];
        }

        for (int_t i = 0; i < n; i++)
        {
            tmp->rowpointer[i + 1] += tmp->rowpointer[i];
        }

        for (int_t i = 0; i < n; i++)
        {
            int_t row = perm[i];
            calculate_type rs = row_scale[i];
            int_t tmp_index = tmp->rowpointer[row];
            for (int_t j = origin_matrix->rowpointer[i]; j < origin_matrix->rowpointer[i + 1]; j++)
            {
                int_t col = origin_matrix->columnindex[j];
                tmp->columnindex[tmp_index] = col;
                tmp->value[tmp_index] = (origin_matrix->value[j] * rs * col_scale[col]);
                if (col == row)
                {
                    if (fabs(fabs(tmp->value[tmp_index]) - 1.0) > 1e-10)
                    {
                    }
                }
                tmp_index++;
            }
        }
    }

#ifdef METIS
    if(RANK==0){
        pangulu_metis(tmp, &metis_perm);
    }
    // MPI_Barrier(MPI_COMM_WORLD);
    // pangulu_parmetis(tmp, &metis_perm);
    // MPI_Barrier(MPI_COMM_WORLD);
#else
    if(RANK==0){
        pangulu_add_diagonal_element(tmp);
        metis_perm = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * n);
        for (int_t i = 0; i < n; i++)
        {
            metis_perm[i] = i;
        }
    }

#endif
    if(RANK==0){
        //pangulu_display_pangulu_Smatrix(tmp);
        pangulu_origin_Smatrix_transport_transport_iperm(tmp, reorder_matrix, metis_perm);
        pangulu_sort_pangulu_origin_Smatrix(reorder_matrix);

        block_Smatrix->row_perm = perm;
        block_Smatrix->col_perm = iperm;
        block_Smatrix->row_scale = row_scale;
        block_Smatrix->col_scale = col_scale;
        block_Smatrix->metis_perm = metis_perm;

        // pangulu_display_pangulu_Smatrix(origin_matrix);
        // pangulu_display_pangulu_Smatrix(tmp);
        // pangulu_display_pangulu_Smatrix(reorder_matrix);

        pangulu_free(__FILE__, __LINE__, tmp->rowpointer);
        pangulu_free(__FILE__, __LINE__, tmp->columnindex);
        pangulu_free(__FILE__, __LINE__, tmp->value);
        pangulu_free(__FILE__, __LINE__, tmp);
    }
}

#endif