#ifndef PANGULU_UTILS_H
#define PANGULU_UTILS_H

#include "mmio.h"
#include "mmio_highlevel.h"
#include "pangulu_common.h"

#include "pangulu_time.h"

#include <string.h>
#include "pangulu_mpi.h"
#include "pangulu_malloc.h"

#ifdef GPU_OPEN
#include "pangulu_cuda_interface.h"
#endif

#include <getopt.h>

#define PANGULU_MAX(A, B) \
    (((A) > (B)) ? (A) : (B))
#define PANGULU_MIN(A, B) \
    (((A) < (B)) ? (A) : (B))
#define PANGULU_ABS(A) \
    ((A > 0) ? (A) : (-A))

#define ZERO_ELEMENT 1e-12

void pangulu_get_common(pangulu_common *common,
                        int_t ARGC, char **ARGV, int_32t size)
{
    int_t c;
    extern char *optarg;
    common->NB = 0;
    common->P = 0;
    common->Q = 0;
    common->sum_rank_size = size;
    common->omp_thread = 1;
    while ((c = getopt(ARGC, ARGV, "NB:F:O:")) != EOF)
    {
        switch (c)
        {
        case 'B':
            common->NB = atoi(optarg);
            continue;
        case 'O':
            common->NB = atoi(optarg);
            continue;
        case 'F':
            common->file_name = (char *)pangulu_malloc((strlen(optarg) + 10) * sizeof(char));
            strcpy(common->file_name, optarg);
            continue;
        }
    }

    int_t tmp_p = sqrt(common->sum_rank_size);
    while (((common->sum_rank_size) % tmp_p) != 0)
    {
        tmp_p--;
    }

    common->P = tmp_p;
    common->Q = common->sum_rank_size / tmp_p;
    if (common->rank == 0)
    {
        printf("MPI Processes %d NB is %d\n", common->P * common->Q, common->NB);
        printf("Matrix is %s\n", common->file_name);
        fflush(NULL);
    }
    if (common->sum_rank_size <= 0)
    {
        printf("error don't any process sum_rank_size is %d\n", common->sum_rank_size);
        exit(2);
    }
    if ((common->NB) == 0)
    {
        printf("error the NB is 0\n");
        exit(4);
    }
}

void pangulu_init_pangulu_block_Smatrix(pangulu_block_Smatrix *block_Smatrix)
{
    // reorder array
    block_Smatrix->row_perm = NULL;
    block_Smatrix->col_perm = NULL;
    block_Smatrix->metis_perm = NULL;
    block_Smatrix->row_scale = NULL;
    block_Smatrix->col_scale = NULL;

    // symbolic
    block_Smatrix->symbolic_rowpointer = NULL;
    block_Smatrix->symbolic_columnindex = NULL;

    // LU
    block_Smatrix->block_Smatrix_nnzA_num = NULL;
    block_Smatrix->block_Smatrix_non_zero_vector_L = NULL;
    block_Smatrix->block_Smatrix_non_zero_vector_U = NULL;
    block_Smatrix->mapper_Big_pangulu_Smatrix = NULL;
    block_Smatrix->Big_pangulu_Smatrix_rowpointer = NULL;
    block_Smatrix->Big_pangulu_Smatrix_columnindex = NULL;
    block_Smatrix->Big_pangulu_Smatrix_value = NULL;
    block_Smatrix->Big_pangulu_Smatrix_copy_value = NULL;
    block_Smatrix->L_pangulu_Smatrix_columnpointer = NULL;
    block_Smatrix->L_pangulu_Smatrix_rowindex = NULL;
    block_Smatrix->L_Smatrix_nzz = 0;
    block_Smatrix->L_pangulu_Smatrix_value = NULL;
    block_Smatrix->U_pangulu_Smatrix_rowpointer = NULL;
    block_Smatrix->U_pangulu_Smatrix_columnindex = NULL;
    block_Smatrix->U_Smatrix_nzz = 0;
    block_Smatrix->U_pangulu_Smatrix_value = NULL;
    block_Smatrix->mapper_diagonal = NULL;
    block_Smatrix->diagonal_Smatrix_L = NULL;
    block_Smatrix->diagonal_Smatrix_U = NULL;
    block_Smatrix->calculate_L = NULL;
    block_Smatrix->calculate_U = NULL;
    block_Smatrix->calculate_X = NULL;

    block_Smatrix->task_level_length = 0;
    block_Smatrix->task_level_num = NULL;
    block_Smatrix->mapper_LU = NULL;
    block_Smatrix->task_flag_id = NULL;
    block_Smatrix->heap = NULL;
    block_Smatrix->now_level_L_length = NULL;
    block_Smatrix->now_level_U_length = NULL;
    block_Smatrix->save_now_level_L = NULL;
    block_Smatrix->save_now_level_U = NULL;
    block_Smatrix->send_flag = NULL;
    block_Smatrix->send_diagonal_flag_L = NULL;
    block_Smatrix->send_diagonal_flag_U = NULL;
    block_Smatrix->grid_process_id = NULL;
    block_Smatrix->save_send_rank_flag = NULL;
    block_Smatrix->level_task_rank_id = NULL;
    block_Smatrix->real_matrix_flag = NULL;
    block_Smatrix->sum_flag_block_num = NULL;
    block_Smatrix->receive_level_num = NULL;
    block_Smatrix->save_tmp = NULL;

    block_Smatrix->level_index = NULL;
    block_Smatrix->level_index_reverse = NULL;
    block_Smatrix->mapper_mpi = NULL;
    block_Smatrix->mapper_mpi_reverse = NULL;
    block_Smatrix->mpi_level_num = NULL;

    block_Smatrix->flag_save_L = NULL;
    block_Smatrix->flag_save_U = NULL;
    block_Smatrix->flag_dignon_L = NULL;
    block_Smatrix->flag_dignon_U = NULL;

#ifdef OVERLAP
    block_Smatrix->run_bsem1 = NULL;
    block_Smatrix->run_bsem2 = NULL;

#endif

    // sptrsv  malloc
    block_Smatrix->Big_row_vector = NULL;
    block_Smatrix->Big_col_vector = NULL;
    block_Smatrix->diagonal_flag = NULL;
    block_Smatrix->L_row_task_nnz = NULL;
    block_Smatrix->L_col_task_nnz = NULL;
    block_Smatrix->U_row_task_nnz = NULL;
    block_Smatrix->U_col_task_nnz = NULL;
    block_Smatrix->sptrsv_heap = NULL;
    block_Smatrix->save_vector = NULL;
    block_Smatrix->L_send_flag = NULL;
    block_Smatrix->U_send_flag = NULL;
    block_Smatrix->L_sptrsv_task_columnpointer = NULL;
    block_Smatrix->L_sptrsv_task_rowindex = NULL;
    block_Smatrix->U_sptrsv_task_columnpointer = NULL;
    block_Smatrix->U_sptrsv_task_rowindex = NULL;
}

void pangulu_init_pangulu_Smatrix(pangulu_Smatrix *S)
{
    S->value = NULL;
    S->value_CSC = NULL;
    S->CSR_to_CSC_index = NULL;
    S->CSC_to_CSR_index = NULL;
    S->rowpointer = NULL;
    S->columnindex = NULL;
    S->columnpointer = NULL;
    S->rowindex = NULL;
    S->column = 0;
    S->row = 0;
    S->nnz = 0;

    S->nnzU = NULL;
    S->bin_rowpointer = NULL;
    S->bin_rowindex = NULL;

#ifdef GPU_OPEN
    S->CUDA_rowpointer = NULL;
    S->CUDA_columnindex = NULL;
    S->CUDA_value = NULL;
    S->CUDA_nnzU = NULL;
    S->CUDA_bin_rowpointer = NULL;
    S->CUDA_bin_rowindex = NULL;
#else
    S->num_lev = 0;
    S->level_idx = NULL;
    S->level_size = NULL;

#endif
}

void pangulu_binary_write_csc_pangulu_Smatrix(pangulu_Smatrix *S, char *filename)
{
    int_t nnz = S->nnz;
    int_t nrow = S->row;
    int_t ncol = S->column;
    int_t *columnpointer = S->columnpointer;
    idx_int *rowindex = S->rowindex;
    FILE *fp = fopen(filename, "w+");
    if (fp != NULL)
    {
        fwrite(&nrow, sizeof(int_t), 1, fp);
        fwrite(&ncol, sizeof(int_t), 1, fp);
        fwrite(&nnz, sizeof(int_t), 1, fp);
        fwrite(columnpointer, sizeof(int_t), nrow + 1, fp);
        fwrite(rowindex, sizeof(idx_int), nnz, fp);
    }
    else
    {
        printf("open error\n");
        return;
    }
    fclose(fp);
}

void pangulu_binary_write_csc_value_pangulu_Smatrix(pangulu_Smatrix *S, char *filename)
{
    int_t nnz = S->nnz;
    int_t nrow = S->row;
    int_t ncol = S->column;
    int_t *columnpointer = S->columnpointer;
    idx_int *rowindex = S->rowindex;
    calculate_type *value_CSC = S->value_CSC;
    FILE *fp = fopen(filename, "w+");
    if (fp != NULL)
    {
        fwrite(&nrow, sizeof(int_t), 1, fp);
        fwrite(&ncol, sizeof(int_t), 1, fp);
        fwrite(&nnz, sizeof(int_t), 1, fp);
        fwrite(columnpointer, sizeof(int_t), nrow + 1, fp);
        fwrite(rowindex, sizeof(idx_int), nnz, fp);
        fwrite(value_CSC, sizeof(calculate_type), nnz, fp);
    }
    else
    {
        printf("open error\n");
        return;
    }

    fclose(fp);
}

void pangulu_binary_write_csv_pangulu_Smatrix(pangulu_Smatrix *S, char *filename)
{
    int_t nnz = S->nnz;
    int_t nrow = S->row;
    int_t ncol = S->column;
    printf("write binary file %s\n", filename);
    int_t *rowpointer = S->rowpointer;
    idx_int *columnindex = S->columnindex;
    calculate_type *value = S->value;
    FILE *fp = fopen(filename, "w+");
    if (fp != NULL)
    {
        printf("write nrow is %ld ncol is %ld nnz is %ld\n", nrow, ncol, nnz);
        fwrite(&nrow, sizeof(int_t), 1, fp);
        fwrite(&ncol, sizeof(int_t), 1, fp);
        fwrite(&nnz, sizeof(int_t), 1, fp);
        fwrite(rowpointer, sizeof(int_t), nrow + 1, fp);
        fwrite(columnindex, sizeof(idx_int), nnz, fp);
        fwrite(value, sizeof(calculate_type), nnz, fp);
    }
    else
    {
        printf("open error\n");
        return;
    }

    fclose(fp);
}

int_t pangulu_binary_read_csv_pangulu_Smatrix(pangulu_Smatrix *S, char *filename)
{
    int_t nrow, ncol;
    int_t nnz;
    FILE *fp = NULL;
    fp = fopen(filename, "r");
    if (fp == NULL)
    {
        return -1;
    }
    if (1 != fread(&nrow, sizeof(int_t), 1, fp))
    {
        printf("read nrow error\n");
        return -1;
    }

    if (1 != fread(&ncol, sizeof(int_t), 1, fp))
    {
        printf("read ncol error\n");
        return -1;
    }
    if (1 != fread(&nnz, sizeof(int_t), 1, fp))
    {
        printf("read nnz error\n");
        return -1;
    }
    int_t *rowpointer = (int_t *)pangulu_malloc(sizeof(int_t) * (nrow + 1));
    if ((unsigned int_t)(nrow + 1) != fread(rowpointer, sizeof(int_t), nrow + 1, fp))
    {
        printf("read rowptr error\n");
        return -1;
    }
    // printf("nnz = %ld\n",nnz);
    idx_int *columnindex = (idx_int *)pangulu_malloc(sizeof(idx_int) * nnz);
    calculate_type *value = (calculate_type *)pangulu_malloc(sizeof(calculate_type) * nnz);

    if ((unsigned int_t)nnz != fread(columnindex, sizeof(idx_int), nnz, fp))
    {
        printf("read columnindex error\n");
        return -1;
    }
    if ((unsigned int_t)nnz != fread(value, sizeof(calculate_type), nnz, fp))
    {
        printf("read value error\n");
        return -1;
    }

    S->row = nrow;
    S->column = ncol;
    S->rowpointer = rowpointer;
    S->columnindex = columnindex;
    S->value = value;
    S->nnz = nnz;
    fclose(fp);

    return 0;
}

void pangulu_read_pangulu_Smatrix(pangulu_Smatrix *S,
                                  char *filename)
{
    int_t isSymmeticeR;
    int_t nnz;
    int_t nrow, ncol;
    char binary_name[512];
    int_t file_length = strlen(filename);
    strcpy(binary_name, filename);
    binary_name[file_length - 1] = 'v';
    binary_name[file_length - 2] = 's';
    binary_name[file_length - 3] = 't';
    int_t flag = pangulu_binary_read_csv_pangulu_Smatrix(S, binary_name);
    if (flag == -1)
    {
        // printf("matrix is %s\n",filename);
        mmio_info(&nrow, &ncol, &nnz, &isSymmeticeR, filename);
        int_t *rowpointer = (int_t *)pangulu_malloc(sizeof(int_t) * (nrow + 1));
        idx_int *columnindex = (idx_int *)pangulu_malloc(sizeof(idx_int) * nnz);
        VALUE_TYPE *save_value = (VALUE_TYPE *)pangulu_malloc(sizeof(VALUE_TYPE) * nnz);
        mmio_data_csr(rowpointer, columnindex, save_value, filename);
        S->row = nrow;
        S->column = ncol;
        S->rowpointer = rowpointer;
        S->columnindex = columnindex;
        calculate_type *value = (calculate_type *)pangulu_malloc(sizeof(calculate_type) * nnz);
        for (int_t i = 0; i < nnz; i++)
        {
            value[i] = (calculate_type)(save_value[i]);
        }
        free(save_value);
        S->nnz = nnz;
        S->value = value;
        pangulu_binary_write_csv_pangulu_Smatrix(S, binary_name);
    }
}

void pangulu_write_pangulu_Smatrix(pangulu_Smatrix *S, char *string)
{

    FILE *fp = fopen(string, "w+");

    if (fp == NULL)
    {
        printf("eeror the file don't open\n");
        return;
    }
    int_t column = S->column;
    int_t row = S->row;
    fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(fp, "%ld ", row);
    fprintf(fp, "%ld ", column);
    fprintf(fp, "%ld\n", S->rowpointer[row]);

    for (int_t i = 0; i < row; i++)
    {
        for (int_t j = S->rowpointer[i]; j < S->rowpointer[i + 1]; j++)
        {
            fprintf(fp, "%d %ld %f\n", S->columnindex[j] + 1, i + 1, S->value[j]);
        }
    }
    fclose(fp);
}

void pangulu_time_start(pangulu_common *common)
{
    gettimeofday(&(common->start_time), NULL);
}

void pangulu_time_stop(pangulu_common *common)
{
    gettimeofday(&(common->stop_time), NULL);
}

int_t pangulu_mapper_A_Smatrix(int_t row, int_t col, int_t *mapper_A, int_t col_length, int_t P, int_t Q)
{
    int_t mapper_index = mapper_A[(row / P) * col_length + (col / Q)];
    return mapper_index;
}

void pangulu_display_pangulu_Smatrix(pangulu_Smatrix *S)
{
    printf("------------------\n\n\n");
    if (S == NULL)
    {
        printf("\nno i am null\n");
        return;
    }
    printf("row is %ld column is %ld\n", S->row, S->column);
    printf("rowpointer:");
    for (int_t i = 0; i < S->row + 1; i++)
    {
        printf("%ld ", S->rowpointer[i]);
    }
    printf("\n");
    printf("columnindex:\n");
    for (int_t i = 0; i < S->row; i++)
    {
        for (int_t j = S->rowpointer[i]; j < S->rowpointer[i + 1]; j++)
        {
            printf("%d ", S->columnindex[j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("value:\n");
    for (int_t i = 0; i < S->row; i++)
    {
        for (int_t j = S->rowpointer[i]; j < S->rowpointer[i + 1]; j++)
        {
            printf("%lf ", S->value[j]);
        }
        printf("\n");
    }
    printf("\n\n\n--------------------");
}
void pangulu_memcpy_zero_pangulu_Smatrix_CSC_value(pangulu_Smatrix *S)
{
    for (int_t i = 0; i < S->nnz; i++)
    {
        S->value_CSC[i] = 0.0;
    }
}
void pangulu_memcpy_zero_pangulu_Smatrix_CSR_value(pangulu_Smatrix *S)
{
    for (int_t i = 0; i < S->nnz; i++)
    {
        S->value[i] = 0.0;
    }
}
void pangulu_display_pangulu_Smatrix_CSC(pangulu_Smatrix *S)
{
    printf("------------------\n\n\n");
    if (S == NULL)
    {
        printf("\nno i am null\n");
        return;
    }
    printf("row is %ld column is %ld\n", S->row, S->column);
    printf("columnpointer:");
    for (int_t i = 0; i < S->row + 1; i++)
    {
        printf("%ld ", S->columnpointer[i]);
    }
    printf("\n");
    printf("rowindex:\n");
    for (int_t i = 0; i < S->row; i++)
    {
        for (int_t j = S->columnpointer[i]; j < S->columnpointer[i + 1]; j++)
        {
            printf("%d ", S->rowindex[j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("value_csc:\n");
    for (int_t i = 0; i < S->row; i++)
    {
        // printf("i %ld\n",i);
        for (int_t j = S->columnpointer[i]; j < S->columnpointer[i + 1]; j++)
        {
            printf("%lf ", S->value_CSC[j]);
        }
        printf("\n");
    }
    printf("\n\n\n--------------------");
}

void pangulu_display_pangulu_vector(pangulu_vector *V)
{
    printf("this vector length is %ld\n", V->row);
    for (int_t i = 0; i < V->row; i++)
    {
        printf("%ld %lf\n", i, V->value[i]);
    }
}

double pangulu_get_spend_time(pangulu_common *common)
{
    double time = (common->stop_time.tv_sec - common->start_time.tv_sec) * 1000.0 + (common->stop_time.tv_usec - common->start_time.tv_usec) / 1000.0;
    return time / 1000.0;
}

void pangulu_transport_pangulu_Smatrix_CSC_to_CSR(pangulu_Smatrix *S)
{
    for (int_t i = 0; i < S->nnz; i++)
    {
        int_t index = S->CSC_to_CSR_index[i];
        S->value[index] = S->value_CSC[i];
    }
}
void pangulu_transport_pangulu_Smatrix_CSR_to_CSC(pangulu_Smatrix *S)
{
    for (int_t i = 0; i < S->nnz; i++)
    {
        int_t index = S->CSC_to_CSR_index[i];
        S->value_CSC[i] = S->value[index];
    }
}

void pangulu_pangulu_Smatrix_memcpy_rowpointer_CSR(pangulu_Smatrix *S, pangulu_Smatrix *copy_S)
{
    int_t n = S->row;
    for (int_t i = 0; i < (n + 1); i++)
    {
        S->rowpointer[i] = copy_S->rowpointer[i];
    }
}

void pangulu_pangulu_Smatrix_memcpy_value_CSR(pangulu_Smatrix *S, pangulu_Smatrix *copy_S)
{
    for (int_t i = 0; i < S->nnz; i++)
    {
        S->value[i] = copy_S->value[i];
    }
}

void pangulu_pangulu_Smatrix_memcpy_value_CSR_copy_length(pangulu_Smatrix *S, pangulu_Smatrix *copy_S)
{
    memcpy(S->value, copy_S->value, sizeof(calculate_type) * copy_S->nnz);
}

void pangulu_pangulu_Smatrix_memcpy_value_CSC_copy_length(pangulu_Smatrix *S, pangulu_Smatrix *copy_S)
{
    memcpy(S->value_CSC, copy_S->value_CSC, sizeof(calculate_type) * copy_S->nnz);
}

void pangulu_pangulu_Smatrix_memcpy_struct_CSC(pangulu_Smatrix *S, pangulu_Smatrix *copy_S)
{
    S->column = copy_S->column;
    S->row = copy_S->row;
    S->nnz = copy_S->nnz;
    for (int_t i = 0; i < S->column + 1; i++)
    {
        S->columnpointer[i] = copy_S->columnpointer[i];
    }
    for (int_t i = 0; i < S->nnz; i++)
    {
        S->rowindex[i] = copy_S->rowindex[i];
    }
}

void pangulu_pangulu_Smatrix_memcpy_columnpointer_CSC(pangulu_Smatrix *S, pangulu_Smatrix *copy_S)
{
    memcpy(S->columnpointer, copy_S->columnpointer, sizeof(int_t) * (copy_S->row + 1));
}

void pangulu_pangulu_Smatrix_memcpy_value_CSC(pangulu_Smatrix *S, pangulu_Smatrix *copy_S)
{
    for (int_t i = 0; i < S->nnz; i++)
    {

        S->value_CSC[i] = copy_S->value_CSC[i];
    }
}

void pangulu_pangulu_Smatrix_memcpy_complete_CSC(pangulu_Smatrix *S, pangulu_Smatrix *copy_S)
{
    pangulu_pangulu_Smatrix_memcpy_struct_CSC(S, copy_S);
    pangulu_pangulu_Smatrix_memcpy_value_CSC(S, copy_S);
}

void pangulu_pangulu_Smatrix_multiple_pangulu_vector_CSR(pangulu_Smatrix *A,
                                                         pangulu_vector *X,
                                                         pangulu_vector *B)
{
    int_t n = A->row;
    calculate_type *X_value = X->value;
    calculate_type *B_value = B->value;
    for (int_t i = 0; i < n; i++)
    {
        B_value[i] = 0.0;
        for (int_t j = A->rowpointer[i]; j < A->rowpointer[i + 1]; j++)
        {
            int_t col = A->columnindex[j];
            B_value[i] += A->value[j] * X_value[col];
        }
    }
}

void pangulu_pangulu_Smatrix_multiple_pangulu_vector(pangulu_Smatrix *A,
                                                     pangulu_vector *X,
                                                     pangulu_vector *B)
{
    int_t n = A->row;
    calculate_type *X_value = X->value;
    calculate_type *B_value = B->value;
    for(int_t i = 0 ; i < n; i++){
        B_value[i] = 0.0;
    }
    for (int_t i = 0; i < n; i++)
    {
        for (int_t j = A->rowpointer[i]; j < A->rowpointer[i + 1]; j++)
        {
            int_t row = A->columnindex[j];
            B_value[row] += A->value[j] * X_value[i];
        }
    }
}

void pangulu_pangulu_Smatrix_multiply_block_pangulu_vector_CSR(pangulu_Smatrix *A,
                                                               calculate_type *X,
                                                               calculate_type *B)
{
    int_t n = A->row;
    for (int_t i = 0; i < n; i++)
    {
        for (int_t j = A->rowpointer[i]; j < A->rowpointer[i + 1]; j++)
        {
            int_t col = A->columnindex[j];
            B[i] += A->value[j] * X[col];
        }
    }
}

void pangulu_pangulu_Smatrix_multiple_pangulu_vector_CSC(pangulu_Smatrix *A,
                                                         pangulu_vector *X,
                                                         pangulu_vector *B)
{
    int_t n = A->column;
    calculate_type *X_value = X->value;
    calculate_type *B_value = B->value;
    for (int_t i = 0; i < n; i++)
    {
        for (int_t j = A->columnpointer[i]; j < A->columnpointer[i + 1]; j++)
        {
            int_t row = A->rowindex[j];
            B_value[row] += A->value_CSC[j] * X_value[i];
        }
    }
}

void pangulu_pangulu_Smatrix_multiply_block_pangulu_vector_CSC(pangulu_Smatrix *A,
                                                               calculate_type *X,
                                                               calculate_type *B)
{
    int_t n = A->column;
    for (int_t i = 0; i < n; i++)
    {
        for (int_t j = A->columnpointer[i]; j < A->columnpointer[i + 1]; j++)
        {
            int_t row = A->rowindex[j];
            B[row] += A->value_CSC[j] * X[i];
        }
    }
}

void pangulu_read_pangulu_vector(pangulu_vector *X, int_t n, char *filename)
{
    FILE *fp;
    if (filename != NULL)
    {
        fp = fopen(filename, "r");
        if (fp != NULL)
        {
            int_t vector_n = 0;
            int rt = fscanf(fp, "%ld", &vector_n);
            if (vector_n >= n)
            {
                printf("read vector is %s\n", filename);
                X->value = (calculate_type *)pangulu_malloc(sizeof(calculate_type) * n);
                for (int_t i = 0; i < n; i++)
                {
                    rt = fscanf(fp, "%lf", &(X->value[i]));
                }
                X->row = n;
                fclose(fp);
                return;
            }
            else
            {
                printf("read vector is %s row is %ld don't have enough element\n", filename, vector_n);
                fclose(fp);
            }
        }
    }
    X->value = (calculate_type *)pangulu_malloc(sizeof(calculate_type) * n);
    for (int_t i = 0; i < n; i++)
    {
        X->value[i] = 2.0;
    }
    X->row = n;
}

void pangulu_get_init_value_pangulu_vector(pangulu_vector *X, int_t n)
{
    X->value = (calculate_type *)pangulu_malloc(sizeof(calculate_type) * n);
    for (int_t i = 0; i < n; i++)
    {
       //X->value[i] = (calculate_type)i;
       X->value[i] = 2.0;
    }
    X->row = n;
}

void pangulu_init_pangulu_vector(pangulu_vector *B, int_t n)
{
    B->value = (calculate_type *)pangulu_malloc(sizeof(calculate_type) * n);
    for (int_t i = 0; i < n; i++)
    {
        B->value[i] = (calculate_type)0.0;
    }
    B->row = n;
}

void pangulu_zero_pangulu_vector(pangulu_vector *v)
{
    for (int i = 0; i < v->row; i++)
    {

        v->value[i] = 0.0;
    }
}

void pangulu_add_diagonal_element(pangulu_Smatrix *S){
        int_t diagonal_add=0;
        int_t n=S->row;
        int_t *new_rowpointer=(int_t *)pangulu_malloc(sizeof(int_t)*(n+5));
        for(int_t i=0;i<n;i++){
            int_t flag=0;
            for(int_t j=S->rowpointer[i];j<S->rowpointer[i+1];j++){
                if(S->columnindex[j]==i){
                    flag=1;
                    break;
                }
            }
            new_rowpointer[i]=S->rowpointer[i]+diagonal_add;
            diagonal_add+=(!flag);
        }
        // if(diagonal_add==0){
        //     free(new_rowpointer);
        //     return ;
        // }
        new_rowpointer[n]=S->rowpointer[n]+diagonal_add;
        
        int_32t *new_columnindex=(int_32t *)pangulu_malloc(sizeof(int_32t)*new_rowpointer[n]);
        calculate_type *new_value=(calculate_type *)pangulu_malloc(sizeof(calculate_type)*new_rowpointer[n]);
        
        for(int_t i=0;i<n;i++){
            if((new_rowpointer[i+1]-new_rowpointer[i])==(S->rowpointer[i+1]-S->rowpointer[i])){
                for(int_t j=new_rowpointer[i],k=S->rowpointer[i];j<new_rowpointer[i+1];j++,k++){
                    new_columnindex[j]=S->columnindex[k];
                    new_value[j]=S->value[k];
                }
            }
            else{
                int_t flag=0;
                for(int_t j=new_rowpointer[i],k=S->rowpointer[i];k<S->rowpointer[i+1];j++,k++){
                    if(S->columnindex[k]<i){
                        new_columnindex[j]=S->columnindex[k];
                        new_value[j]=S->value[k];
                    }
                    else if(S->columnindex[k]>i){
                        if(flag==0){
                            new_columnindex[j]=i;
                            new_value[j]=ZERO_ELEMENT;
                            k--;
                            flag=1;
                        }
                        else{
                            new_columnindex[j]=S->columnindex[k];
                            new_value[j]=S->value[k];
                        }
                        
                    }
                    else{
                        printf("error\n");
                    }
                    
                }
                if(flag==0){
                    new_columnindex[new_rowpointer[i+1]-1]=i;
                    new_value[new_rowpointer[i+1]-1]=ZERO_ELEMENT;
                }
            }
        }

        free(S->rowpointer);
        free(S->columnindex);
        free(S->value);
        S->rowpointer=new_rowpointer;
        S->columnindex=new_columnindex;
        S->value=new_value;
        S->nnz=new_rowpointer[n];
    }


void pangulu_send_pangulu_vector_value(pangulu_vector *S,
                                       int_t send_id, int_t signal, int_t vector_length)
{
    MPI_Send(S->value, vector_length, MPI_VAL_TYPE, send_id, signal, MPI_COMM_WORLD);
}

void pangulu_isend_pangulu_vector_value(pangulu_vector *S,
                                        int send_id, int signal, int vector_length)
{
    MPI_Request req;
    MPI_Isend(S->value, vector_length, MPI_VAL_TYPE, send_id, signal, MPI_COMM_WORLD, &req);
}

void pangulu_recv_pangulu_vector_value(pangulu_vector *S, int_t receive_id, int_t signal, int_t vector_length)
{
    MPI_Status status;
    MPI_Recv(S->value, vector_length, MPI_VAL_TYPE, receive_id, signal, MPI_COMM_WORLD, &status);
}

void pangulu_init_vector_int(int_t *vector, int_t length)
{
    for (int_t i = 0; i < length; i++)
    {
        vector[i] = 0;
    }
}

int_t pangulu_choose_pivot(int_t i, int_t j)
{
    return (i + j) / 2;
}

void pangulu_swap_int(int_t *a, int_t *b)
{
    int_t tmp = *a;
    *a = *b;
    *b = tmp;
}

void pangulu_quicksort_keyval(int_t *key, int_t *val, int_t start, int_t end)
{
    int_t pivot;
    int_t i, j, k;

    if (start < end)
    {
        k = pangulu_choose_pivot(start, end);
        pangulu_swap_int(&key[start], &key[k]);
        pangulu_swap_int(&val[start], &val[k]);
        pivot = key[start];

        i = start + 1;
        j = end;
        while (i <= j)
        {
            while ((i <= end) && (key[i] <= pivot))
                i++;
            while ((j >= start) && (key[j] > pivot))
                j--;
            if (i < j)
            {
                pangulu_swap_int(&key[i], &key[j]);
                pangulu_swap_int(&val[i], &val[j]);
            }
        }

        // swap two elements
        pangulu_swap_int(&key[start], &key[j]);
        pangulu_swap_int(&val[start], &val[j]);

        // recursively sort the lesser key
        pangulu_quicksort_keyval(key, val, start, j - 1);
        pangulu_quicksort_keyval(key, val, j + 1, end);
    }
}

double pangulu_standard_deviation(int_t *p, int_t num)
{
    double average = 0.0;
    for (int_t i = 0; i < num; i++)
    {
        average += (double)p[i];
    }
    average /= (double)num;
    double answer = 0.0;
    for (int_t i = 0; i < num; i++)
    {
        answer += (double)(((double)p[i] - average) * ((double)p[i] - average));
    }
    return sqrt(answer / (double)(num));
}

#ifndef GPU_OPEN
void pangulu_init_level_array(pangulu_Smatrix *A, int_t *work_space)
{
    int_t n = A->row;
    int_t *level_size = A->level_size;
    int_t *level_idx = A->level_idx;
    int_t index_inlevel = 0;
    int_t index_level_ptr = 0;
    int_t num_lev = 0;

    int_t *l_col_ptr = work_space;
    int_t *csr_diag_ptr = work_space + n + 1;
    int_t *inlevel = work_space + (n + 1) * 2;
    int_t *level_ptr = work_space + (n + 1) * 3;

    for (int_t i = 0; i < n; i++)
    {
        level_idx[i] = 0;
        level_size[i] = 0;
        inlevel[i] = 0;
        level_ptr[i] = 0;
        l_col_ptr[i] = 0;
        csr_diag_ptr[i] = 0;
    }

    for (int_t i = 0; i < n; i++) // each csc column
    {
        for (int_t j = A->columnpointer[i]; j < A->columnpointer[i + 1]; j++)
        {
            int_t row = A->rowindex[j];
            if (row == i)
            {
                l_col_ptr[i] = j;
            }
        }
    }

    for (int_t i = 0; i < n; i++) // each csr row
    {
        for (int_t j = A->rowpointer[i]; j < A->rowpointer[i + 1]; j++)
        {
            int_t column = A->columnindex[j];
            if (column == i)
            {
                csr_diag_ptr[i] = j;
                continue;
            }
            else
            {
                csr_diag_ptr[i] = -1;
            }
        }
    }

    for (int_t i = 0; i < n; i++) // each csc column
    {
        int_t max_lv = -1;
        int_t lv;
        // search dependent columns on the left
        for (int_t j = A->columnpointer[i]; j < l_col_ptr[i]; j++)
        {
            unsigned nz_idx = A->rowindex[j]; // Nonzero row in col i, U part

            // L part of col nz_idx exists , U-dependency found
            if (l_col_ptr[nz_idx] + 1 != A->columnpointer[nz_idx + 1])
            {
                lv = inlevel[nz_idx];
                if (lv > max_lv)
                {
                    max_lv = lv;
                }
            }
        }
        for (int_t j = A->rowpointer[i]; j < csr_diag_ptr[i]; j++)
        {
            unsigned nz_idx = A->columnindex[j];
            lv = inlevel[nz_idx];
            if (lv > max_lv)
            {
                max_lv = lv;
            }
        }
        lv = max_lv + 1;
        inlevel[index_inlevel++] = lv;
        ++level_size[lv];
        if (lv > num_lev)
        {
            num_lev = lv;
        }
    }

    ++num_lev;

    level_ptr[index_level_ptr++] = 0;
    for (int_t i = 0; i < num_lev; i++)
    {
        level_ptr[index_level_ptr++] = level_ptr[i] + level_size[i];
    }

    for (int_t i = 0; i < n; i++)
    {
        level_idx[level_ptr[inlevel[i]]++] = i;
    }

    A->num_lev = num_lev;
}

#endif

int_t choose_pivot(int_t i, int_t j)
{
    return (i + j) / 2;
}

void swap_value(calculate_type *a, calculate_type *b)
{
    calculate_type tmp = *a;
    *a = *b;
    *b = tmp;
}

void swap_index(int32_t *a, int32_t *b)
{
    int32_t tmp = *a;
    *a = *b;
    *b = tmp;
}

void pangulu_sort(int32_t *key, calculate_type *val, int_t start, int_t end)
{
    int_t pivot;
    int_t i, j, k;

    if (start < end)
    {
        k = choose_pivot(start, end);
        swap_index(&key[start], &key[k]);
        swap_value(&val[start], &val[k]);
        pivot = key[start];

        i = start + 1;
        j = end;
        while (i <= j)
        {
            while ((i <= end) && (key[i] <= pivot))
                i++;
            while ((j >= start) && (key[j] > pivot))
                j--;
            if (i < j)
            {
                swap_index(&key[i], &key[j]);
                swap_value(&val[i], &val[j]);
            }
        }

        // swap two elements
        swap_index(&key[start], &key[j]);
        swap_value(&val[start], &val[j]);

        // recursively sort the lesser key
        pangulu_sort(key, val, start, j - 1);
        pangulu_sort(key, val, j + 1, end);
    }
}

void pangulu_sort_struct(int32_t *key, int_t start, int_t end)
{
    int_t pivot;
    int_t i, j, k;

    if (start < end)
    {
        k = choose_pivot(start, end);
        swap_index(&key[start], &key[k]);
        pivot = key[start];

        i = start + 1;
        j = end;
        while (i <= j)
        {
            while ((i <= end) && (key[i] <= pivot))
                i++;
            while ((j >= start) && (key[j] > pivot))
                j--;
            if (i < j)
            {
                swap_index(&key[i], &key[j]);
            }
        }

        // swap two elements
        swap_index(&key[start], &key[j]);

        // recursively sort the lesser key
        pangulu_sort_struct(key, start, j - 1);
        pangulu_sort_struct(key, j + 1, end);
    }
}

void pangulu_sort_pangulu_matrix(int_t n, int_t *rowpointer, int_32t *columnindex)
{
    for (int_t i = 0; i < n; i++)
    {
        pangulu_sort_struct(columnindex, rowpointer[i], rowpointer[i + 1] - 1);
    }
}
void pangulu_sort_pangulu_matrix(pangulu_Smatrix *S)
{
    for (int_t i = 0; i < S->row; i++)
    {
        pangulu_sort(S->columnindex, S->value, S->rowpointer[i], S->rowpointer[i + 1] - 1);
    }
}
#ifdef GPU_OPEN
void TRIANGLE_PRE_CPU(idx_int *L_rowindex,
                      const int_t n,
                      const int_t nnzL,
                      int *d_graphInDegree)
{
    for (int i = 0; i < nnzL; i++)
    {
        d_graphInDegree[L_rowindex[i]] += 1;
    }
}

void pangulu_gessm_preprocess(pangulu_Smatrix *L)
{
    int_t n = L->row;
    int_t nnzL = L->nnz;

    /**********************************L****************************************/

    int *graphInDegree = L->graphInDegree;
    memset(graphInDegree, 0, n * sizeof(int));

    TRIANGLE_PRE_CPU(L->rowindex, n, nnzL, graphInDegree);
}

void pangulu_tstrf_preprocess(pangulu_Smatrix *U)
{
    int_t n = U->row;
    int_t nnzU = U->nnz;

    /**********************************L****************************************/

    int *graphInDegree = U->graphInDegree;
    memset(graphInDegree, 0, n * sizeof(int));

    TRIANGLE_PRE_CPU(U->columnindex, n, nnzU, graphInDegree);
}
#endif

#endif