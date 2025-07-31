#define PANGULU_PLATFORM_ENV
#include "../../../../pangulu_common.h"
#include "cblas.h"

pangulu_int32_t **ssssm_hash_lu = NULL;
pangulu_int32_t **ssssm_hash_l_row = NULL;
pangulu_int32_t **ssssm_hash_l_row_inv = NULL;
pangulu_int32_t **ssssm_hash_u_col = NULL;
pangulu_int32_t **ssssm_hash_u_col_inv = NULL;
calculate_type **ssssm_l_value = NULL;
calculate_type **ssssm_u_value = NULL;
calculate_type **temp_a_value;

void pangulu_platform_0100000_malloc(void** platform_address, size_t size){
    *platform_address = pangulu_malloc(__FILE__, __LINE__, size);
}

void pangulu_platform_0100000_malloc_pinned(void** platform_address, size_t size){
    *platform_address = pangulu_malloc(__FILE__, __LINE__, size);
}

void pangulu_platform_0100000_synchronize(){}

void pangulu_platform_0100000_memset(void* s, int c, size_t n){
    memset(s, c, n);
}

void pangulu_platform_0100000_create_stream(void** stream){}

void pangulu_platform_0100000_memcpy(void *dst, const void *src, size_t count, unsigned int kind){
    memcpy(dst, src, count);
}

void pangulu_platform_0100000_memcpy_async(void *dst, const void *src, size_t count, unsigned int kind, void* stream){
    memcpy(dst, src, count);
}

void pangulu_platform_0100000_free(void* devptr){
    free(devptr);
}

void pangulu_platform_0100000_get_device_num(int* device_num){
    *device_num = 1;
}

void pangulu_platform_0100000_set_default_device(int device_num){}

void pangulu_platform_0100000_get_device_name(char* name, int device_num){
    strcpy(name, "CPU");
}

void pangulu_platform_0100000_get_device_memory_usage(size_t *used_byte)
{
    *used_byte = 0;
}

void pangulu_platform_0100000_getrf(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t* opdst,
    int tid
){
    pangulu_storage_slot_t* upper_diag;
    pangulu_storage_slot_t* lower_diag;
    if(opdst->is_upper){
        upper_diag = opdst;
        lower_diag = opdst->related_block;
    }else{
        upper_diag = opdst->related_block;
        lower_diag = opdst;
    }

    for (pangulu_int32_t level = 0; level < nb; level++)
    {
        if (upper_diag->columnpointer[level] == upper_diag->columnpointer[level + 1])
        {
            continue;
        }

        calculate_type diag;
        if(fabs(upper_diag->value[upper_diag->columnpointer[level]]) < PANGULU_TOL){
            diag = PANGULU_TOL;
        }else{
            diag = upper_diag->value[upper_diag->columnpointer[level]];
        }
        for(pangulu_int32_t csc_idx = lower_diag->columnpointer[level]; csc_idx < lower_diag->columnpointer[level+1]; csc_idx++){
            pangulu_int32_t row = lower_diag->rowindex[csc_idx];
            lower_diag->value[csc_idx] /= diag;
        }

        for(pangulu_int32_t csc_idx = lower_diag->columnpointer[level]; csc_idx < lower_diag->columnpointer[level+1]; csc_idx++){
            pangulu_int32_t row = lower_diag->rowindex[csc_idx];
            calculate_type op1 = lower_diag->value[csc_idx];
            pangulu_int32_t csr_idx_op2 = upper_diag->columnpointer[level];
            pangulu_int32_t csr_idx_op2_ub = upper_diag->columnpointer[level+1];
            pangulu_int32_t csr_idx_opdst = upper_diag->columnpointer[row];
            pangulu_int32_t csr_idx_opdst_ub = upper_diag->columnpointer[row+1];
            while(csr_idx_op2 < csr_idx_op2_ub && csr_idx_opdst < csr_idx_opdst_ub){
                if(upper_diag->rowindex[csr_idx_opdst] == upper_diag->rowindex[csr_idx_op2]){
                    upper_diag->value[csr_idx_opdst] -= op1 * upper_diag->value[csr_idx_op2];
                    csr_idx_op2++;
                    csr_idx_opdst++;
                }
                while(csr_idx_op2 < csr_idx_op2_ub && upper_diag->rowindex[csr_idx_op2] < upper_diag->rowindex[csr_idx_opdst]){
                    csr_idx_op2++;
                }
                while(csr_idx_opdst < csr_idx_opdst_ub && upper_diag->rowindex[csr_idx_opdst] < upper_diag->rowindex[csr_idx_op2]){
                    csr_idx_opdst++;
                }
            }
        }

        for(pangulu_int32_t csr_idx = upper_diag->columnpointer[level]+1; csr_idx < upper_diag->columnpointer[level+1]; csr_idx++){
            pangulu_int32_t col = upper_diag->rowindex[csr_idx];
            calculate_type op2 = upper_diag->value[csr_idx];
            pangulu_int32_t csc_idx_op1 = lower_diag->columnpointer[level];
            pangulu_int32_t csc_idx_op1_ub = lower_diag->columnpointer[level+1];
            pangulu_int32_t csc_idx_opdst = lower_diag->columnpointer[col];
            pangulu_int32_t csc_idx_opdst_ub = lower_diag->columnpointer[col+1];
            while(csc_idx_op1 < csc_idx_op1_ub && csc_idx_opdst < csc_idx_opdst_ub){
                if(lower_diag->rowindex[csc_idx_opdst] == lower_diag->rowindex[csc_idx_op1]){
                    lower_diag->value[csc_idx_opdst] -= lower_diag->value[csc_idx_op1] * op2;
                    csc_idx_op1++;
                    csc_idx_opdst++;
                }
                while(csc_idx_op1 < csc_idx_op1_ub && lower_diag->rowindex[csc_idx_op1] < lower_diag->rowindex[csc_idx_opdst]){
                    csc_idx_op1++;
                }
                while(csc_idx_opdst < csc_idx_opdst_ub && lower_diag->rowindex[csc_idx_opdst] < lower_diag->rowindex[csc_idx_op1]){
                    csc_idx_opdst++;
                }
            }
        }

    }
}

void pangulu_platform_0100000_tstrf(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t* opdst,
    pangulu_storage_slot_t* opdiag,
    int tid
){
    if(opdiag->is_upper == 0){
        opdiag = opdiag->related_block;
    }

    for(pangulu_int32_t rhs_id=0;rhs_id<nb;rhs_id++){
        for(pangulu_int32_t rhs_idx = opdst->rowpointer[rhs_id]; rhs_idx < opdst->rowpointer[rhs_id+1]; rhs_idx++){
            pangulu_int32_t mul_row = opdst->columnindex[rhs_idx];
            pangulu_int32_t lsum_idx = rhs_idx+1;
            pangulu_int32_t diag_idx = opdiag->columnpointer[mul_row];
            calculate_type diag;
            if(fabs(opdiag->value[diag_idx]) < PANGULU_TOL){
                diag = PANGULU_TOL;
            }else{
                diag = opdiag->value[diag_idx];
            }
            opdst->value[opdst->idx_of_csc_value_for_csr[rhs_idx]] /= diag;
            calculate_type mul_val = opdst->value[opdst->idx_of_csc_value_for_csr[rhs_idx]];
            while(lsum_idx < opdst->rowpointer[rhs_id+1] && diag_idx < opdiag->columnpointer[mul_row+1]){
                if(opdiag->rowindex[diag_idx] == opdst->columnindex[lsum_idx]){
                    opdst->value[opdst->idx_of_csc_value_for_csr[lsum_idx]] -= mul_val * opdiag->value[diag_idx];
                    lsum_idx++;
                    diag_idx++;
                }
                while(lsum_idx < opdst->rowpointer[rhs_id+1] && opdst->columnindex[lsum_idx] < opdiag->rowindex[diag_idx]){
                    lsum_idx++;
                }
                while(diag_idx < opdiag->columnpointer[mul_row+1] && opdiag->rowindex[diag_idx] < opdst->columnindex[lsum_idx]){
                    diag_idx++;
                }
            }
        }
    }
}


void pangulu_platform_0100000_gessm(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t* opdst,
    pangulu_storage_slot_t* opdiag,
    int tid
){
    if(opdiag->is_upper == 1){
        opdiag = opdiag->related_block;
    }

    for(pangulu_int32_t rhs_id=0;rhs_id<nb;rhs_id++){
        for(pangulu_int32_t rhs_idx = opdst->columnpointer[rhs_id]; rhs_idx < opdst->columnpointer[rhs_id+1]; rhs_idx++){
            pangulu_int32_t mul_row = opdst->rowindex[rhs_idx];
            calculate_type mul_val = opdst->value[rhs_idx];
            pangulu_int32_t lsum_idx = rhs_idx+1;
            pangulu_int32_t diag_idx = opdiag->columnpointer[mul_row];
            while(lsum_idx < opdst->columnpointer[rhs_id+1] && diag_idx < opdiag->columnpointer[mul_row+1]){
                if(opdiag->rowindex[diag_idx] == opdst->rowindex[lsum_idx]){
                    opdst->value[lsum_idx] -= mul_val * opdiag->value[diag_idx];
                    lsum_idx++;
                    diag_idx++;
                }
                while(lsum_idx < opdst->columnpointer[rhs_id+1] && opdst->rowindex[lsum_idx] < opdiag->rowindex[diag_idx]){
                    lsum_idx++;
                }
                while(diag_idx < opdiag->columnpointer[mul_row+1] && opdiag->rowindex[diag_idx] < opdst->rowindex[lsum_idx]){
                    diag_idx++;
                }
            }
        }
    }
}

void pangulu_platform_0100000_ssssm(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t* opdst,
    pangulu_storage_slot_t* op1,
    pangulu_storage_slot_t* op2,
    int tid
){
    if(ssssm_hash_lu[tid] == NULL){
        ssssm_hash_lu[tid] = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * (nb));
    }
    if(ssssm_hash_l_row[tid] == NULL){
        ssssm_hash_l_row[tid] = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * (nb));
    }
    if(ssssm_hash_l_row_inv[tid] == NULL){
        ssssm_hash_l_row_inv[tid] = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * (nb));
    }
    if(ssssm_hash_u_col[tid] == NULL){
        ssssm_hash_u_col[tid] = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * (nb));
    }
    if(ssssm_hash_u_col_inv[tid] == NULL){
        ssssm_hash_u_col_inv[tid] = (pangulu_int32_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_int32_t) * (nb));
    }
    if(ssssm_l_value[tid] == NULL){
        ssssm_l_value[tid] = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nb * nb);
        memset(ssssm_l_value[tid], 0, sizeof(calculate_type) * nb * nb);
    }
    if(ssssm_u_value[tid] == NULL){
        ssssm_u_value[tid] = (calculate_type *)pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nb * nb);
        memset(ssssm_u_value[tid], 0, sizeof(calculate_type) * nb * nb);
    }
    if(temp_a_value[tid] == NULL){
        temp_a_value[tid] = pangulu_malloc(__FILE__, __LINE__, sizeof(calculate_type) * nb * nb);
    }

    int n = nb;
    int L_row_num = 0, U_col_num = 0, LU_rc_num = 0;
    int *blas_dense_hash_col_LU = NULL;
    int *blas_dense_hash_row_L = NULL;

    blas_dense_hash_col_LU = ssssm_hash_lu[tid];
    blas_dense_hash_row_L = ssssm_hash_l_row[tid];
    for (int i = 0; i < n; i++)
    {
        blas_dense_hash_row_L[i] = -1;
        ssssm_hash_u_col[tid][i] = -1;
    }
    for (int i = 0; i < n; i++)
    {
        if (op1->columnpointer[i + 1] > ((i==0)?0:op1->columnpointer[i]))
        {
            blas_dense_hash_col_LU[i] = LU_rc_num;
            LU_rc_num++;
        }
    }
    for (int i = 0; i < n; i++)
    {
        int col_begin = ((i==0)?0:op1->columnpointer[i]);
        int col_end = op1->columnpointer[i + 1];
        for (int j = col_begin; j < col_end; j++)
        {
            int L_row = op1->rowindex[j];
            if (blas_dense_hash_row_L[L_row] == -1)
            {
                blas_dense_hash_row_L[L_row] = L_row_num;
                ssssm_hash_l_row_inv[tid][L_row_num] = L_row;
                L_row_num++;
            }
        }
    }

    for (int i = 0; i < n; i++)
    {
        int begin = ((i==0)?0:op2->columnpointer[i]);
        int end = op2->columnpointer[i + 1];
        if (end > begin)
        {
            calculate_type *U_temp_value = ssssm_u_value[tid] + U_col_num * LU_rc_num; // op2 column based

            for (int j = begin; j < end; j++)
            {
                int U_row = op2->rowindex[j];
                if (op1->columnpointer[U_row + 1] > ((U_row==0)?0:op1->columnpointer[U_row]) > 0)
                {
                    U_temp_value[blas_dense_hash_col_LU[U_row]] = op2->value[j];
                }
            }
            ssssm_hash_u_col_inv[tid][U_col_num] = i;
            ssssm_hash_u_col[tid][i] = U_col_num;
            U_col_num++;
        }
    }
    for (int i = 0; i < n; i++)
    {
        int col_begin = ((i==0)?0:op1->columnpointer[i]);
        int col_end = op1->columnpointer[i + 1];
        calculate_type *temp_data = ssssm_l_value[tid] + L_row_num * blas_dense_hash_col_LU[i];
        for (int j = col_begin; j < col_end; j++)
        {
            temp_data[blas_dense_hash_row_L[op1->rowindex[j]]] = op1->value[j];
        }
    }
    int m = L_row_num;
    int k = LU_rc_num;
    n = U_col_num;

    calculate_type alpha = 1.0, beta = 0.0;
#if defined(CALCULATE_TYPE_CR64)
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, ssssm_l_value[tid], m, ssssm_u_value[tid], k, &beta, temp_a_value[tid], m);
#elif defined(CALCULATE_TYPE_R64)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, ssssm_l_value[tid], m, ssssm_u_value[tid], k, beta, temp_a_value[tid], m);
#elif defined(CALCULATE_TYPE_R32)
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, ssssm_l_value[tid], m, ssssm_u_value[tid], k, beta, temp_a_value[tid], m);
#elif defined(CALCULATE_TYPE_CR32)
    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, ssssm_l_value[tid], m, ssssm_u_value[tid], k, &beta, temp_a_value[tid], m);
#else
#error[PanguLU Compile Error] Unsupported CALCULATE_TYPE for the selected BLAS library. Please recompile with a compatible value.
#endif

    memset(ssssm_l_value[tid], 0, sizeof(calculate_type) * m * k);
    memset(ssssm_u_value[tid], 0, sizeof(calculate_type) * k * n);

    if(opdst->bcol_pos == opdst->brow_pos){
        pangulu_storage_slot_t* upper_diag;
        pangulu_storage_slot_t* lower_diag;
        if(opdst->is_upper){
            upper_diag = opdst;
            lower_diag = opdst->related_block;
        }else{
            upper_diag = opdst->related_block;
            lower_diag = opdst;
        }
        #pragma omp critical
        {
            for (int i = 0; i < U_col_num; i++)
            {
                int col_num = ssssm_hash_u_col_inv[tid][i];
                calculate_type *temp_value = temp_a_value[tid] + i * m;
                int j_begin = lower_diag->columnpointer[col_num];
                int j_end = lower_diag->columnpointer[col_num + 1];
                for (int j = j_begin; j < j_end; j++)
                {
                    int row = lower_diag->rowindex[j];
                    if (blas_dense_hash_row_L[row] != -1)
                    {
                        int row_index = blas_dense_hash_row_L[row];
                        lower_diag->value[j] -= temp_value[row_index];
                    }
                }
            }
            for (int i = 0; i < L_row_num; i++)
            {
                int row = ssssm_hash_l_row_inv[tid][i];
                int j_begin = upper_diag->columnpointer[row];
                int j_end = upper_diag->columnpointer[row + 1];
                for (int j = j_begin; j < j_end; j++)
                {
                    int col = upper_diag->rowindex[j];
                    if (ssssm_hash_u_col[tid][col] != -1)
                    {
                        int col_index = ssssm_hash_u_col[tid][col];
                        upper_diag->value[j] -= temp_a_value[tid][col_index * m + i];
                    }
                }
            }
        }
    }else{
        #pragma omp critical
        {
            for (int i = 0; i < U_col_num; i++)
            {
                int col_num = ssssm_hash_u_col_inv[tid][i];
                calculate_type *temp_value = temp_a_value[tid] + i * m;
                int j_begin = ((col_num==0)?0:opdst->columnpointer[col_num]);
                int j_end = opdst->columnpointer[col_num + 1];
                for (int j = j_begin; j < j_end; j++)
                {
                    int row = opdst->rowindex[j];
                    if (blas_dense_hash_row_L[row] != -1)
                    {
                        int row_index = blas_dense_hash_row_L[row];
                        opdst->value[j] -= temp_value[row_index];
                    }
                }
            }
        }
    }
}

void pangulu_platform_0100000_ssssm_batched(
    pangulu_inblock_idx nb,
    pangulu_uint64_t ntask,
    pangulu_task_t* tasks
){
    for(pangulu_uint64_t itask = 0; itask < ntask; itask++){
        pangulu_platform_0100000_ssssm(nb, tasks[itask].opdst, tasks[itask].op1, tasks[itask].op2, 0);
    }
}


void pangulu_platform_0100000_hybrid_batched(
    pangulu_inblock_idx nb,
    pangulu_uint64_t ntask,
    pangulu_task_t* tasks
){
    for(pangulu_uint64_t itask = 0; itask < ntask; itask++){
        switch(tasks[itask].kernel_id){
            case PANGULU_TASK_GETRF:
                pangulu_platform_0100000_getrf(nb, tasks[itask].opdst, 0);
                break;
            case PANGULU_TASK_TSTRF:
                pangulu_platform_0100000_tstrf(nb, tasks[itask].opdst, tasks[itask].op1, 0);
                break;
            case PANGULU_TASK_GESSM:
                pangulu_platform_0100000_gessm(nb, tasks[itask].opdst, tasks[itask].op1, 0);
                break;
            case PANGULU_TASK_SSSSM:
                pangulu_platform_0100000_ssssm(nb, tasks[itask].opdst, tasks[itask].op1, tasks[itask].op2, 0);
                break;
        }
    }
}



void pangulu_platform_0100000_spmv(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t* a,
    calculate_type* x,
    calculate_type* y
){
    if(nb > 0){
        for(int idx = 0; idx < a->columnpointer[1]; idx++){
            int row = a->rowindex[idx];
            y[row] -= a->value[idx] * x[0];
        }
    }
    for(int col = 1; col < nb; col++){
        for(int idx = a->columnpointer[col]; idx < a->columnpointer[col+1]; idx++){
            int row = a->rowindex[idx];
            y[row] -= a->value[idx] * x[col];
        }
    }
}

void pangulu_platform_0100000_vecadd(
    pangulu_int64_t length,
    calculate_type *bval, 
    calculate_type *xval
){
    for (pangulu_int64_t i = 0; i < length; i++)
    {
        bval[i] += xval[i];
    }
}

void pangulu_platform_0100000_sptrsv(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *s,
    calculate_type* xval,
    pangulu_int64_t uplo
){
    if(uplo==0){
        pangulu_inblock_ptr *l_colptr=s->columnpointer;
        pangulu_inblock_idx *l_rowidx=s->rowindex;
        calculate_type *l_value = s->value;
        for(pangulu_int32_t i=0;i<nb;i++) 
        {
            for(pangulu_int32_t j=l_colptr[i];j<l_colptr[i+1];j++)
            {
                pangulu_inblock_idx row=l_rowidx[j];
                xval[row]-=l_value[j]*xval[i];
            }
        }
    }else{
        pangulu_inblock_ptr *u_rowptr=s->columnpointer;
        pangulu_inblock_idx *u_coidx=s->rowindex;
        calculate_type *u_value = s->value;
        for(pangulu_int32_t i=nb-1;i>=0;i--) 
        {
            if(u_rowptr[i+1] == u_rowptr[i]){
                continue;
            }

            for(pangulu_int32_t j=u_rowptr[i]+1; j<u_rowptr[i+1]; j++)
            {
                pangulu_inblock_idx col=u_coidx[j];
                xval[i]-=u_value[j]*xval[col];
            }

            if(fabs(u_value[u_rowptr[i]])>PANGULU_SPTRSV_TOL)
                xval[i]=xval[i]/u_value[u_rowptr[i]];
            else
                xval[i]=xval[i]/PANGULU_SPTRSV_TOL;
        }
    }
}
