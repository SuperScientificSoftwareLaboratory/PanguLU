#ifndef PANGULU_INTERFACE_H
#define PANGULU_INTERFACE_H

#include "pangulu_interface_common.h"
#include <mpi.h>

int_t CPU_MEMORY = 0;
int_t CPU_PEAK_MEMORY = 0;
int_t GPU_MEMORY = 0;
int_t HEAP_SELECT;
calculate_type *TEMP_A_value = NULL;
pangulu_inblock_idx *CUDA_B_idx_COL = NULL;
calculate_type *CUDA_TEMP_value = NULL;
int_t *ssssm_col_ops_u = NULL;
idx_int *ssssm_ops_pointer = NULL;
idx_int *getrf_diagIndex_csc = NULL;
idx_int *getrf_diagIndex_csr = NULL;

int_t STREAM_DENSE_INDEX = 0;
int_t INDEX_NUM = 0;
idx_int PANGU_OMP_NUM_THREADS = 1;

int_t FLOP = 0;
double TIME_transport = 0.0;
double TIME_isend = 0.0;
double TIME_receive = 0.0;
double TIME_getrf = 0.0;
double TIME_tstrf = 0.0;
double TIME_gessm = 0.0;
double TIME_gessm_dense = 0.0;
double TIME_gessm_sparse = 0.0;
double TIME_ssssm = 0.0;
double TIME_cuda_memcpy = 0.0;
double TIME_wait = 0.0;
double calculate_TIME_wait = 0.0;
int_t calculate_time = 0;

idx_int *SSSSM_hash_LU = NULL;
char *SSSSM_flag_LU = NULL;
char *SSSSM_flag_L_row = NULL;
idx_int *SSSSM_hash_L_row = NULL;
idx_int zip_max_id = 0;
idx_int zip_cur_id = 0;
calculate_type *SSSSM_L_value = NULL;
calculate_type *SSSSM_U_value = NULL;
idx_int *zip_rows = NULL;
idx_int *zip_cols = NULL;
idx_int *SSSSM_hash_U_col = NULL;

int_32t RANK;
int_32t LEVEL;
int_32t OMP_THREAD;



#endif