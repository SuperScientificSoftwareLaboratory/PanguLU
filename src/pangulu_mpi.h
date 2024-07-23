#ifndef PANGULU_MPI_H
#define PANGULU_MPI_H

#include <mpi.h>
#include "pangulu_common.h"

#ifdef CHECK_TIME
#include "pangulu_time.h"
#endif

int have_msg;
void pangulu_probe_message(MPI_Status *status)
{
    have_msg=0;
    do{
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &have_msg, status);
        if(have_msg){
            return;
        }
        usleep(10);
    }while(!have_msg);
}

int_t pangulu_Bcast_N(int_t N, int_t send_rank)
{
    MPI_Bcast(&N, 1, MPI_INT_TYPE, send_rank, MPI_COMM_WORLD);
    return N;
}

void pangulu_Bcast_vector(int_t *vector, int_t length, int_t send_rank)
{
    int_t everry_length = 100000000;
    for (int_t i = 0; i < length; i += everry_length)
    {
        if ((i + everry_length) > length)
        {
            MPI_Bcast(vector + i, length - i, MPI_INT_TYPE, send_rank, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Bcast(vector + i, everry_length, MPI_INT_TYPE, send_rank, MPI_COMM_WORLD);
        }
    }
}
void pangulu_mpi_waitall(MPI_Request *Request, int num)
{
    MPI_Status Status;
    for(int i = 0; i < num; i++)
    {
        MPI_Wait(&Request[i], &Status);
    }
}
void pangulu_isend_vector_char_wait(char *A, int_t N, int_t send_id, int signal, MPI_Request* req)
{
    MPI_Isend(A, N, MPI_CHAR, send_id, signal, MPI_COMM_WORLD, req);
}

void pangulu_send_vector_int(int_t *A, int_t N, int_t send_id, int signal)
{
    MPI_Send(A, N, MPI_INT_TYPE, send_id, signal, MPI_COMM_WORLD);
}

void pangulu_recv_vector_int(int_t *A, int_t N, int_t receive_id, int signal)
{
    MPI_Status status;
    for (int_t i = 0; i < N; i++)
    {
        A[i] = 0;
    }
    MPI_Recv(A, N, MPI_INT_TYPE, receive_id, signal, MPI_COMM_WORLD, &status);
}

void pangulu_send_vector_char(char *A, int_t N, int_t send_id, int signal)
{
    MPI_Send(A, N, MPI_CHAR, send_id, signal, MPI_COMM_WORLD);
}

void pangulu_recv_vector_char(char *A, int_t N, int_t receive_id, int signal)
{
    MPI_Status status;
    for (int_t i = 0; i < N; i++)
    {
        A[i] = 0;
    }
    pangulu_probe_message(&status);
    MPI_Recv(A, N, MPI_CHAR, receive_id, signal, MPI_COMM_WORLD, &status);
}

void pangulu_send_vector_value(calculate_type *A, int_t N, int_t send_id, int signal)
{
    MPI_Send(A, N, MPI_VAL_TYPE, send_id, signal, MPI_COMM_WORLD);
}

void pangulu_recv_vector_value(calculate_type *A, int_t N, int_t receive_id, int signal)
{
    MPI_Status status;
    for (int_t i = 0; i < N; i++)
    {
        A[i] = 0.0;
    }
    MPI_Recv(A, N, MPI_VAL_TYPE, receive_id, signal, MPI_COMM_WORLD, &status);
}

void pangulu_send_pangulu_Smatrix_value_CSR(pangulu_Smatrix *S,
                                            int_t send_id, int signal, int_t NB)
{

    MPI_Send(S->value, S->nnz, MPI_VAL_TYPE, send_id, signal + 2, MPI_COMM_WORLD);
}
void pangulu_send_pangulu_Smatrix_struct_CSR(pangulu_Smatrix *S,
                                             int_t send_id, int signal, int_t NB)
{

    MPI_Send(S->rowpointer, S->row + 1, MPI_INT_TYPE, send_id, signal, MPI_COMM_WORLD);
    MPI_Send(S->columnindex, S->nnz, MPI_INT_TYPE, send_id, signal + 1, MPI_COMM_WORLD);
}

void pangulu_send_pangulu_Smatrix_complete_CSR(pangulu_Smatrix *S,
                                               int_t send_id, int signal, int_t NB)
{
    pangulu_send_pangulu_Smatrix_struct_CSR(S, send_id, signal * 3, NB);
    pangulu_send_pangulu_Smatrix_value_CSR(S, send_id, signal * 3, NB);
}

void pangulu_recv_pangulu_Smatrix_struct_CSR(pangulu_Smatrix *S,
                                             int_t receive_id, int signal, int_t NB)
{

    MPI_Status status;
    for (int_t i = 0; i < (S->row + 1); i++)
    {
        S->rowpointer[i] = 0;
    }

    MPI_Recv(S->rowpointer, S->row + 1, MPI_INT_TYPE, receive_id, signal, MPI_COMM_WORLD, &status);
    S->nnz = S->rowpointer[S->row];
    for (int_t i = 0; i < S->nnz; i++)
    {
        S->columnindex[i] = 0;
    }
    MPI_Recv(S->columnindex, S->nnz, MPI_INT_TYPE, receive_id, signal + 1, MPI_COMM_WORLD, &status);
}
void pangulu_recv_pangulu_Smatrix_value_CSR(pangulu_Smatrix *S,
                                            int_t receive_id, int signal, int_t NB)
{
    MPI_Status status;
    for (int_t i = 0; i < S->nnz; i++)
    {
        S->value[i] = (calculate_type)0.0;
    }
    MPI_Recv(S->value, S->nnz, MPI_VAL_TYPE, receive_id, signal + 2, MPI_COMM_WORLD, &status);
}

void pangulu_recv_pangulu_Smatrix_value_CSR_in_signal(pangulu_Smatrix *S,
                                                      int_t receive_id, int signal, int_t NB)
{
    MPI_Status status;
    for (int_t i = 0; i < S->nnz; i++)
    {
        S->value[i] = (calculate_type)0.0;
    }
    MPI_Recv(S->value, S->nnz, MPI_VAL_TYPE, receive_id, signal, MPI_COMM_WORLD, &status);
}

void pangulu_recv_pangulu_Smatrix_complete_CSR(pangulu_Smatrix *S,
                                               int_t receive_id, int signal, int_t NB)
{

    pangulu_recv_pangulu_Smatrix_struct_CSR(S, receive_id, signal * 3, NB);
    pangulu_recv_pangulu_Smatrix_value_CSR(S, receive_id, signal * 3, NB);
}

void pangulu_recv_whole_pangulu_Smatrix_CSR(pangulu_Smatrix *S,
                                            int_t receive_id, int signal, int_t nnz, int_t NB)
{
#ifdef CHECK_TIME
    struct timeval GET_TIME_START;
    pangulu_time_check_begin(&GET_TIME_START);
#endif
    int_t length = sizeof(pangulu_inblock_ptr) * (NB + 1) + sizeof(pangulu_inblock_idx) * nnz + sizeof(calculate_type) * nnz;
    MPI_Status status;
    char *now_vector = (char *)(S->rowpointer);
    for (int_t i = 0; i < length; i++)
    {
        now_vector[i] = 0;
    }
    S->columnindex = (pangulu_inblock_idx *)(now_vector + sizeof(pangulu_inblock_ptr) * (NB + 1));
    S->value = (calculate_type *)(now_vector + sizeof(pangulu_inblock_ptr) * (NB + 1) + sizeof(pangulu_inblock_idx) * nnz);
    MPI_Recv(now_vector, length, MPI_CHAR, receive_id, signal, MPI_COMM_WORLD, &status);
    S->nnz = nnz;
#ifdef CHECK_TIME
    TIME_receive += pangulu_time_check_end(&GET_TIME_START);
#endif
}

void pangulu_send_pangulu_Smatrix_value_CSC(pangulu_Smatrix *S,
                                            int_t send_id, int signal, int_t NB)
{
    MPI_Send(S->value_CSC, S->nnz, MPI_VAL_TYPE, send_id, signal + 2, MPI_COMM_WORLD);
}

void pangulu_send_pangulu_Smatrix_struct_CSC(pangulu_Smatrix *S,
                                             int_t send_id, int signal, int_t NB)
{

    MPI_Send(S->columnpointer, S->row + 1, MPI_INT_TYPE, send_id, signal, MPI_COMM_WORLD);
    MPI_Send(S->rowindex, S->nnz, MPI_INT_TYPE, send_id, signal + 1, MPI_COMM_WORLD);
}

void pangulu_send_pangulu_Smatrix_complete_CSC(pangulu_Smatrix *S,
                                               int_t send_id, int signal, int_t NB)
{
    pangulu_send_pangulu_Smatrix_struct_CSC(S, send_id, signal * 3, NB);
    pangulu_send_pangulu_Smatrix_value_CSC(S, send_id, signal * 3, NB);
}

void pangulu_recv_pangulu_Smatrix_struct_CSC(pangulu_Smatrix *S,
                                             int_t receive_id, int signal, int_t NB)
{

    MPI_Status status;
    for (int_t i = 0; i < (S->row + 1); i++)
    {
        S->columnpointer[i] = 0;
    }

    MPI_Recv(S->columnpointer, S->row + 1, MPI_INT_TYPE, receive_id, signal, MPI_COMM_WORLD, &status);
    S->nnz = S->columnpointer[S->row];
    for (int_t i = 0; i < S->nnz; i++)
    {
        S->rowindex[i] = 0;
    }
    MPI_Recv(S->rowindex, S->nnz, MPI_INT_TYPE, receive_id, signal + 1, MPI_COMM_WORLD, &status);
}
void pangulu_recv_pangulu_Smatrix_value_CSC(pangulu_Smatrix *S,
                                            int_t receive_id, int signal, int_t NB)
{

    MPI_Status status;
    for (int_t i = 0; i < S->nnz; i++)
    {
        S->value_CSC[i] = (calculate_type)0.0;
    }

    MPI_Recv(S->value_CSC, S->nnz, MPI_VAL_TYPE, receive_id, signal + 2, MPI_COMM_WORLD, &status);
}

void pangulu_recv_pangulu_Smatrix_value_CSC_in_signal(pangulu_Smatrix *S,
                                                      int_t receive_id, int signal, int_t NB)
{
    MPI_Status status;
    for (int_t i = 0; i < S->nnz; i++)
    {
        S->value_CSC[i] = (calculate_type)0.0;
    }
    MPI_Recv(S->value_CSC, S->nnz, MPI_VAL_TYPE, receive_id, signal, MPI_COMM_WORLD, &status);
}

void pangulu_recv_pangulu_Smatrix_complete_CSC(pangulu_Smatrix *S,
                                               int_t receive_id, int signal, int_t NB)
{

    pangulu_recv_pangulu_Smatrix_struct_CSC(S, receive_id, signal * 3, NB);
    pangulu_recv_pangulu_Smatrix_value_CSC(S, receive_id, signal * 3, NB);
}

void pangulu_recv_whole_pangulu_Smatrix_CSC(pangulu_Smatrix *S,
                                            int_t receive_id, int signal, int_t nnz, int_t NB)
{
#ifdef CHECK_TIME
    struct timeval GET_TIME_START;
    pangulu_time_check_begin(&GET_TIME_START);
#endif
    int_t length = sizeof(pangulu_inblock_ptr) * (NB + 1) + sizeof(pangulu_inblock_idx) * nnz + sizeof(calculate_type) * nnz;
    MPI_Status status;
    char *now_vector = (char *)(S->columnpointer);
    for (int_t i = 0; i < length; i++)
    {
        now_vector[i] = 0;
    }
    S->rowindex = (pangulu_inblock_idx *)(now_vector + sizeof(pangulu_inblock_ptr) * (NB + 1));
    S->value_CSC = (calculate_type *)(now_vector + sizeof(pangulu_inblock_ptr) * (NB + 1) + sizeof(pangulu_inblock_idx) * nnz);
    MPI_Recv(now_vector, length, MPI_CHAR, receive_id, signal, MPI_COMM_WORLD, &status);
    S->nnz = nnz;
#ifdef CHECK_TIME
    TIME_receive += pangulu_time_check_end(&GET_TIME_START);
#endif
}

int pangulu_iprobe_message(MPI_Status *status)
{
    int flag;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, status);
    return flag;
}

void pangulu_isend_pangulu_Smatrix_value_CSR(pangulu_Smatrix *S,
                                             int_t send_id, int signal, int_t NB)
{

    MPI_Request req;
    MPI_Isend(S->value, S->nnz, MPI_VAL_TYPE, send_id, signal + 2, MPI_COMM_WORLD, &req);
}
void pangulu_isend_pangulu_Smatrix_struct_CSR(pangulu_Smatrix *S,
                                              int_t send_id, int signal, int_t NB)
{
    MPI_Request req;
    MPI_Isend(S->rowpointer, S->row + 1, MPI_INT_TYPE, send_id, signal, MPI_COMM_WORLD, &req);
    MPI_Isend(S->columnindex, S->nnz, MPI_INT_TYPE, send_id, signal + 1, MPI_COMM_WORLD, &req);
}

void pangulu_isend_pangulu_Smatrix_complete_CSR(pangulu_Smatrix *S,
                                                int_t send_id, int signal, int_t NB)
{

    pangulu_isend_pangulu_Smatrix_struct_CSR(S, send_id, signal * 3, NB);
    pangulu_isend_pangulu_Smatrix_value_CSR(S, send_id, signal * 3, NB);
}

void pangulu_isend_whole_pangulu_Smatrix_CSR(pangulu_Smatrix *S,
                                             int_t receive_id, int signal, int_t NB)
{
    int_t nnz = S->nnz;
    int_t length = sizeof(pangulu_inblock_ptr) * (NB + 1) + sizeof(pangulu_inblock_idx) * nnz + sizeof(calculate_type) * nnz;
    MPI_Request req;
    char *now_vector = (char *)(S->rowpointer);
    calculate_type *value = (calculate_type *)(now_vector + sizeof(pangulu_inblock_ptr) * (NB + 1) + sizeof(pangulu_inblock_idx) * nnz);
    if (value != S->value)
    {
        printf(PANGULU_E_ISEND_CSR);
    }
    MPI_Isend(now_vector, length, MPI_CHAR, receive_id, signal, MPI_COMM_WORLD, &req);
}

void pangulu_isend_pangulu_Smatrix_value_CSC(pangulu_Smatrix *S,
                                             int_t send_id, int signal, int_t NB)
{
    MPI_Request req;
    MPI_Isend(S->value_CSC, S->nnz, MPI_VAL_TYPE, send_id, signal + 2, MPI_COMM_WORLD, &req);
}

void pangulu_isend_pangulu_Smatrix_value_CSC_in_signal(pangulu_Smatrix *S,
                                                       int_t send_id, int signal, int_t NB)
{
    MPI_Request req;
    MPI_Isend(S->value_CSC, S->nnz, MPI_VAL_TYPE, send_id, signal, MPI_COMM_WORLD, &req);
}

void pangulu_isend_pangulu_Smatrix_struct_CSC(pangulu_Smatrix *S,
                                              int_t send_id, int signal, int_t NB)
{
    MPI_Request req;
    MPI_Isend(S->columnpointer, S->row + 1, MPI_INT_TYPE, send_id, signal, MPI_COMM_WORLD, &req);
    MPI_Isend(S->rowindex, S->nnz, MPI_INT_TYPE, send_id, signal + 1, MPI_COMM_WORLD, &req);
}

void pangulu_isend_pangulu_Smatrix_complete_CSC(pangulu_Smatrix *S,
                                                int_t send_id, int signal, int_t NB)
{

    pangulu_isend_pangulu_Smatrix_struct_CSC(S, send_id, signal * 3, NB);
    pangulu_isend_pangulu_Smatrix_value_CSC(S, send_id, signal * 3, NB);
}

void pangulu_isend_whole_pangulu_Smatrix_CSC(pangulu_Smatrix *S,
                                             int_t receive_id, int signal, int_t NB)
{
    int_t nnz = S->nnz;
    int_t length = sizeof(pangulu_inblock_ptr) * (NB + 1) + sizeof(pangulu_inblock_idx) * nnz + sizeof(calculate_type) * nnz;
    MPI_Request req;
    char *now_vector = (char *)(S->columnpointer);
    calculate_type *value = (calculate_type *)(now_vector + sizeof(pangulu_inblock_ptr) * (NB + 1) + sizeof(pangulu_inblock_idx) * nnz);
    if (value != S->value_CSC)
    {
        printf(PANGULU_E_ISEND_CSC);
    }
    MPI_Isend(now_vector, length, MPI_CHAR, receive_id, signal, MPI_COMM_WORLD, &req);
}

#endif