#include "pangulu_common.h"

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

pangulu_int64_t pangulu_bcast_n(pangulu_int64_t n, pangulu_int64_t send_rank)
{
    MPI_Bcast(&n, 1, MPI_PANGULU_INT64_T, send_rank, MPI_COMM_WORLD);
    return n;
}

void pangulu_bcast_vector(pangulu_inblock_ptr *vector, pangulu_int32_t length, pangulu_int64_t send_rank)
{
    pangulu_int64_t everry_length = 100000000;
    for (pangulu_int64_t i = 0; i < length; i += everry_length)
    {
        if ((i + everry_length) > length)
        {
            MPI_Bcast(vector + i, length - i, MPI_PANGULU_INBLOCK_PTR, send_rank, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Bcast(vector + i, everry_length, MPI_PANGULU_INBLOCK_PTR, send_rank, MPI_COMM_WORLD);
        }
    }
}
void pangulu_bcast_vector_int64(pangulu_int64_t *vector, pangulu_int32_t length, pangulu_int64_t send_rank)
{
    pangulu_int64_t everry_length = 100000000;
    for (pangulu_int64_t i = 0; i < length; i += everry_length)
    {
        if ((i + everry_length) > length)
        {
            MPI_Bcast(vector + i, length - i, MPI_PANGULU_INT64_T, send_rank, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Bcast(vector + i, everry_length, MPI_PANGULU_INT64_T, send_rank, MPI_COMM_WORLD);
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
void pangulu_isend_vector_char_wait(char *a, pangulu_int64_t n, pangulu_int64_t send_id, int signal, MPI_Request* req)
{
    MPI_Isend(a, n, MPI_CHAR, send_id, signal, MPI_COMM_WORLD, req);
}

void pangulu_send_vector_int(pangulu_int64_t *a, pangulu_int64_t n, pangulu_int64_t send_id, int signal)
{
    MPI_Send(a, n, MPI_PANGULU_INT64_T, send_id, signal, MPI_COMM_WORLD);
}

void pangulu_recv_vector_int(pangulu_int64_t *a, pangulu_int64_t n, pangulu_int64_t receive_id, int signal)
{
    MPI_Status status;
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        a[i] = 0;
    }
    MPI_Recv(a, n, MPI_PANGULU_INT64_T, receive_id, signal, MPI_COMM_WORLD, &status);
}

void pangulu_send_vector_char(char *a, pangulu_int64_t n, pangulu_int64_t send_id, int signal)
{
    MPI_Send(a, n, MPI_CHAR, send_id, signal, MPI_COMM_WORLD);
}

void pangulu_recv_vector_char(char *a, pangulu_int64_t n, pangulu_int64_t receive_id, int signal)
{
    MPI_Status status;
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        a[i] = 0;
    }
    pangulu_probe_message(&status);
    MPI_Recv(a, n, MPI_CHAR, receive_id, signal, MPI_COMM_WORLD, &status);
}

void pangulu_send_vector_value(calculate_type *a, pangulu_int64_t n, pangulu_int64_t send_id, int signal)
{
    MPI_Send(a, n, MPI_VAL_TYPE, send_id, signal, MPI_COMM_WORLD);
}

void pangulu_recv_vector_value(calculate_type *a, pangulu_int64_t n, pangulu_int64_t receive_id, int signal)
{
    MPI_Status status;
    for (pangulu_int64_t i = 0; i < n; i++)
    {
        a[i] = 0.0;
    }
    MPI_Recv(a, n, MPI_VAL_TYPE, receive_id, signal, MPI_COMM_WORLD, &status);
}

void pangulu_send_pangulu_smatrix_value_csr(pangulu_smatrix *s,
                                            pangulu_int64_t send_id, int signal, pangulu_int64_t nb)
{

    MPI_Send(s->value, s->nnz, MPI_VAL_TYPE, send_id, signal + 2, MPI_COMM_WORLD);
}
void pangulu_send_pangulu_smatrix_struct_csr(pangulu_smatrix *s,
                                             pangulu_int64_t send_id, int signal, pangulu_int64_t nb)
{

    MPI_Send(s->rowpointer, s->row + 1, MPI_PANGULU_INBLOCK_PTR, send_id, signal, MPI_COMM_WORLD);
    MPI_Send(s->columnindex, s->nnz, MPI_PANGULU_INBLOCK_IDX, send_id, signal + 1, MPI_COMM_WORLD);
}

void pangulu_send_pangulu_smatrix_complete_csr(pangulu_smatrix *s,
                                               pangulu_int64_t send_id, int signal, pangulu_int64_t nb)
{
    pangulu_send_pangulu_smatrix_struct_csr(s, send_id, signal * 3, nb);
    pangulu_send_pangulu_smatrix_value_csr(s, send_id, signal * 3, nb);
}

void pangulu_recv_pangulu_smatrix_struct_csr(pangulu_smatrix *s,
                                             pangulu_int64_t receive_id, int signal, pangulu_int64_t nb)
{

    MPI_Status status;
    for (pangulu_int64_t i = 0; i < (s->row + 1); i++)
    {
        s->rowpointer[i] = 0;
    }

    MPI_Recv(s->rowpointer, s->row + 1, MPI_PANGULU_INBLOCK_PTR, receive_id, signal, MPI_COMM_WORLD, &status);
    s->nnz = s->rowpointer[s->row];
    for (pangulu_int64_t i = 0; i < s->nnz; i++)
    {
        s->columnindex[i] = 0;
    }
    MPI_Recv(s->columnindex, s->nnz, MPI_PANGULU_INBLOCK_IDX, receive_id, signal + 1, MPI_COMM_WORLD, &status);
}
void pangulu_recv_pangulu_smatrix_value_csr(pangulu_smatrix *s,
                                            pangulu_int64_t receive_id, int signal, pangulu_int64_t nb)
{
    MPI_Status status;
    for (pangulu_int64_t i = 0; i < s->nnz; i++)
    {
        s->value[i] = (calculate_type)0.0;
    }
    MPI_Recv(s->value, s->nnz, MPI_VAL_TYPE, receive_id, signal + 2, MPI_COMM_WORLD, &status);
}

void pangulu_recv_pangulu_smatrix_value_csr_in_signal(pangulu_smatrix *s,
                                                      pangulu_int64_t receive_id, int signal, pangulu_int64_t nb)
{
    MPI_Status status;
    for (pangulu_int64_t i = 0; i < s->nnz; i++)
    {
        s->value[i] = (calculate_type)0.0;
    }
    MPI_Recv(s->value, s->nnz, MPI_VAL_TYPE, receive_id, signal, MPI_COMM_WORLD, &status);
}

void pangulu_recv_pangulu_smatrix_complete_csr(pangulu_smatrix *s,
                                               pangulu_int64_t receive_id, int signal, pangulu_int64_t nb)
{

    pangulu_recv_pangulu_smatrix_struct_csr(s, receive_id, signal * 3, nb);
    pangulu_recv_pangulu_smatrix_value_csr(s, receive_id, signal * 3, nb);
}

void pangulu_recv_whole_pangulu_smatrix_csr(pangulu_smatrix *s,
                                            pangulu_int64_t receive_id, int signal, pangulu_int64_t nnz, pangulu_int64_t nb)
{
#ifdef CHECK_TIME
    struct timeval GET_TIME_START;
    pangulu_time_check_begin(&GET_TIME_START);
#endif
    pangulu_int64_t length = sizeof(pangulu_inblock_ptr) * (nb + 1) + sizeof(pangulu_inblock_idx) * nnz + sizeof(calculate_type) * nnz;
    MPI_Status status;
    char *now_vector = (char *)(s->rowpointer);
    for (pangulu_int64_t i = 0; i < length; i++)
    {
        now_vector[i] = 0;
    }
    s->columnindex = (pangulu_inblock_idx *)(now_vector + sizeof(pangulu_inblock_ptr) * (nb + 1));
    s->value = (calculate_type *)(now_vector + sizeof(pangulu_inblock_ptr) * (nb + 1) + sizeof(pangulu_inblock_idx) * nnz);
    MPI_Recv(now_vector, length, MPI_CHAR, receive_id, signal, MPI_COMM_WORLD, &status);
    s->nnz = nnz;
#ifdef CHECK_TIME
    time_receive += pangulu_time_check_end(&GET_TIME_START);
#endif
}

void pangulu_send_pangulu_smatrix_value_csc(pangulu_smatrix *s,
                                            pangulu_int64_t send_id, int signal, pangulu_int64_t nb)
{
    MPI_Send(s->value_csc, s->nnz, MPI_VAL_TYPE, send_id, signal + 2, MPI_COMM_WORLD);
}

void pangulu_send_pangulu_smatrix_struct_csc(pangulu_smatrix *s,
                                             pangulu_int64_t send_id, int signal, pangulu_int64_t nb)
{

    MPI_Send(s->columnpointer, s->row + 1, MPI_PANGULU_INBLOCK_PTR, send_id, signal, MPI_COMM_WORLD);
    MPI_Send(s->rowindex, s->nnz, MPI_PANGULU_INBLOCK_IDX, send_id, signal + 1, MPI_COMM_WORLD);
}

void pangulu_send_pangulu_smatrix_complete_csc(pangulu_smatrix *s,
                                               pangulu_int64_t send_id, int signal, pangulu_int64_t nb)
{
    pangulu_send_pangulu_smatrix_struct_csc(s, send_id, signal * 3, nb);
    pangulu_send_pangulu_smatrix_value_csc(s, send_id, signal * 3, nb);
}

void pangulu_recv_pangulu_smatrix_struct_csc(pangulu_smatrix *s,
                                             pangulu_int64_t receive_id, int signal, pangulu_int64_t nb)
{

    MPI_Status status;
    for (pangulu_int64_t i = 0; i < (s->row + 1); i++)
    {
        s->columnpointer[i] = 0;
    }

    MPI_Recv(s->columnpointer, s->row + 1, MPI_PANGULU_INBLOCK_PTR, receive_id, signal, MPI_COMM_WORLD, &status);
    s->nnz = s->columnpointer[s->row];
    for (pangulu_int64_t i = 0; i < s->nnz; i++)
    {
        s->rowindex[i] = 0;
    }
    MPI_Recv(s->rowindex, s->nnz, MPI_PANGULU_INBLOCK_IDX, receive_id, signal + 1, MPI_COMM_WORLD, &status);
}
void pangulu_recv_pangulu_smatrix_value_csc(pangulu_smatrix *s,
                                            pangulu_int64_t receive_id, int signal, pangulu_int64_t nb)
{

    MPI_Status status;
    for (pangulu_int64_t i = 0; i < s->nnz; i++)
    {
        s->value_csc[i] = (calculate_type)0.0;
    }

    MPI_Recv(s->value_csc, s->nnz, MPI_VAL_TYPE, receive_id, signal + 2, MPI_COMM_WORLD, &status);
}

void pangulu_recv_pangulu_smatrix_value_csc_in_signal(pangulu_smatrix *s,
                                                      pangulu_int64_t receive_id, int signal, pangulu_int64_t nb)
{
    MPI_Status status;
    for (pangulu_int64_t i = 0; i < s->nnz; i++)
    {
        s->value_csc[i] = (calculate_type)0.0;
    }
    MPI_Recv(s->value_csc, s->nnz, MPI_VAL_TYPE, receive_id, signal, MPI_COMM_WORLD, &status);
}

void pangulu_recv_pangulu_smatrix_complete_csc(pangulu_smatrix *s,
                                               pangulu_int64_t receive_id, int signal, pangulu_int64_t nb)
{

    pangulu_recv_pangulu_smatrix_struct_csc(s, receive_id, signal * 3, nb);
    pangulu_recv_pangulu_smatrix_value_csc(s, receive_id, signal * 3, nb);
}

void pangulu_recv_whole_pangulu_smatrix_csc(pangulu_smatrix *s,
                                            pangulu_int64_t receive_id, int signal, pangulu_int64_t nnz, pangulu_int64_t nb)
{
#ifdef CHECK_TIME
    struct timeval GET_TIME_START;
    pangulu_time_check_begin(&GET_TIME_START);
#endif
    pangulu_int64_t length = sizeof(pangulu_inblock_ptr) * (nb + 1) + sizeof(pangulu_inblock_idx) * nnz + sizeof(calculate_type) * nnz;
    MPI_Status status;
    char *now_vector = (char *)(s->columnpointer);
    for (pangulu_int64_t i = 0; i < length; i++)
    {
        now_vector[i] = 0;
    }
    s->rowindex = (pangulu_inblock_idx *)(now_vector + sizeof(pangulu_inblock_ptr) * (nb + 1));
    s->value_csc = (calculate_type *)(now_vector + sizeof(pangulu_inblock_ptr) * (nb + 1) + sizeof(pangulu_inblock_idx) * nnz);
    MPI_Recv(now_vector, length, MPI_CHAR, receive_id, signal, MPI_COMM_WORLD, &status);
    s->nnz = nnz;
#ifdef CHECK_TIME
    time_receive += pangulu_time_check_end(&GET_TIME_START);
#endif
}

int pangulu_iprobe_message(MPI_Status *status)
{
    int flag;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, status);
    return flag;
}

void pangulu_isend_pangulu_smatrix_value_csr(pangulu_smatrix *s,
                                             pangulu_int64_t send_id, int signal, pangulu_int64_t nb)
{

    MPI_Request req;
    MPI_Isend(s->value, s->nnz, MPI_VAL_TYPE, send_id, signal + 2, MPI_COMM_WORLD, &req);
}
void pangulu_isend_pangulu_smatrix_struct_csr(pangulu_smatrix *s,
                                              pangulu_int64_t send_id, int signal, pangulu_int64_t nb)
{
    MPI_Request req;
    MPI_Isend(s->rowpointer, s->row + 1, MPI_PANGULU_INBLOCK_PTR, send_id, signal, MPI_COMM_WORLD, &req);
    MPI_Isend(s->columnindex, s->nnz, MPI_PANGULU_INBLOCK_IDX, send_id, signal + 1, MPI_COMM_WORLD, &req);
}

void pangulu_isend_pangulu_smatrix_complete_csr(pangulu_smatrix *s,
                                                pangulu_int64_t send_id, int signal, pangulu_int64_t nb)
{

    pangulu_isend_pangulu_smatrix_struct_csr(s, send_id, signal * 3, nb);
    pangulu_isend_pangulu_smatrix_value_csr(s, send_id, signal * 3, nb);
}

void pangulu_isend_whole_pangulu_smatrix_csr(pangulu_smatrix *s,
                                             pangulu_int64_t receive_id, int signal, pangulu_int64_t nb)
{
    pangulu_int64_t nnz = s->nnz;
    pangulu_int64_t length = sizeof(pangulu_inblock_ptr) * (nb + 1) + sizeof(pangulu_inblock_idx) * nnz + sizeof(calculate_type) * nnz;
    MPI_Request req;
    char *now_vector = (char *)(s->rowpointer);
    calculate_type *value = (calculate_type *)(now_vector + sizeof(pangulu_inblock_ptr) * (nb + 1) + sizeof(pangulu_inblock_idx) * nnz);
    if (value != s->value)
    {
        printf(PANGULU_E_ISEND_CSR);
        pangulu_exit(1);
    }
    MPI_Isend(now_vector, length, MPI_CHAR, receive_id, signal, MPI_COMM_WORLD, &req);
}

void pangulu_isend_pangulu_smatrix_value_csc(pangulu_smatrix *s,
                                             pangulu_int64_t send_id, int signal, pangulu_int64_t nb)
{
    MPI_Request req;
    MPI_Isend(s->value_csc, s->nnz, MPI_VAL_TYPE, send_id, signal + 2, MPI_COMM_WORLD, &req);
}

void pangulu_isend_pangulu_smatrix_value_csc_in_signal(pangulu_smatrix *s,
                                                       pangulu_int64_t send_id, int signal, pangulu_int64_t nb)
{
    MPI_Request req;
    MPI_Isend(s->value_csc, s->nnz, MPI_VAL_TYPE, send_id, signal, MPI_COMM_WORLD, &req);
}

void pangulu_isend_pangulu_smatrix_struct_csc(pangulu_smatrix *s,
                                              pangulu_int64_t send_id, int signal, pangulu_int64_t nb)
{
    MPI_Request req;
    MPI_Isend(s->columnpointer, s->row + 1, MPI_PANGULU_INBLOCK_PTR, send_id, signal, MPI_COMM_WORLD, &req);
    MPI_Isend(s->rowindex, s->nnz, MPI_PANGULU_INBLOCK_IDX, send_id, signal + 1, MPI_COMM_WORLD, &req);
}

void pangulu_isend_pangulu_smatrix_complete_csc(pangulu_smatrix *s,
                                                pangulu_int64_t send_id, int signal, pangulu_int64_t nb)
{

    pangulu_isend_pangulu_smatrix_struct_csc(s, send_id, signal * 3, nb);
    pangulu_isend_pangulu_smatrix_value_csc(s, send_id, signal * 3, nb);
}

void pangulu_isend_whole_pangulu_smatrix_csc(pangulu_smatrix *s,
                                             pangulu_int64_t receive_id, int signal, pangulu_int64_t nb)
{
    pangulu_int64_t nnz = s->nnz;
    pangulu_int64_t length = sizeof(pangulu_inblock_ptr) * (nb + 1) + sizeof(pangulu_inblock_idx) * nnz + sizeof(calculate_type) * nnz;
    MPI_Request req;
    char *now_vector = (char *)(s->columnpointer);
    calculate_type *value = (calculate_type *)(now_vector + sizeof(pangulu_inblock_ptr) * (nb + 1) + sizeof(pangulu_inblock_idx) * nnz);
    if (value != s->value_csc)
    {
        printf(PANGULU_E_ISEND_CSC);
        pangulu_exit(1);
    }
    MPI_Isend(now_vector, length, MPI_CHAR, receive_id, signal, MPI_COMM_WORLD, &req);
}