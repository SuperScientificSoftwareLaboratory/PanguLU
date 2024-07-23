#include "pangulu.h"
#include <getopt.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include "mmio_highlevel.h"

void read_command_params(int argc, char** argv, char* mtx_name, char* rhs_name, int* nb)
{
    int c;
    extern char *optarg;
    while ((c = getopt(argc, argv, "NB:F:R:")) != EOF)
    {
        switch (c)
        {
        case 'B':
            *nb = atoi(optarg);
            continue;
        case 'F':
            strcpy(mtx_name, optarg);
            continue;
        case 'R':
            strcpy(rhs_name, optarg);
            continue;
        }
    }
    if ((nb) == 0)
    {
        printf("Error : NB is 0\n");
        exit(1);
    }
}

int main(int ARGC ,char **ARGV){
    // Step 1: Create varibles, initialize MPI environment.
    int provided = 0;
    int RANK = 0, SIZE = 0;
    MPI_Init_thread(&ARGC, &ARGV, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &SIZE);
    long m = 0, n = 0, nnz = 0, isSym = 0;
    long* rowptr = NULL;
    int* colidx = NULL;
    double* value = NULL;
    double* sol = NULL;
    double* rhs = NULL;
    int nb = 0;


    // Step 2: Read matrix and rhs vectors.
    if(RANK==0){
        char mtx_name[200];
        char rhs_name[200];
        read_command_params(ARGC, ARGV, mtx_name, rhs_name, &nb);

        printf("Reading matrix %s\n", mtx_name);
        mmio_info(&m, &n, &nnz, &isSym, mtx_name);
        rowptr = (long*)malloc(sizeof(long)*(n+1));
        colidx = (int*)malloc(sizeof(int)*nnz);
        value = (double*)malloc(sizeof(double)*nnz);
        mmio_data_csr(rowptr, colidx, value, mtx_name);
        printf("Read mtx done.\n");

        sol = (double*)malloc(sizeof(double)*n);
        rhs = (double*)malloc(sizeof(double)*n);
        for(int i=0;i<n;i++){
            rhs[i] = 0;
            for(long long j=rowptr[i]; j<rowptr[i+1]; j++){
                rhs[i] += value[j];
            }
            sol[i] = rhs[i];
        }
        printf("Generate rhs done.\n");
    }
    MPI_Bcast(&n, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nb, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);


    // Step 3: Initialize PanguLU solver.
    pangulu_init_options init_options;
    init_options.nb = nb;
    init_options.nthread = 20;
    void* pangulu_handler;
    pangulu_init(n, nnz, rowptr, colidx, value, &init_options, &pangulu_handler);


    // Step 4: Execute LU factorization.
    pangulu_gstrf_options gstrf_options;
    gstrf_options.tol = 1e-8;
    pangulu_gstrf(&gstrf_options, &pangulu_handler);


    // Step 5: Execute triangle solve using factorize results.
    pangulu_gstrs_options gstrs_options;
    gstrs_options.nrhs = 1;
    pangulu_gstrs(sol ,&gstrs_options, &pangulu_handler);
    MPI_Barrier(MPI_COMM_WORLD);


    // Step 6: Check the answer. 
    double* rhs_computed;
    if(RANK==0){
        // Step 6.1: Calculate rhs_computed = A * x.
        rhs_computed = (double*)malloc(sizeof(double)*n);
        for(int i=0;i<n;i++){
            rhs_computed[i] = 0.0;
            double c = 0.0;
            for(long long int j=rowptr[i]; j<rowptr[i+1]; j++){
                double num = value[j] * sol[colidx[j]];
                double z = num - c;
                double t = rhs_computed[i] + z;
                c = (t - rhs_computed[i]) - z;
                rhs_computed[i] = t;
            }
        }
        
        // Step 6.2: Calculate residual residual = rhs_comuted - rhs.
        double* residual = rhs_computed;
        for(int i=0;i<n;i++){
            residual[i] = rhs_computed[i] - rhs[i];
        }
        
        double sum, c;
        // Step 6.2: Calculte norm2 of residual.
        sum = 0.0;
        c   = 0.0;
        for (int i = 0; i < n; i++) {
            double num = residual[i] * residual[i];
            double z   = num - c;
            double t   = sum + z;
            c               = (t - sum) - z;
            sum             = t;
        }
        double residual_norm2 = sqrt(sum);

        // Step 6.3: Calculte norm2 of original rhs.
        sum = 0.0;
        c   = 0.0;
        for (int i = 0; i < n; i++) {
            double num = rhs[i] * rhs[i];
            double z   = num - c;
            double t   = sum + z;
            c               = (t - sum) - z;
            sum             = t;
        }
        double rhs_norm2 = sqrt(sum);

        // Step 6.4: Calculate relative residual.
        printf("|| Ax - b || / || b || = %le\n", residual_norm2 / rhs_norm2);
    }
    
    
    // Step 7: Clean and finalize.
    if(RANK=0){
        free(rowptr);
        free(colidx);
        free(value);
        free(sol);
        free(rhs);
        free(rhs_computed);
    }
    MPI_Finalize();
}
