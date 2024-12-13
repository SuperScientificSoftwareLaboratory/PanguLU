#include "pangulu_common.h"

void at_plus_a_dist(
    const pangulu_exblock_idx n,    /* number of columns in reorder_matrix A. */
    const pangulu_exblock_ptr nz,   /* number of nonzeros in reorder_matrix A */
    pangulu_exblock_ptr *colptr,    /* column pointer of size n+1 for reorder_matrix A. */
    pangulu_exblock_idx *rowind,    /* row indices of size nz for reorder_matrix A. */
    pangulu_exblock_ptr *bnz,       /* out - on exit, returns the actual number of nonzeros in reorder_matrix A'+A. */
    pangulu_exblock_ptr **b_colptr, /* out - size n+1 */
    pangulu_exblock_idx **b_rowind  /* out - size *bnz */
)
{

    register pangulu_exblock_idx i, j, k, col; 
    register pangulu_exblock_ptr num_nz;
    pangulu_exblock_ptr *t_colptr;
    pangulu_exblock_idx *t_rowind; /* a column oriented form of T = A' */
    pangulu_int32_t *marker;


    marker = (pangulu_int32_t*)pangulu_malloc(__FILE__, __LINE__,  n * sizeof(pangulu_int32_t));
    t_colptr = (pangulu_exblock_ptr*)pangulu_malloc(__FILE__, __LINE__,  (n+1) * sizeof(pangulu_exblock_ptr));
    t_rowind = (pangulu_exblock_idx*)pangulu_malloc(__FILE__, __LINE__,  nz * sizeof(pangulu_exblock_idx));

    /* Get counts of each column of T, and set up column pointers */
    for (i = 0; i < n; ++i) marker[i] = 0;
    for (j = 0; j < n; ++j) {
	for (i = colptr[j]; i < colptr[j+1]; ++i)
	    ++marker[rowind[i]];
    }

    t_colptr[0] = 0;
    for (i = 0; i < n; ++i) {
	t_colptr[i+1] = t_colptr[i] + marker[i];
	marker[i] = t_colptr[i];
    }

    /* Transpose the reorder_matrix from A to T */
    for (j = 0; j < n; ++j) {
	for (i = colptr[j]; i < colptr[j+1]; ++i) {
	    col = rowind[i];
	    t_rowind[marker[col]] = j;
	    ++marker[col]; 
	}
    }

    /* ----------------------------------------------------------------
       compute B = A + T, where column j of B is:

       Struct (B_*j) = Struct (A_*k) UNION Struct (T_*k)

       do not include the diagonal entry
       ---------------------------------------------------------------- */

    /* Zero the diagonal flag */
    for (i = 0; i < n; ++i) marker[i] = -1;

    /* First pass determines number of nonzeros in B */
    num_nz = 0;
    for (j = 0; j < n; ++j) {
	/* Flag the diagonal so it's not included in the B reorder_matrix */
	//marker[j] = j;

	/* Add pattern of column A_*k to B_*j */
	for (i = colptr[j]; i < colptr[j+1]; ++i) {
	    k = rowind[i];
	    if ( marker[k] != j ) {
		marker[k] = j;
		++num_nz;
	    }
	}

	/* Add pattern of column T_*k to B_*j */
	for (i = t_colptr[j]; i < t_colptr[j+1]; ++i) {
	    k = t_rowind[i];
	    if ( marker[k] != j ) {
		marker[k] = j;
		++num_nz;
	    }
	}
    }
    *bnz = num_nz;


    /* Allocate storage for A+A' */
    *b_colptr = (pangulu_exblock_ptr*)pangulu_malloc(__FILE__, __LINE__,  (n+1) * sizeof(pangulu_exblock_ptr));
    *b_rowind = (pangulu_exblock_idx*)pangulu_malloc(__FILE__, __LINE__,  *bnz * sizeof(pangulu_exblock_idx));

    /* Zero the diagonal flag */
    for (i = 0; i < n; ++i) marker[i] = -1;
    
    /* Compute each column of B, one at a time */
    num_nz = 0;
    for (j = 0; j < n; ++j) {
	(*b_colptr)[j] = num_nz;
	
	/* Flag the diagonal so it's not included in the B reorder_matrix */
	//marker[j] = j;

	/* Add pattern of column A_*k to B_*j */
	for (i = colptr[j]; i < colptr[j+1]; ++i) {
	    k = rowind[i];
	    if ( marker[k] != j ) {
		marker[k] = j;
		(*b_rowind)[num_nz++] = k;
	    }
	}

	/* Add pattern of column T_*k to B_*j */
	for (i = t_colptr[j]; i < t_colptr[j+1]; ++i) {
	    k = t_rowind[i];
	    if ( marker[k] != j ) {
		marker[k] = j;
		(*b_rowind)[num_nz++] = k;
	    }
	}
    }
    (*b_colptr)[n] = num_nz;
       
    pangulu_free(__FILE__, __LINE__, marker);
    pangulu_free(__FILE__, __LINE__, t_colptr);
    pangulu_free(__FILE__, __LINE__, t_rowind);
} /* at_plus_a_dist */

void add_prune(node *prune,node *prune_next,pangulu_int64_t num,pangulu_int64_t num_value,pangulu_int64_t p)
{
    prune[num].value++;
    prune_next[p].value=num_value;
    prune_next[p].next=NULL;
    node *p2=&prune[num];
   for(;;)
   {
        if(p2->next==NULL)
        {
            break;
        }
        p2=p2->next;
   }
   p2->next= &prune_next[p];
}


void fill_in_sym_prune(
    pangulu_exblock_idx n,pangulu_exblock_ptr nnz,pangulu_exblock_idx *ai, pangulu_exblock_ptr *ap,
    pangulu_exblock_ptr **symbolic_rowpointer,pangulu_exblock_idx **symbolic_columnindex,
    pangulu_inblock_idx nb,pangulu_exblock_idx block_length,
    pangulu_inblock_ptr *block_smatrix_non_zero_vector_L,
    pangulu_inblock_ptr *block_smatrix_non_zero_vector_U,
    pangulu_inblock_ptr *block_smatrix_nnzA_num,
    pangulu_exblock_ptr *symbolic_nnz
)
{
    pangulu_exblock_ptr relloc_zjq=nnz;
    pangulu_exblock_idx  *L_r_idx=(pangulu_exblock_idx*)pangulu_malloc(__FILE__, __LINE__, relloc_zjq*sizeof(pangulu_exblock_idx));//include diagonal
    pangulu_exblock_ptr  *L_c_ptr= (pangulu_exblock_ptr*)pangulu_malloc(__FILE__, __LINE__, (n+1)*sizeof(pangulu_exblock_ptr));
    L_c_ptr[0]=0;

    node *prune=(node*)pangulu_malloc(__FILE__, __LINE__, n*sizeof(node));
    node *prune_next=(node*)pangulu_malloc(__FILE__, __LINE__, n*sizeof(node));
    node *p1;

    pangulu_int64_t *work_space = (pangulu_int64_t*)pangulu_malloc(__FILE__, __LINE__, n*sizeof(pangulu_int64_t));
    pangulu_int64_t *merge=(pangulu_int64_t*)pangulu_malloc(__FILE__, __LINE__, n*sizeof(pangulu_int64_t));

    for(pangulu_exblock_idx i = 0;i<n;i++)
    {
        work_space[i]=-1;
        prune[i].value=0;
        prune[i].next=NULL;
        prune_next[i].value=-1;
        prune_next[i].next=NULL;
    }
    pangulu_int64_t L_maxsize=relloc_zjq;
    pangulu_int64_t L_size=0;

    pangulu_int64_t row=-1;
    pangulu_int64_t num_merge=0;
    
    pangulu_int64_t p=0;
    for (pangulu_exblock_idx i = 0; i < n; i++)
    {

        pangulu_exblock_idx n_rows=ap[i+1]-ap[i];

        for(pangulu_exblock_idx k=0;k<n_rows;k++)
        {
            
            row = (ai+ap[i])[k];
            if(row >= i)
            {
                work_space[row]=i;
                L_r_idx[L_size]=row;
                L_size++;
                block_smatrix_nnzA_num[(i/nb)*block_length+(row/nb)]++;
                if(L_size + 1 > L_maxsize)
                {
                    L_r_idx=(pangulu_exblock_idx*)pangulu_realloc(__FILE__, __LINE__, L_r_idx,(L_maxsize+relloc_zjq)*sizeof(pangulu_exblock_idx));
                    L_maxsize=L_maxsize+relloc_zjq;
                }

            }
        }

        num_merge=prune[i].value;
        p1 = &prune[i];
        for(pangulu_int64_t k=0;;k++)
        {
            if(p1->next==NULL)
            break;
            p1=p1->next;
            merge[k]=p1->value;
        }
        for(pangulu_int64_t k=0;k<num_merge;k++)
        {
            row=merge[k];
            pangulu_int64_t min=L_c_ptr[row];
            pangulu_int64_t max=L_c_ptr[row+1];
            for(pangulu_int64_t j=min;j<max;j++)
            {
                pangulu_int64_t crow=L_r_idx[j];

                if(crow>i&&work_space[crow]!=i)
                {
                    work_space[crow]=i;
                    L_r_idx[L_size]=crow;
                    L_size++;
                    block_smatrix_nnzA_num[(i/nb)*block_length+(crow/nb)]++;
                    if(L_size + 1 > L_maxsize)
                    {
                        L_r_idx=(pangulu_exblock_idx*)pangulu_realloc(__FILE__, __LINE__, L_r_idx,(L_maxsize+relloc_zjq)*sizeof(pangulu_exblock_idx));
                        L_maxsize=L_maxsize+relloc_zjq;
                    }
                }
                
            }
        }
        L_c_ptr[i+1]=L_size;

        if(L_c_ptr[i+1]-L_c_ptr[i]>1)
        {
            pangulu_int64_t todo_prune=n+1;
            for(pangulu_int64_t k=L_c_ptr[i];k<L_c_ptr[i+1];k++)
            {
                if(todo_prune>L_r_idx[k]&&L_r_idx[k]>i)
                todo_prune=L_r_idx[k];
            }
            add_prune(prune,prune_next,todo_prune,i,p);
            p++;
        }
        
        
    }
    pangulu_free(__FILE__, __LINE__, work_space);
    pangulu_free(__FILE__, __LINE__, merge);
    *symbolic_nnz = L_size*2-n;
    printf(PANGULU_I_SYMBOLIC_NONZERO);
    for(int i=0;i<block_length;i++)
    {
        for(int j=0;j<block_length;j++)
        {
            if(j==i)
            {
                block_smatrix_non_zero_vector_U[i]=block_smatrix_nnzA_num[i*block_length+j];
                block_smatrix_non_zero_vector_L[i]=block_smatrix_nnzA_num[i*block_length+j];
                block_smatrix_nnzA_num[i*block_length+j]=block_smatrix_nnzA_num[i*block_length+j]*2-nb;
            }
            if(j>i){
                block_smatrix_nnzA_num[j*block_length+i]=block_smatrix_nnzA_num[i*block_length+j];
            }
        }
    }
    block_smatrix_nnzA_num[block_length*block_length-1]=block_smatrix_nnzA_num[block_length*block_length-1]+(nb-n%nb);

    pangulu_free(__FILE__, __LINE__, prune);
    pangulu_free(__FILE__, __LINE__, prune_next);

    *symbolic_rowpointer=L_c_ptr;
    *symbolic_columnindex=L_r_idx;
}

pangulu_int32_t pruneL(
    pangulu_exblock_idx jcol,
    pangulu_exblock_idx *U_r_idx,
    pangulu_exblock_ptr *U_c_ptr,
    pangulu_exblock_idx *L_r_idx,
    pangulu_exblock_ptr *L_c_ptr,
    pangulu_int64_t *work_space,
    pangulu_int64_t *prune_space
)
{
    if(jcol == 0)
    return 0;

    pangulu_int64_t min=U_c_ptr[jcol];
    pangulu_int64_t max=U_c_ptr[jcol+1];
    pangulu_int64_t cmin,cmax,crow,doprune;
    doprune=0;

    for(pangulu_int64_t i=min;i<max;i++)
    {
        doprune=0;
        crow=U_r_idx[i];
        cmin=L_c_ptr[crow];
        cmax=prune_space[crow];
        for(pangulu_int64_t j=cmin;j<cmax;j++)
        {
            if(L_r_idx[j]==jcol)
            {
                doprune=1;
                break;
            }
        }

        if(doprune==1)
        {
            for(pangulu_int64_t j=cmin;j<cmax;j++)
            {
                pangulu_int32_t ccrow=L_r_idx[j];
                if(ccrow>jcol&&work_space[ccrow]==jcol)
                {
                    
                    pangulu_int32_t temp=L_r_idx[cmax-1];
                    L_r_idx[cmax-1]=L_r_idx[j];
                    L_r_idx[j]=temp;
                    cmax--;
                    j--;
                }
            }
        }
        prune_space[crow]=cmax;
    }
    return 0;
}

void fill_in_2_no_sort_pruneL(
    pangulu_exblock_idx n,
    pangulu_exblock_ptr nnz,
    pangulu_exblock_idx *ai, 
    pangulu_exblock_ptr *ap,
    pangulu_exblock_ptr **L_rowpointer,
    pangulu_exblock_idx **L_columnindex,
    pangulu_exblock_ptr **U_rowpointer,
    pangulu_exblock_idx **U_columnindex
)
{
    pangulu_exblock_ptr relloc_zjq=nnz;
    pangulu_exblock_idx  *U_r_idx=(pangulu_exblock_idx*)pangulu_malloc(__FILE__, __LINE__, relloc_zjq*sizeof(pangulu_exblock_idx));//exclude diagonal
    pangulu_exblock_ptr  *U_c_ptr= (pangulu_exblock_ptr*)pangulu_malloc(__FILE__, __LINE__, (n+1)*sizeof(pangulu_exblock_ptr));
    pangulu_exblock_idx  *L_r_idx=(pangulu_exblock_idx*)pangulu_malloc(__FILE__, __LINE__, relloc_zjq*sizeof(pangulu_exblock_idx));//include diagonal
    pangulu_exblock_ptr  *L_c_ptr= (pangulu_exblock_ptr*)pangulu_malloc(__FILE__, __LINE__, (n+1)*sizeof(pangulu_exblock_ptr));
    U_c_ptr[0]=0;
    L_c_ptr[0]=0;

    pangulu_int64_t *parent= (pangulu_int64_t*)pangulu_malloc(__FILE__, __LINE__, (n+1)*sizeof(pangulu_int64_t));//for dfs
    pangulu_int64_t *xplore = (pangulu_int64_t*)pangulu_malloc(__FILE__, __LINE__, (n+1)*sizeof(pangulu_int64_t));

    pangulu_int64_t *work_space= (pangulu_int64_t*)pangulu_malloc(__FILE__, __LINE__, n*sizeof(pangulu_int64_t)); //use this to avoid sorting
    pangulu_int64_t *prune_space=(pangulu_int64_t*)pangulu_malloc(__FILE__, __LINE__, n*sizeof(pangulu_int64_t));

    pangulu_int64_t U_maxsize=relloc_zjq;
    pangulu_int64_t L_maxsize=relloc_zjq;

    pangulu_int64_t U_size=0;
    pangulu_int64_t L_size=0;//record quantity

    pangulu_int32_t row=-1;
    pangulu_int32_t oldrow=-1;
    pangulu_int64_t xdfs=-1;
    pangulu_int64_t maxdfs=-1;
    pangulu_int32_t kchild=-1;
    pangulu_int32_t kpar=-1;

    for(pangulu_int64_t k=0;k<n;k++)
    {
        work_space[k]=-1;  //avoid conflict
        parent[k]=-1;
        xplore[k]=0;
    }

    for (pangulu_int64_t i = 0; i < n; i++)
    {
    

        pangulu_int64_t n_rows=ap[i+1]-ap[i];        
    
        for(pangulu_int64_t k=0;k<n_rows;k++)
        {
            row = (ai+ap[i])[k];            
            if(work_space[row]==i)
            continue;

            work_space[row]=i;
            if(row >= i)
            {
                L_r_idx[L_size]=row;
                L_size++;

                if(L_size >= L_maxsize-100)
                {
                    L_r_idx=(pangulu_exblock_idx*)pangulu_realloc(__FILE__, __LINE__, L_r_idx,(L_maxsize+relloc_zjq)*sizeof(pangulu_exblock_idx));
                    L_maxsize=L_maxsize+nnz;
                }
            }
            else
            {
                U_r_idx[U_size]=row;
                U_size++;
                if(U_size >= U_maxsize-100)
                {
                    U_r_idx=(pangulu_exblock_idx*)pangulu_realloc(__FILE__, __LINE__, U_r_idx,(U_maxsize+relloc_zjq)*sizeof(pangulu_exblock_idx));
                    U_maxsize=U_maxsize+nnz;
                }
                 //do dfs
                oldrow = -1;
                parent[row] = oldrow;
                xdfs = L_c_ptr[row];
                maxdfs = prune_space[row];//prune
                do
                {
                    /* code */
                    while(xdfs < maxdfs)
                    {
                    
                        kchild=L_r_idx[xdfs];
                        xdfs++;
                        
                        if(work_space[kchild]!=i)
                        {
        
                            work_space[kchild]=i;
                            if(kchild>=i)
                            {
                                L_r_idx[L_size]=kchild;
                                L_size++;
                                if(L_size >= L_maxsize-100)
                                {
                                    L_r_idx=(pangulu_exblock_idx*)pangulu_realloc(__FILE__, __LINE__, L_r_idx,(L_maxsize+relloc_zjq)*sizeof(pangulu_exblock_idx));
                                    L_maxsize=L_maxsize+nnz;
                                } 
                            }
                            else
                            {
                                U_r_idx[U_size]=kchild;
                                U_size++;
                                if(U_size >= U_maxsize-100)
                                {
                                    U_r_idx=(pangulu_exblock_idx*)pangulu_realloc(__FILE__, __LINE__, U_r_idx,(U_maxsize+relloc_zjq)*sizeof(pangulu_exblock_idx));
                                    U_maxsize=U_maxsize+nnz;
                                }

                                xplore[row]=xdfs;
                                oldrow = row;
                                row = kchild;
                                parent[row]=oldrow;
                                xdfs =  L_c_ptr[row];
                                maxdfs = prune_space[row];//prune


                            }

                        }
                    }
               
                    kpar = parent[row];

                    if(kpar == -1)
                    break;
                    
                    row = kpar;
                    xdfs = xplore[row];
                    maxdfs = prune_space[row];



                } while (kpar != -1);
                
            }

        }
        U_c_ptr[i+1]=U_size;
        L_c_ptr[i+1]=L_size;
        prune_space[i]=L_size;

        pruneL(i,U_r_idx,U_c_ptr,L_r_idx,L_c_ptr,work_space,prune_space); 
        
    }
    pangulu_sort_pangulu_matrix(n,L_c_ptr,L_r_idx);
    pangulu_sort_pangulu_matrix(n,U_c_ptr,U_r_idx);

    pangulu_free(__FILE__, __LINE__, parent);
    pangulu_free(__FILE__, __LINE__, xplore);
    pangulu_free(__FILE__, __LINE__, work_space);
    pangulu_free(__FILE__, __LINE__, prune_space);
    
    *L_rowpointer=L_c_ptr;
    *L_columnindex=L_r_idx;
    *U_rowpointer=U_c_ptr;
    *U_columnindex=U_r_idx;
}

void pangulu_symbolic(pangulu_block_common *block_common,
                      pangulu_block_smatrix *block_smatrix,
                      pangulu_origin_smatrix *reorder_matrix){
#ifndef symmetric    
    fill_in_2_no_sort_pruneL(reorder_matrix->row,reorder_matrix->nnz,reorder_matrix->columnindex,reorder_matrix->rowpointer,L_rowpointer,L_columnindex,U_rowpointer,U_columnindex);
#else
    pangulu_exblock_ptr *new_rowpointer=NULL;
    pangulu_exblock_idx *new_columnindex=NULL;
    pangulu_exblock_ptr new_nnz;
    struct timeval start_time;
    pangulu_time_start(&start_time);
    at_plus_a_dist(reorder_matrix->row,reorder_matrix->nnz,
                   reorder_matrix->rowpointer,reorder_matrix->columnindex,
                   &new_nnz,&new_rowpointer,&new_columnindex);
    printf("[PanguLU Info] 4 PanguLU A+AT (before symbolic) time is %lf s.\n", pangulu_time_stop(&start_time));
    pangulu_exblock_ptr *symbolic_rowpointer = NULL;
    pangulu_exblock_idx *symbolic_columnindex = NULL;
    pangulu_inblock_idx nb = block_common->nb;
    pangulu_exblock_idx block_length = block_common->block_length;
    pangulu_inblock_ptr *block_smatrix_nnzA_num = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * block_length * block_length);
    pangulu_inblock_ptr *block_smatrix_non_zero_vector_L = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * block_length);
    pangulu_inblock_ptr *block_smatrix_non_zero_vector_U = (pangulu_inblock_ptr *)pangulu_malloc(__FILE__, __LINE__, sizeof(pangulu_inblock_ptr) * block_length);
    
    for (pangulu_exblock_idx i = 0; i < block_length; i++)
    {
        block_smatrix_non_zero_vector_L[i] = 0;
    }

    for (pangulu_exblock_idx i = 0; i < block_length; i++)
    {
        block_smatrix_non_zero_vector_U[i] = 0;
    }
    for (pangulu_exblock_idx i = 0; i < block_length * block_length; i++)
    {
        block_smatrix_nnzA_num[i] = 0;
    }

    pangulu_time_start(&start_time);
    fill_in_sym_prune(reorder_matrix->row,new_nnz,new_columnindex,new_rowpointer,
                    &symbolic_rowpointer,&symbolic_columnindex,
                    nb,block_length,
                    block_smatrix_non_zero_vector_L,
                    block_smatrix_non_zero_vector_U,
                    block_smatrix_nnzA_num,
                    &block_smatrix->symbolic_nnz
                    );
    printf("[PanguLU Info] 5 PanguLU symbolic time is %lf s.\n", pangulu_time_stop(&start_time));
    
    pangulu_free(__FILE__, __LINE__, new_rowpointer);
    pangulu_free(__FILE__, __LINE__, new_columnindex);
    
    block_smatrix->block_smatrix_nnza_num=block_smatrix_nnzA_num;
    block_smatrix->block_smatrix_non_zero_vector_l=block_smatrix_non_zero_vector_L;
    block_smatrix->block_smatrix_non_zero_vector_u=block_smatrix_non_zero_vector_U;
    block_smatrix->symbolic_rowpointer=symbolic_rowpointer;
    block_smatrix->symbolic_columnindex=symbolic_columnindex;

#endif  

}
