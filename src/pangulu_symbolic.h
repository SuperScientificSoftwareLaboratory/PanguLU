#ifndef PANGULU_SYMBOLIC_H

#define PANGULU_SYMBOLIC_H

#include "pangulu_common.h"
#include "pangulu_utils.h"


void
at_plus_a_dist(
	       const int_32t n,    /* number of columns in reorder_matrix A. */
	       const int_t nz,   /* number of nonzeros in reorder_matrix A */
	       int_t *colptr,    /* column pointer of size n+1 for reorder_matrix A. */
	       int_32t *rowind,    /* row indices of size nz for reorder_matrix A. */
	       int_t *bnz,       /* out - on exit, returns the actual number of
				    nonzeros in reorder_matrix A'+A. */
	       int_t **b_colptr, /* out - size n+1 */
	       int_32t **b_rowind  /* out - size *bnz */
	       )
{

    register int_32t i, j, k, col; 
    register int_t num_nz;
    int_t *t_colptr;
    int_32t *t_rowind; /* a column oriented form of T = A' */
    int_32t *marker;


    marker = (int_32t*)pangulu_malloc(__FILE__, __LINE__,  n * sizeof(int_32t));
    t_colptr = (int_t*)pangulu_malloc(__FILE__, __LINE__,  (n+1) * sizeof(int_t));
    t_rowind = (int_32t*)pangulu_malloc(__FILE__, __LINE__,  nz * sizeof(int_32t));

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
    *b_colptr = (int_t*)pangulu_malloc(__FILE__, __LINE__,  (n+1) * sizeof(int_t));
    *b_rowind = (int_32t*)pangulu_malloc(__FILE__, __LINE__,  *bnz * sizeof(int_32t));

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

typedef struct node{
	int_t value;
	struct node *next;
}node;
void add_prune(node *prune,node *prune_next,int_t num,int_t num_value,int_t p)
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


void fill_in_sym_prune(int_t n,int_t nnz,int_32t *ai, int_t *ap,
                        int_t **symbolic_rowpointer,int_32t **symbolic_columnindex,
                        int_t N,int_32t NB,int_32t block_length,
                        int_t *block_Smatrix_non_zero_vector_L,
                        int_t *block_Smatrix_non_zero_vector_U,
                        int_t *block_Smatrix_nnzA_num,
                        int_t *symbolic_nnz)
{
    int_t relloc_zjq=nnz;
    int_32t  *L_r_idx=(int_32t*)pangulu_malloc(__FILE__, __LINE__, relloc_zjq*sizeof(int_32t));//include diagonal
    int_t  *L_c_ptr= (int_t*)pangulu_malloc(__FILE__, __LINE__, (n+1)*sizeof(int_t));
    L_c_ptr[0]=0;

    node *prune=(node*)pangulu_malloc(__FILE__, __LINE__, n*sizeof(node));
    node *prune_next=(node*)pangulu_malloc(__FILE__, __LINE__, n*sizeof(node));
    node *p1;

    int_t *work_space = (int_t*)pangulu_malloc(__FILE__, __LINE__, n*sizeof(int_t));
    int_t *merge=(int_t*)pangulu_malloc(__FILE__, __LINE__, n*sizeof(int_t));

    for(int_t i = 0;i<n;i++)
    {
        work_space[i]=-1;
        prune[i].value=0;
        prune[i].next=NULL;
        prune_next[i].value=-1;
        prune_next[i].next=NULL;
    }
    int_t L_maxsize=relloc_zjq;
    int_t L_size=0;

    int_t row=-1;
    int_t num_merge=0;
    
    int_t p=0;
        for (int_32t i = 0; i < n; i++)
        {
        

            int_32t n_rows=ap[i+1]-ap[i];

            for(int_32t k=0;k<n_rows;k++)
            {
                
                row = (ai+ap[i])[k];
                if(row >= i)
                {
                    work_space[row]=i;
                    L_r_idx[L_size]=row;
                    L_size++;
                    block_Smatrix_nnzA_num[(i/NB)*block_length+(row/NB)]++;
                    if(L_size >= L_maxsize-100)
                    {
                        L_r_idx=(int_32t*)pangulu_realloc(__FILE__, __LINE__, L_r_idx,(L_maxsize+relloc_zjq)*sizeof(int_32t));
                        L_maxsize=L_maxsize+relloc_zjq;
                    }

                }
            }

            num_merge=prune[i].value;
             p1 = &prune[i];
            for(int_t k=0;;k++)
            {
                if(p1->next==NULL)
                break;
                p1=p1->next;
                merge[k]=p1->value;
            }
            for(int_t k=0;k<num_merge;k++)
            {
                row=merge[k];
                // 
                int_t min=L_c_ptr[row];
                int_t max=L_c_ptr[row+1];
                for(int_t j=min;j<max;j++)
                {
                    int_t crow=L_r_idx[j];

                    if(crow>i&&work_space[crow]!=i)
                    {
                        work_space[crow]=i;
                        L_r_idx[L_size]=crow;
                        L_size++;
                        block_Smatrix_nnzA_num[(i/NB)*block_length+(crow/NB)]++;
                        if(L_size >= L_maxsize-100)
                        {
                            L_r_idx=(int_32t*)pangulu_realloc(__FILE__, __LINE__, L_r_idx,(L_maxsize+relloc_zjq)*sizeof(int_32t));
                            L_maxsize=L_maxsize+relloc_zjq;
                        }
                    }
                    
                }
            }
            L_c_ptr[i+1]=L_size;

            if(L_c_ptr[i+1]-L_c_ptr[i]>1)
            {
                int_t todo_prune=n+1;
                for(int_t k=L_c_ptr[i];k<L_c_ptr[i+1];k++)
                {
                    if(todo_prune>L_r_idx[k]&&L_r_idx[k]>i)
                    todo_prune=L_r_idx[k];
                }
                add_prune(prune,prune_next,todo_prune,i,p);
                p++;
            }
            
            
        }
        printf("Symbolic nonzero = %ld\n",L_size*2-n);        
        *symbolic_nnz = L_size*2-n;
        for(int i=0;i<block_length;i++)
        {
            for(int j=0;j<block_length;j++)
            {
                if(j==i)
                {
                    block_Smatrix_non_zero_vector_U[i]=block_Smatrix_nnzA_num[i*block_length+j];
                    block_Smatrix_non_zero_vector_L[i]=block_Smatrix_nnzA_num[i*block_length+j];
                    block_Smatrix_nnzA_num[i*block_length+j]=block_Smatrix_nnzA_num[i*block_length+j]*2-NB;
                }
                if(j>i)
                block_Smatrix_nnzA_num[j*block_length+i]=block_Smatrix_nnzA_num[i*block_length+j];
            }
        }
        block_Smatrix_nnzA_num[block_length*block_length-1]=block_Smatrix_nnzA_num[block_length*block_length-1]+(NB-N%NB);

        pangulu_free(__FILE__, __LINE__, prune);
        pangulu_free(__FILE__, __LINE__, prune_next);

        *symbolic_rowpointer=L_c_ptr;
        *symbolic_columnindex=L_r_idx;
}

int_32t pruneL(int_32t jcol,int_32t *U_r_idx,int_t *U_c_ptr,int_32t *L_r_idx,int_t *L_c_ptr,int_t *work_space,int_t *prune_space)
{
    if(jcol == 0)
    return 0;

    int_t min=U_c_ptr[jcol];
    int_t max=U_c_ptr[jcol+1];
    int_t cmin,cmax,crow,doprune;
    doprune=0;

    for(int_t i=min;i<max;i++)
    {
        doprune=0;
        crow=U_r_idx[i];
        cmin=L_c_ptr[crow];
        cmax=prune_space[crow];
        for(int_t j=cmin;j<cmax;j++)
        {
            if(L_r_idx[j]==jcol)
            {
                doprune=1;
                break;
            }
        }

        if(doprune==1)
        {
            for(int_t j=cmin;j<cmax;j++)
            {
                int_32t ccrow=L_r_idx[j];
                if(ccrow>jcol&&work_space[ccrow]==jcol)
                {
                    
                    int_32t temp=L_r_idx[cmax-1];
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
void fill_in_2_no_sort_pruneL(int_t n,int_t nnz,int_32t *ai, int_t *ap,int_t **L_rowpointer,int_32t **L_columnindex,int_t **U_rowpointer,int_32t **U_columnindex)
{
    int_32t relloc_zjq=nnz;
    int_32t  *U_r_idx=(int_32t*)pangulu_malloc(__FILE__, __LINE__, relloc_zjq*sizeof(int_32t));//exclude diagonal
    int_t  *U_c_ptr= (int_t*)pangulu_malloc(__FILE__, __LINE__, (n+1)*sizeof(int_t));
    int_32t  *L_r_idx=(int_32t*)pangulu_malloc(__FILE__, __LINE__, relloc_zjq*sizeof(int_32t));//include diagonal
    int_t  *L_c_ptr= (int_t*)pangulu_malloc(__FILE__, __LINE__, (n+1)*sizeof(int_t));
    U_c_ptr[0]=0;
    L_c_ptr[0]=0;

    int_t *parent= (int_t*)pangulu_malloc(__FILE__, __LINE__, (n+1)*sizeof(int_t));//for dfs
    int_t *xplore = (int_t*)pangulu_malloc(__FILE__, __LINE__, (n+1)*sizeof(int_t));

    int_t *work_space= (int_t*)pangulu_malloc(__FILE__, __LINE__, n*sizeof(int_t)); //use this to avoid sorting
    int_t *prune_space=(int_t*)pangulu_malloc(__FILE__, __LINE__, n*sizeof(int_t));

    int_t U_maxsize=relloc_zjq;
    int_t L_maxsize=relloc_zjq;

    int_t U_size=0;
    int_t L_size=0;//record quantity

    int_32t row=-1;
    int_32t oldrow=-1;
    int_t xdfs=-1;
    int_t maxdfs=-1;
    int_32t kchild=-1;
    int_32t kpar=-1;

    for(int_t k=0;k<n;k++)
    {
        work_space[k]=-1;  //avoid conflict
        parent[k]=-1;
        xplore[k]=0;
    }

    for (int_t i = 0; i < n; i++)
    {
    

        int_t n_rows=ap[i+1]-ap[i];        
    
        for(int_t k=0;k<n_rows;k++)
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
                    L_r_idx=(int_32t*)pangulu_realloc(__FILE__, __LINE__, L_r_idx,(L_maxsize+relloc_zjq)*sizeof(int_32t));
                    L_maxsize=L_maxsize+nnz;
                }
            }
            else
            {
                U_r_idx[U_size]=row;
                U_size++;
                if(U_size >= U_maxsize-100)
                {
                    U_r_idx=(int_32t*)pangulu_realloc(__FILE__, __LINE__, U_r_idx,(U_maxsize+relloc_zjq)*sizeof(int_32t));
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
                                    L_r_idx=(int_32t*)pangulu_realloc(__FILE__, __LINE__, L_r_idx,(L_maxsize+relloc_zjq)*sizeof(int_32t));
                                    L_maxsize=L_maxsize+nnz;
                                } 
                            }
                            else
                            {
                                U_r_idx[U_size]=kchild;
                                U_size++;
                                if(U_size >= U_maxsize-100)
                                {
                                    U_r_idx=(int_32t*)pangulu_realloc(__FILE__, __LINE__, U_r_idx,(U_maxsize+relloc_zjq)*sizeof(int_32t));
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
                      pangulu_block_Smatrix *block_Smatrix,
                      pangulu_origin_Smatrix *reorder_matrix){
#ifndef symmetric    
    fill_in_2_no_sort_pruneL(reorder_matrix->row,reorder_matrix->nnz,reorder_matrix->columnindex,reorder_matrix->rowpointer,L_rowpointer,L_columnindex,U_rowpointer,U_columnindex);
#else
    int_t *new_rowpointer=NULL;
    int_32t *new_columnindex=NULL;
    int_t new_nnz;
    at_plus_a_dist(reorder_matrix->row,reorder_matrix->nnz,
                   reorder_matrix->rowpointer,reorder_matrix->columnindex,
                   &new_nnz,&new_rowpointer,&new_columnindex);
    int_t *symbolic_rowpointer = NULL;
    int_32t *symbolic_columnindex = NULL;
    int_t N = block_common->N;
    int_32t NB = block_common->NB;
    int_32t block_length = block_common->block_length;
    int_t *block_Smatrix_nnzA_num = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * block_length * block_length);
    int_t *block_Smatrix_non_zero_vector_L = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * block_length);
    int_t *block_Smatrix_non_zero_vector_U = (int_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(int_t) * block_length);
    
    for (int_t i = 0; i < block_length; i++)
    {
        block_Smatrix_non_zero_vector_L[i] = 0;
    }

    for (int_t i = 0; i < block_length; i++)
    {
        block_Smatrix_non_zero_vector_U[i] = 0;
    }
    for (int_t i = 0; i < block_length * block_length; i++)
    {
        block_Smatrix_nnzA_num[i] = 0;
    }

    fill_in_sym_prune(reorder_matrix->row,new_nnz,new_columnindex,new_rowpointer,
                    &symbolic_rowpointer,&symbolic_columnindex,
                    N,NB,block_length,
                    block_Smatrix_non_zero_vector_L,
                    block_Smatrix_non_zero_vector_U,
                    block_Smatrix_nnzA_num,
                    &block_Smatrix->symbolic_nnz);
    
    pangulu_free(__FILE__, __LINE__, new_rowpointer);
    pangulu_free(__FILE__, __LINE__, new_columnindex);
    
    block_Smatrix->block_Smatrix_nnzA_num=block_Smatrix_nnzA_num;
    block_Smatrix->block_Smatrix_non_zero_vector_L=block_Smatrix_non_zero_vector_L;
    block_Smatrix->block_Smatrix_non_zero_vector_U=block_Smatrix_non_zero_vector_U;
    block_Smatrix->symbolic_rowpointer=symbolic_rowpointer;
    block_Smatrix->symbolic_columnindex=symbolic_columnindex;

#endif  

}

#endif

