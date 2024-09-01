#include "pangulu_common.h"

void  pangulu_sptrsv_cpu_choumi(pangulu_smatrix *s,pangulu_vector *x,pangulu_vector *b)
{
    calculate_type *value=s->value;
    calculate_type *bval=b->value;
    calculate_type *xval=x->value;
    pangulu_int64_t n=s->column;
    for(pangulu_int64_t i=0;i<n;i++){
        for(pangulu_int64_t j=0;j<n;j++) {
            if(i==j){
                 xval[i]=bval[i]/value[i*n+j];
                 for(pangulu_int64_t k=i+1;k<n;k++){
                     bval[k]-=xval[i]*value[k*n+j];
                 }
             }
        }
    }
}

void pangulu_sptrsv_cpu_xishu(pangulu_smatrix *s,pangulu_vector *x,pangulu_vector *b,pangulu_int64_t vector_number)
{
    pangulu_int64_t row=s->row;  
    pangulu_inblock_ptr *csr_row_ptr_tmp=s->rowpointer;
    pangulu_inblock_idx *csr_col_idx_tmp=s->columnindex;
    calculate_type *csr_val_tmp=s->value;
    for(pangulu_int64_t vector_index=0;vector_index<vector_number;vector_index++){
        calculate_type *xval=x->value+vector_index*row;
        calculate_type *bval=b->value+vector_index*row;
	    for(pangulu_int64_t i=0;i<row;i++) 
	    {
		    calculate_type sumf=0;
            pangulu_int64_t have=0;
		    for(pangulu_int64_t j=csr_row_ptr_tmp[i];j<csr_row_ptr_tmp[i+1];j++)
		    {
                if(i!=csr_col_idx_tmp[j])   sumf+=csr_val_tmp[j]*xval[csr_col_idx_tmp[j]];
                else    have =1;
	    	}
            if(have==0){	
                xval[i]=0.0;
            }
            else 
		    {
                xval[i]=(bval[i]-sumf)/csr_val_tmp[ csr_row_ptr_tmp[i+1]-1];
            }
	    }
    }
}
void pangulu_sptrsv_cpu_xishu_csc(pangulu_smatrix *s,pangulu_vector *x,pangulu_vector *b,pangulu_int64_t vector_number,pangulu_int64_t tag)
{
    pangulu_int64_t col=s->column;  
    pangulu_inblock_ptr *csc_column_ptr_tmp=s->columnpointer;
    pangulu_inblock_idx *csc_row_idx_tmp=s->rowindex;
    calculate_type *cscVal_tmp = s->value_csc;
    if(tag==0){
        for(pangulu_int64_t vector_index=0;vector_index<vector_number;vector_index++){
            calculate_type *xval=x->value+vector_index*col;
            calculate_type *bval=b->value+vector_index*col;
	        for(pangulu_int64_t i=0;i<col;i++) 
    	    {
                if(csc_row_idx_tmp[csc_column_ptr_tmp[i]]==i){
                    if(fabs(cscVal_tmp[csc_column_ptr_tmp[i]])>SPTRSV_ERROR)
                    xval[i]=bval[i]/cscVal_tmp[csc_column_ptr_tmp[i]];
                    else
                    xval[i]=bval[i]/SPTRSV_ERROR;
                }
                else{
                    xval[i]=0.0;
                    continue;
                }
		        for(pangulu_int64_t j=csc_column_ptr_tmp[i]+1;j<csc_column_ptr_tmp[i+1];j++)
    		    {
                    pangulu_inblock_idx row=csc_row_idx_tmp[j];
                    bval[row]-=cscVal_tmp[j]*xval[i];
	    	    }
	        }
        }
    }
    else{
        for(pangulu_int64_t vector_index=0;vector_index<vector_number;vector_index++){
            calculate_type *xval=x->value+vector_index*col;
            calculate_type *bval=b->value+vector_index*col;
	        for(pangulu_int64_t i=col-1;i>=0;i--) 
    	    {
                if(csc_row_idx_tmp[csc_column_ptr_tmp[i+1]-1]==i){
                    if(fabs(cscVal_tmp[csc_column_ptr_tmp[i+1]-1])>SPTRSV_ERROR)
                    xval[i]=bval[i]/cscVal_tmp[csc_column_ptr_tmp[i+1]-1];
                    else
                    xval[i]=bval[i]/SPTRSV_ERROR;
                }
                else{
                    xval[i]=0.0;
                    continue;
                }
                if(csc_column_ptr_tmp[i+1]>=2){ // Don't modify this to csc_column_ptr_tmp[i+1]-2>=0, because values in array csc_column_ptr_tmp are unsigned.
                    for(pangulu_int64_t j=csc_column_ptr_tmp[i+1]-2;j>=csc_column_ptr_tmp[i];j--)
                    {
                        pangulu_inblock_idx row=csc_row_idx_tmp[j];
                        bval[row]-=cscVal_tmp[j]*xval[i];
                    }
                }
	        }
        }
    }
    
}
