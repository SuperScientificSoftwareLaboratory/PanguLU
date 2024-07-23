#include<stdio.h>
#include<stdlib.h>

typedef int64_t int_t;
typedef int32_t int_32t;
typedef int idx_int;
typedef unsigned long long pangulu_exblock_ptr;
typedef unsigned int pangulu_exblock_idx;
typedef unsigned int pangulu_inblock_ptr; // 块内rowptr和colptr类型
typedef unsigned int pangulu_inblock_idx; // 块内colidx和rowidx类型
typedef long double pangulu_refinement_hp;

struct pangulu_gstrf_options
{
    double tol;
};
struct pangulu_init_options
{
    int nthread;
    int nb;
};
struct pangulu_gstrs_options
{
    int nrhs;

};