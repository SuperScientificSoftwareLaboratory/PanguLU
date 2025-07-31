#ifndef DEFINE_H
#define DEFINE_H

#include <stdint.h>
#include "struct.h"
#include "mkqsort.h"

#define IDX_MAX   INT64_MAX
#define IDX_MIN   INT64_MIN

#define SCIDX  SCNd64
#define PRIDX  PRId64

#define NUM_OPTIONS     40
#define PTYPE_RB        0
#define PTYPE_KWAY      1

#define OBJTYPE_CUT     0
#define OBJTYPE_VOL     1
#define OBJTYPE_NODE    2

#define CTYPE_RM        0
#define CTYPE_SHEM      1

#define OP_PMETIS       0
#define OP_KMETIS       1
#define OP_OMETIS       2

#define CTYPE_RM        0
#define CTYPE_SHEM      1

#define IPTYPE_GROW     0
#define IPTYPE_RANDOM   1
#define IPTYPE_EDGE     2
#define IPTYPE_NODE     3
#define IPTYPE_METISRB  4


#define RTYPE_FM        0
#define RTYPE_GREEDY    1
#define RTYPE_SEP2SIDED 2
#define RTYPE_SEP1SIDED 3

#define PMETIS_DEFAULT_UFACTOR          1
#define MCPMETIS_DEFAULT_UFACTOR        10
#define PartGraphKway_DEFAULT_UFACTOR   30
#define OMETIS_DEFAULT_UFACTOR          200

#define OPTION_PTYPE        0
#define OPTION_OBJTYPE      1
#define OPTION_CTYPE        2
#define OPTION_IPTYPE       3
#define OPTION_RTYPE        4
#define OPTION_DBGLVL       5
#define OPTION_NITER        6
#define OPTION_NCUTS        7
#define OPTION_SEED         8
#define OPTION_NO2HOP       9
#define OPTION_MINCONN      10
#define OPTION_CONTIG       11
#define OPTION_COMPRESS     12
#define OPTION_CCORDER      13
#define OPTION_PFACTOR      14
#define OPTION_NSEPS        15
#define OPTION_UFACTOR      16
#define OPTION_NUMBERING    17

#define OPTION_HELP         18
#define OPTION_TPWGTS       19
#define OPTION_NCOMMON      20
#define OPTION_NOOUTPUT     21
#define OPTION_BALANCE      22
#define OPTION_GTYPE        23
#define OPTION_UBVEC        24

#define GETOPTION(options, idx, defval) ((options) == NULL || (options)[idx] == -1 ? defval : (options)[idx]) 

#define COMPUTE_UBFACTORS(ufactor) (1.0 + 0.001 * (ufactor))

#define lyj_max(a, b) ((a) >= (b) ? (a) : (b))
#define lyj_min(a, b) ((a) <= (b) ? (a) : (b))
#define lyj_swap(a, b, z) (z = a, a = b, b = z)
#define m_gt_n(m,n) ((m)>(n))

#define MAKECSR(i, n, a) \
   do { \
     for (i=1; i<n; i++) a[i] += a[i-1]; \
     for (i=n; i>0; i--) a[i] = a[i-1]; \
     a[0] = 0; \
   } while(0) 

#define SHIFTCSR(i, n, a) \
   do { \
     for (i=n; i>0; i--) a[i] = a[i-1]; \
     a[0] = 0; \
   } while(0) 

void ikvsorti(size_t n, ikv_t *base)
{
#define ikey_lt(a, b) ((a)->key < (b)->key)
    GK_MKQSORT(ikv_t, base, n, ikey_lt);
#undef ikey_lt
}

#define CONTROL_COMMAND(a, flag, cmd) if ((a)&(flag)) (cmd);

#endif