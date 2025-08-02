#ifndef FUNCTIONSET_H
#define FUNCTIONSET_H

#include <unistd.h>
#define __USE_GNU
#include <sched.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h> 
#include <malloc.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>
#include "../include/mynd_types.h"

/* typedef.h */
#ifndef MYND_OMP_H
#define MYND_OMP_H

/*--------------------------------------------------------------------------
 Specifies the width of the elementary data type that will hold information
 about vertices and their adjacency lists.

 Possible values:
   32 : Use 32 bit signed integers
   64 : Use 64 bit signed integers

 A width of 64 should be specified if the number of vertices or the total
 number of edges in the graph exceed the limits of a 32 bit signed integer
 i.e., 2^31-1.
 Proper use of 64 bit integers requires that the c99 standard datatypes
 int32_t and int64_t are supported by the compiler.
 GCC does provides these definitions in stdint.h, but it may require some
 modifications on other architectures.
--------------------------------------------------------------------------*/
// #define IDXTYPEWIDTH 64


/*--------------------------------------------------------------------------
 Specifies the data type that will hold floating-point style information.

 Possible values:
   32 : single precission floating point (float)
   64 : double precission floating point (double)
--------------------------------------------------------------------------*/
// #define REALTYPEWIDTH 64

/****************************************************************************
* In principle, nothing needs to be changed beyond this point, unless the
* int32_t and int64_t cannot be found in the normal places.
*****************************************************************************/

/* Uniform definitions for various compilers */
#if defined(_MSC_VER)
  #define COMPILER_MSC
#endif
#if defined(__ICC)
  #define COMPILER_ICC
#endif
#if defined(__GNUC__)
  #define COMPILER_GCC
#endif

/* Include c99 int definitions and need constants. When building the library,
 * these are already defined by GKlib; hence the test for _GKLIB_H_ */
#ifndef _GKLIB_H_
#ifdef COMPILER_MSC
#include <limits.h>

typedef __int32 int32_t;
typedef __int64 int64_t;
#define PRId32       "I32d"
#define PRId64       "I64d"
#define SCNd32       "ld"
#define SCNd64       "I64d"
#define INT32_MIN    ((int32_t)_I32_MIN)
#define INT32_MAX    _I32_MAX
#define INT64_MIN    ((int64_t)_I64_MIN)
#define INT64_MAX    _I64_MAX
#else
#include <inttypes.h>
#endif
#endif


/*------------------------------------------------------------------------
* Setup the basic datatypes
*-------------------------------------------------------------------------*/
#if IDXTYPEWIDTH == 32
  typedef int32_t reordering_int_t;

  #define IDX_MAX   INT32_MAX
  #define IDX_MIN   INT32_MIN

  #define SCIDX  SCNd32
  #define PRIDX  PRId32

  #define strtoidx      strtol
  #define lyj_abs          abs
#elif IDXTYPEWIDTH == 64
  typedef int64_t reordering_int_t;

  #define IDX_MAX   INT64_MAX
  #define IDX_MIN   INT64_MIN

  #define SCIDX  SCNd64
  #define PRIDX  PRId64

#ifdef COMPILER_MSC
  #define strtoidx      _strtoi64
#else
  #define strtoidx      strtoll
#endif
  #define lyj_abs          labs
#else
  #error "Incorrect user-supplied value fo IDXTYPEWIDTH"
#endif


#if REALTYPEWIDTH == 32
  typedef float reordering_real_t;

  #define SCREAL         "f"
  #define PRREAL         "f"
  #define REAL_MAX       FLT_MAX
  #define REAL_MIN       FLT_MIN
  #define REAL_EPSILON   FLT_EPSILON

  #define rabs          fabsf
  #define REALEQ(x,y) ((rabs((x)-(y)) <= FLT_EPSILON))

#ifdef COMPILER_MSC
  #define strtoreal     (float)strtod
#else
  #define strtoreal     strtof
#endif
#elif REALTYPEWIDTH == 64
  typedef double reordering_real_t;

  #define SCREAL         "lf"
  #define PRREAL         "lf"
  #define REAL_MAX       DBL_MAX
  #define REAL_MIN       DBL_MIN
  #define REAL_EPSILON   DBL_EPSILON

  #define rabs          fabs
  #define REALEQ(x,y) ((rabs((x)-(y)) <= DBL_EPSILON))

  #define strtoreal     strtod
#else
  #error "Incorrect user-supplied value for REALTYPEWIDTH"
#endif

#endif

/* control.h */
#ifndef CONTROL_H
#define CONTROL_H

extern reordering_int_t control;

#define ALL_Time                    7
#define PRINTTIMEGENERAL            7

#define NESTEDBISECTION_Time        3
#define BISECTIONBEST_Time          3
#define COARSEN_Time                3
#define REORDERBISECTION_Time       3
#define REFINE2WAYNODE_Time         3
#define SPLITGRAPHREORDER_Time      3
#define PRINTTIMEPHASES             3

#define MATCH_Time                  1
#define CREATCOARSENGRAPH_Time      1
#define PARTITIOBINF2WAY            1 
#define FM2WAYCUTBALANCE_Time       1
#define FM2WAYCUTREFINE_Time        1
#define REORDERINF2WAY_Time         1
#define FMNODEBALANCE_Time          1
#define FM1SIDENODEREFINE_Time      1
#define FM2SIDENODEREFINE_Time      1
#define PRINTTIMESTEPS              1

#endif

/* mkqsort.h */
#ifndef MKQSORT_H
#define MKQSORT_H

/* Swap two items pointed to by A and B using temporary buffer t. */
#define _GKQSORT_SWAP(a, b, t) ((void)((t = *a), (*a = *b), (*b = t)))

/* Discontinue quicksort algorithm when partition gets below this size.
   This particular magic number was chosen to work best on a Sun 4/260. */
#define _GKQSORT_MAX_THRESH 4

/* The next 4 #defines implement a very fast in-line stack abstraction. */
#define _GKQSORT_STACK_SIZE	    (8 * sizeof(size_t))
#define _GKQSORT_PUSH(top, low, high) (((top->_lo = (low)), (top->_hi = (high)), ++top))
#define	_GKQSORT_POP(low, high, top)  ((--top, (low = top->_lo), (high = top->_hi)))
#define	_GKQSORT_STACK_NOT_EMPTY	    (_stack < _top)


/* The main code starts here... */
#define GK_MKQSORT(GKQSORT_TYPE,GKQSORT_BASE,GKQSORT_NELT,GKQSORT_LT)   \
{									\
  GKQSORT_TYPE *const _base = (GKQSORT_BASE);				\
  const size_t _elems = (GKQSORT_NELT);					\
  GKQSORT_TYPE _hold;							\
									\
  if (_elems == 0)                                                      \
    return;                                                             \
                                                                        \
  /* Don't declare two variables of type GKQSORT_TYPE in a single	\
   * statement: eg `TYPE a, b;', in case if TYPE is a pointer,		\
   * expands to `type* a, b;' wich isn't what we want.			\
   */									\
									\
  if (_elems > _GKQSORT_MAX_THRESH) {					\
    GKQSORT_TYPE *_lo = _base;						\
    GKQSORT_TYPE *_hi = _lo + _elems - 1;				\
    struct {								\
      GKQSORT_TYPE *_hi; GKQSORT_TYPE *_lo;				\
    } _stack[_GKQSORT_STACK_SIZE], *_top = _stack + 1;			\
									\
    while (_GKQSORT_STACK_NOT_EMPTY) {					\
      GKQSORT_TYPE *_left_ptr; GKQSORT_TYPE *_right_ptr;		\
									\
      /* Select median value from among LO, MID, and HI. Rearrange	\
         LO and HI so the three values are sorted. This lowers the	\
         probability of picking a pathological pivot value and		\
         skips a comparison for both the LEFT_PTR and RIGHT_PTR in	\
         the while loops. */						\
									\
      GKQSORT_TYPE *_mid = _lo + ((_hi - _lo) >> 1);			\
									\
      if (GKQSORT_LT (_mid, _lo))					\
        _GKQSORT_SWAP (_mid, _lo, _hold);				\
      if (GKQSORT_LT (_hi, _mid))					\
        _GKQSORT_SWAP (_mid, _hi, _hold);				\
      else								\
        goto _jump_over;						\
      if (GKQSORT_LT (_mid, _lo))					\
        _GKQSORT_SWAP (_mid, _lo, _hold);				\
  _jump_over:;								\
									\
      _left_ptr  = _lo + 1;						\
      _right_ptr = _hi - 1;						\
									\
      /* Here's the famous ``collapse the walls'' section of quicksort.	\
         Gotta like those tight inner loops!  They are the main reason	\
         that this algorithm runs much faster than others. */		\
      do {								\
        while (GKQSORT_LT (_left_ptr, _mid))				\
         ++_left_ptr;							\
									\
        while (GKQSORT_LT (_mid, _right_ptr))				\
          --_right_ptr;							\
									\
        if (_left_ptr < _right_ptr) {					\
          _GKQSORT_SWAP (_left_ptr, _right_ptr, _hold);			\
          if (_mid == _left_ptr)					\
            _mid = _right_ptr;						\
          else if (_mid == _right_ptr)					\
            _mid = _left_ptr;						\
          ++_left_ptr;							\
          --_right_ptr;							\
        }								\
        else if (_left_ptr == _right_ptr) {				\
          ++_left_ptr;							\
          --_right_ptr;							\
          break;							\
        }								\
      } while (_left_ptr <= _right_ptr);				\
									\
     /* Set up pointers for next iteration.  First determine whether	\
        left and right partitions are below the threshold size.  If so,	\
        ignore one or both.  Otherwise, push the larger partition's	\
        bounds on the stack and continue sorting the smaller one. */	\
									\
      if (_right_ptr - _lo <= _GKQSORT_MAX_THRESH) {			\
        if (_hi - _left_ptr <= _GKQSORT_MAX_THRESH)			\
          /* Ignore both small partitions. */				\
          _GKQSORT_POP (_lo, _hi, _top);				\
        else								\
          /* Ignore small left partition. */				\
          _lo = _left_ptr;						\
      }									\
      else if (_hi - _left_ptr <= _GKQSORT_MAX_THRESH)			\
        /* Ignore small right partition. */				\
        _hi = _right_ptr;						\
      else if (_right_ptr - _lo > _hi - _left_ptr) {			\
        /* Push larger left partition indices. */			\
        _GKQSORT_PUSH (_top, _lo, _right_ptr);				\
        _lo = _left_ptr;						\
      }									\
      else {								\
        /* Push larger right partition indices. */			\
        _GKQSORT_PUSH (_top, _left_ptr, _hi);				\
        _hi = _right_ptr;						\
      }									\
    }									\
  }									\
									\
  /* Once the BASE array is partially sorted by quicksort the rest	\
     is completely sorted using insertion sort, since this is efficient	\
     for partitions below MAX_THRESH size. BASE points to the		\
     beginning of the array to sort, and END_PTR points at the very	\
     last element in the array (*not* one beyond it!). */		\
									\
  {									\
    GKQSORT_TYPE *const _end_ptr = _base + _elems - 1;			\
    GKQSORT_TYPE *_tmp_ptr = _base;					\
    register GKQSORT_TYPE *_run_ptr;					\
    GKQSORT_TYPE *_thresh;						\
									\
    _thresh = _base + _GKQSORT_MAX_THRESH;				\
    if (_thresh > _end_ptr)						\
      _thresh = _end_ptr;						\
									\
    /* Find smallest element in first threshold and place it at the	\
       array's beginning.  This is the smallest array element,		\
       and the operation speeds up insertion sort's inner loop. */	\
									\
    for (_run_ptr = _tmp_ptr + 1; _run_ptr <= _thresh; ++_run_ptr)	\
      if (GKQSORT_LT (_run_ptr, _tmp_ptr))				\
        _tmp_ptr = _run_ptr;						\
									\
    if (_tmp_ptr != _base)						\
      _GKQSORT_SWAP (_tmp_ptr, _base, _hold);				\
									\
    /* Insertion sort, running from left-hand-side			\
     * up to right-hand-side.  */					\
									\
    _run_ptr = _base + 1;						\
    while (++_run_ptr <= _end_ptr) {					\
      _tmp_ptr = _run_ptr - 1;						\
      while (GKQSORT_LT (_run_ptr, _tmp_ptr))				\
        --_tmp_ptr;							\
									\
      ++_tmp_ptr;							\
      if (_tmp_ptr != _run_ptr) {					\
        GKQSORT_TYPE *_trav = _run_ptr + 1;				\
        while (--_trav >= _run_ptr) {					\
          GKQSORT_TYPE *_hi; GKQSORT_TYPE *_lo;				\
          _hold = *_trav;						\
									\
          for (_hi = _lo = _trav; --_lo >= _tmp_ptr; _hi = _lo)		\
            *_hi = *_lo;						\
          *_hi = _hold;							\
        }								\
      }									\
    }									\
  }									\
									\
}

#endif

/* struct.h */
#ifndef STRUCT_H
#define STRUCT_H

/*************************************************************************/
/*! This data structure stores the various command line arguments */
/*************************************************************************/
typedef struct 
{
	reordering_int_t ptype;
	reordering_int_t objtype;
	reordering_int_t ctype;
	reordering_int_t iptype;
	reordering_int_t rtype;

	reordering_int_t no2hop;
	reordering_int_t minconn;
	reordering_int_t contig;

	reordering_int_t nooutput;

	reordering_int_t balance;
	reordering_int_t ncuts;
	reordering_int_t niter;

	reordering_int_t gtype;
	reordering_int_t ncommon;

	reordering_int_t seed;
	reordering_int_t dbglvl;

	reordering_int_t nparts;

	reordering_int_t nseps;
	reordering_int_t ufactor;
	reordering_int_t pfactor;
	reordering_int_t compress;
	reordering_int_t ccorder;

	char *filename;
	char *outfile;
	char *xyzfile;
	char *tpwgtsfile;
	char *ubvecstr;

	reordering_int_t wgtflag;
	reordering_int_t numflag;
	reordering_real_t *tpwgts;
	reordering_real_t *ubvec;

	reordering_real_t iotimer;
	reordering_real_t parttimer;
	reordering_real_t reporttimer;

	reordering_int_t maxmemory;
} params_t;

/*************************************************************************/
/*! The following data structure stores holds information on degrees for k-way
    partition */
/*************************************************************************/
typedef struct ckrinfo_t 
{
	reordering_int_t id;              /*!< The internal degree of a vertex (sum of weights) */
	reordering_int_t ed;            	/*!< The total external degree of a vertex */
	reordering_int_t nnbrs;          	/*!< The number of neighboring subdomains */
	reordering_int_t inbr;            /*!< The index in the cnbr_t array where the nnbrs list 
                             of neighbors is stored */
} ckrinfo_t;

/*************************************************************************/
/*! The following data structure holds information on degrees for k-way
    vol-based partition */
/*************************************************************************/
typedef struct vkrinfo_t 
{
	reordering_int_t nid;             /*!< The internal degree of a vertex (count of edges) */
	reordering_int_t ned;            	/*!< The total external degree of a vertex (count of edges) */
	reordering_int_t gv;            	/*!< The volume gain of moving that vertex */
	reordering_int_t nnbrs;          	/*!< The number of neighboring subdomains */
	reordering_int_t inbr;            /*!< The index in the vnbr_t array where the nnbrs list 
                             of neighbors is stored */
} vkrinfo_t;

/*************************************************************************/
/*! The following data structure holds information on degrees for k-way
    partition */
/*************************************************************************/
typedef struct nrinfo_t 
{
	reordering_int_t edegrees[2];  
} nrinfo_t;

/*************************************************************************/
/*! Binary Search Tree node */
/*************************************************************************/
typedef struct treenode_t
{ 
	reordering_int_t val;  //  ncy
	reordering_int_t key;  //  wgt
	struct treenode_t *left;
	struct treenode_t *right;
} treenode_t;

/*************************************************************************/
/*! Binary Search Tree */
/*************************************************************************/
typedef struct binary_search_tree_t
{ 
	reordering_int_t nownodes; 
	// size_t maxnodes; 
	treenode_t *treenode; 
	
} binary_search_tree_t;

/*************************************************************************/
/*! Hash Table Element */
/*************************************************************************/
typedef struct hashelement_t
{ 
	reordering_int_t val;  //  ncy
	reordering_int_t key;  //  wgt
} hashelement_t;

/*************************************************************************/
/*! Hash Table */
/*************************************************************************/
typedef struct hash_table_t
{ 
	reordering_int_t nownodes; 
	reordering_int_t maxnodes; 
	hashelement_t *hashelement; 
} hash_table_t;

/*************************************************************************/
/*! Hash Table */
/*************************************************************************/
typedef struct hash_table2_t
{ 
	reordering_int_t nownodes; 
	reordering_int_t maxnodes; 
	reordering_int_t *hashelement; 
} hash_table2_t;

/*************************************************************************/
/*! Hash Table */
/*************************************************************************/
typedef struct hash_table_omp_t
{ 
	// reordering_int_t *nownodes; 
	reordering_int_t *maxnodes; 
	reordering_int_t *index;
	reordering_int_t *val;
	reordering_int_t *key;
} hash_table_omp_t;

/*************************************************************************/
/*! Hash Table */
/*************************************************************************/
typedef struct hash_table_omp2_t
{
	reordering_int_t maxnodes;
	reordering_int_t *val;
	reordering_int_t *key;
} hash_table_omp2_t;

/*************************************************************************/
/*! Binary Search Tree node */
/*************************************************************************/
typedef struct treenode2_t
{ 
	reordering_int_t val;  //  ncy
	reordering_int_t key;  //  wgt
} treenode2_t;

/*************************************************************************/
/*! Binary Search Tree */
/*************************************************************************/
typedef struct binary_search_tree2_t
{ 
	reordering_int_t nownodes; 
	reordering_int_t maxnodes; 
	treenode2_t *treenode; 
} binary_search_tree2_t;

/*************************************************************************/
/*! This data structure holds a graph */
/*************************************************************************/
typedef struct graph_t 
{
    reordering_int_t nvtxs, nedges;	/* The # of vertices and edges in the graph */
    // reordering_int_t ncon;		/* The # of constrains */ 
    reordering_int_t *xadj;		/* Pointers to the locally stored vertices */
    reordering_int_t *vwgt;		/* Vertex weights */
    // reordering_int_t *vsize;		/* Vertex sizes for min-volume formulation */
    reordering_int_t *adjncy;        /* Array that stores the adjacency lists of nvtxs */
    reordering_int_t *adjwgt;        /* Array that stores the weights of the adjacency lists */

    reordering_int_t *tvwgt;         /* The sum of the vertex weights in the graph */
    reordering_real_t *invtvwgt;     /* The inverse of the sum of the vertex weights in the graph */


    /* These are to keep track control if the corresponding fields correspond to
        application or library memory */
    // reordering_int_t free_xadj, free_vwgt, free_vsize, free_adjncy, free_adjwgt;

    reordering_int_t *label;
    reordering_int_t *cmap;
    reordering_int_t *match;
    void **addr;
    binary_search_tree2_t *tree;
    hash_table2_t *hash;

    /* Partition parameters */
    reordering_int_t mincut, minvol;
    reordering_int_t *where, *pwgts;
    reordering_int_t nbnd;
    reordering_int_t *bndptr, *bndind;

    /* Bisection refinement parameters */
    reordering_int_t *id, *ed;

    /* K-way refinement parameters */
    // ckrinfo_t *ckrinfo;   /*!< The per-vertex cut-based refinement info */
    // vkrinfo_t *vkrinfo;   /*!< The per-vertex volume-based refinement info */

    /* Node refinement information */
    nrinfo_t *nrinfo;	//	replace after

    struct graph_t *coarser, *finer;
} graph_t;

typedef struct ikv_t
{
  reordering_int_t key;
  reordering_int_t val;
} ikv_t;

/*************************************************************************/
/*! The following structure stores information used by Metis */
/*************************************************************************/
typedef struct ctrl_t 
{
	reordering_int_t optype;
	reordering_int_t objtype;
	reordering_int_t ctype;
	reordering_int_t iptype;
	reordering_int_t rtype;
	// moptype_et  optype;	        /* Type of operation */
	// mobjtype_et objtype;          /* Type of refinement objective */
	// mdbglvl_et  dbglvl;		/* Controls the debuging output of the program */
	// mctype_et   ctype;		/* The type of coarsening */
	// miptype_et  iptype;		/* The type of initial partitioning */
	// mrtype_et   rtype;		/* The type of refinement */

	reordering_int_t CoarsenTo;		/* The # of vertices in the coarsest graph */
	reordering_int_t nIparts;                /* The number of initial partitions to compute */
	reordering_int_t no2hop;                 /* Indicates if 2-hop matching will be used */
	reordering_int_t minconn;                /* Indicates if the subdomain connectivity will be minimized */
	reordering_int_t contig;                 /* Indicates if contigous partitions are required */
	reordering_int_t nseps;			/* The number of separators to be found during multiple bisections */
	reordering_int_t ufactor;                /* The user-supplied load imbalance factor */
	reordering_int_t compress;               /* If the graph will be compressed prior to ordering */
	reordering_int_t ccorder;                /* If connected components will be ordered separately */
	reordering_int_t seed;                   /* The seed for the random number generator */
	reordering_int_t ncuts;                  /* The number of different partitionings to compute */
	reordering_int_t niter;                  /* The number of iterations during each refinement */
	reordering_int_t numflag;                /* The user-supplied numflag for the graph */
	reordering_int_t *maxvwgt;		/* The maximum allowed weight for a vertex */

	reordering_int_t ncon;                   /*!< The number of balancing constraints */
	reordering_int_t nparts;                 /*!< The number of partitions */

	reordering_real_t pfactor;		/* .1*(user-supplied prunning factor) */

	reordering_real_t *ubfactors;            /*!< The per-constraint ubfactors */
	
	reordering_real_t *tpwgts;               /*!< The target partition weights */
	reordering_real_t *pijbm;                /*!< The nparts*ncon multiplies for the ith partition
										and jth constraint for obtaining the balance */

	reordering_real_t cfactor;               /*!< The achieved compression factor */

  	/* These are for use by the k-way refinement routines */
//   size_t nbrpoolsize;      /*!< The number of {c,v}nbr_t entries that have been allocated */
//   size_t nbrpoolcpos;      /*!< The position of the first free entry in the array */
//   size_t nbrpoolreallocs;  /*!< The number of times the pool was resized */

//   cnbr_t *cnbrpool;     /*!< The pool of cnbr_t entries to be used during refinement.
//                              The size and current position of the pool is controlled
//                              by nnbrs & cnbrs */
//   vnbr_t *vnbrpool;     /*!< The pool of vnbr_t entries to be used during refinement.
//                              The size and current position of the pool is controlled
//                              by nnbrs & cnbrs */

  	/* The subdomain graph, in sparse format  */ 
	reordering_int_t *maxnads;               /* The maximum allocated number of adjacent domains */
	reordering_int_t *nads;                  /* The number of adjacent domains */
	reordering_int_t **adids;                /* The IDs of the adjacent domains */
	reordering_int_t **adwgts;               /* The edge-weight to the adjacent domains */
	reordering_int_t *pvec1, *pvec2;         /* Auxiliar nparts-size vectors for efficiency */

} ctrl_t;

/*************************************************************************/
/*! binary heap node */
/*************************************************************************/
typedef struct node_t
{ 
	reordering_int_t key;  //  ed - id
	reordering_int_t val;  //  vertex
} node_t;

/*************************************************************************/
/*! Priority queue based on binary heap */
/*************************************************************************/
typedef struct priority_queue_t
{ 
	reordering_int_t nownodes; 
	reordering_int_t maxnodes; 
	node_t *heap; 
	reordering_int_t *locator;
} priority_queue_t;

/*************************************************************************/
/*! The following data structure stores information about a memory 
    allocation operation that can either be served from gk_mcore_t or by
    a gk_malloc if not sufficient workspace memory is available. */
/*************************************************************************/
typedef struct memory_block {
	// reordering_int_t type;
	reordering_int_t nbytes;
	void *ptr;
} memory_block;

typedef struct memory_manage {
	reordering_int_t used_block;
	reordering_int_t all_block;
	reordering_int_t now_memory;
	reordering_int_t max_memory;
	memory_block *memoryblock;
} memory_manage;

#endif

/* define.h*/
#ifndef DEFINE_H
#define DEFINE_H

//  options
/* The maximum length of the options[] array */
#define NUM_OPTIONS     40
// #define OPTION_OBJTYPE  1
// #define OPTION_CTYPE    2
// #define OPTION_IPTYPE   3
// #define OPTION_RTYPE    4
// #define OPTION_NO2HOP   9
// #define OPTION_MINCONN  10
// #define OPTION_CONTIG   11
// #define OPTION_SEED     8
// #define OPTION_NITER    6
// #define OPTION_NCUTS    7
// #define OPTION_UFACTOR  16
// #define OPTION_DBGLVL   5

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

/*! Coarsening Schemes */
#define CTYPE_RM        0
#define CTYPE_SHEM      1

/*! Initial partitioning schemes */
#define IPTYPE_GROW     0
#define IPTYPE_RANDOM   1
#define IPTYPE_EDGE     2
#define IPTYPE_NODE     3
#define IPTYPE_METISRB  4


/*! Refinement schemes */
#define RTYPE_FM        0
#define RTYPE_GREEDY    1
#define RTYPE_SEP2SIDED 2
#define RTYPE_SEP1SIDED 3

/* Default ufactors for the various operational modes */
#define PMETIS_DEFAULT_UFACTOR          1
#define MCPMETIS_DEFAULT_UFACTOR        10
#define PartGraphKway_DEFAULT_UFACTOR   30
#define OMETIS_DEFAULT_UFACTOR          200

/*! Options codes (i.e., options[]) */
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

/* Used for command-line parameter purposes */
#define OPTION_HELP         18
#define OPTION_TPWGTS       19
#define OPTION_NCOMMON      20
#define OPTION_NOOUTPUT     21
#define OPTION_BALANCE      22
#define OPTION_GTYPE        23
#define OPTION_UBVEC        24

/* gets the appropriate option value */
#define GETOPTION(options, idx, defval) ((options) == NULL || (options)[idx] == -1 ? defval : (options)[idx]) 

//  compute ubfactors
#define COMPUTE_UBFACTORS(ufactor) (1.0 + 0.001 * (ufactor))

#define lyj_max(a, b) ((a) >= (b) ? (a) : (b))
#define lyj_min(a, b) ((a) <= (b) ? (a) : (b))
#define lyj_swap(a, b, z) (z = a, a = b, b = z)
#define m_gt_n(m,n) ((m)>(n))

/*-------------------------------------------------------------
 * CSR conversion macros
 *-------------------------------------------------------------*/
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

#define CONTROL_COMMAND(a, flag, cmd) if ((a)&(flag)) (cmd);

#endif

/* timer.h */
extern reordering_real_t time_all;
extern struct timeval start_all;
extern struct timeval end_all;

extern reordering_real_t time_nestedbisection;
extern struct timeval start_nestedbisection;
extern struct timeval end_nestedbisection;

extern reordering_real_t time_bisectionbest;
extern struct timeval start_bisectionbest;
extern struct timeval end_bisectionbest;

extern reordering_real_t time_coarsen;
extern struct timeval start_coarsen;
extern struct timeval end_coarsen;

extern reordering_real_t time_reorderbisection;
extern struct timeval start_reorderbisection;
extern struct timeval end_reorderbisection;

extern reordering_real_t time_refine2waynode;
extern struct timeval start_refine2waynode;
extern struct timeval end_refine2waynode;

extern reordering_real_t time_splitgraphreorder;
extern struct timeval start_splitgraphreorder;
extern struct timeval end_splitgraphreorder;

extern reordering_real_t time_match;
extern struct timeval start_match;
extern struct timeval end_match;

extern reordering_real_t time_createcoarsengraph;
extern struct timeval start_createcoarsengraph;
extern struct timeval end_createcoarsengraph;

extern reordering_real_t time_partitioninf2way;
extern struct timeval start_partitioninf2way;
extern struct timeval end_partitioninf2way;

extern reordering_real_t time_fm2waycutbalance;
extern struct timeval start_fm2waycutbalance;
extern struct timeval end_fm2waycutbalance;

extern reordering_real_t time_fm2waycutrefine;
extern struct timeval start_fm2waycutrefine;
extern struct timeval end_fm2waycutrefine;

extern reordering_real_t time_reorderinf2way;
extern struct timeval start_reorderinf2way;
extern struct timeval end_reorderinf2way;

extern reordering_real_t time_fmnodebalance;
extern struct timeval start_fmnodebalance;
extern struct timeval end_fmnodebalance;

extern reordering_real_t time_fm1sidenoderefine;
extern struct timeval start_fm1sidenoderefine;
extern struct timeval end_fm1sidenoderefine;

extern reordering_real_t time_fm2sidenoderefine;
extern struct timeval start_fm2sidenoderefine;
extern struct timeval end_fm2sidenoderefine;

extern reordering_real_t time_malloc;
extern struct timeval start_malloc;
extern struct timeval end_malloc;

extern reordering_real_t time_free;
extern struct timeval start_free;
extern struct timeval end_free;

/* reordergraph.c */
void mynd_Reorderpartition(graph_t *graph, reordering_int_t niparts, reordering_int_t level);
void mynd_Bisection(graph_t *graph, reordering_int_t niparts, reordering_int_t nthreads, reordering_int_t level);
void mynd_BisectionBest(graph_t *graph, reordering_int_t nthreads, reordering_int_t level);
void mynd_NestedBisection(graph_t *graph, reordering_int_t *reflect, reordering_int_t *reordernumend, reordering_int_t nthreads, reordering_int_t level);
void mynd_NestedBisection_omp(graph_t *graph, reordering_int_t *reflect, reordering_int_t *reordernum, reordering_int_t nthreads, reordering_int_t level);
void mynd_ReorderGraph(reordering_int_t *nvtxs, reordering_int_t *nedges, reordering_int_t *xadj, reordering_int_t *vwgt, reordering_int_t *adjncy, reordering_int_t *adjwgt, 
    reordering_int_t *treflect, reordering_int_t *reflect, reordering_int_t *compress, reordering_int_t *tcontrol, reordering_int_t *is_memery_manage_before, reordering_int_t nthreads);

/* compressgraph.c */
graph_t *mynd_Compress_Graph(reordering_int_t nvtxs, reordering_int_t *xadj, reordering_int_t *adjncy, reordering_int_t *vwgt, reordering_int_t *cptr, reordering_int_t *cind);

/* coarsen.c */
graph_t *mynd_CoarsenGraph(graph_t *graph, reordering_int_t Coarsen_Threshold);
graph_t *mynd_CoarsenGraphNlevels_metis(graph_t *graph, reordering_int_t Coarsen_Threshold, reordering_int_t nlevels);

/* initialpartition.c */
//  InitSeparator + GrowBisectionNode
void mynd_ReorderBisection(graph_t *graph, reordering_int_t niparts);

/* refine.c */
void mynd_Compute_Partition_Informetion_2way(graph_t *graph);
void mynd_Compute_Reorder_Informetion_2way(graph_t *graph);
void mynd_project_Reorder(graph_t *graph);
void mynd_FM_2WayCutRefine(graph_t *graph, reordering_real_t *ntpwgts, reordering_int_t niter);
void mynd_FM_2WayNodeRefine2Sided(graph_t *graph, reordering_int_t niter);
void mynd_FM_2WayNodeRefine1Sided(graph_t *graph, reordering_int_t niter);
void mynd_Refine2WayNode(graph_t *graph, graph_t *origraph);

/* splitgraph.c */
void mynd_SplitGraphReorder(graph_t *graph, graph_t **sub_lgraph, graph_t **sub_rgraph, reordering_int_t level);

/* mmdorder.c */
void mynd_MMD_Order_line(graph_t *graph, reordering_int_t *reflect, reordering_int_t *reordernum, reordering_int_t task_id);
reordering_int_t mynd_mmdint(reordering_int_t neqns, reordering_int_t *xadj, reordering_int_t *adjncy, reordering_int_t *head, reordering_int_t *forward,
    reordering_int_t *backward, reordering_int_t *qsize, reordering_int_t *list, reordering_int_t *marker);
void mynd_mmdnum(reordering_int_t neqns, reordering_int_t *perm, reordering_int_t *invp, reordering_int_t *qsize);
void mynd_mmdelm(reordering_int_t mdeg_node, reordering_int_t *xadj, reordering_int_t *adjncy, reordering_int_t *head, reordering_int_t *forward,
    reordering_int_t *backward, reordering_int_t *qsize, reordering_int_t *list, reordering_int_t *marker, reordering_int_t maxint, reordering_int_t tag);
void mynd_mmdupd(reordering_int_t ehead, reordering_int_t neqns, reordering_int_t *xadj, reordering_int_t *adjncy, reordering_int_t delta, reordering_int_t *mdeg,
    reordering_int_t *head, reordering_int_t *forward, reordering_int_t *backward, reordering_int_t *qsize, reordering_int_t *list,
    reordering_int_t *marker, reordering_int_t maxint, reordering_int_t *tag);
void mynd_genmmd(reordering_int_t neqns, reordering_int_t *xadj, reordering_int_t *adjncy, reordering_int_t *invp, reordering_int_t *perm,
    reordering_int_t delta, reordering_int_t *head, reordering_int_t *qsize, reordering_int_t *list, reordering_int_t *marker,
    reordering_int_t maxint, reordering_int_t *ncsub);
void mynd_MMD_Order(graph_t *graph, reordering_int_t *order, reordering_int_t *reordernum, reordering_int_t task_id);

/* match.c */
reordering_int_t mynd_Match_2HopAny(graph_t *graph, reordering_int_t *perm, reordering_int_t *match, reordering_int_t cnvtxs, reordering_int_t *r_nunmatched, reordering_int_t maxdegree);
reordering_int_t mynd_Match_2HopAll(graph_t *graph, reordering_int_t *perm, reordering_int_t *match, reordering_int_t cnvtxs, reordering_int_t *r_nunmatched, reordering_int_t maxdegree);
reordering_int_t mynd_Match_2Hop(graph_t *graph, reordering_int_t *perm, reordering_int_t *match, reordering_int_t cnvtxs, reordering_int_t nunmatched);
reordering_int_t mynd_Match_RM(graph_t *graph, reordering_int_t maxvwgt);
void mynd_BucketSortKeysInc(reordering_int_t n, reordering_int_t max, reordering_int_t *keys, reordering_int_t *tperm, reordering_int_t *perm);
reordering_int_t mynd_Match_SHEM(graph_t *graph, reordering_int_t maxvwgt);

/* createcoarsegraph.c */
void mynd_CreateCoarseGraph(graph_t *graph, reordering_int_t cnvtxs);
void mynd_CreateCoarseGraph_S(graph_t *graph, reordering_int_t cnvtxs);
void mynd_CreateCoarseGraph_BST(graph_t *graph, reordering_int_t cnvtxs);
void mynd_CreateCoarseGraph_BST_2(graph_t *graph, reordering_int_t cnvtxs);
void mynd_CreateCoarseGraph_HT(graph_t *graph, reordering_int_t cnvtxs);
void mynd_CreateCoarseGraph_HT_2(graph_t *graph, reordering_int_t cnvtxs);

/* balance.c */
reordering_real_t mynd_ComputeLoadImbalanceDiff(graph_t *graph, reordering_int_t nparts, reordering_real_t ubvec);
void mynd_Bnd2WayBalance(graph_t *graph, reordering_real_t *ntpwgts);
void mynd_General2WayBalance(graph_t *graph, reordering_real_t *ntpwgts);
void mynd_Balance2Way(graph_t *graph, reordering_real_t *ntpwgts);
void mynd_FM_2WayNodeBalance(graph_t *graph);

/* ikvsorti.c */
void mynd_ikvsorti(size_t n, ikv_t *base);

/* commom.c*/
reordering_int_t lyj_log2(reordering_int_t a);
void mynd_set_value_int(reordering_int_t n, reordering_int_t val, reordering_int_t *src);
void mynd_set_value_double(reordering_int_t n, reordering_real_t val, reordering_real_t *src);
void mynd_copy_double(reordering_int_t n, reordering_real_t *src, reordering_real_t *dst);
void mynd_copy_int(reordering_int_t n, reordering_int_t *src, reordering_int_t *dst);
reordering_int_t mynd_sum_int(reordering_int_t n, reordering_int_t *src, reordering_int_t ncon);
void mynd_select_sort(reordering_int_t *num, reordering_int_t length);
void mynd_select_sort_val(reordering_int_t *num, reordering_int_t length);
void mynd_gk_randinit(uint64_t seed);
void mynd_isrand(reordering_int_t seed);
void mynd_InitRandom(reordering_int_t seed);
uint64_t mynd_gk_randint64(void);
uint32_t mynd_gk_randint32(void);
reordering_int_t mynd_irand();
reordering_int_t mynd_rand_count();
reordering_int_t mynd_irandInRange(reordering_int_t max);
void mynd_irandArrayPermute(reordering_int_t n, reordering_int_t *p, reordering_int_t nshuffles, reordering_int_t flag);

/* graph.c */
void mynd_InitGraph(graph_t *graph);
graph_t *mynd_CreateGraph(void);
void mynd_SetupGraph_tvwgt(graph_t *graph);
void mynd_SetupGraph_label(graph_t *graph);
graph_t *mynd_SetupGraph(reordering_int_t nvtxs, reordering_int_t *xadj, reordering_int_t *adjncy, reordering_int_t *vwgt, reordering_int_t *adjwgt);
graph_t *mynd_SetupCoarseGraph(graph_t *graph, reordering_int_t cnvtxs);
graph_t *mynd_SetupSplitGraph(graph_t *graph, reordering_int_t subnvtxs, reordering_int_t subnedges);
void mynd_FreeRefineData(graph_t *graph);
void mynd_FreeGraph(graph_t **r_graph);
void mynd_Change2CNumbering(reordering_int_t nvtxs, reordering_int_t *xadj, reordering_int_t *adjncy);
void mynd_Change2FNumbering(reordering_int_t nvtxs, reordering_int_t *xadj, reordering_int_t *adjncy, reordering_int_t *vector);
void mynd_exam_nvtxs_nedges(graph_t *graph);
void mynd_exam_xadj(graph_t *graph);
void mynd_exam_vwgt(graph_t *graph);
void mynd_exam_adjncy_adjwgt(graph_t *graph);
void mynd_exam_label(graph_t *graph);
void mynd_exam_where(graph_t *graph);
void mynd_exam_pwgts(graph_t *graph);
void mynd_exam_edid(graph_t *graph);
void mynd_exam_bnd(graph_t *graph);
void mynd_exam_nrinfo(graph_t *graph);
void mynd_exam_num(reordering_int_t *num, reordering_int_t n);

/* hashtable.c */
//  Hash Table Version 1.0
void mynd_hash_table_Init(hash_table_t *hash, reordering_int_t size);
hash_table_t *mynd_hash_table_Create(reordering_int_t size);
void mynd_hashelement_Free(hash_table_t *hash);
void mynd_hash_table_Destroy(hash_table_t *hash);
reordering_int_t mynd_hash_table_Length(hash_table_t *hash);
reordering_int_t mynd_hashFunction(reordering_int_t val, reordering_int_t size);
reordering_int_t mynd_Insert_hashelement(hashelement_t *element, reordering_int_t size, reordering_int_t val, reordering_int_t key, reordering_int_t index);
void mynd_hash_table_Insert(hash_table_t *hash, reordering_int_t val, reordering_int_t key);
void mynd_Traversal_hashelement(hashelement_t *element, reordering_int_t size, reordering_int_t *dst1, reordering_int_t *dst2, reordering_int_t ptr);
void mynd_hash_table_Traversal(hash_table_t *hash, reordering_int_t *dst1, reordering_int_t *dst2);
//  Hash Table Version 2.0
void mynd_hash_table_Init2(hash_table2_t *hash, reordering_int_t size);
hash_table2_t *mynd_hash_table_Create2(reordering_int_t size);
void mynd_hashelement_Free2(hash_table2_t *hash);
void mynd_hash_table_Destroy2(hash_table2_t *hash);
reordering_int_t mynd_hash_table_Length2(hash_table2_t *hash);
void mynd_hash_table_Reset2(hash_table2_t *hash, reordering_int_t *src);
reordering_int_t mynd_Insert_hashelement2(reordering_int_t *element, reordering_int_t val, reordering_int_t key);
reordering_int_t mynd_hash_table_Insert2(hash_table2_t *hash, reordering_int_t val, reordering_int_t key);
reordering_int_t mynd_hash_table_Find2(hash_table2_t *hash, reordering_int_t val);

/* priorityqueue.c */
void mynd_priority_queue_Init(priority_queue_t *queue, reordering_int_t maxnodes);
priority_queue_t *mynd_priority_queue_Create(reordering_int_t maxnodes);
void mynd_priority_queue_Reset(priority_queue_t *queue);
void mynd_priority_queue_Free(priority_queue_t *queue);
void mynd_priority_queue_Destroy(priority_queue_t *queue);
reordering_int_t mynd_priority_queue_Length(priority_queue_t *queue);
reordering_int_t mynd_priority_queue_Insert(priority_queue_t *queue, reordering_int_t node, reordering_int_t key);
reordering_int_t mynd_priority_queue_Delete(priority_queue_t *queue, reordering_int_t node);
void mynd_priority_queue_Update(priority_queue_t *queue, reordering_int_t node, reordering_int_t newkey);
reordering_int_t mynd_priority_queue_GetTop(priority_queue_t *queue);
reordering_int_t mynd_priority_queue_SeeTopVal(priority_queue_t *queue);
reordering_int_t mynd_priority_queue_SeeTopKey(priority_queue_t *queue);
void mynd_exam_priority_queue(priority_queue_t *queue);

/* queue.c */
reordering_int_t mynd_init_queue(reordering_int_t ptr, reordering_int_t *bndptr, reordering_int_t nvtxs);
reordering_int_t mynd_insert_queue(reordering_int_t nbnd, reordering_int_t *bndptr, reordering_int_t *bndind, reordering_int_t vertex);
reordering_int_t mynd_delete_queue(reordering_int_t nbnd, reordering_int_t *bndptr, reordering_int_t *bndind, reordering_int_t vertex);

/* searchtree.h */
//  Binary Search Tree Version 1.0
void mynd_binary_search_tree_Init(binary_search_tree_t *tree);
binary_search_tree_t *mynd_binary_search_tree_Create();
void mynd_Free_Treenode(treenode_t *node);
void mynd_binary_search_tree_Free(binary_search_tree_t *tree);
void mynd_binary_search_tree_Destroy(binary_search_tree_t *tree);
reordering_int_t mynd_binary_search_tree_Length(binary_search_tree_t *tree);
treenode_t *mynd_Create_TreeNode(reordering_int_t val, reordering_int_t key);
treenode_t *mynd_Insert_TreeNode(treenode_t *node, reordering_int_t val, reordering_int_t key, reordering_int_t *nownodes);
void mynd_binary_search_tree_Insert(binary_search_tree_t *tree, reordering_int_t val, reordering_int_t key);
reordering_int_t mynd_InorderTraversal_TreeNode(treenode_t *root, reordering_int_t *dst1, reordering_int_t *dst2, reordering_int_t *ptr);
void mynd_binary_search_tree_Traversal(binary_search_tree_t *tree, reordering_int_t *dst1, reordering_int_t *dst2);
//  Binary Search Tree Version 2.0
void mynd_binary_search_tree_Init2(binary_search_tree2_t *tree, reordering_int_t size);
binary_search_tree2_t *mynd_binary_search_tree_Create2(reordering_int_t size);
void mynd_exam_binary_search_tree2(binary_search_tree2_t *tree);
void mynd_exam_binary_search_tree2_flag(binary_search_tree2_t *tree);
void mynd_binary_search_tree_Free2(binary_search_tree2_t *tree);
void mynd_binary_search_tree_Destroy2(binary_search_tree2_t *tree);
reordering_int_t mynd_binary_search_tree_Length2(binary_search_tree2_t *tree);
void mynd_Insert_TreeNode2(binary_search_tree2_t *tree, reordering_int_t val, reordering_int_t key);
void mynd_binary_search_tree_Insert2(binary_search_tree2_t *tree, reordering_int_t val, reordering_int_t key);
void mynd_InorderTraversal_TreeNode2(binary_search_tree2_t *tree, treenode2_t *treenode, reordering_int_t maxnodes, reordering_int_t *dst1, reordering_int_t *dst2, reordering_int_t located, reordering_int_t *ptr);
void mynd_binary_search_tree_Traversal2(binary_search_tree2_t *tree, reordering_int_t *dst1, reordering_int_t *dst2);
void mynd_Reset_TreeNode2(treenode2_t *treenode, reordering_int_t maxnodes, reordering_int_t located);
void mynd_binary_search_tree_Reset2(binary_search_tree2_t *tree);

/* timer.c */
void mynd_Timer_Init();
void mynd_gettimebegin(struct timeval *start, struct timeval *end, reordering_real_t *time);
void mynd_gettimeend(struct timeval *start, struct timeval *end, reordering_real_t *time);
void mynd_PrintTimeGeneral();
void mynd_PrintTimePhases();
void mynd_PrintTimeSteps();
void mynd_PrintTime(reordering_int_t control);

/* memory.c */
void mynd_error_exit(const char *error_message);
reordering_int_t mynd_find_between_last_slash_and_dotgraph(const char *filename);
reordering_int_t mynd_init_memery_manage(char *filename);
void mynd_log_memory(reordering_int_t task_type, reordering_int_t nbytes, void *ptr, char *message);
void mynd_add_memory_block(void *ptr, reordering_int_t nbytes, char *message);
void mynd_update_memory_block(void *ptr, void *oldptr, reordering_int_t nbytes, reordering_int_t old_nbytes, char *message);
void mynd_delete_memory_block(void *ptr, char *message);
void mynd_free_memory_block();
void *mynd_check_malloc(reordering_int_t nbytes, char *message);
void *mynd_check_realloc(void *oldptr, reordering_int_t nbytes, reordering_int_t old_nbytes, char *message);
void mynd_check_free(void *ptr, reordering_int_t nbytes, char *message);
void mynd_PrintMemory();
void mynd_exam_memory();

/* read.h */
reordering_int_t mynd_Is_file_exists(char *fname);
FILE *mynd_check_fopen(char *fname, char *mode, const char *message);
ssize_t mynd_check_getline(char **lineptr, reordering_int_t *n, FILE *stream);
void mynd_check_fclose(FILE *fp);
void mynd_ReadGraph(char *filename, reordering_int_t *nvtxs, reordering_int_t *nedges, reordering_int_t **txadj, reordering_int_t **tvwgt, reordering_int_t **tadjncy, reordering_int_t **tadjwgt);

#endif