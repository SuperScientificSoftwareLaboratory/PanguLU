#ifndef STRUCT_H
#define STRUCT_H

#include "typedef.h"

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