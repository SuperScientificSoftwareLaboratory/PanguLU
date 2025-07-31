#ifndef STRUCT_H
#define STRUCT_H

#include "typedef.h"

/*************************************************************************/
/*! This data structure stores the various command line arguments */
/*************************************************************************/
typedef struct 
{
	Hunyuan_int_t ptype;
	Hunyuan_int_t objtype;
	Hunyuan_int_t ctype;
	Hunyuan_int_t iptype;
	Hunyuan_int_t rtype;

	Hunyuan_int_t no2hop;
	Hunyuan_int_t minconn;
	Hunyuan_int_t contig;

	Hunyuan_int_t nooutput;

	Hunyuan_int_t balance;
	Hunyuan_int_t ncuts;
	Hunyuan_int_t niter;

	Hunyuan_int_t gtype;
	Hunyuan_int_t ncommon;

	Hunyuan_int_t seed;
	Hunyuan_int_t dbglvl;

	Hunyuan_int_t nparts;

	Hunyuan_int_t nseps;
	Hunyuan_int_t ufactor;
	Hunyuan_int_t pfactor;
	Hunyuan_int_t compress;
	Hunyuan_int_t ccorder;

	char *filename;
	char *outfile;
	char *xyzfile;
	char *tpwgtsfile;
	char *ubvecstr;

	Hunyuan_int_t wgtflag;
	Hunyuan_int_t numflag;
	Hunyuan_real_t *tpwgts;
	Hunyuan_real_t *ubvec;

	Hunyuan_real_t iotimer;
	Hunyuan_real_t parttimer;
	Hunyuan_real_t reporttimer;

	Hunyuan_int_t maxmemory;
} params_t;

/*************************************************************************/
/*! The following data structure stores holds information on degrees for k-way
    partition */
/*************************************************************************/
typedef struct ckrinfo_t 
{
	Hunyuan_int_t id;              /*!< The Hunyuan_int_ternal degree of a vertex (sum of weights) */
	Hunyuan_int_t ed;            	/*!< The total external degree of a vertex */
	Hunyuan_int_t nnbrs;          	/*!< The number of neighboring subdomains */
	Hunyuan_int_t inbr;            /*!< The index in the cnbr_t array where the nnbrs list 
                             of neighbors is stored */
} ckrinfo_t;

/*************************************************************************/
/*! The following data structure holds information on degrees for k-way
    vol-based partition */
/*************************************************************************/
typedef struct vkrinfo_t 
{
	Hunyuan_int_t nid;             /*!< The Hunyuan_int_ternal degree of a vertex (count of edges) */
	Hunyuan_int_t ned;            	/*!< The total external degree of a vertex (count of edges) */
	Hunyuan_int_t gv;            	/*!< The volume gain of moving that vertex */
	Hunyuan_int_t nnbrs;          	/*!< The number of neighboring subdomains */
	Hunyuan_int_t inbr;            /*!< The index in the vnbr_t array where the nnbrs list 
                             of neighbors is stored */
} vkrinfo_t;

/*************************************************************************/
/*! The following data structure holds information on degrees for k-way
    partition */
/*************************************************************************/
typedef struct nrinfo_t 
{
	Hunyuan_int_t edegrees[2];  
} nrinfo_t;

/*************************************************************************/
/*! Binary Search Tree node */
/*************************************************************************/
typedef struct treenode_t
{ 
	Hunyuan_int_t val;  //  ncy
	Hunyuan_int_t key;  //  wgt
	struct treenode_t *left;
	struct treenode_t *right;
} treenode_t;

/*************************************************************************/
/*! Binary Search Tree */
/*************************************************************************/
typedef struct binary_search_tree_t
{ 
	Hunyuan_int_t nownodes; 
	// size_t maxnodes; 
	treenode_t *treenode; 
	
} binary_search_tree_t;

/*************************************************************************/
/*! Hash Table Element */
/*************************************************************************/
typedef struct hashelement_t
{ 
	Hunyuan_int_t val;  //  ncy
	Hunyuan_int_t key;  //  wgt
} hashelement_t;

/*************************************************************************/
/*! Hash Table */
/*************************************************************************/
typedef struct hash_table_t
{ 
	Hunyuan_int_t nownodes; 
	Hunyuan_int_t maxnodes; 
	hashelement_t *hashelement; 
} hash_table_t;

/*************************************************************************/
/*! Hash Table */
/*************************************************************************/
typedef struct hash_table2_t
{ 
	Hunyuan_int_t nownodes; 
	Hunyuan_int_t maxnodes; 
	Hunyuan_int_t *hashelement; 
} hash_table2_t;

/*************************************************************************/
/*! Hash Table */
/*************************************************************************/
typedef struct hash_table_omp_t
{ 
	// Hunyuan_int_t *nownodes; 
	Hunyuan_int_t *maxnodes; 
	Hunyuan_int_t *index;
	Hunyuan_int_t *val;
	Hunyuan_int_t *key;
} hash_table_omp_t;

/*************************************************************************/
/*! Hash Table */
/*************************************************************************/
typedef struct hash_table_omp2_t
{
	Hunyuan_int_t maxnodes;
	Hunyuan_int_t *val;
	Hunyuan_int_t *key;
} hash_table_omp2_t;

/*************************************************************************/
/*! Binary Search Tree node */
/*************************************************************************/
typedef struct treenode2_t
{ 
	Hunyuan_int_t val;  //  ncy
	Hunyuan_int_t key;  //  wgt
} treenode2_t;

/*************************************************************************/
/*! Binary Search Tree */
/*************************************************************************/
typedef struct binary_search_tree2_t
{ 
	Hunyuan_int_t nownodes; 
	Hunyuan_int_t maxnodes; 
	treenode2_t *treenode; 
} binary_search_tree2_t;

/*************************************************************************/
/*! This data structure holds a graph */
/*************************************************************************/
typedef struct graph_t 
{
    Hunyuan_int_t nvtxs, nedges;	/* The # of vertices and edges in the graph */
    // Hunyuan_int_t ncon;		/* The # of constrains */ 
    Hunyuan_int_t *xadj;		/* PoHunyuan_int_ters to the locally stored vertices */
    Hunyuan_int_t *vwgt;		/* Vertex weights */
    // Hunyuan_int_t *vsize;		/* Vertex sizes for min-volume formulation */
    Hunyuan_int_t *adjncy;        /* Array that stores the adjacency lists of nvtxs */
    Hunyuan_int_t *adjwgt;        /* Array that stores the weights of the adjacency lists */

    Hunyuan_int_t *tvwgt;         /* The sum of the vertex weights in the graph */
    Hunyuan_real_t *invtvwgt;     /* The inverse of the sum of the vertex weights in the graph */


    /* These are to keep track control if the corresponding fields correspond to
        application or library memory */
    // Hunyuan_int_t free_xadj, free_vwgt, free_vsize, free_adjncy, free_adjwgt;

    Hunyuan_int_t *label;
    Hunyuan_int_t *cmap;
	Hunyuan_int_t *match;
	void **addr;
	binary_search_tree2_t *tree;
	hash_table2_t *hash;

    /* Partition parameters */
    Hunyuan_int_t mincut, minvol;
    Hunyuan_int_t *where, *pwgts;
    Hunyuan_int_t nbnd;
    Hunyuan_int_t *bndptr, *bndind;

    /* Bisection refinement parameters */
    Hunyuan_int_t *id, *ed;

    /* K-way refinement parameters */
    // ckrinfo_t *ckrinfo;   /*!< The per-vertex cut-based refinement info */
    // vkrinfo_t *vkrinfo;   /*!< The per-vertex volume-based refinement info */

    /* Node refinement information */
    nrinfo_t *nrinfo;	//	replace after

    struct graph_t *coarser, *finer;
} graph_t;

typedef struct ikv_t
{
  Hunyuan_int_t key;
  Hunyuan_int_t val;
} ikv_t;

/*************************************************************************/
/*! The following structure stores information used by Metis */
/*************************************************************************/
typedef struct ctrl_t 
{
	Hunyuan_int_t optype;
	Hunyuan_int_t objtype;
	Hunyuan_int_t ctype;
	Hunyuan_int_t iptype;
	Hunyuan_int_t rtype;
	// moptype_et  optype;	        /* Type of operation */
	// mobjtype_et objtype;          /* Type of refinement objective */
	// mdbglvl_et  dbglvl;		/* Controls the debuging output of the program */
	// mctype_et   ctype;		/* The type of coarsening */
	// miptype_et  iptype;		/* The type of initial partitioning */
	// mrtype_et   rtype;		/* The type of refinement */

	Hunyuan_int_t CoarsenTo;		/* The # of vertices in the coarsest graph */
	Hunyuan_int_t nIparts;                /* The number of initial partitions to compute */
	Hunyuan_int_t no2hop;                 /* Indicates if 2-hop matching will be used */
	Hunyuan_int_t minconn;                /* Indicates if the subdomain connectivity will be minimized */
	Hunyuan_int_t contig;                 /* Indicates if contigous partitions are required */
	Hunyuan_int_t nseps;			/* The number of separators to be found during multiple bisections */
	Hunyuan_int_t ufactor;                /* The user-supplied load imbalance factor */
	Hunyuan_int_t compress;               /* If the graph will be compressed prior to ordering */
	Hunyuan_int_t ccorder;                /* If connected components will be ordered separately */
	Hunyuan_int_t seed;                   /* The seed for the random number generator */
	Hunyuan_int_t ncuts;                  /* The number of different partitionings to compute */
	Hunyuan_int_t niter;                  /* The number of iterations during each refinement */
	Hunyuan_int_t numflag;                /* The user-supplied numflag for the graph */
	Hunyuan_int_t *maxvwgt;		/* The maximum allowed weight for a vertex */

	Hunyuan_int_t ncon;                   /*!< The number of balancing constraHunyuan_int_ts */
	Hunyuan_int_t nparts;                 /*!< The number of partitions */

	Hunyuan_real_t pfactor;		/* .1*(user-supplied prunning factor) */

	Hunyuan_real_t *ubfactors;            /*!< The per-constraHunyuan_int_t ubfactors */
	
	Hunyuan_real_t *tpwgts;               /*!< The target partition weights */
	Hunyuan_real_t *pijbm;                /*!< The nparts*ncon multiplies for the ith partition
										and jth constraHunyuan_int_t for obtaining the balance */

	Hunyuan_real_t cfactor;               /*!< The achieved compression factor */

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
	Hunyuan_int_t *maxnads;               /* The maximum allocated number of adjacent domains */
	Hunyuan_int_t *nads;                  /* The number of adjacent domains */
	Hunyuan_int_t **adids;                /* The IDs of the adjacent domains */
	Hunyuan_int_t **adwgts;               /* The edge-weight to the adjacent domains */
	Hunyuan_int_t *pvec1, *pvec2;         /* Auxiliar nparts-size vectors for efficiency */

} ctrl_t;

/*************************************************************************/
/*! binary heap node */
/*************************************************************************/
typedef struct node_t
{ 
	Hunyuan_int_t key;  //  ed - id
	Hunyuan_int_t val;  //  vertex
} node_t;

/*************************************************************************/
/*! Priority queue based on binary heap */
/*************************************************************************/
typedef struct priority_queue_t
{ 
	Hunyuan_int_t nownodes; 
	Hunyuan_int_t maxnodes; 
	node_t *heap; 
	Hunyuan_int_t *locator;
} priority_queue_t;

/*************************************************************************/
/*! The following data structure stores information about a memory 
    allocation operation that can either be served from gk_mcore_t or by
    a gk_malloc if not sufficient workspace memory is available. */
/*************************************************************************/
typedef struct memory_block {
	// Hunyuan_int_t type;
	Hunyuan_int_t nbytes;
	void *ptr;
} memory_block;

typedef struct memory_manage {
	Hunyuan_int_t used_block;
	Hunyuan_int_t all_block;
	Hunyuan_int_t now_memory;
	Hunyuan_int_t max_memory;
	memory_block *memoryblock;
} memory_manage;

#endif