#ifndef MYND_OMP_H
#define MYND_OMP_H

#include "../include/mynd_types.h"

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