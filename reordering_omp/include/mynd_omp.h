#ifndef MYND_OMP_H
#define MYND_OMP_H

#include "mynd_types.h"
#if defined(_MSC_VER)
  #define COMPILER_MSC
#endif
#if defined(__ICC)
  #define COMPILER_ICC
#endif
#if defined(__GNUC__)
  #define COMPILER_GCC
#endif

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

#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C"
{
#endif

    void mynd_ReorderGraph(reordering_int_t *nvtxs, reordering_int_t *nedges, reordering_int_t *xadj, reordering_int_t *vwgt, reordering_int_t *adjncy, reordering_int_t *adjwgt, 
        reordering_int_t *treflect, reordering_int_t *reflect, reordering_int_t *compress, reordering_int_t *tcontrol, reordering_int_t *is_memery_manage_before, reordering_int_t nthreads);
    void mynd_ReadGraph(char *filename, reordering_int_t *nvtxs, reordering_int_t *nedges, reordering_int_t **txadj, reordering_int_t **tvwgt, reordering_int_t **tadjncy, reordering_int_t **tadjwgt);

#ifdef __cplusplus
}
#endif

#endif
