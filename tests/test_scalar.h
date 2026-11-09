#ifndef PANGULU_TEST_SCALAR_H
#define PANGULU_TEST_SCALAR_H
#include <complex.h>
typedef unsigned int sparse_index_t;
typedef unsigned long long sparse_pointer_t;
#if defined(CALCULATE_TYPE_CR64)
typedef double _Complex sparse_value_t;
#define COMPLEX_VALUES 1
#define SMALL_SCALE 1e-80
#define LARGE_SCALE 1e80
#define FORWARD_TOL 1e-11
#define BACKWARD_TOL 1e-12
#elif defined(CALCULATE_TYPE_CR32)
typedef float _Complex sparse_value_t;
#define COMPLEX_VALUES 1
#define SMALL_SCALE 1e-20
#define LARGE_SCALE 1e20
#define FORWARD_TOL 2e-5
#define BACKWARD_TOL 2e-6
#elif defined(CALCULATE_TYPE_R32)
typedef float sparse_value_t;
#define COMPLEX_VALUES 0
#define SMALL_SCALE 1e-20
#define LARGE_SCALE 1e20
#define FORWARD_TOL 2e-5
#define BACKWARD_TOL 2e-6
#else
typedef double sparse_value_t;
#define COMPLEX_VALUES 0
#define SMALL_SCALE 1e-80
#define LARGE_SCALE 1e80
#define FORWARD_TOL 1e-11
#define BACKWARD_TOL 1e-12
#endif
#include "pangulu.h"
#endif
