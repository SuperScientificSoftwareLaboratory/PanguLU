#ifndef IKVSORTI_H
#define IKVSORTI_H

#include "mynd_functionset.h"

void mynd_ikvsorti(size_t n, ikv_t *base)
{
#define ikey_lt(a, b) ((a)->key < (b)->key)
    GK_MKQSORT(ikv_t, base, n, ikey_lt);
#undef ikey_lt
}

#endif