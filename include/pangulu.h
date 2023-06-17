#ifndef PANGULU_H
#define PANGULU_H

#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <math.h>

/*
    this function is panguLU test file
    The input parameters are as follows:
    -np is MPI process numbers.
    -NB is the matrix block size.
    -F is the solution matrix name.
*/
    void pangulu(int ARGC, char **ARGV);


#endif
