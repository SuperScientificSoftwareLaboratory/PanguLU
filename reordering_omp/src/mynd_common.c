#ifndef COMMON_H
#define COMMON_H

#include "mynd_functionset.h"

/*************************************************************************
* This function returns the log2(x)
**************************************************************************/
reordering_int_t lyj_log2(reordering_int_t a) 
{
    if (a <= 0) 
	{
        fprintf(stderr, "lyj_log2: Input must be greater than 0.\n");
        return -1;
    }

    reordering_int_t i = 0;
    while (a > 1) 
	{
        a >>= 1;
        i++;
    }

    return i;
}

void mynd_set_value_int(reordering_int_t n, reordering_int_t val, reordering_int_t *src)
{
	reordering_int_t i;
	for (i = 0; i < n; i++)
    	src[i] = val;
}

void mynd_set_value_double(reordering_int_t n, reordering_real_t val, reordering_real_t *src)
{
	reordering_int_t i;
	for (i = 0; i < n; i++)
    	src[i] = val;
}

void mynd_copy_double(reordering_int_t n, reordering_real_t *src, reordering_real_t *dst)
{
	for (reordering_int_t i = 0; i < n; i++)
    	dst[i] = src[i];
}

void mynd_copy_int(reordering_int_t n, reordering_int_t *src, reordering_int_t *dst)
{
	for (reordering_int_t i = 0; i < n; i++)
    	dst[i] = src[i];
}

reordering_int_t mynd_sum_int(reordering_int_t n, reordering_int_t *src, reordering_int_t ncon)
{
	reordering_int_t sum = 0;
	for(reordering_int_t i = 0;i < n;i++)
		sum += src[i];
	return sum;
}

void mynd_select_sort(reordering_int_t *num, reordering_int_t length)
{
	for(reordering_int_t i = 0;i < length;i++)
	{
		reordering_int_t t = i;
		for(reordering_int_t j = i + 1;j < length;j++)
			if(num[j] < num[t]) t = j;
		reordering_int_t z;
		lyj_swap(num[t], num[i],z);
		// printf("i=%d t=%d num: ",i,t);
		// for(reordering_int_t j = 0;j < length;j++)
		// 	printf("%d ",num[j]);
		// printf("\n");
	}
}

void mynd_select_sort_val(reordering_int_t *num, reordering_int_t length)
{
	for(reordering_int_t i = 0;i < length;i++)
	{
		reordering_int_t t = i;
		for(reordering_int_t j = i + 1;j < length;j++)
			if(num[j] < num[t]) t = j;
		reordering_int_t z;
		lyj_swap(num[t], num[i], z);
		lyj_swap(num[t + length], num[i + length],z);
	}
}
//	USE_GKRAND ???
void mynd_gk_randinit(uint64_t seed)
{
#ifdef USE_GKRAND
  mt[0] = seed;
  for (mti=1; mti<NN; mti++) 
    mt[mti] = (6364136223846793005ULL * (mt[mti-1] ^ (mt[mti-1] >> 62)) + mti);
#else
  srand((unsigned int) seed);
#endif
}

/*************************************************************************/
/*! Initializes the generator */ 
/**************************************************************************/
void mynd_isrand(reordering_int_t seed)
{
  mynd_gk_randinit((uint64_t) seed);
}

/*************************************************************************/
/*! This function initializes the random number generator 
  */
/*************************************************************************/
void mynd_InitRandom(reordering_int_t seed)
{
	mynd_isrand((seed == -1 ? 4321 : seed));
}

/* generates a random number on [0, 2^64-1]-interval */
uint64_t mynd_gk_randint64(void)
{
#ifdef USE_GKRAND
#else
  return (uint64_t)(((uint64_t) rand()) << 32 | ((uint64_t) rand()));
#endif
}

/* generates a random number on [0, 2^32-1]-interval */
uint32_t mynd_gk_randint32(void)
{
#ifdef USE_GKRAND
#else
  return (uint32_t)rand();
#endif
}

/*************************************************************************/
/*! Returns a random number */ 
/**************************************************************************/
reordering_int_t mynd_irand()
{
  if (sizeof(reordering_int_t) <= sizeof(int32_t)) 
    return (reordering_int_t)mynd_gk_randint32();
  else 
    return (reordering_int_t)mynd_gk_randint64(); 
}

reordering_int_t mynd_rand_count()
{
	static int ccnt = 0;  
	ccnt++;   
	return ccnt;
}

/*************************************************************************/
/*! Returns a random number between [0, max) */ 
/**************************************************************************/
reordering_int_t mynd_irandInRange(reordering_int_t max) 
{
	// reordering_int_t t = rand_count(); 
	// if(t % 10000 == 0)  printf("ccnt=%d\n",t);
	return (reordering_int_t)((mynd_irand())%max); 
}

/*************************************************************************/
/*! Randomly permutes the elements of an array p[]. 
    flag == 1, p[i] = i prior to permutation, 
    flag == 0, p[] is not initialized. */
/**************************************************************************/
void mynd_irandArrayPermute(reordering_int_t n, reordering_int_t *p, reordering_int_t nshuffles, reordering_int_t flag)
{
	reordering_int_t i, u, v;
	reordering_int_t tmp;

	if (flag == 1) 
	{
		for (i = 0; i < n; i++)
			p[i] = (reordering_int_t)i;
	}

	if (n < 10) 
	{
		for (i = 0; i < n; i++) 
		{
			v = mynd_irandInRange(n);
			u = mynd_irandInRange(n);
			lyj_swap(p[v], p[u], tmp);
		}
	}
	else 
	{
		for (i = 0; i < nshuffles; i++) 
		{
			v = mynd_irandInRange(n - 3);
			u = mynd_irandInRange(n - 3);
			lyj_swap(p[v + 0], p[u + 2], tmp);
			lyj_swap(p[v + 1], p[u + 3], tmp);
			lyj_swap(p[v + 2], p[u + 0], tmp);
			lyj_swap(p[v + 3], p[u + 1], tmp);
		}
	}
}

#endif