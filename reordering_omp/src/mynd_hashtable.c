#ifndef HASHTABLE_H
#define HASHTABLE_H

#include "mynd_functionset.h"

//  Hash Table Version 1.0
void mynd_hash_table_Init(hash_table_t *hash, reordering_int_t size)
{
    hash->nownodes = 0;
    hash->maxnodes = size;

    hash->hashelement = (hashelement_t *)mynd_check_malloc(sizeof(hashelement_t) * size, "mynd_hash_table_Init: hashelement");

    for(int i = 0;i < size;i++)
        hash->hashelement[i].val = -1;
}

hash_table_t *mynd_hash_table_Create(reordering_int_t size)
{
    hash_table_t *hash;

    hash = (hash_table_t *)mynd_check_malloc(sizeof(hash_table_t), "mynd_hash_table_Create: hash");
    mynd_hash_table_Init(hash, size);

    return hash;
}

void mynd_hashelement_Free(hash_table_t *hash)
{
    if (hash == NULL) return;
    mynd_check_free(hash->hashelement, sizeof(hashelement_t) * hash->maxnodes, "mynd_hashelement_Free: hash->hashelement");
    hash->nownodes = 0;
    hash->maxnodes = 0;
}

void mynd_hash_table_Destroy(hash_table_t *hash)
{
    if (hash == NULL) return;
	mynd_hashelement_Free(hash);
	mynd_check_free(hash, sizeof(hash_table_t), "hash_table_Destroy: hash");
}

reordering_int_t mynd_hash_table_Length(hash_table_t *hash)
{
	return hash->nownodes;
}

reordering_int_t mynd_hashFunction(reordering_int_t val, reordering_int_t size)
{
    return val % size;
}

reordering_int_t mynd_Insert_hashelement(hashelement_t *element, reordering_int_t size, reordering_int_t val, reordering_int_t key, reordering_int_t index)
{
    reordering_int_t start_index = index;
    do
    {
        if(element[index].val == -1)
        {
            element[index].val = val;
            element[index].key = key;
            return 1;
        }
        else if(element[index].val == val)
        {
            element[index].key += key;
            return 0;
        }

        index++;
        if(index >= size)
            index = 0;
    } while (index != start_index);

    return 0;
}

void mynd_hash_table_Insert(hash_table_t *hash, reordering_int_t val, reordering_int_t key)
{
    reordering_int_t index = mynd_hashFunction(val, hash->maxnodes);
    
    hash->nownodes += mynd_Insert_hashelement(hash->hashelement,hash->maxnodes, val, key, index);

	return ;
}

void mynd_Traversal_hashelement(hashelement_t *element, reordering_int_t size, reordering_int_t *dst1, reordering_int_t *dst2, reordering_int_t ptr) 
{
    if (element != NULL) 
	{
        for(reordering_int_t i = 0;i < size;i++)
        {
            reordering_int_t t = element[i].val;
            if(t != -1)
            {
                dst1[ptr] = t;
			    dst2[ptr] = element[i].key;
                ptr++;
            }
        }
    }
}

void mynd_hash_table_Traversal(hash_table_t *hash, reordering_int_t *dst1, reordering_int_t *dst2)
{
    hashelement_t *element = hash->hashelement;

    mynd_Traversal_hashelement(element, hash->maxnodes, dst1, dst2, 0);
}


//  Hash Table Version 2.0
void mynd_hash_table_Init2(hash_table2_t *hash, reordering_int_t size)
{
    hash->nownodes = 0;
    hash->maxnodes = size;

    hash->hashelement = (reordering_int_t *)mynd_check_malloc(sizeof(reordering_int_t) * size, "hash_table_Init2: hashelement");

    for(int i = 0;i < size;i++)
        hash->hashelement[i] = -1;
}

hash_table2_t *mynd_hash_table_Create2(reordering_int_t size)
{
    hash_table2_t *hash;

    hash = (hash_table2_t *)mynd_check_malloc(sizeof(hash_table2_t), "hash_table_Create2: hash");

    mynd_hash_table_Init2(hash, size);

    return hash;
}

void mynd_hashelement_Free2(hash_table2_t *hash)
{
    if (hash == NULL) return;
    mynd_check_free(hash->hashelement, sizeof(reordering_int_t) * hash->maxnodes, "hashelement_Free2: hash->hashelement");
    hash->nownodes = 0;
    hash->maxnodes = 0;
}

void mynd_hash_table_Destroy2(hash_table2_t *hash)
{
    if (hash == NULL) return;
	mynd_hashelement_Free2(hash);
	mynd_check_free(hash, sizeof(hash_table2_t), "hash_table_Destroy2: hash");
}

reordering_int_t mynd_hash_table_Length2(hash_table2_t *hash)
{
	return hash->nownodes;
}

void mynd_hash_table_Reset2(hash_table2_t *hash, reordering_int_t *src)
{
    for(reordering_int_t i = 0;i < hash->nownodes;i++)
        hash->hashelement[src[i]] = -1;
    hash->nownodes = 0;
}

reordering_int_t mynd_Insert_hashelement2(reordering_int_t *element, reordering_int_t val, reordering_int_t key)
{
    if(element[val] == -1)
    {
        element[val] = key;
        return 1;
    }

    return 0;
}

//  0 --> Already Exist
//  1 --> Be Inserting
reordering_int_t mynd_hash_table_Insert2(hash_table2_t *hash, reordering_int_t val, reordering_int_t key)
{
    reordering_int_t flag = 0;
    
    flag = mynd_Insert_hashelement2(hash->hashelement, val, key);

    hash->nownodes += flag;

	return flag;
}

reordering_int_t mynd_hash_table_Find2(hash_table2_t *hash, reordering_int_t val)
{
    if(hash->hashelement[val] == -1)
        return -1;
    else 
        return hash->hashelement[val];
}

#endif