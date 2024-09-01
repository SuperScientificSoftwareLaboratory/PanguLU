#include "pangulu_common.h"

void pangulu_mutex_init(pthread_mutex_t *mutex)
{
    pthread_mutex_init((mutex), NULL);
}

void pangulu_bsem_init(bsem *bsem_p, pangulu_int64_t value)
{
    if (value < 0 || value > 1)
    {
        exit(1);
    }
    bsem_p->mutex = (pthread_mutex_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pthread_mutex_t));
    bsem_p->cond = (pthread_cond_t *)pangulu_malloc(__FILE__, __LINE__, sizeof(pthread_cond_t));
    pangulu_mutex_init((bsem_p->mutex));
    pthread_cond_init((bsem_p->cond), NULL);
    bsem_p->v = value;
}

bsem *pangulu_bsem_destory(bsem *bsem_p)
{
    pangulu_free(__FILE__, __LINE__, bsem_p->mutex);
    bsem_p->mutex = NULL;
    pangulu_free(__FILE__, __LINE__, bsem_p->cond);
    bsem_p->cond = NULL;
    bsem_p->v = 0;
    pangulu_free(__FILE__, __LINE__, bsem_p);
    return NULL;
}

void pangulu_bsem_post(pangulu_heap *heap)
{
    bsem *bsem_p = heap->heap_bsem;
    pthread_mutex_lock(bsem_p->mutex);
    pangulu_int64_t flag = heap_empty(heap);
    if (((bsem_p->v == 0) && (flag == 0)))
    {
        bsem_p->v = 1;
        // get bsem p
        pthread_cond_signal(bsem_p->cond);
        // send
    }
    pthread_mutex_unlock(bsem_p->mutex);
}

pangulu_int64_t pangulu_bsem_wait(pangulu_heap *heap)
{
    bsem *heap_bsem = heap->heap_bsem;
    pthread_mutex_t *heap_mutex = heap_bsem->mutex;

    pthread_mutex_lock(heap_mutex);
    if (heap_empty(heap) == 1)
    {
        heap_bsem->v = 0;
        while (heap_bsem->v == 0)
        {
            // wait
            pthread_cond_wait(heap_bsem->cond, heap_bsem->mutex);
        }
    }

    pangulu_int64_t compare_flag = pangulu_heap_delete(heap);
    heap_bsem->v = 1;
    pthread_mutex_unlock(heap_mutex);
    return compare_flag;
}

void pangulu_bsem_stop(pangulu_heap *heap)
{
    bsem *bsem_p = heap->heap_bsem;
    pthread_mutex_lock(bsem_p->mutex);
    bsem_p->v = 0;
    pthread_mutex_unlock(bsem_p->mutex);
}

void pangulu_bsem_synchronize(bsem *bsem_p)
{
    pthread_mutex_lock((bsem_p->mutex));
    pangulu_int64_t v = bsem_p->v;
    if (v == 1)
    {
        bsem_p->v = 0;
        pthread_cond_signal(bsem_p->cond);
        pthread_mutex_unlock(bsem_p->mutex);
    }
    else
    {
        bsem_p->v = 1;
        while (bsem_p->v == 1)
        {
            pthread_cond_wait((bsem_p->cond), (bsem_p->mutex));
            bsem_p->v = 0;
        }
        bsem_p->v = 0;
        pthread_mutex_unlock(bsem_p->mutex);
    }
}