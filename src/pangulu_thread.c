#include "pangulu_common.h"

void pangulu_bind_to_core(pangulu_int32_t core)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0)
    {
        printf(PANGULU_W_BIND_CORE_FAIL);
    }
}

void pangulu_mutex_init(pthread_mutex_t *mutex)
{
    pthread_mutex_init((mutex), NULL);
}

void pangulu_bsem_init(pangulu_bsem_t *bsem_p, pangulu_int64_t value)
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

pangulu_bsem_t *pangulu_bsem_destory(pangulu_bsem_t *bsem_p)
{
    pangulu_free(__FILE__, __LINE__, bsem_p->mutex);
    bsem_p->mutex = NULL;
    pangulu_free(__FILE__, __LINE__, bsem_p->cond);
    bsem_p->cond = NULL;
    bsem_p->v = 0;
    pangulu_free(__FILE__, __LINE__, bsem_p);
    return NULL;
}

void pangulu_bsem_post(pangulu_task_queue_t *heap)
{
    pangulu_bsem_t *bsem_p = heap->heap_bsem;
    pthread_mutex_lock(bsem_p->mutex);
    pangulu_int64_t flag = pangulu_task_queue_empty(heap);
    if (((bsem_p->v == 0) && (flag == 0)))
    {
        bsem_p->v = 1;
        pthread_cond_signal(bsem_p->cond);
    }
    pthread_mutex_unlock(bsem_p->mutex);
}

void pangulu_bsem_stop(pangulu_task_queue_t *heap)
{
    pangulu_bsem_t *bsem_p = heap->heap_bsem;
    pthread_mutex_lock(bsem_p->mutex);
    bsem_p->v = 0;
    pthread_mutex_unlock(bsem_p->mutex);
}

void pangulu_bsem_synchronize(pangulu_bsem_t *bsem_p)
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