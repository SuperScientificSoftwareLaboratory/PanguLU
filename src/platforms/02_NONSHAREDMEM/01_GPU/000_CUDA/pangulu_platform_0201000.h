#include "../../../../pangulu_common.h"

#ifdef __cplusplus
extern "C"{
#endif

void pangulu_platform_0201000_malloc(void** platform_address, size_t size);

void pangulu_platform_0201000_malloc_pinned(void** platform_address, size_t size);

void pangulu_platform_0201000_synchronize();

void pangulu_platform_0201000_memset(void* s, int c, size_t n);

void pangulu_platform_0201000_create_stream(void** stream);

void pangulu_platform_0201000_memcpy(void *dst, const void *src, size_t count, unsigned int kind);

void pangulu_platform_0201000_memcpy_async(void *dst, const void *src, size_t count, unsigned int kind, void* stream);

void pangulu_platform_0201000_free(void* devptr);

void pangulu_platform_0201000_get_device_num(int* device_num);

void pangulu_platform_0201000_set_default_device(int device_num);

void pangulu_platform_0201000_get_device_name(char* name, int device_num);

void pangulu_platform_0201000_get_device_memory_usage(size_t *used_byte);


void pangulu_platform_0201000_getrf(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t* opdst,
    int tid
);
void pangulu_platform_0201000_tstrf(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t* opdst,
    pangulu_storage_slot_t* opdiag,
    int tid
);
void pangulu_platform_0201000_gessm(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t* opdst,
    pangulu_storage_slot_t* opdiag,
    int tid
);
void pangulu_platform_0201000_ssssm(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t* opdst,
    pangulu_storage_slot_t* op1,
    pangulu_storage_slot_t* op2,
    int tid
);
void pangulu_platform_0201000_ssssm_batched(
    pangulu_inblock_idx nb,
    pangulu_uint64_t ntask,
    pangulu_task_t* tasks
);
void pangulu_platform_0201000_hybrid_batched(
    pangulu_inblock_idx nb,
    pangulu_uint64_t ntask,
    pangulu_task_t* tasks
);

void pangulu_platform_0201000_spmv(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t* a,
    calculate_type* x,
    calculate_type* y
);
void pangulu_platform_0201000_vecadd(
    pangulu_int64_t length,
    calculate_type *bval, 
    calculate_type *xval
);
void pangulu_platform_0201000_sptrsv(
    pangulu_inblock_idx nb,
    pangulu_storage_slot_t *s,
    calculate_type* xval,
    pangulu_int64_t uplo
);

#ifdef __cplusplus
}
#endif