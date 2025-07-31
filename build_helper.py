#!/usr/bin/python3
import csv
import os
import sys
import subprocess
import time

platform_functions_info = [
    ["malloc", "void** platform_address, size_t size", "platform_address, size"],
    ["malloc_pinned", "void** platform_address, size_t size", "platform_address, size"],
    ["synchronize", "", ""],
    ["memset", "void* s, int c, size_t n", "s, c, n"],
    ["create_stream", "void** stream", "stream"],
    ["memcpy", "void *dst, const void *src, size_t count, unsigned int kind", "dst, src, count, kind"],
    ["memcpy_async", "void *dst, const void *src, size_t count, unsigned int kind, void* stream", "dst, src, count, kind, stream"],
    ["free", "void* devptr", "devptr"],
    ["get_device_num", "int* device_num", "device_num"],
    ["set_default_device", "int device_num", "device_num"],
    ["get_device_name", "char* name, int device_num", "name, device_num"],
    ["get_device_memory_usage", "size_t* used_byte", "used_byte"],
    
    ["getrf", "pangulu_inblock_idx nb, pangulu_storage_slot_t* opdst, int tid", "nb, opdst, tid"],
    ["tstrf", "pangulu_inblock_idx nb, pangulu_storage_slot_t* opdst, pangulu_storage_slot_t* opdiag, int tid", "nb, opdst, opdiag, tid"],
    ["gessm", "pangulu_inblock_idx nb, pangulu_storage_slot_t* opdst, pangulu_storage_slot_t* opdiag, int tid", "nb, opdst, opdiag, tid"],
    ["ssssm", "pangulu_inblock_idx nb, pangulu_storage_slot_t* opdst, pangulu_storage_slot_t* op1, pangulu_storage_slot_t* op2, int tid", "nb, opdst, op1, op2, tid"],
    ["ssssm_batched", "pangulu_inblock_idx nb, pangulu_uint64_t ntask, pangulu_task_t* tasks", "nb, ntask, tasks"],
    ["hybrid_batched", "pangulu_inblock_idx nb, pangulu_uint64_t ntask, pangulu_task_t* tasks", "nb, ntask, tasks"],

    ["spmv",  "pangulu_inblock_idx nb, pangulu_storage_slot_t* a, calculate_type* x, calculate_type* y", "nb, a, x, y"],
    ["vecadd","pangulu_int64_t length, calculate_type *bval, calculate_type *xval", "length, bval, xval"],
    ["sptrsv","pangulu_inblock_idx nb, pangulu_storage_slot_t *s, calculate_type* xval, pangulu_int64_t uplo", "nb, s, xval, uplo"],
]

def generate_platform_names(build_list_path, platform_list_path):
    build_name_list = []
    with open(build_list_path, "r") as f:
        build_reader = csv.reader(f)
        for build_item in build_reader:
            if len(build_item) < 1:
                continue
            build_name_list.append(build_item[0])
            
    platform_list = []
    with open(platform_list_path, "r") as f:
        platform_reader = csv.reader(f)
        for platform_item in platform_reader:
            platform_list.append(platform_item)
    
    build_name_list_ret = []
    for name in build_name_list:
        for platform in platform_list:
            if len(platform) < 2:
                continue
            if platform[1] == name:
                build_name_list_ret.append(platform)
                break
    return build_name_list_ret


def generate_platform_paths(build_platform_names, platform_list_path):
    platform_paths = []
    for platform in build_platform_names:
        platform_id = platform[0]
        assert(len(platform_id) == 7)
        platform_id_l1 = platform_id[0:2]
        platform_id_l2 = platform_id[2:4]
        platform_id_l3 = platform_id[4:7]
        dir_l1 = None
        dir_l2 = None
        dir_l3 = None
        dirs_l1 = [file for file in os.listdir(os.path.dirname(platform_list_path))]
        for current_dir_l1 in dirs_l1:
            if current_dir_l1[:2] == platform_id_l1:
                dir_l1 = current_dir_l1
                break
        dirs_l2 = [file for file in os.listdir(os.path.join(os.path.dirname(platform_list_path), dir_l1))]
        for current_dir_l2 in dirs_l2:
            if current_dir_l2[:2] == platform_id_l2:
                dir_l2 = current_dir_l2
                break
        dirs_l3 = [file for file in os.listdir(os.path.join(os.path.dirname(platform_list_path), dir_l1, dir_l2))]
        for current_dir_l3 in dirs_l3:
            if current_dir_l3[:3] == platform_id_l3:
                dir_l3 = current_dir_l3
                break
        platform_paths.append([platform_id, f"platforms/{dir_l1}/{dir_l2}/{dir_l3}"])
    return platform_paths


def generate_platform_function(platforms, function_info):
    function_lastname, formal_params, real_params = function_info
    ret = ""
    if len(formal_params) == 0:
        ret += f"void pangulu_platform_{function_lastname}(pangulu_platform_t platform)" + "{"
    else:
        ret += f"void pangulu_platform_{function_lastname}({formal_params}, pangulu_platform_t platform)" + "{"
    ret += "switch(platform){"
    for platform in platforms:
        ret += f'case PANGULU_PLATFORM_{platform[1]} : pangulu_platform_{platform[0]}_{function_lastname}({real_params}); break;'
    ret += f'default: printf("No platform implementation for function pangulu_platform_{platform[0]}_{function_lastname}.\\n"); break;'
    ret += "}}"
    return ret


def generate_platform_helper_c(dst_dir, build_list_path, platform_list_path):
    dont_modify_warning = '''// Warning : Don't modify this file directly.
// This file is automatically generated by build_helper.py.
// This file will be regenerated after the next compilation.
// All changes will be lost.
'''
    h_file_str = ""
    h_file_str += dont_modify_warning
    h_file_str += """#include "pangulu_common.h"
#include "./platforms/pangulu_platform_common.h"
"""
    build_platform_names = generate_platform_names(build_list_path, platform_list_path)
        
    for function_info in platform_functions_info:
        h_file_str += generate_platform_function(build_platform_names, function_info)
        h_file_str += "\n"
    with open(os.path.join(dst_dir, "pangulu_platform_helper.c"), "w") as f:
        f.write(h_file_str)


def generate_platform_helper_h(dst_dir, build_list_path, platform_list_path):
    dont_modify_warning = '''// Warning : Don't modify this file directly.
// This file is automatically generated by build_helper.py.
// This file will be regenerated after the next compilation.
// All changes will be lost.
'''
    h_file_str = ""
    h_file_str += dont_modify_warning
    h_file_str += """#ifndef PANGULU_PLATFORM_HELPER
#define PANGULU_PLATFORM_HELPER
"""
    build_platform_names = generate_platform_names(build_list_path, platform_list_path)
    
    build_platform_paths = generate_platform_paths(build_platform_names, platform_list_path)
    for platform_path in build_platform_paths:
        h_file_str += f'#include "{os.path.join(platform_path[1].split("platforms/")[-1], f"pangulu_platform_{platform_path[0]}.h")}"\n'

    h_file_str += "typedef unsigned long long pangulu_platform_t;\n"
    
    for platform in build_platform_names:
        h_file_str += f"#define PANGULU_PLATFORM_{platform[1]} 0x{platform[0]}\n"

    for function_info in platform_functions_info:
        h_file_str += generate_platform_function(build_platform_names, function_info).split("{")[0]+";"
        h_file_str += "\n"

    h_file_str += "#endif\n"
    with open(os.path.join(dst_dir, "pangulu_platform_common.h"), "w") as f:
        f.write(h_file_str)


def compile_platform_code(build_list_path, platform_list_path):
    build_platform_names = generate_platform_names(build_list_path, platform_list_path)
    build_platform_paths = generate_platform_paths(build_platform_names, platform_list_path)
    for build_platform_path in build_platform_paths:
        command = f"make -C src/{build_platform_path[1]}"
        print(command)
        return_code = subprocess.call(command.split())
        if return_code != 0:
            exit(return_code)

def generate_build_info_h(dst_dir):
    dont_modify_warning = '''// Warning : Don't modify this file directly.
// This file is automatically generated by build_helper.py.
// This file will be regenerated after the next compilation.
// All changes will be lost.
'''
    h_file_str = ""
    h_file_str += dont_modify_warning
    h_file_str += """#ifndef PANGULU_BUILD_INFO
#define PANGULU_BUILD_INFO
"""

    h_file_str += F'''const char* pangulu_build_time = "{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}";\n'''
    
    h_file_str += "#endif"
    with open(os.path.join(dst_dir, "pangulu_build_info.h"), "w") as f:
        f.write(h_file_str)

    
if __name__ == "__main__":
    if sys.argv[1] == "generate_platform_helper":
        # generate_build_info_h("src/")
        generate_platform_helper_h("src/platforms/", "build_list.csv", "src/platforms/platform_list.csv")
        generate_platform_helper_c("src/", "build_list.csv", "src/platforms/platform_list.csv")
    elif sys.argv[1] == "compile_platform_code":
        compile_platform_code("build_list.csv", "src/platforms/platform_list.csv")
    else:
        print("[BUILD_HELPER_ERROR] Unknown command.")
        exit(1)