#!/usr/bin/python3
import csv
import os
import sys
import subprocess

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


def compile_platform_code(build_list_path, platform_list_path):
    build_platform_names = generate_platform_names(build_list_path, platform_list_path)
    build_platform_paths = generate_platform_paths(build_platform_names, platform_list_path)
    for build_platform_path in build_platform_paths:
        command = f"make -C src/{build_platform_path[1]}"
        print(command)
        return_code = subprocess.call(command.split())
        if return_code != 0:
            exit(return_code)


if __name__ == "__main__":
    if sys.argv[1] == "compile_platform_code":
        compile_platform_code("build_list.csv", "src/platforms/platform_list.csv")
    else:
        print("[BUILD_HELPER_ERROR] Unknown command.")
        exit(1)