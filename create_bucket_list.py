import os
import math
import glob
import json
import argparse
from tqdm import tqdm


def calc_crop_res(orig_res, target_size, res_steps):
    orig_size = orig_res[0] * orig_res[1]
    if orig_size > target_size:
        scale = math.sqrt(orig_size / target_size)
        new_res = [orig_res[0]/scale, orig_res[1]/scale]
    else:
        new_res = orig_res

    diff = new_res[0] % res_steps
    if diff != 0:
        new_res[0] = new_res[0] - diff
        if diff > res_steps / 2 and orig_res[0] >= new_res[0] + res_steps:
            new_res[0] = new_res[0] + res_steps

    diff = new_res[1] % res_steps
    if diff != 0:
        new_res[1] = new_res[1] - diff
        if diff > res_steps / 2 and orig_res[1] >= new_res[1] + res_steps:
            new_res[1] = new_res[1] + res_steps

    new_res[0] = int(new_res[0])
    new_res[1] = int(new_res[1])
    return new_res


def write_bucket_list(dataset_path, target_size, res_steps, image_ext, size_function):
    res_map = {}
    file_list = glob.glob(f"{dataset_path}/**/*{image_ext}", recursive=True)
    for image in tqdm(file_list):
        width, height = size_function(image)
        new_res = calc_crop_res([width, height], target_size, res_steps)
        bucket_name = f"{new_res[0]}x{new_res[1]}"
        if bucket_name not in res_map:
            res_map[bucket_name] = []
        res_map[bucket_name].append(image[len(dataset_path)+1:])
    with open(os.path.join(dataset_path, "bucket_list.json"), "w") as f:
        json.dump(res_map, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a bucket list with a given dataset path')
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('--pixel_count', default=1048576, type=int)
    parser.add_argument('--res_steps', default=64, type=int)
    parser.add_argument('--image_ext', default=".jxl", type=str)
    args = parser.parse_args()

    if args.image_ext == ".jxl":
        from utils.jpeg_xl_utils import get_jxl_size as size_function
    else:
        from imagesize import get as size_function

    write_bucket_list(args.dataset_path, args.pixel_count, args.res_steps, args.image_ext, size_function)
