import os
import math
import json
import argparse
import imagesize

from glob import glob
from tqdm import tqdm
from typing import Tuple
from utils.jpeg_xl_utils import get_jxl_size

img_ext_list = ("jpg", "png", "webp", "jpeg", "jxl")


def calc_crop_res(width: int, height: int, target_size: int, res_steps: int) -> Tuple[int, int]:
    orig_size = width * height
    if orig_size > target_size:
        scale = math.sqrt(orig_size / target_size)
        new_width = width/scale
        new_height = height/scale
    else:
        new_width = width
        new_height = height

    diff = new_width % res_steps
    if diff != 0:
        new_width = new_width - diff
        if diff > res_steps / 2 and width >= new_width + res_steps:
            new_width = new_width + res_steps

    diff = new_height % res_steps
    if diff != 0:
        new_height = new_height - diff
        if diff > res_steps / 2 and height >= new_height + res_steps:
            new_height = new_height + res_steps

    return int(new_width), int(new_height)


def write_bucket_list(dataset_path: str, target_size: int, res_steps: int) -> None:
    res_map = {}
    file_list = []
    for ext in img_ext_list:
        file_list.extend(glob(f"{dataset_path}/**/*{ext}"))
    for image_path in tqdm(file_list):
        if os.path.splitext(image_path)[-1] == ".jxl":
            width, height = get_jxl_size(image_path)
        else:
            width, height = imagesize.get(image_path)
        new_width, new_height = calc_crop_res(width, height, target_size, res_steps)
        bucket_name = f"{new_width}x{new_height}"
        if bucket_name not in res_map.keys():
            res_map[bucket_name] = []
        res_map[bucket_name].append(image_path[len(dataset_path)+1:])
    with open(os.path.join(dataset_path, "bucket_list.json"), "w") as f:
        json.dump(res_map, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a bucket list with a given dataset path')
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('--pixel_count', default=1048576, type=int)
    parser.add_argument('--res_steps', default=32, type=int)
    args = parser.parse_args()

    if args.dataset_path[-1] == "/":
        args.dataset_path = args.dataset_path[:-1]

    print(f"Searching for {img_ext_list} files...")
    write_bucket_list(args.dataset_path, args.pixel_count, args.res_steps)
