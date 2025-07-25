import os
import gc
import json
import time
import torch

if not torch.version.cuda:
    import transformers
    transformers.utils.is_flash_attn_2_available = lambda: False

import atexit
import argparse
from tqdm import tqdm
from PIL import Image

from typing import Dict, List, Tuple
from transformers import ImageProcessingMixin
from diffusers.models.modeling_utils import ModelMixin

from utils import loader_utils, latent_utils

print_filler = "--------------------------------------------------"


def get_bucket_list(model_type: str, dataset_path: str, out_path: str) -> Dict[str, List[str]]:
    print(print_filler)
    print("Creating bucket list")
    total_image_count = 0
    images_to_encode = 0
    new_bucket_list = {}
    bucket_list = {}
    with open(os.path.join(dataset_path, "bucket_list.json"), "r") as f:
        bucket = json.load(f)
    for key in bucket.keys():
        if key not in bucket_list.keys():
            bucket_list[key] = []
            new_bucket_list[key] = []
        for i in range(len(bucket[key])):
            latent_path = os.path.splitext(bucket[key][i])[0] + "_" + model_type + "_latent.pt"
            latent_path_full = os.path.join(out_path, latent_path)
            if not os.path.exists(latent_path_full) or os.path.getsize(latent_path_full) == 0:
                bucket_list[key].append(os.path.join(dataset_path, bucket[key][i]))
                images_to_encode = images_to_encode + 1
            new_bucket_list[key].append(latent_path)
            total_image_count = total_image_count + 1
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path, "bucket_list.json"), "w") as f:
        json.dump(new_bucket_list, f)
    print(f"Found {total_image_count} images")
    print(f"Found {images_to_encode} images to encode")
    print(print_filler)
    return bucket_list


def get_batches(batch_size: int, model_type: str, dataset_path: str, out_path: str) -> List[Tuple[List[str], str]]:
    bucket_list = get_bucket_list(model_type, dataset_path, out_path)
    epoch_batch = []
    for key, bucket in bucket_list.items():
        bucket_len = len(bucket)
        if bucket_len > 0:
            if bucket_len > batch_size:
                images_left_out = bucket_len % batch_size
                for i in range(int((bucket_len - images_left_out) / batch_size)):
                    epoch_batch.append((bucket[i*batch_size:(i+1)*batch_size], key))
                if images_left_out > 0:
                    epoch_batch.append((bucket[-images_left_out:], key))
            else:
                epoch_batch.append((bucket, key))
        print(f"Images to encode in the bucket {key}: {bucket_len}")
    return epoch_batch


def write_latents(
    latent_model: ModelMixin,
    image_processor: ImageProcessingMixin,
    device: torch.device,
    args: argparse.Namespace,
    cache_backend: loader_utils.SaveBackend,
    save_image_backend: loader_utils.SaveImageBackend,
    batch: List[Tuple[Image.Image, str]],
) -> None:
    images = []
    latent_paths = []
    save_image_paths = []
    for item in batch:
        latent_path = os.path.splitext(item[1][len(args.dataset_path)+1:])[0] + "_" + args.model_type + "_latent.pt"
        latent_path = os.path.join(args.out_path, latent_path)
        latent_paths.append(latent_path)
        if args.save_images:
            save_image_path = os.path.splitext(item[1][len(args.dataset_path)+1:])[0] + "_" + args.model_type + "_image" + args.save_images_ext
            save_image_path = os.path.join(args.save_images_path, save_image_path)
            save_image_paths.append(save_image_path)
        images.append(item[0])
    with torch.no_grad():
        latents = latent_utils.encode_latents(latent_model, image_processor, images, args.model_type, device)
    getattr(torch, device.type).synchronize(device)
    for i in range(len(latent_paths)):
        cache_backend.save(latents[i], latent_paths[i])
        if args.save_images:
            save_image_backend.save(images[i], save_image_paths[i])


@torch.no_grad()
def main():
    print("\n" + print_filler)
    parser = argparse.ArgumentParser(description='Create latent cache')

    parser.add_argument('model_path', type=str)
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('out_path', type=str)
    parser.add_argument('--model_type', default="sd3", type=str)

    parser.add_argument('--device', default="auto", type=str)
    parser.add_argument('--dtype', default="float16", type=str)
    parser.add_argument('--save_dtype', default="float16", type=str)
    parser.add_argument('--batch_size', default=4, type=int)

    parser.add_argument('--gc_steps', default=2048, type=int)
    parser.add_argument('--dynamo_backend', default="no", type=str)
    parser.add_argument('--tunableop', default=False, action='store_true')

    parser.add_argument('--save_images', default=False, action='store_true')
    parser.add_argument('--save_images_path', default="cropped_images", type=str)
    parser.add_argument('--save_images_ext', default=".jxl", type=str)
    parser.add_argument('--save_images_lossless', default=True, action='store_true')
    parser.add_argument('--save_images_quality', default=100, type=int)

    parser.add_argument('--load_queue_lenght', default=32, type=int)
    parser.add_argument('--save_queue_lenght', default=4096, type=int)
    parser.add_argument('--save_image_queue_lenght', default=4096, type=int)
    parser.add_argument('--max_load_workers', default=4, type=int)
    parser.add_argument('--max_save_workers', default=8, type=int)
    parser.add_argument('--max_save_image_workers', default=4, type=int)

    args = parser.parse_args()

    if args.dataset_path[-1] == "/":
        args.dataset_path = args.dataset_path[:-1]
    if args.save_images_path[-1] == "/":
        args.save_images_path = args.save_images_path[:-1]

    if torch.version.hip:
        try:
            # don't use this for training models, only for inference with latent encoder and embed encoder
            # https://github.com/huggingface/diffusers/discussions/7172
            from functools import wraps
            from flash_attn import flash_attn_func
            backup_sdpa = torch.nn.functional.scaled_dot_product_attention
            @wraps(torch.nn.functional.scaled_dot_product_attention)
            def sdpa_hijack(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
                if query.shape[-1] <= 128 and attn_mask is None and query.dtype != torch.float32:
                    return flash_attn_func(q=query.transpose(1, 2), k=key.transpose(1, 2), v=value.transpose(1, 2), dropout_p=dropout_p, causal=is_causal, softmax_scale=scale).transpose(1, 2)
                else:
                    return backup_sdpa(query=query, key=key, value=value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
            torch.nn.functional.scaled_dot_product_attention = sdpa_hijack
        except Exception as e:
            print(f"Failed to enable Flash Atten for ROCm: {e}")

    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)

    if args.tunableop:
        torch.cuda.tunable.enable(val=True)

    dtype = getattr(torch, args.dtype)
    save_dtype = getattr(torch, args.save_dtype)
    if args.device == "auto":
        device = torch.device("xpu" if hasattr(torch,"xpu") and torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Loading latent encoder models with dtype {dtype} to device {device}")
    print(print_filler)
    latent_model, image_processor = latent_utils.get_latent_model(args.model_type, args.model_path, device, dtype, args.dynamo_backend)

    epoch_batches = get_batches(args.batch_size, args.model_type, args.dataset_path, args.out_path)
    epoch_len = len(epoch_batches)
    cache_backend = loader_utils.SaveBackend(model_type=args.model_type, save_dtype=save_dtype, save_queue_lenght=args.save_queue_lenght, max_save_workers=args.max_save_workers)
    image_backend = loader_utils.ImageBackend(epoch_batches, load_queue_lenght=args.load_queue_lenght, max_load_workers=args.max_load_workers)
    if args.save_images:
        save_image_backend = loader_utils.SaveImageBackend(save_queue_lenght=args.save_image_queue_lenght, max_save_workers=args.max_save_image_workers, lossless=args.save_images_lossless, quality=args.save_images_quality)
    else:
        save_image_backend = None

    def exit_handler(image_backend, cache_backend, save_image_backend):
        image_backend.keep_loading = False
        image_backend.load_thread.shutdown(wait=True)
        del image_backend

        while not cache_backend.save_queue.empty():
            print(f"Waiting for the remaining writes: {cache_backend.save_queue.qsize()}")
            time.sleep(1)
        cache_backend.keep_saving = False
        cache_backend.save_thread.shutdown(wait=True)
        del cache_backend

        if save_image_backend is not None:
            while not save_image_backend.save_queue.empty():
                print(f"Waiting for the remaining image writes: {save_image_backend.save_queue.qsize()}")
                time.sleep(1)
            save_image_backend.keep_saving = False
            save_image_backend.save_thread.shutdown(wait=True)
            del save_image_backend

    atexit.register(exit_handler, image_backend, cache_backend, save_image_backend)

    print(print_filler)
    print(f"Starting to encode {epoch_len} batches with batch size {args.batch_size}")
    for steps_done in tqdm(range(epoch_len)):
        try:
            batch = image_backend.get_images()
            write_latents(latent_model, image_processor, device, args, cache_backend, save_image_backend, batch)
            if steps_done % args.gc_steps == 0:
                gc.collect()
                getattr(torch, device.type).synchronize(device)
                getattr(torch, device.type).empty_cache()
        except Exception as e:
            print("ERROR: ", str(e))
            break # break so torch can save the new tunableops table

    atexit.unregister(exit_handler)
    exit_handler(image_backend, cache_backend, save_image_backend)


if __name__ == '__main__':
    main()
