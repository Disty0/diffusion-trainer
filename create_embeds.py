import os
import gc
import glob
import time
import torch
import atexit
import argparse
from tqdm import tqdm
from utils import loader_utils, embed_utils

print_filler = "--------------------------------------------------"


def get_paths(dataset_path, out_path, model_type, text_ext):
    print(print_filler)
    print(f"Discovering {text_ext} files")
    file_list = glob.glob(f"{dataset_path}/**/*{text_ext}", recursive=True)
    print(f"Found {len(file_list)} {text_ext} files")
    paths = []
    texts = []
    for text_file in file_list:
        embed_path = os.path.splitext(text_file[len(dataset_path)+1:])[0] + "_" + model_type + "_embed.pt"
        embed_path = os.path.join(out_path, embed_path)
        if not os.path.exists(embed_path) or os.path.getsize(embed_path) == 0:
            paths.append(embed_path)
            with open(text_file, "r") as file:
                text = file.read()
            if text[-1] == "\n":
                text = text[:-1]
            texts.append(text)
    print(f"Found {len(paths)} {text_ext} files to encode")
    return texts, paths


def get_batches(batch_size, dataset_path, out_path, model_type, text_ext):
    texts, paths = get_paths(dataset_path, out_path, model_type, text_ext)
    embed_pathes = []
    embed_path = []
    text_batches = []
    text_batch = []
    for i in range(len(paths)):
        embed_path.append(paths[i])
        text_batch.append(texts[i])
        if len(embed_path) >= batch_size:
            embed_pathes.append(embed_path)
            text_batches.append(text_batch)
            embed_path = []
            text_batch = []
    if len(embed_path) != 0:
        embed_pathes.append(embed_path)
        text_batches.append(text_batch)
    return text_batches, embed_pathes


def write_embeds(embed_encoder, device, model_type, cache_backend, text_batch, embed_path):
    embeds = embed_utils.encode_embeds(embed_encoder, device, model_type, text_batch)
    getattr(torch, device.type).synchronize(device)
    for i in range(len(text_batch)):
        cache_backend.save(embeds[i], embed_path[i])


if __name__ == '__main__':
    print("\n" + print_filler)
    parser = argparse.ArgumentParser(description='Create embed cache')
    parser.add_argument('model_path', type=str)
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('out_path', type=str)
    parser.add_argument('--model_type', default="sd3", type=str)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--dtype', default="bfloat16", type=str)
    parser.add_argument('--dynamo_backend', default="inductor", type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--save_queue_lenght', default=4096, type=int)
    parser.add_argument('--max_save_workers', default=12, type=int)
    parser.add_argument('--gc_steps', default=2048, type=int)
    parser.add_argument('--text_ext', default=".txt", type=str)
    parser.add_argument('--disable_tunableop', default=False, action='store_true')
    args = parser.parse_args()

    if args.dataset_path[-1] == "/":
        args.dataset_path = args.dataset_path[:-1]

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

    if not args.disable_tunableop:
        torch.cuda.tunable.enable(val=True)
    try:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
    except Exception:
        pass

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    print(f"Loading embed encoder models with dtype {dtype} to device {device}")
    print(print_filler)
    embed_encoder = embed_utils.get_embed_encoder(args.model_type, args.model_path, device, dtype, args.dynamo_backend)

    cache_backend = loader_utils.SaveBackend(args.model_type, save_queue_lenght=args.save_queue_lenght, max_save_workers=args.max_save_workers)
    text_batches, embed_pathes = get_batches(args.batch_size, args.dataset_path, args.out_path, args.model_type, args.text_ext)
    epoch_len = len(text_batches)

    def exit_handler(cache_backend):
        while not cache_backend.save_queue.empty():
            print(f"Waiting for the remaining writes: {cache_backend.save_queue.qsize()}")
            time.sleep(1)
        cache_backend.keep_saving = False
        cache_backend.save_thread.shutdown(wait=True)
        del cache_backend
    atexit.register(exit_handler, cache_backend)

    print(f"Starting to encode {epoch_len} batches with batch size {args.batch_size}")
    for steps_done in tqdm(range(epoch_len)):
        try:
            write_embeds(embed_encoder, device, args.model_type, cache_backend, text_batches.pop(0), embed_pathes.pop(0))
            if steps_done % args.gc_steps == 0:
                gc.collect()
                getattr(torch, device.type).synchronize(device)
                getattr(torch, device.type).empty_cache()
        except Exception as e:
            print("ERROR: ", embed_pathes[0], " : ", str(e))
            break # break so torch can save the new tunableops table
    atexit.unregister(exit_handler)
    exit_handler(cache_backend)