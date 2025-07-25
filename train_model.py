import os
import gc
import math
import json
import torch

if not torch.version.cuda:
    import transformers
    transformers.utils.is_flash_attn_2_available = lambda: False

import wandb
import random
import shutil
import argparse
from tqdm import tqdm
from utils import loader_utils, train_utils

from accelerate import Accelerator
from torch.utils.data import DataLoader
from diffusers.training_utils import EMAModel

from typing import Dict, List, Tuple

print_filler = "--------------------------------------------------"


def get_bucket_list(batch_size: int, dataset_paths: List[Tuple[str, List[str], int]], empty_embed_path: str, latent_type: str = "latent", embed_type: str = "embed") -> Dict[str, List[str]]:
    if embed_type == "embed":
        embed_suffix = "_" + empty_embed_path.rsplit("empty_", maxsplit=1)[-1]
    else:
        embed_suffix = ".txt"
        empty_embed_path = ""
    print("Creating bucket list")
    bucket_list = {}

    for latent_dataset, embed_datasets, repeat in dataset_paths:
        with open(os.path.join(latent_dataset, "bucket_list.json"), "r") as f:
            bucket = json.load(f)
        for key in bucket.keys():
            if key not in bucket_list.keys():
                bucket_list[key] = []
            for i in range(len(bucket[key])):
                for _ in range(repeat):
                    for embed_dataset in embed_datasets:
                        latent_path = os.path.join(latent_dataset, bucket[key][i])
                        if embed_dataset == "empty_embed":
                            embed_path = empty_embed_path
                        elif latent_type == "latent":
                            embed_path = os.path.join(embed_dataset, bucket[key][i][:-9]+"embed.pt")
                        else:
                            embed_path = os.path.join(embed_dataset, os.path.splitext(bucket[key][i])[0] + embed_suffix)
                        if not os.path.exists(latent_path):
                            print(f"Latent file not found: {bucket[key][i]}")
                        elif not os.path.exists(embed_path):
                            print(f"Embed file not found: {embed_path}")
                        else:
                            bucket_list[key].append((latent_path, embed_path))

    keys_to_remove = []
    total_image_count = 0
    for key in bucket_list.keys():
        if len(bucket_list[key]) < batch_size:
            keys_to_remove.append(key)
        else:
            random.shuffle(bucket_list[key])
            total_image_count = total_image_count + len(bucket_list[key])

    removed_image_count = 0
    for key in keys_to_remove:
        count = len(bucket_list[key])
        print(f"Removing bucket {key} with {count} images")
        bucket_list.pop(key)
        removed_image_count = removed_image_count + count

    print(print_filler)
    print(f"Removed {removed_image_count} images in total")
    print(f"Images left in the dataset: {total_image_count}")
    print(print_filler + "\n")

    return bucket_list


def get_batches(batch_size: int, dataset_paths: List[Tuple[str, List[str], int]], dataset_index: str, empty_embed_path: str, latent_type: str = "latent", embed_type: str = "embed") -> None:
    bucket_list = get_bucket_list(batch_size, dataset_paths, empty_embed_path, latent_type=latent_type, embed_type=embed_type)
    print("Creating epoch batches")
    epoch_batch = []
    images_left_out_count = 0

    for key, bucket in bucket_list.items():
        random.shuffle(bucket)
        bucket_len = len(bucket)
        images_left_out = bucket_len % batch_size
        images_left_out_count= images_left_out_count + images_left_out
        if latent_type == "latent":
            for i in range(int((bucket_len - images_left_out) / batch_size)):
                epoch_batch.append(bucket[i*batch_size:(i+1)*batch_size])
        elif latent_type in {"image", "jpeg"}:
            for i in range(int((bucket_len - images_left_out) / batch_size)):
                epoch_batch.append((bucket[i*batch_size:(i+1)*batch_size], key))
        else:
            raise NotImplementedError(F"Latent type {latent_type} is not implemented")
        print(print_filler)
        print(f"Images left out from bucket {key}: {images_left_out}")
        print(f"Images left in the bucket {key}: {bucket_len - images_left_out}")

    print(print_filler)
    print(f"Images that got left out from the epoch: {images_left_out_count}")
    print(f"Total images left in the epoch: {len(epoch_batch) * batch_size}")
    print(f"Batches * Batch Size: {len(epoch_batch)} * {batch_size}")
    print(print_filler + "\n")

    random.shuffle(epoch_batch)
    os.makedirs(os.path.dirname(dataset_index), exist_ok=True)
    with open(dataset_index, "w") as f:
        json.dump(epoch_batch, f)


def main():
    parser = argparse.ArgumentParser(description='Train a model with a given config')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    if config["tunableop"]:
        torch.cuda.tunable.enable(val=True)
    if config["dynamo_backend"] != "no":
        torch._dynamo.config.cache_size_limit = 64

    if config["allow_tf32"]:
        torch.set_float32_matmul_precision('high')
    else:
        torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = config["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = config["allow_tf32"]

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = config["allow_reduced_precision"]
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = config["allow_reduced_precision"]

    torch.backends.cuda.enable_flash_sdp(config["flash_sdp"])
    torch.backends.cuda.enable_mem_efficient_sdp(config["mem_efficient_sdp"])
    torch.backends.cuda.enable_math_sdp(config["math_sdp"])
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(config["math_sdp_reduction"])
    if config["dynamic_sdp"]:
        from utils.dynamic_sdp import dynamic_scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = dynamic_scaled_dot_product_attention

    os.makedirs(config["project_dir"], exist_ok=True)
    empty_embed_path = os.path.join("empty_embeds", "empty_" + config["model_type"] + "_embed.pt")
    if config["embed_type"] == "token":
        empty_embed = None
    else:
        empty_embed = loader_utils.load_from_file(empty_embed_path)
    first_epoch = 0
    current_epoch = 0
    current_step = 0
    start_step = 0

    accelerator = Accelerator(
        mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        log_with=config["log_with"],
        project_dir=config["project_dir"],
        dynamo_backend=config["dynamo_backend"],
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        return model._orig_mod if isinstance(model, torch._dynamo.eval_frame.OptimizedModule) else model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(unwrap_model(model), train_utils.get_model_class(config["model_type"])):
                    unwrap_model(model).save_pretrained(os.path.join(output_dir, "diffusion_model"))
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")
                weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            model = models.pop()
            if isinstance(unwrap_model(model), train_utils.get_model_class(config["model_type"])):
                load_model = train_utils.get_model_class(config["model_type"]).from_pretrained(input_dir, subfolder="diffusion_model")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
            else:
                raise ValueError(f"Unsupported model found: {type(model)=}")
            del load_model

    def fused_optimizer_hook(parameter):
        if config["log_grad_stats"]:
            global grad_max, grad_mean, grad_mean_count
            param_grad_abs = parameter.grad.abs()
            grad_max = max(grad_max, param_grad_abs.max().item())
            grad_mean += param_grad_abs.mean().item()
            grad_mean_count += 1
        if accelerator.sync_gradients and config["max_grad_clip"] > 0:
            # this is **very** slow with fp16
            accelerator.clip_grad_value_(parameter, config["max_grad_clip"])
            if config["log_grad_stats"]:
                global clipped_grad_mean, clipped_grad_mean_count
                clipped_grad_mean += parameter.grad.abs().mean().item()
                clipped_grad_mean_count += 1
        if accelerator.sync_gradients and config["max_grad_norm"] > 0:
            # this is **very** slow with fp16 and norming per parameter isn't ideal
            global grad_norm, grad_norm_count
            grad_norm += accelerator.clip_grad_norm_(parameter, config["max_grad_norm"])
            grad_norm_count += 1
        optimizer[parameter][0].step()
        optimizer[parameter][1].step()
        optimizer[parameter][0].zero_grad()

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    accelerator.print("\n" + print_filler)
    accelerator.print("Initializing the trainer")
    accelerator.print(print_filler)

    batch_size = config["batch_size"]
    if accelerator.is_local_main_process and not os.path.exists(config["dataset_index"]):
        get_batches(batch_size, config["dataset_paths"], config["dataset_index"], empty_embed_path, latent_type=config["latent_type"], embed_type=config["embed_type"])
    accelerator.wait_for_everyone()
    with open(config["dataset_index"], "r") as f:
        epoch_batch = json.load(f)

    dtype = getattr(torch, config["weights_dtype"])
    print(f"Loading diffusion models with dtype {dtype} to device {accelerator.device}")
    accelerator.print(print_filler)
    model, model_processor = train_utils.get_diffusion_model(config, accelerator.device, dtype)
    if config["gradient_checkpointing"]:
        model.enable_gradient_checkpointing()
    model = accelerator.prepare(model)

    optimizer, lr_scheduler = train_utils.get_optimizer_and_lr_scheduler(config, model, accelerator, fused_optimizer_hook)

    if config["latent_type"] == "latent":
        dataset = loader_utils.LatentsAndEmbedsDataset(epoch_batch)
    elif config["latent_type"] == "image":
        dataset = loader_utils.ImageTensorsAndEmbedsDataset(epoch_batch)
    elif config["latent_type"] == "jpeg":
        if config["encode_dcts_with_cpu"]:
            if config["embed_type"] == "token":
                from transformers import AutoTokenizer
                dataset = loader_utils.DCTsAndTokensDataset(epoch_batch, image_encoder=model_processor, tokenizer=AutoTokenizer.from_pretrained(config["model_path"], subfolder="tokenizer"))
            else:
                dataset = loader_utils.DCTsAndEmbedsDataset(epoch_batch, image_encoder=model_processor)
        else:
            if config["embed_type"] == "token":
                from transformers import AutoTokenizer
                dataset = loader_utils.ImagesAndTokensDataset(epoch_batch, tokenizer=AutoTokenizer.from_pretrained(config["model_path"], subfolder="tokenizer"))
            else:
                dataset = loader_utils.ImagesAndEmbedsDataset(epoch_batch)
    else:
        raise NotImplementedError(F'Latent type {config["latent_type"]} is not implemented')
    train_dataloader = DataLoader(dataset=dataset, batch_size=None, batch_sampler=None, shuffle=False, pin_memory=True, num_workers=config["max_load_workers"], prefetch_factor=int(config["load_queue_lenght"]/config["max_load_workers"]))
    train_dataloader = accelerator.prepare(train_dataloader)

    resume_checkpoint = None
    if config.get("resume_from", "") and config["resume_from"] != "none":
        if config["resume_from"] == "latest":
            checkpoints = os.listdir(config["project_dir"])
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint-")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            if len(checkpoints) > 0:
                resume_checkpoint = checkpoints[-1]
        else:
            resume_checkpoint = config['resume_from']
        if resume_checkpoint is None:
            accelerator.print("No checkpoint found, starting a fresh training run")
        else:
            accelerator.print(f"Resuming from: {resume_checkpoint}")
            accelerator.load_state(os.path.join(config["project_dir"], resume_checkpoint))
            current_step = int(resume_checkpoint.split("-")[1])
            first_epoch = current_step // math.ceil(len(train_dataloader) / config["gradient_accumulation_steps"])
            current_epoch = first_epoch
            start_step = current_step

    if config["ema_update_steps"] > 0 and accelerator.is_main_process:
        ema_dtype = getattr(torch, config["ema_weights_dtype"])
        accelerator.print(print_filler)
        accelerator.print(f'Loading EMA models with dtype {ema_dtype} to device {"cpu" if config["update_ema_on_cpu"] or config["offload_ema_to_cpu"] else accelerator.device}')
        if resume_checkpoint is not None:
            accelerator.print(f"Resuming EMA from: {resume_checkpoint}")
            accelerator.print(print_filler)
            ema_model = EMAModel.from_pretrained(os.path.join(config["project_dir"], resume_checkpoint, "diffusion_ema_model"), train_utils.get_model_class(config["model_type"]), foreach=config["use_foreach_ema"])
            ema_model.to("cpu" if config["update_ema_on_cpu"] or config["offload_ema_to_cpu"] else accelerator.device, dtype=ema_dtype)
        else:
            accelerator.print(print_filler)
            ema_model, _ = train_utils.get_diffusion_model(config, "cpu" if config["update_ema_on_cpu"] or config["offload_ema_to_cpu"] else accelerator.device, ema_dtype)
            ema_model = EMAModel(ema_model.parameters(), model_cls=train_utils.get_model_class(config["model_type"]), model_config=ema_model.config, foreach=config["use_foreach_ema"], decay=config["ema_decay"])
        if config["offload_ema_pin_memory"]:
            ema_model.pin_memory()
        accelerator.print(print_filler)

    accelerator.init_trackers(project_name=config["project_name"], config=config)

    progress_bar = tqdm(
        range(0, math.ceil(len(train_dataloader) / config["gradient_accumulation_steps"]) * config["epochs"]),
        initial=current_step,
        disable=not accelerator.is_local_main_process,
    )

    total_empty_embeds_count = 0
    total_nan_embeds_count = 0
    total_self_correct_count = 0
    total_masked_count = 0
    timesteps_list = []
    grad_norm = torch.tensor(0.0, dtype=dtype, device=accelerator.device)
    grad_mean = 0
    clipped_grad_mean = 0
    grad_norm_count = 0
    skip_grad_norm_count = 0
    grad_mean_count = 0
    clipped_grad_mean_count = 0
    grad_max = 0
    loss = torch.tensor(1.0, dtype=dtype, device=accelerator.device)
    loss_func = train_utils.get_loss_func(config)
    model.train()
    getattr(torch, accelerator.device.type).empty_cache()
    for _ in range(first_epoch, config["epochs"]):
        for epoch_step, (latents_list, embeds_list) in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                last_loss = loss
                loss, model_pred, target, log_dict = train_utils.run_model(model, model_processor, config, accelerator, latents_list, embeds_list, empty_embed, loss_func)
                accelerator.backward(loss)
                if not config["fused_optimizer"]:
                    if accelerator.sync_gradients:
                        if config["log_grad_stats"]:
                            for parameter in model.parameters():
                                if hasattr(parameter, "grad") and parameter.grad is not None:
                                    param_grad_abs = parameter.grad.abs()
                                    grad_max = max(grad_max, param_grad_abs.max().item())
                                    grad_mean += param_grad_abs.mean().item()
                                    grad_mean_count += 1
                        if config["max_grad_clip"] > 0:
                            accelerator.clip_grad_value_(model.parameters(), config["max_grad_clip"])
                            if config["log_grad_stats"]:
                                for parameter in model.parameters():
                                    if hasattr(parameter, "grad") and parameter.grad is not None:
                                        clipped_grad_mean += parameter.grad.abs().mean().item()
                                        clipped_grad_mean_count += 1
                        if config["max_grad_norm"] > 0:
                            grad_norm += accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                            grad_norm_count += 1
                            if grad_norm.isnan() or (config["skip_grad_norm"] > 0 and current_step > config["skip_grad_norm_steps"] and (grad_norm / grad_norm_count) > config["skip_grad_norm"]):
                                loss = last_loss
                                skip_grad_norm_count += 1
                                optimizer.zero_grad(set_to_none=True)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if log_dict.get("timesteps", None) is not None:
                    timesteps_list.extend(log_dict["timesteps"].to("cpu", dtype=torch.float32).detach().tolist())
                if log_dict.get("empty_embeds_count", None) is not None:
                    total_empty_embeds_count += log_dict["empty_embeds_count"]
                if log_dict.get("nan_embeds_count", None) is not None:
                    total_nan_embeds_count += log_dict["nan_embeds_count"]
                if log_dict.get("self_correct_count", None) is not None:
                    total_self_correct_count += log_dict["self_correct_count"]
                if log_dict.get("masked_count", None) is not None:
                    total_masked_count += log_dict["masked_count"]

                if accelerator.sync_gradients:
                    if config["ema_update_steps"] > 0 and current_step % config["ema_update_steps"] == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            if config["update_ema_on_cpu"]:
                                gc.collect()
                                model.to(device="cpu", non_blocking=False)
                            elif config["offload_ema_to_cpu"]:
                                ema_model.to(device=accelerator.device, non_blocking=config["offload_ema_non_blocking"])
                            ema_model.step(model.parameters())
                            if config["update_ema_on_cpu"]:
                                model.to(device=accelerator.device, non_blocking=False)
                                gc.collect()
                            elif config["offload_ema_to_cpu"]:
                                ema_model.to(device="cpu", non_blocking=config["offload_ema_non_blocking"])
                        accelerator.wait_for_everyone()
                    progress_bar.update(1)
                    current_step = current_step + 1

                    if current_step % config["checkpoint_save_steps"] == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            accelerator.print("\n" + print_filler)
                            if config["checkpoints_limit"] != 0:
                                checkpoints = os.listdir(config["project_dir"])
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint-")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                                if len(checkpoints) >= config["checkpoints_limit"]:
                                    num_to_remove = len(checkpoints) - config["checkpoints_limit"] + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]
                                    accelerator.print(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                                    accelerator.print(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(config["project_dir"], removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(config["project_dir"], f"checkpoint-{current_step}")
                            accelerator.print(f"Saving state to {save_path}")
                            accelerator.save_state(save_path)
                            if config["ema_update_steps"] > 0:
                                gc.collect()
                                accelerator.print(f"Saving EMA state to {save_path}")
                                save_ema_model, _ = train_utils.get_diffusion_model(config, "cpu", ema_dtype)
                                save_ema_model_state_dict = ema_model.state_dict()
                                save_ema_model_state_dict.pop("shadow_params", None)
                                save_ema_model.register_to_config(**save_ema_model_state_dict)
                                ema_model.copy_to(save_ema_model.parameters())
                                save_ema_model.save_pretrained(os.path.join(save_path, "diffusion_ema_model"))
                                del save_ema_model
                            gc.collect()
                            accelerator.print(f"\nSaved states to {save_path}")
                            accelerator.print(print_filler)
                        accelerator.wait_for_everyone()

                    logs = {"loss": loss.detach().item(), "epoch": current_epoch}
                    if config["fused_optimizer"]:
                        last_lr = optimizer[list(optimizer.keys())[0]][1].get_last_lr()
                    else:
                        last_lr = lr_scheduler.get_last_lr()
                    logs["lr"] = last_lr[0]
                    if isinstance(logs["lr"], torch.Tensor):
                        logs["lr"] = logs["lr"].item()
                    if len(last_lr) > 1:
                        logs["lr_2"] = last_lr[1]
                        if isinstance(logs["lr_2"], torch.Tensor):
                            logs["lr_2"] = logs["lr_2"].item()

                    if config["log_grad_stats"]:
                        logs["grad_max"] = grad_max
                        grad_max = 0
                        if grad_mean_count > 0:
                            logs["grad_mean"] = grad_mean / grad_mean_count
                            grad_mean = 0
                            grad_mean_count = 0
                        if clipped_grad_mean_count > 0:
                            logs["clipped_grad_mean"] = clipped_grad_mean / clipped_grad_mean_count
                            clipped_grad_mean = 0
                            clipped_grad_mean_count = 0
                    if grad_norm_count > 0:
                        logs["grad_norm"] = (grad_norm / grad_norm_count).item()
                        grad_norm = torch.tensor(0.0, dtype=dtype, device=accelerator.device)
                        grad_norm_count = 0
                    if skip_grad_norm_count > 0:
                        logs["skip_grad_norm_count"] = skip_grad_norm_count
                    if accelerator.is_main_process:
                        if config["ema_update_steps"] > 0:
                            logs["ema_decay"] = ema_model.get_decay(ema_model.optimization_step)
                    if log_dict.get("seq_len", None) is not None:
                        logs["seq_len"] = log_dict["seq_len"]

                    progress_bar.set_postfix(**logs)
                    if config["dropout_rate"] > 0:
                        logs["total_empty_embeds_count"] = total_empty_embeds_count
                    if total_nan_embeds_count > 0:
                        logs["total_nan_embeds_count"] = total_nan_embeds_count
                    if config["self_correct_rate"] > 0:
                        logs["total_self_correct_count"] = total_self_correct_count
                    if config["mask_rate"] > 0:
                        logs["total_masked_count"] = total_masked_count
                    if timesteps_list:
                        logs["timesteps"] = wandb.Histogram(timesteps_list)
                        logs["timesteps_min"] = min(timesteps_list)
                        logs["timesteps_max"] = max(timesteps_list)
                        timesteps_list = []
                    accelerator.log(logs, step=current_step)

                    if current_step == start_step + 1 or (config["gc_steps"] != 0 and current_step % config["gc_steps"] == 0):
                        gc.collect()
                        getattr(torch, accelerator.device.type).empty_cache()

        current_epoch = current_epoch + 1
        accelerator.print("\n" + print_filler)
        accelerator.print(f"Starting epoch {current_epoch}")
        accelerator.print(f"Current steps done: {current_step}")
        if config["reshuffle"]:
            del dataset, train_dataloader
            if accelerator.is_local_main_process:
                os.rename(config["dataset_index"], config["dataset_index"]+"-epoch_"+str(current_epoch-1)+".json")
                get_batches(batch_size, config["dataset_paths"], config["dataset_index"], empty_embed_path, latent_type=config["latent_type"], embed_type=config["embed_type"])
            accelerator.wait_for_everyone()
            with open(config["dataset_index"], "r") as f:
                epoch_batch = json.load(f)
            if config["latent_type"] == "latent":
                dataset = loader_utils.LatentsAndEmbedsDataset(epoch_batch)
            elif config["latent_type"] == "image":
                dataset = loader_utils.ImageTensorsAndEmbedsDataset(epoch_batch)
            elif config["latent_type"] == "jpeg":
                if config["encode_dcts_with_cpu"]:
                    if config["embed_type"] == "token":
                        from transformers import AutoTokenizer
                        dataset = loader_utils.DCTsAndTokensDataset(epoch_batch, image_encoder=model_processor, tokenizer=AutoTokenizer.from_pretrained(config["model_path"], subfolder="tokenizer"))
                    else:
                        dataset = loader_utils.DCTsAndEmbedsDataset(epoch_batch, image_encoder=model_processor)
                else:
                    if config["embed_type"] == "token":
                        from transformers import AutoTokenizer
                        dataset = loader_utils.ImagesAndTokensDataset(epoch_batch, tokenizer=AutoTokenizer.from_pretrained(config["model_path"], subfolder="tokenizer"))
                    else:
                        dataset = loader_utils.ImagesAndEmbedsDataset(epoch_batch)
            else:
                raise NotImplementedError(F'Latent type {config["latent_type"]} is not implemented')
            train_dataloader = DataLoader(dataset=dataset, batch_size=None, batch_sampler=None, shuffle=False, pin_memory=True, num_workers=config["max_load_workers"], prefetch_factor=int(config["load_queue_lenght"]/config["max_load_workers"]))
            train_dataloader = accelerator.prepare(train_dataloader)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = unwrap_model(model)
        save_path = os.path.join(config["project_dir"], "checkpoint-final")
        accelerator.print("\n" + print_filler)
        accelerator.print(f"Saving state to {save_path}")
        accelerator.save_state(save_path)
        if config["ema_update_steps"] > 0:
            gc.collect()
            accelerator.print(f"Saving EMA state to {save_path}")
            save_ema_model, _ = train_utils.get_diffusion_model(config, "cpu", ema_dtype)
            save_ema_model_state_dict = ema_model.state_dict()
            save_ema_model_state_dict.pop("shadow_params", None)
            save_ema_model.register_to_config(**save_ema_model_state_dict)
            ema_model.copy_to(save_ema_model.parameters())
            save_ema_model.save_pretrained(os.path.join(save_path, "diffusion_ema_model"))
            del save_ema_model
        gc.collect()
        accelerator.print(f"\nSaved states to {save_path}")
    accelerator.end_training()


if __name__ == '__main__':
    main()
