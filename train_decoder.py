from typing import Dict, List, Tuple

import os
import gc
import math
import json
import torch

if not torch.version.cuda:
    import transformers
    transformers.utils.is_flash_attn_2_available = lambda: False

import random
import shutil
import argparse
from tqdm import tqdm

from accelerate import Accelerator
from torch.utils.data import DataLoader

from utils import loader_utils, train_utils, latent_utils
from utils.ema_model import EMAModel

print_filler = "--------------------------------------------------"


def get_bucket_list(batch_size: int, dataset_paths: List[dict], image_ext: str) -> Dict[str, List[str]]:
    print("Creating bucket list")
    bucket_list = {}

    dataset_progress_bar = tqdm(total=len(dataset_paths))
    dataset_progress_bar.set_description("Loading datasets")
    bucket_progress_bar = tqdm()
    bucket_progress_bar.set_description("Loading buckets")
    image_progress_bar = tqdm()
    image_progress_bar.set_description("Loading images")

    for dataset in dataset_paths:
        dataset_progress_bar.set_postfix(current=dataset["path"])

        bucket_list_path = dataset["bucket_list"]
        if not os.path.exists(bucket_list_path):
            bucket_list_path = os.path.join(dataset["path"], bucket_list_path)
        with open(bucket_list_path, "r") as f:
            bucket = json.load(f)
        gc.collect()

        bucket_progress_bar.reset(total=len(bucket.keys()))

        for key in bucket.keys():
            current_bucket_list = []
            if key not in bucket_list.keys():
                bucket_list[key] = []

            bucket_progress_bar.set_postfix(current=key)
            image_progress_bar.set_postfix(current=key)
            image_progress_bar.reset(total=len(bucket[key]))

            for file_name in bucket[key]:
                latent_path = os.path.join(dataset["path"], file_name)
                image_path = os.path.join(dataset["images"], file_name[:-9]+"image"+image_ext)
                if not os.path.exists(latent_path):
                    print(f"Latent file not found: {latent_path}")
                elif not os.path.exists(image_path):
                    print(f"Image file not found: {image_path}")
                else:
                    current_bucket_list.extend([(latent_path, image_path)]*dataset["repeats"])
                image_progress_bar.update(1)

            bucket_list[key].extend(current_bucket_list)
            bucket_progress_bar.update(1)
        dataset_progress_bar.update(1)

    dataset_progress_bar.close()
    bucket_progress_bar.close()
    image_progress_bar.close()

    keys_to_remove = []
    total_image_count = 0
    
    bucket_progress_bar = tqdm(total=len(bucket_list.keys()))
    bucket_progress_bar.set_description("Processing buckets")
    
    for key in bucket_list.keys():
        bucket_progress_bar.set_postfix(current=key)
        bucket_len = len(bucket_list[key])
        if bucket_len < batch_size:
            keys_to_remove.append(key)
        else:
            random.shuffle(bucket_list[key])
            total_image_count = total_image_count + bucket_len
        bucket_progress_bar.update(1)
    
    bucket_progress_bar.close()

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


def get_batches(batch_size: int, dataset_paths: List[Tuple[str, str, int]], dataset_index: str, image_ext: str) -> None:
    bucket_list = get_bucket_list(batch_size, dataset_paths, image_ext)
    print("Creating epoch batches")

    epoch_batch = []
    images_left_out_count = 0

    bucket_progress_bar = tqdm(total=len(bucket_list.keys()))
    bucket_progress_bar.set_description("Loading batches")

    for key, bucket in bucket_list.items():
        bucket_progress_bar.set_postfix(current=key)
        random.shuffle(bucket)
        bucket_len = len(bucket)
        images_left_out = bucket_len % batch_size
        images_left_out_count= images_left_out_count + images_left_out
        for i in range(int((bucket_len - images_left_out) / batch_size)):
            epoch_batch.append((bucket[i*batch_size:(i+1)*batch_size], key))
        print(print_filler)
        print(f"Images left out from bucket {key}: {images_left_out}")
        print(f"Images left in the bucket {key}: {bucket_len - images_left_out}")
        bucket_progress_bar.update(1)

    bucket_progress_bar.close()

    print(print_filler)
    print(f"Images that got left out from the epoch: {images_left_out_count}")
    print(f"Total images left in the epoch: {len(epoch_batch) * batch_size}")
    print(f"Batches * Batch Size: {len(epoch_batch)} * {batch_size}")
    print(print_filler + "\n")

    random.shuffle(epoch_batch)
    os.makedirs(os.path.dirname(dataset_index), exist_ok=True)
    print(f"Saving dataset index to: {dataset_index}")
    with open(dataset_index, "w") as f:
        json.dump(epoch_batch, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model with a given config")
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)
    gc.collect()

    if config["tunableop"] != "default":
        torch.cuda.tunable.enable(config["tunableop"])
    torch.backends.cudnn.enabled = config["cudnn_enabled"]

    if config["dynamo_backend"] != "no":
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)

    if config["allow_tf32"]:
        torch.backends.fp32_precision = "tf32"
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        torch.backends.cudnn.rnn.fp32_precision = "tf32"
    else:
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cudnn.rnn.fp32_precision = "ieee"

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

    def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        model = accelerator.unwrap_model(model)
        return model._orig_mod if isinstance(model, torch._dynamo.eval_frame.OptimizedModule) else model

    def save_model_hook(models: List[torch.nn.Module], weights: List[Dict[str, torch.Tensor]], output_dir: str) -> None:
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(unwrap_model(model), latent_utils.get_latent_model_class(config["model_type"])):
                    unwrap_model(model).save_pretrained(os.path.join(output_dir, "decoder_model"))
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")
                weights.pop()

    def load_model_hook(models: List[torch.nn.Module], input_dir: str) -> None:
        for _ in range(len(models)):
            model = models.pop()
            if isinstance(unwrap_model(model), latent_utils.get_latent_model_class(config["model_type"])):
                load_model = latent_utils.get_latent_model_class(config["model_type"]).from_pretrained(input_dir, subfolder="decoder_model")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
            else:
                raise ValueError(f"Unsupported model found: {type(model)=}")
            del load_model

    def fused_optimizer_hook(parameter: torch.nn.Parameter) -> None:
        global grad_scaler
        if config["log_grad_stats"]:
            global grad_max, grad_mean, grad_mean_count
            param_grad_abs = parameter.grad.abs()
            grad_max = max(grad_max, param_grad_abs.max().item())
            grad_mean += param_grad_abs.mean().item()
            grad_mean_count += 1
        if accelerator.sync_gradients:
            if grad_scaler is not None and (config["max_grad_clip"] > 0 or config["max_grad_norm"] > 0):
                grad_scaler.unscale_(optimizer[parameter][0])
            if config["max_grad_clip"] > 0:
                # this is **very** slow with fp16
                accelerator.clip_grad_value_(parameter, config["max_grad_clip"])
                if config["log_grad_stats"]:
                    global clipped_grad_mean, clipped_grad_mean_count
                    clipped_grad_mean += parameter.grad.abs().mean().item()
                    clipped_grad_mean_count += 1
            if config["max_grad_norm"] > 0:
                # this is **very** slow with fp16 and norming per parameter isn't ideal
                global grad_norm, grad_norm_count
                grad_norm += accelerator.clip_grad_norm_(parameter, config["max_grad_norm"])
                grad_norm_count += 1
        if grad_scaler is not None:
            grad_scaler.step(optimizer[parameter][0])
        else:
            optimizer[parameter][0].step()
        optimizer[parameter][1].step()
        optimizer[parameter][0].zero_grad()

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    accelerator.print("\n" + print_filler)
    accelerator.print("Initializing the trainer")
    accelerator.print(print_filler)

    dtype = getattr(torch, config["weights_dtype"])
    print(f"Loading latent models with dtype {dtype} to device {accelerator.device}")
    accelerator.print(print_filler)
    model, image_processor = latent_utils.get_latent_model(config["model_type"], config["model_path"], accelerator.device, dtype, "no")
    if config["gradient_checkpointing"]:
        model.enable_gradient_checkpointing()
    model = accelerator.prepare(model)
    gc.collect()

    optimizer, lr_scheduler = train_utils.get_optimizer_and_lr_scheduler(config, model, accelerator, fused_optimizer_hook)

    global grad_scaler
    if config["use_grad_scaler"]:
        from utils.grad_scaler import GradScaler
        grad_scaler = GradScaler(accelerator.device.type)
        grad_scaler = accelerator.prepare(grad_scaler)
    else:
        grad_scaler = None

    batch_size = config["batch_size"]
    if accelerator.is_local_main_process and not os.path.exists(config["dataset_index"]):
        get_batches(batch_size, config["dataset_paths"], config["dataset_index"], config["image_ext"])
        gc.collect()
    accelerator.wait_for_everyone()

    accelerator.print(f'Loading dataset index: {config["dataset_index"]}')
    with open(config["dataset_index"], "r") as f:
        epoch_batch = json.load(f)
    gc.collect()

    dataset = loader_utils.LatentsAndImagesDataset(epoch_batch, image_processor)
    del epoch_batch
    gc.collect()

    train_dataloader = DataLoader(dataset=dataset, batch_size=None, batch_sampler=None, shuffle=False, pin_memory=True, num_workers=config["max_load_workers"], prefetch_factor=int(config["load_queue_lenght"]/config["max_load_workers"]))
    train_dataloader = accelerator.prepare(train_dataloader)
    del dataset
    gc.collect()

    resume_checkpoint = None
    if config.get("resume_from", "") and config["resume_from"] != "none":
        if config["resume_from"] == "latest":
            checkpoints = os.listdir(config["project_dir"])
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint-")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            if len(checkpoints) > 0:
                resume_checkpoint = checkpoints[-1]
        else:
            resume_checkpoint = config["resume_from"]
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
        print(f'Loading EMA models with dtype {ema_dtype} to device {"cpu" if config["update_ema_on_cpu"] or config["offload_ema_to_cpu"] else accelerator.device}')
        accelerator.print(print_filler)
        if resume_checkpoint is not None:
            accelerator.print(f"Resuming EMA from: {resume_checkpoint}")
            accelerator.print(print_filler)
            ema_model = EMAModel.from_pretrained(os.path.join(config["project_dir"], config["resume_from"], "decoder_ema_model"), latent_utils.get_latent_model_class(config["model_type"]), foreach=config["use_foreach_ema"], torch_dtype=ema_dtype)
            ema_model.to("cpu" if config["update_ema_on_cpu"] or config["offload_ema_to_cpu"] else accelerator.device)
        else:
            accelerator.print(print_filler)
            orig_model, _ = latent_utils.get_latent_model(config["model_type"], config["model_path"], "cpu" if config["update_ema_on_cpu"] or config["offload_ema_to_cpu"] else accelerator.device, ema_dtype, "no")
            ema_model = EMAModel(orig_model.parameters(), model_cls=latent_utils.get_latent_model_class(config["model_type"]), model_config=orig_model.config, foreach=config["use_foreach_ema"], decay=config["ema_decay"])
            orig_model = orig_model.to("meta")
            orig_model = None
            del orig_model
        if config["offload_ema_pin_memory"]:
            ema_model.pin_memory()
        accelerator.print(print_filler)

    accelerator.init_trackers(project_name=config["project_name"], config=config)

    progress_bar = tqdm(
        range(0, math.ceil(len(train_dataloader) / config["gradient_accumulation_steps"]) * config["epochs"]),
        initial=current_step,
        disable=not accelerator.is_local_main_process,
    )

    global grad_max, grad_mean, grad_mean_count, clipped_grad_mean, clipped_grad_mean_count, grad_norm, grad_norm_count
    grad_norm = torch.tensor(0.0, dtype=dtype, device=accelerator.device)
    grad_mean = 0
    clipped_grad_mean = 0
    grad_norm_count = 0
    skip_grad_norm_count = 0
    grad_mean_count = 0
    clipped_grad_mean_count = 0
    grad_max = 0
    loss = 1.0
    loss_func = train_utils.get_loss_func(config)

    if hasattr(model, "decoder") and hasattr(model, "encoder"):
        model.eval()
        model.requires_grad_(False)
        model.encoder.eval()
        model.encoder.requires_grad_(False)
        model.decoder.train()
        model.decoder.requires_grad_(True)
    else:
        model.train()

    gc.collect()
    if accelerator.device.type != "cpu":
        getattr(torch, accelerator.device.type).empty_cache()

    for _ in range(first_epoch, config["epochs"]):
        for epoch_step, (latents_list, image_tensors_list) in enumerate(train_dataloader):
            with torch.no_grad():
                latents = []
                for i in range(len(latents_list)):
                    latents.append(latents_list[i].to(accelerator.device, dtype=torch.float32))
                latents = torch.stack(latents).to(accelerator.device, dtype=torch.float32)
                image_tensors = []
                for i in range(len(image_tensors_list)):
                    image_tensors.append(image_tensors_list[i].to(accelerator.device, dtype=torch.float32))
                image_tensors = torch.stack(image_tensors).to(accelerator.device, dtype=torch.float32)
            with accelerator.accumulate(model):
                last_loss = loss
                model_pred = latent_utils.decode_latents(model, image_processor, latents, config["model_type"], accelerator.device, return_image=False, mixed_precision=config["mixed_precision"])
                loss = loss_func(model_pred, image_tensors, reduction=config["loss_reduction"])
                del model_pred
                if grad_scaler is not None:
                    accelerator.backward(grad_scaler.scale(loss))
                else:
                    accelerator.backward(loss)
                loss = loss.detach().item()
                if not config["fused_optimizer"]:
                    if accelerator.sync_gradients:
                        if grad_scaler is not None and (config["max_grad_clip"] > 0 or config["max_grad_norm"] > 0):
                            grad_scaler.unscale_(optimizer)
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
                    if grad_scaler is not None:
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                    else:
                        optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                else:
                    if grad_scaler is not None:
                        grad_scaler.update()

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
                                model.to(device=accelerator.device, non_blocking=config["offload_ema_non_blocking"])
                            elif config["offload_ema_to_cpu"]:
                                ema_model.to(device="cpu", non_blocking=False)
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
                                save_ema_model, _ = latent_utils.get_latent_model(config["model_type"], config["model_path"], "cpu", ema_dtype, "no")
                                ema_model.copy_to(save_ema_model.parameters())
                                save_ema_model.save_pretrained(os.path.join(save_path, "decoder_ema_model"))
                                save_ema_model = save_ema_model.to("meta")
                                save_ema_model = None
                                del save_ema_model
                                save_ema_model_state_dict = ema_model.state_dict()
                                save_ema_model_state_dict.pop("shadow_params", None)
                                with open(os.path.join(save_path, "decoder_ema_model", "ema_state.json"), "w") as f:
                                    json.dump(save_ema_model_state_dict, f)
                                del save_ema_model_state_dict
                            gc.collect()
                            accelerator.print(f"\nSaved states to {save_path}")
                            accelerator.print(print_filler)
                        accelerator.wait_for_everyone()

                    logs = {"loss": loss, "epoch": current_epoch}
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

                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=current_step)

                    if current_step == start_step + 1 or (config["gc_steps"] != 0 and current_step % config["gc_steps"] == 0):
                        gc.collect()
                        if accelerator.device.type != "cpu":
                            getattr(torch, accelerator.device.type).empty_cache()

        current_epoch = current_epoch + 1
        accelerator.print("\n" + print_filler)
        accelerator.print(f"Starting epoch {current_epoch}")
        accelerator.print(f"Current steps done: {current_step}")
        if config["reshuffle"]:
            del train_dataloader
            gc.collect()
            if accelerator.is_local_main_process:
                os.rename(config["dataset_index"], config["dataset_index"]+"-epoch_"+str(current_epoch-1)+".json")
                get_batches(batch_size, config["dataset_paths"], config["dataset_index"], config["image_ext"])
                gc.collect()
            accelerator.wait_for_everyone()

            accelerator.print(f'Loading dataset index: {config["dataset_index"]}')
            with open(config["dataset_index"], "r") as f:
                epoch_batch = json.load(f)
            gc.collect()

            accelerator.print('Setting up dataset loader: LatentsAndImagesDataset')
            dataset = loader_utils.LatentsAndImagesDataset(epoch_batch, image_processor)
            del epoch_batch
            gc.collect()

            train_dataloader = DataLoader(dataset=dataset, batch_size=None, batch_sampler=None, shuffle=False, pin_memory=True, num_workers=config["max_load_workers"], prefetch_factor=int(config["load_queue_lenght"]/config["max_load_workers"]))
            train_dataloader = accelerator.prepare(train_dataloader)
            del dataset
            gc.collect()

    progress_bar.close()
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
            save_ema_model, _ = latent_utils.get_latent_model(config["model_type"], config["model_path"], "cpu", ema_dtype, "no")
            ema_model.copy_to(save_ema_model.parameters())
            save_ema_model.save_pretrained(os.path.join(save_path, "decoder_ema_model"))
            save_ema_model = save_ema_model.to("meta")
            save_ema_model = None
            del save_ema_model
            save_ema_model_state_dict = ema_model.state_dict()
            save_ema_model_state_dict.pop("shadow_params", None)
            with open(os.path.join(save_path, "decoder_ema_model", "ema_state.json"), "w") as f:
                json.dump(save_ema_model_state_dict, f)
            del save_ema_model_state_dict
        gc.collect()
        accelerator.print(f"\nSaved states to {save_path}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
