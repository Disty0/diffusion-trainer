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

print_filler = "--------------------------------------------------"


def get_bucket_list(batch_size, dataset_paths, empty_embed_path, latent_type="latent"):
    embed_suffix = "_" + empty_embed_path.rsplit("empty_", maxsplit=1)[-1]
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
                            bucket_list[key].append([latent_path, embed_path])

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


def get_batches(batch_size, dataset_paths, dataset_index, empty_embed_path, latent_type="latent"):
    bucket_list = get_bucket_list(batch_size, dataset_paths, empty_embed_path, latent_type=latent_type)
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
        elif latent_type == "image":
            for i in range(int((bucket_len - images_left_out) / batch_size)):
                epoch_batch.append([key, bucket[i*batch_size:(i+1)*batch_size]])
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with a given config')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    if config["tunableop"]:
        torch.cuda.tunable.enable(val=True)
    if config["dynamo_backend"] != "no":
        torch._dynamo.config.cache_size_limit = 64
    try:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
    except Exception:
        pass

    empty_embed_path = os.path.join("empty_embeds", "empty_" + config["model_type"] + "_embed.pt")
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

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    accelerator.print("\n" + print_filler)

    batch_size = config["batch_size"]
    if accelerator.is_local_main_process and not os.path.exists(config["dataset_index"]):
        get_batches(batch_size, config["dataset_paths"], config["dataset_index"], empty_embed_path, latent_type=config["latent_type"])
    accelerator.wait_for_everyone()
    with open(config["dataset_index"], "r") as f:
        epoch_batch = json.load(f)
    if config["latent_type"] == "latent":
        dataset = loader_utils.LatentsAndEmbedsDataset(epoch_batch)
    elif config["latent_type"] == "image":
        dataset = loader_utils.ImagesAndEmbedsDataset(epoch_batch)
    train_dataloader = DataLoader(dataset=dataset, batch_size=None, batch_sampler=None, shuffle=False, pin_memory=True, num_workers=config["max_load_workers"], prefetch_factor=int(config["load_queue_lenght"]/config["max_load_workers"]))

    dtype = getattr(torch, config["weights_dtype"])
    print(f"Loading diffusion models with dtype {dtype} to device {accelerator.device}")
    accelerator.print(print_filler)
    model, scheduler = train_utils.get_diffusion_model(config["model_type"], config["model_path"], accelerator.device, dtype)
    if config["gradient_checkpointing"]:
        model.enable_gradient_checkpointing()

    if config["fused_optimizer"]:
        optimizer_dict = {p: accelerator.prepare(
                train_utils.get_optimizer(config["optimizer"], [p], config["learning_rate"], **config["optimizer_args"])
            ) for p in model.parameters()
        }
        for key, optimizer in optimizer_dict.items():
            optimizer_dict[key] = [optimizer, accelerator.prepare(train_utils.get_lr_scheduler(config["lr_scheduler"], optimizer, **config["lr_scheduler_args"]))]
        def optimizer_hook(parameter):
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
            optimizer_dict[parameter][0].step()
            optimizer_dict[parameter][1].step()
            optimizer_dict[parameter][0].zero_grad()
        for p in model.parameters():
            p.register_post_accumulate_grad_hook(optimizer_hook)
        train_dataloader, model = accelerator.prepare(train_dataloader, model)
    else:
        optimizer = train_utils.get_optimizer(config["optimizer"], model.parameters(), config["learning_rate"], **config["optimizer_args"])
        lr_scheduler = train_utils.get_lr_scheduler(config["lr_scheduler"], optimizer, **config["lr_scheduler_args"])
        train_dataloader, model, optimizer, lr_scheduler = accelerator.prepare(train_dataloader, model, optimizer, lr_scheduler)

    if config.get("resume_from", "") and config["resume_from"] != "none":
        accelerator.print(f"Resuming from: {config['resume_from']}")
        accelerator.load_state(os.path.join(config["project_dir"], config["resume_from"]))
        current_step = int(config["resume_from"].split("-")[1])
        first_epoch = current_step // math.ceil(len(train_dataloader) / config["gradient_accumulation_steps"])
        current_epoch = first_epoch
        start_step = current_step

    if config["ema_update_steps"] > 0 and accelerator.is_main_process:
        ema_dtype = getattr(torch, config["ema_weights_dtype"])
        accelerator.print("\n" + print_filler)
        print(f'Loading EMA models with dtype {ema_dtype} to device {"cpu" if config["update_ema_on_cpu"] or config["offload_ema_to_cpu"] else accelerator.device}')
        accelerator.print(print_filler)
        if config.get("resume_from", "") and config["resume_from"] != "none":
            ema_model = EMAModel.from_pretrained(os.path.join(config["project_dir"], config["resume_from"], "diffusion_ema_model"), train_utils.get_model_class(config["model_type"]), foreach=config["use_foreach_ema"])
            ema_model.to("cpu" if config["update_ema_on_cpu"] or config["offload_ema_to_cpu"] else accelerator.device, dtype=ema_dtype)
        else:
            ema_model, _ = train_utils.get_diffusion_model(config["model_type"], config["model_path"], "cpu" if config["update_ema_on_cpu"] or config["offload_ema_to_cpu"] else accelerator.device, ema_dtype)
            ema_model = EMAModel(ema_model.parameters(), model_cls=train_utils.get_model_class(config["model_type"]), model_config=ema_model.config, foreach=config["use_foreach_ema"], decay=config["ema_decay"])
        if config["offload_ema_pin_memory"]:
            ema_model.pin_memory()

    accelerator.init_trackers(project_name=config["project_name"], config=config)

    progress_bar = tqdm(
        range(0, math.ceil(len(train_dataloader) / config["gradient_accumulation_steps"]) * config["epochs"]),
        initial=current_step,
        disable=not accelerator.is_local_main_process,
    )

    empty_embeds_added_count = 0
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
    model.train()
    getattr(torch, accelerator.device.type).empty_cache()
    for _ in range(first_epoch, config["epochs"]):
        for epoch_step, (latents_list, embeds_list) in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                last_loss = loss
                model_pred, target, timesteps, empty_embeds_added = train_utils.run_model(model, scheduler, config, accelerator, dtype, latents_list, embeds_list, empty_embed)
                if config["loss_type"] == "mae":
                    loss = torch.nn.functional.l1_loss(model_pred, target, reduction=config["loss_reduction"])
                elif config["loss_type"] == "mse":
                    loss = torch.nn.functional.mse_loss(model_pred, target, reduction=config["loss_reduction"])
                else:
                    loss = getattr(torch.nn.functional, config["loss_type"])(model_pred, target, reduction=config["loss_reduction"])
                accelerator.backward(loss)
                if not config["fused_optimizer"]:
                    if accelerator.sync_gradients:
                        if config["log_grad_stats"]:
                            for parameter in model.parameters():
                                if hasattr(parameter, "grad"):
                                    param_grad_abs = parameter.grad.abs()
                                    grad_max = max(grad_max, param_grad_abs.max().item())
                                    grad_mean += param_grad_abs.mean().item()
                                    grad_mean_count += 1
                        if config["max_grad_clip"] > 0:
                            accelerator.clip_grad_value_(model.parameters(), config["max_grad_clip"])
                            if config["log_grad_stats"]:
                                for parameter in model.parameters():
                                    if hasattr(parameter, "grad"):
                                        clipped_grad_mean += parameter.grad.abs().mean().item()
                                        clipped_grad_mean_count += 1
                        if config["max_grad_norm"] > 0:
                            grad_norm += accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                            grad_norm_count += 1
                    if grad_norm_count > 0 and (grad_norm.isnan() or (config["skip_grad_norm"] > 0 and current_step > config["skip_grad_norm_steps"] and (grad_norm / grad_norm_count) > config["skip_grad_norm"])):
                        loss = last_loss
                        skip_grad_norm_count += 1
                    else:
                        optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if timesteps is not None:
                    timesteps_list.extend(timesteps.to("cpu", dtype=torch.float32).detach().tolist())
                if empty_embeds_added is not None:
                    empty_embeds_added_count += empty_embeds_added

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
                            os.makedirs(config["project_dir"], exist_ok=True)
                            if config["checkpoints_limit"] != 0:
                                checkpoints = os.listdir(config["project_dir"])
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
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
                                save_ema_model, _ = train_utils.get_diffusion_model(config["model_type"], config["model_path"], "cpu", ema_dtype)
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
                    if not config["fused_optimizer"]:
                        logs["lr"] = lr_scheduler.get_last_lr()[0]
                    else:
                        logs["lr"] = optimizer_dict[list(optimizer_dict.keys())[0]][1].get_last_lr()[0]
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
                    if config["skip_grad_norm"] > 0:
                        logs["skip_grad_norm_count"] = skip_grad_norm_count
                    if accelerator.is_main_process:
                        if config["ema_update_steps"] > 0:
                            logs["ema_decay"] = ema_model.get_decay(ema_model.optimization_step)

                    progress_bar.set_postfix(**logs)
                    if config["dropout_rate"] > 0:
                        logs["empty_embeds_added_count"] = empty_embeds_added_count
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
            train_dataloader = accelerator.unwrap_model(train_dataloader)
            del dataset, train_dataloader
            if accelerator.is_local_main_process:
                os.rename(config["dataset_index"], config["dataset_index"]+"-epoch_"+str(current_epoch-1)+".json")
                get_batches(batch_size, config["dataset_paths"], config["dataset_index"], empty_embed_path, latent_type=config["latent_type"])
            accelerator.wait_for_everyone()
            with open(config["dataset_index"], "r") as f:
                epoch_batch = json.load(f)
            dataset = loader_utils.LatentsAndEmbedsDataset(epoch_batch)
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
            save_ema_model, _ = train_utils.get_diffusion_model(config["model_type"], config["model_path"], "cpu", ema_dtype)
            save_ema_model_state_dict = ema_model.state_dict()
            save_ema_model_state_dict.pop("shadow_params", None)
            save_ema_model.register_to_config(**save_ema_model_state_dict)
            ema_model.copy_to(save_ema_model.parameters())
            save_ema_model.save_pretrained(os.path.join(save_path, "diffusion_ema_model"))
            del save_ema_model
        gc.collect()
        accelerator.print(f"\nSaved states to {save_path}")
    accelerator.end_training()
