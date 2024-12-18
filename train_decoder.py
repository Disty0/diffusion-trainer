import os
import gc
import math
import json
import torch
import random
import shutil
import argparse
from tqdm import tqdm
from utils import loader_utils, train_utils, latent_utils

from accelerate import Accelerator
from torch.utils.data import DataLoader

print_filler = "--------------------------------------------------"


def get_bucket_list(batch_size, dataset_paths, image_ext):
    print("Creating bucket list")
    bucket_list = {}

    for latent_dataset, image_dataset, repeat in dataset_paths:
        with open(os.path.join(latent_dataset, "bucket_list.json"), "r") as f:
            bucket = json.load(f)
        for key in bucket.keys():
            if key not in bucket_list.keys():
                bucket_list[key] = []
            for i in range(len(bucket[key])):
                for _ in range(repeat):
                    latent_path = os.path.join(latent_dataset, bucket[key][i])
                    image_path = os.path.join(image_dataset, bucket[key][i][:-9]+"image"+image_ext)
                    if os.path.exists(latent_path) and os.path.exists(image_path):
                        bucket_list[key].append([latent_path, image_path])
                    else:
                        print(f"File not found: {bucket[key][i]}")

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


def get_batches(batch_size, dataset_paths, dataset_index, image_ext):
    bucket_list = get_bucket_list(batch_size, dataset_paths, image_ext)
    print("Creating epoch batches")
    epoch_batch = []
    images_left_out_count = 0

    for key, bucket in bucket_list.items():
        random.shuffle(bucket)
        bucket_len = len(bucket)
        images_left_out = bucket_len % batch_size
        images_left_out_count= images_left_out_count + images_left_out
        for i in range(int((bucket_len - images_left_out) / batch_size)):
            epoch_batch.append(bucket[i*batch_size:(i+1)*batch_size])
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
    try:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
    except Exception:
        pass

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
        model = model._orig_mod if isinstance(model, torch._dynamo.eval_frame.OptimizedModule) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(unwrap_model(model), latent_utils.get_latent_model_class(config["model_type"])):
                    unwrap_model(model).save_pretrained(os.path.join(output_dir, "decoder"))
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")
                weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            model = models.pop()
            if isinstance(unwrap_model(model), latent_utils.get_latent_model_class(config["model_type"])):
                load_model = latent_utils.get_latent_model_class(config["model_type"]).from_pretrained(input_dir, subfolder="decoder")
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
        get_batches(batch_size, config["dataset_paths"], config["dataset_index"], config["image_ext"])
    accelerator.wait_for_everyone()
    with open(config["dataset_index"], "r") as f:
        epoch_batch = json.load(f)

    dtype = getattr(torch, config["weights_dtype"])
    print(f"Loading latent models with dtype {dtype} to device {accelerator.device}")
    accelerator.print(print_filler)
    model, image_processor = latent_utils.get_latent_model(config["model_type"], config["model_path"], accelerator.device, dtype, "no")
    if config["gradient_checkpointing"]:
        model.enable_gradient_checkpointing()

    dataset = loader_utils.LatentAndImagesDataset(epoch_batch, image_processor)
    train_dataloader = DataLoader(dataset=dataset, batch_size=None, batch_sampler=None, shuffle=False, pin_memory=True, num_workers=config["max_load_workers"], prefetch_factor=int(config["load_queue_lenght"]/config["max_load_workers"]))

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

    accelerator.init_trackers(project_name=config["project_name"], config=config)

    progress_bar = tqdm(
        range(0, math.ceil(len(train_dataloader) / config["gradient_accumulation_steps"]) * config["epochs"]),
        initial=current_step,
        disable=not accelerator.is_local_main_process,
    )

    grad_norm = []
    if hasattr(model, "decoder") and hasattr(model, "encoder"):
        model.eval()
        model.requires_grad_(False)
        model.encoder.eval()
        model.encoder.requires_grad_(False)
        model.decoder.train()
        model.decoder.requires_grad_(True)
    else:
        model.train()
    getattr(torch, accelerator.device.type).empty_cache()

    for _ in range(first_epoch, config["epochs"]):
        for epoch_step, (latents_list, image_tensors_list) in enumerate(train_dataloader):
            with torch.no_grad():
                latents = []
                for i in range(len(latents_list)):
                    latents.append(latents_list[i].to(dtype=torch.float32))
                latents = torch.stack(latents).to(accelerator.device, dtype=torch.float32)
                image_tensors = []
                for i in range(len(image_tensors_list)):
                    image_tensors.append(image_tensors_list[i].to(dtype=torch.float32))
                image_tensors = torch.stack(image_tensors).to(accelerator.device, dtype=torch.float32)
            with accelerator.accumulate(model):
                model_pred = latent_utils.decode_latents(model, image_processor, latents, config["model_type"], accelerator.device, return_image=False, mixed_precision=config["mixed_precision"])
                loss = torch.nn.functional.l1_loss(model_pred, image_tensors, reduction="mean")
                accelerator.backward(loss)
                if accelerator.sync_gradients and config["max_grad_norm"] > 0:
                    grad_norm.append(accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"]))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    current_step = current_step + 1

                    if accelerator.is_main_process:
                        if current_step % config["checkpoint_save_steps"] == 0:
                            os.makedirs(config["project_dir"], exist_ok=True)
                            if config["checkpoints_limit"] != 0:
                                checkpoints = os.listdir(config["project_dir"])
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                                if len(checkpoints) >= config["checkpoints_limit"]:
                                    num_to_remove = len(checkpoints) - config["checkpoints_limit"] + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]
                                    accelerator.print(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                                    accelerator.print(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(config["project_dir"], removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(config["project_dir"], f"checkpoint-{current_step}")
                            accelerator.save_state(save_path)
                            accelerator.print(f"Saved state to {save_path}")

                    logs = {"loss": loss.detach().item(), "epoch": current_epoch}
                    logs["lr"] = lr_scheduler.get_last_lr()[0]
                    progress_bar.set_postfix(**logs)
                    if len(grad_norm) > 0:
                        avg_grad_norm = 0
                        for i in grad_norm:
                            avg_grad_norm += i
                        avg_grad_norm = avg_grad_norm / len(grad_norm)
                        grad_norm = []
                        logs["grad_norm"] = avg_grad_norm
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
                get_batches(batch_size, config["dataset_paths"], config["dataset_index"], config["image_ext"])
            accelerator.wait_for_everyone()
            with open(config["dataset_index"], "r") as f:
                epoch_batch = json.load(f)
            dataset = loader_utils.LatentAndImagesDataset(epoch_batch, image_processor)
            train_dataloader = DataLoader(dataset=dataset, batch_size=None, batch_sampler=None, shuffle=False, pin_memory=True, num_workers=config["max_load_workers"], prefetch_factor=int(config["load_queue_lenght"]/config["max_load_workers"]))
            train_dataloader = accelerator.prepare(train_dataloader)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = unwrap_model(model)
        save_path = os.path.join(config["project_dir"], "checkpoint-final")
        accelerator.save_state(save_path)
        accelerator.print(f"Saved state to {save_path}")
    accelerator.end_training()
