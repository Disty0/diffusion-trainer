import os
import math
import json
import torch
import random
import shutil
import argparse
from tqdm import tqdm
from utils import loader_utils, train_utils

from accelerate import Accelerator
from torch.utils.data import DataLoader

print_filler = "--------------------------------------------------"


def get_bucket_list(batch_size, dataset_paths):
    print("Creating bucket list")
    bucket_list = {}

    for latent_dataset, embed_datasets, repeat in dataset_paths:
        with open(os.path.join(latent_dataset, "bucket_list.json"), "r") as f:
            bucket = json.load(f)
        for key in bucket.keys():
            if key not in bucket_list:
                bucket_list[key] = []
            for i in range(len(bucket[key])):
                for _ in range(repeat):
                    for embed_dataset in embed_datasets:
                        latent_path = os.path.join(latent_dataset, bucket[key][i])
                        embed_path = os.path.join(embed_dataset, bucket[key][i][:-9]+"embed.pt")
                        bucket_list[key].append([latent_path, embed_path])

    keys_to_remove = []
    total_image_count = 0
    for key in bucket_list:
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


def get_batches(batch_size, dataset_paths, dataset_index):
    bucket_list = get_bucket_list(batch_size, dataset_paths)
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

    first_epoch = 0
    current_epoch = 0
    current_step = 0

    accelerator = Accelerator(
        mixed_precision=config["accelerate_mixed_precision"],
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
                if isinstance(unwrap_model(model), train_utils.get_model_class(config["model_type"])):
                    unwrap_model(model).save_pretrained(os.path.join(output_dir, "transformer"))
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")
                weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            model = models.pop()
            if isinstance(unwrap_model(model), train_utils.get_model_class(config["model_type"])):
                load_model = train_utils.get_model_class(config["model_type"]).from_pretrained(input_dir, subfolder="transformer")
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
        get_batches(batch_size, config["dataset_paths"], config["dataset_index"])
    with open(config["dataset_index"], "r") as f:
        epoch_batch = json.load(f)
    dataset = loader_utils.LatentAndEmbedsDataset(epoch_batch)
    train_dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=config["max_load_workers"], prefetch_factor=int(config["load_queue_lenght"]/config["max_load_workers"]))

    dtype = getattr(torch, config["weights_dtype"])
    print(f"Loading diffusion models with dtype {dtype} to device {accelerator.device}")
    accelerator.print(print_filler)
    model, scheduler = train_utils.get_diffusion_model(config["model_type"], config["model_path"], accelerator.device, dtype)
    if config["gradient_checkpointing"]:
        model.enable_gradient_checkpointing()
    if config["fused_optimizer"]:
        optimizer_dict = {p: accelerator.prepare(
                train_utils.get_optimizer(config["optimizer"], [p], config["learning_rate"], foreach=False, **config["optimizer_args"])
            ) for p in model.parameters()
        }
        for key, optimizer in optimizer_dict.items():
            optimizer_dict[key] = [optimizer, accelerator.prepare(train_utils.get_lr_scheduler(config["lr_scheduler"], optimizer, config["lr_scheduler_args"]))]
        def optimizer_hook(parameter):
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

    empty_embed = loader_utils.load_from_file(os.path.join("empty_embeds", "empty_" + config["model_type"] + "_embed.pt"))

    if config["resume_from"] != "none":
        accelerator.print(f"Resuming from: {config['resume_from']}")
        accelerator.load_state(os.path.join(config["project_dir"], config["resume_from"]))
        current_step = int(config["resume_from"].split("-")[1])
        first_epoch = current_step // math.ceil(len(train_dataloader) / config["gradient_accumulation_steps"])

    accelerator.init_trackers(project_name=config["project_name"], config=config)

    progress_bar = tqdm(
        range(0, len(train_dataloader) * config["epochs"]),
        initial=current_step,
        disable=not accelerator.is_local_main_process,
    )

    for _ in range(first_epoch, config["epochs"]):
        model.train()

        for epoch_step, (latents, embeds) in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                model_pred, target = train_utils.run_model(model, scheduler, config["model_type"], accelerator, dtype, latents, embeds, empty_embed, config["dropout_rate"])
                loss = torch.nn.functional.l1_loss(model_pred, target, reduction="mean")
                accelerator.backward(loss)
                if not config["fused_optimizer"]:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    current_step = current_step + 1

                    if accelerator.is_main_process:
                        if current_step % config["checkpoint_save_steps"] == 0:
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
                    if not config["fused_optimizer"]:
                        logs["lr"] = lr_scheduler.get_last_lr()[0]
                    else:
                        logs["lr"] = optimizer_dict[list(optimizer_dict.keys())[0]][1].get_last_lr()[0]
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=current_step)

        current_epoch = current_epoch + 1
        accelerator.print("\n" + print_filler)
        accelerator.print(f"Starting epoch {current_epoch}")
        accelerator.print(f"Current steps done: {current_step}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = unwrap_model(model)
        save_path = os.path.join(config["project_dir"], "checkpoint-final")
        accelerator.save_state(save_path)
        accelerator.print(f"Saved state to {save_path}")
    accelerator.end_training()
