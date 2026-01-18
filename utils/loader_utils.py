import os
import gc
import math
import time
import random
import warnings

import brotli
import torch
import numpy as np

import pillow_jxl # noqa: F401
from PIL import Image
from io import BytesIO
from queue import Queue
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor

from typing import List, Tuple, Union
from transformers import ImageProcessingMixin, AutoTokenizer

Image.MAX_IMAGE_PIXELS = 999999999 # 178956970
warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning)


def load_from_file(path: str) -> torch.FloatTensor:
    with open(path, "rb") as file:
        data = file.read()
    decompressed_data = brotli.decompress(data)
    stored_tensor = BytesIO(decompressed_data)
    stored_tensor.seek(0)
    return torch.load(stored_tensor, map_location="cpu", weights_only=True)


def load_image_from_file(image_path: str, target_size: str) -> Image.Image:
        if isinstance(target_size, str):
            target_size = target_size.split("x")
            target_size[0] = int(target_size[0])
            target_size[1] = int(target_size[1])

        with Image.open(image_path) as img:
            background = Image.new("RGBA", img.size, (255, 255, 255))
            image = Image.alpha_composite(background, img.convert("RGBA")).convert("RGB")

        orig_size = image.size
        new_size = [math.ceil(target_size[1] * orig_size[0] / orig_size[1]), math.ceil(target_size[0] * orig_size[1] / orig_size[0])]
        if new_size[0] > target_size[0]:
            image = image.resize((new_size[0], target_size[1]), Image.BICUBIC)
        else:
            image = image.resize((target_size[0], new_size[1]), Image.BICUBIC)

        new_size = image.size
        left_diff = new_size[0] - target_size[0]
        if left_diff < 0: # can go off by 1 or 2 pixel
            image = image.resize((target_size[0], new_size[1]), Image.BICUBIC)
            new_size = image.size
            left_diff = new_size[0] - target_size[0]

        top_diff = (new_size[1] - target_size[1])
        if top_diff < 0: # can go off by 1 or 2 pixel
            image = image.resize((new_size[0], target_size[1]), Image.BICUBIC)
            new_size = image.size
            top_diff = (new_size[1] - target_size[1])

        left_shift = random.randint(0, left_diff)
        top_shift = random.randint(0, top_diff)
        image = image.crop((left_shift, top_shift, (left_shift + target_size[0]), (top_shift + target_size[1])))

        new_size = image.size
        if new_size[0] != target_size[0] or new_size[1] != target_size[1]: # sanity check
            image = image.resize((target_size[0], target_size[1]), Image.BICUBIC)

        return image


class DiffusionDataset(Dataset):
    def __init__(self, batches: List[Tuple[List[Tuple[str, str]], str]], config: dict):
        self.batches = batches
        self.config = config

        if self.config["embed_type"] == "token":
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_path"], subfolder="tokenizer")

        if self.config["latent_type"] == "jpeg" and self.config["encode_dcts_with_cpu"]:
            from raiflow import RaiFlowImageEncoder
            self.image_encoder = RaiFlowImageEncoder.from_pretrained(self.config["model_path"], subfolder="image_encoder")

    def __len__(self) -> int:
        return len(self.batches)

    @torch.no_grad()
    def __getitem__(self, index: int) -> Tuple[List[torch.FloatTensor], List[torch.FloatTensor]]:
        latents = []
        embeds = []
        resolution = self.batches[index][1]
        for batch in self.batches[index][0]:
            if self.config["latent_type"] in {"image_tensor", "image_pt"}:
                with load_image_from_file(batch[0], resolution) as image:
                    latents.append(torch.from_numpy(np.asarray(image).copy()))
            elif self.config["latent_type"] != "latent":
                latents.append(load_image_from_file(batch[0], resolution))
            else:
                latents.append(load_from_file(batch[0]).to(dtype=torch.float32))

            if self.config["embed_type"] in {"text", "token"}:
                if batch[1] == "":
                    text = ""
                else:
                    with open(batch[1], "r") as file:
                        text = file.read()
                    if text and text[-1] == "\n":
                        text = text[:-1]
                embeds.append(text)
            else:
                embeds.append(load_from_file(batch[1]))

        if self.config["embed_type"] == "token":
            embeds = self.tokenizer(text=embeds, padding="longest", pad_to_multiple_of=256, max_length=self.config["max_token_length"], truncation=True, add_special_tokens=True, return_tensors="pt").input_ids

        if self.config["latent_type"] in {"latent", "image_pt"}:
            latents = torch.stack(latents)
        elif self.config["latent_type"] == "image_tensor":
            latents = torch.stack(latents)
            latents = latents.permute(0,3,1,2).to(dtype=torch.float32)
            latents = latents.div_(127.5).sub_(1)
        elif self.config["latent_type"] == "jpeg" and self.config["encode_dcts_with_cpu"]:
            latents = self.image_encoder.encode(latents, device="cpu")

        return (latents, embeds)


class LatentDecoderDataset(Dataset):
    def __init__(self, batches: List[Tuple[str, str]], image_processor: ImageProcessingMixin):
        self.batches = batches
        self.image_processor = image_processor

    def __len__(self) -> int:
        return len(self.batches)

    @torch.no_grad()
    def __getitem__(self, index: int) -> Tuple[List[torch.FloatTensor], List[torch.FloatTensor]]:
        latents = []
        image_tensors = []
        # resolution = self.batches[index][1]
        for batch in self.batches[index][0]:
            latents.append(load_from_file(batch[0]).to(dtype=torch.float32))
            with Image.open(batch[1]) as image:
                image_tensors.append(self.image_processor.preprocess(image)[0])
        latents = torch.stack(latents)
        image_tensors = torch.stack(image_tensors)
        return (latents, image_tensors)


class SaveBackend():
    def __init__(self, model_type: str, save_dtype: torch.dtype, save_queue_lenght: int = 4096, max_save_workers: int = 8):
        self.save_queue_lenght = 0
        self.model_type = model_type
        self.save_dtype = save_dtype
        self.keep_saving = True
        self.max_save_queue_lenght = save_queue_lenght
        self.save_queue = Queue()
        self.save_thread = ThreadPoolExecutor(max_workers=max_save_workers)
        for _ in range(max_save_workers):
            self.save_thread.submit(self.save_thread_func)

    def save(self, data: Union[List[torch.FloatTensor], torch.FloatTensor], path: str) -> None:
        if isinstance(data, torch.Tensor):
            data = data.to("cpu", dtype=self.save_dtype).clone()
        elif isinstance(data, list):
            for i in range(len(data)):
                if isinstance(data[i], torch.Tensor):
                    data[i] = data[i].to("cpu", dtype=self.save_dtype).clone()
        torch.cpu.synchronize()
        if self.save_queue_lenght > self.max_save_queue_lenght:
            print(f"Hit the max save queue lenght of {self.max_save_queue_lenght}. Sleeping for 10 seconds")
            time.sleep(10)
            gc.collect()
        self.save_queue.put((data,path))
        self.save_queue_lenght += 1

    @torch.no_grad()
    def save_thread_func(self) -> None:
        while self.keep_saving:
            if not self.save_queue.empty():
                data = self.save_queue.get()
                self.save_to_file(data[0], data[1])
                self.save_queue_lenght -= 1
            else:
                time.sleep(0.25)
        print("Stopping the save backend threads")

    @staticmethod
    def save_to_file(data: torch.FloatTensor, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.cpu.synchronize()
        data_is_nan = bool((isinstance(data, torch.Tensor) and data.isnan().any()) or (isinstance(data, list) and any(tensor.isnan().any() for tensor in data)))
        if data_is_nan:
            print("NaN found in:", path)
        else:
            output_data_container = BytesIO()
            torch.save(data, output_data_container)
            output_data_container.seek(0)
            compressed_data = brotli.compress(output_data_container.getvalue(), quality=10)
            with open(path, "wb") as file:
                file.write(compressed_data)


class ImageBackend():
    def __init__(self, batches: List[Tuple[List[str], str]], load_queue_lenght: int = 32, max_load_workers: int = 8):
        self.load_queue_lenght = 0
        self.keep_loading = True
        self.batches = Queue()
        for batch in batches:
            self.batches.put(batch)
        self.max_load_queue_lenght = load_queue_lenght
        self.load_queue = Queue()
        self.load_thread = ThreadPoolExecutor(max_workers=max_load_workers)
        for _ in range(max_load_workers):
            self.load_thread.submit(self.load_thread_func)

    def get_images(self) -> List[Tuple[Image.Image, str]]:
        result = self.load_queue.get()
        self.load_queue_lenght -= 1
        return result

    @torch.no_grad()
    def load_thread_func(self) -> None:
        while self.keep_loading:
            if self.load_queue_lenght >= self.max_load_queue_lenght:
                time.sleep(0.25)
            elif not self.batches.empty():
                batches = self.batches.get()
                current_batch = []
                resolution = batches[1]
                for batch in batches[0]:
                    current_batch.append((load_image_from_file(batch, resolution), batch))
                self.load_queue.put(current_batch)
                self.load_queue_lenght += 1
            else:
                time.sleep(5)
        print("Stopping the image loader threads")


class SaveImageBackend():
    def __init__(self, save_queue_lenght: int = 4096, max_save_workers: int = 8, lossless: bool = True, quality: int = 100):
        self.lossless = lossless
        self.quality = quality
        self.save_queue_lenght = 0
        self.keep_saving = True
        self.max_save_queue_lenght = save_queue_lenght
        self.save_queue = Queue()
        self.save_thread = ThreadPoolExecutor(max_workers=max_save_workers)
        for _ in range(max_save_workers):
            self.save_thread.submit(self.save_thread_func)

    def save(self, data: Image.Image, path: str) -> None:
        if self.save_queue_lenght > self.max_save_queue_lenght:
            print(f"Hit the max image save queue lenght of {self.max_save_queue_lenght}. Sleeping for 10 seconds")
            time.sleep(10)
            gc.collect()
        self.save_queue.put((data,path))
        self.save_queue_lenght += 1

    @torch.no_grad()
    def save_thread_func(self) -> None:
        while self.keep_saving:
            if not self.save_queue.empty():
                data = self.save_queue.get()
                self.save_to_file(data[0], data[1])
                self.save_queue_lenght -= 1
            else:
                time.sleep(0.25)
        print("Stopping the save backend threads")

    def save_to_file(self, image: Image.Image, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path, lossless=self.lossless, quality=self.quality)
        image.close()
