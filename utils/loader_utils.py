import os
import gc
import math
import brotli
import random
import time
import torch
import numpy as np

import pillow_jxl # noqa: F401
from PIL import Image
from io import BytesIO
from queue import Queue
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor

Image.MAX_IMAGE_PIXELS = 999999999 # 178956970


def load_from_file(path):
    with open(path, "rb") as file:
        data = file.read()
    decompressed_data = brotli.decompress(data)
    stored_tensor = BytesIO(decompressed_data)
    stored_tensor.seek(0)
    return torch.load(stored_tensor, map_location="cpu", weights_only=True)


def load_image_from_file(image_path, target_size):
        if isinstance(target_size, str):
            target_size = target_size.split("x")
            target_size[0] = int(target_size[0])
            target_size[1] = int(target_size[1])

        image = Image.open(image_path)
        background = Image.new('RGBA', image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image.convert("RGBA")).convert("RGB")

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

        return [image, image_path]


class LatentsAndEmbedsDataset(Dataset):
    def __init__(self, batches):
        self.batches = batches
    def __len__(self):
        return len(self.batches)
    def __getitem__(self, index):
        latents = []
        embeds = []
        for batch in self.batches[index]:
            latents.append(load_from_file(batch[0]))
            embeds.append(load_from_file(batch[1]))
        return [latents, embeds]


class LatentsAndImagesDataset(Dataset):
    def __init__(self, batches, image_processor):
        self.batches = batches
        self.image_processor = image_processor
    def __len__(self):
        return len(self.batches)
    def __getitem__(self, index):
        latents = []
        image_tensors = []
        for batch in self.batches[index]:
            latents.append(load_from_file(batch[0]))
            with Image.open(batch[1]) as image:
                image_tensors.append(self.image_processor.preprocess(image)[0])
        return [latents, image_tensors]


class ImagesAndEmbedsDataset(Dataset):
    def __init__(self, batches):
        self.batches = batches
    def __len__(self):
        return len(self.batches)
    def __getitem__(self, index):
        images = []
        embeds = []
        resoluion = self.batches[index][0]
        for batch in self.batches[index][1]:
            image_tensor = torch.from_numpy(np.asarray(load_image_from_file(batch[0], resoluion)[0]).copy()).transpose(2,0).transpose(1,2)
            image_tensor = ((image_tensor.float() / 255) - 0.5) * 2 # -1 to 1 range
            images.append(image_tensor)
            embeds.append(load_from_file(batch[1]))
        return [images, embeds]


class DCTsAndEmbedsDataset(Dataset):
    def __init__(self, batches, image_encoder):
        self.batches = batches
        self.image_encoder = image_encoder
    def __len__(self):
        return len(self.batches)
    def __getitem__(self, index):
        images = []
        embeds = []
        resoluion = self.batches[index][0]
        for batch in self.batches[index][1]:
            images.append(self.image_encoder.encode(load_image_from_file(batch[0], resoluion)[0], device="cpu")[0])
            embeds.append(load_from_file(batch[1]))
        return [images, embeds]


class SaveBackend():
    def __init__(self, model_type, save_queue_lenght=4096, max_save_workers=8):
        self.save_queue_lenght = 0
        self.model_type = model_type
        self.keep_saving = True
        self.max_save_queue_lenght = save_queue_lenght
        self.save_queue = Queue()
        self.save_thread = ThreadPoolExecutor(max_workers=max_save_workers)
        for _ in range(max_save_workers):
            self.save_thread.submit(self.save_thread_func)


    def save(self, data, path):
        if isinstance(data, torch.Tensor):
            data = data.to("cpu", dtype=torch.float16).clone()
        elif isinstance(data, list):
            for i in range(len(data)):
                if isinstance(data[i], torch.Tensor):
                    data[i] = data[i].to("cpu", dtype=torch.float16).clone()
        torch.cpu.synchronize()
        if self.save_queue_lenght > self.max_save_queue_lenght:
            print(f"Hit the max save queue lenght of {self.max_save_queue_lenght}. Sleeping for 10 seconds")
            time.sleep(10)
            gc.collect()
        self.save_queue.put([data,path])
        self.save_queue_lenght += 1


    def save_thread_func(self):
        while self.keep_saving:
            if not self.save_queue.empty():
                data = self.save_queue.get()
                self.save_to_file(data[0], data[1])
                self.save_queue_lenght -= 1
            else:
                time.sleep(0.25)
        print("Stopping the save backend threads")


    def save_to_file(self, data, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.cpu.synchronize()
        output_data_container = BytesIO()
        torch.save(data, output_data_container)
        output_data_container.seek(0)
        compressed_data = brotli.compress(output_data_container.getvalue(), quality=10)
        with open(path, "wb") as file:
            file.write(compressed_data)


class ImageBackend():
    def __init__(self, batches, load_queue_lenght=32, max_load_workers=8):
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


    def get_images(self):
        result = self.load_queue.get()
        self.load_queue_lenght -= 1
        return result


    def load_thread_func(self):
        while self.keep_loading:
            if self.load_queue_lenght >= self.max_load_queue_lenght:
                time.sleep(0.25)
            elif not self.batches.empty():
                batches = self.batches.get()
                curren_batch = []
                for batch in batches[0]:
                    curren_batch.append(load_image_from_file(batch, batches[1]))
                self.load_queue.put(curren_batch)
                self.load_queue_lenght += 1
            else:
                time.sleep(5)
        print("Stopping the image loader threads")


class SaveImageBackend():
    def __init__(self, save_queue_lenght=4096, max_save_workers=8, lossless=True, quality=100):
        self.lossless = lossless
        self.quality = quality
        self.save_queue_lenght = 0
        self.keep_saving = True
        self.max_save_queue_lenght = save_queue_lenght
        self.save_queue = Queue()
        self.save_thread = ThreadPoolExecutor(max_workers=max_save_workers)
        for _ in range(max_save_workers):
            self.save_thread.submit(self.save_thread_func)


    def save(self, data, path):
        if self.save_queue_lenght > self.max_save_queue_lenght:
            print(f"Hit the max image save queue lenght of {self.max_save_queue_lenght}. Sleeping for 10 seconds")
            time.sleep(10)
            gc.collect()
        self.save_queue.put([data,path])
        self.save_queue_lenght += 1


    def save_thread_func(self):
        while self.keep_saving:
            if not self.save_queue.empty():
                data = self.save_queue.get()
                self.save_to_file(data[0], data[1])
                self.save_queue_lenght -= 1
            else:
                time.sleep(0.25)
        print("Stopping the save backend threads")


    def save_to_file(self, image, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path, lossless=self.lossless, quality=self.quality)
        image.close()
