import os
import brotli
import random
import time
import torch
import pillow_jxl
from PIL import Image
from io import BytesIO
from queue import Queue
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor


def load_from_file(path):
    with open(path, "rb") as file:
        data = file.read()
    decompressed_data = brotli.decompress(data)
    stored_tensor = BytesIO(decompressed_data)
    stored_tensor.seek(0)
    return torch.load(stored_tensor, map_location="cpu", weights_only=True)


class LatentAndEmbedsDataset(Dataset):
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


class SaveBackend():
    def __init__(self, model_type, max_save_workers=8):
        self.model_type = model_type
        self.keep_saving = True
        self.save_queue = Queue()
        self.save_thread = ThreadPoolExecutor(max_workers=max_save_workers)
        for _ in range(max_save_workers):
            self.save_thread.submit(self.save_thread_func)


    def save(self, data, path):
        self.save_queue.put([data,path])


    def save_thread_func(self):
        while self.keep_saving:
            if not self.save_queue.empty():
                data = self.save_queue.get()
                self.save_to_file(data[0], data[1])
            else:
                time.sleep(0.1)
        print("Stopping the save backend threads")
        return


    def save_to_file(self, data, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(data, torch.Tensor):
            data = data.to("cpu", torch.float16).clone()
        elif isinstance(data, list):
            for i in range(len(data)):
                if isinstance(data[i], torch.Tensor):
                    data[i] = data[i].to("cpu", torch.float16).clone()
        output_data_container = BytesIO()
        torch.save(data, output_data_container)
        output_data_container.seek(0)
        compressed_data = brotli.compress(output_data_container.getvalue(), quality=10)
        with open(path, "wb") as file:
            file.write(compressed_data)


class ImageBackend():
    def __init__(self, batches, load_queue_lenght=32, max_load_workers=8):
        self.keep_loading = True
        self.batches = Queue()
        for batch in batches:
            self.batches.put(batch)
        self.load_queue_lenght = load_queue_lenght
        self.load_queue = Queue()
        self.load_thread = ThreadPoolExecutor(max_workers=max_load_workers)
        for _ in range(max_load_workers):
            self.load_thread.submit(self.load_thread_func)


    def get_images(self):
        return self.load_queue.get()


    def load_thread_func(self):
        while self.keep_loading:
            if self.load_queue.qsize() >= self.load_queue_lenght:
                time.sleep(0.1)
            elif not self.batches.empty():
                batches = self.batches.get()
                curren_batch = []
                for batch in batches[0]:
                    curren_batch.append(self.load_from_file(batch, batches[1]))
                self.load_queue.put(curren_batch)
        print("Stopping the image loader threads")
        return


    def load_from_file(self, image_path, target_size):
        if isinstance(target_size, str):
            target_size = target_size.split("x")
            target_size[0] = int(target_size[0])
            target_size[1] = int(target_size[1])

        image = Image.open(image_path)

        orig_size = image.size
        new_size = [int(target_size[1] * orig_size[0] / orig_size[1]), int(target_size[0] * orig_size[1] / orig_size[0])]
        if new_size[0] > target_size[0]:
            image = image.resize((new_size[0], target_size[1]), Image.LANCZOS)
        else:
            image = image.resize((target_size[0], new_size[1]), Image.LANCZOS)

        new_size = image.size
        left_diff = new_size[0] - target_size[0]
        if left_diff < 0: # can go off by 1 or 2 pixel
            image = image.resize((target_size[0], new_size[1]), Image.LANCZOS)
            new_size = image.size
            left_diff = new_size[0] - target_size[0]

        top_diff = (new_size[1] - target_size[1])
        if top_diff < 0: # can go off by 1 or 2 pixel
            image = image.resize((new_size[0], target_size[1]), Image.LANCZOS)
            new_size = image.size
            top_diff = (new_size[1] - target_size[1])

        left_shift = random.randint(0, left_diff)
        top_shift = random.randint(0, top_diff)
        image = image.crop((left_shift, top_shift, (left_shift + target_size[0]), (top_shift + target_size[1])))

        new_size = image.size
        if new_size[0] != target_size[0] or new_size[1] != target_size[1]: # sanity check
            image = image.resize((target_size[0], target_size[1]), Image.LANCZOS)

        return [image, image_path]
