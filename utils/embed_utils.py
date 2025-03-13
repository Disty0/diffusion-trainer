import torch
import diffusers
from PIL import Image
from utils.models import sd3_utils, sotev3_utils

from typing import List, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer


def get_embed_encoder(model_type: str, path: str, device: torch.device, dtype: torch.dtype, dynamo_backend: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    if model_type == "sd3":
        return get_sd3_embed_encoder(path, device, dtype, dynamo_backend)
    elif model_type == "sotev3":
        return get_sotev3_embed_encoder(path, device, dtype, dynamo_backend)
    else:
        raise NotImplementedError


def encode_embeds(embed_encoder: Tuple[PreTrainedModel, PreTrainedTokenizer], device: torch.device, model_type: str, texts: List[str], prompt_images=List[Image.Image]) -> torch.FloatTensor:
    with torch.no_grad():
        #return [torch.zeros((len(texts), 333, 4096)), torch.zeros((len(texts), 2048))]
        if model_type == "sd3":
            return encode_sd3_embeds(embed_encoder, device, texts)
        elif model_type == "sotev3":
            return encode_sotev3_embeds(embed_encoder, device, texts, prompt_images=prompt_images)
        else:
            raise NotImplementedError


def encode_sd3_embeds(embed_encoders: Tuple[Tuple[PreTrainedModel], Tuple[PreTrainedTokenizer]], device: torch.device, texts: List[str]) -> List[List[torch.FloatTensor]]:
    prompt_embeds, pooled_prompt_embeds = sd3_utils.encode_sd3_prompt(embed_encoders[0], embed_encoders[1], texts, device=device, no_clip=True)
    embeds = []
    for i in range(len(prompt_embeds)):
        embeds.append([prompt_embeds[i], pooled_prompt_embeds[i]])
    return embeds

def get_sd3_embed_encoder(path: str, device: torch.device, dtype: torch.dtype, dynamo_backend: str) -> Tuple[Tuple[PreTrainedModel], Tuple[PreTrainedTokenizer]]:
    pipe = diffusers.AutoPipelineForText2Image.from_pretrained(path, transformer=None, vae=None, torch_dtype=dtype)
    text_encoder = pipe.text_encoder.to(device, dtype=dtype).eval()
    text_encoder_2 = pipe.text_encoder_2.to(device, dtype=dtype).eval()
    text_encoder_3 = pipe.text_encoder_3.to(device, dtype=dtype).eval()
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    text_encoder_3.requires_grad_(False)
    if dynamo_backend != "no":
        text_encoder = torch.compile(text_encoder, backend=dynamo_backend)
        text_encoder_2 = torch.compile(text_encoder_2, backend=dynamo_backend)
        text_encoder_3 = torch.compile(text_encoder_3, backend=dynamo_backend)
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    tokenizer_3 = pipe.tokenizer_3
    return ((text_encoder, text_encoder_2, text_encoder_3), (tokenizer, tokenizer_2, tokenizer_3))


def encode_sotev3_embeds(embed_encoders: Tuple[PreTrainedModel, PreTrainedTokenizer], device: torch.device, texts: List[str], prompt_images=List[Image.Image]) -> List[torch.FloatTensor]:
    return sotev3_utils.encode_sotev3_prompt(embed_encoders[0], embed_encoders[1], prompt=texts, prompt_images=prompt_images, device=device)

def get_sotev3_embed_encoder(path: str, device: torch.device, dtype: torch.dtype, dynamo_backend: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    from sotev3 import SoteDiffusionV3Pipeline
    pipe = SoteDiffusionV3Pipeline.from_pretrained(path, transformer=None, vae=None, torch_dtype=dtype)
    text_encoder = pipe.text_encoder.to(device, dtype=dtype).eval()
    text_encoder.requires_grad_(False)
    if dynamo_backend != "no":
        text_encoder = torch.compile(text_encoder, backend=dynamo_backend)
    tokenizer = pipe.tokenizer
    return (text_encoder, tokenizer)
