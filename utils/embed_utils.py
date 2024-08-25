import torch
import diffusers
from utils.models import sd3_utils


def get_embed_encoder(model_type, path, device, dtype, dynamo_backend):
    if model_type == "sd3":
        return get_sd3_embed_encoder(path, device, dtype, dynamo_backend)
    else:
        raise NotImplementedError

def encode_embeds(embed_encoder, device, model_type, texts):
    with torch.no_grad():
        #return [torch.zeros((len(texts), 333, 4096)), torch.zeros((len(texts), 2048))]
        if model_type == "sd3":
            return encode_sd3_embeds(embed_encoder, device, texts)
        else:
            raise NotImplementedError


def get_sd3_embed_encoder(path, device, dtype, dynamo_backend):
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
    return [[text_encoder, text_encoder_2, text_encoder_3], [tokenizer, tokenizer_2, tokenizer_3]]


def encode_sd3_embeds(embed_encoders, device, texts):
    prompt_embeds, pooled_prompt_embeds = sd3_utils.encode_sd3_prompt(embed_encoders[0], embed_encoders[1], texts, device=device, return_masked_embed=True)
    embeds = []
    for i in range(len(prompt_embeds)):
        embeds.append([prompt_embeds[i], pooled_prompt_embeds[i]])
    return embeds