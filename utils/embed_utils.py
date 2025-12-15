import torch
from PIL import Image

from typing import List, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer


def get_embed_encoder(model_type: str, path: str, device: torch.device, dtype: torch.dtype, dynamo_backend: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    if model_type == "sd3":
        from .models.sd3_utils import get_sd3_embed_encoder
        return get_sd3_embed_encoder(path, device, dtype, dynamo_backend)
    elif model_type == "raiflow":
        from .models.raiflow_utils import get_raiflow_embed_encoder
        return get_raiflow_embed_encoder(path, device, dtype, dynamo_backend)
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented")


def encode_embeds(embed_encoder: Tuple[PreTrainedModel, PreTrainedTokenizer], device: torch.device, model_type: str, texts: List[str], prompt_images: Optional[List[Image.Image]] = None) -> torch.FloatTensor:
    with torch.no_grad():
        if model_type == "sd3":
            from .models.sd3_utils import encode_sd3_embeds
            return encode_sd3_embeds(embed_encoder, device, texts)
        elif model_type == "raiflow":
            from .models.raiflow_utils import encode_raiflow_embeds
            return encode_raiflow_embeds(embed_encoder, device, texts, prompt_images=prompt_images)
        else:
            raise NotImplementedError(f"Model type {model_type} is not implemented")
