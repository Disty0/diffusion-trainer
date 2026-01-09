import torch
from PIL import Image

from typing import List, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer


def get_embed_encoder(model_type: str, path: str, device: torch.device, dtype: torch.dtype, dynamo_backend: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    match model_type:
        case "sd3":
            from .models.sd3_utils import get_sd3_embed_encoder
            return get_sd3_embed_encoder(path, device, dtype, dynamo_backend)
        case "sdxl":
            from .models.sdxl_utils import get_sdxl_embed_encoder
            return get_sdxl_embed_encoder(path, device, dtype, dynamo_backend)
        case "raiflow":
            from .models.raiflow_utils import get_raiflow_embed_encoder
            return get_raiflow_embed_encoder(path, device, dtype, dynamo_backend)
        case "z_image":
            from .models.z_image_utils import get_z_image_embed_encoder
            return get_z_image_embed_encoder(path, device, dtype, dynamo_backend)
        case _:
            raise NotImplementedError(f"Model type {model_type} is not implemented")


def encode_embeds(embed_encoder: Tuple[PreTrainedModel, PreTrainedTokenizer], device: torch.device, model_type: str, texts: List[str], prompt_images: Optional[List[Image.Image]] = None) -> torch.FloatTensor:
    with torch.no_grad():
        match model_type:
            case "sd3":
                from .models.sd3_utils import encode_sd3_embeds
                return encode_sd3_embeds(embed_encoder, device, texts)
            case "sdxl":
                from .models.sdxl_utils import encode_sdxl_embeds
                return encode_sdxl_embeds(embed_encoder, device, texts)
            case "raiflow":
                from .models.raiflow_utils import encode_raiflow_embeds
                return encode_raiflow_embeds(embed_encoder, device, texts, prompt_images=prompt_images)
            case "z_image":
                from .models.z_image_utils import encode_z_image_embeds
                return encode_z_image_embeds(embed_encoder, device, texts)
            case _:
                raise NotImplementedError(f"Model type {model_type} is not implemented")
