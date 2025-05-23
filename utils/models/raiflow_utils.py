import torch

from typing import List, Optional, Union
from transformers import PreTrainedModel, PreTrainedTokenizer
from diffusers.image_processor import PipelineImageInput


def encode_raiflow_prompt(
    text_encoder: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: Union[str, List[str]],
    prompt_images: Optional[PipelineImageInput] = None,
    device: Optional[torch.device] = None,
    max_sequence_length: int = 1024,
) -> List[torch.FloatTensor]:
    device = device or text_encoder.device

    if prompt_images is None and (prompt == "" or prompt == [""]):
        # encoding the empty embed via the text encoder is the same as using zeros
        return [torch.zeros((1, text_encoder.config.hidden_size), device=device, dtype=text_encoder.dtype)]

    prompt = [prompt] if isinstance(prompt, str) else prompt

    if prompt_images is not None and not isinstance(prompt_images, list):
        prompt_images = [prompt_images]

    inputs = tokenizer(
        text=prompt.copy(), # tokenizer overwrites
        images=prompt_images,
        padding="longest",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_embeds = text_encoder(**inputs, output_hidden_states=True).hidden_states[-2]
    prompt_embeds = prompt_embeds.to(device, dtype=text_encoder.dtype)

    attention_mask = inputs["attention_mask"].to(device, dtype=text_encoder.dtype)
    prompt_embeds = prompt_embeds * attention_mask.unsqueeze(-1).expand(prompt_embeds.shape)

    prompt_embeds_list = []
    for i in range(prompt_embeds.size(0)):
        count = 0
        for j in reversed(attention_mask[i]):
            if j == 0:
                break
            count += 1
        count = max(count,1)
        prompt_embeds_list.append(prompt_embeds[i, -count:])

    return prompt_embeds_list
