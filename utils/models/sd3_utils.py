import torch


def _encode_sd3_prompt_with_t5(text_encoder, tokenizer, prompt=None, device=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    attention_mask = text_inputs.attention_mask.to(device)
    return prompt_embeds * attention_mask.unsqueeze(-1).expand(prompt_embeds.shape)


def _encode_sd3_prompt_with_clip(text_encoder, tokenizer, prompt=None, device=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    return prompt_embeds, pooled_prompt_embeds


def encode_sd3_prompt(text_encoders, tokenizers, prompt=None, device=None, no_clip=True):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        clip_tokenizers = tokenizers[:2]
        clip_text_encoders = text_encoders[:2]

        clip_prompt_embeds_list = []
        clip_pooled_prompt_embeds_list = []
        for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
            prompt_embeds, pooled_prompt_embeds = _encode_sd3_prompt_with_clip(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
            )
            clip_prompt_embeds_list.append(prompt_embeds)
            clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

        clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

        t5_prompt_embed = _encode_sd3_prompt_with_t5(
            text_encoders[-1],
            tokenizers[-1],
            prompt=prompt,
            device=device,
        )

        if no_clip:
            prompt_embeds = t5_prompt_embed
        else:
            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds,
                (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
            )
            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

        return prompt_embeds, pooled_prompt_embeds
