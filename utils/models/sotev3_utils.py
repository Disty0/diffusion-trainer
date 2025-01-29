def encode_sotev3_prompt(
    text_encoder, tokenizer,
    prompt, prompt_images=None,
    max_sequence_length=512,
    device=None,
    dtype=None,
):
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt

    if prompt_images is not None and not isinstance(prompt_images, list):
        prompt_images = [prompt_images]

    inputs = tokenizer(
        text=prompt.copy(), # tokenizer overwrites
        images=prompt_images,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_embeds = text_encoder(**inputs, output_hidden_states=True).hidden_states[-1]
    prompt_embeds = prompt_embeds.to(device, dtype=dtype)

    attention_mask = inputs["attention_mask"].to(device, dtype=dtype)
    prompt_embeds = prompt_embeds * attention_mask.unsqueeze(-1).expand(prompt_embeds.shape)

    return prompt_embeds
