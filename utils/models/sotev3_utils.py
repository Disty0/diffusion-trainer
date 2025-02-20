def encode_sotev3_prompt(
    text_encoder, tokenizer,
    prompt, prompt_images=None,
    max_sequence_length=1024,
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
        padding="longest",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_embeds = text_encoder(**inputs, output_hidden_states=True).hidden_states[-2]
    prompt_embeds = prompt_embeds.to(device, dtype=dtype)

    attention_mask = inputs["attention_mask"].to(device, dtype=dtype)
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
