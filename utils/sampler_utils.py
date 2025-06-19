from typing import Optional, Tuple
import random
import torch


def get_meanflow_target(target: torch.FloatTensor, sigmas: torch.FloatTensor, sigmas_next: torch.FloatTensor, jvp_out: torch.FloatTensor) -> torch.FloatTensor:
    with torch.no_grad():
        return (target.float() - (sigmas.float() - sigmas_next.float()) * jvp_out.float()).detach()


def get_flowmatch_inputs(
    latents: torch.FloatTensor,
    device: torch.device,
    num_train_timesteps: int = 1000,
    shift: float = 3.0,
    noise: Optional[torch.FloatTensor] = None,
    meanflow: bool = False,
) -> Tuple[torch.FloatTensor]:
    # use timestep 1000 as well for zero snr
    # torch.randn is not random so we use uniform instead
    # uniform range is larger than 1.0 to hit the timestep 1000 more
    if meanflow:
        u = torch.empty((2, latents.shape[0]), device=device, dtype=torch.float32).uniform_(0.0, 1.0056)
        sigmas_next = torch.amin(u, dim=0).clamp(0, 1.0 - 1/num_train_timesteps).view(-1, 1, 1, 1)
        u = torch.amax(u, dim=0)
    else:
        u = torch.empty((latents.shape[0],), device=device, dtype=torch.float32).uniform_(0.0, 1.0056)

    u = (u * shift) / (1 + (shift - 1) * u)
    u = u.clamp(1/num_train_timesteps,1.0)
    timesteps = torch.mul(u, num_train_timesteps)
    sigmas = u.view(-1, 1, 1, 1)
    if not meanflow:
        sigmas_next = sigmas

    if noise is None:
        noise = torch.randn_like(latents, device=device, dtype=torch.float32)
    # noisy_model_input = ((1.0 - sigmas) * latents) + (sigmas * noise)
    noisy_model_input = torch.addcmul(torch.mul(sigmas, noise), torch.sub(1.0, sigmas), latents)
    target = noise - latents

    return noisy_model_input, timesteps, target, sigmas, sigmas_next, noise


def mask_noisy_model_input(noisy_model_input: torch.FloatTensor, config: dict, device: torch.device) -> Tuple[torch.FloatTensor, int]:
    masked_count = 0
    batch_size, channels, height, width = noisy_model_input.shape
    unmask = torch.ones((height, width), device=device, dtype=torch.float32)

    mask = []
    for _ in range(batch_size):
        if random.randint(0,100) > config["mask_rate"] * 100:
            mask.append(unmask)
        else:
            masked_count += 1
            mask.append(torch.randint(random.randint(config["mask_low_rate"],0), random.randint(2,config["mask_high_rate"]), (height, width), device=device).float().clamp(0,1))

    mask = torch.stack(mask, dim=0).unsqueeze(1).to(device, dtype=torch.float32)
    mask = mask.repeat(1,channels,1,1)
    noisy_model_input = ((noisy_model_input - 1) * mask) + 1 # mask with ones

    return noisy_model_input, masked_count


def get_self_corrected_targets(
    noisy_model_input: torch.FloatTensor,
    target: torch.FloatTensor,
    sigmas: torch.FloatTensor,
    noise: torch.FloatTensor,
    model_pred: torch.FloatTensor,
    x0_pred: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, int]:
    if x0_pred:
        model_x0_pred = model_pred
    model_x0_pred = noisy_model_input.float() - (model_pred * sigmas)

    # new_noisy_model_input = ((1.0 - sigmas) * model_x0_pred) + (sigmas * noise)
    new_noisy_model_input = torch.addcmul(torch.mul(sigmas, noise), torch.sub(1.0, sigmas), model_x0_pred)
    # new_target = target + ((new_noisy_model_input - noisy_model_input) / sigmas)
    new_target = torch.addcdiv(target, torch.sub(new_noisy_model_input, noisy_model_input), sigmas)

    self_correct_count = new_noisy_model_input.shape[0]
    return new_noisy_model_input, new_target, self_correct_count
