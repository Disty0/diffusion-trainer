from typing import Optional, Tuple
import random
import torch


def get_meanflow_target(target: torch.FloatTensor, sigmas: torch.FloatTensor, sigmas_next: torch.FloatTensor, jvp_out: torch.FloatTensor) -> torch.FloatTensor:
    with torch.no_grad():
        return torch.addcmul(target.to(dtype=torch.float32), (sigmas_next.to(dtype=torch.float32) - sigmas.to(dtype=torch.float32)), jvp_out.to(dtype=torch.float32)).detach()


def get_flowmatch_inputs(
    latents: torch.FloatTensor,
    device: torch.device,
    sampler_config: dict,
    num_train_timesteps: int = 1000,
    meanflow: bool = False,
    noise: Optional[torch.FloatTensor] = None,
) -> Tuple[torch.FloatTensor]:
    shape = (2, latents.shape[0]) if meanflow else (latents.shape[0],)

    if sampler_config["weighting_scheme"] == "uniform":
        # uniform range is larger than 1.0 to hit the timestep 1000 more
        u = torch.empty(shape, device=device, dtype=torch.float32).uniform_(0.0, 1.0056)
    elif sampler_config["weighting_scheme"] in {"logit_normal", "lognorm"}:
        u = torch.normal(sampler_config["logit_mean"], sampler_config["logit_std"], shape, device=device, dtype=torch.float32).sigmoid_()
    elif sampler_config["weighting_scheme"] == "mode":
        u = torch.rand(shape, device=device, dtype=torch.float32)
        # u = 1 - u - mode_scale * (torch.cos(torch.pi * u / 2) ** 2 - 1 + u)
        u = (1 - u).sub_((torch.cos(u.mul(torch.pi / 2)).square_().add_(-1).add_(u)).mul_(sampler_config["mode_scale"]))
    else:
        raise NotImplementedError(f'weighting_scheme type {sampler_config["weighting_scheme"]} is not implemented')

    if meanflow:
        sigmas_next = torch.amin(u, dim=0).clamp_(0, 1.0 - 1/num_train_timesteps).view(-1, 1, 1, 1)
        u = torch.amax(u, dim=0)

    if sampler_config["shift"] != 0:
        # u = (u * shift) / (1 + (shift - 1) * u)
        u = torch.mul(u, sampler_config["shift"]).div_(u.mul_((sampler_config["shift"] - 1)).add_(1))

    u = u.clamp_(1/num_train_timesteps,1.0)
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


def get_loss_weighting(loss_weighting, model_pred, target, sigmas):
    if loss_weighting == "none":
        return model_pred, target
    elif loss_weighting == "sigma_sqrt_clamp":
        weight = sigmas.sqrt().clamp(min=0.1, max=None)
    elif loss_weighting == "sigma_sqrt":
        weight = sigmas.sqrt()
    elif loss_weighting == "cosmap":
        # weighting = 2 / (torch.pi * (1 - (2 * sigmas) + 2 * sigmas**2))
        double_sigmas = 2 * sigmas
        weight = (2 / torch.pi) / (1 - double_sigmas + (double_sigmas * sigmas))
    else:
        raise NotImplementedError(f'loss_weighting type {loss_weighting} is not implemented')
    return torch.mul(model_pred, weight), torch.mul(target, weight)


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
            mask.append(torch.randint(random.randint(config["mask_low_rate"],0), random.randint(2,config["mask_high_rate"]), (height, width), device=device).to(dtype=torch.float32).clamp(0,1))

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
    #model_x0_pred = noisy_model_input.to(dtype=torch.float32) - (model_pred * sigmas)
    model_x0_pred = torch.addcmul(noisy_model_input.to(dtype=torch.float32), model_pred, -sigmas)

    # new_noisy_model_input = ((1.0 - sigmas) * model_x0_pred) + (sigmas * noise)
    new_noisy_model_input = torch.addcmul(torch.mul(sigmas, noise), torch.sub(1.0, sigmas), model_x0_pred)
    # new_target = target + ((new_noisy_model_input - noisy_model_input) / sigmas)
    new_target = torch.addcdiv(target, torch.sub(new_noisy_model_input, noisy_model_input), sigmas)

    self_correct_count = new_noisy_model_input.shape[0]
    return new_noisy_model_input, new_target, self_correct_count
