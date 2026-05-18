import math

import torch
import nodes


def asymflux2_sqrt_shift(seq_len: int) -> float:
    base_seq_len = 1024 ** 2
    max_seq_len = 2048 ** 2
    base_shift = 17.0
    max_shift = 34.0

    slope = (max_shift - base_shift) / (math.sqrt(max_seq_len) - math.sqrt(base_seq_len))
    return (math.sqrt(seq_len) - math.sqrt(base_seq_len)) * slope + base_shift


def get_asymflux2_schedule(num_steps: int, image_seq_len: int) -> torch.Tensor:
    shift = asymflux2_sqrt_shift(image_seq_len)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)[:-1]
    sigmas = shift * timesteps / (1.0 + (shift - 1.0) * timesteps)
    return torch.cat([sigmas, sigmas.new_zeros(1)])


class AsymFlux2Scheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 38, "min": 1, "max": 4096}),
                "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 1}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "LakonLab"

    def get_sigmas(self, steps, width, height):
        return (get_asymflux2_schedule(steps, width * height),)
