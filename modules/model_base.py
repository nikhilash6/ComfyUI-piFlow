import torch
import comfy
from functools import partial
from enum import Enum
from comfy.model_base import BaseModel, convert_tensor, QwenImage as _QwenImage, Flux as _Flux
from .piflow_policies import POLICY_CLASSES
from . import architectures


class ModelType(Enum):
    PIFLOW = 1


class ModelSamplingPiFlow(torch.nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}

        self.set_parameters(
            shift=sampling_settings.get("shift", 3.2),
            multiplier=sampling_settings.get("multiplier", 1.0))

    def set_parameters(self, shift=3.2, multiplier=1.0):
        self.shift = shift
        self.multiplier = multiplier

    def timestep(self, sigma):
        return sigma * self.multiplier

    def warp_t(self, t):
        shift = self.shift
        return shift * t / (1 + (shift - 1) * t)

    def unwarp_t(self, t):
        shift = self.shift
        return t / (shift + (1 - shift) * t)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return self.warp_t(1.0 - percent)


# stash original
_original_model_sampling = comfy.model_base.model_sampling


def model_sampling(model_config, model_type):
    if model_type == ModelType.PIFLOW:
        c = comfy.model_sampling.CONST
        s = ModelSamplingPiFlow

        class ModelSampling(s, c):
            pass

        return ModelSampling(model_config)

    # fallback to original
    return _original_model_sampling(model_config, model_type)


# patch comfyui model_sampling
comfy.model_base.model_sampling = model_sampling


class BasePiFlow(BaseModel):

    def __init__(self, model_config, diffusion_model, model_type=ModelType.PIFLOW, device=None):
        BaseModel.__init__(
            self, model_config, model_type=model_type, device=device, unet_model=diffusion_model)

        policy_config = model_config.policy_config.copy()
        policy_type = policy_config.pop("type")
        self.policy_class = partial(POLICY_CLASSES[policy_type], **policy_config)

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        sigma = t
        xc = self.model_sampling.calculate_input(sigma, x)

        if c_concat is not None:
            xc = torch.cat([xc] + [comfy.model_management.cast_to_device(c_concat, xc.device, xc.dtype)], dim=1)

        context = c_crossattn
        dtype = self.get_dtype()

        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        xc = xc.to(dtype)
        device = xc.device
        t = self.model_sampling.timestep(t).float()
        if context is not None:
            context = comfy.model_management.cast_to_device(context, device, dtype)

        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]

            if hasattr(extra, "dtype"):
                extra = convert_tensor(extra, dtype, device)
            elif isinstance(extra, list):
                ex = []
                for ext in extra:
                    ex.append(convert_tensor(ext, dtype, device))
                extra = ex
            extra_conds[o] = extra

        t = self.process_timestep(t, x=x, **extra_conds)
        assert "latent_shapes" not in extra_conds, \
            "`pack_latents` and `unpack_latents` are currently not supported in PiFlow models."

        model_output = self.diffusion_model(xc, t, context=context, control=control,
                                            transformer_options=transformer_options, **extra_conds)
        if isinstance(model_output, dict):
            model_output = {k: v.float() for k, v in model_output.items()}
        else:
            model_output = model_output.float()
        return self.policy_class(model_output, x, sigma)


class GMQwenImage(_QwenImage, BasePiFlow):

    def __init__(self, model_config, device=None):
        BasePiFlow.__init__(self, model_config, architectures.GMQwenImageTransformer2DModel, device=device)
        self.memory_usage_factor_conds = ("ref_latents",)


class QwenImage(_QwenImage, BasePiFlow):

    def __init__(self, model_config, device=None):
        BasePiFlow.__init__(self, model_config, architectures.QwenImageTransformer2DModelMod, device=device)
        self.memory_usage_factor_conds = ("ref_latents",)


class GMFlux(_Flux, BasePiFlow):
    def __init__(self, model_config, device=None):
        BasePiFlow.__init__(self, model_config, architectures.GMFlux, device=device)
        self.memory_usage_factor_conds = ("ref_latents",)


class Flux(_Flux, BasePiFlow):
    def __init__(self, model_config, device=None):
        BasePiFlow.__init__(self, model_config, architectures.FluxMod, device=device)
        self.memory_usage_factor_conds = ("ref_latents",)
