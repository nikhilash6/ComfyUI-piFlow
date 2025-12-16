import math
import torch
import comfy
from functools import partial
from enum import Enum
from comfy.model_base import BaseModel, convert_tensor, QwenImage as _QwenImage, Flux as _Flux, utils
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


class GMQwenImage(BasePiFlow, _QwenImage):

    def __init__(self, model_config, device=None):
        super().__init__(model_config, architectures.GMQwenImageTransformer2DModel, device=device)
        self.memory_usage_factor_conds = ("ref_latents",)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        ref_latents = kwargs.get("reference_latents", None)
        if ref_latents is not None:
            latents = []
            for lat in ref_latents:
                latents.append(self.process_latent_in(lat))
            out['ref_latents'] = comfy.conds.CONDList(latents)

            ref_latents_method = kwargs.get("reference_latents_method", None)
            if ref_latents_method is not None:
                out['ref_latents_method'] = comfy.conds.CONDConstant(ref_latents_method)
        return out


class QwenImage(BasePiFlow, _QwenImage):

    def __init__(self, model_config, device=None):
        super().__init__(model_config, architectures.QwenImageTransformer2DModelMod, device=device)
        self.memory_usage_factor_conds = ("ref_latents",)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        ref_latents = kwargs.get("reference_latents", None)
        if ref_latents is not None:
            latents = []
            for lat in ref_latents:
                latents.append(self.process_latent_in(lat))
            out['ref_latents'] = comfy.conds.CONDList(latents)

            ref_latents_method = kwargs.get("reference_latents_method", None)
            if ref_latents_method is not None:
                out['ref_latents_method'] = comfy.conds.CONDConstant(ref_latents_method)
        return out


class GMFlux(BasePiFlow, _Flux):
    def __init__(self, model_config, device=None):
        super().__init__(model_config, architectures.GMFlux, device=device)
        self.memory_usage_factor_conds = ("ref_latents",)

    def concat_cond(self, **kwargs):
        try:
            # Handle Flux control loras dynamically changing the img_in weight.
            num_channels = self.diffusion_model.img_in.weight.shape[1] // (self.diffusion_model.patch_size * self.diffusion_model.patch_size)
        except:
            # Some cases like tensorrt might not have the weights accessible
            num_channels = self.model_config.unet_config["in_channels"]

        out_channels = self.model_config.unet_config["out_channels"]

        if num_channels <= out_channels:
            return None

        image = kwargs.get("concat_latent_image", None)
        noise = kwargs.get("noise", None)
        device = kwargs["device"]

        if image is None:
            image = torch.zeros_like(noise)

        image = utils.common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
        image = utils.resize_to_batch_size(image, noise.shape[0])
        image = self.process_latent_in(image)
        if num_channels <= out_channels * 2:
            return image

        # inpaint model
        mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
        if mask is None:
            mask = torch.ones_like(noise)[:, :1]

        mask = torch.mean(mask, dim=1, keepdim=True)
        mask = utils.common_upscale(mask.to(device), noise.shape[-1] * 8, noise.shape[-2] * 8, "bilinear", "center")
        mask = mask.view(mask.shape[0], mask.shape[2] // 8, 8, mask.shape[3] // 8, 8).permute(0, 2, 4, 1, 3).reshape(mask.shape[0], -1, mask.shape[2] // 8, mask.shape[3] // 8)
        mask = utils.resize_to_batch_size(mask, noise.shape[0])
        return torch.cat((image, mask), dim=1)

    def encode_adm(self, **kwargs):
        return kwargs["pooled_output"]

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        # upscale the attention mask, since now we
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            shape = kwargs["noise"].shape
            mask_ref_size = kwargs["attention_mask_img_shape"]
            # the model will pad to the patch size, and then divide
            # essentially dividing and rounding up
            (h_tok, w_tok) = (math.ceil(shape[2] / self.diffusion_model.patch_size), math.ceil(shape[3] / self.diffusion_model.patch_size))
            attention_mask = utils.upscale_dit_mask(attention_mask, mask_ref_size, (h_tok, w_tok))
            out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)

        guidance = kwargs.get("guidance", 3.5)
        if guidance is not None:
            out['guidance'] = comfy.conds.CONDRegular(torch.FloatTensor([guidance]))

        ref_latents = kwargs.get("reference_latents", None)
        if ref_latents is not None:
            latents = []
            for lat in ref_latents:
                latents.append(self.process_latent_in(lat))
            out['ref_latents'] = comfy.conds.CONDList(latents)

            ref_latents_method = kwargs.get("reference_latents_method", None)
            if ref_latents_method is not None:
                out['ref_latents_method'] = comfy.conds.CONDConstant(ref_latents_method)
        return out

    def extra_conds_shapes(self, **kwargs):
        out = {}
        ref_latents = kwargs.get("reference_latents", None)
        if ref_latents is not None:
            out['ref_latents'] = list([1, 16, sum(map(lambda a: math.prod(a.size()), ref_latents)) // 16])
        return out


class Flux(BasePiFlow, _Flux):
    def __init__(self, model_config, device=None):
        super().__init__(model_config, architectures.FluxMod, device=device)
        self.memory_usage_factor_conds = ("ref_latents",)

    def concat_cond(self, **kwargs):
        try:
            # Handle Flux control loras dynamically changing the img_in weight.
            num_channels = self.diffusion_model.img_in.weight.shape[1] // (self.diffusion_model.patch_size * self.diffusion_model.patch_size)
        except:
            # Some cases like tensorrt might not have the weights accessible
            num_channels = self.model_config.unet_config["in_channels"]

        out_channels = self.model_config.unet_config["out_channels"]

        if num_channels <= out_channels:
            return None

        image = kwargs.get("concat_latent_image", None)
        noise = kwargs.get("noise", None)
        device = kwargs["device"]

        if image is None:
            image = torch.zeros_like(noise)

        image = utils.common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
        image = utils.resize_to_batch_size(image, noise.shape[0])
        image = self.process_latent_in(image)
        if num_channels <= out_channels * 2:
            return image

        # inpaint model
        mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
        if mask is None:
            mask = torch.ones_like(noise)[:, :1]

        mask = torch.mean(mask, dim=1, keepdim=True)
        mask = utils.common_upscale(mask.to(device), noise.shape[-1] * 8, noise.shape[-2] * 8, "bilinear", "center")
        mask = mask.view(mask.shape[0], mask.shape[2] // 8, 8, mask.shape[3] // 8, 8).permute(0, 2, 4, 1, 3).reshape(mask.shape[0], -1, mask.shape[2] // 8, mask.shape[3] // 8)
        mask = utils.resize_to_batch_size(mask, noise.shape[0])
        return torch.cat((image, mask), dim=1)

    def encode_adm(self, **kwargs):
        return kwargs["pooled_output"]

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        # upscale the attention mask, since now we
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            shape = kwargs["noise"].shape
            mask_ref_size = kwargs["attention_mask_img_shape"]
            # the model will pad to the patch size, and then divide
            # essentially dividing and rounding up
            (h_tok, w_tok) = (math.ceil(shape[2] / self.diffusion_model.patch_size), math.ceil(shape[3] / self.diffusion_model.patch_size))
            attention_mask = utils.upscale_dit_mask(attention_mask, mask_ref_size, (h_tok, w_tok))
            out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)

        guidance = kwargs.get("guidance", 3.5)
        if guidance is not None:
            out['guidance'] = comfy.conds.CONDRegular(torch.FloatTensor([guidance]))

        ref_latents = kwargs.get("reference_latents", None)
        if ref_latents is not None:
            latents = []
            for lat in ref_latents:
                latents.append(self.process_latent_in(lat))
            out['ref_latents'] = comfy.conds.CONDList(latents)

            ref_latents_method = kwargs.get("reference_latents_method", None)
            if ref_latents_method is not None:
                out['ref_latents_method'] = comfy.conds.CONDConstant(ref_latents_method)
        return out

    def extra_conds_shapes(self, **kwargs):
        out = {}
        ref_latents = kwargs.get("reference_latents", None)
        if ref_latents is not None:
            out['ref_latents'] = list([1, 16, sum(map(lambda a: math.prod(a.size()), ref_latents)) // 16])
        return out
