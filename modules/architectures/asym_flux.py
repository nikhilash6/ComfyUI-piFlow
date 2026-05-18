import torch
from einops import rearrange
from comfy.ldm.flux.model import Flux
from .asymflow import AsymFlowMixin


class AsymFlux(AsymFlowMixin, Flux):
    def __init__(
        self,
        *args,
        base_rank=128,
        sigma_min=1e-4,
        num_timesteps=1,
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ):
        super().__init__(*args, dtype=dtype, device=device, operations=operations, **kwargs)
        self.base_rank = base_rank
        self.sigma_min = sigma_min
        self.num_timesteps = num_timesteps
        self.init_asymflow_buffers(self.in_channels, self.base_rank)

    def _forward(
        self,
        x,
        timestep,
        context,
        y=None,
        guidance=None,
        ref_latents=None,
        control=None,
        transformer_options={},
        **kwargs,
    ):
        bs, c, h_orig, w_orig = x.shape
        patch_size = self.patch_size

        h_len = ((h_orig + (patch_size // 2)) // patch_size)
        w_len = ((w_orig + (patch_size // 2)) // patch_size)
        img, img_ids = self.process_img(x, transformer_options=transformer_options)
        img_tokens = img.shape[1]
        x_t_packed = img

        calibration = self.asymflow_calibration(timestep, bs, x_t_packed.ndim)
        img = x_t_packed * calibration.k.to(x_t_packed.dtype)
        model_timestep = calibration.timestep

        timestep_zero_index = None
        if ref_latents is not None:
            ref_num_tokens = []
            h = 0
            w = 0
            index = 0
            ref_latents_method = kwargs.get("ref_latents_method", self.params.default_ref_method)
            timestep_zero = ref_latents_method == "index_timestep_zero"
            for ref in ref_latents:
                if ref_latents_method in ("index", "index_timestep_zero"):
                    index += self.params.ref_index_scale
                    h_offset = 0
                    w_offset = 0
                elif ref_latents_method == "uxo":
                    index = 0
                    h_offset = h_len * patch_size + h
                    w_offset = w_len * patch_size + w
                    h += ref.shape[-2]
                    w += ref.shape[-1]
                else:
                    index = 1
                    h_offset = 0
                    w_offset = 0
                    if ref.shape[-2] + h > ref.shape[-1] + w:
                        w_offset = w
                    else:
                        h_offset = h
                    h = max(h, ref.shape[-2] + h_offset)
                    w = max(w, ref.shape[-1] + w_offset)

                kontext, kontext_ids = self.process_img(
                    ref, index=index, h_offset=h_offset, w_offset=w_offset,
                    transformer_options=transformer_options)
                img = torch.cat([img, kontext / calibration.s.to(kontext.dtype)], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)
                ref_num_tokens.append(kontext.shape[1])
            if timestep_zero and index > 0:
                model_timestep = torch.cat([calibration.timestep, calibration.timestep * 0], dim=0)
                timestep_zero_index = [[img_tokens, img_ids.shape[1]]]
            transformer_options = transformer_options.copy()
            transformer_options["reference_image_num_tokens"] = ref_num_tokens

        txt_ids = torch.zeros((bs, context.shape[1], len(self.params.axes_dim)), device=x.device, dtype=torch.float32)
        if len(self.params.txt_ids_dims) > 0:
            for i in self.params.txt_ids_dims:
                txt_ids[:, :, i] = torch.linspace(
                    0, context.shape[1] - 1, steps=context.shape[1],
                    device=x.device, dtype=torch.float32)

        u_a_packed = self.forward_orig(
            img, img_ids, context, txt_ids, model_timestep, y, guidance, control,
            timestep_zero_index=timestep_zero_index, transformer_options=transformer_options,
            attn_mask=kwargs.get("attention_mask", None))
        u_a_packed = u_a_packed[:, :img_tokens]
        output_packed = self.asymflow_velocity(u_a_packed, x_t_packed, calibration)
        return rearrange(
            output_packed.to(u_a_packed.dtype),
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h_len, w=w_len, c=c, ph=patch_size, pw=patch_size
        )[:, :, :h_orig, :w_orig]
