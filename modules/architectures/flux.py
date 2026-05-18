import inspect
import torch
from einops import rearrange
from comfy.ldm.flux.model import Flux


class FluxMod(Flux):

    def _forward(self, x, timestep, context, y=None, guidance=None, ref_latents=None, control=None,
                 transformer_options={}, **kwargs):
        bs, c, h_orig, w_orig = x.shape
        patch_size = self.patch_size

        h_len = ((h_orig + (patch_size // 2)) // patch_size)
        w_len = ((w_orig + (patch_size // 2)) // patch_size)
        if 'transformer_options' in inspect.signature(self.process_img).parameters:
            img, img_ids = self.process_img(x, transformer_options=transformer_options)
        else:  # fallback for older versions
            img, img_ids = self.process_img(x)
        img_tokens = img.shape[1]
        if ref_latents is not None:
            h = 0
            w = 0
            index = 0
            ref_latents_method = kwargs.get(
                "ref_latents_method",
                getattr(self.params, 'default_ref_method', 'offset')
            )
            for ref in ref_latents:
                if ref_latents_method == "index":
                    index += getattr(self.params, 'ref_index_scale', 1)
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

                if 'transformer_options' in inspect.signature(self.process_img).parameters:
                    kontext, kontext_ids = self.process_img(
                        ref, index=index, h_offset=h_offset, w_offset=w_offset,
                        transformer_options=transformer_options)
                else:  # fallback for older versions
                    kontext, kontext_ids = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset)
                img = torch.cat([img, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)

        txt_ids = torch.zeros((bs, context.shape[1], len(self.params.axes_dim)), device=x.device, dtype=torch.float32)

        txt_ids_dims = getattr(self.params, "txt_ids_dims", [3] if len(self.params.axes_dim) == 4 else [])
        for i in txt_ids_dims:
            txt_ids[:, :, i] = torch.linspace(
                0, context.shape[1] - 1, steps=context.shape[1], device=x.device, dtype=torch.float32)

        out = self.forward_orig(
            img, img_ids, context, txt_ids, timestep, y, guidance, control,
            transformer_options=transformer_options, attn_mask=kwargs.get("attention_mask", None))
        out = out[:, :img_tokens]
        return rearrange(
            out, "b (h w) (n c ph pw) -> b n c (h ph) (w pw)",
            h=h_len, w=w_len, c=c, ph=self.patch_size, pw=self.patch_size
        )[..., :h_orig, :w_orig]
