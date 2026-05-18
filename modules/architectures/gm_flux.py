import torch
import torch.nn as nn
from comfy.ldm.flux.layers import apply_mod, timestep_embedding
from comfy.ldm.flux.model import Flux, invert_slices


class LastLayer(nn.Module):
    """Modulation only, no linear projection to out_channels."""

    def __init__(self, hidden_size: int, bias=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final = operations.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(hidden_size, 2 * hidden_size, bias=bias, dtype=dtype, device=device)
        )

    def forward(self, x, vec, modulation_dims=None):
        if vec.ndim == 2:
            vec = vec[:, None, :]

        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        x = apply_mod(self.norm_final(x), (1 + scale), shift, modulation_dims)
        return x


class GMFlux(Flux):
    def __init__(
            self,
            *args,
            num_gaussians=8,
            constant_logstd=None,
            logstd_inner_dim=1024,
            gm_num_logstd_layers=2,
            final_layer=True,
            dtype=None,
            device=None,
            operations=None,
            **kwargs):
        super().__init__(
            *args, final_layer=False, dtype=dtype, device=device, operations=operations, **kwargs)

        if final_layer:
            self.num_gaussians = num_gaussians
            self.constant_logstd = constant_logstd
            self.logstd_inner_dim = logstd_inner_dim
            self.gm_num_logstd_layers = gm_num_logstd_layers

            self.final_layer = LastLayer(
                self.hidden_size,
                bias=getattr(self.params, 'ops_bias', True),
                dtype=dtype,
                device=device,
                operations=operations)
            self.proj_out_means = operations.Linear(
                self.hidden_size, self.num_gaussians * self.out_channels,
                bias=True, device=device)
            self.proj_out_logweights = operations.Linear(
                self.hidden_size, self.num_gaussians * self.patch_size * self.patch_size,
                bias=True, device=device)
            self.constant_logstd = constant_logstd

            if self.constant_logstd is None:
                assert gm_num_logstd_layers >= 1
                in_dim = self.hidden_size
                logstd_layers = []
                for _ in range(gm_num_logstd_layers - 1):
                    logstd_layers.extend([
                        nn.SiLU(),
                        operations.Linear(in_dim, logstd_inner_dim, bias=True, device=device)])
                    in_dim = logstd_inner_dim
                self.proj_out_logstds = nn.Sequential(
                    *logstd_layers,
                    nn.SiLU(),
                    operations.Linear(in_dim, 1, bias=True, device=device))

    def forward_orig(
            self,
            img,
            img_ids,
            txt,
            txt_ids,
            timesteps,
            y,
            guidance=None,
            control=None,
            timestep_zero_index=None,
            transformer_options={},
            attn_mask=None):

        transformer_options = transformer_options.copy()
        patches = transformer_options.get("patches", {})
        patches_replace = transformer_options.get("patches_replace", {})
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is not None:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        if self.vector_in is not None:
            if y is None:
                y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)
            vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])

        if self.txt_norm is not None:
            txt = self.txt_norm(txt)
        txt = self.txt_in(txt)

        if "post_input" in patches:
            for p in patches["post_input"]:
                out = p({"img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids, "transformer_options": transformer_options})
                img = out["img"]
                txt = out["txt"]
                img_ids = out["img_ids"]
                txt_ids = out["txt_ids"]

        if img_ids is not None:
            ids = torch.cat((txt_ids, img_ids), dim=1)
            pe = self.pe_embedder(ids)
        else:
            pe = None

        vec_orig = vec
        txt_vec = vec
        extra_kwargs = {}
        if timestep_zero_index is not None:
            modulation_dims = []
            batch = vec.shape[0] // 2
            vec_orig = vec_orig.reshape(2, batch, vec.shape[1]).movedim(0, 1)
            invert = invert_slices(timestep_zero_index, img.shape[1])
            for s in invert:
                modulation_dims.append((s[0], s[1], 0))
            for s in timestep_zero_index:
                modulation_dims.append((s[0], s[1], 1))
            extra_kwargs["modulation_dims_img"] = modulation_dims
            txt_vec = vec[:batch]

        if getattr(self.params, 'global_modulation', False):
            vec = (self.double_stream_modulation_img(vec_orig), self.double_stream_modulation_txt(txt_vec))

        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(self.double_blocks)
        transformer_options["block_type"] = "double"
        for i, block in enumerate(self.double_blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"],
                                                   txt=args["txt"],
                                                   vec=args["vec"],
                                                   pe=args["pe"],
                                                   attn_mask=args.get("attn_mask"),
                                                   transformer_options=args.get("transformer_options"),
                                                   **extra_kwargs)
                    return out

                out = blocks_replace[("double_block", i)]({"img": img,
                                                           "txt": txt,
                                                           "vec": vec,
                                                           "pe": pe,
                                                           "attn_mask": attn_mask,
                                                           "transformer_options": transformer_options},
                                                          {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img,
                                 txt=txt,
                                 vec=vec,
                                 pe=pe,
                                 attn_mask=attn_mask,
                                 transformer_options=transformer_options,
                                 **extra_kwargs)

            if control is not None:  # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img[:, :add.shape[1]] += add

        if img.dtype == torch.float16:
            img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

        img = torch.cat((txt, img), 1)

        if getattr(self.params, 'global_modulation', False):
            vec, _ = self.single_stream_modulation(vec_orig)

        extra_kwargs = {}
        if timestep_zero_index is not None:
            modulation_dims_combined = list(map(lambda x: (0 if x[0] == 0 else x[0] + txt.shape[1], x[1] + txt.shape[1], x[2]), modulation_dims))
            extra_kwargs["modulation_dims"] = modulation_dims_combined

        transformer_options["total_blocks"] = len(self.single_blocks)
        transformer_options["block_type"] = "single"
        transformer_options["img_slice"] = [txt.shape[1], img.shape[1]]
        for i, block in enumerate(self.single_blocks):
            transformer_options["block_index"] = i
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"],
                                       vec=args["vec"],
                                       pe=args["pe"],
                                       attn_mask=args.get("attn_mask"),
                                       transformer_options=args.get("transformer_options"),
                                       **extra_kwargs)
                    return out

                out = blocks_replace[("single_block", i)]({"img": img,
                                                           "vec": vec,
                                                           "pe": pe,
                                                           "attn_mask": attn_mask,
                                                           "transformer_options": transformer_options},
                                                          {"original_block": block_wrap})
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, transformer_options=transformer_options, **extra_kwargs)

            if control is not None:  # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1]:txt.shape[1] + add.shape[1], ...] += add

        img = img[:, txt.shape[1]:, ...]

        extra_kwargs = {}
        if timestep_zero_index is not None:
            extra_kwargs["modulation_dims"] = modulation_dims

        hidden_states = self.final_layer(img, vec_orig, **extra_kwargs)
        vec_out = vec_orig[:, 0] if vec_orig.ndim == 3 else vec_orig
        return hidden_states, vec_out

    def _forward(self, x, timestep, context, y=None, guidance=None, ref_latents=None, control=None,
                 transformer_options={}, **kwargs):
        bs, c, h_orig, w_orig = x.shape
        patch_size = self.patch_size

        h_len = ((h_orig + (patch_size // 2)) // patch_size)
        w_len = ((w_orig + (patch_size // 2)) // patch_size)
        img, img_ids = self.process_img(x, transformer_options=transformer_options)
        img_tokens = img.shape[1]

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

                kontext, kontext_ids = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset, transformer_options=transformer_options)
                img = torch.cat([img, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)
                ref_num_tokens.append(kontext.shape[1])
            if timestep_zero and index > 0:
                timestep = torch.cat([timestep, timestep * 0], dim=0)
                timestep_zero_index = [[img_tokens, img_ids.shape[1]]]
            transformer_options = transformer_options.copy()
            transformer_options["reference_image_num_tokens"] = ref_num_tokens

        txt_ids = torch.zeros((bs, context.shape[1], len(self.params.axes_dim)), device=x.device, dtype=torch.float32)

        if len(self.params.txt_ids_dims) > 0:
            for i in self.params.txt_ids_dims:
                txt_ids[:, :, i] = torch.linspace(0, context.shape[1] - 1, steps=context.shape[1], device=x.device, dtype=torch.float32)

        hidden_states, vec = self.forward_orig(
            img, img_ids, context, txt_ids, timestep, y, guidance, control,
            timestep_zero_index=timestep_zero_index, transformer_options=transformer_options,
            attn_mask=kwargs.get("attention_mask", None))
        hidden_states = hidden_states[:, :img_tokens].to(self.proj_out_means.weight.dtype)

        bs = hidden_states.size(0)
        k = self.num_gaussians
        c = self.out_channels
        out_means = self.proj_out_means(hidden_states).reshape(
            bs, h_len, w_len, k, c // (patch_size * patch_size), patch_size, patch_size
        ).permute(
            0, 3, 4, 1, 5, 2, 6
        ).reshape(
            bs, k, c // (patch_size * patch_size), h_len * patch_size, w_len * patch_size
        )[..., :h_orig, :w_orig]
        out_logweights = self.proj_out_logweights(hidden_states).reshape(
            bs, h_len, w_len, k, 1, patch_size, patch_size
        ).permute(
            0, 3, 4, 1, 5, 2, 6
        ).reshape(
            bs, k, 1, h_len * patch_size, w_len * patch_size
        )[..., :x.shape[-2], :x.shape[-1]].log_softmax(dim=1)
        if self.constant_logstd is None:
            out_logstds = self.proj_out_logstds(
                vec.detach().to(self.proj_out_logstds[-1].weight.dtype)).reshape(bs, 1, 1, 1, 1)
        else:
            out_logstds = hidden_states.new_full((bs, 1, 1, 1, 1), float(self.constant_logstd))

        return dict(
            means=out_means,
            logweights=out_logweights,
            logstds=out_logstds)
