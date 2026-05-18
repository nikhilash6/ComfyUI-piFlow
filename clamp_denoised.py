class ClampDenoised:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "vae": ("VAE",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "LakonLab"

    def patch(self, model, vae):
        model = model.clone()

        def post_cfg_function(args):
            denoised = args["denoised"]
            if denoised.ndim != 4:
                return denoised
            dtype = denoised.dtype
            image = vae.decode(denoised).clamp(0, 1)
            return vae.encode(image).to(device=denoised.device, dtype=dtype)

        model.set_model_sampler_post_cfg_function(post_cfg_function)
        return (model,)
