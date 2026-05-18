import torch


class OklabColorEncoder:

    lrgb_to_lms = torch.tensor([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ], dtype=torch.float32)
    lms_to_oklab = torch.tensor([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ], dtype=torch.float32)
    oklab_to_lms = torch.linalg.inv(lms_to_oklab)
    lms_to_lrgb = torch.linalg.inv(lrgb_to_lms)

    def __init__(self, use_affine_norm=True, mean=(0.56, 0.0, 0.01), std=0.16):
        self.use_affine_norm = use_affine_norm
        self.mean = mean
        self.std = std

    @staticmethod
    def srgb_to_lrgb(srgb):
        a = 0.055
        return torch.where(srgb <= 0.04045, srgb / 12.92, ((srgb + a) / (1 + a)) ** 2.4)

    @staticmethod
    def lrgb_to_srgb(lrgb):
        lrgb = lrgb.clamp(min=0)
        a = 0.055
        return torch.where(lrgb <= 0.0031308, lrgb * 12.92, (1 + a) * (lrgb ** (1 / 2.4)) - a)

    @classmethod
    def _matrix(cls, name, device):
        return getattr(cls, name).to(device=device)

    def _mean_std(self, tensor):
        n_dim = tensor.dim() - 2
        mean = torch.tensor(self.mean, device=tensor.device, dtype=torch.float32).reshape(-1, *([1] * n_dim))
        std = torch.tensor(self.std, device=tensor.device, dtype=torch.float32).reshape(*([1] * tensor.dim()))
        return mean, std

    def encode(self, image):
        image = image.movedim(-1, 1).float()
        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False):
            lrgb = self.srgb_to_lrgb(image)
            lrgb_to_lms = self._matrix("lrgb_to_lms", image.device)
            lms_to_oklab = self._matrix("lms_to_oklab", image.device)
            lms = torch.einsum("ij,bj...->bi...", lrgb_to_lms, lrgb).clamp(min=0)
            oklab = torch.einsum("ij,bj...->bi...", lms_to_oklab, lms.pow(1 / 3))
            if self.use_affine_norm:
                mean, std = self._mean_std(oklab)
                oklab = (oklab - mean) / std
            return oklab.to(dtype=image.dtype)

    def decode(self, oklab):
        dtype = oklab.dtype
        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False):
            oklab = oklab.float()
            if self.use_affine_norm:
                mean, std = self._mean_std(oklab)
                oklab = oklab * std + mean
            oklab_to_lms = self._matrix("oklab_to_lms", oklab.device)
            lms_to_lrgb = self._matrix("lms_to_lrgb", oklab.device)
            lms = torch.einsum("ij,bj...->bi...", oklab_to_lms, oklab).pow(3)
            lrgb = torch.einsum("ij,bj...->bi...", lms_to_lrgb, lms).clamp(0, 1)
            image = self.lrgb_to_srgb(lrgb).clamp(0, 1).to(dtype=dtype)
            return image.movedim(1, -1)


class OklabColorEncoderNode:

    DESCRIPTION = "Creates an Oklab color encoder/decoder compatible with ComfyUI VAE Encode and VAE Decode nodes."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "use_affine_norm": ("BOOLEAN", {"default": True}),
                "mean_l": ("FLOAT", {"default": 0.56, "min": -10.0, "max": 10.0, "step": 0.001}),
                "mean_a": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "mean_b": ("FLOAT", {"default": 0.01, "min": -10.0, "max": 10.0, "step": 0.001}),
                "std": ("FLOAT", {"default": 0.16, "min": 0.0001, "max": 10.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "make_vae"
    CATEGORY = "LakonLab"

    def make_vae(self, use_affine_norm=True, mean_l=0.56, mean_a=0.0, mean_b=0.01, std=0.16):
        return (OklabColorEncoder(
            use_affine_norm=use_affine_norm,
            mean=(mean_l, mean_a, mean_b),
            std=std,
        ),)
