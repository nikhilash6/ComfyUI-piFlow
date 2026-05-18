from .lakonlab_loaders import PiFlowLoader, PiFlowLoaderGGUF, AsymFlowLoader, AsymFlowLoaderGGUF
from .piflow_sampler import PiFlowSampler
from .model_sampling_piflow import ModelSamplingPiFlow
from .oklab_color_encoder import OklabColorEncoderNode
from .clamp_denoised import ClampDenoised
from .asymflux2_scheduler import AsymFlux2Scheduler


NODE_CLASS_MAPPINGS = {
    "Load pi-Flow Model": PiFlowLoader,
    "Load AsymFlow Model": AsymFlowLoader,
    "Oklab Color Encoder": OklabColorEncoderNode,
    "Clamp Denoised": ClampDenoised,
    "AsymFlux2Scheduler": AsymFlux2Scheduler,
    "pi-Flow Sampler": PiFlowSampler,
    "ModelSamplingPiFlow": ModelSamplingPiFlow,
}

if PiFlowLoaderGGUF is not None:
    NODE_CLASS_MAPPINGS["Load pi-Flow Model (GGUF)"] = PiFlowLoaderGGUF
if AsymFlowLoaderGGUF is not None:
    NODE_CLASS_MAPPINGS["Load AsymFlow Model (GGUF)"] = AsymFlowLoaderGGUF


__all__ = ['NODE_CLASS_MAPPINGS']
