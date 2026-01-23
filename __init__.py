from .piflow_loader import PiFlowLoader, PiFlowLoaderGGUF
from .piflow_sampler import PiFlowSampler
from .model_sampling_piflow import ModelSamplingPiFlow


NODE_CLASS_MAPPINGS = {
    "Load pi-Flow Model": PiFlowLoader,
    "pi-Flow Sampler": PiFlowSampler,
    "ModelSamplingPiFlow": ModelSamplingPiFlow,
}

if PiFlowLoaderGGUF is not None:
    NODE_CLASS_MAPPINGS["Load pi-Flow Model (GGUF)"] = PiFlowLoaderGGUF


__all__ = ['NODE_CLASS_MAPPINGS']
