from .piflow_loader import PiFlowLoader, PiFlowLoaderGGUF
from .piflow_sampler import PiFlowSampler
from .model_sampling_piflow import ModelSamplingPiFlow


NODE_CLASS_MAPPINGS = {
    "Load pi-Flow Model": PiFlowLoader,
    "Load pi-Flow Model (GGUF)": PiFlowLoaderGGUF,
    "pi-Flow Sampler": PiFlowSampler,
    "ModelSamplingPiFlow": ModelSamplingPiFlow,
}

__all__ = ['NODE_CLASS_MAPPINGS']
