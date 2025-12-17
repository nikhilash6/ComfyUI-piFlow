import os
import sys
import importlib
import importlib.util
import importlib.machinery
import logging
import torch
import comfy
import folder_paths
from comfy import model_management
from comfy.model_detection import unet_prefix_from_state_dict, convert_diffusers_mmdit, detect_unet_config
from .modules.model_detection import model_config_from_piflow
try:
    from comfy.utils import convert_old_quants
except ImportError:
    convert_old_quants = None


def import_comfyui_gguf_nodes():
    """
    Import custom_nodes/ComfyUI-GGUF/nodes.py as a proper package module so that
    relative imports inside it (e.g. from .ops import ...) work.

    Returns the imported nodes module, or None if ComfyUI-GGUF is not present.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    custom_nodes_dir = os.path.abspath(os.path.join(here, ".."))  # .../custom_nodes
    gguf_dir = os.path.join(custom_nodes_dir, "ComfyUI-GGUF")
    nodes_py = os.path.join(gguf_dir, "nodes.py")

    if not os.path.isfile(nodes_py):
        return None

    pkg_name = "comfyui_gguf"  # safe alias (valid identifier)
    mod_name = f"{pkg_name}.nodes"  # import as a submodule of that package

    # 1) Ensure the parent package exists (with a __path__)
    if pkg_name not in sys.modules:
        pkg = importlib.util.module_from_spec(
            importlib.machinery.ModuleSpec(pkg_name, loader=None, is_package=True)
        )
        pkg.__path__ = [gguf_dir]  # where to find ops.py, nodes.py, etc.
        sys.modules[pkg_name] = pkg

    # 2) Import nodes as pkg submodule (so __package__ is set correctly)
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    spec = importlib.util.spec_from_file_location(mod_name, nodes_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


_nodes = import_comfyui_gguf_nodes()
if _nodes is None:
    GGUFModelPatcher = gguf_sd_loader = GGMLOps = None
else:
    GGUFModelPatcher = _nodes.GGUFModelPatcher
    gguf_sd_loader = _nodes.gguf_sd_loader
    GGMLOps = _nodes.GGMLOps


def convert_diffusers_to_comfyui(state_dict, diffusers_weight, comfy_weight_map):
    """Modified from convert_diffusers_mmdit

    Note: This is an in-place operation. Tensors in state_dict and diffusers_weight may be updated in place.
    """
    if isinstance(comfy_weight_map, str):
        comfy_weight_key = comfy_weight_map
        state_dict[comfy_weight_key] = diffusers_weight
    else:
        comfy_weight_key = comfy_weight_map[0]
        if len(comfy_weight_map) > 2:
            weight_convert_fun = comfy_weight_map[2]
        else:
            weight_convert_fun = lambda a: a
        offset = comfy_weight_map[1]
        if offset is not None:
            updated_weight = state_dict.get(comfy_weight_key, None)
            if updated_weight is None:
                updated_weight = diffusers_weight
            if updated_weight.shape[offset[0]] < offset[1] + offset[2]:
                expanded_shape = list(diffusers_weight.shape)
                expanded_shape[offset[0]] = offset[1] + offset[2]
                expanded_weight = torch.empty(
                    expanded_shape, device=diffusers_weight.device, dtype=diffusers_weight.dtype)
                _updated_weight = expanded_weight.narrow(offset[0], 0, updated_weight.shape[offset[0]])
                _updated_weight[:] = updated_weight
                updated_weight = expanded_weight
            target_slice = updated_weight.narrow(offset[0], offset[1], offset[2])
        else:
            target_slice = updated_weight = diffusers_weight
        target_slice[:] = weight_convert_fun(diffusers_weight)
        state_dict[comfy_weight_key] = updated_weight
    return comfy_weight_key


def load_piflow_model_state_dict(
        base_model_sd, adapter_sd=None, model_options={},
        base_metadata=None, adapter_metadata=None):
    if base_metadata is None:
        base_metadata = {}
    if adapter_metadata is None:
        adapter_metadata = {}

    # prepare base_model_sd first
    diffusion_model_prefix = unet_prefix_from_state_dict(base_model_sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(base_model_sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        base_model_sd = temp_sd
    base_unet_config = detect_unet_config(base_model_sd, "", metadata=base_metadata)
    if base_unet_config is None:
        base_model_sd = convert_diffusers_mmdit(base_model_sd, "")
        base_unet_config = detect_unet_config(base_model_sd, "", metadata=base_metadata)
        if base_unet_config is None:
            return None, None

    weight_dtype = comfy.utils.weight_dtype(base_model_sd)

    metadata = base_metadata.copy()
    new_sd = base_model_sd.copy()
    lora_sd = {}

    if adapter_sd is not None:
        updated_weight_layers = set()
        updated_keys = set()

        key_mapping = {}
        base_image_model = base_unet_config["image_model"]
        if base_image_model == 'flux':  # requires conversion
            key_mapping = comfy.utils.flux_to_diffusers(base_unet_config, output_prefix="")

        for k in adapter_sd.keys():
            if "lora" in k:
                if base_image_model == 'flux' and not k.startswith("transformer."):
                    lora_sd["transformer." + k] = adapter_sd[k]
                else:
                    lora_sd[k] = adapter_sd[k]
            else:
                if k in key_mapping:  # convert_diffusers_mmdit
                    comfy_weight_key = convert_diffusers_to_comfyui(new_sd, adapter_sd[k], key_mapping[k])
                else:
                    new_sd[k] = adapter_sd[k]
                    comfy_weight_key = k
                updated_keys.add(comfy_weight_key)
                if comfy_weight_key.endswith(".weight"):
                    layer_name = comfy_weight_key[:-7]
                    updated_weight_layers.add(layer_name)

        # unset scales in the base model for updated layers
        for layer in updated_weight_layers:
            for scale_postfix in ["scale_input", "scale_weight"]:
                scale_key = '.'.join([layer, scale_postfix])
                if scale_key in new_sd and scale_key not in updated_keys:
                    del new_sd[scale_key]

        metadata.update(adapter_metadata)

    parameters = comfy.utils.calculate_parameters(new_sd)

    if convert_old_quants is not None:
        if model_options.get("custom_operations", None) is None:
            new_sd, metadata = convert_old_quants(
                new_sd, "", metadata=metadata)

    model_config = model_config_from_piflow(new_sd, "", metadata=metadata)
    if model_config is None:
        return None, None

    offload_device = model_management.unet_offload_device()
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if hasattr(model_config, 'quant_config'):
        has_quant_config = model_config.quant_config is not None
        if has_quant_config:
            weight_dtype = None
    else:
        has_quant_config = getattr(model_config, 'layer_quant_config', None) is not None
        if getattr(model_config, 'scaled_fp8', None) is not None:
            weight_dtype = None

    dtype = model_options.get("dtype", None)

    if dtype is None:
        unet_dtype = model_management.unet_dtype(
            model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype)
    else:
        unet_dtype = dtype

    load_device = model_management.get_torch_device()

    if has_quant_config:
        manual_cast_dtype = model_management.unet_manual_cast(
            None, load_device, model_config.supported_inference_dtypes)
    else:
        manual_cast_dtype = model_management.unet_manual_cast(
            unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model_config.custom_operations = model_options.get("custom_operations", model_config.custom_operations)
    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    model = model_config.get_model(new_sd, "")
    model = model.to(offload_device)
    model.load_model_weights(new_sd, "")
    left_over = new_sd.keys()
    if len(left_over) > 0:
        logging.info("left over keys in diffusion model: {}".format(left_over))
    model = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)

    return model, lora_sd


def load_piflow_model(base_model_path, adapter_path, model_options={}, adapter_strength=1.0):
    base_model_sd, base_metadata = comfy.utils.load_torch_file(base_model_path, return_metadata=True)
    adapter_sd = adapter_metadata = None
    if adapter_path is not None:
        adapter_sd, adapter_metadata = comfy.utils.load_torch_file(adapter_path, return_metadata=True)
    model, lora_sd = load_piflow_model_state_dict(
        base_model_sd, adapter_sd=adapter_sd, model_options=model_options,
        base_metadata=base_metadata, adapter_metadata=adapter_metadata)
    if model is None:
        logging.error("ERROR UNSUPPORTED PIFLOW MODEL")
        raise RuntimeError("ERROR: Could not detect model type of: {}\n".format(base_model_path))
    if len(lora_sd) > 0:
        model, _ = comfy.sd.load_lora_for_models(model, None, lora_sd, adapter_strength, None)
    return model


def load_piflow_model_gguf(
        base_model_path, adapter_path, model_options={}, adapter_strength=1.0,
        dequant_dtype=None, patch_dtype=None, patch_on_device=None):
    if GGUFModelPatcher is None:
        raise RuntimeError(
            "ComfyUI-GGUF not found. Please install the ComfyUI-GGUF custom nodes to enable GGUF loading.")

    ops = GGMLOps()

    if dequant_dtype in ("default", None):
        ops.Linear.dequant_dtype = None
    elif dequant_dtype in ["target"]:
        ops.Linear.dequant_dtype = dequant_dtype
    else:
        ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)

    if patch_dtype in ("default", None):
        ops.Linear.patch_dtype = None
    elif patch_dtype in ["target"]:
        ops.Linear.patch_dtype = patch_dtype
    else:
        ops.Linear.patch_dtype = getattr(torch, patch_dtype)

    # Todo: load metadata (policy_config) from GGUF?
    base_model_sd = gguf_sd_loader(base_model_path)
    model_options.update(custom_operations=ops)

    adapter_sd = adapter_metadata = None
    if adapter_path is not None:
        adapter_sd, adapter_metadata = comfy.utils.load_torch_file(adapter_path, return_metadata=True)
    model, lora_sd = load_piflow_model_state_dict(
        base_model_sd, adapter_sd=adapter_sd, model_options=model_options,
        adapter_metadata=adapter_metadata)
    if model is None:
        logging.error("ERROR UNSUPPORTED PIFLOW MODEL")
        raise RuntimeError("ERROR: Could not detect model type of: {}\n".format(base_model_path))

    model = GGUFModelPatcher.clone(model)
    model.patch_on_device = patch_on_device

    if len(lora_sd) > 0:
        model, _ = comfy.sd.load_lora_for_models(model, None, lora_sd, adapter_strength, None)

    return model


class PiFlowLoader:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],)
            },
            "optional": {
                "adapter_name": (folder_paths.get_filename_list("loras"), {"default": None}),
                "adapter_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_piflow"

    CATEGORY = "piflow"

    def load_piflow(self, model_name, weight_dtype, adapter_name=None, adapter_strength=1.0):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        base_model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        if adapter_name is not None:
            adapter_path = folder_paths.get_full_path_or_raise("loras", adapter_name)
        else:
            adapter_path = None
        model = load_piflow_model(
            base_model_path, adapter_path,
            model_options=model_options, adapter_strength=adapter_strength)

        return (model,)


class PiFlowLoaderGGUF:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("unet_gguf"),),
            },
            "optional": {
                "dequant_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_on_device": ("BOOLEAN", {"default": False}),
                "adapter_name": (folder_paths.get_filename_list("loras"), {"default": None}),
                "adapter_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_piflow_gguf"

    CATEGORY = "piflow"

    def load_piflow_gguf(
            self, model_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None,
            adapter_name=None, adapter_strength=1.0):

        base_model_path = folder_paths.get_full_path_or_raise("unet", model_name)
        if adapter_name is not None:
            adapter_path = folder_paths.get_full_path_or_raise("loras", adapter_name)
        else:
            adapter_path = None

        model = load_piflow_model_gguf(
            base_model_path, adapter_path, adapter_strength=adapter_strength,
            dequant_dtype=dequant_dtype, patch_dtype=patch_dtype, patch_on_device=patch_on_device)

        return (model,)
