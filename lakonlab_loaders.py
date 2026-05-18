import os
import sys
import importlib
import importlib.util
import importlib.machinery
from dataclasses import dataclass
import torch
import comfy
import folder_paths
from comfy.utils import swap_scale_shift
from .modules.model_detection import model_config_from_piflow, model_config_from_asymflow
from .modules.loader_utils import (
    flux_norm_target_suffix,
    load_lakonlab_model_from_files,
    load_lakonlab_model_from_gguf,
)


old_flux_to_diffusers = comfy.utils.flux_to_diffusers


def flux_to_diffusers(mmdit_config, output_prefix=""):
    # Todo: need a better way to determine Flux.1 vs Flux.2
    if mmdit_config.get('image_model', 'flux') in ('flux', 'gm_flux'):
        return old_flux_to_diffusers(mmdit_config, output_prefix=output_prefix)

    n_double_layers = mmdit_config.get("depth", 0)
    n_single_layers = mmdit_config.get("depth_single_blocks", 0)
    hidden_size = mmdit_config.get("hidden_size", 0)

    key_map = {}
    norm_suffix = flux_norm_target_suffix()

    # --- double blocks: diffusers transformer_blocks.{i} -> comfy double_blocks.{i} ---
    for index in range(n_double_layers):
        prefix_from = f"transformer_blocks.{index}"
        prefix_to = f"{output_prefix}double_blocks.{index}"

        # q/k/v for image stream packed into img_attn.qkv.weight
        k_attn = f"{prefix_from}.attn."
        qkv_img = f"{prefix_to}.img_attn.qkv.weight"
        key_map[f"{k_attn}to_q.weight"] = (qkv_img, (0, 0, hidden_size))
        key_map[f"{k_attn}to_k.weight"] = (qkv_img, (0, hidden_size, hidden_size))
        key_map[f"{k_attn}to_v.weight"] = (qkv_img, (0, hidden_size * 2, hidden_size))

        # q/k/v for text(additional) stream packed into txt_attn.qkv.weight
        qkv_txt = f"{prefix_to}.txt_attn.qkv.weight"
        key_map[f"{k_attn}add_q_proj.weight"] = (qkv_txt, (0, 0, hidden_size))
        key_map[f"{k_attn}add_k_proj.weight"] = (qkv_txt, (0, hidden_size, hidden_size))
        key_map[f"{k_attn}add_v_proj.weight"] = (qkv_txt, (0, hidden_size * 2, hidden_size))

        # the rest are mostly 1:1 renames
        block_map = {
            # attn proj
            "attn.to_out.0.weight": "img_attn.proj.weight",
            "attn.to_add_out.weight": "txt_attn.proj.weight",

            # mlps
            "ff.linear_in.weight": "img_mlp.0.weight",
            "ff.linear_out.weight": "img_mlp.2.weight",
            "ff_context.linear_in.weight": "txt_mlp.0.weight",
            "ff_context.linear_out.weight": "txt_mlp.2.weight",

            # norms
            "attn.norm_q.weight": f"img_attn.norm.query_norm.{norm_suffix}",
            "attn.norm_k.weight": f"img_attn.norm.key_norm.{norm_suffix}",
            "attn.norm_added_q.weight": f"txt_attn.norm.query_norm.{norm_suffix}",
            "attn.norm_added_k.weight": f"txt_attn.norm.key_norm.{norm_suffix}",
        }
        for k_from, k_to in block_map.items():
            key_map[f"{prefix_from}.{k_from}"] = f"{prefix_to}.{k_to}"

    # --- single blocks: diffusers single_transformer_blocks.{i} -> comfy single_blocks.{i} ---
    for index in range(n_single_layers):
        prefix_from = f"single_transformer_blocks.{index}"
        prefix_to = f"{output_prefix}single_blocks.{index}"

        # Flux.2 diffusers already fuses (qkv + mlp_in) into one big mat:
        #   to_qkv_mlp_proj.weight [55296, 6144]  <->  comfy linear1.weight [55296, 6144]
        # and attn.to_out.weight [6144, 24576]   <->  comfy linear2.weight [6144, 24576]
        key_map[f"{prefix_from}.attn.to_qkv_mlp_proj.weight"] = f"{prefix_to}.linear1.weight"
        key_map[f"{prefix_from}.attn.to_out.weight"] = f"{prefix_to}.linear2.weight"

        # norms
        key_map[f"{prefix_from}.attn.norm_q.weight"] = f"{prefix_to}.norm.query_norm.{norm_suffix}"
        key_map[f"{prefix_from}.attn.norm_k.weight"] = f"{prefix_to}.norm.key_norm.{norm_suffix}"

    # --- top-level modules ---
    MAP_BASIC = {
        # embeds
        ("img_in.weight", "x_embedder.weight"),
        ("txt_in.weight", "context_embedder.weight"),

        # time + guidance (Flux.2 diffusers name)
        ("time_in.in_layer.weight", "time_guidance_embed.timestep_embedder.linear_1.weight"),
        ("time_in.out_layer.weight", "time_guidance_embed.timestep_embedder.linear_2.weight"),
        ("guidance_in.in_layer.weight", "time_guidance_embed.guidance_embedder.linear_1.weight"),
        ("guidance_in.out_layer.weight", "time_guidance_embed.guidance_embedder.linear_2.weight"),

        # stream modulation
        ("double_stream_modulation_img.lin.weight", "double_stream_modulation_img.linear.weight"),
        ("double_stream_modulation_txt.lin.weight", "double_stream_modulation_txt.linear.weight"),
        ("single_stream_modulation.lin.weight", "single_stream_modulation.linear.weight"),

        # output head
        ("final_layer.linear.weight", "proj_out.weight"),
        ("final_layer.adaLN_modulation.1.weight", "norm_out.linear.weight", swap_scale_shift),
    }

    for item in MAP_BASIC:
        if len(item) == 3:
            comfy_k, diffusers_k, fn = item
            key_map[diffusers_k] = (f"{output_prefix}{comfy_k}", None, fn)
        else:
            comfy_k, diffusers_k = item
            key_map[diffusers_k] = f"{output_prefix}{comfy_k}"

    return key_map


comfy.utils.flux_to_diffusers = flux_to_diffusers


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


@dataclass(frozen=True)
class LakonLabModelRegistry:
    key: str
    model_config_factory: object
    error_label: str


MODEL_CONFIG_REGISTRIES = {
    "piflow": LakonLabModelRegistry(
        key="piflow",
        model_config_factory=model_config_from_piflow,
        error_label="piflow",
    ),
    "asymflow": LakonLabModelRegistry(
        key="asymflow",
        model_config_factory=model_config_from_asymflow,
        error_label="asymflow",
    ),
}


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

    CATEGORY = "LakonLab"

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
        registry = MODEL_CONFIG_REGISTRIES["piflow"]
        model = load_lakonlab_model_from_files(
            base_model_path, adapter_path,
            model_options=model_options, adapter_strength=adapter_strength,
            model_config_factory=registry.model_config_factory,
            error_label=registry.error_label)

        return (model,)


class AsymFlowLoader:

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
    FUNCTION = "load_asymflow"

    CATEGORY = "LakonLab"

    def load_asymflow(self, model_name, weight_dtype, adapter_name=None, adapter_strength=1.0):
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
        registry = MODEL_CONFIG_REGISTRIES["asymflow"]
        model = load_lakonlab_model_from_files(
            base_model_path, adapter_path,
            model_options=model_options, adapter_strength=adapter_strength,
            model_config_factory=registry.model_config_factory,
            error_label=registry.error_label)

        return (model,)


if _nodes is not None:
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

        CATEGORY = "LakonLab"

        def load_piflow_gguf(
                self, model_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None,
                adapter_name=None, adapter_strength=1.0):

            base_model_path = folder_paths.get_full_path_or_raise("unet", model_name)
            if adapter_name is not None:
                adapter_path = folder_paths.get_full_path_or_raise("loras", adapter_name)
            else:
                adapter_path = None

            registry = MODEL_CONFIG_REGISTRIES["piflow"]
            model = load_lakonlab_model_from_gguf(
                base_model_path, adapter_path, adapter_strength=adapter_strength,
                dequant_dtype=dequant_dtype, patch_dtype=patch_dtype,
                patch_on_device=patch_on_device,
                model_config_factory=registry.model_config_factory,
                error_label=registry.error_label,
                gguf_model_patcher=GGUFModelPatcher,
                gguf_sd_loader=gguf_sd_loader,
                ggml_ops_class=GGMLOps)

            return (model,)

    class AsymFlowLoaderGGUF:

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
        FUNCTION = "load_asymflow_gguf"

        CATEGORY = "LakonLab"

        def load_asymflow_gguf(
                self, model_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None,
                adapter_name=None, adapter_strength=1.0):

            base_model_path = folder_paths.get_full_path_or_raise("unet", model_name)
            if adapter_name is not None:
                adapter_path = folder_paths.get_full_path_or_raise("loras", adapter_name)
            else:
                adapter_path = None

            registry = MODEL_CONFIG_REGISTRIES["asymflow"]
            model = load_lakonlab_model_from_gguf(
                base_model_path, adapter_path, adapter_strength=adapter_strength,
                dequant_dtype=dequant_dtype, patch_dtype=patch_dtype,
                patch_on_device=patch_on_device,
                model_config_factory=registry.model_config_factory,
                error_label=registry.error_label,
                gguf_model_patcher=GGUFModelPatcher,
                gguf_sd_loader=gguf_sd_loader,
                ggml_ops_class=GGMLOps)

            return (model,)

else:
    PiFlowLoaderGGUF = None
    AsymFlowLoaderGGUF = None
