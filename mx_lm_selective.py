import sys, os, gc, copy
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ.setdefault("TRANSFORMERS_NO_FLASH_ATTENTION", "1")
os.environ.setdefault("NVTE_FLASH_ATTN", "0")

import math, torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from ignite.engine import Engine
from ignite.metrics import Loss, Metric
from mx.specs import finalize_mx_specs
from mx.mx_ops import quantize_mx_op
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import argparse
import logging
from contextlib import nullcontext

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # bitsandbytes optional
    BitsAndBytesConfig = None

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format as TEFormat
    TE_AVAILABLE = True
except Exception:  # NVIDIA Transformer Engine optional (handles missing extension)
    te = None
    DelayedScaling = None
    TEFormat = None
    TE_AVAILABLE = False

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

_MODEL_LOAD_LOCK = threading.Lock()

LOG_FILE = "mx_selective_log.txt"
LOGGER_NAME = "mx.selective"

logger = logging.getLogger(LOGGER_NAME)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

# ==================== MX Configuration ====================
# Standard MX format: FP8 E4M3 with 32-element blocks
DEFAULT_MX_SPEC = {
    "w_elem_format": "fp8_e4m3",
    "a_elem_format": "fp8_e4m3",
    "scale_bits": 8,
    "block_size": 32,
    "custom_cuda": True,
    "quantize_backprop": False,
    "round": "even",  # Round-to-nearest-even (OCP MX v1.0 default)
}

# MX Format Variants for Testing
MX_FORMAT_PRESETS = {
    "fp8_e4m3": {
        "w_elem_format": "fp8_e4m3",
        "a_elem_format": "fp8_e4m3",
        "scale_bits": 8,
        "block_size": 32,
        "custom_cuda": True,
        "quantize_backprop": False,
        "round": "even",
    },
    "fp8_e5m2": {
        "w_elem_format": "fp8_e5m2",
        "a_elem_format": "fp8_e5m2",
        "scale_bits": 8,
        "block_size": 32,
        "custom_cuda": True,
        "quantize_backprop": False,
        "round": "even",
    },
    "fp6_e3m2": {
        "w_elem_format": "fp6_e3m2",
        "a_elem_format": "fp6_e3m2",
        "scale_bits": 8,
        "block_size": 32,
        "custom_cuda": True,
        "quantize_backprop": False,
        "round": "even",
    },
    "fp6_e2m3": {
        "w_elem_format": "fp6_e2m3",
        "a_elem_format": "fp6_e2m3",
        "scale_bits": 8,
        "block_size": 32,
        "custom_cuda": True,
        "quantize_backprop": False,
        "round": "even",
    },
    "fp4_e2m1": {
        "w_elem_format": "fp4_e2m1",
        "a_elem_format": "fp4_e2m1",
        "scale_bits": 8,
        "block_size": 32,
        "custom_cuda": True,
        "quantize_backprop": False,
        "round": "even",
    },
}

DEFAULT_MX_FORMATS = [
    "fp8_e4m3",
    "fp8_e5m2",
    "fp6_e3m2",
    "fp6_e2m3",
    "fp4_e2m1",
]

_MX_SPEC_CACHE = {}


def get_mx_spec(format_key):
    if format_key not in MX_FORMAT_PRESETS:
        raise ValueError(f"Unknown MX format preset '{format_key}'")
    if format_key not in _MX_SPEC_CACHE:
        _MX_SPEC_CACHE[format_key] = finalize_mx_specs(dict(MX_FORMAT_PRESETS[format_key]))
    return _MX_SPEC_CACHE[format_key]

NVFP8_EMULATION_PRESETS = {
    "fp8_e4m3_no_shared": {
        "w_elem_format": "fp8_e4m3",
        "a_elem_format": "fp8_e4m3",
        "scale_bits": 8,
        "block_size": 0,
        "shared_exp_method": "none",
        "custom_cuda": False,
        "quantize_backprop": False,
        "round": "even",
    },
    "fp8_e5m2_no_shared": {
        "w_elem_format": "fp8_e5m2",
        "a_elem_format": "fp8_e5m2",
        "scale_bits": 8,
        "block_size": 0,
        "shared_exp_method": "none",
        "custom_cuda": False,
        "quantize_backprop": False,
        "round": "even",
    },
}

BFP16_MX_PRESET = {
    "w_elem_format": "bfloat16",
    "a_elem_format": "bfloat16",
    "scale_bits": 8,
    "block_size": 0,  # per-tensor
    "shared_exp_method": "none",  # no shared exponent
    "custom_cuda": False,
    "quantize_backprop": False,
    "round": "even",
    "bfloat": 16,  # enable bfloat quantization in MX
}

DEFAULT_NVFP8_RECIPE = (
    DelayedScaling(fp8_format=TEFormat.E4M3, amax_history_len=16, amax_compute_algo="max")
    if TE_AVAILABLE and DelayedScaling and TEFormat
    else None
)

# ==================== Device ====================
DEFAULT_DEVICE = torch.device("cpu")

def _resolve_device(device):
    if device is None:
        return DEFAULT_DEVICE
    return device if isinstance(device, torch.device) else torch.device(device)

# ==================== Selective Linear Replacement ====================
class MxLinear(nn.Module):
    """
    MX-quantized Linear layer that wraps nn.Linear.
    Replaces only the GEMM operation with MX quantization.
    """
    def __init__(self, original_linear, mx_specs):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        self.mx_specs = mx_specs
        
    def forward(self, x):
        def _quantize_bfp16(tensor):
            block_size = self.mx_specs.get('block_size', 0) or 0
            if block_size <= 0:
                return tensor.to(torch.bfloat16).to(tensor.dtype)

            axis = -1
            orig_dtype = tensor.dtype
            work = tensor.float()
            axis = work.dim() + axis if axis < 0 else axis
            work = work.movedim(axis, -1)
            orig_shape = work.shape
            length = orig_shape[-1]
            pad = (0, (block_size - length % block_size) % block_size)
            if pad[1] > 0:
                work = F.pad(work, pad)
            blocks_height = work.shape[-1] // block_size
            blocks = work.view(*orig_shape[:-1], blocks_height, block_size)
            max_abs = blocks.abs().amax(dim=-1, keepdim=True)
            mask = max_abs > 0
            shared_exp = torch.zeros_like(max_abs)
            if mask.any():
                shared_exp[mask] = torch.floor(torch.log2(max_abs[mask]))
            scale = torch.pow(2.0, shared_exp)
            normalized = torch.where(mask, blocks / scale, blocks)
            quantized = normalized.to(torch.bfloat16).to(torch.float32)
            restored = torch.where(mask, quantized * scale, torch.zeros_like(blocks))
            restored = restored.view(*orig_shape)
            if pad[1] > 0:
                restored = restored[..., :length]
            restored = restored.movedim(-1, axis)
            return restored.to(orig_dtype)

        use_bfp16 = (
            str(self.mx_specs.get('w_elem_format')) == 'bfloat16'
            and str(self.mx_specs.get('a_elem_format')) == 'bfloat16'
        )

        if use_bfp16:
            w_mx = _quantize_bfp16(self.weight)
            x_mx = _quantize_bfp16(x)
        else:
            w_mx = quantize_mx_op(
                self.weight,
                self.mx_specs,
                elem_format=self.mx_specs['w_elem_format'],
                axes=[-1],
                round=self.mx_specs.get('round', 'even')
            )
            x_mx = quantize_mx_op(
                x,
                self.mx_specs,
                elem_format=self.mx_specs['a_elem_format'],
                axes=[-1],
                round=self.mx_specs.get('round', 'even')
            )
        # Perform matmul with quantized tensors
        out = F.linear(x_mx, w_mx, self.bias)
        return out

def replace_linear_with_mx(model, mx_specs, target_modules=None):
    """
    Recursively replace nn.Linear modules with MxLinear.
    
    Args:
        model: The model to modify
        mx_specs: MX specifications for quantization
        target_modules: List of module name patterns to replace (e.g., ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'mlp'])
                       If None, replaces ALL Linear layers
    """
    count = 0
    
    def _recursive_replace(module, prefix=""):
        nonlocal count
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            
            # Check if this Linear should be replaced
            if isinstance(child, nn.Linear):
                should_replace = target_modules is None
                if target_modules is not None:
                    # Check if any pattern matches the full module path
                    for pattern in target_modules:
                        if pattern in full_name.lower():
                            should_replace = True
                            break
                
                if should_replace:
                    # Replace with MxLinear
                    mx_linear = MxLinear(child, mx_specs)
                    setattr(module, child_name, mx_linear)
                    count += 1
                    logger.debug(f"Replaced {full_name} with MxLinear")
            else:
                # Recurse into submodules
                _recursive_replace(child, full_name)
    
    _recursive_replace(model)
    logger.info(f"Replaced {count} Linear layers with MxLinear")
    return model

# ==================== NVIDIA FP8 Replacement ====================
class NvFp8Linear(nn.Module):
    """Wrap nn.Linear with Transformer Engine FP8 Linear."""

    def __init__(self, original_linear):
        if not TE_AVAILABLE or te is None:
            raise ImportError("transformer_engine is required for NVFP8 backend")
        super().__init__()
        bias = original_linear.bias is not None
        self.linear = te.Linear(
            original_linear.in_features,
            original_linear.out_features,
            bias=bias,
            params_dtype=original_linear.weight.dtype,
        )
        # Copy weights from original linear
        with torch.no_grad():
            self.linear.weight.copy_(original_linear.weight)
            if bias:
                self.linear.bias.copy_(original_linear.bias)

    def forward(self, x):
        return self.linear(x)


def replace_linear_with_nvfp8(model, target_modules=None):
    """
    Replace selected nn.Linear modules with Transformer Engine FP8 linears.
    """

    if not TE_AVAILABLE or te is None:
        raise ImportError("transformer_engine is required for NVFP8 backend")

    count = 0

    def _recursive_replace(module, prefix=""):
        nonlocal count
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name

            if isinstance(child, nn.Linear):
                should_replace = target_modules is None
                if target_modules is not None:
                    for pattern in target_modules:
                        if pattern in full_name.lower():
                            should_replace = True
                            break
                if should_replace:
                    try:
                        nv_linear = NvFp8Linear(child)
                        setattr(module, child_name, nv_linear)
                        count += 1
                        logger.debug(f"Replaced {full_name} with NvFp8Linear")
                    except Exception as e:
                        logger.warning(f"Failed to replace {full_name} with NvFp8Linear: {e}")
            else:
                _recursive_replace(child, full_name)

    _recursive_replace(model)
    logger.info(f"Replaced {count} Linear layers with NvFp8Linear")
    return model

# ==================== Model Factory ====================
def make_model_factory(
    model_id,
    tokenizer_kwargs=None,
    model_kwargs=None,
    use_mx=False,
    mx_specs=None,
    mx_spec_groups=None,
    target_modules=None,
    multi_gpu=False,
    max_memory=None,
):
    """
    Create a factory function that loads a model and optionally applies MX quantization.
    
    Args:
        model_id: HuggingFace model identifier
        tokenizer_kwargs: Kwargs for tokenizer
        model_kwargs: Kwargs for model loading
        use_mx: Whether to apply MX quantization
        mx_specs: MX specifications (required if use_mx=True)
        target_modules: Which modules to replace (None = all Linear layers)
    """
    tokenizer_kwargs = tokenizer_kwargs or {}
    model_kwargs = model_kwargs or {}
    
    def _build_multi_gpu_max_memory(default_mem, visible_devices):
        if isinstance(default_mem, dict):
            return default_mem
        if isinstance(default_mem, str):
            return {idx: default_mem for idx in visible_devices}
        if default_mem is None:
            values = {}
            for idx in visible_devices:
                try:
                    props = torch.cuda.get_device_properties(idx)
                    total_gib = max(int(props.total_memory // (1024 ** 3)) - 2, 8)
                except RuntimeError:
                    total_gib = 20
                values[idx] = f"{total_gib}GiB"
            return values
        return {idx: f"{default_mem}GiB" for idx in visible_devices}

    def factory(target_device=None):
        load_kwargs = dict(model_kwargs)
        load_kwargs.setdefault("low_cpu_mem_usage", False)

        if multi_gpu:
            if not torch.cuda.is_available():
                raise RuntimeError(f"Model {model_id} requires CUDA for multi-GPU loading, but CUDA is unavailable.")
            visible_devices = list(range(torch.cuda.device_count()))
            if not visible_devices:
                raise RuntimeError(f"No CUDA devices visible for multi-GPU model {model_id}.")
            load_kwargs.setdefault("device_map", "balanced")
            if load_kwargs["device_map"] == "auto":
                load_kwargs["device_map"] = "balanced"
            load_kwargs.setdefault("max_memory", _build_multi_gpu_max_memory(max_memory, visible_devices))
        else:
            load_kwargs.pop("device_map", None)
            load_kwargs.pop("max_memory", None)
        
        # Force eager attention (no Flash Attention)
        load_kwargs["attn_implementation"] = "eager"
        
        target = _resolve_device(target_device)
        with _MODEL_LOAD_LOCK:
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)

        dispatch_map = getattr(model, "hf_device_map", None)
        if dispatch_map:
            unique_devices = sorted(
                {
                    dev if isinstance(dev, str) else f"cuda:{dev}"
                    for dev in dispatch_map.values()
                }
            )
            logger.info(f"Loaded {model_id} across devices: {unique_devices}")
            try:
                primary_location = next(iter(dispatch_map.values()))
                if isinstance(primary_location, str):
                    primary_device = torch.device(primary_location)
                else:
                    primary_device = torch.device(f"cuda:{primary_location}")
            except StopIteration:
                primary_device = torch.device("cuda:0") if torch.cuda.is_available() else DEFAULT_DEVICE
            setattr(model, "_mx_primary_device", primary_device)
        else:
            setattr(model, "_mx_primary_device", target)
        
        if any(p.is_meta for p in model.parameters()):
            raise RuntimeError(f"{model_id} weights were initialized on a meta device; disable checkpoint sharding.")
        
        # Apply MX quantization if requested
        if use_mx:
            if mx_spec_groups:
                logger.info(f"Applying MX quantization groups to {model_id}")
                for group in mx_spec_groups:
                    group_specs = group.get("mx_specs")
                    group_targets = group.get("target_modules")
                    if group_specs is None:
                        raise ValueError("mx_spec_groups entries must include 'mx_specs'")
                    model = replace_linear_with_mx(model, group_specs, group_targets)
            else:
                if mx_specs is None:
                    raise ValueError("mx_specs required when use_mx=True")
                logger.info(f"Applying MX quantization to {model_id}")
                model = replace_linear_with_mx(model, mx_specs, target_modules)

        dispatch_map = getattr(model, "hf_device_map", None)
        if dispatch_map is None:
            if target.type == "cuda":
                device_index = target.index if target.index is not None else torch.cuda.current_device()
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA target requested but no CUDA devices are available.")
                with torch.cuda.device(device_index):
                    model = model.to(target, non_blocking=True)
            else:
                model = model.to(target)
            setattr(model, "_mx_primary_device", target)
        
        return model, tokenizer
    
    return factory

def make_int8_factory(
    model_id,
    tokenizer_kwargs=None,
    model_kwargs=None,
    quant_kwargs=None,
):
    if BitsAndBytesConfig is None:
        raise ImportError("BitsAndBytesConfig not available. Install bitsandbytes to use INT8 backend.")

    tokenizer_kwargs = tokenizer_kwargs or {}
    model_kwargs = model_kwargs or {}
    quant_kwargs = quant_kwargs or {}

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=quant_kwargs.get("llm_int8_threshold", 6.0),
        llm_int8_skip_modules=quant_kwargs.get("llm_int8_skip_modules"),
    )

    def factory(target_device=None):
        load_kwargs = dict(model_kwargs)
        load_kwargs.setdefault("low_cpu_mem_usage", True)
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs.setdefault("device_map", "auto")
        load_kwargs.pop("torch_dtype", None)
        load_kwargs.setdefault("attn_implementation", "eager")

        with _MODEL_LOAD_LOCK:
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)

        dispatch_map = getattr(model, "hf_device_map", None)
        if dispatch_map:
            try:
                primary_location = next(iter(dispatch_map.values()))
                primary_device = torch.device(primary_location if isinstance(primary_location, str) else f"cuda:{primary_location}")
            except StopIteration:
                primary_device = torch.device("cuda:0") if torch.cuda.is_available() else DEFAULT_DEVICE
            setattr(model, "_mx_primary_device", primary_device)
        else:
            target = _resolve_device(target_device)
            if target.type != "cuda":
                raise RuntimeError("INT8 backend requires CUDA device")
            model = model.to(target)
            setattr(model, "_mx_primary_device", target)

        return model, tokenizer

    return factory


def make_nvfp8_factory(
    model_id,
    tokenizer_kwargs=None,
    model_kwargs=None,
    target_modules=None,
    recipe=None,
):
    if not TE_AVAILABLE or te is None:
        raise ImportError("transformer_engine is required for NVFP8 backend")
    tokenizer_kwargs = tokenizer_kwargs or {}
    model_kwargs = model_kwargs or {}
    recipe = recipe or DEFAULT_NVFP8_RECIPE
    if recipe is None:
        raise RuntimeError("No NVFP8 recipe available; ensure Transformer Engine is installed correctly.")

    def factory(target_device=None):
        load_kwargs = dict(model_kwargs)
        load_kwargs.setdefault("low_cpu_mem_usage", False)
        load_kwargs.setdefault("torch_dtype", torch.bfloat16)  # Use bfloat16 for better TE compatibility
        load_kwargs.setdefault("attn_implementation", "eager")
        load_kwargs.pop("device_map", None)
        load_kwargs.pop("max_memory", None)

        target = _resolve_device(target_device)
        if target.type != "cuda":
            raise RuntimeError("NVFP8 backend requires CUDA device")

        with _MODEL_LOAD_LOCK:
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)

        if any(p.is_meta for p in model.parameters()):
            raise RuntimeError("NVFP8 backend does not support meta device weights; disable checkpoint sharding.")

        # Move to target device before replacing
        model = model.to(target)
        
        # Replace linear layers with TE FP8 layers
        model = replace_linear_with_nvfp8(model, target_modules)
        
        setattr(model, "_mx_primary_device", target)
        setattr(model, "_nvfp8_recipe", recipe)
        setattr(
            model,
            "_mx_context_factory",
            lambda: te.fp8_autocast(enabled=True, fp8_recipe=recipe)
        )

        return model, tokenizer

    return factory

# ==================== Data ====================
def load_pile_data(split="validation", max_samples=1000):
    """
    Load data from The Pile dataset (Parquet-backed on HuggingFace Hub).
    Falls back to WikiText if The Pile is not available.
    """
    try:
        # Use monology/pile-uncopyrighted which has Parquet files
        logger.info(f"Loading {max_samples} samples from The Pile ({split} split)...")
        ds = load_dataset("monology/pile-uncopyrighted", split=split, streaming=True)
        texts = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            texts.append(item["text"])
        logger.info(f"Loaded {len(texts)} samples from The Pile")
        return texts
    except Exception as e:
        logger.warning(f"Could not load The Pile: {e}. Falling back to WikiText.")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return ds["text"][:max_samples]

def get_available_devices(min_free_ratio=0.5):
    """Get list of CUDA device indices with sufficient free memory."""
    if not torch.cuda.is_available():
        return []
    eligible = []
    for idx in range(torch.cuda.device_count()):
        with torch.cuda.device(idx):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        if total_bytes and free_bytes / total_bytes >= min_free_ratio:
            eligible.append(idx)
    return eligible

def make_loader(tokenizer, data, block_size=512, batch_size=8, num_workers=0):
    """Create a DataLoader for evaluation."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokens = tokenizer(
        data,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=block_size
    )

    ds = torch.utils.data.TensorDataset(tokens.input_ids, tokens.attention_mask)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )

# ==================== Metrics ====================
class Entropy(Metric):
    def reset(self): 
        self.sum, self.n = 0.0, 0
    
    def update(self, output):
        logits, _ = output
        probs = torch.softmax(logits, dim=-1)
        ent = -(probs * (probs + 1e-12).log()).sum(dim=-1)
        self.sum += ent.sum().item()
        self.n += ent.numel()
    
    def compute(self): 
        return self.sum / self.n

def perplexity(loss_val): 
    return math.exp(loss_val)

# ==================== Evaluation ====================
def eval_model(model, loader, device, context_factory=None):
    """Evaluate a model on the given loader."""
    dispatch_map = getattr(model, "hf_device_map", None)
    if dispatch_map:
        target = getattr(model, "_mx_primary_device", None)
        if target is None:
            try:
                primary_location = next(iter(dispatch_map.values()))
                if isinstance(primary_location, str):
                    target = torch.device(primary_location)
                else:
                    target = torch.device(f"cuda:{primary_location}")
            except StopIteration:
                target = torch.device("cuda:0") if torch.cuda.is_available() else DEFAULT_DEVICE
            setattr(model, "_mx_primary_device", target)
    else:
        target = _resolve_device(device)
        if any(p.is_meta for p in model.parameters()):
            raise RuntimeError("Model contains meta tensors; instantiate the model on a real device.")

        param_devices = {p.device for p in model.parameters()}
        if not param_devices or len(param_devices) > 1 or target not in param_devices:
            try:
                model.to(target)
            except NotImplementedError as err:
                raise RuntimeError("Failed to move model to target device; ensure factory loads weights on that device.") from err
    
    model.eval()
    
    def step(engine, batch):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(target)
        attention_mask = attention_mask.to(target)

        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

        return out.logits, out.loss

    engine = Engine(step)
    Loss(F.cross_entropy, output_transform=lambda x: (x[0].reshape(-1, x[0].size(-1)),
                                                      x[0].argmax(-1).reshape(-1))).attach(engine, "xent")
    Entropy(output_transform=lambda x: x).attach(engine, "entropy")
    context = context_factory() if context_factory else nullcontext()
    with context:
        state = engine.run(loader)
    loss = state.metrics["xent"]
    return {
        "PPL": perplexity(loss),
        "Xent": loss,
        "Entr": state.metrics["entropy"]
    }

def pretty(name, tag, metrics):
    """Pretty-print evaluation metrics."""
    msg = f"{name:32}{tag:15}  PPL {metrics['PPL']:.2f}  Xent {metrics['Xent']:.4f}  Entr {metrics['Entr']:.4f}"
    logger.info(msg)

# ==================== Parallel Evaluation ====================
def run_eval(model_builders, tag, data, devices, batch_size=8, workers_per_device=1, loader_workers=0):
    """
    Run evaluation across multiple GPUs in parallel.
    
    Args:
        model_builders: Dict of {name: factory_fn}
        tag: Tag for logging (e.g., "FP32", "MX-FP8")
        data: Text data to evaluate on
        devices: List of CUDA device indices to use
        batch_size: Batch size for evaluation
        workers_per_device: Number of parallel workers per GPU
        loader_workers: Number of DataLoader workers
    """
    workers_per_device = max(1, workers_per_device)
    loader_workers = max(0, loader_workers)
    
    if devices:
        logger.info(f"{tag:15}  Using GPUs: {devices} with {workers_per_device} worker(s)/device")
    else:
        logger.warning(f"{tag:15}  No GPUs available. Using {DEFAULT_DEVICE}.")
    
    if not devices:
        # CPU fallback
        for name, build_fn in model_builders.items():
            model = tok = loader = None
            try:
                model, tok = build_fn(DEFAULT_DEVICE)
                loader = make_loader(tok, data, batch_size=batch_size, num_workers=loader_workers)
                context_factory = getattr(model, "_mx_context_factory", None)
                res = eval_model(model, loader, DEFAULT_DEVICE, context_factory=context_factory)
                pretty(name, tag, res)
            finally:
                if loader is not None:
                    del loader
                if tok is not None:
                    del tok
                if model is not None:
                    model.to("cpu")
                    del model
                gc.collect()
        return

    # GPU parallel execution
    task_queue = queue.Queue()
    for item in model_builders.items():
        task_queue.put(item)

    def worker(device_id):
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()
        while True:
            try:
                name, build_fn = task_queue.get_nowait()
            except queue.Empty:
                break
            model = tok = loader = None
            try:
                torch.cuda.reset_peak_memory_stats()
                target_device = torch.device(f"cuda:{device_id}")
                model, tok = build_fn(target_device)
                loader = make_loader(tok, data, batch_size=batch_size, num_workers=loader_workers)
                context_factory = getattr(model, "_mx_context_factory", None)
                res = eval_model(model, loader, target_device, context_factory=context_factory)
                pretty(name, tag, res)
            finally:
                if loader is not None:
                    del loader
                if tok is not None:
                    del tok
                dispatch_devices = set()
                if model is not None:
                    device_map = getattr(model, "hf_device_map", None)
                    if device_map:
                        for dev in device_map.values():
                            if isinstance(dev, str):
                                dispatch_devices.add(dev)
                            else:
                                dispatch_devices.add(f"cuda:{dev}")
                    else:
                        model.to("cpu")
                    del model
                if dispatch_devices:
                    for dev in dispatch_devices:
                        with torch.cuda.device(torch.device(dev)):
                            torch.cuda.empty_cache()
                else:
                    torch.cuda.empty_cache()
                gc.collect()

    with ThreadPoolExecutor(max_workers=len(devices) * workers_per_device) as pool:
        futures = [
            pool.submit(worker, dev)
            for dev in devices
            for _ in range(workers_per_device)
        ]
        for fut in futures:
            fut.result()

# ==================== Model Configurations ====================
MODEL_CONFIGS = {
    # Small models for testing
    "tinystories-33M": {
        "model_id": "roneneldan/TinyStories-33M",
        "tokenizer_kwargs": {"use_fast": True},
        "model_kwargs": {
            "torch_dtype": torch.float32,
            "trust_remote_code": True,
            "use_safetensors": True
        }
    },
    "qwen1.5-0.5B": {
        "model_id": "Qwen/Qwen1.5-0.5B",
        "tokenizer_kwargs": {"trust_remote_code": True},
        "model_kwargs": {
            "torch_dtype": torch.float32,
            "trust_remote_code": True
        }
    },
    "phi-1_5": {
        "model_id": "microsoft/phi-1_5",
        "tokenizer_kwargs": {},
        "model_kwargs": {
            "torch_dtype": torch.float32,
            "trust_remote_code": True,
            "use_safetensors": True
        }
    },
    # 7B-30B range models (primary targets)
    "llama-3.1-8b": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B",
        "tokenizer_kwargs": {"use_fast": True},
        "model_kwargs": {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "use_safetensors": True,
        }
    },
    "qwen2.5-7b": {
        "model_id": "Qwen/Qwen2.5-7B",
        "tokenizer_kwargs": {"use_fast": True},
        "model_kwargs": {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "use_safetensors": True,
        }
    },
    "qwen2.5-14b": {
        "model_id": "Qwen/Qwen2.5-14B",
        "tokenizer_kwargs": {"use_fast": True},
        "model_kwargs": {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "use_safetensors": True,
        },
        "multi_gpu": True,
        "max_memory": "12GiB",
        "eval_batch_size": 1,
    },
    "phi-3-medium-14b": {
        "model_id": "microsoft/Phi-3-medium-128k-instruct",
        "tokenizer_kwargs": {"use_fast": True},
        "model_kwargs": {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "use_safetensors": True,
        }
    },
    "mixtral-8x7b": {
        "model_id": "mistralai/Mixtral-8x7B-v0.1",
        "tokenizer_kwargs": {"use_fast": True},
        "model_kwargs": {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "use_safetensors": True,
        }
    },
    "mpt-30b": {
        "model_id": "mosaicml/mpt-30b",
        "tokenizer_kwargs": {"use_fast": True},
        "model_kwargs": {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
    },
    "yi-34b": {
        "model_id": "01-ai/Yi-34B",
        "tokenizer_kwargs": {"use_fast": True},
        "model_kwargs": {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "use_safetensors": True,
        }
    },
        # DeepSeek LLM 67B (base)
    "deepseek-llm-67b-base": {
        "model_id": "deepseek-ai/deepseek-llm-67b-base",
        "tokenizer_kwargs": {
            "use_fast": True,
            "trust_remote_code": True
        },
        "model_kwargs": {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "use_safetensors": True,
            "attn_implementation": "eager",   # ensure GEMMs are visible for MX
            "low_cpu_mem_usage": True,
            "device_map": "auto"              # enable model-parallel across GPUs
        },
        "multi_gpu": True,
        "max_memory": "20GiB",
        "eval_batch_size": 1,
    },
}

# Target modules for selective replacement (normalized patterns)
MLP_MODULE_PATTERNS = [
    "gate_proj", "up_proj", "down_proj",      # LLaMA / GPT-NeoX style MLP
    "fc1", "fc2",                              # BERT-style feed-forward
    "w1", "w2", "w3",                          # Mixtral experts
    "dense_h_to_4h", "dense_4h_to_h",         # GPT-NeoX / MPT
    "mlp.c_fc", "mlp.c_proj",                 # GPT-2 / GPT-J
]

ATTENTION_MODULE_PATTERNS = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # Standard attention projections
    "c_attn", "c_proj",                        # GPT-style packed projections
    "query", "key", "value", "dense",          # Alternative naming
    "wqkv", "out_proj",                        # Phi / MPT variations
]

ALL_TIERS_ORDER = ["A", "B", "C", "D", "E"]

TIER_DEFINITIONS = {
    "A": {
        "description": "MLP-only MXFP8 (very safe)",
        "target_patterns": MLP_MODULE_PATTERNS,
        "allow_format_override": True,
        "default_format": "fp8_e4m3",
    },
    "B": {
        "description": "MLP + attention projections in MXFP8 (safe)",
        "target_patterns": MLP_MODULE_PATTERNS + ATTENTION_MODULE_PATTERNS,
        "allow_format_override": True,
        "default_format": "fp8_e4m3",
    },
    "C": {
        "description": "MLP in MXFP6, attention projections in MXFP8 (balanced)",
        "target_patterns": MLP_MODULE_PATTERNS + ATTENTION_MODULE_PATTERNS,
        "module_groups": [
            {"patterns": MLP_MODULE_PATTERNS, "format": "fp6_e3m2"},
            {"patterns": ATTENTION_MODULE_PATTERNS, "format": "fp8_e4m3"},
        ],
    },
    "D": {
        "description": "MLP in MXFP4, attention projections in MXFP6 (aggressive)",
        "target_patterns": MLP_MODULE_PATTERNS + ATTENTION_MODULE_PATTERNS,
        "module_groups": [
            {"patterns": MLP_MODULE_PATTERNS, "format": "fp4_e2m1"},
            {"patterns": ATTENTION_MODULE_PATTERNS, "format": "fp6_e3m2"},
        ],
    },
    "E": {
        "description": "MLP + attention projections in MXFP4 (most aggressive)",
        "target_patterns": MLP_MODULE_PATTERNS + ATTENTION_MODULE_PATTERNS,
        "module_groups": [
            {"patterns": MLP_MODULE_PATTERNS + ATTENTION_MODULE_PATTERNS, "format": "fp4_e2m1"},
        ],
    },
}


def _normalize_patterns(patterns):
    if patterns is None:
        return None
    seen = set()
    ordered = []
    for pat in patterns:
        lowered = pat.lower()
        if lowered not in seen:
            seen.add(lowered)
            ordered.append(lowered)
    return ordered


def build_tier_mx_plans(tier_name, mx_formats):
    cfg = TIER_DEFINITIONS[tier_name]
    plans = []
    if cfg.get("allow_format_override", False):
        formats = list(mx_formats) if mx_formats else [cfg.get("default_format", "fp8_e4m3")]
        for fmt in formats:
            plans.append({
                "label": f"Tier{tier_name}-{fmt}",
                "groups": [
                    {"patterns": cfg.get("target_patterns"), "format": fmt}
                ],
            })
    else:
        groups = []
        for group in cfg.get("module_groups", []):
            groups.append({
                "patterns": group.get("patterns"),
                "format": group["format"],
            })
        plans.append({
            "label": f"Tier{tier_name}",
            "groups": groups,
        })
    return plans

def main():
    parser = argparse.ArgumentParser(description="Selective MX GEMM Replacement Benchmark")
    default_model_list = list(MODEL_CONFIGS.keys())
    parser.add_argument("--models", nargs="+", default=default_model_list, 
                       help="Models to evaluate")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="Batch size for evaluation")
    parser.add_argument("--min_free_ratio", type=float, default=0.5, 
                       help="Minimum free memory ratio to select a GPU")
    parser.add_argument("--workers_per_device", type=int, default=1, 
                       help="Number of parallel evaluation workers per GPU")
    parser.add_argument("--loader_workers", type=int, default=2, 
                       help="Number of DataLoader workers")
    parser.add_argument("--max_samples", type=int, default=20000,
                       help="Maximum number of samples from dataset")
    parser.add_argument("--tier", choices=["A", "B", "C", "D", "E", "all"], default="B",
                       help="Quantization tier: A=MLP FP8, B=MLP+attention FP8 (default), C=MLP FP6 + attention FP8, D=MLP FP4 + attention FP6, E=MLP+attention FP4, or 'all' to run tiers A-E sequentially")
    parser.add_argument("--all_linears", action="store_true",
                       help="Replace ALL Linear layers (not just attention/MLP)")
    parser.add_argument("--mlp_only", action="store_true",
                       help="(Deprecated) Equivalent to specifying --tier A")
    parser.add_argument("--skip_mx", action="store_true",
                       help="Skip MX selective replacement stage")
    parser.add_argument("--mx_format", nargs="+", choices=list(MX_FORMAT_PRESETS.keys()), default=list(DEFAULT_MX_FORMATS),
                       help=f"MX format preset(s) to use (default: {', '.join(DEFAULT_MX_FORMATS)})")
    parser.add_argument("--skip_nvfp8_emulation", action="store_true",
                       help="Skip NV-FP8 emulation (no shared exponent) stage")
    parser.add_argument("--skip_bfp16", action="store_true",
                       help="Skip BFP16 block-floating stage")
    parser.add_argument("--skip_bf16", action="store_true",
                       help="Skip BF16 per-tensor stage")
    parser.add_argument("--full_pipeline", action="store_true",
                       help="Run all tiers and stages with no skips for the primary MX comparison set")
    args = parser.parse_args()

    if args.full_pipeline:
        args.tier = "all"
        args.skip_mx = False
        args.skip_nvfp8_emulation = False
        args.skip_bfp16 = False
        args.skip_bf16 = False
        if sorted(args.models) == sorted(default_model_list):
            args.models = ["llama-3.1-8b", "qwen2.5-14b"]

    # Filter selected models
    selected_configs = {name: cfg for name, cfg in MODEL_CONFIGS.items() if name in args.models}
    if not selected_configs:
        raise ValueError("No models selected for evaluation.")

    if getattr(args, "no_mx", False):
        args.skip_mx = True
    
    # Determine tier program
    if args.all_linears:
        tiers_to_run = ["ALL"]
        tier_target_modules = {"ALL": None}
        logger.info("Will replace ALL Linear layers with MX (overrides tier selection)")
    else:
        tier_arg = (args.tier or "B").upper()
        if args.mlp_only:
            logger.warning("--mlp_only is deprecated; use --tier A instead. Proceeding with Tier A.")
            tier_arg = "A"
        if tier_arg == "ALL":
            tiers_to_run = list(ALL_TIERS_ORDER)
        else:
            if tier_arg not in TIER_DEFINITIONS:
                raise ValueError(f"Unsupported tier '{tier_arg}'. Available tiers: {ALL_TIERS_ORDER + ['all']}")
            tiers_to_run = [tier_arg]

        tier_target_modules = {}
        for tier_name in tiers_to_run:
            cfg = TIER_DEFINITIONS[tier_name]
            tier_target_modules[tier_name] = _normalize_patterns(cfg.get("target_patterns"))
            logger.info(f"Tier {tier_name} selected: {cfg['description']}")
        if args.skip_mx:
            logger.info("MX stage skipped by flag; tier selection will apply to NV-FP8/BFP16 only.")

    if isinstance(args.mx_format, str):
        mx_formats = [args.mx_format]
    else:
        mx_formats = list(args.mx_format)
    if not mx_formats:
        mx_formats = list(DEFAULT_MX_FORMATS)
    args.mx_format = mx_formats

    logged_formats = set()

    def resolve_spec(format_key):
        spec = get_mx_spec(format_key)
        if format_key not in logged_formats:
            logger.info(f"MX Configuration ({format_key}): {spec}")
            logged_formats.add(format_key)
        return spec
    
    # Load data
    logger.info("Loading evaluation data...")
    data = load_pile_data(max_samples=args.max_samples)
    
    logger.info("=" * 80)
    logger.info("SELECTIVE MX GEMM REPLACEMENT BENCHMARK")
    logger.info("=" * 80)
    
    for model_name, config in selected_configs.items():
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"{'=' * 80}")

        model_batch_size = config.get("eval_batch_size", args.batch_size)
        if model_batch_size != args.batch_size:
            logger.info(f"Using batch size override {model_batch_size} for {model_name}")
        
        # Build FP32/BF16 baseline
        logger.info("\n--- Baseline (FP32/BF16) ---")
        baseline_factory = {
            f"{model_name}": make_model_factory(
                model_id=config["model_id"],
                tokenizer_kwargs=config["tokenizer_kwargs"],
                model_kwargs=config["model_kwargs"],
                use_mx=False,
                multi_gpu=config.get("multi_gpu", False),
                max_memory=config.get("max_memory"),
            )
        }
        devices = get_available_devices(min_free_ratio=args.min_free_ratio)
        run_eval(
            baseline_factory,
            tag="Baseline",
            data=data,
            devices=devices,
            batch_size=model_batch_size,
            workers_per_device=args.workers_per_device,
            loader_workers=args.loader_workers
        )

        # BF16 per-tensor baseline
        if not args.skip_bf16:
            logger.info("\n--- BF16 (per-tensor cast) ---")
            try:
                bf16_model_kwargs = copy.deepcopy(config["model_kwargs"])
                bf16_model_kwargs["torch_dtype"] = torch.bfloat16
                bf16_factory = {
                    f"{model_name}-BF16": make_model_factory(
                        model_id=config["model_id"],
                        tokenizer_kwargs=config["tokenizer_kwargs"],
                        model_kwargs=bf16_model_kwargs,
                        use_mx=False,
                        multi_gpu=config.get("multi_gpu", False),
                        max_memory=config.get("max_memory"),
                    )
                }
                devices = get_available_devices(min_free_ratio=args.min_free_ratio)
                run_eval(
                    bf16_factory,
                    tag="BF16",
                    data=data,
                    devices=devices,
                    batch_size=model_batch_size,
                    workers_per_device=args.workers_per_device,
                    loader_workers=args.loader_workers
                )
            except Exception as exc:
                logger.warning(f"Skipping BF16 benchmark for {model_name}: {exc}")

        # Tier-specific evaluation stages
        bfp16_specs_cache = None
        nvfp8_emulation_cache = {}

        for tier_name in tiers_to_run:
            tier_patterns = tier_target_modules.get(tier_name)
            if tier_name == "ALL":
                tier_label = "ALL"
                tier_description = "All Linear layers"
            else:
                tier_label = f"T{tier_name}"
                tier_description = TIER_DEFINITIONS[tier_name]["description"]

            logger.info(f"\n--- Tier {tier_label}: {tier_description} ---")

            # BFP16 block-floating via MX
            if not args.skip_bfp16:
                try:
                    if bfp16_specs_cache is None:
                        bfp16_specs_cache = finalize_mx_specs(dict(BFP16_MX_PRESET))
                        logger.info(f"BFP16 MX Configuration: {bfp16_specs_cache}")
                    bfp16_factory = {
                        f"{model_name}-BFP16-{tier_label}": make_model_factory(
                            model_id=config["model_id"],
                            tokenizer_kwargs=config["tokenizer_kwargs"],
                            model_kwargs=config["model_kwargs"],
                            use_mx=True,
                            mx_specs=bfp16_specs_cache,
                            target_modules=tier_patterns,
                            multi_gpu=config.get("multi_gpu", False),
                            max_memory=config.get("max_memory"),
                        )
                    }
                    devices = get_available_devices(min_free_ratio=args.min_free_ratio)
                    run_eval(
                        bfp16_factory,
                        tag=f"BFP16-{tier_label}",
                        data=data,
                        devices=devices,
                        batch_size=model_batch_size,
                        workers_per_device=args.workers_per_device,
                        loader_workers=args.loader_workers
                    )
                except Exception as exc:
                    logger.warning(f"Skipping BFP16 benchmark for {model_name} [{tier_label}]: {exc}")

            # NV-FP8 emulation with no shared exponent bits
            if not args.skip_nvfp8_emulation:
                try:
                    nvfp8_factories = {}
                    for preset_name, preset in NVFP8_EMULATION_PRESETS.items():
                        if preset_name not in nvfp8_emulation_cache:
                            spec = finalize_mx_specs(dict(preset))
                            nvfp8_emulation_cache[preset_name] = spec
                            logger.info(f"NV-FP8 Emulation ({preset_name}) MX Configuration: {spec}")
                        spec = nvfp8_emulation_cache[preset_name]
                        factory_name = f"{model_name}-NVFP8-Emu-{preset_name}-{tier_label}"
                        nvfp8_factories[factory_name] = make_model_factory(
                            model_id=config["model_id"],
                            tokenizer_kwargs=config["tokenizer_kwargs"],
                            model_kwargs=config["model_kwargs"],
                            use_mx=True,
                            mx_specs=spec,
                            target_modules=tier_patterns,
                            multi_gpu=config.get("multi_gpu", False),
                            max_memory=config.get("max_memory"),
                        )
                    devices = get_available_devices(min_free_ratio=args.min_free_ratio)
                    run_eval(
                        nvfp8_factories,
                        tag=f"NVFP8Emu-{tier_label}",
                        data=data,
                        devices=devices,
                        batch_size=model_batch_size,
                        workers_per_device=args.workers_per_device,
                        loader_workers=args.loader_workers
                    )
                except Exception as exc:
                    logger.warning(f"Skipping NV-FP8 emulation for {model_name} [{tier_label}]: {exc}")

            # Build MX variant (selective GEMM replacement)
            if not args.skip_mx:
                plans = []
                if tier_name == "ALL":
                    for fmt in args.mx_format:
                        plans.append({
                            "label": f"AllLinears-{fmt}",
                            "groups": [{"patterns": None, "format": fmt}]
                        })
                else:
                    plans = build_tier_mx_plans(tier_name, args.mx_format)

                mx_factories = {}
                for plan in plans:
                    spec_groups = []
                    for group in plan["groups"]:
                        fmt = group["format"]
                        spec = resolve_spec(fmt)
                        spec_groups.append({
                            "mx_specs": spec,
                            "target_modules": _normalize_patterns(group.get("patterns")),
                        })
                    factory_name = f"{model_name}-{plan['label']}"
                    mx_factories[factory_name] = make_model_factory(
                        model_id=config["model_id"],
                        tokenizer_kwargs=config["tokenizer_kwargs"],
                        model_kwargs=config["model_kwargs"],
                        use_mx=True,
                        mx_spec_groups=spec_groups,
                        multi_gpu=config.get("multi_gpu", False),
                        max_memory=config.get("max_memory"),
                    )

                if not mx_factories:
                    logger.warning(f"No MX formats configured for tier {tier_label}; skipping MX stage.")
                else:
                    devices = get_available_devices(min_free_ratio=args.min_free_ratio)
                    run_eval(
                        mx_factories,
                        tag=f"MX-{tier_label}",
                        data=data,
                        devices=devices,
                        batch_size=model_batch_size,
                        workers_per_device=args.workers_per_device,
                        loader_workers=args.loader_workers
                    )
            else:
                logger.info(f"--- MX (Selective GEMM) [{tier_label}] --- Skipped (--skip_mx)")

            # NV-FP8 (Transformer Engine) for the tier
            if TE_AVAILABLE and te is not None and torch.cuda.is_available():
                try:
                    nvfp8_factory = {
                        f"{model_name}-NVFP8-{tier_label}": make_nvfp8_factory(
                            model_id=config["model_id"],
                            tokenizer_kwargs=config["tokenizer_kwargs"],
                            model_kwargs=config["model_kwargs"],
                            target_modules=tier_patterns,
                            recipe=config.get("nvfp8_recipe", DEFAULT_NVFP8_RECIPE),
                        )
                    }
                    devices = get_available_devices(min_free_ratio=args.min_free_ratio)
                    if not devices:
                        raise RuntimeError("No eligible CUDA devices for NV-FP8 backend")
                    run_eval(
                        nvfp8_factory,
                        tag=f"NVFP8-{tier_label}",
                        data=data,
                        devices=devices,
                        batch_size=model_batch_size,
                        workers_per_device=args.workers_per_device,
                        loader_workers=args.loader_workers
                    )
                except Exception as exc:
                    logger.warning(f"Skipping NV-FP8 benchmark for {model_name} [{tier_label}]: {exc}")
            else:
                if tier_name == tiers_to_run[0]:
                    reason = []
                    if not TE_AVAILABLE or te is None:
                        reason.append("transformer_engine not available")
                    if not torch.cuda.is_available():
                        reason.append("CUDA unavailable")
                    if reason:
                        logger.info(f"\n--- NV-FP8 (Transformer Engine) --- Skipped ({', '.join(reason)})")

        # Build INT8 variant (bitsandbytes)
        if BitsAndBytesConfig is not None and torch.cuda.is_available():
            logger.info("\n--- INT8 (bitsandbytes) ---")
            try:
                int8_factory = {
                    f"{model_name}-INT8": make_int8_factory(
                        model_id=config["model_id"],
                        tokenizer_kwargs=config["tokenizer_kwargs"],
                        model_kwargs=config["model_kwargs"],
                        quant_kwargs=config.get("int8_quant_kwargs"),
                    )
                }
                devices = get_available_devices(min_free_ratio=args.min_free_ratio)
                if not devices:
                    raise RuntimeError("No eligible CUDA devices for INT8 backend")
                run_eval(
                    int8_factory,
                    tag="INT8",
                    data=data,
                    devices=devices,
                    batch_size=model_batch_size,
                    workers_per_device=args.workers_per_device,
                    loader_workers=args.loader_workers
                )
            except Exception as exc:
                logger.warning(f"Skipping INT8 benchmark for {model_name}: {exc}")
        else:
            logger.info("\n--- INT8 (bitsandbytes) --- Skipped (dependency or CUDA unavailable)")

    
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
