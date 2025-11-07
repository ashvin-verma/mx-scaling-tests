import sys, os, gc
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import matplotlib.pyplot as plt
# sys.modules["flash_attn"] = None
# sys.modules["flash_attn_2_cuda"] = None
# os.environ["FLASH_ATTENTION_SKIP_IMPORT"] = "1"

import math, torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from ignite.engine import Engine
from ignite.metrics import Loss, Metric
from mx.mx_mapping import inject_pyt_ops
from mx.specs import finalize_mx_specs
import json
from datetime import datetime
import logging
import argparse
import transformers

# Check transformers version and warn if potentially incompatible
print(f"Transformers version: {transformers.__version__}")
if transformers.__version__ < "4.36.0":
    print("Warning: phi-3 models may require transformers >= 4.36.0 for full compatibility")

# Silence HF hub/tqdm progress and tokenizer threads for cleaner stdout
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

# ==================== Setup Logging ====================
# Configure file handler (INFO) and console handler (WARNING) to avoid INFO duplication on console
file_handler = logging.FileHandler("mx_scaling_results.log")
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[file_handler, console_handler],
)
logger = logging.getLogger(__name__)
# Hide Ignite noisy logs
logging.getLogger("ignite").setLevel(logging.ERROR)
logging.getLogger("ignite.engine").setLevel(logging.ERROR)
logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.ERROR)

# ==================== Devices ====================
DEFAULT_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def _resolve_device(device=None):
    if device is None:
        return DEFAULT_DEVICE
    if isinstance(device, torch.device):
        return device
    return torch.device(device)

# ==================== Data ====================
def load_data(dataset_name="wikitext"):
    if dataset_name == "wikitext":
        return load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"][:1000]
    elif dataset_name == "code":
        # public, no auth required
        ds = load_dataset("code_search_net", "python", split="train[:1000]")
        return ds["code"]  # field "code" contains the source snippets
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# Remove accidental eager dataset load at module import (kept here for reference)
# data = load_dataset("wikitext", "wikit-2-raw-v1", split="test")["text"][:1000]

# data = load_dataset("bigcode/the-stack", data_dir="data/python", split="train[:1000]") // more code related dataset

def make_loader(tokenizer, data, block_size=8, batch_size=1):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # print("✅ Added pad_token as eos_token")

    tokens = tokenizer(
        data,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=block_size
    )

    ds = torch.utils.data.TensorDataset(tokens.input_ids, tokens.attention_mask)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

def make_model_factory(
    model_cls,
    model_id,
    tokenizer_id=None,
    *,
    model_kwargs=None,
    tokenizer_kwargs=None,
    tokenizer_setup=None,
):
    tokenizer_id = tokenizer_id or model_id
    model_kwargs = dict(model_kwargs or {})
    tokenizer_kwargs = dict(tokenizer_kwargs or {})

    def factory(target_device=None):
        target = _resolve_device(target_device)
        load_kwargs = dict(model_kwargs)
        load_kwargs.setdefault("low_cpu_mem_usage", False)
        load_kwargs.pop("device_map", None)
        model = model_cls.from_pretrained(model_id, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **tokenizer_kwargs)
        if tokenizer_setup:
            tokenizer_setup(tokenizer)

        if any(p.is_meta for p in model.parameters()):
            raise RuntimeError(f"{model_id} weights were initialized on a meta device; disable checkpoint sharding.")

        if target.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA target requested but no CUDA devices are available.")
            device_index = target.index if target.index is not None else torch.cuda.current_device()
            with torch.cuda.device(device_index):
                model = model.to(target, non_blocking=True)
        else:
            model = model.to(target)

        return model, tokenizer

    return factory


def plot_exponent_hist(model_factories, device=torch.device("cpu")):
    """Generate exponent histograms for each model produced by the factories."""
    target = _resolve_device(device)
    for name, factory in model_factories.items():
        model = tok = None
        try:
            model, tok = factory(target)
            model = model.to("cpu").eval()
            exponents = []
            with torch.no_grad():
                for p in model.parameters():
                    flat = p.data.view(-1).float().cpu()
                    _, exp = torch.frexp(flat)
                    exponents.append(exp)
            all_exp = torch.cat(exponents).numpy()

            plt.figure(figsize=(8, 4))
            plt.hist(
                all_exp,
                bins=range(int(all_exp.min()), int(all_exp.max()) + 2),
                align="left",
            )
            plt.yscale("log")
            plt.title(f"Exponent Distribution — {name}")
            plt.xlabel("Biased Exponent (0…255)")
            plt.ylabel("Count (log scale)")
            plt.grid(True, which="both", ls=":")
            plt.tight_layout()
            plt.savefig(f"{name}_exp_hist.png")
            plt.close()
        finally:
            if model is not None:
                model.to("cpu")
                del model
            if tok is not None:
                del tok
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# ==================== Metrics ====================
class Entropy(Metric):
    def reset(self): self.sum, self.n = 0.0, 0
    def update(self, output):
        logits, _ = output
        probs = torch.softmax(logits, dim=-1)
        ent = -(probs * (probs+1e-12).log()).sum(dim=-1)
        self.sum += ent.sum().item(); self.n += ent.numel()
    def compute(self): return self.sum / self.n

def perplexity(loss_val): return math.exp(loss_val)

# ==================== Evaluation ====================
def eval_model(model, loader, device):
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
    state = engine.run(loader)
    loss = state.metrics["xent"]
    return {
        "PPL": perplexity(loss),
        "Xent": loss,
        "Entr": state.metrics["entropy"]
    }

def pretty(name, tag, metrics, config=None):
    msg = f"{name:32}{tag:10}  PPL {metrics['PPL']:.2f}  Xent {metrics['Xent']:.2f}  Entr {metrics['Entr']:.2f}"
    if config:
        details = []
        if config.get("block_size") is not None:
            details.append(f"block_size: {config['block_size']}")
        if config.get("scale_bits") is not None:
            details.append(f"scale_bits: {config['scale_bits']}")
        if details:
            msg += " | " + ", ".join(details)
    logger.info(msg)
    print(msg)

# ==================== Eval Wrapper ====================
def run_eval(model_builders, tag, data, *, device, config=None, batch_size=1):
    results = {}
    target = _resolve_device(device)
    for name, build_fn in model_builders.items():
        model = tok = loader = None
        try:
            if target.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            model, tok = build_fn(target)
            block_size = config.get("block_size") if config and config.get("block_size") else 8
            loader = make_loader(tok, data, block_size=block_size, batch_size=batch_size)
            res = eval_model(model, loader, target)
            pretty(name, tag, res, config)
            results[name] = {
                "model": name,
                "format": tag,
                "metrics": res,
                "config": config,
            }
        finally:
            if loader is not None:
                del loader
            if tok is not None:
                del tok
            if model is not None:
                model.to("cpu")
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results

# ==================== MX specs ====================
mx_formats = ["int8", "fp8_e4m3", "fp8_e5m2", "fp6_e3m2", "fp6_e2m3", "fp4_e2m1"]
block_sizes = [32, 64, 128, 256]  # Different block sizes to test
scale_bits_options = [4, 8, 16]  # Different scale bits to test

models_fp32 = {
    "bert-base-uncased":
        (AutoModelForMaskedLM.from_pretrained("bert-base-uncased"),
         AutoTokenizer.from_pretrained("bert-base-uncased")),
    "phi-1_5":
        (AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", torch_dtype=torch.float32),
         AutoTokenizer.from_pretrained("microsoft/phi-1_5")),
    # "tinystories-33M":
    #     (AutoModelForCausalLM.from_pretrained(
    #         "roneneldan/TinyStories-33M",
    #         torch_dtype=torch.float32,
    #         trust_remote_code=True,
    #         use_safetensors=True
    #     ),
    #     AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M", use_fast=True)
    # ),
    "phi-3-instruct":
        (AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-3-mini-128k-instruct",
            torch_dtype=torch.float32,
            trust_remote_code=True
        ),
         AutoTokenizer.from_pretrained("microsoft/phi-3-mini-128k-instruct")),
    "qwen1.5-0.5B":
        (AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True, torch_dtype=torch.float32),
         AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True))
     }

def main():
    parser = argparse.ArgumentParser(description='Run MX scaling tests with different configurations')
    parser.add_argument('--datasets', nargs='+', default=["wikitext", "code"], help='Datasets to evaluate on')
    parser.add_argument('--formats', nargs='+', default=mx_formats, help='MX formats to test')
    parser.add_argument('--block_sizes', nargs='+', type=int, default=block_sizes, help='Block sizes to test')
    parser.add_argument('--scale_bits', nargs='+', type=int, default=scale_bits_options, help='Scale bits to test')
    parser.add_argument('--models', nargs='+', default=list(models_fp32.keys()), help='Models to test')
    args = parser.parse_args()

    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"mx_scaling_results_{timestamp}.json"

    for dataset_name in args.datasets:
        logger.info(f"\n\n======= EVALUATING ON {dataset_name.upper()} =======")
        data = load_data(dataset_name)
        dataset_results = {}

        # Only run selected models
        filtered_models_fp32 = {k: v for k, v in models_fp32.items() if k in args.models}

        logger.info("\n=== FP32 Evaluation ===")
        fp32_results = run_eval(filtered_models_fp32, tag="(FP32)", data=data)
        dataset_results["fp32"] = fp32_results

        # ==================== MX Formats Evaluation ====================
        for mx_format in args.formats:
            format_results = {}
            
            # Special-case int8: it does not use block_size/scale_bits — run a single configuration
            if mx_format == "int8":
                config = {
                    "format": mx_format,
                    "block_size": None,
                    "scale_bits": None
                }
                
                logger.info(f"\n=== MX Format: {mx_format.upper()} (no block_size/scale_bits variation) ===")
                mx_cfg = finalize_mx_specs({
                    "w_elem_format": mx_format,
                    "a_elem_format": mx_format,
                    "custom_cuda": True,
                    "quantize_backprop": False,
                })
                inject_pyt_ops(mx_cfg)

                # Create copies of models with the same configuration
                models_mx = {
                    "bert-base-uncased":
                        (AutoModelForMaskedLM.from_pretrained("bert-base-uncased"),
                         AutoTokenizer.from_pretrained("bert-base-uncased")),
                    "phi-1_5":
                        (AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", torch_dtype=torch.float32, trust_remote_code=True,
                        use_safetensors=True),
                         AutoTokenizer.from_pretrained("microsoft/phi-1_5")),
                    "phi-3-instruct":
                        (AutoModelForCausalLM.from_pretrained(
                            "microsoft/phi-3-mini-128k-instruct",
                            torch_dtype=torch.float32,
                            trust_remote_code=True
                        ),
                         AutoTokenizer.from_pretrained("microsoft/phi-3-mini-128k-instruct")),
                    "qwen1.5-0.5B":
                        (AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True, torch_dtype=torch.float32),
                         AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True))
                }

                # Filter to only selected models
                filtered_models_mx = {k: v for k, v in models_mx.items() if k in args.models}
                
                if is_main_process:
                    plot_exponent_hist(filtered_models_mx, device="cpu")
                    
                config_key = f"{mx_format}"
                config_results = run_eval(filtered_models_mx, tag=f"(MX-{mx_format.upper()})", data=data, config=config)
                format_results[config_key] = config_results
                
                # Save results after this configuration
                if is_main_process:
                    dataset_results[mx_format] = format_results
                    all_results[dataset_name] = dataset_results
                    with open(results_file, 'w') as f:
                        json.dump(all_results, f, indent=2)

            else:
                # For other MX formats, vary block_size and scale_bits as before
                for block_size in args.block_sizes:
                    for scale_bits in args.scale_bits:
                        config = {
                            "format": mx_format,
                            "block_size": block_size,
                            "scale_bits": scale_bits
                        }
                        
                        logger.info(f"\n=== MX Format: {mx_format.upper()} | Block Size: {block_size} | Scale Bits: {scale_bits} ===")
                        mx_cfg = finalize_mx_specs({
                            "w_elem_format": mx_format,
                            "a_elem_format": mx_format,
                            "scale_bits": scale_bits,
                            "block_size": block_size,
                            "custom_cuda": True,
                            "quantize_backprop": False,
                        })
                        inject_pyt_ops(mx_cfg)

                        # Create copies of models with the same configuration
                        models_mx = {
                            "bert-base-uncased":
                                (AutoModelForMaskedLM.from_pretrained("bert-base-uncased"),
                                 AutoTokenizer.from_pretrained("bert-base-uncased")),
                            "phi-1_5":
                                (AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", torch_dtype=torch.float32, trust_remote_code=True,
                                use_safetensors=True),
                                 AutoTokenizer.from_pretrained("microsoft/phi-1_5")),
                            "phi-3-instruct":
                                (AutoModelForCausalLM.from_pretrained(
                                    "microsoft/phi-3-mini-128k-instruct",
                                    torch_dtype=torch.float32,
                                    trust_remote_code=True
                                ),
                                 AutoTokenizer.from_pretrained("microsoft/phi-3-mini-128k-instruct")),
                            "qwen1.5-0.5B":
                                (AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True, torch_dtype=torch.float32),
                                 AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True))
                        }
                        
                        # Filter to only selected models
                        filtered_models_mx = {k: v for k, v in models_mx.items() if k in args.models}
                        
                        if is_main_process:
                            plot_exponent_hist(filtered_models_mx, device="cpu")
                            
                        config_key = f"{mx_format}_b{block_size}_s{scale_bits}"
                        config_results = run_eval(filtered_models_mx, tag=f"(MX-{mx_format.upper()})", data=data, config=config)
                        format_results[config_key] = config_results
                        
                        # Save results after each configuration to avoid losing data if the process crashes
                        if is_main_process:
                            dataset_results[mx_format] = format_results
                            all_results[dataset_name] = dataset_results
                            with open(results_file, 'w') as f:
                                json.dump(all_results, f, indent=2)

        logger.info("\n" + "="*60)
        
    if is_main_process:
        logger.info(f"All results saved to {results_file}")

if __name__ == "__main__":
    main()
