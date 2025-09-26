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
from concurrent.futures import ThreadPoolExecutor
import queue

from transformers import logging
logging.set_verbosity_error()


# ==================== Device ====================
DEFAULT_DEVICE = torch.device("cpu")

def _resolve_device(device):
    if device is None:
        return DEFAULT_DEVICE
    return device if isinstance(device, torch.device) else torch.device(device)

def make_model_factory(model_cls, model_id, tokenizer_id=None, tokenizer_kwargs=None, model_kwargs=None):
    tokenizer_id = tokenizer_id or model_id
    tokenizer_kwargs = tokenizer_kwargs or {}
    model_kwargs = model_kwargs or {}
    def factory(target_device=None):
        load_kwargs = dict(model_kwargs)
        load_kwargs.setdefault("low_cpu_mem_usage", False)
        target = _resolve_device(target_device)
        if target.type == "cuda":
            device_index = target.index if target.index is not None else torch.cuda.current_device()
            load_kwargs.update({"device_map": {"": device_index}})
        else:
            load_kwargs.update({"device_map": "cpu"})
        model = model_cls.from_pretrained(model_id, **load_kwargs)
        if any(p.is_meta for p in model.parameters()):
            raise RuntimeError(f"{model_id} weights were initialized on a meta device; disable checkpoint sharding.")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **tokenizer_kwargs)
        return model, tokenizer
    return factory


# ==================== Data ====================
def load_data(dataset_name="wikitext"):
    if dataset_name == "wikitext":
        return load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"][:1000]
    elif dataset_name == "code":
        # public, no auth required
        ds = load_dataset("code_search_net", "python", split="train[:1000]")
        return ds["code"]  # field “code” contains the source snippets
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_available_devices(min_free_ratio=0.8):
    if not torch.cuda.is_available():
        return []
    eligible = []
    for idx in range(torch.cuda.device_count()):
        with torch.cuda.device(idx):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        if total_bytes and free_bytes / total_bytes >= min_free_ratio:
            eligible.append(idx)
    return eligible


data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"][:1000]

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

def plot_exponent_hist(model_dict, device="cpu"):
    """
    For each model in model_dict (name:(model, tok)), 
    compute biased exponent bits of all weights and plot.
    """
    target_device = _resolve_device(device)
    for name, entry in model_dict.items():
        model = tok = None
        try:
            model, tok = entry(target_device)
            model.eval()

            exponents = []
            with torch.no_grad():
                for p in model.parameters():
                    flat = p.data.view(-1).float().cpu()
                    _, exp = torch.frexp(flat)
                    exponents.append(exp)
            all_exp = torch.cat(exponents).numpy()

            plt.figure(figsize=(8,4))
            plt.hist(all_exp, 
                     bins=range(int(all_exp.min()), int(all_exp.max())+2),
                     align="left")
            plt.yscale("log")
            plt.title(f"Exponent Distribution — {name}")
            plt.xlabel("Biased Exponent (0…255)")
            plt.ylabel("Count (log scale)")
            plt.grid(True, which="both", ls=":")
            plt.tight_layout()
            plt.savefig(f"{name}_exp_hist_gpus.png")
            plt.close()
        finally:
            if model is not None:
                model.cpu()
                del model
            if tok is not None:
                del tok
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

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

def pretty(name, tag, metrics):
    print(f"{name:32}{tag:10}  PPL {metrics['PPL']:.2f}  Xent {metrics['Xent']:.2f}  Entr {metrics['Entr']:.2f}")

# ==================== Eval Wrapper ====================
def run_eval(model_builders, tag, data, devices, batch_size=1):
    if devices:
        print(f"{tag:10}  Using GPUs (>=80% free): {devices}")
    else:
        print(f"{tag:10}  No GPUs meet free-memory threshold. Using {DEFAULT_DEVICE}.")
    if not devices:
        for name, build_fn in model_builders.items():
            model = tok = loader = None
            try:
                model, tok = build_fn(DEFAULT_DEVICE)
                loader = make_loader(tok, data, batch_size=batch_size)
                res = eval_model(model, loader, DEFAULT_DEVICE)
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
                loader = make_loader(tok, data, batch_size=batch_size)
                res = eval_model(model, loader, target_device)
                pretty(name, tag, res)
            finally:
                if loader is not None:
                    del loader
                if tok is not None:
                    del tok
                if model is not None:
                    model.to("cpu")
                    del model
                torch.cuda.empty_cache()
                gc.collect()

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(worker, dev) for dev in devices]
        for fut in futures:
            fut.result()

# ==================== MX specs ====================
mx_formats = ["int8", "fp8_e4m3", "fp8_e5m2", "fp6_e3m2", "fp6_e2m3", "fp4_e2m1"]

MODEL_FACTORIES = {
    "bert-base-uncased": make_model_factory(
        AutoModelForMaskedLM,
        "bert-base-uncased"
    ),
    "phi-1_5": make_model_factory(
        AutoModelForCausalLM,
        "microsoft/phi-1_5",
        model_kwargs={
            "torch_dtype": torch.float32,
            "trust_remote_code": True,
            "use_safetensors": True
        }
    ),
    "tinystories-33M": make_model_factory(
        AutoModelForCausalLM,
        "roneneldan/TinyStories-33M",
        tokenizer_kwargs={"use_fast": True},
        model_kwargs={
            "torch_dtype": torch.float32,
            "trust_remote_code": True,
            "use_safetensors": True
        }
    ),
    "qwen1.5-0.5B": make_model_factory(
        AutoModelForCausalLM,
        "Qwen/Qwen1.5-0.5B",
        tokenizer_kwargs={"trust_remote_code": True},
        model_kwargs={
            "torch_dtype": torch.float32,
            "trust_remote_code": True
        }
    )
}

for dataset in ["wikitext", "code"]:
    print(f"\n\n======= EVALUATING ON {dataset.upper()} =======")
    data = load_data(dataset)

    print("\n=== FP32 Evaluation ===")
    devices = get_available_devices()
    run_eval(MODEL_FACTORIES, tag="(FP32)", data=data, devices=devices)

    # ==================== MX Formats Evaluation ====================
    for mx_format in mx_formats:
        print(f"\n=== MX Format: {mx_format.upper()} ===")
        mx_cfg = finalize_mx_specs({
            "w_elem_format": mx_format,
            "a_elem_format": mx_format,
            "scale_bits": 8,
            "block_size": 32,
            "custom_cuda": True,
            "quantize_backprop": False,
        })
        inject_pyt_ops(mx_cfg)

        plot_exponent_hist(MODEL_FACTORIES, device="cpu")
        devices = get_available_devices()
        run_eval(MODEL_FACTORIES, tag=f"(MX-{mx_format.upper()})", data=data, devices=devices)
    print("\n" + "="*60)
