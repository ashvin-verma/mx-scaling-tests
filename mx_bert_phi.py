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

from transformers import logging
logging.set_verbosity_error()


# ==================== Device ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    for name, (model, _) in model_dict.items():
        # 1) Move model to CPU and eval
        model.to(device).eval()

        # 2) Gather all exponent bits
        exponents = []
        with torch.no_grad():
            for p in model.parameters():
                # flat float32 on CPU
                flat = p.data.view(-1).float().cpu()
                _, exp = torch.frexp(flat)
                exponents.append(exp)
        all_exp = torch.cat(exponents).numpy()

        # 3) Plot histogram
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
        plt.savefig(f"{name}_exp_hist.png")

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
def eval_model(model, tokenizer, loader):
    model.eval()
    def step(engine, batch):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

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
def run_eval(model_dict, tag, data):
    for name, (model, tok) in model_dict.items():

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # model = torch.nn.DataParallel(model, device_ids=[0])  

        model.to(device)
        loader = make_loader(tok, data)
        res = eval_model(model, tok, loader)
        pretty(name, tag, res)
        model.to("cpu")
        del model
        torch.cuda.empty_cache()
        gc.collect()

# ==================== MX specs ====================
mx_formats = ["int8", "fp8_e4m3", "fp8_e5m2", "fp6_e3m2", "fp6_e2m3", "fp4_e2m1"]

models_fp32 = {
    "bert-base-uncased":
        (AutoModelForMaskedLM.from_pretrained("bert-base-uncased"),
         AutoTokenizer.from_pretrained("bert-base-uncased")),
"phi-1_5":
    (AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", torch_dtype=torch.float32),
     AutoTokenizer.from_pretrained("microsoft/phi-1_5")),
     "tinystories-33M":
    (AutoModelForCausalLM.from_pretrained(
    "roneneldan/TinyStories-33M",
    torch_dtype=torch.float32,
    trust_remote_code=True,
    use_safetensors=True
),
     AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M", use_fast=True)
),
    # "phi-3-instruct":
    #     (AutoModelForCausalLM.from_pretrained("microsoft/phi-3-mini-128k-instruct", torch_dtype=torch.float32,
    #                                           trust_remote_code=True),
    #      AutoTokenizer.from_pretrained("microsoft/phi-3-mini-128k-instruct")),

    "qwen1.5-0.5B":
        (AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True, torch_dtype=torch.float32),
         AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True))

}
# plot_exponent_hist(models_fp32, device="cpu")

for dataset in ["wikitext", "code"]:
    print(f"\n\n======= EVALUATING ON {dataset.upper()} =======")
    data = load_data(dataset)

    print("\n=== FP32 Evaluation ===")
    run_eval(models_fp32, tag="(FP32)", data=data)

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

        models_mx = {
            "bert-base-uncased":
                (AutoModelForMaskedLM.from_pretrained("bert-base-uncased"),
                AutoTokenizer.from_pretrained("bert-base-uncased")),
    "phi-1_5":
        (AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", torch_dtype=torch.float32, trust_remote_code=True,
        use_safetensors=True),
        AutoTokenizer.from_pretrained("microsoft/phi-1_5")),
            "tinystories-33M":
        (AutoModelForCausalLM.from_pretrained(
        "roneneldan/TinyStories-33M",
        torch_dtype=torch.float32,
        trust_remote_code=True,
        use_safetensors=True,
    ),
        AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M", use_fast=True)
    ),
        # "phi-3-instruct":
        #     (AutoModelForCausalLM.from_pretrained("microsoft/phi-3-mini-128k-instruct", torch_dtype=torch.float32,
        #                                         trust_remote_code=True),
        #     AutoTokenizer.from_pretrained("microsoft/phi-3-mini-128k-instruct")),

        "qwen1.5-0.5B":
            (AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True, torch_dtype=torch.float32),
            AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True))
        }
        plot_exponent_hist(models_mx, device="cpu")
        run_eval(models_mx, tag=f"(MX-{mx_format.upper()})", data=data)
    print("\n" + "="*60)
