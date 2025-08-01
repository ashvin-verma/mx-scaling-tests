import sys, os, gc
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
data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"][:1000]

# data = load_dataset("bigcode/the-stack", data_dir="data/python", split="train[:1000]") // more code related dataset

def make_loader(tokenizer, block_size=8, batch_size=1):
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
def run_eval(model_dict, tag):
    for name, (model, tok) in model_dict.items():

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model.to(device)
        loader = make_loader(tok)
        res = eval_model(model, tok, loader)
        pretty(name, tag, res)
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

}

print("\n=== FP32 Evaluation ===")
run_eval(models_fp32, tag="(FP32)")

# ==================== MX Formats Evaluation ====================
for mx_format in mx_formats:
    print(f"\n=== MX Format: {mx_format.upper()} ===")
    mx_cfg = finalize_mx_specs({
        "w_elem_format": mx_format,
        "a_elem_format": mx_format,
        "scale_bits": 4,
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
    }
    run_eval(models_mx, tag=f"(MX-{mx_format.upper()})")
print("\n" + "="*60)
