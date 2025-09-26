# ------------ install (once) ------------------
# pip install datasets torchvision ignite torch

# ------------ mx_cnn_clean.py -----------------
import json, urllib.request, torch, torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from ignite.engine import create_supervised_evaluator
from ignite.metrics import TopKCategoricalAccuracy, Loss, Metric
import os

# Helper function to safely move models to device, handling meta tensors
def safe_model_to_device(model, device):
    """Safely move model to device, handling meta tensors properly."""
    try:
        return model.to(device)
    except NotImplementedError as e:
        if "Cannot copy out of meta tensor" in str(e):
            # Model has meta tensors, use to_empty instead
            return model.to_empty(device=device)
        else:
            raise e

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 1. Get the global 1 000-class JSON ───────────────────────
idx_file = Path("imagenet_class_index.json")
# if not idx_file.exists():
#     urllib.request.urlretrieve(
#         "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_class_index.json",
#         idx_file)
class_idx = json.load(idx_file.open())            # "0": ["n01440764","tench"]
wnid2idx1000 = {v[0]: int(k) for k, v in class_idx.items()}

# ─── 2. Download ImageNet-100 validation set (WordNet IDs!) ───
root = Path("imagenet100_subset/val")
if not root.exists():                      # first run → download
    ds = load_dataset("clane9/imagenet-100", split="validation")
    root.mkdir(parents=True)
    label_names = ds.features["label"].names   # WordNet IDs
    for i, ex in enumerate(ds):
        folder = root/label_names[ex["label"]]
        folder.mkdir(exist_ok=True)
        ex["image"].save(folder/f"{i}.jpeg")
else:
    print("Using existing ImageNet-100 dataset")

# ─── 3. Build subset→1000 mapping tensor ─────────────────────
subset_wnids = sorted(p.name for p in root.iterdir() if p.is_dir())
subset_to_1000 = torch.tensor(
    [wnid2idx1000[w] for w in subset_wnids], dtype=torch.long, device=device)

def output_tf(x, y, y_pred):
    """Updated signature for newer Ignite versions."""
    return y_pred.index_select(1, subset_to_1000), y


# ─── 4. DataLoader ───────────────────────────────────────────
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_ds = datasets.ImageFolder(root=str(root), transform=val_tf)
val_loader = DataLoader(val_ds, batch_size=32,
                        shuffle=False, num_workers=4, pin_memory=True)

# ─── 5. Extra metrics ────────────────────────────────────────
class Confidence(Metric):
    def reset(self): self.t, self.n = 0.0, 0
    def update(self, out):
        p = torch.softmax(out[0], 1).max(1)[0]
        self.t += p.sum().item(); self.n += p.size(0)
    def compute(self): return self.t / self.n

class Entropy(Metric):
    def reset(self): self.t, self.n = 0.0, 0
    def update(self, out):
        p = torch.softmax(out[0], 1)
        e = -(p*(p+1e-12).log()).sum(1)
        self.t += e.sum().item(); self.n += e.size(0)
    def compute(self): return self.t / self.n

metrics = {
    "top1": TopKCategoricalAccuracy(k=1),
    "top5": TopKCategoricalAccuracy(k=5),
    "xent": Loss(F.cross_entropy),
    "conf": Confidence(),
    "entr": Entropy(),
}

def eval_and_print(name, model):
    model = safe_model_to_device(model, device).eval()
    evaluator = create_supervised_evaluator(
        model, metrics, device=device, output_transform=output_tf)
    s = evaluator.run(val_loader).metrics
    print(f"{name:15}  Top1 {s['top1']:.4f}  Top5 {s['top5']:.4f}  "
          f"Xent {s['xent']:.2f}  Conf {s['conf']:.2f}  Entr {s['entr']:.2f}")

# ─── 6. FP32 baseline ────────────────────────────────────────
eval_and_print("AlexNet (FP32)",  models.alexnet(pretrained=True))
eval_and_print("ResNet50 (FP32)", models.resnet50(pretrained=True))

# torch.set_float32_matmul_precision("medium")

# ─── 7. (Optional) MicroXcaling INT8 emulation ───────────────
mx_format_configs = [
    ("MXINT8",  {"w_elem_format": "int8",         "a_elem_format": "int8"}),
    ("MXFP8:E4M3",   {"w_elem_format": "fp8_e4m3",     "a_elem_format": "fp8_e4m3"}),
    ("MXFP8:E5M2",   {"w_elem_format": "fp8_e5m2",     "a_elem_format": "fp8_e5m2"}),
    ("MXFP6:E3M2",   {"w_elem_format": "fp6_e3m2",     "a_elem_format": "fp6_e3m2"}),
    ("MXFP6:E2M3",   {"w_elem_format": "fp6_e2m3",     "a_elem_format": "fp6_e2m3"}),
    ("MXFP4:E2M1",   {"w_elem_format": "fp4_e2m1",     "a_elem_format": "fp4_e2m1"}),
]

try:
    from mx.mx_mapping import inject_pyt_ops
    from mx.specs      import finalize_mx_specs

    for fmt_name, base_spec in mx_format_configs:
        mx_cfg = finalize_mx_specs({
            **base_spec,
            "scale_bits": 8, "block_size": 32, "custom_cuda": False,
        })
        inject_pyt_ops(mx_cfg)
        eval_and_print(f"AlexNet ({fmt_name})", models.alexnet(pretrained=True))
        eval_and_print(f"ResNet50 ({fmt_name})", models.resnet50(pretrained=True))

    # mx_cfg = finalize_mx_specs({
    #     "w_elem_format":"int8", "a_elem_format":"int8",
    #     "scale_bits":8, "block_size":32, "custom_cuda":False,
    # })
    # inject_pyt_ops(mx_cfg)
    # eval_and_print("AlexNet (MXINT8)",  models.alexnet(pretrained=True))
    # eval_and_print("ResNet50 (MXINT8)", models.resnet50(pretrained=True))
except ImportError:
    print("MicroXcaling not installed – skipping MXINT8 run")
