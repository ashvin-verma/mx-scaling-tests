import json
import urllib.request
import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from ignite.engine import create_supervised_evaluator
from ignite.metrics import TopKCategoricalAccuracy, Loss, Metric
from PIL import Image


# ----------------------------------------------------------------------------------------------------------------
# 1) Download ImageNet class-index JSON if missing
# ----------------------------------------------------------------------------------------------------------------
idx_file = Path("imagenet_class_index.json")
if not idx_file.exists():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_class_index.json",
        idx_file
    )
idx2syn = json.load(open(idx_file))  # {"0": ["n01440764","tench"], ...}
syn2global = {v[0]: int(k) for k, v in idx2syn.items()}

# ----------------------------------------------------------------------------------------------------------------
# 2) Download ImageNet-100 synset list if missing
# ----------------------------------------------------------------------------------------------------------------
syn_file = Path("imagenet100.txt")
if not syn_file.exists():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/HobbitLong/CMC/master/imagenet100.txt",
        syn_file
    )
synsets = syn_file.read_text().split()  # list of 100 synset IDs
local2global = {i: syn2global[s] for i, s in enumerate(synsets)}

# Safe mapping function to avoid KeyError
def map_target(local_idx):
    return local2global.get(local_idx, -1)

# ----------------------------------------------------------------------------------------------------------------
# 3) Build ImageFolder with correct mapping
# ----------------------------------------------------------------------------------------------------------------
val_root = "imagenet100_subset/val"  # HF-dumped folder structure
assert Path(val_root).is_dir(), "Validation directory missing. Run the HF dump cell first."

# Use Scale for older torchvision
eval_tf = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Create dataset without target_transform to inspect raw local targets
raw_ds = datasets.ImageFolder(root=val_root, transform=eval_tf)
raw_local_targets = [label for _, label in raw_ds.imgs[:10]]
print("Class to index mapping (local):", raw_ds.class_to_idx)
print("Sample raw local targets:", raw_local_targets)

# Now dataset with correct target_transform to global indices
val_ds = datasets.ImageFolder(
    root=val_root,
    transform=eval_tf,
    target_transform=map_target
)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Inspect mapping on first batch with exception handling
glob_targets = []
for idx in raw_local_targets:
    gl = map_target(idx)
    if gl < 0:
        print(f"Warning: no global mapping for local index {idx}")
    glob_targets.append(gl)
print("Mapped global targets:", glob_targets)

# ----------------------------------------------------------------------------------------------------------------
# 4) Define custom metrics
# ----------------------------------------------------------------------------------------------------------------
class Confidence(Metric):
    def reset(self): self.total, self.count = 0.0, 0
    def update(self, output):
        p = torch.softmax(output[0], dim=1).max(1)[0]
        self.total += p.sum().item()
        self.count += p.size(0)
    def compute(self): return self.total / self.count

class Entropy(Metric):
    def reset(self): self.total, self.count = 0.0, 0
    def update(self, output):
        p = torch.softmax(output[0], dim=1)
        e = -(p * (p + 1e-12).log()).sum(dim=1)
        self.total += e.sum().item()
        self.count += p.size(0)
    def compute(self): return self.total / self.count

metrics = {
    "top1": TopKCategoricalAccuracy(k=1),
    "top5": TopKCategoricalAccuracy(k=5),
    "xent": Loss(F.cross_entropy),
    "conf": Confidence(),
    "entr": Entropy(),
}

# ----------------------------------------------------------------------------------------------------------------
# 5) Evaluation + print helper
# ----------------------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_and_print(models_dict, tag=""):
    for name, model in models_dict.items():
        evaluator = create_supervised_evaluator(model, metrics, device=device)
        state = evaluator.run(val_loader)
        m = state.metrics
        print(f"{name+' '+tag:16}  Top-1 {m['top1']:.4f}  Top-5 {m['top5']:.4f}  "
              f"Xent {m['xent']:.2f}  Conf {m['conf']:.2f}  Entr {m['entr']:.2f}")

# ----------------------------------------------------------------------------------------------------------------
# 6) Run FP32 evaluation
# ----------------------------------------------------------------------------------------------------------------
models_fp32 = {
    "AlexNet":  models.alexnet(pretrained=True).to(device).eval(),
    "ResNet50": models.resnet50(pretrained=True).to(device).eval(),
}
eval_and_print(models_fp32, tag="(FP32)")

# ----------------------------------------------------------------------------------------------------------------
# 7) (Optional) Inject MX and re-run
# ----------------------------------------------------------------------------------------------------------------
from mx.mx_mapping import inject_pyt_ops
from mx.specs import finalize_mx_specs

raw_mx = {
    "w_elem_format":   "int8",
    "a_elem_format":   "int8",
    "scale_bits":      8,
    "block_size":     32,
    "custom_cuda":   False,
    "quantize_backprop": False,
}
mx_spec = finalize_mx_specs(raw_mx)
inject_pyt_ops(mx_spec)

models_mx = {
    "AlexNet":  models.alexnet(pretrained=True).to(device).eval(),
    "ResNet50": models.resnet50(pretrained=True).to(device).eval(),
}
eval_and_print(models_mx, tag="(MXINT8)")

img_path, local_label = raw_ds.imgs[0]
img = Image.open(img_path).convert("RGB")
x   = eval_tf(img).unsqueeze(0).to(device)
with torch.no_grad():
    logits = models.resnet50(pretrained=True).to(device)(x)
    probs  = torch.softmax(logits, dim=1)
    top5   = torch.topk(probs, 5).indices.cpu().numpy().tolist()[0]
print("True global label:", map_target(local_label))
print("Model top-5 preds:", top5)
