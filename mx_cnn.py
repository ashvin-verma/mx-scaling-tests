# □□ Step 0: Install dependencies (run this first)
# pip install neural-compressor[pt] amd-quark datasets torchvision ignite

# ─── Step 1: Imports ─────────────────────────────────────────────
import json, urllib.request, torch, torch.nn.functional as F
from datasets import load_dataset
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from ignite.engine import create_supervised_evaluator
from ignite.metrics import TopKCategoricalAccuracy, Loss, Metric
from neural_compressor.torch.quantization import MXQuantConfig, prepare, convert
from quark.torch.quantization.config.config import QuantizationConfig
from quark.torch.quantization.config.type import Dtype
from quark.torch.quantization.config.config import Config
from quark.torch.quantization.observer.observer import PerBlockMXObserver

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

# ─── Step 2: Download ImageNet-100 subset ───────────────────────
idx_file = Path("imagenet_class_index.json")
if not idx_file.exists():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_class_index.json",
        idx_file
    )
class_idx = json.load(open(idx_file))
ds = load_dataset("clane9/imagenet-100", split="validation")
root = Path("imagenet100_subset/val")
root.mkdir(parents=True, exist_ok=True)
label_names = ds.features["label"].names
for i, ex in enumerate(ds):
    wnid = label_names[ex["label"]]
    d = root/wnid; d.mkdir(exist_ok=True)
    ex["image"].save(d/f"{i}.png")

# ─── Step 3: Prepare DataLoader ─────────────────────────────────
eval_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_ds = datasets.ImageFolder(root=str(root), transform=eval_tf)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

# ─── Step 4: Set global-to-subset mapping ───────────────────────
wnid2idx1000 = {v[0]:int(k) for k,v in class_idx.items()}
subset_wnids = sorted(d.name for d in root.iterdir() if d.is_dir())
subset_to_1000 = torch.tensor([wnid2idx1000[wnid] for wnid in subset_wnids], dtype=torch.long)

def output_transform(output):
    y_pred,y = output
    return y_pred.index_select(1, subset_to_1000), y

# ─── Step 5: Set metrics ────────────────────────────────────────
class Confidence(Metric):
    def reset(self): self.total=0;self.n=0
    def update(self,out):
        p = torch.softmax(out[0],1).max(1)[0]
        self.total+=p.sum().item(); self.n+=len(p)
    def compute(self): return self.total/self.n

class Entropy(Metric):
    def reset(self): self.total=0; self.n=0
    def update(self,out):
        p = torch.softmax(out[0],1)
        e = -(p*(p+1e-12).log()).sum(1)
        self.total+=e.sum().item(); self.n+=len(e)
    def compute(self): return self.total/self.n

metrics = {
    "top1": TopKCategoricalAccuracy(k=1),
    "top5": TopKCategoricalAccuracy(k=5),
    "xent": Loss(torch.nn.CrossEntropyLoss()),
    "conf": Confidence(), "entr": Entropy()
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Step 6: Define eval runner ─────────────────────────────────
def eval_model(model, tag=""):
    model = safe_model_to_device(model, device).eval()
    evaluator = create_supervised_evaluator(model, metrics, device=device,
                                            output_transform=output_transform)
    state = evaluator.run(val_loader)
    m = state.metrics
    print(f"{tag:15} Top1={m['top1']:.4f} | Top5={m['top5']:.4f} | Xent={m['xent']:.2f} | Conf={m['conf']:.2f} | Entr={m['entr']:.2f}")

# ─── Step 7: Baseline FP32 eval ──────────────────────────────────
models_fp32 = {
    "ResNet50": models.resnet50(pretrained=True),
    "MobileNet-v2": models.mobilenet_v2(pretrained=True),
}
for name,mdl in models_fp32.items():
    eval_model(mdl, tag=f"{name} (FP32)")

# ─── Step 8: Intel INC MXQuant (MXINT8 or MXFP8) ───────────────
for name,mdl in models_fp32.items():
    m = safe_model_to_device(mdl, device).eval()
    inc_cfg = MXQuantConfig(weight_dtype="mxint8", act_dtype="mxint8")
    q = prepare(m, quant_config=inc_cfg, calib_dataloader=val_loader)
    q = convert(q)
    eval_model(q, tag=f"{name} (INC-MXINT8)")

# ─── Step 9: AMD Quark MXFP8 Quant ──────────────────────────────
for name,mdl in models_fp32.items():
    qcfg = QuantizationConfig(
        weight=QuantizationConfig(weight=None),  # dummy
    )
    from quark.torch.quantization.config.config import QuantizationSpec
    spec = QuantizationSpec(dtype=Dtype.mx, mx_element_dtype=Dtype.fp8_e4m3,
                             observer_cls=PerBlockMXObserver, 
                             qscheme=None, ch_axis=0, is_dynamic=True)
    cfg = Config(global_quant_config=spec)
    q = safe_model_to_device(mdl, device).eval()
    from quark.torch.quantization import quantizer
    qmodel = quantizer.quantize(model=q, config=cfg, calib_dataloader=val_loader)
    eval_model(qmodel, tag=f"{name} (Quark-MXFP8)")

# ─── DONE ─────────────────────────────────────────────────────
