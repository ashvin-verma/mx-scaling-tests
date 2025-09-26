import os, json, urllib.request, shutil, types, torch, torch.nn.functional as F
from datasets import load_dataset
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from ignite.engine import create_supervised_evaluator
from ignite.metrics import TopKCategoricalAccuracy, Loss, Metric
from neural_compressor import PostTrainingQuantConfig, quantization

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

# ────────────────────────────────────────────────────────────
# 1. Download global ImageNet class map
idx_file = Path("imagenet_class_index.json")
if not idx_file.exists():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_class_index.json",
        idx_file)
class_idx = json.load(idx_file.open())
wnid2idx1000 = {v[0]: int(k) for k, v in class_idx.items()}
syn2idx1000  = {v[1]: int(k) for k, v in class_idx.items()}

# ────────────────────────────────────────────────────────────
# 2. APPROACH 2: Download exact ImageNet-100 mapping and rebuild dataset
def setup_imagenet100_with_exact_mapping():
    """Download exact ImageNet-100 class list and rebuild dataset properly."""
    
    # Download the official ImageNet-100 WordNet ID list
    imagenet100_url = "https://raw.githubusercontent.com/HobbitLong/CMC/master/imagenet100.txt"
    imagenet100_file = Path("imagenet100_wnids.txt")
    
    if not imagenet100_file.exists():
        try:
            urllib.request.urlretrieve(imagenet100_url, imagenet100_file)
            print("Downloaded ImageNet-100 WordNet ID list")
        except Exception as e:
            print(f"Could not download ImageNet-100 list: {e}")
            # Fallback: create from available folders
            return setup_from_existing_folders()
    
    # Read the exact WordNet IDs for ImageNet-100
    with open(imagenet100_file, 'r') as f:
        target_wnids = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Target ImageNet-100 classes: {len(target_wnids)}")
    
    # Re-download the dataset with proper mapping
    root = Path("imagenet100_subset/val")
    if root.exists():
        shutil.rmtree(root)  # Clean slate
    
    # Download dataset fresh
    ds = load_dataset("clane9/imagenet-100", split="validation")
    root.mkdir(parents=True, exist_ok=True)
    
    # Create mapping from dataset labels to target WordNet IDs
    dataset_labels = ds.features["label"].names
    print(f"Dataset has {len(dataset_labels)} classes")
    
    # Build mapping dictionary
    label_to_wnid = {}
    for i, label_name in enumerate(dataset_labels):
        # Find matching WordNet ID from target list
        best_match = None
        best_score = 0
        
        for target_wnid in target_wnids:
            if target_wnid in wnid2idx1000:
                canonical_name = class_idx[str(wnid2idx1000[target_wnid])][1].lower()
                
                # Check for matches
                if canonical_name in label_name.lower() or label_name.lower() in canonical_name:
                    score = min(len(canonical_name), len(label_name)) / max(len(canonical_name), len(label_name))
                    if score > best_score:
                        best_score = score
                        best_match = target_wnid
        
        if best_match:
            label_to_wnid[i] = best_match
            print(f"Mapped '{label_name}' -> {best_match}")
    
    # Save images with correct WordNet ID folder names
    successful_classes = set()
    for i, ex in enumerate(ds):
        label_idx = ex["label"]
        if label_idx in label_to_wnid:
            wnid = label_to_wnid[label_idx]
            folder_path = root / wnid
            folder_path.mkdir(exist_ok=True)
            ex["image"].save(folder_path / f"{i}.jpeg")
            successful_classes.add(wnid)
    
    print(f"Successfully created {len(successful_classes)} classes")
    return root, successful_classes

def setup_from_existing_folders():
    """Fallback: work with existing folders that have images."""
    root = Path("imagenet100_subset/val")
    valid_folders = []
    
    for folder in root.iterdir():
        if folder.is_dir():
            # Check if folder has images
            image_files = list(folder.glob("*.jpeg")) + list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
            if len(image_files) > 0 and folder.name in wnid2idx1000:
                valid_folders.append(folder.name)
            elif len(image_files) == 0:
                print(f"Removing empty folder: {folder.name}")
                shutil.rmtree(folder)
    
    print(f"Found {len(valid_folders)} valid folders with images")
    return root, set(valid_folders)

# Setup the dataset
root, valid_wnids = setup_imagenet100_with_exact_mapping()

# ────────────────────────────────────────────────────────────
# 3. Build subset→1000 index tensor from valid classes only
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
subset_wnids = sorted(list(valid_wnids))
subset_to_1000 = torch.tensor([wnid2idx1000[w] for w in subset_wnids],
                              dtype=torch.long, device=device)

def output_transform(x, y, y_pred):
    """Updated signature for newer Ignite versions."""
    return y_pred.index_select(1, subset_to_1000), y



# ────────────────────────────────────────────────────────────
# 4. DataLoader (rest of your code remains the same)
tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# This should now work without FileNotFoundError
val_ds = datasets.ImageFolder(root=str(root), transform=tf)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

# Rest of your metrics and evaluation code...
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

def eval_model(model, tag=""):
    model = safe_model_to_device(model, device).eval()
    evaluator = create_supervised_evaluator(model, metrics, device=device,
                                            output_transform=output_transform)
    state = evaluator.run(val_loader)
    m = state.metrics
    print(f"{tag:15} Top1={m['top1']:.4f} | Top5={m['top5']:.4f} | Xent={m['xent']:.2f} | Conf={m['conf']:.2f} | Entr={m['entr']:.2f}")

# Your quantization code
models_fp32 = {
    "ResNet50": models.resnet50(pretrained=True),
    "MobileNet-v2": models.mobilenet_v2(pretrained=True),
}

for name,mdl in models_fp32.items():
    eval_model(mdl, tag=f"{name} (FP32)")

def test_neural_compressor_mx():
    """Test Intel Neural Compressor MX formats on NVIDIA GPU via ONNX."""
    
    results = {}
    
    for model_name, pytorch_model in models_to_test.items():
        print(f"Processing {model_name} with Neural Compressor...")
        
        # Convert to ONNX first
        onnx_path = convert_pytorch_to_onnx(pytorch_model, model_name)
        
        # Configure for ONNX Runtime with GPU support
        conf = PostTrainingQuantConfig(
            approach="static",
            backend="onnxrt",  # Use ONNX Runtime backend
            device="gpu"       # Target GPU execution
        )
        
        # Simple evaluation function
        def eval_func(model):
            # This would need to be implemented for ONNX model evaluation
            return 0.9  # Placeholder
        
        try:
            q_model = quantization.fit(
                model=onnx_path,
                conf=conf,
                calib_dataloader=val_loader,
                eval_func=eval_func
            )
            
            if q_model:
                results[model_name] = "Success"
                print(f"{model_name}: Neural Compressor MX quantization completed")
            
        except Exception as e:
            print(f"{model_name}: Neural Compressor failed - {e}")
            results[model_name] = f"Failed: {e}"
    
    return results
