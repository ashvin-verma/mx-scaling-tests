from pytorch_cifar_models import cifar100_resnet20, cifar100_mobilenetv2
from neural_compressor import PostTrainingQuantConfig, quantization

# Load pretrained CIFAR-100 models (no modification needed)
models_fp32 = {
    "ResNet-20": cifar100_resnet20(pretrained=True),
    "MobileNet-v2": cifar100_mobilenetv2(pretrained=True),
}

# CIFAR-100 dataloader (32x32 images, no resizing needed)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

val_ds = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

# Quantization (same as your original code)
for name, fp32_model in models_fp32.items():
    inc_cfg = PostTrainingQuantConfig(approach="static", mx_formats=["MXINT8"])
    q_model = quantization.fit(model=fp32_model, conf=inc_cfg,
                               calib_dataloader=val_loader,
                               eval_dataloader=val_loader)
    eval_model(q_model, tag=f"{name} (INC‑MXINT8)")
