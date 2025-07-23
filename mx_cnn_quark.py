# pip install amd-quark datasets ignite torchvision

import json, urllib.request, torch
from datasets import load_dataset
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from ignite.engine import create_supervised_evaluator
from ignite.metrics import TopKCategoricalAccuracy, Loss, Metric
from quark.torch.quantization.config.config import Config, QuantizationSpec, QuantizationConfig
from quark.torch.quantization.config.type import Dtype
from quark.torch.quantization.observer.observer import PerBlockMXObserver
from quark.torch.quantization import ModelQuantizer

# (Download/ImageNet‑100 setup & mapping as above) ...

for fmt, e in [("MXINT8", Dtype.mx_int8),
               ("MXFP8_E4M3", Dtype.mx_fp8_e4m3),
               ("MXFP6_E3M2", Dtype.mx_fp6_e3m2)]:
    spec = QuantizationSpec(dtype=Dtype.mx, mx_element_dtype=e,
                            observer_cls=PerBlockMXObserver,
                            group_size=32, ch_axis=1)
    cfg = Config(global_quant_config=QuantizationConfig(weight=spec))
    for name, fp32_model in {"ResNet50": models.resnet50(pretrained=True),
                             "MobileNet-v2": models.mobilenet_v2(pretrained=True)}.items():
        quant_model = ModelQuantizer(cfg).quantize_model(fp32_model.cuda(), calib_dataloader=val_loader)
        eval_model(quant_model, tag=f"{name} ({fmt})")
