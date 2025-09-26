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


# ==================== Device ====================
device = torch.device("cuda")

# ==================== Data ====================
data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"][:1000]

# data = load_dataset("bigcode/the-stack", data_dir="data/python", split="train[:1000]") // more code related dataset

def make_loader(tokenizer, block_size=16, batch_size=1):
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

        safe_model_to_device(model, device)
        loader = make_loader(tok)
        res = eval_model(model, tok, loader)
        pretty(name, tag, res)
        del model
        torch.cuda.empty_cache()
        gc.collect()

# ==================== MX specs ====================
mx_formats = ["int8", "fp8_e4m3", "fp6_e3m2", "fp4_e2m1"]

models_fp32 = {
    # "bert-base-uncased":
    #     (AutoModelForMaskedLM.from_pretrained("bert-base-uncased"),
    #      AutoTokenizer.from_pretrained("bert-base-uncased")),
"phi-1_5":
    (AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", torch_dtype=torch.float32),
     AutoTokenizer.from_pretrained("microsoft/phi-1_5")),
#      "tinystories-33M":
#     (AutoModelForCausalLM.from_pretrained(
#     "roneneldan/TinyStories-33M",
#     torch_dtype=torch.float32,
#     trust_remote_code=True,
#     use_safetensors=True
# ),
#      AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M", use_fast=True)
# ),

}

print("\n=== FP32 Evaluation ===")
run_eval(models_fp32, tag="(FP32)")

# ==================== MX Formats Evaluation ====================
for mx_format in mx_formats:
    print(f"\n=== MX Format: {mx_format.upper()} ===")
    mx_cfg = finalize_mx_specs({
        "w_elem_format": mx_format,
        "a_elem_format": mx_format,
        "scale_bits": 8,
        "block_size": 64,
        "custom_cuda": True,
        "quantize_backprop": False,
    })
    inject_pyt_ops(mx_cfg)

    models_mx = {
        # "bert-base-uncased":
        #     (AutoModelForMaskedLM.from_pretrained("bert-base-uncased"),
        #      AutoTokenizer.from_pretrained("bert-base-uncased")),
"phi-1_5":
    (AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", torch_dtype=torch.float32, trust_remote_code=True,
    use_safetensors=True),
     AutoTokenizer.from_pretrained("microsoft/phi-1_5")),
#         "tinystories-33M":
#     (AutoModelForCausalLM.from_pretrained(
#     "roneneldan/TinyStories-33M",
#     torch_dtype=torch.float32,
#     trust_remote_code=True,
#     use_safetensors=True,
# ),
#      AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M", use_fast=True)
# ),
    }
    run_eval(models_mx, tag=f"(MX-{mx_format.upper()})")
print("\n" + "="*60)
print("=== Intel Neural Compressor + ONNX Runtime Quantization ===")
print("="*60)

def convert_pytorch_to_onnx(model, model_name, device):
    """Convert PyTorch model to ONNX format."""
    print(f"\n🔄 Converting {model_name} to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    onnx_path = f"{model_name}_imagenet.onnx"
    
    try:
        torch.onnx.export(
            model, dummy_input, onnx_path,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=11,
            verbose=False
        )
        print(f"  ✅ Successfully saved to {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"  ❌ ONNX conversion failed: {e}")
        return None

def evaluate_onnx_model_detailed(onnx_path, tag, max_batches=100):
    """Evaluate ONNX model with detailed metrics matching your original format."""
    print(f"\n📊 Evaluating {tag}...")
    
    try:
        import onnxruntime as ort
        
        # Setup providers
        providers = []
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
            print(f"  🖥️  Using CUDA for {tag}")
        providers.append('CPUExecutionProvider')
        
        session = ort.InferenceSession(onnx_path, providers=providers)
        input_name = session.get_inputs()[0].name
        
        # Metrics tracking
        correct_top1 = 0
        correct_top5 = 0
        total_loss = 0.0
        total_conf = 0.0
        total_entr = 0.0
        total_samples = 0
        num_batches = 0
        
        criterion = torch.nn.CrossEntropyLoss()
        
        print(f"  🔍 Running evaluation on up to {max_batches} batches...")
        
        for batch_idx, (data, target) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
                
            # ONNX inference
            data_np = data.cpu().numpy()
            outputs = session.run(None, {input_name: data_np})
            full_predictions = torch.tensor(outputs[0])
            
            # Apply ImageNet subset mapping (same as your output_transform)
            subset_predictions = full_predictions.index_select(1, subset_to_1000)
            
            # Calculate metrics
            _, pred_top1 = subset_predictions.topk(1, 1, True, True)
            _, pred_top5 = subset_predictions.topk(5, 1, True, True)
            
            correct_top1 += pred_top1.eq(target.view_as(pred_top1)).sum().item()
            correct_top5 += pred_top5.eq(target.view(-1, 1).expand_as(pred_top5)).sum().item()
            
            # Loss calculation
            loss = criterion(subset_predictions, target)
            total_loss += loss.item()
            
            # Confidence (max softmax probability)
            probs = torch.softmax(subset_predictions, 1)
            conf = probs.max(1)[0].mean().item()
            total_conf += conf
            
            # Entropy
            entropy = -(probs * (probs + 1e-12).log()).sum(1).mean().item()
            total_entr += entropy
            
            total_samples += target.size(0)
            num_batches += 1
        
        # Calculate final metrics
        top1_acc = correct_top1 / total_samples if total_samples > 0 else 0.0
        top5_acc = correct_top5 / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_conf = total_conf / num_batches if num_batches > 0 else 0.0
        avg_entr = total_entr / num_batches if num_batches > 0 else 0.0
        
        # Print in same format as your original eval_model
        print(f"{tag:25} Top1={top1_acc:.4f} | Top5={top5_acc:.4f} | Xent={avg_loss:.2f} | Conf={avg_conf:.2f} | Entr={avg_entr:.2f}")
        
        return top1_acc
        
    except Exception as e:
        print(f"  ❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

# Process each model
for name, fp32_model in models_fp32.items():
    print(f"\n{'='*50}")
    print(f"🚀 Processing {name}")
    print(f"{'='*50}")
    
    try:
        # Step 1: Convert to ONNX
        onnx_path = convert_pytorch_to_onnx(fp32_model[0], name, device)
        
        if not onnx_path:
            print(f"❌ Skipping {name} due to ONNX conversion failure")
            continue
        
        # Step 2: Evaluate original ONNX model (baseline)
        print(f"\n📋 Step 1: Baseline ONNX Evaluation")
        original_acc = evaluate_onnx_model_detailed(onnx_path, f"{name} (ONNX-Original)")
        
        # Step 3: Quantize with Intel Neural Compressor
        print(f"\n⚙️  Step 2: Intel Neural Compressor Quantization")
        quantized_path = f"{name}_quantized.onnx"
        
        # Import Neural Compressor components
        try:
            from neural_compressor.config import PostTrainingQuantConfig
            from neural_compressor import quantization
            
            print(f"  📦 Neural Compressor imported successfully")
            
            # Configure quantization for ONNX backend
            conf = PostTrainingQuantConfig(
                approach="static",
                backend="onnxrt"
            )
            
            print(f"  🔧 Configuration created: static quantization with ONNX Runtime backend")
            
            # Create evaluation function for Neural Compressor
            def inc_eval_func(model_path):
                """Quick evaluation function for INC calibration."""
                try:
                    import onnxruntime as ort
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    session = ort.InferenceSession(model_path, providers=providers)
                    input_name = session.get_inputs()[0].name
                    
                    correct = 0
                    total = 0
                    
                    for batch_idx, (data, target) in enumerate(val_loader):
                        if batch_idx >= 10:  # Quick evaluation
                            break
                            
                        data_np = data.cpu().numpy()
                        outputs = session.run(None, {input_name: data_np})
                        full_pred = torch.tensor(outputs[0])
                        subset_pred = full_pred.index_select(1, subset_to_1000)
                        predictions = subset_pred.argmax(dim=1)
                        
                        correct += (predictions == target).sum().item()
                        total += target.size(0)
                    
                    accuracy = correct / total if total > 0 else 0.0
                    print(f"    📈 INC calibration accuracy: {accuracy:.4f}")
                    return accuracy
                except Exception as e:
                    print(f"    ❌ INC evaluation error: {e}")
                    return 0.0
            
            print(f"  🎯 Starting quantization process...")
            
            # Perform quantization
            quantized_model = quantization.fit(
                model=onnx_path,
                conf=conf,
                eval_func=inc_eval_func
            )
            
            if quantized_model is not None:
                print(f"  ✅ Quantization completed successfully!")
                
                # Handle different return types
                if isinstance(quantized_model, str):
                    final_quantized_path = quantized_model
                else:
                    # Try to save the quantized model
                    try:
                        if hasattr(quantized_model, 'save'):
                            quantized_model.save(quantized_path)
                            final_quantized_path = quantized_path
                        else:
                            final_quantized_path = quantized_path
                        print(f"  💾 Quantized model saved to {final_quantized_path}")
                    except:
                        final_quantized_path = quantized_path
                
                # Step 4: Evaluate quantized model
                print(f"\n📊 Step 3: Quantized Model Evaluation")
                quantized_acc = evaluate_onnx_model_detailed(final_quantized_path, f"{name} (INC-Quantized)")
                
                # Calculate and report accuracy drop
                acc_drop = original_acc - quantized_acc
                drop_percent = (acc_drop / original_acc * 100) if original_acc > 0 else 0
                
                print(f"\n📈 {name} Quantization Summary:")
                print(f"  • Original accuracy:  {original_acc:.4f}")
                print(f"  • Quantized accuracy: {quantized_acc:.4f}")
                print(f"  • Accuracy drop:      {acc_drop:.4f} ({drop_percent:.2f}%)")
                
                if acc_drop < 0.02:  # Less than 2% drop
                    print(f"  🎉 Excellent quantization result!")
                elif acc_drop < 0.05:  # Less than 5% drop
                    print(f"  ✅ Good quantization result!")
                else:
                    print(f"  ⚠️  Significant accuracy drop detected")
                
            else:
                print(f"  ❌ Quantization failed - Neural Compressor returned None")
                print(f"  🤔 This might indicate the model couldn't be quantized with current settings")
                
        except ImportError as e:
            print(f"  ❌ Neural Compressor import failed: {e}")
            print(f"  💡 Try: pip install neural-compressor")
            
    except Exception as e:
        print(f"❌ Error processing {name}: {e}")
        print("📋 Full error traceback:")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print("🏁 Quantization Analysis Complete!")
print(f"{'='*60}")
