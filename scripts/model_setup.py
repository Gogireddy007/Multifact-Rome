"""Shared clean GPT-2 XL loader for baseline and causal-tracing work."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROME = ROOT / 'vendor/rome'
MODEL_ID = 'gpt2-xl'
MODEL_REVISION = '15ea56dee5df4983c59b2538573817e1667135e2'

def prepare_rome():
    os.environ['MPLBACKEND'] = 'Agg'
    if not (ROME/'globals.yml').exists():
        raise RuntimeError('Run scripts/setup.py first.')
    sys.path.insert(0, str(ROME))
    # Upstream util.globals reads globals.yml relative to the current directory.
    os.chdir(ROME)

def load_model_and_tokenizer():
    prepare_rome()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('An NVIDIA CUDA GPU is required.')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(32)
    torch.cuda.manual_seed_all(32)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float32,
        attn_implementation='eager', use_safetensors=True).eval().cuda()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

if __name__ == '__main__':
    model, tokenizer = load_model_and_tokenizer()
    from experiments.causal_trace import ModelAndTokenizer
    mt = ModelAndTokenizer(model=model, tokenizer=tokenizer)
    print(mt)
    print('MODEL READY for causal tracing')
