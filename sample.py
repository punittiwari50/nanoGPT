"""
Sample from a trained model
"""
import os
import sys
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# Ensure clean UTF-8 standard output and error across all terminals and environments
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
dataset = '' # optional dataset name override (e.g. 'combined')
configurator_path = os.path.join(os.path.dirname(__file__), 'configurator.py') if '__file__' in globals() else 'configurator.py'
if os.path.exists(configurator_path):
    exec(open(configurator_path).read())
elif os.path.exists('configurator.py'):
    exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
target_dataset = dataset if dataset else (checkpoint.get('config', {}).get('dataset', '') if 'checkpoint' in globals() and 'config' in checkpoint else '')
candidate_meta_paths = [
    os.path.join('data', target_dataset, 'meta.pkl') if target_dataset else '',
    os.path.join(os.path.dirname(__file__), 'data', target_dataset, 'meta.pkl') if ('__file__' in globals() and target_dataset) else '',
    os.path.join(out_dir, 'meta.pkl'),
    os.path.join(out_dir, '..', 'meta.pkl'),
]
for cmp in candidate_meta_paths:
    if cmp and os.path.exists(cmp):
        meta_path = cmp
        load_meta = True
        break
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    def decode(tokens):
        valid_tokens = [t for t in tokens if t < 50257]
        try:
            # Decode token bytes into clean UTF-8 string
            raw_bytes = enc.decode_bytes(valid_tokens)
            decoded_str = raw_bytes.decode('utf-8', errors='replace')
            # Strip dangling replacement characters (\ufffd) caused by isolated byte tokens
            return decoded_str.replace('\ufffd', '')
        except Exception:
            text = enc.decode(valid_tokens)
            return text.replace('\ufffd', '')

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

print("\n" + "=" * 60)
print(f"INPUT PROMPT: {start}")
print("=" * 60, flush=True)

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            full_text = decode(y[0].tolist())
            prompt_token_count = x.shape[1]
            generated_tokens = y[0][prompt_token_count:].tolist()
            completion_text = decode(generated_tokens)
            
            print(f"\n--- [SAMPLE {k+1}/{num_samples}] ---")
            print(">>> GENERATED OUTPUT:")
            print(completion_text)
            print("\n>>> FULL SEQUENCE (PROMPT + GENERATION):")
            print(full_text)
            print("-" * 60, flush=True)
