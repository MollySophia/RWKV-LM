import torch
import types

# Load checkpoint
checkpoint_path = "out/qat-rwkv7-g1d-0.1b-test/rwkv-final.pth"
w = torch.load(checkpoint_path, map_location='cpu')

# Find all q_scale keys
q_scale_keys = [k for k in w.keys() if '.q_scale' in k]
print(f"Found {len(q_scale_keys)} q_scale keys")

# Check a few scales
for k in q_scale_keys[:5]:
    scale = w[k]
    print(f"\n{k}:")
    print(f"  Shape: {scale.shape}")
    print(f"  Min: {scale.min().item():.6f}, Max: {scale.max().item():.6f}")
    print(f"  Mean: {scale.mean().item():.6f}")
    print(f"  Has zero: {(scale == 0).any().item()}")
    print(f"  Has nan: {torch.isnan(scale).any().item()}")
    print(f"  Has inf: {torch.isinf(scale).any().item()}")

# Check module names (Linear layers)
linear_keys = [k for k in w.keys() if '.weight' in k and 'emb.' not in k and 'ln' not in k]
print(f"\n\nFound {len(linear_keys)} weight keys")
for k in linear_keys[:10]:
    print(f"  {k}")

# Check if module names match q_scale names
print("\n\nMatching check:")
for wk in linear_keys[:5]:
    module_name = wk.replace('.weight', '')
    qk = module_name + '.q_scale'
    if qk in q_scale_keys:
        print(f"  OK: {module_name}")
    else:
        print(f"  MISSING: {module_name}")
