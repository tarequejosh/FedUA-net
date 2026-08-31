import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import fedua_net as m

net = m.ClientNet(num_classes=4, backbone=m.cfg.BACKBONE, emb=m.cfg.EMB, dropout=m.cfg.DROPOUT)
body = net.body

total_params = sum(p.numel() for p in body.parameters())
bn_only = m.bn_param_names(body)
bn_count = sum(p.numel() for n, p in body.named_parameters() if n in bn_only)

m.cfg.PERSONALIZE_DEEP = True
deep_only = m.deep_param_names(body)
deep_count = sum(p.numel() for n, p in body.named_parameters() if n in deep_only)

uploaded_before = total_params - bn_count                    # current 'fedua' uniform behavior
uploaded_after  = total_params - bn_count - deep_count        # with --personalize_deep

def mb(n_params, bytes_per_param):
    return n_params * bytes_per_param / (1024**2)

print(f"Total body params:          {total_params:,}")
print(f"BN-only params (never sent):{bn_count:,}")
print(f"Deep-personalize params (never sent, new): {deep_count:,}")
print(f"Uploaded params BEFORE (fedua uniform):  {uploaded_before:,}  ->  fp32 {mb(uploaded_before,4):.2f} MB / fp16 {mb(uploaded_before,2):.2f} MB")
print(f"Uploaded params AFTER (personalize_deep): {uploaded_after:,}  ->  fp32 {mb(uploaded_after,4):.2f} MB / fp16 {mb(uploaded_after,2):.2f} MB")
print(f"Relative payload reduction: {100*(1 - uploaded_after/uploaded_before):.2f}%")
