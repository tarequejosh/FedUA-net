"""Reconstruct per-round fed curves from train_final.log text and plot the figure."""
import re, os, sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log = r'D:/Research/FedUA-Net/outputs_final/train_final.log'
out = r'D:/Research/FedUA-Net/outputs_final'
rows = []
pat = re.compile(r"R\s+(\d+)\s+LRb=[\d.e-]+\s+C0=([\d.]+)\s+C1=([\d.]+)\s+C2=([\d.]+)\s+mean=([\d.]+)")
for line in open(log, encoding='utf-8', errors='ignore'):
    mt = pat.search(line)
    if mt:
        r = mt.groups()
        rows.append({'round': int(r[0]), 'c0_test_acc': float(r[1]),
                     'c1_test_acc': float(r[2]), 'c2_test_acc': float(r[3]),
                     'mean_test_acc': float(r[4])})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(out, 'reports', 'final_fed_log.csv'), index=False)
print(df.to_string(index=False))

# uncertainty final value for the report line
plt.style.use('seaborn-v0_8-whitegrid')
cols = ['#1976D2', '#F57C00', '#388E3C']
names = ['Hospital_A (Brain-Tumor MRI)', 'Hospital_B (Breast Ultrasound)',
         'Hospital_C (COVID-19 X-Ray)']
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
ax = axes[0]
ax.plot(df['round'], df['mean_test_acc'], 'k-o', lw=2.5, label='Mean client test acc')
for c in range(3):
    ax.plot(df['round'], df[f'c{c}_test_acc'], '--', marker='s', ms=5,
            color=cols[c], label=names[c])
ax.set_xlabel('Communication round'); ax.set_ylabel('Test accuracy')
ax.set_title('FedUA-Net - federated learning phase', fontweight='bold')
ax.legend(fontsize=8); ax.set_xticks(df['round']) if len(df) <= 15 else None
ax.set_ylim(0.5, 1.0)

ax = axes[1]
fin = pd.read_csv(os.path.join(out, 'reports', 'final_client_summary.csv'))
bar_names = ['v1\nglobal', 'MobileNetV2\ncentralized', 'v4\npersonalized FL']
vals = [0.6517, 0.7643, fin['accuracy'].mean()]
ax.bar(bar_names, vals, color=['#9E9E9E', '#B0BEC5', '#1976D2'])
for i, v in enumerate(vals):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
ax.set_ylim(0, 1.0); ax.set_ylabel('Mean client accuracy')
ax.set_title('vs prior baselines', fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(out, 'figures', 'final_training_curves.png'), dpi=200)
print('[OK] figure saved')