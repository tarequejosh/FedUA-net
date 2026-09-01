# Table III: Systematic Factorial Ablation Study (3-Seed Mean ± Std)

| Configuration | Attention Module | Local FT | Hospital A (MRI) | Hospital B (US) | Hospital C (X-Ray) | Multi-Task Mean Acc. (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| FedBN Baseline | None | No | 96.02 ± 0.34 | 54.99 ± 3.00 | 95.65 ± 0.39 | 82.22 ± 0.96 |
| Base Personalization | None | Yes | 96.40 ± 0.31 | 82.91 ± 3.73 | 95.30 ± 0.60 | 91.53 ± 1.19 |
| + Spatial Attention | Spatial | Yes | 95.98 ± 0.32 | 80.06 ± 2.15 | 95.56 ± 0.25 | 90.53 ± 0.68 |
| + Channel Attention | Channel | Yes | 96.00 ± 0.25 | 81.48 ± 3.00 | 95.22 ± 0.22 | 90.90 ± 0.94 |
| **FedUA-Net (Proposed)** | **Dual CBAM** | **Yes** | **96.29 ± 0.28** | **83.48 ± 4.04** | **95.50 ± 0.41** | **91.75 ± 1.34** |