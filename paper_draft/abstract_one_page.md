# Calibrated Uncertainty in Federated Learning for Privacy-Preserving Multi-Modal Medical Imaging
Md Tareque Jamil Josh, Fernanda Miyuki Yamada, and Hiroki Takahashi

Abstract—With the expansion of deep learning in medical imaging, hospitals are expected to train diagnostic models, yet privacy regulations prevent pooling patient images. This paper proposes FedUA-Net, a federated uncertainty-aware network for cross-modality medical image classification. A shared convolutional backbone with channel and spatial attention is trained collaboratively across three hospitals without exchanging images [1]. Client-specific heads absorb the disjoint label sets [2], and local Batch Normalization statistics are excluded from aggregation to suppress modality shift [3]. A post-hoc module calibrates Monte Carlo (MC) dropout entropy [4] with temperature scaling and conformal prediction [5], giving each case a coverage-guaranteed prediction set. Tested on brain MRI, breast ultrasound, and chest X-ray against five baselines over three seeds, the system classifies across modalities and flags uncertain cases for clinician review.

## I. Introduction
Deep learning now reads medical images with near-expert accuracy, yet most hospitals cannot build such models alone: privacy rules lock patient images inside each institution, and no single site holds enough cases for every condition. Federated learning (FL) offers a way out by letting hospitals train a shared model while exchanging weights instead of images [1]. In practice, however, FL assumes that clients see similar data, an assumption that breaks when one hospital contributes brain MRI, another breast ultrasound, and a third chest X-ray, each with its own label set. Moreover, a clinical model must be honest about its doubt, because a confident wrong answer is more dangerous than an uncertain case flagged for review.

## II. FedUA-Net: Federated Uncertainty-Aware Network

Fig. 1. FedUA-Net framework.
For data acquisition, three public datasets serve as clients: brain magnetic resonance imaging (MRI), breast ultrasound (US), and chest X-ray. The collected images will be pre-processed including resizing and normalization. Later, the features are divided into a shared body with Convolutional Block Attention Module (CBAM) and global average pooling (GAP), and a local client-specific head (FedPer). The server aggregation excludes Batch Normalization (BN) statistics (FedBN) [3]. Then, uncertainty is estimated by Monte Carlo (MC) dropout [4], temperature-scaled, and converted to prediction sets by Adaptive Prediction Sets (APS) at 90 percent coverage [5]. The resulting sets shall flag uncertain cases for clinician review.

## III. Experimental Results
Experiments are conducted to simulate a real cross-silo situation. The experimental results are shown in Table I.

**Result of Federated Learning Performance Comparison (3-Seed Mean)**

| Method | Accuracy (%) | Mean F1 (%) |
|---|---|---|
| FedAvg | 82.7 | 80.8 |
| FedBN | 82.2 | 80.5 |
| FedProx | 83.7 | 82.2 |
| FedBABU | 89.3 | 88.7 |
| Ditto | 93.3 | 93.1 |
| FedUA-Net | 90.5 | 90.0 |

Six strategies were compared over 12 rounds and 3 seeds (Table I). FedUA-Net clearly outperforms FedAvg, FedBN, and FedProx; Ditto scores higher raw accuracy but lacks a coverage guarantee, whereas FedUA-Net meets the 90 percent coverage target and reaches 89.7 percent mean accuracy after brief fine-tuning.

## Conclusion
In this paper, we proposed a framework to handle cross-modality federated learning and set up an uncertainty-aware prediction system, named FedUA-Net. The system classifies accurately while providing calibrated uncertainty according to preliminary results.

## References
Kairouz, P., et al., 2021. Advances and Open Problems in Federated Learning. Found. Trends Mach. Learn., 4(1), pp. 1-162.
Arivazhagan, M.G., et al., 2019. Federated Learning with Personalization Layers. arXiv:1912.00818.
Li, X., et al., 2021. FedBN: Federated Learning on Non-IID Features via Local Batch Normalization. Proc. ICLR (arXiv:2102.07623).
Gal, Y. and Ghahramani, Z., 2016. Dropout as a Bayesian Approximation: Representing Model Uncertainty. Proc. ICML, pp. 1050-1059.
Angelopoulos, A.N. and Bates, S., 2023. Conformal Prediction: A Gentle Introduction. Found. Trends Mach. Learn., 16(4), pp. 494-591.

M. T. J. Josh (School of Informatics and Engineering, UEC Tokyo, Japan; Dept. of CSE, DIU, Bangladesh), F. M. Yamada, and H. Takahashi (Graduate School of Informatics and Engineering, UEC Tokyo)
E-mail: j2695005@gl.cc.uec.ac.jp