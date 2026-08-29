# FedUA-Net — Literature Review Notes

**Project:** FedUA-Net — Federated Uncertainty-Aware Network for multi-modal medical image classification
**Date:** 2026-08-18
**Scope:** 140 verified references across 4 topic clusters (see `references_master.bib` / `.csv` / `.ris`)
**Target journal:** Medical Image Analysis (MedIA) primary; TMI / npj Digital Medicine alternatives

---

## 1. Federated Learning in Medical Imaging (Surveys & Applications)

### 1.1 Foundational FL works (must-cite)
- **FedAvg** [mcmahan2017communication]: the canonical aggregation algorithm; all FL builds on it. AISTATS 2017.
- **Kairouz et al. "Advances and Open Problems in FL"** [kairouz2021advances]: defines the non-IID taxonomy (covariate/feature shift, label skew, quantity skew, concurrent skew). *Cite for formal definitions of feature shift + label skew.*
- **Rieke et al. npj Digital Medicine 2020** [rieke2020future]: the canonical FL-for-healthcare perspective. *Cite for medical motivation.*

### 1.2 Medical imaging FL surveys
- **Guan et al. Pattern Recognition 2024** [guan2024federated] — comprehensive survey of FL for medical image analysis.
- **Kaissis et al. Nature Machine Intelligence 2020** [kaissis2020secure] — privacy-preserving federated ML in imaging.
- **Darzidehkalani JACR 2022 (Part I & II)** [darzidehkalani2022federatedI, darzidehkalani2022federatedII] — multi-centric health-care ecosystems.
- **Koutsoubis et al. Radiology AI 2025** [koutsoubis2025privacy] — review combining FL + uncertainty (directly supports the MC-dropout/conformal angle).
- **Sohan 2023, Sandhu 2023, Rehman BJR 2023, Raza CMPB 2025** [sohan2023systematic, sandhu2023medical, rehman2023federated, raza2025federated] — additional surveys.

### 1.3 FL applied to medical tasks (real deployments / demonstrations)
- **Sheller et al. Sci Rep 2020** [sheller2020federated]: real multi-institution brain-tumor segmentation; FL ≈ pooled centralized. *Core motivation for hospital collaboration.*
- **Li et al. 2019 brain tumour** [li2019privacy]: early FL on brain MRI.
- **Dou et al. 2021 COVID CT** [dou2021federated]: FL for COVID-19.
- **EXAM / Dayan et al. 2021** [dayan2021federated]: worldwide FL for COVID-19 outcome prediction.
- **FeTS 2022** [pati2022federated]: federated brain tumor segmentation challenge.
- **FLOW 2023** [ogierduterrail2023federated]: multi-task federated learning.
- **Sarma 2021, Adnan 2022, Swarm Learning 2021** [sarma2021federated, adnan2022federated, warnatherresthal2021swarm].
- **Albalawi 2024** [albalawi2024integrated]: brain MRI classification in FL.
- **DMFL_Net 2023** [malik2023dmfl]: COVID X-ray FL.
- **DRFL 2023** [mohan2023drfl]: decentralized FL.

### 1.4 Healthcare FL surveys
- **Xu et al. JHIR 2021** [xujie2020federated]: healthcare informatics FL survey.
- **Nguyen 2022, Antunes 2022, Pfitzner 2021, Joshi 2022, Teo 2024, Zhang Patterns 2024, Pati Patterns 2024, Rahman 2022, Aouedi 2022** [nguyen2022federated, antunes2022federated, pfitzner2021federated, joshi2022federated, teo2024federated, zhangfan2024recent, pati2024privacy, rahman2022federated, aouedi2022handling].
- **Che 2023 Multimodal FL survey** [che2023multimodal]: relevant to our multi-modality setting.
- **Lin 2023** [lin2023federated].

---

## 2. Personalized FL & Non-IID Statistical Heterogeneity

### 2.1 Our two architectural pillars
- **FedPer** [arivazhagan2019federated]: shared base + personalized head. *Origin of FedUA-Net's shared-body/local-head design.*
- **FedBN** [lixiaoxiao2021fedbn]: keep BatchNorm local (never aggregated). *Origin of FedUA-Net's BN-not-aggregated design; handles feature shift.*

### 2.2 Our baselines
- **FedAvg** [mcmahan2017communication]
- **FedProx** [litian2020fedprox]: proximal term limits client drift.
- **Ditto** [litian2021ditto]: personalization via proximal term on global; improves fairness & robustness. *Our closest accuracy competitor — key differentiator is calibrated uncertainty.*
- **FedBABU** [oh2022fedbabu]: freeze head during FL, fine-tune locally.

### 2.3 Shared-representation personalization theory
- **FedRep** [collins2021fedrep]: proves convergence to shared representation + local heads. *Best theoretical anchor for FedPer-style design.*
- **FedRoD** [chen2022fedrod]: balanced-softmax decoupling handles label skew.
- **LG-FedAvg** [liang2020lgfedavg]: local + global representations.
- **pFedMe** [dinh2022pfedme], **Per-FedAvg** [fallah2020perfedavg] (meta-learning), **APFL** [deng2020apfl], **FedALA** [zhang2023fedala], **pFL-Bench** [chen2023pflbench] (standardized PFL benchmark).

### 2.4 Non-IID data challenges
- **Zhao et al. 2018** [zhao2018noniid]: first systematic study; accuracy drops up to 55% under skew; "weight divergence" via EMD.
- **Hsu et al. 2019** [hsu2019measuring]: continuous "identicalness" metric for synthesizing non-IID partitions. *Standard partition protocol.*
- **Li et al. ICDE 2022** [li2022noniidsilos]: largest experimental study of non-IID; label distribution skew most harmful in silos. *Direct evidence for our label-skew scenario.*
- **FedRS** [li2021fedrs]: restricted softmax for label skew (missing classes).
- **Zhang et al. ICML 2022** [zhang2022logits]: logits calibration under label skew.
- **Qu et al. JBHI 2022** [qu2022data]: data heterogeneity in FL for MEDICAL imaging (closest study to our setting); proposes BN averaging. *Must-cite for related work positioning.*
- **FedAlign** [mendieta2022local]: local learning generality mitigates non-IID.
- **Casella et al. TNNLS 2024** [casella2023normalization]: BN vs GroupNorm under non-IID FL.

### 2.5 Heterogeneity theory
- **Li et al. ICLR 2020** [li2020convergence]: O(1/T) convergence of FedAvg under non-IID.
- **SCAFFOLD** [karimireddy2020scaffold]: client drift correction.
- **FedDyn** [acar2021feddyn], **FedNova** [wang2020fednova], **MOCHA** [smith2017mocha] (multi-task ancestor of personalization).

### 2.6 Surveys
- **Tan et al. TNNLS 2023** [tan2023towards]: most-cited PFL survey; standard taxonomy.
- **Kulkarni 2020** [kulkarni2020survey], **Ye et al. ACM CSUR 2024** [ye2023heterogeneous].
- **Huang et al. AAAI 2021** [huang2021personalized]: personalized CROSS-SILO FL — directly matches our 3-hospital scenario.

---

## 3. Uncertainty Quantification, Calibration & Conformal Prediction

### 3.1 Epistemic vs aleatoric foundations
- **Kendall & Gal NeurIPS 2017** [kendall2017uncertainties]: aleatoric vs epistemic distinction.
- **Hüllermeier & Waegeman ML 2021** [hullermeier2021aleatoric]: formal definitions.
- **Abdar et al. Inf. Fusion 2021** [abdar2021review]: most-cited UQ survey.
- **Gawlikowski et al. AI Review 2023** [gawlikowski2023survey]: taxonomy of uncertainty methods.

### 3.2 MC-dropout (our uncertainty mechanism)
- **Gal & Ghahramani ICML 2016** [gal2016dropout]: THE MC-dropout paper. *We use dropout 0.30; entropy from MC samples = epistemic uncertainty signal.*
- **Kendall et al. BMVC 2015 (Bayesian SegNet)** [kendall2015bayesian]: pixel-wise MC-dropout uncertainty.

### 3.3 Calibration
- **Guo et al. ICML 2017** [guo2017calibration]: defines ECE, reliability diagrams, temperature scaling. *We apply temperature scaling post-FL.*
- **Naeini et al. AAAI 2015 (BBQ)** [naeini2015bbq]: origin of ECE formulation.
- **Brier 1950** [brier1950verification]: Brier score (we report).
- **Kull et al. NeurIPS 2019 (Dirichlet)** [kull2019dirichlet], **Minderer et al. NeurIPS 2021** [minderer2021revisiting], **Kumar et al. NeurIPS 2019 (Verified)** [kumar2019verified], **Vaicenavicius et al. AISTATS 2019** [vaicenavicius2019evaluating].
- **Deep ensembles alternative** [lakshminarayanan2017ensembles]: baseline to compare against MC-dropout.
- **Ovadia et al. NeurIPS 2019** [ovadia2019trust]: calibration degrades under dataset shift — motivates evaluating under cross-client shift.

### 3.4 Conformal prediction (our prediction-set mechanism)
- **Vovk et al. 2005 book** [vovk2005algorithmic]: foundations.
- **Shafer & Vovk JMLR 2008** [shafer2008tutorial]: distribution-free marginal coverage guarantees.
- **Angelopoulos & Bates FnT-ML 2023** [angelopoulos2023gentle]: definitive modern tutorial.
- **Fontana et al. Bernoulli 2023** [fontana2023conformal]: taxonomy.
- **Lei et al. JASA 2018** [lei2018distribution]: split conformal guarantees.
- **APS — Romano, Sesia & Candès NeurIPS 2020** [romano2020aps]: **the method we implement** (adaptive prediction sets with valid, adaptive coverage).
- **RAPS** [angelopoulos2021raps]: regularized APS for large label sets.
- **Barber et al. 2023** [barber2023beyond]: conformal beyond exchangeability (federated/drifted data).
- **Gibbs & Candès NeurIPS 2021** [gibbs2021adaptive]: adaptive conformal under distribution shift.

### 3.5 Conformal in medicine & federated settings
- **Olsson et al. Nat Comm 2022** [olsson2022diagnostic]: clinical pathology conformal — flagship medical precedent.
- **Lu et al. AAAI 2022** [lu2022fair]: fair conformal for medical imaging.
- **Vazquez & Facelli JHIR 2022** [vazquez2022conformal]: clinical conformal review.
- **Lu et al. ICML 2023** [lu2023federated]: federated conformal predictors (CLOSEST prior work to us).
- **Humbert et al. ICML 2023** [humbert2023oneshot]: one-shot federated conformal.
- **Plassier et al. ICML 2023** [plassier2023conformal]: label-shift-aware federated conformal.
- **Lu & Kalpathy-Cramer FL-AAAI 2022** [lu2022distributionfree]: federated conformal on MedMNIST; correlates entropy with set size.

### 3.6 Medical imaging uncertainty methods
- **Mehrtash et al. TMI 2020** [mehrtash2021confidence]: MC-dropout + calibration for medical segmentation — closest methodology.
- **Nair et al. MedIA 2020** [nair2020uncertainty]: entropy-based MC-dropout uncertainty in MS lesion segmentation.
- **Jungo & Reyes MICCAI 2019** [jungo2019assessing]: caveat that MC-dropout uncertainty is not always reliability-correlated.
- **Kompa et al. npj DM 2021** [kompa2021second]: clinical motivation for uncertainty-aware ML.

---

## 4. Architectures, Attention, Datasets & Transfer Learning

### 4.1 Backbone
- **EfficientNet ICML 2019** [tan2019efficientnet]: compound scaling.
- **EfficientNetV2 ICML 2021** [tan2021efficientnetv2]: our exact backbone (EfficientNetV2-S, ImageNet-pretrained).

### 4.2 Attention
- **CBAM ECCV 2018** [woo2018cbam]: channel+spatial attention — our exact module.
- **SE-Net CVPR 2018** [hu2018senet]: channel-attention predecessor.
- **Guo et al. 2022 attention survey** [guo2022attention].

### 4.3 Dataset-origin papers (NON-NEGOTIABLE citations)
- **Brain tumor MRI (Figshare) — Cheng et al. PLOS ONE 2015** [cheng2015enhanced]: 3064 T1-CE slices, 233 patients, 4 classes. *Origin of our C0.*
- **BUSI — Al-Dhabyani et al. Data in Brief 2020** [aldhabyani2020dataset]: 780 ultrasound images (437 benign / 210 malignant / 133 normal). *Origin of our C1.*
- **COVIDx — Wang & Wong Sci Rep 2020** [wang2020covidnet] (+ CXR-2 [pavlova2022covidnetcxr2]).
- **COVID-19 Radiography Database — Chowdhury et al. IEEE Access 2020** [chowdhury2020ai] **AND Rahman et al. CBM 2021** [rahman2021exploring] (defines the exact 4-class scheme: covid, lung_opacity, normal, pneumonia — our C2).

### 4.4 Task baselines (dataset-specific)
- Brain: **Sultan et al. 2019** [sultan2019multi], **Badža & Barjaktarović 2020** [badza2020classification], **Deepak & Ameer 2019** [deepak2019brain] (transfer learning), **Díaz-Pernas 2021** [diazpernas2021deep].
- BUSI: **Ayana et al. 2022** [ayana2022novel], **AAU-Net TMI 2023** [chen2023aau].
- COVID: **Marques et al. 2020** [marques2020automated] (EfficientNet for COVID).

### 4.5 Benchmarks & transfer learning
- **MedMNIST 2021 / v2 2023** [yang2021medmnist, yang2023medmnistv2].
- **Tajbakhsh et al. TMI 2016** [tajbakhsh2016convolutional]: fine-tuning > from-scratch in medical imaging — justifies ImageNet pretraining.
- **Yosinski et al. 2014** [yosinski2014transferable]: feature transferability theory.
- **Zhuang et al. Proc IEEE 2021** [zhuang2021survey]: transfer learning taxonomy.
- **Litjens et al. MedIA 2017** [litjens2017survey]: most-cited medical DL survey.

---

## 5. Positioning FedUA-Net (for Related Work / Contributions)

1. **Gap in FL + uncertainty:** existing FL-for-medicine work focuses on accuracy/privacy; few methods provide calibrated, distribution-free uncertainty with formal coverage guarantees in the federated setting [lu2023federated, humbert2023oneshot, plassier2023conformal]. FedUA-Net combines MC-dropout entropy + temperature scaling + APS conformal prediction **within** a personalized FL architecture.
2. **Personalization for multi-modal disjoint-label hospitals:** FedPer-style local heads [arivazhagan2019federated, collins2021fedrep] + FedBN local BatchNorm [lixiaoxiao2021fedbn] jointly handle label skew (disjoint label sets) + feature shift (different modalities). Qu et al. [qu2022data] is the closest experimental study and motivates BN handling in medical FL.
3. **Uncertainty as differentiator vs Ditto/FedBABU:** even where accuracy ties [litian2021ditto, oh2022fedbabu], FedUA-Net provides calibrated confidence and valid adaptive prediction sets, aligning with clinical-uncertainty motivation [kompa2021second, olsson2022diagnostic].
4. **Cross-silo realism:** 3 hospitals, disjoint modalities + labels matches the personalized cross-silo setting [huang2021personalized] better than typical cross-device benchmarks.

---

## 6. Suggested Citation Order for the Intro / Related Work

1. Medical DL context: litjens2017survey, tajbakhsh2016convolutional
2. FL foundations: mcmahan2017communication, kairouz2021advances
3. FL for medicine: rieke2020future, sheller2020federated, xu2020federated, guan2024federated
4. Non-IID problem: zhao2018noniid, hsu2019measuring, li2022noniidsilos, qu2022data
5. Personalization: tan2023towards, arivazhagan2019federated, litian2021ditto, litian2020fedprox, oh2022fedbabu, lixiaoxiao2021fedbn, collins2021fedrep
6. Uncertainty: gal2016dropout, kendall2017uncertainties, guo2017calibration, abdar2021review
7. Conformal: shafer2008tutorial, romano2020aps, angelopoulos2023gentle, lu2023federated
8. Backbones: tan2021efficientnetv2, woo2018cbam
9. Datasets: cheng2015enhanced, aldhabyani2020dataset, chowdhury2020ai, rahman2021exploring