# ============================================================
# FedUA-Net - PyTorch GPU port
# Federated Uncertainty-Aware Attention Network
# Improvements vs v3 (TF):
#   1) Personalized federated learning: shared feature body
#      (EfficientNetV2-S + CBAM) + per-client local heads
#      (FedPer) -> fixes the incoherent averaged 11-class head.
#   2) FedBN-style: BatchNorm params stay local per client
#      (feature-shift = different imaging modality per hospital).
#   3) Fixed MC-Dropout (active at inference) for uncertainty.
#   4) AMP (fp16) training on GPU, gradient accumulation,
#      per-client LR + class weights, warmup/cosine server LR.
#   5) Per-client + pooled evaluation with temperature scaling.
# ============================================================
import os, sys, gc, time, json, copy, random, warnings, argparse, contextlib
from collections import OrderedDict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
class Config:
    IMG_SIZE     = 224
    BACKBONE     = 'efficientnet_v2_s'
    NUM_CLIENTS  = 3
    COMM_ROUNDS  = 12
    LOCAL_EPOCHS = {0: 4, 1: 10, 2: 2}   # per-client local epochs/round
    BATCH_SIZE   = 32
    GRAD_ACCUM   = 2                      # effective batch = BATCH_SIZE * GRAD_ACCUM
    WORKERS      = 0
    EMB          = 512
    DROPOUT      = 0.30                   # MC dropout rate (feature head)
    CBAM_RATIO   = 8
    CBAM_KERNEL  = 7
    BASE_LR      = 3e-4                   # head / top layers
    BACKBONE_LR  = 1e-4
    MIN_LR       = 5e-6
    AMP          = False                  # fp16 autocast (off = stable fp32)
    WARMUP_ROUNDS = 2
    WD           = 1e-4
    LABEL_SMOOTHING = 0.1
    TEMP_TUNE    = True                   # temperature scaling after training
    MC_ITERS     = 30                     # MC dropout forward passes for UQ
    VAL_FRAC     = 0.15                   # brain tumor: val out of train
    PERSONALIZE_DEEP = False              # Depth-adaptive: keep CBAM + projection local, freeze/aggregate body only
    ULTRASOUND_AUG   = False              # Ultrasound-specific augmentations (SpeckleNoise + ElasticTransform) for Hospital B
    ULTRASOUND_AUG_MILD = False           # Milder ultrasound augmentations (SpeckleNoise(0.04, 0.3) + ElasticTransform(15.0, 4.0))
    AGG_WEIGHT_TYPE = 'uniform'           # 'uniform' (1/K) or 'sample' (N_k/N_total)
    HOSPITAL_B_SUBSET_SIZE = None         # None or int > 0 for data scarcity subsampling
    DATA_ROOT    = os.environ.get('DATA_ROOT', './Dataset')
    OUTPUT_DIR   = os.environ.get('OUTPUT_DIR', './outputs_final')

cfg = Config()

def client_meta(cid):
    """(dataset_name, prefix, classes as on disk, name)"""
    return [
        ('brain_tumor', 'bt',
         ['glioma', 'meningioma', 'notumor', 'pituitary'],
         'Hospital_A (Brain-Tumor MRI)'),
        ('busi', 'bu',
         ['benign', 'malignant', 'normal'],
         'Hospital_B (Breast Ultrasound)'),
        ('covid', 'cr',
         ['covid', 'lung_opacity', 'normal', 'pneumonia'],
         'Hospital_C (COVID-19 X-Ray)'),
    ][cid]

ROOT = Path(cfg.DATA_ROOT)
DATASET_DIR = {
    'brain_tumor': ROOT / 'Brain Tumor MRI Dataset',
    'busi':        ROOT / 'Dataset_BUSI_with_GT',
    'covid':       ROOT / 'COVID-19_Radiography_Dataset',
}

# ------------------------------------------------------------
# Data discovery
# ------------------------------------------------------------
VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def list_images(d):
    d = Path(d)
    if not d.exists():
        return []
    out = []
    for p in sorted(d.rglob('*')):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            if '_mask' in p.stem.lower():
                continue
            out.append(str(p))
    return out

def build_client_rows(cid, smoke=False):
    """Return DataFrame columns: path, label(global idx), split, client"""
    ds_name, prefix, cls_names, _ = client_meta(cid)
    base = DATASET_DIR[ds_name]
    rows = []  # (path, local_class_name)

    if ds_name == 'brain_tumor':
        # Training -> train/val, Testing -> test (use the held-out site test set)
        tr_rows, te_rows = [], []
        for cdir in sorted((base / 'Training').iterdir()):
            if not cdir.is_dir():
                continue
            cname = cdir.name.lower().strip()
            if cname not in cls_names:
                continue
            for p in list_images(cdir):
                tr_rows.append((p, f'{prefix}_{cname}'))
        for cdir in sorted((base / 'Testing').iterdir()):
            if not cdir.is_dir():
                continue
            cname = cdir.name.lower().strip()
            if cname not in cls_names:
                continue
            for p in list_images(cdir):
                te_rows.append((p, f'{prefix}_{cname}'))
        tr_df = pd.DataFrame(tr_rows, columns=['path', 'label'])
        from sklearn.model_selection import train_test_split
        trp, valp, trl, vall = train_test_split(
            tr_df['path'].tolist(), tr_df['label'].tolist(),
            test_size=cfg.VAL_FRAC, random_state=SEED, stratify=tr_df['label'])
        te_df = pd.DataFrame(te_rows, columns=['path', 'label'])
        df = pd.concat([
            pd.DataFrame({'path': trp, 'label': trl, 'split': 'train'}),
            pd.DataFrame({'path': valp, 'label': vall, 'split': 'val'}),
            pd.DataFrame({'path': te_df['path'].tolist(),
                          'label': te_df['label'].tolist(), 'split': 'test'}),
        ], ignore_index=True)
    elif ds_name == 'busi':
        for cdir in sorted(base.iterdir()):
            if not cdir.is_dir():
                continue
            cname = cdir.name.lower().strip()
            if cname not in cls_names:
                continue
            for p in list_images(cdir):
                rows.append((p, f'{prefix}_{cname}'))
        tmp = pd.DataFrame(rows, columns=['path', 'label'])
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(tmp, test_size=0.30, random_state=SEED, stratify=tmp['label'])
        val, te = train_test_split(te, test_size=0.50, random_state=SEED, stratify=te['label'])
        tr['split'] = 'train'; val['split'] = 'val'; te['split'] = 'test'
        df = pd.concat([tr, val, te])
    else:  # covid
        name_map = {'covid': 'covid', 'lung_opacity': 'lung_opacity',
                    'normal': 'normal', 'pneumonia': 'pneumonia',
                    'viral_pneumonia': 'pneumonia'}
        for cdir in sorted(base.iterdir()):
            if not cdir.is_dir():
                continue
            raw = cdir.name.lower().replace('-', '_').replace(' ', '_')
            cname = name_map.get(raw)
            if cname is None:
                continue
            img_dir = cdir / 'images'
            for p in list_images(img_dir if img_dir.exists() else cdir):
                rows.append((p, f'{prefix}_{cname}'))
        tmp = pd.DataFrame(rows, columns=['path', 'label'])
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(tmp, test_size=0.30, random_state=SEED, stratify=tmp['label'])
        val, te = train_test_split(te, test_size=0.50, random_state=SEED, stratify=te['label'])
        tr['split'] = 'train'; val['split'] = 'val'; te['split'] = 'test'
        df = pd.concat([tr, val, te])

    if smoke:
        n = getattr(cfg, 'SMOKE_PER_CLASS', 30)
        keep = []
        for (_s, _l), g in df.groupby(['split', 'label'], sort=False):
            keep.append(g.head(n))
        df = pd.concat(keep, ignore_index=True)
    df['client'] = cid
    return df

def discover_all(smoke=False):
    frames = [build_client_rows(cid, smoke) for cid in range(cfg.NUM_CLIENTS)]
    all_df = pd.concat(frames, ignore_index=True)
    classes = sorted(all_df['label'].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    all_df['gid'] = all_df['label'].map(class_to_idx)
    return all_df, classes, class_to_idx

# ------------------------------------------------------------
# Transforms & Dataset
# ------------------------------------------------------------
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)

class SpeckleNoise(nn.Module):
    """Multiplicative speckle noise for ultrasound simulations: x = x + x * N(0, sigma^2)."""
    def __init__(self, sigma=0.08, p=0.5):
        super().__init__()
        self.sigma = sigma
        self.p = p

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(img) * self.sigma
            img = torch.clamp(img + img * noise, 0.0, 1.0)
        return img

def train_transforms(ultrasound=False):
    tfs = [
        torchvision.transforms.ConvertImageDtype(torch.float),
        torchvision.transforms.Resize((256, 256), antialias=True),
    ]
    if ultrasound == 'mild':
        tfs.extend([
            torchvision.transforms.ElasticTransform(alpha=15.0, sigma=4.0),
            SpeckleNoise(sigma=0.04, p=0.3),
        ])
    elif ultrasound:
        tfs.extend([
            torchvision.transforms.ElasticTransform(alpha=25.0, sigma=4.0),
            SpeckleNoise(sigma=0.08, p=0.5),
        ])
    tfs.extend([
        torchvision.transforms.RandomResizedCrop(cfg.IMG_SIZE, scale=(0.7, 1.0), antialias=True),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(p=0.1),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        torchvision.transforms.Normalize(IMG_MEAN, IMG_STD),
    ])
    return torchvision.transforms.Compose(tfs)

def eval_transforms():
    return torchvision.transforms.Compose([
        torchvision.transforms.ConvertImageDtype(torch.float),
        torchvision.transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE), antialias=True),
        torchvision.transforms.Normalize(IMG_MEAN, IMG_STD),
    ])

class MedImgDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df['path'].tolist()
        self.labels = df['local'].tolist()
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        img = torchvision.io.read_image(self.paths[i])  # (C,H,W) uint8
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] > 3:
            img = img[:3]
        return self.transform(img), self.labels[i]

def build_loaders(df, batch_size, workers, hospital_b_subset_size=None, client_id=None, seed=None):
    loaders = {}
    is_hospital_b = (client_id == 1) or \
                    ('client' in df.columns and (df['client'] == 1).all()) or \
                    (len(df) > 0 and 'busi' in str(df['label'].iloc[0]))
    if is_hospital_b and getattr(cfg, 'ULTRASOUND_AUG_MILD', False):
        use_us_aug = 'mild'
    elif is_hospital_b and getattr(cfg, 'ULTRASOUND_AUG', False):
        use_us_aug = True
    else:
        use_us_aug = False
    for split in ('train', 'val', 'test'):
        sub = df[df['split'] == split]
        if len(sub) == 0:
            loaders[split] = None
            continue
        tf = train_transforms(ultrasound=use_us_aug) if split == 'train' else eval_transforms()
        ds = MedImgDataset(sub, tf)

        # Hospital B data scarcity subsampling for training split
        if split == 'train' and hospital_b_subset_size is not None and hospital_b_subset_size > 0:
            if is_hospital_b:
                n_total = len(ds)
                subset_size = min(int(hospital_b_subset_size), n_total)
                # Fix: subsample seed was previously hardcoded to 42 and didn't vary with the experiment seed
                active_seed = seed if seed is not None else getattr(cfg, 'SEED', SEED)
                rng = np.random.default_rng(active_seed)
                indices = rng.choice(n_total, size=subset_size, replace=False).tolist()
                ds = torch.utils.data.Subset(ds, indices)
                print(f"[DATA SCARCITY] Hospital B training set reduced to {subset_size} images (seed={active_seed}).")

        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == 'train'),
            num_workers=workers, pin_memory=True, drop_last=(split == 'train' and len(ds) >= batch_size),
            prefetch_factor=(2 if workers > 0 else None))
    return loaders

def class_weights(df, hospital_b_subset_size=None, seed=None):
    """Balanced class weights across the client's train set, indexed by LOCAL class."""
    tr = df[df['split'] == 'train'].copy()
    if hospital_b_subset_size is not None and hospital_b_subset_size > 0:
        is_hospital_b = ('client' in tr.columns and (tr['client'] == 1).all()) or \
                        (len(tr) > 0 and 'busi' in str(tr['label'].iloc[0]))
        if is_hospital_b:
            # Fix: subsample seed was previously hardcoded to 42 and didn't vary with the experiment seed
            active_seed = seed if seed is not None else getattr(cfg, 'SEED', SEED)
            rng = np.random.default_rng(active_seed)
            n_sample = min(int(hospital_b_subset_size), len(tr))
            idx = rng.choice(len(tr), size=n_sample, replace=False)
            tr = tr.iloc[idx]
    counts = Counter(tr['local'])
    n = len(tr)
    w = {c: n / (len(counts) * cnt) for c, cnt in counts.items()}
    return w

# ------------------------------------------------------------
# Model: CBAM + shared body + per-client head
# ------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        r = max(in_planes // ratio, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        a = self.avg_pool(x); m = self.max_pool(x)
        s = self.sigmoid(self.fc(a) + self.fc(m))
        return x * s

class SpatialAttention(nn.Module):
    def __init__(self, kernel=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        a = torch.mean(x, dim=1, keepdim=True)
        m = torch.max(x, dim=1, keepdim=True)[0]
        s = self.sigmoid(self.conv(torch.cat([a, m], dim=1)))
        return x * s

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel)
    def forward(self, x):
        return self.sa(self.ca(x))

BACKBONE_OUT = {'efficientnet_v2_s': 1280, 'efficientnet_b0': 1280,
                'mobilenet_v3_large': 960, 'resnet50': 2048}

class SharedBody(nn.Module):
    """Feature extractor: backbone features + CBAM (or ablation) + GAP + PReLU + MC dropout."""
    def __init__(self, backbone='efficientnet_v2_s', emb=512, dropout=0.3, attention='cbam'):
        super().__init__()
        self.backbone_name = backbone
        self.attention_type = attention
        self.dropout_rate = dropout
        if backbone == 'efficientnet_v2_s':
            base = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
            self.features = base.features
        elif backbone == 'efficientnet_b0':
            base = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
            self.features = base.features
        elif backbone == 'mobilenet_v3_large':
            base = torchvision.models.mobilenet_v3_large(weights='IMAGENET1K_V1')
            self.features = base.features
        else:
            raise ValueError(backbone)
        cin = BACKBONE_OUT[backbone]
        if attention == 'cbam':
            self.attention = CBAM(cin, ratio=cfg.CBAM_RATIO, kernel=cfg.CBAM_KERNEL)
        elif attention == 'channel':
            self.attention = ChannelAttention(cin, ratio=cfg.CBAM_RATIO)
        elif attention == 'spatial':
            self.attention = SpatialAttention(kernel=cfg.CBAM_KERNEL)
        else:
            self.attention = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, emb)
        self.prelu = nn.PReLU()

    def forward(self, x, mc=False):
        h = self.features(x)
        h = self.attention(h)
        v = self.pool(h).flatten(1)
        v = self.prelu(self.fc(v))
        if mc or self.training:
            v = F.dropout(v, p=self.dropout_rate, training=True)
        return v

    def forward_features(self, x):
        return self.attention(self.features(x))

class LocalHead(nn.Module):
    def __init__(self, emb, ncls):
        super().__init__()
        self.fc = nn.Linear(emb, ncls)
    def forward(self, x):
        return self.fc(x)

class ClientNet(nn.Module):
    """Full per-client network = shared body + local head."""
    def __init__(self, num_classes, backbone='efficientnet_v2_s', emb=512, dropout=0.3, attention='cbam'):
        super().__init__()
        self.body = SharedBody(backbone, emb, dropout, attention=attention)
        self.head = LocalHead(emb, num_classes)
    def forward(self, x, mc=False):
        v = self.body(x, mc=mc)
        return self.head(v)

# ------------------------------------------------------------
# Federated weight handling (FedPer + FedBN-style local BN + CKA-guided deep personalization)
# ------------------------------------------------------------
def bn_param_names(module):
    """Names of params that live inside BatchNorm modules (kept local)."""
    names = set()
    for name, m in module.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            prefix = name + '.' if name else ''
            for k in m.state_dict():
                names.add(prefix + k)
    return names

# Parameter-name prefixes identified by the CKA analysis (Fig. 7) as
# modality-specific (low cross-client representational alignment).
# CBAM attention: CKA ~0.694. Final 512-D projection (fc/prelu): CKA ~0.446.
DEEP_PERSONALIZE_PREFIXES = ('attention.', 'fc.', 'prelu.')

def deep_param_names(module):
    """Body-level parameter names kept fully local when cfg.PERSONALIZE_DEEP is True.
    Returns an empty set otherwise (so behavior is unchanged by default and
    existing 'uniform' runs are unaffected)."""
    if not getattr(cfg, 'PERSONALIZE_DEEP', False):
        return set()
    names = {n for n, _ in module.named_parameters() if n.startswith(DEEP_PERSONALIZE_PREFIXES)}
    assert len(names) > 0, (
        "PERSONALIZE_DEEP is True but DEEP_PERSONALIZE_PREFIXES matched zero "
        "parameters — the prefix list does not match this model's naming. "
        "Fix DEEP_PERSONALIZE_PREFIXES before proceeding."
    )
    return names

def frozen_local_param_names(module):
    """Union of BatchNorm params + (optionally) CKA-guided deep-personalized params.
    This is the single source of truth for 'what never crosses the federation boundary'
    — use it everywhere bn_param_names was previously used for that purpose."""
    return bn_param_names(module) | deep_param_names(module)

def shared_param_names(module):
    return {n for n, _ in module.named_parameters()}

def copy_shared(src_body, dst_body, keep_bn=True):
    """Copy shared weights from src_body into dst_body.
    With keep_bn=True, dst keeps its own BatchNorm (FedBN) and, if cfg.PERSONALIZE_DEEP,
    its own attention/fc/prelu weights too."""
    frozen = frozen_local_param_names(dst_body) if keep_bn else set()
    src = src_body.state_dict()
    with torch.no_grad():
        for n, p in dst_body.named_parameters():
            if n in frozen:
                continue
            if n in src:
                p.data.copy_(src[n].data)

def state_dict_excluding_bn(module):
    frozen = frozen_local_param_names(module)
    sd = module.state_dict()
    return {k: v for k, v in sd.items() if k not in frozen}

def weighted_average_bodies(weighted_states, bn_states=None):
    """weighted_states: list of (weight, state_dict-excluding-frozen-local-params).
    Returns averaged non-frozen state."""
    keys = list(weighted_states[0][1].keys())
    W = sum(w for w, _ in weighted_states)
    avg = {k: sum(w * st[k] for w, st in weighted_states) / W for k in keys}
    return avg

def set_state_dict(module, sd, exclude_bn=False):
    frozen = frozen_local_param_names(module) if exclude_bn else set()
    with torch.no_grad():
        for n, p in module.named_parameters():
            if n in frozen or n not in sd:
                continue
            p.data.copy_(sd[n].data)

def check_finite(module, tag=''):
    for n, p in module.named_parameters():
        if not torch.isfinite(p.data).all():
            print(f'    [WARN] non-finite param {tag}/{n}')
            return False
    return True

# ------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------
def softmax_loss(logits, labels, smoothing):
    return F.cross_entropy(logits, labels, label_smoothing=smoothing)

@contextlib.contextmanager
def amp():
    """Mixed-precision context; no-op when AMP disabled (fp32 -> stable).
    cfg.AMP = False (default) — fp16 caused NaN in roc_curve on this setup.
    Set cfg.AMP = True only if you have confirmed numerical stability.
    """
    if cfg.AMP and torch.cuda.is_available():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            yield
    else:
        yield

def train_client(net, loaders, cw, epochs, grad_accum, lr_backbone, lr_head,
                 scheduler_round_progress=None, mc_dropout=True, device='cuda'):
    """Local training for one client. Returns train/val metrics."""
    body_params = [p for p in net.body.parameters() if p.requires_grad]
    head_params = [p for p in net.head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {'params': body_params, 'lr': lr_backbone, 'weight_decay': cfg.WD},
        {'params': head_params, 'lr': lr_head, 'weight_decay': cfg.WD},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=None, label_smoothing=cfg.LABEL_SMOOTHING)

    # build per-batch class weight vector (on device)
    w_tensor = torch.zeros(max(cw) + 1, dtype=torch.float32)
    for k, v in cw.items():
        w_tensor[k] = v
    w_tensor = w_tensor.to(device)

    net.train()
    tr_loss, tr_acc, n_tr = 0.0, 0.0, 0
    for ep in range(epochs):
        for i, (x, y) in enumerate(loaders['train']):
            x, y = x.to(device), y.to(device)
            w = w_tensor[y]
            with amp():
                out = net(x, mc=True)
                loss = F.cross_entropy(out, y, reduction='none',
                                       label_smoothing=cfg.LABEL_SMOOTHING)
                loss = (loss * w).mean() / grad_accum
            loss.backward()
            if (i + 1) % grad_accum == 0:
                opt.step(); opt.zero_grad(set_to_none=True)
            tr_loss += loss.item() * x.size(0) * grad_accum
            tr_acc += (out.argmax(1) == y).sum().item()
            n_tr += x.size(0)
        sched.step()

    # validation
    net.eval()
    val_loss, val_acc, n_val = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loaders['val']:
            x, y = x.to(device), y.to(device)
            with amp():
                out = net(x)
                loss = F.cross_entropy(out, y, label_smoothing=cfg.LABEL_SMOOTHING)
            val_loss += loss.item() * x.size(0)
            val_acc += (out.argmax(1) == y).sum().item()
            n_val += x.size(0)
    return (tr_loss / max(n_tr, 1), tr_acc / max(n_tr, 1),
            val_loss / max(n_val, 1), val_acc / max(n_val, 1))

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
def evaluate(net, loader, device='cuda', ncls=None):
    net.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            with amp():
                out = net(x).float()
            y_true.append(y.numpy())
            y_prob.append(out.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_prob = np.nan_to_num(y_prob, nan=1.0 / y_prob.shape[1], posinf=1.0, neginf=0.0)
    y_pred = y_prob.argmax(1)
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 roc_auc_score)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    from sklearn.metrics import roc_curve, auc as _auc
    aucs = []
    for c in range(y_prob.shape[1]):
        if np.sum(y_true == c) == 0:
            continue
        fpr, tpr, _ = roc_curve((y_true == c).astype(int), y_prob[:, c])
        if len(np.unique((y_true == c).astype(int))) < 2:
            continue
        aucs.append(_auc(fpr, tpr))
    auc = float(np.nanmean(aucs)) if aucs else float('nan')
    return {'acc': acc, 'precision': p, 'recall': r, 'f1': f1, 'auc': auc,
            'y_true': y_true, 'y_prob': y_prob}

def tune_temperature(net, loader, device='cuda'):
    """Temperature scaling on the validation split."""
    logits, labels = [], []
    net.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            with amp():
                out = net(x).float()
            logits.append(out.cpu()); labels.append(y)
    logits = torch.cat(logits); labels = torch.cat(labels)
    T = nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss
    for _ in range(20):
        opt.step(closure)
    return float(T.item())

def mc_predict(net, loader, device='cuda', iters=cfg.MC_ITERS, max_batches=None):
    """MC-dropout stochastic forward passes -> mean prob, entropy, uncertainty.
    Maintains net.eval() so BatchNorm statistics remain frozen while MC dropout is active."""
    means, ents, uncs = [], [], []
    net.eval()
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = []
            for _ in range(iters):
                with amp():
                    probs.append(torch.softmax(net(x, mc=True).float(), dim=-1).cpu())
            p = torch.stack(probs).mean(0)          # (B, C)
            ent = -(p * (p + 1e-10).log()).sum(-1)  # (B,)
            unc = 1 - p.max(-1).values
            means.append(p.numpy()); ents.append(ent.numpy()); uncs.append(unc.numpy())
            count += 1
            if max_batches and count >= max_batches:
                break
    return {'mean_prob': np.concatenate(means), 'entropy': np.concatenate(ents),
            'uncertainty': np.concatenate(uncs)}

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=cfg.COMM_ROUNDS)
    ap.add_argument('--batch', type=int, default=cfg.BATCH_SIZE)
    ap.add_argument('--agg_weight_type', type=str, default=cfg.AGG_WEIGHT_TYPE, choices=['uniform', 'sample'],
                    help="Server aggregation weighting: 'uniform' (1/K) or 'sample' (N_k/N_total)")
    ap.add_argument('--hospital_b_subset_size', type=int, default=0,
                    help="Subsample size for Hospital B training dataset (0 = use all data)")
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    cfg.COMM_ROUNDS = args.rounds
    cfg.BATCH_SIZE = args.batch
    cfg.SMOKE = args.smoke or cfg.SMOKE
    cfg.AGG_WEIGHT_TYPE = args.agg_weight_type
    cfg.HOSPITAL_B_SUBSET_SIZE = args.hospital_b_subset_size if args.hospital_b_subset_size > 0 else None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'PyTorch {torch.__version__} | device={device} | '
          f'GPU={torch.cuda.get_device_name(0) if device=="cuda" else "n/a"}')
    if device == 'cuda':
        free, total = torch.cuda.mem_get_info()
        print(f'VRAM free={free/2**30:.2f}GB total={total/2**30:.2f}GB '
              f'(batch {cfg.BATCH_SIZE})')
        if free < 1.5e9:
            print('[!] Low free VRAM - consider closing Lightroom/Teams/Edge.')

    out = Path(cfg.OUTPUT_DIR)
    for sub in ('models', 'reports', 'figures'):
        (out / sub).mkdir(parents=True, exist_ok=True)

    print('[i] Discovering data ...')
    all_df, classes, class_to_idx = discover_all(smoke=cfg.SMOKE)
    print(f'[OK] {len(classes)} classes: {classes}')

    client_dfs = {cid: all_df[all_df['client'] == cid] for cid in range(cfg.NUM_CLIENTS)}
    for cid in range(cfg.NUM_CLIENTS):
        uniq = sorted(client_dfs[cid]['gid'].unique())
        g2l = {g: i for i, g in enumerate(uniq)}
        client_dfs[cid] = client_dfs[cid].copy()
        client_dfs[cid]['local'] = client_dfs[cid]['gid'].map(g2l)

    if cfg.AGG_WEIGHT_TYPE == 'uniform':
        K = cfg.NUM_CLIENTS
        agg_w = {cid: 1.0 / K for cid in range(cfg.NUM_CLIENTS)}
    else:
        sizes = {cid: int((client_dfs[cid]['split'] == 'train').sum())
                 for cid in range(cfg.NUM_CLIENTS)}
        tot = sum(sizes.values())
        agg_w = {cid: s / tot for cid, s in sizes.items()}
    print(f'  Aggregation weights ({cfg.AGG_WEIGHT_TYPE}):', {k: round(v, 4) for k, v in agg_w.items()})

    loaders = {cid: build_loaders(client_dfs[cid], cfg.BATCH_SIZE, cfg.WORKERS,
                                  hospital_b_subset_size=cfg.HOSPITAL_B_SUBSET_SIZE,
                                  client_id=cid)
               for cid in range(cfg.NUM_CLIENTS)}
    cweights = {cid: class_weights(client_dfs[cid], hospital_b_subset_size=cfg.HOSPITAL_B_SUBSET_SIZE)
                for cid in range(cfg.NUM_CLIENTS)}

    client_uniq = {cid: sorted(client_dfs[cid]['gid'].unique())
                   for cid in range(cfg.NUM_CLIENTS)}
    ncls_local = {cid: len(client_uniq[cid]) for cid in range(cfg.NUM_CLIENTS)}
    all_cls_idx = sorted(all_df['gid'].unique())

    global_body = SharedBody(cfg.BACKBONE, cfg.EMB, cfg.DROPOUT).to(device)

    # per-client persistent nets
    nets = {cid: ClientNet(ncls_local[cid], cfg.BACKBONE, cfg.EMB, cfg.DROPOUT).to(device)
            for cid in range(cfg.NUM_CLIENTS)}
    # initialize each client body from global (round 0); copy BN on first round
    for cid in range(cfg.NUM_CLIENTS):
        copy_shared(global_body, nets[cid].body, keep_bn=False)

    log_rows = []
    best_global_acc = 0.0
    t_start = time.time()

    print('=' * 72)
    print('  FEDUA-NET (PyTorch) - PERSONALIZED FEDERATED TRAINING')
    print(f'  Rounds={cfg.COMM_ROUNDS}  local_epochs={cfg.LOCAL_EPOCHS}  '
          f'agg_weights={agg_w}')
    print(f'  Backbone={cfg.BACKBONE}  FedPer heads + FedBN-style local BN')
    print('=' * 72)

    for rnd in range(1, cfg.COMM_ROUNDS + 1):
        t0 = time.time()
        # server LR schedule (cosine w/ warmup)
        if rnd <= cfg.WARMUP_ROUNDS:
            frac = rnd / cfg.WARMUP_ROUNDS
        else:
            prog = (rnd - cfg.WARMUP_ROUNDS) / max(1, cfg.COMM_ROUNDS - cfg.WARMUP_ROUNDS)
            frac = 0.5 * (1 + np.cos(np.pi * prog))
        lr_backbone = cfg.MIN_LR + (cfg.BACKBONE_LR - cfg.MIN_LR) * frac
        lr_head = cfg.MIN_LR + (cfg.BASE_LR - cfg.MIN_LR) * frac

        weighted_states = []
        row = {'round': rnd, 'lr_backbone': lr_backbone, 'lr_head': lr_head}
        if getattr(cfg, 'PERSONALIZE_DEEP', False):
            print(f"  [CKA-Personalize] Round {rnd:02d}: deep_param_names(nets[1].body) count = {len(deep_param_names(nets[1].body))}")
        for cid in range(cfg.NUM_CLIENTS):
            # broadcast global shared (non-BN) into client, keep local BN
            copy_shared(global_body, nets[cid].body, keep_bn=True)
            trl, tra, vl, va = train_client(
                nets[cid], loaders[cid], cweights[cid], cfg.LOCAL_EPOCHS[cid],
                cfg.GRAD_ACCUM, lr_backbone, lr_head, device=device)
            if not check_finite(nets[cid], f'client{cid}'):
                copy_shared(global_body, nets[cid].body, keep_bn=True)
            row[f'c{cid}_tr_acc'] = tra; row[f'c{cid}_val_acc'] = va
            row[f'c{cid}_val_loss'] = vl
            weighted_states.append((agg_w[cid], state_dict_excluding_bn(nets[cid].body)))

        # weighted FedAvg over non-BN shared weights (FedBN)
        avg_sd = weighted_average_bodies(weighted_states)
        set_state_dict(global_body, avg_sd, exclude_bn=True)

        # refresh global BN as weighted average for the reference model
        with torch.no_grad():
            bn_accum = {}
            for cid in range(cfg.NUM_CLIENTS):
                for n, p in nets[cid].body.named_parameters():
                    if n in bn_param_names(nets[cid].body):
                        bn_accum[n] = bn_accum.get(n, 0) + agg_w[cid] * p.detach().cpu().clone()
            for n, p in global_body.named_parameters():
                if n in bn_param_names(global_body) and n in bn_accum:
                    p.data.copy_(bn_accum[n].data)

        # per-client personalized eval on each client's test
        client_evals = {}
        for cid in range(cfg.NUM_CLIENTS):
            if loaders[cid]['test'] is not None:
                ev = evaluate(nets[cid], loaders[cid]['test'], device)
                client_evals[cid] = ev
                row[f'c{cid}_test_acc'] = ev['acc']
        # global reference: pooled test via per-client nets (weighted mean)
        mean_acc = np.mean([client_evals[c]['acc'] for c in client_evals])
        row['mean_test_acc'] = mean_acc
        log_rows.append(row)
        hosp_tags = {0: 'HospA (Brain)', 1: 'HospB (BreastUS)', 2: 'HospC (COVID)'}
        print(f'  R{rnd:>2}/{cfg.COMM_ROUNDS}  LRb={lr_backbone:.1e} | ' +
              ' | '.join(f'{hosp_tags.get(c, f"C{c}")}: {client_evals[c]["acc"]*100:.2f}%' for c in sorted(client_evals)) +
              f' | Mean: {mean_acc*100:.2f}% ({time.time()-t0:.1f}s)')

        # checkpoint
        torch.save({'global_body': global_body.state_dict(),
                    'nets': {cid: nets[cid].state_dict() for cid in nets}},
                   out / 'models' / f'checkpoint_r{rnd:02d}.pt')

    # -------------------- evaluation --------------------
    print('=' * 72)
    print('  FINAL EVALUATION (personalized models, per-client test)')
    print('=' * 72)
    summary = []
    per_class_rows = []
    from sklearn.metrics import classification_report
    for cid in range(cfg.NUM_CLIENTS):
        if loaders[cid]['test'] is None:
            continue
        ev = evaluate(nets[cid], loaders[cid]['test'], device)
        local_labels = list(range(ncls_local[cid]))
        local_names = [classes[g] for g in client_uniq[cid]]
        print(f"\n  {client_meta(cid)[3]}")
        print(f"    Acc={ev['acc']:.4f}  F1={ev['f1']:.4f}  AUC={ev['auc']:.4f}")
        summary.append({'client': cid, 'name': client_meta(cid)[3],
                        'accuracy': ev['acc'], 'precision': ev['precision'],
                        'recall': ev['recall'], 'f1': ev['f1'], 'auc': ev['auc']})
        rep = classification_report(ev['y_true'], ev['y_prob'].argmax(1),
                                    labels=local_labels, target_names=local_names,
                                    output_dict=True, zero_division=0)
        for cl in local_names:
            d = rep.get(cl, {})
            per_class_rows.append({'client': client_meta(cid)[3], 'class': cl,
                                   'precision': d.get('precision'),
                                   'recall': d.get('recall'),
                                   'f1': d.get('f1-score'),
                                   'support': d.get('support')})

    summary_df = pd.DataFrame(summary)
    print('\n  --- Summary table ---')
    print(summary_df.to_string(index=False))
    summary_df.to_csv(out / 'reports' / 'final_client_summary.csv', index=False)
    pd.DataFrame(per_class_rows).to_csv(out / 'reports' / 'final_per_class.csv', index=False)
    pd.DataFrame(log_rows).to_csv(out / 'reports' / 'final_fed_log.csv', index=False)

    with open(out / 'reports' / 'final_report.txt', 'w') as f:
        f.write('FedUA-Net (PyTorch) evaluation report\n')
        f.write('=' * 60 + '\n')
        f.write(summary_df.to_string(index=False) + '\n')
        f.write('\nPer-class:\n')
        f.write(pd.DataFrame(per_class_rows).to_string(index=False) + '\n')

    # uncertainty on a sample of each client's test (MC dropout)
    print('\n[MC] Uncertainty estimation (MC dropout) ...')
    uq_rows = []
    for cid in range(cfg.NUM_CLIENTS):
        if loaders[cid]['test'] is None:
            continue
        uq = mc_predict(nets[cid], loaders[cid]['test'], device, iters=cfg.MC_ITERS,
                        max_batches=8)
        y = []
        for xb, yb in loaders[cid]['test']:
            y.append(yb.numpy())
            if len(y) * yb.size(0) >= len(uq['mean_prob']):
                break
        y = np.concatenate(y)[:len(uq['mean_prob'])]
        pred = uq['mean_prob'].argmax(1)
        correct = (pred == y).astype(int)
        for i in range(len(y)):
            uq_rows.append({'client': client_meta(cid)[3],
                            'correct': correct[i], 'entropy': uq['entropy'][i],
                            'uncertainty': uq['uncertainty'][i]})
    if uq_rows:
        uq_df = pd.DataFrame(uq_rows)
        uq_df.to_csv(out / 'reports' / 'final_uncertainty.csv', index=False)
        print(uq_df.groupby('client')['correct'].mean().to_string())

    # save final artifacts
    torch.save({'global_body': global_body.state_dict(),
                'nets': {cid: nets[cid].state_dict() for cid in nets}},
               out / 'models' / 'fedua_net_final.pt')
    with open(out / 'reports' / 'final_meta.json', 'w') as f:
        json.dump({'classes': classes, 'agg_weights': {str(k): v for k, v in agg_w.items()},
                   'rounds': cfg.COMM_ROUNDS, 'batch': cfg.BATCH_SIZE,
                   'elapsed_sec': time.time() - t_start,
                   'summary': summary_df.to_dict(orient='records')}, f, indent=2)
    print(f'\n[OK] Done in {time.time()-t_start:.1f}s. Artifacts in {out}')

if __name__ == '__main__':
    main()
