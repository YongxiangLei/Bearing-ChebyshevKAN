"""Reconstructed from the authors' supplementary PDF, pp. 14–15.

The model and experimental protocol match the provided code. The downloader
fails fast rather than silently skipping missing files. Set CWRU_DEVICE=cuda
to use a CUDA-enabled PyTorch installation; the original selects MPS or CPU.
"""
import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from scipy.io import loadmat
from adaptive_chebyshev_kan import AdaptiveChebyshevKANLayer

DATA_DIR = Path(os.environ.get("CWRU_DATA_DIR", Path(__file__).parent / "cwru_raw"))
OUT_DIR = Path(__file__).parent / "outputs"
SEED, SEG_LEN = 42, 512
FILES = {
    (0, 0): ("97.mat", "X097_DE_time"), (0, 1): ("98.mat", "X098_DE_time"),
    (0, 2): ("99.mat", "X099_DE_time"), (0, 3): ("100.mat", "X100_DE_time"),
    (1, 0): ("105.mat", "X105_DE_time"), (1, 1): ("106.mat", "X106_DE_time"),
    (1, 2): ("107.mat", "X107_DE_time"), (1, 3): ("108.mat", "X108_DE_time"),
    (2, 0): ("130.mat", "X130_DE_time"), (2, 1): ("131.mat", "X131_DE_time"),
    (2, 2): ("132.mat", "X132_DE_time"), (2, 3): ("133.mat", "X133_DE_time"),
    (3, 0): ("118.mat", "X118_DE_time"), (3, 1): ("119.mat", "X119_DE_time"),
    (3, 2): ("120.mat", "X120_DE_time"), (3, 3): ("121.mat", "X121_DE_time"),
}
NAMES = ["Normal", "Inner Race", "Outer Race", "Ball"]
BASE_URL = "https://engineering.case.edu/sites/default/files/"


def set_seed(s=42):
    np.random.seed(s); torch.manual_seed(s)


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for fn, _ in FILES.values():
        fp = DATA_DIR / fn
        if fp.exists() and fp.stat().st_size > 1000:
            continue
        print(f"Downloading {fn} ...", flush=True)
        urlretrieve(BASE_URL + fn, str(fp))
        if fp.stat().st_size <= 1000:
            raise ValueError(f"File is unexpectedly small: {fp}")


def load_sig(fn, key):
    fp = DATA_DIR / fn
    if not fp.exists():
        return None
    mat = loadmat(str(fp))
    if key in mat:
        return mat[key].flatten()
    # Do not silently substitute another recording's channel: e.g. 99.mat
    # contains both X098_DE_time and X099_DE_time.
    raise KeyError(f"Expected channel {key} absent in {fp}; available keys: {list(mat)}")


def segment(sig, length=512, overlap=0.5):
    step = int(length * (1 - overlap))
    return np.array([sig[i:i+length] for i in range(0, len(sig)-length+1, step)])


def build_dataset(load_hp):
    vibs, labels = [], []
    for (cls, hp), (fn, key) in FILES.items():
        if hp != load_hp:
            continue
        sig = load_sig(fn, key)
        if sig is None:
            raise ValueError(f"Missing or unsupported signal: {fn}, {key}")
        s = segment(sig, SEG_LEN)
        vibs.append(s); labels.append(np.full(len(s), cls))
    return np.concatenate(vibs), np.concatenate(labels).astype(np.int64)


def featurize_simple(X_vib):
    fft_vals = np.abs(np.fft.rfft(X_vib, axis=1))
    mean = np.mean(fft_vals, axis=0, keepdims=True)
    std = np.std(fft_vals, axis=0, keepdims=True) + 1e-8
    return (fft_vals - mean) / std


class AdaptiveChebyshevClassifier(nn.Module):
    def __init__(self, in_features, num_classes=4, hidden_dim=64, max_degree=8):
        super().__init__()
        self.layer1 = AdaptiveChebyshevKANLayer(in_features, hidden_dim, max_degree)
        self.layer2 = AdaptiveChebyshevKANLayer(hidden_dim, num_classes, max_degree)

    def forward(self, x):
        return self.layer2(self.layer1(x))


def train(X, y, nc=4, epochs=40, device="cpu"):
    m = AdaptiveChebyshevClassifier(X.shape[1], nc, hidden_dim=32, max_degree=5).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=0.005, weight_decay=1e-4)
    xt, yt = torch.from_numpy(X).float().to(device), torch.from_numpy(y).long().to(device)
    ce = nn.CrossEntropyLoss()
    m.train()
    for _ in range(epochs):
        for i in range(0, len(xt), 64):
            idx = torch.randperm(len(xt), device=device)[:64]
            loss = ce(m(xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return m.cpu().eval()


def evaluate(m, X, y):
    with torch.no_grad():
        p = m(torch.from_numpy(X).float()).argmax(1).numpy()
    return float(accuracy_score(y, p)), float(f1_score(y, p, average="macro", zero_division=0))


def main():
    set_seed(SEED)
    dev = os.environ.get("CWRU_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Running Adaptive ChebyshevKAN on CWRU ({dev})", flush=True)
    download()
    X0, y0 = build_dataset(0)
    X1, y1 = build_dataset(1)
    F0, F1 = featurize_simple(X0), featurize_simple(X1)
    print(f"Training on 0HP (Shape: {F0.shape})...", flush=True)
    m = train(F0, y0, epochs=40, device=dev)
    acc_indom, f1_indom = evaluate(m, F0, y0)
    acc_cross, f1_cross = evaluate(m, F1, y1)
    print(f"In-domain (0HP -> 0HP): Acc={acc_indom*100:.2f}%, F1={f1_indom:.4f}")
    print(f"Cross-load (0HP -> 1HP): Acc={acc_cross*100:.2f}%, F1={f1_cross:.4f}")
    print("\nExtracting Symbolic Expressions for Class 0 (Normal) ...")
    print(m.layer2.get_symbolic_expression(threshold=0.8)[0][:300] + " ...")


if __name__ == "__main__":
    main()
