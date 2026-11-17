"""Reconstructed from the authors' supplementary PDF, pp. 8–9.

This reproduces the supplied overlapping-window random split and per-test-set
normalization. These choices mean the result is not a leakage-free estimate.
Set CWRU_DEVICE=cuda to enable a CUDA-capable PyTorch installation.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from run_adaptive_cwru import AdaptiveChebyshevClassifier, set_seed, build_dataset, featurize_simple, download


class StandardMLP(nn.Module):
    def __init__(self, in_features, num_classes=4, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        return self.net(x)


def add_awgn_noise(signal, snr_db):
    sig_power = np.mean(signal ** 2, axis=1, keepdims=True)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise


def main():
    set_seed(42)
    dev = os.environ.get("CWRU_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running SNR Robustness Evaluation on CWRU ({dev})", flush=True)
    download()
    X_clean, y = build_dataset(0)
    idx = np.random.permutation(len(X_clean))
    train_size = int(0.8 * len(X_clean))
    train_idx, test_idx = idx[:train_size], idx[train_size:]
    X_train_clean, y_train = X_clean[train_idx], y[train_idx]
    X_test_clean, y_test = X_clean[test_idx], y[test_idx]
    F_train_clean = featurize_simple(X_train_clean)
    models = {
        "AdaptiveChebyshevKAN": AdaptiveChebyshevClassifier(F_train_clean.shape[1], 4, hidden_dim=32, max_degree=5).to(dev),
        "StandardMLP Baseline": StandardMLP(F_train_clean.shape[1], 4, hidden_dim=32).to(dev),
    }
    for model_name, m in models.items():
        print(f"Training {model_name} on clean data...", flush=True)
        opt = torch.optim.AdamW(m.parameters(), lr=0.005, weight_decay=1e-4)
        ce = nn.CrossEntropyLoss()
        xt = torch.from_numpy(F_train_clean).float().to(dev)
        yt = torch.from_numpy(y_train).long().to(dev)
        m.train()
        for _ in range(40):
            for i in range(0, len(xt), 64):
                b_idx = torch.randperm(len(xt), device=dev)[:64]
                loss = ce(m(xt[b_idx]), yt[b_idx])
                opt.zero_grad(); loss.backward(); opt.step()
        m.cpu().eval()
        print(f"\n--- Testing Robustness under varying SNR for {model_name} ---", flush=True)
        for snr in [10, 5, 0, -2, -4, -6, -8, -10]:
            X_test_noisy = add_awgn_noise(X_test_clean, snr)
            F_test_noisy = featurize_simple(X_test_noisy)
            with torch.no_grad():
                p = m(torch.from_numpy(F_test_noisy).float()).argmax(1).numpy()
            acc = accuracy_score(y_test, p)
            print(f"SNR {snr:3d}dB : Accuracy = {acc*100:.2f}%", flush=True)
    print("\nRobustness evaluation complete.")


if __name__ == "__main__":
    main()
