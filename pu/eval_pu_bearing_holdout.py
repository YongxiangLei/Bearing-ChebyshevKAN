"""Exploratory Paderborn University three-class bearing-held-out experiment.

Uses the named vibration channel from raw official MATLAB files and separates
entire physical bearing IDs. This is an independent protocol inspired by the
paper, not an exact reproduction of its PU benchmark.
"""
import argparse
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from adaptive_chebyshev_kan import AdaptiveChebyshevKANLayer


SPLITS = {
    'train': {'normal': ['K001', 'K002', 'K003'], 'inner': ['KI01', 'KI03', 'KI04'], 'outer': ['KA01', 'KA03', 'KA04']},
    'validation': {'normal': ['K004'], 'inner': ['KI05'], 'outer': ['KA05']},
    'test': {'normal': ['K005', 'K006'], 'inner': ['KI07', 'KI16'], 'outer': ['KA06', 'KA07']},
}
NAMES = ['normal', 'inner', 'outer']
CONDITION = 'N15_M07_F10'


class Classifier(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layer1 = AdaptiveChebyshevKANLayer(n, 32, 5)
        self.layer2 = AdaptiveChebyshevKANLayer(32, 3, 5)

    def forward(self, x):
        return self.layer2(self.layer1(x))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True, help='directory containing official bearing RAR files')
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'outputs')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    return p.parse_args()


def load_data(root):
    target = root / 'selected_mat'
    target.mkdir(parents=True, exist_ok=True)
    data, evidence = {}, []
    for split, groups in SPLITS.items():
        xs, ys = [], []
        for label, ids in groups.items():
            for bearing in ids:
                archive = root / f'{bearing}.rar'
                if not archive.exists() or archive.stat().st_size < 1_000_000:
                    raise FileNotFoundError(f'Missing or incomplete archive: {archive}')
                for recording in (1, 2):
                    stem = f'{CONDITION}_{bearing}_{recording}'
                    relative = f'{bearing}/{stem}.mat'
                    path = target / relative
                    if not path.exists():
                        subprocess.run(['tar', '-xf', str(archive.resolve()), '-C', str(target.resolve()), relative], check=True)
                    m = loadmat(path, simplify_cells=True)
                    channels = m[stem]['Y']
                    candidates = [c for c in channels if c['Name'] == 'vibration_1']
                    if len(candidates) != 1:
                        raise ValueError(f'Expected one vibration_1 channel: {path}')
                    raw = np.asarray(candidates[0]['Data'], dtype=np.float64).reshape(-1)
                    n = len(raw) // 1024
                    if n < 100 or not np.all(np.isfinite(raw)):
                        raise ValueError(f'Insufficient or invalid signal: {path}')
                    # 100 windows evenly spread across each recording, no overlap.
                    indices = np.linspace(0, n-1, 100, dtype=int)
                    windows = np.stack([raw[i*1024:(i+1)*1024] for i in indices])
                    # Standardize each window before the FFT; no cross-split statistics.
                    windows = (windows - windows.mean(axis=1, keepdims=True)) / (windows.std(axis=1, keepdims=True) + 1e-8)
                    xs.append(np.abs(np.fft.rfft(windows, axis=1)))
                    ys.append(np.full(len(windows), NAMES.index(label), dtype=np.int64))
                    evidence.append({'bearing': bearing, 'class': label, 'split': split,
                                     'file': relative, 'channel': 'vibration_1', 'samples': len(raw), 'windows': len(windows)})
        data[split] = (np.concatenate(xs), np.concatenate(ys))
    return data, evidence


def evaluate(model, x, y, device, batch):
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=batch)
    pred = []
    model.eval()
    with torch.no_grad():
        for (a,) in loader:
            pred.extend(model(a.to(device)).argmax(1).cpu().tolist())
    return {'n': len(y), 'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'class_order': NAMES,
            'confusion_matrix_rows_true_columns_predicted': confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
            'per_class': classification_report(y, pred, labels=[0, 1, 2], target_names=NAMES, output_dict=True, zero_division=0)}


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device == 'cuda': torch.cuda.manual_seed_all(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    data, evidence = load_data(args.data_dir)
    xtr, ytr = data['train']; xv, yv = data['validation']; xt, yt = data['test']
    mean = xtr.mean(axis=0, keepdims=True); std = xtr.std(axis=0, keepdims=True) + 1e-8
    xtr = ((xtr - mean) / std).astype(np.float32)
    xv = ((xv - mean) / std).astype(np.float32)
    xt = ((xt - mean) / std).astype(np.float32)
    dl = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)), batch_size=args.batch_size,
                    shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    model = Classifier(xtr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best, best_epoch, state, stale, history = -1, 0, None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_total = 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); optimizer.step()
            loss_total += loss.item() * len(y)
        val = evaluate(model, xv, yv, device, args.batch_size)
        history.append({'epoch': epoch, 'train_loss': loss_total/len(ytr), 'validation_accuracy': val['accuracy'], 'validation_macro_f1': val['macro_f1']})
        print(f"epoch={epoch} train_loss={loss_total/len(ytr):.4f} val_acc={val['accuracy']:.4f} val_f1={val['macro_f1']:.4f}", flush=True)
        if val['macro_f1'] > best:
            best, best_epoch, stale = val['macro_f1'], epoch, 0
            state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(state)
    test = evaluate(model, xt, yt, device, args.batch_size)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'protocol': 'Three-class exploratory task. Hold out whole bearing IDs by class; use two recordings per bearing at N15_M07_F10; take 100 non-overlapping length-1024 windows per recording, distributed across the trace. Per-window z-score, magnitude FFT, then train-only feature scaling. Best validation macro-F1, AdamW.',
              'limitations': 'Only 3/1/2 bearings per class for training/validation/test; only one operating condition, two recordings per bearing. This independent protocol is not the supplied paper script and cannot estimate broad population performance.',
              'bearing_splits': SPLITS, 'files': evidence, 'seed': args.seed, 'best_epoch': best_epoch,
              'training': {'epochs_max': args.epochs, 'patience': args.patience, 'batch_size': args.batch_size,
                           'device': device, 'optimizer': 'AdamW lr=0.005 weight_decay=0.0001',
                           'torch': torch.__version__, 'numpy': np.__version__},
              'history': history, 'test': test}
    torch.save({'model_state_dict': state, 'fft_mean': mean, 'fft_std': std, 'classes': NAMES}, args.output/'model.pt')
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"TEST accuracy={test['accuracy']:.4f} macro_f1={test['macro_f1']:.4f}; saved to {args.output}")


if __name__ == '__main__':
    main()
