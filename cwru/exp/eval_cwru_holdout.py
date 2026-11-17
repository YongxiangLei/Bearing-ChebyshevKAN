"""Evaluate a modified CWRU protocol with raw-time-disjoint splits.

The paper's model and CWRU file list are retained; the split, training, model
selection and preprocessing differ from its supplementary scripts. Each 0 hp
recording is split into contiguous 60/20/20% portions *before* segmentation.
This avoids overlapping raw samples across train/validation/test, but it does
not establish generalization to new bearings or acquisition sessions.
"""
import argparse
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from scipy.signal import resample_poly
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_adaptive_cwru import (AdaptiveChebyshevClassifier, DATA_DIR, FILES,
                               NAMES, download, load_sig, segment)


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--resample-normal', action='store_true', help='sensitivity experiment: downsample normal recordings 4x within each split (48 kHz assumption is unverified)')
    p.add_argument('--augment-noise', action='store_true', help='train with three noisy copies per clean window, SNR uniform from -6 to 10 dB')
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'outputs' / 'cwru_holdout')
    return p.parse_args()


def split_recordings(load=0, resample_normal=False):
    """Split each class recording by raw sample index, prior to windowing."""
    parts = {name: ([], []) for name in ('train', 'validation', 'test')}
    provenance = []
    for (cls, hp), (fn, key) in sorted(FILES.items()):
        if hp != load:
            continue
        sig = load_sig(fn, key)
        if sig is None:
            raise FileNotFoundError(f'Missing channel {key} in {DATA_DIR / fn}')
        cut1, cut2 = int(len(sig) * .6), int(len(sig) * .8)
        for name, start, end in [('train', 0, cut1), ('validation', cut1, cut2), ('test', cut2, len(sig))]:
            block = sig[start:end]
            # Filter only inside the already-disjoint raw-time block.
            if resample_normal and cls == 0:
                block = resample_poly(block, 1, 4)
            windows = segment(block, length=512, overlap=.5)
            if not len(windows):
                raise ValueError(f'Too few samples in {fn}/{name}')
            parts[name][0].append(windows)
            parts[name][1].append(np.full(len(windows), cls, dtype=np.int64))
            provenance.append(dict(file=fn, cls=cls, label=NAMES[cls], split=name,
                                   raw_start=int(start), raw_end_exclusive=int(end), windows=len(windows)))
    return {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in parts.items()}, provenance


def full_load(load=1, resample_normal=False):
    xs, ys = [], []
    for (cls, hp), (fn, key) in sorted(FILES.items()):
        if hp == load:
            sig = load_sig(fn, key)
            if sig is None:
                raise FileNotFoundError(f'Missing channel {key} in {DATA_DIR / fn}')
            if resample_normal and cls == 0:
                sig = resample_poly(sig, 1, 4)
            w = segment(sig, length=512, overlap=.5)
            xs.append(w); ys.append(np.full(len(w), cls, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


def fft(x):
    return np.abs(np.fft.rfft(x, axis=1))


def metrics(model, features, y, device, batch_size):
    loader = DataLoader(TensorDataset(torch.from_numpy(features.astype(np.float32))), batch_size=batch_size)
    predictions = []
    model.eval()
    with torch.no_grad():
        for (x,) in loader:
            predictions.extend(model(x.to(device)).argmax(dim=1).cpu().numpy().tolist())
    return {
        'n': len(y),
        'accuracy': float(accuracy_score(y, predictions)),
        'macro_f1': float(f1_score(y, predictions, average='macro', zero_division=0)),
        'class_order': NAMES,
        'confusion_matrix_rows_true_columns_predicted': confusion_matrix(y, predictions, labels=list(range(4))).tolist(),
        'per_class': classification_report(y, predictions, labels=list(range(4)), target_names=NAMES,
                                           output_dict=True, zero_division=0),
    }


def main():
    args = get_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError('epochs, patience and batch-size must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but PyTorch cannot access it')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    download()
    splits, provenance = split_recordings(resample_normal=args.resample_normal)
    xtr, ytr = splits['train']
    xval, yval = splits['validation']
    xtest, ytest = splits['test']
    if args.augment_noise:
        augmentation_rng = np.random.default_rng(args.seed + 100)
        power = np.mean(xtr ** 2, axis=1, keepdims=True)
        augmented = [xtr]
        for _ in range(3):
            snr = augmentation_rng.uniform(-6, 10, size=(len(xtr), 1))
            augmented.append(xtr + augmentation_rng.normal(0, np.sqrt(power / (10 ** (snr / 10))), xtr.shape))
        xtr = np.concatenate(augmented)
        ytr = np.tile(ytr, 4)
    train_fft = fft(xtr)
    mean = train_fft.mean(axis=0, keepdims=True)
    std = train_fft.std(axis=0, keepdims=True) + 1e-8
    transform = lambda x: ((fft(x) - mean) / std).astype(np.float32)
    ftrain, fval = transform(xtr), transform(xval)
    if args.augment_noise:
        validation_rng = np.random.default_rng(args.seed + 101)
        power = np.mean(xval ** 2, axis=1, keepdims=True)
        val_noisy = xval + validation_rng.normal(0, np.sqrt(power / (10 ** (-4 / 10))), xval.shape)
        fval_noisy = transform(val_noisy)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(ftrain), torch.from_numpy(ytr)),
                              batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    model = AdaptiveChebyshevClassifier(ftrain.shape[1], 4, hidden_dim=32, max_degree=5).to(device)
    counts = np.bincount(ytr, minlength=4)
    weights = len(ytr) / (len(counts) * counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=1e-4)
    history, best_f1, best_epoch, best_state, stale = [], -1., 0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(y)
        v = metrics(model, fval, yval, device, args.batch_size)
        noisy_f1 = metrics(model, fval_noisy, yval, device, args.batch_size)['macro_f1'] if args.augment_noise else None
        score = (v['macro_f1'] + noisy_f1) / 2 if noisy_f1 is not None else v['macro_f1']
        history.append(dict(epoch=epoch, train_loss=total_loss/len(ytr), validation_accuracy=v['accuracy'], validation_macro_f1=v['macro_f1'], validation_noisy_minus4_macro_f1=noisy_f1, selection_score=score))
        print(f"epoch={epoch:02d} loss={history[-1]['train_loss']:.4f} val_acc={v['accuracy']:.4f} selection_score={score:.4f}", flush=True)
        if score > best_f1:
            best_f1, best_epoch, stale = score, epoch, 0
            best_state = {k: value.detach().cpu().clone() for k, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    test = metrics(model, transform(xtest), ytest, device, args.batch_size)
    noise_results = {}
    rng = np.random.default_rng(args.seed)
    for snr in [10, 5, 0, -2, -4, -6, -8, -10]:
        power = np.mean(xtest ** 2, axis=1, keepdims=True)
        noise = rng.normal(0, np.sqrt(power / (10 ** (snr / 10))), xtest.shape)
        noise_results[str(snr)] = metrics(model, transform(xtest + noise), ytest, device, args.batch_size)
    xcross, ycross = full_load(1, resample_normal=args.resample_normal)
    cross = metrics(model, transform(xcross), ycross, device, args.batch_size)
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': 'Raw-recording temporal 60/20/20 splits before 512-sample windows, 50% within-split overlap; 0hp train/validation/test; per-class split; train-only FFT feature normalization; weighted loss; validation-based early stopping.',
        'limitations': 'Same recording contributes to all temporal splits; not independent bearings/runs. Exact sample rate of normal baseline files is not confirmed by the source consulted. Optional 4x downsampling is a sensitivity experiment under an unverified 48 kHz assumption. No guarantees of exact GPU determinism.',
        'settings': {'seed': args.seed, 'epochs_max': args.epochs, 'patience': args.patience,
                     'batch_size': args.batch_size, 'device': device, 'window': 512, 'overlap': .5,
                     'normal_downsample_4x_unverified_48khz_assumption': args.resample_normal,
                     'noise_augmentation': '3 copies per training window, SNR uniform -6 to 10 dB' if args.augment_noise else 'none',
                     'model': '2-layer adaptive Chebyshev KAN, hidden 32, degree 5',
                     'optimizer': 'AdamW lr=0.005 weight_decay=0.0001', 'selection': 'best mean of clean and fixed -4dB validation macro F1' if args.augment_noise else 'best clean validation macro F1',
                     'python': platform.python_version(), 'torch': torch.__version__,
                     'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'source_files_and_raw_ranges': provenance, 'class_counts_train': counts.tolist(),
        'best_epoch': best_epoch, 'history': history, 'test_clean_0hp': test,
        'test_awgn_0hp_by_snr_db': noise_results, 'cross_load_1hp_clean': cross,
    }
    torch.save({'model_state_dict': best_state, 'fft_mean': mean, 'fft_std': std,
                'settings': report['settings']}, args.output / 'model.pt')
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"Test clean accuracy={test['accuracy']:.4f}, macro F1={test['macro_f1']:.4f}")
    print(f"Test -4 dB accuracy={noise_results['-4']['accuracy']:.4f}, macro F1={noise_results['-4']['macro_f1']:.4f}")
    print(f"Cross-load 1 hp accuracy={cross['accuracy']:.4f}, macro F1={cross['macro_f1']:.4f}")
    print(f"Saved {args.output / 'metrics.json'} and model.pt")


if __name__ == '__main__':
    main()
