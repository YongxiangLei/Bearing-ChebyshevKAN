"""Runnable adaptation of TIMC-26-1736 supplementary run_adaptive_xjtu.py.

The model, training and train/test bearing assignments follow the supplied PDF.
This CLI adds seed control, path checks and saved results. Real extracted XJTU-SY
CSV files are required; it never substitutes simulated data.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from adaptive_chebyshev_kan import AdaptiveChebyshevKANLayer

BEARINGS = {
    'train_1': ('35Hz12kN', 'Bearing1_1'),
    'train_2': ('35Hz12kN', 'Bearing1_2'),
    'same_domain_test': ('35Hz12kN', 'Bearing1_3'),
    'cross_domain_test': ('37.5Hz11kN', 'Bearing2_1'),
}

class AdaptiveChebyshevRegressor(nn.Module):
    def __init__(self, in_features, hidden_dim=32, max_degree=5):
        super().__init__()
        self.layer1 = AdaptiveChebyshevKANLayer(in_features, hidden_dim, max_degree)
        self.layer2 = AdaptiveChebyshevKANLayer(hidden_dim, 1, max_degree)
    def forward(self, x):
        return torch.sigmoid(self.layer2(self.layer1(x)))

class StandardMLPRegressor(nn.Module):
    def __init__(self, in_features, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
    def forward(self, x):
        return self.net(x)

def load_bearing(path, expected_count):
    if not path.is_dir():
        raise FileNotFoundError(f'Missing bearing directory: {path}')
    files = list(path.glob('*.csv'))
    if len(files) != expected_count:
        raise ValueError(f'Expected {expected_count} CSVs in {path}, found {len(files)}')
    try:
        files.sort(key=lambda f: int(f.stem))
    except ValueError as exc:
        raise ValueError(f'Expected numeric CSV names (1.csv, 2.csv, ...) in {path}') from exc
    if [int(f.stem) for f in files] != list(range(1, expected_count+1)):
        raise ValueError(f'Expected contiguous CSV names 1.csv through {expected_count}.csv in {path}')
    X = []
    for i, file in enumerate(files):
        df = pd.read_csv(file)
        if list(df.columns) != ['Horizontal_vibration_signals', 'Vertical_vibration_signals'] or len(df) != 32768:
            raise ValueError(f'Expected 32,768 samples and the two official vibration columns: {file}')
        arr = df.to_numpy(dtype=np.float64)
        if not np.isfinite(arr).all():
            raise ValueError(f'Non-finite values in {file}')
        feats = []
        for col in df.columns:
            sig = df[col].to_numpy(dtype=np.float64)
            feats.extend([np.sqrt(np.mean(sig**2)), np.max(np.abs(sig)), np.var(sig),
                          pd.Series(sig).kurtosis(), pd.Series(sig).skew()])
        if not np.isfinite(feats).all():
            raise ValueError(f'Non-finite extracted features in {file}')
        X.append(feats)
        if (i+1) % 100 == 0:
            print(f'{path.name}: {i+1}/{len(files)}', flush=True)
    return np.asarray(X), np.linspace(1.0, 0.0, len(files), dtype=np.float64)[:,None]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent.parent / 'data' / 'XJTU-SY' / 'XJTU-SY_Bearing_Datasets', help='Extracted XJTU-SY_Bearing_Datasets directory')
    ap.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'results')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--device', default='cpu', choices=['cpu','cuda','mps'])
    args = ap.parse_args()
    if args.epochs < 1: ap.error('--epochs must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available(): ap.error('CUDA is not available')
    if args.device == 'mps' and not torch.backends.mps.is_available(): ap.error('MPS is not available')
    paths = {name: args.data_dir.joinpath(*parts) for name, parts in BEARINGS.items()}
    missing = [str(p) for p in paths.values() if not p.is_dir()]
    if missing:
        raise FileNotFoundError('Extract the official dataset first; required directories missing:\n' + '\n'.join(missing))
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device == 'cuda': torch.cuda.manual_seed_all(args.seed)
    expected_counts = {'train_1': 123, 'train_2': 161, 'same_domain_test': 158, 'cross_domain_test': 491}
    data = {name: load_bearing(path, expected_counts[name]) for name, path in paths.items()}
    X = np.vstack([data['train_1'][0], data['train_2'][0]])
    y = np.vstack([data['train_1'][1], data['train_2'][1]])
    mean, std = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True) + 1e-8
    X = (X - mean) / std
    loader = DataLoader(TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y)), batch_size=32, shuffle=True)
    test = {name: (torch.FloatTensor((data[name][0] - mean) / std).to(args.device),
                   torch.FloatTensor(data[name][1]).to(args.device))
            for name in ('same_domain_test', 'cross_domain_test')}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {'seed': args.seed, 'epochs': args.epochs, 'device': args.device,
              'bearings': {k: str(v) for k,v in paths.items()},
              'sample_counts': {k: len(v[0]) for k,v in data.items()},
              'target': 'normalized remaining life: 1 at first file, 0 at last file',
              'feature_order': 'for each CSV column: RMS, peak absolute, variance, kurtosis, skewness',
              'models': {}}
    models = {'AdaptiveChebyshevKAN': AdaptiveChebyshevRegressor(X.shape[1]),
              'StandardMLP': StandardMLPRegressor(X.shape[1])}
    for name, model in models.items():
        model.to(args.device); model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        for epoch in range(args.epochs):
            total = 0.0
            for xb,yb in loader:
                xb, yb = xb.to(args.device), yb.to(args.device)
                optimizer.zero_grad(); loss=criterion(model(xb),yb); loss.backward(); optimizer.step()
                total += loss.item() * len(xb)
            if epoch == 0 or (epoch+1) % 10 == 0 or epoch+1 == args.epochs:
                print(f'{name} epoch {epoch+1}/{args.epochs}, train MSE={total/len(X):.6f}', flush=True)
        model.eval(); result['models'][name] = {}
        with torch.no_grad():
            for domain,(xt,yt) in test.items():
                pred = model(xt)
                mse = torch.mean((pred-yt)**2).item()
                mae = torch.mean(torch.abs(pred-yt)).item()
                result['models'][name][domain] = {'rmse': mse**0.5, 'mae': mae}
                print(f'{name} {domain}: RMSE={mse**0.5:.4f}, MAE={mae:.4f}', flush=True)
        torch.save({'state_dict': model.state_dict(), 'feature_mean': mean, 'feature_std': std,
                    'seed': args.seed, 'epochs': args.epochs}, args.output_dir / f'{name}.pt')
    (args.output_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

if __name__ == '__main__': main()
