"""Audit existing XJTU-SY RUL model on its two held-out bearings.

This analyzes the historical normalized-life model, not minute RUL or an
online-calibrated prediction interval. Output changes under masking show model
reliance, not causal effects or validated failure mechanisms.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    root = args.project.resolve()
    sys.path.insert(0, str(root / 'xjtu'))
    from run_xjtu import AdaptiveChebyshevRegressor, load_bearing

    torch.set_num_threads(2)
    np.random.seed(42); random.seed(42); torch.manual_seed(42)
    checkpoint_path = root / 'xjtu' / 'results' / 'AdaptiveChebyshevKAN.pt'
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    mu = np.asarray(ckpt['feature_mean'], dtype=np.float64)
    sd = np.asarray(ckpt['feature_std'], dtype=np.float64)
    model = AdaptiveChebyshevRegressor(10)
    model.load_state_dict(ckpt['state_dict']); model.eval()
    names = [f'{channel}_{metric}' for channel in ['horizontal', 'vertical'] for metric in ['rms', 'peak_abs', 'variance', 'kurtosis', 'skewness']]
    bearings = [('same_domain', '35Hz12kN', 'Bearing1_3', 158), ('cross_domain', '37.5Hz11kN', 'Bearing2_1', 491)]
    ds_root = root / 'data' / 'XJTU-SY' / 'XJTU-SY_Bearing_Datasets'
    reports = {}
    rankings = {}
    for domain, condition, bearing, count in bearings:
        raw, y = load_bearing(ds_root / condition / bearing, count)
        X = torch.tensor((raw - mu) / sd, dtype=torch.float32)
        with torch.no_grad():
            pred = model(X).squeeze(1)
            baseline_prediction = float(model(torch.zeros((1, 10))).item())
        # 64 midpoint integrated gradients from training feature means.
        grad_sum = torch.zeros_like(X)
        for alpha in (torch.arange(64, dtype=torch.float32) + 0.5) / 64:
            z = (alpha * X).detach().requires_grad_(True)
            model(z).sum().backward()
            grad_sum += z.grad.detach()
        ig = X * grad_sum / 64
        err = torch.abs(ig.sum(dim=1) - (pred - baseline_prediction))
        delta = torch.abs(pred - baseline_prediction)
        abs_ig = ig.abs()
        order = torch.argsort(abs_ig, dim=1, descending=True)
        top_idx = order[:, :3]
        top_x = X.clone(); top_x.scatter_(1, top_idx, 0.)
        with torch.no_grad():
            ptop = model(top_x).squeeze(1)
        rng = np.random.default_rng(42)
        random_shifts = []
        with torch.no_grad():
            for _ in range(50):
                cols = np.stack([rng.choice(10, 3, replace=False) for _ in range(len(X))])
                z = X.clone(); z.scatter_(1, torch.from_numpy(cols), 0.)
                random_shifts.append(float(torch.abs(model(z).squeeze(1) - pred).mean()))
        global_importance = abs_ig.mean(dim=0).numpy()
        rank = np.argsort(-global_importance)
        rankings[domain] = [names[i] for i in rank]
        reports[domain] = {
            'bearing': bearing, 'n': count, 'true_target': 'relative remaining lifetime, linearly 1 to 0 using observed run endpoint; not minutes',
            'rmse': float(np.sqrt(mean_squared_error(y.reshape(-1), pred.numpy()))),
            'mae': float(mean_absolute_error(y.reshape(-1), pred.numpy())),
            'baseline_prediction': baseline_prediction,
            'ig_mean_absolute_completeness_error': float(err.mean()),
            'ig_mean_absolute_prediction_difference_from_baseline': float(delta.mean()),
            'ig_completeness_ratio_of_means': float(err.mean() / (delta.mean() + 1e-12)),
            'global_mean_abs_attribution': {name: float(global_importance[i]) for i, name in enumerate(names)},
            'top5_feature_ranking': [names[i] for i in rank[:5]],
            'top3_absolute_attribution_share': float((abs_ig.gather(1, top_idx).sum(1) / abs_ig.sum(1).clamp_min(1e-12)).mean()),
            'mask_top3_mean_absolute_prediction_change': float((ptop-pred).abs().mean()),
            'mask_random3_mean_absolute_prediction_change_50_repeats': float(np.mean(random_shifts)),
        }
        print(domain, bearing, json.dumps(reports[domain], ensure_ascii=False), flush=True)
    a, b = set(rankings['same_domain'][:5]), set(rankings['cross_domain'][:5])
    result = {'model':str(checkpoint_path), 'baseline':'training feature mean (standardized zero)', 'ig_steps':64,
              'top5_jaccard_between_the_two_test_bearings':len(a & b)/len(a | b),
              'limitations':['Only two held-out bearings and one random seed; time samples within one bearing are correlated.',
                             'The model target is normalized using full observed endpoint, not online available minutes. This audit explains predictions, not actual causal mechanism.',
                             'The two bearings differ in both condition and identity; their explanation differences cannot be attributed solely to condition.',
                             'Features are correlated and masking features to their training means can create off-manifold examples.'],
              'feature_names': names, 'results':reports}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('top5_jaccard',result['top5_jaccard_between_the_two_test_bearings'])

if __name__ == '__main__':
    main()
