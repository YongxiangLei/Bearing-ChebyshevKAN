"""Evaluate integrated-gradient explanations on saved CWRU test windows.

All results are model- and baseline-specific; input masking is an intervention
in standardized FFT space, not a physical vibration signal intervention.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
os.environ['CWRU_DATA_DIR'] = str(PROJECT / 'data' / 'CWRU')
sys.path.insert(0, str(PROJECT / 'cwru' / 'exp'))
from eval_cwru_holdout import split_recordings, fft
from run_adaptive_cwru import AdaptiveChebyshevClassifier, NAMES

torch.set_num_threads(2)
torch.manual_seed(42)
np.random.seed(42)
model_path = PROJECT / 'cwru' / 'exp' / 'outputs' / 'cwru_augmented' / 'model.pt'
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
net = AdaptiveChebyshevClassifier(257, 4, hidden_dim=32, max_degree=5)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()
splits, provenance = split_recordings()
raw, y = splits['test']
mean, std = checkpoint['fft_mean'], checkpoint['fft_std']
x = torch.tensor(((fft(raw)-mean)/std).astype(np.float32))
with torch.no_grad():
    scores = net(x)
    top = scores.topk(2, dim=1).indices
pred, rival = top[:, 0], top[:, 1]
num, features = x.shape
assert num == len(y) == 470 and features == 257
steps, k = 64, 26

def margin(t, a, b):
    out = net(t)
    return out.gather(1, a[:, None]).squeeze(1) - out.gather(1, b[:, None]).squeeze(1)

def integrated_gradients(inp, a, b):
    # Midpoint rule over baseline (zero normalized features) to input.
    all_attr = []
    alphas = (torch.arange(steps, dtype=torch.float32)+.5) / steps
    for start in range(0, len(inp), 8):
        xx = inp[start:start+8]
        aa, bb = a[start:start+8], b[start:start+8]
        path = (xx[:, None, :] * alphas[None, :, None]).reshape(-1, features).requires_grad_(True)
        margins = margin(path, aa.repeat_interleave(steps), bb.repeat_interleave(steps))
        grad = torch.autograd.grad(margins.sum(), path)[0].reshape(len(xx), steps, features).mean(1)
        all_attr.append((xx * grad).detach())
    return torch.cat(all_attr)

attr = integrated_gradients(x, pred, rival)
with torch.no_grad():
    base = torch.zeros_like(x)
    unmasked_margin = margin(x, pred, rival)
    delta = unmasked_margin - margin(base, pred, rival)
absolute_error = (attr.sum(1) - delta).abs()
ranked = attr.abs().argsort(dim=1, descending=True)
energy = attr.abs()
attribution_share = energy.gather(1, ranked[:, :k]).sum(1) / energy.sum(1).clamp(min=1e-12)

def evaluate_mask(ids):
    altered = x.clone()
    altered.scatter_(1, ids, 0.)
    with torch.no_grad():
        out = net(altered)
        altered_pred = out.argmax(1)
        diff = margin(altered, pred, rival)
    return {'accuracy': float((altered_pred == torch.as_tensor(y)).float().mean()),
            'prediction_flip_rate': float((altered_pred != pred).float().mean()),
            'mean_margin_drop': float((unmasked_margin - diff).mean()),
            'median_margin_drop': float((unmasked_margin - diff).median())}

top_result = evaluate_mask(ranked[:, :k])
bottom_result = evaluate_mask(ranked[:, -k:])
masked_top = x.clone().scatter_(1, ranked[:, :k], 0.)
with torch.no_grad():
    masked_top_pred = net(masked_top).argmax(1)
    masked_top_margin = margin(masked_top, pred, rival)
per_class_masking = {}
for c, name in enumerate(NAMES):
    ix = torch.tensor(y == c)
    per_class_masking[name] = {'n': int(ix.sum()),
        'prediction_flip_rate': float((masked_top_pred[ix] != pred[ix]).float().mean()),
        'mean_margin_drop': float((unmasked_margin[ix] - masked_top_margin[ix]).mean())}
generator = torch.Generator().manual_seed(100)
random_results = [evaluate_mask(torch.rand((num, features), generator=generator).argsort(1)[:, :k]) for _ in range(20)]
random_summary = {key: float(np.mean([r[key] for r in random_results])) for key in random_results[0]}
random_summary['repeats'] = len(random_results)

stability = {}
for snr in [20, -4]:
    power = np.mean(raw**2, axis=1, keepdims=True)
    topmask = torch.zeros((num,features), dtype=torch.bool).scatter_(1, ranked[:, :k], True)
    draws = []
    for seed in range(2026, 2031):
        rng = np.random.default_rng(seed)
        perturbed = raw + rng.normal(0, np.sqrt(power / (10**(snr/10))), raw.shape)
        xn = torch.tensor(((fft(perturbed)-mean)/std).astype(np.float32))
        pert_attr = integrated_gradients(xn, pred, rival)
        pert_rank = pert_attr.abs().argsort(1, descending=True)[:, :k]
        pertmask = torch.zeros((num,features), dtype=torch.bool).scatter_(1, pert_rank, True)
        jaccard = (topmask & pertmask).sum(1).float() / (topmask | pertmask).sum(1).float()
        with torch.no_grad(): noisy_pred = net(xn).argmax(1)
        draws.append({'seed':seed,'mean_top26_jaccard':float(jaccard.mean()),
                      'prediction_agreement':float((noisy_pred == pred).float().mean()),
                      'test_accuracy':float((noisy_pred == torch.as_tensor(y)).float().mean())})
    stability[str(snr)] = {'repeats':len(draws),
                           'mean_top26_jaccard':float(np.mean([d['mean_top26_jaccard'] for d in draws])),
                           'prediction_agreement':float(np.mean([d['prediction_agreement'] for d in draws])),
                           'test_accuracy':float(np.mean([d['test_accuracy'] for d in draws])),
                           'draws':draws}

class_info = {}
for label, name in enumerate(NAMES):
    selected = energy[torch.tensor(y == label)].mean(0)
    bins = selected.topk(10).indices.tolist()
    class_info[name] = {'n': int(np.sum(y == label)), 'top_10_bins': bins,
                        'top_10_hz_if_fs_12000': [round(i * 12000/512, 3) for i in bins]}

report = {
 'model': str(model_path), 'test_n': int(num), 'test_accuracy': float((pred == torch.as_tensor(y)).float().mean()),
 'protocol': '0hp held-out final 20% temporal block per class, 512 samples, 50% overlap. Original augmented model and saved training normalization. Explain predicted-vs-runner-up logit margin with 64-midpoint integrated gradients along path from standardized FFT mean (zero).',
 'limitations': ['Mean-vector baseline and zeroing spectral bins may be off-manifold. Attribution is model reliance, not physical causation.',
                 'Train and test windows come from different time blocks of the same per-class recording; overlapping windows are correlated and do not constitute independent bearings.',
                 'Stability uses the same fixed original predicted class and runner-up margin before and after noise; top-k Jaccard is an explanation similarity, not ground-truth correctness.',
                 '12 kHz frequency mapping is applicable to 12 kHz fault recordings, but normal sampling rate is not established for these files.'],
 'ig_steps': steps, 'masked_feature_count': k, 'masked_feature_fraction': k/features,
 'completeness': {'mean_absolute_margin_error': float(absolute_error.mean()), 'median_absolute_margin_error': float(absolute_error.median()),
                  'mean_abs_true_margin_delta': float(delta.abs().mean()),
                  'mean_absolute_error_over_mean_abs_delta': float(absolute_error.mean()/delta.abs().mean())},
 'top26_share_of_absolute_attribution_mean': float(attribution_share.mean()),
 'top26_share_of_absolute_attribution_median': float(attribution_share.median()),
 'masking_top26': top_result, 'masking_top26_by_class': per_class_masking,
 'masking_bottom26': bottom_result, 'masking_random26_mean': random_summary,
 'noise_stability': stability, 'class_top_bins': class_info,
 'class_order': NAMES, 'test_class_counts': np.bincount(y, minlength=4).tolist(),
 'provenance': [p for p in provenance if p['split'] == 'test']}
out = Path(__file__).with_name('metrics.json')
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
