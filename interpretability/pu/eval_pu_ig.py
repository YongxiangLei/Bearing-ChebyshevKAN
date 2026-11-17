"""IG evaluation on Paderborn whole-bearing holdout, no retraining.

Per-window preprocessing and test windows reproduce pu/eval_pu_bearing_holdout.py.
Stability noise is added in vibration space prior to the original preprocessing.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

parser = argparse.ArgumentParser()
parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().with_name('metrics.json'))
parser.add_argument('--steps', type=int, default=64)
parser.add_argument('--noise-repeats', type=int, default=5)
args = parser.parse_args()
root = args.project_root
sys.path.insert(0, str(root / 'pu'))
from eval_pu_bearing_holdout import Classifier, CONDITION, NAMES, SPLITS

torch.set_num_threads(2)
torch.manual_seed(42)
if args.steps < 1 or args.noise_repeats < 1:
    parser.error('steps and noise-repeats must be positive')
windows, labels, bearing_ids, metadata = [], [], [], []
for label, ids in SPLITS['test'].items():
    for bearing in ids:
        for recording in (1, 2):
            stem = f'{CONDITION}_{bearing}_{recording}'
            file = root / 'data' / 'PU' / 'selected_mat' / bearing / f'{stem}.mat'
            mat = loadmat(file, simplify_cells=True)
            channels = [c for c in mat[stem]['Y'] if c['Name'] == 'vibration_1']
            if len(channels) != 1:
                raise ValueError(f'Expected one vibration channel in {file}')
            raw = np.asarray(channels[0]['Data'], dtype=np.float64).reshape(-1)
            n = len(raw) // 1024
            if n < 100 or not np.isfinite(raw).all():
                raise ValueError(f'Bad recording: {file}')
            indices = np.linspace(0, n-1, 100, dtype=int)
            windows.extend([raw[i*1024:(i+1)*1024] for i in indices])
            labels.extend([NAMES.index(label)]*100)
            bearing_ids.extend([bearing]*100)
            metadata.append({'bearing': bearing, 'label': label, 'file': str(file.relative_to(root)), 'recording': recording, 'windows': 100})
raw = np.asarray(windows)
y = np.asarray(labels)
bearing_ids = np.asarray(bearing_ids)
checkpoint = torch.load(root/'pu'/'outputs'/'model.pt', map_location='cpu', weights_only=False)
mean, std = checkpoint['fft_mean'], checkpoint['fft_std']
net = Classifier(513)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

def transform(z):
    z = (z-z.mean(axis=1, keepdims=True))/(z.std(axis=1, keepdims=True)+1e-8)
    return torch.tensor(((np.abs(np.fft.rfft(z, axis=1))-mean)/std).astype(np.float32))

x = transform(raw)
with torch.no_grad():
    scores = net(x)
    pred, rival = scores.topk(2,dim=1).indices.unbind(1)
num, n_features = x.shape
assert num == 1200 and n_features == 513
assert abs(float((pred == torch.as_tensor(y)).float().mean())-0.3866666667)<1e-6, 'Loaded test data or checkpoint differs from baseline'
k = 52

def margin(z, a, b):
    v = net(z)
    return v.gather(1,a[:,None]).squeeze(1)-v.gather(1,b[:,None]).squeeze(1)

def ig(z, a, b):
    alphas=(torch.arange(args.steps,dtype=torch.float32)+.5)/args.steps
    chunks=[]
    for start in range(0,len(z),8):
        xx=z[start:start+8]; aa=a[start:start+8]; bb=b[start:start+8]
        path=(xx[:,None,:]*alphas[None,:,None]).reshape(-1,n_features).requires_grad_(True)
        grads=torch.autograd.grad(margin(path,aa.repeat_interleave(args.steps),bb.repeat_interleave(args.steps)).sum(),path)[0]
        chunks.append((xx*grads.reshape(len(xx),args.steps,n_features).mean(1)).detach())
    return torch.cat(chunks)

attribution=ig(x,pred,rival)
args.output.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(args.output.with_name('ig_test.npz'), attribution=attribution.numpy(),
                    true_label=y, pred=pred.numpy(), rival=rival.numpy(), bearing=bearing_ids,
                    standardized_fft=x.numpy())
print('Saved per-window attribution arrays', flush=True)
rank=attribution.abs().argsort(1,descending=True)
top_ids=rank[:,:k]
with torch.no_grad():
    base_margin=margin(x,pred,rival)
    delta=base_margin-margin(torch.zeros_like(x),pred,rival)
complete=(attribution.sum(1)-delta).abs()
attr_share=attribution.abs().gather(1,top_ids).sum(1)/attribution.abs().sum(1).clamp_min(1e-12)

def mask_result(ids):
    z=x.clone().scatter_(1,ids,0.)
    with torch.no_grad():
        p=net(z).argmax(1)
        diff=base_margin-margin(z,pred,rival)
    return {'accuracy':float((p==torch.as_tensor(y)).float().mean()),
            'prediction_agreement':float((p==pred).float().mean()),
            'mean_margin_drop':float(diff.mean())}

top_mask=mask_result(top_ids)
gen=torch.Generator().manual_seed(100)
random_runs=[mask_result(torch.rand((num,n_features),generator=gen).argsort(1)[:,:k]) for _ in range(20)]
random_result={key:float(np.mean([r[key] for r in random_runs])) for key in random_runs[0]}
random_result['repeats']=len(random_runs)
top_mask_bool=torch.zeros((num,n_features),dtype=torch.bool).scatter_(1,top_ids,True)
with torch.no_grad():
    top_pred=net(x.clone().scatter_(1,top_ids,0.)).argmax(1)
per_bearing={}
for b in sorted(set(bearing_ids)):
    ix=torch.tensor(bearing_ids==b)
    per_bearing[b]={'class':NAMES[y[bearing_ids==b][0]],'windows':int(ix.sum()),
                    'accuracy':float((pred[ix]==torch.as_tensor(y)[ix]).float().mean()),
                    'top_mask_prediction_agreement':float((top_pred[ix]==pred[ix]).float().mean()),
                    'top_attribution_share':float(attr_share[ix].mean())}
per_class={}
for cls,name in enumerate(NAMES):
    ix=torch.tensor(y==cls)
    per_class[name]={'windows':int(ix.sum()),
                     'accuracy':float((pred[ix]==torch.as_tensor(y)[ix]).float().mean()),
                     'top_mask_prediction_agreement':float((top_pred[ix]==pred[ix]).float().mean()),
                     'top_attribution_share':float(attr_share[ix].mean())}

stability={}
power=np.mean(raw**2,axis=1,keepdims=True)
for snr in (20,-4):
    draws=[]
    for seed in range(2026,2026+args.noise_repeats):
        rng=np.random.default_rng(seed)
        pert=raw+rng.normal(0,np.sqrt(power/(10**(snr/10))),raw.shape)
        xn=transform(pert)
        noisy_attr=ig(xn,pred,rival)
        noisy_top=noisy_attr.abs().argsort(1,descending=True)[:,:k]
        noisy_mask=torch.zeros((num,n_features),dtype=torch.bool).scatter_(1,noisy_top,True)
        j=(top_mask_bool&noisy_mask).sum(1).float()/(top_mask_bool|noisy_mask).sum(1).float()
        with torch.no_grad(): noisy_pred=net(xn).argmax(1)
        per={}
        for b in sorted(set(bearing_ids)):
            ix=torch.tensor(bearing_ids==b)
            per[b]={'mean_jaccard':float(j[ix].mean()),'prediction_agreement':float((noisy_pred[ix]==pred[ix]).float().mean()),
                    'accuracy':float((noisy_pred[ix]==torch.as_tensor(y)[ix]).float().mean())}
        same = noisy_pred == pred
        draws.append({'seed':seed,'mean_jaccard':float(j.mean()),'prediction_agreement':float(same.float().mean()),
                      'mean_jaccard_given_unchanged_prediction':float(j[same].mean()) if same.any() else None,
                      'accuracy':float((noisy_pred==torch.as_tensor(y)).float().mean()),'per_bearing':per})
        print(f'snr={snr} seed={seed} J={draws[-1]["mean_jaccard"]:.4f} agreement={draws[-1]["prediction_agreement"]:.4f}',flush=True)
    per={}
    for b in per_bearing:
        per[b]={key:float(np.mean([d['per_bearing'][b][key] for d in draws])) for key in ('mean_jaccard','prediction_agreement','accuracy')}
    stability[str(snr)]={'repeats':len(draws),
                         'mean_jaccard':float(np.mean([d['mean_jaccard'] for d in draws])),
                         'prediction_agreement':float(np.mean([d['prediction_agreement'] for d in draws])),
                         'mean_jaccard_given_unchanged_prediction':float(np.mean([d['mean_jaccard_given_unchanged_prediction'] for d in draws if d['mean_jaccard_given_unchanged_prediction'] is not None])),
                         'accuracy':float(np.mean([d['accuracy'] for d in draws])),
                         'per_bearing':per,'draws':draws}

output={'model':str(root/'pu'/'outputs'/'model.pt'),'n_windows':num,'n_bearings':len(per_bearing),'n_features':n_features,
 'test_accuracy':float((pred==torch.as_tensor(y)).float().mean()),'ig_steps':args.steps,'k':k,'k_fraction':k/n_features,
 'protocol':'whole-bearing holdout; two recordings and 100 nonoverlapping 1024-sample windows per bearing; original per-window z-score and FFT; saved train-only FFT scaling; IG for predicted vs runner-up logit margin with train-feature-mean baseline; add time-domain per-window-power AWGN before original preprocessing; hold clean classes fixed for IG on perturbed input',
 'caveats':['Accuracy is low: model reliance is not proof of correct fault diagnosis or causal physical relevance.',
            'Only 6 test bearings, 2 per class; windows and noise repeats are correlated, and do not estimate population uncertainty.',
            'Zeroed standardized FFT features and straight path from mean may be out-of-distribution; attribution depends on baseline.',
            'White noise is synthetic and -4 dB is strong; Jaccard compares fixed-class margin explanations even when predictions change.'],
 'sample_files':metadata,'completeness':{'mean_abs_error':float(complete.mean()),'mean_abs_target_delta':float(delta.abs().mean()),
                    'ratio_of_mean_abs_error_to_mean_abs_delta':float(complete.mean()/delta.abs().mean())},
 'top52_mean_abs_attribution_share':float(attr_share.mean()),'top52_mask':top_mask,
 'random52_mask_mean':random_result,'per_bearing':per_bearing,'per_class':per_class,'noise_stability':stability}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(output,ensure_ascii=False,indent=2),flush=True)
