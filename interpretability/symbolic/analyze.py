import sys, json, copy
from pathlib import Path
import numpy as np
import sympy as sp
import torch
from sklearn.metrics import accuracy_score, f1_score

root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'cwru'/'exp'))
import os
os.environ['CWRU_DATA_DIR']=str(root/'data'/'CWRU')
from run_adaptive_cwru import AdaptiveChebyshevClassifier
from eval_cwru_holdout import split_recordings, fft

torch.set_num_threads(2)
weights=root/'cwru'/'exp'/'outputs'/'cwru_augmented'/'model.pt'
ckpt=torch.load(weights,map_location='cpu',weights_only=False)
model=AdaptiveChebyshevClassifier(257,4,hidden_dim=32,max_degree=5)
model.load_state_dict(ckpt['model_state_dict']); model.eval()
splits,provenance=split_recordings()
x,y=splits['test']
features=((fft(x)-ckpt['fft_mean'])/ckpt['fft_std']).astype(np.float32)
X=torch.from_numpy(features)
def preds(m):
    with torch.no_grad():
        z=torch.cat([m(batch) for batch in X.split(64)])
    return z.numpy(),z.argmax(1).numpy()
logits,p=preds(model)
layer=model.layer1
c=(layer.coeffs*torch.sigmoid(layer.order_weights)).detach().numpy()
out_idx,in_idx=np.unravel_index(np.abs(c[:,:,1:]).sum(axis=-1).argmax(),c.shape[:2])
co=c[out_idx,in_idx]
t=sp.Symbol('t',real=True)
expr=sp.expand(sum(float(v)*sp.chebyshevt(i,t) for i,v in enumerate(co)))
z=features[:,in_idx]; t_values=np.tanh(z)
with torch.no_grad():
    basis=layer.chebyshev(torch.tanh(torch.tensor(z)))
    edge=(basis*torch.tensor(co)).sum(dim=1).numpy()
sym=sp.lambdify(t,expr,'numpy')(t_values)
rawcoeff=layer.coeffs[out_idx,in_idx].detach().numpy()
# The PDF helper falls back to raw coeffs because there is no order_attention,
# then generates T_d(x) and thresholds at 0.01 without accounting for tanh.
rawcoeff=np.where(np.abs(rawcoeff)>0.01,rawcoeff,0)
pdf_formula=np.polynomial.chebyshev.chebval(z,rawcoeff)
records=[]
for threshold in [0,.001,.005,.01,.02,.05,.1]:
    pruned=copy.deepcopy(model)
    before=kept=0
    with torch.no_grad():
        for l in [pruned.layer1,pruned.layer2]:
            eff=l.coeffs*torch.sigmoid(l.order_weights)
            mask=(eff.abs()>=threshold)
            before+=mask.numel();kept+=int(mask.sum())
            l.coeffs.mul_(mask)
    pl,pp=preds(pruned)
    records.append({'threshold':threshold,'total_coefficients':before,'retained':kept,'pruned_fraction':(before-kept)/before,
                    'accuracy':accuracy_score(y,pp),'macro_f1':f1_score(y,pp,average='macro'),
                    'prediction_agreement':float(np.mean(p==pp)), 'logit_mae':float(np.mean(np.abs(logits-pl)))})
report={'checkpoint':str(weights),'test_n':len(y),'original_accuracy':accuracy_score(y,p),'original_macro_f1':f1_score(y,p,average='macro'),
        'edge_selection':'first-layer edge with largest sum of absolute effective coefficients degrees 1..5; no claim that this is the most causally important input',
        'hidden_index_zero_based':int(out_idx),'input_bin_zero_based':int(in_idx),'bin_spacing_hz_if_12khz':12000/512,
        'bin_center_hz_if_12khz':float(in_idx*12000/512),'coefficients_T0_to_T5':co.tolist(),
        'expanded_in_t':str(expr),'t_definition':f't=tanh((abs(rfft(window))[{in_idx}]-train_mean[{in_idx}])/train_std[{in_idx}])',
        'max_absolute_edge_reconstruction_error':float(np.max(np.abs(edge-sym))),
        'pdf_extractor_edge_error_on_test':{'mean_absolute':float(np.mean(np.abs(pdf_formula-edge))),
                                            'maximum_absolute':float(np.max(np.abs(pdf_formula-edge))),
                                            'true_edge_mean_absolute':float(np.mean(np.abs(edge)))},
        'test_t_range':[float(t_values.min()),float(t_values.max())],
        'thresholded_model_fidelity_on_same_temporal_test':records,
        'warning':'One edge is not a complete classifier explanation; coefficient thresholding is not proof of physical meaning. Only 0hp time-holdout test; no new bearing generalization.'}
out=Path(__file__).parent/'symbolic_metrics.json';out.write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
