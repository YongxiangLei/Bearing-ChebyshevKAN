"""Raw FFT IG near CWRU fault frequencies; descriptive, not causal validation."""
import os,sys,json
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat

PROJECT=Path(os.environ.get('BEARING_PROJECT',Path(__file__).resolve().parents[2]))
os.environ['CWRU_DATA_DIR']=str(PROJECT/'data'/'CWRU')
sys.path.insert(0,str(PROJECT/'cwru'/'exp'))
from run_adaptive_cwru import AdaptiveChebyshevClassifier
from eval_cwru_holdout import split_recordings,fft
torch.set_num_threads(2)
ckpt=torch.load(PROJECT/'cwru/exp/outputs/cwru_augmented/model.pt',map_location='cpu',weights_only=False)
net=AdaptiveChebyshevClassifier(257,4,hidden_dim=32,max_degree=5)
net.load_state_dict(ckpt['model_state_dict']);net.eval()
raws,_=split_recordings(); raw,y=raws['test']
X=torch.tensor(((fft(raw)-ckpt['fft_mean'])/ckpt['fft_std']).astype(np.float32))
with torch.no_grad(): logits=net(X); top=logits.topk(2,dim=1).indices
def margin(x,a,b):
    z=net(x);return z.gather(1,a[:,None]).squeeze(1)-z.gather(1,b[:,None]).squeeze(1)
def ig(x,a,b):
    steps=64; alphas=(torch.arange(steps,dtype=torch.float32)+0.5)/steps; result=[]
    for i in range(0,len(x),8):
        xi=x[i:i+8]; ai=a[i:i+8]; bi=b[i:i+8]
        path=(xi[:,None,:]*alphas[None,:,None]).reshape(-1,257).requires_grad_(True)
        grads=torch.autograd.grad(margin(path,ai.repeat_interleave(steps),bi.repeat_interleave(steps)).sum(),path)[0]
        result.append((xi*grads.reshape(len(xi),steps,257).mean(1)).detach())
    return torch.cat(result)
attrs=ig(X,top[:,0],top[:,1]).abs().numpy(); xnp=X.numpy(); delta=12000/512
rng=np.random.default_rng(20261111)
cfg=[(1,'Inner Race',105,5.4152),(2,'Outer Race',130,3.5848),(3,'Ball',118,4.7135)]
res={}
for label,name,file,coef in cfg:
    rpm=float(loadmat(PROJECT/'data'/'CWRU'/f'{file}.mat')[f'X{file}RPM'].item())
    f=rpm/60*coef; idx=np.flatnonzero(y==label); relevance=attrs[idx]; denominators=relevance.sum(1)
    rr={'n_windows':len(idx),'file':f'{file}.mat','rpm':rpm,'frequency_coefficient_official':coef,'frequency_hz':f,'masks':{}}
    for num in [1,3]:
        centres=[int(np.floor((f*h/delta)+.5)) for h in range(1,num+1)]
        bins=sorted(set(i for k in centres for i in (k-1,k,k+1) if 1<=i<=42))
        random_masks=np.stack([rng.choice(np.arange(1,43),size=len(bins),replace=False) for _ in range(1000)])
        frac=float((relevance[:,bins].sum(1)/denominators).mean())
        rfrac=np.array([(relevance[:,b].sum(1)/denominators).mean() for b in random_masks])
        x=torch.from_numpy(xnp[idx].copy());a=top[idx,0];b=top[idx,1]
        with torch.no_grad():
            base=margin(x,a,b);original=net(x).argmax(1);x[:,bins]=0
            pert=margin(x,a,b); new=net(x).argmax(1)
        rr['masks'][str(num)+'_harmonics']={'nearest_bins':centres,'bins':bins,'bin_hz':[round(i*delta,3) for i in bins],
            'mean_abs_attribution_share':frac,'random_equal_count_bins_1_to_42_mean':float(rfrac.mean()),
            'random_control_2_5_97_5_percentiles':np.percentile(rfrac,[2.5,97.5]).tolist(),
            'mean_margin_drop_when_masked':float((base-pert).mean()),'flipped_count':int((original!=new).sum())}
    res[name]=rr
out={'task':'CWRU fault-only attribution band description in raw magnitude FFT space',
     'model':'cwru_augmented','sampling_hz':12000,'fft_length':512,'bin_spacing_hz':delta,'steps':64,
     'source':'https://engineering.case.edu/bearingdatacenter/bearing-information',
     'rpm_source':'per-record MAT RPM variable (0hp fault files)',
     'formula':'CWRU published defect-frequency multiplier times rpm / 60',
     'basis':'Predicted-vs-runner-up logit margin IG relative to per-feature training mean. 1000 deterministic equal-size random-bin controls 1..42 (up to 984.375 Hz).',
     'interpretation_caution':'Observational on 0hp held-out temporal intervals, one recording per class, overlapping windows. Averaged samples not independent. Raw FFT is not envelope spectrum: harmonics could be modulations of high-frequency resonances rather than prominent FFT components. Random control ignores spectral autocorrelation and class structure; mean baseline masking may be off manifold. Frequencies are nominal; bin resolution is coarse. Attribution is model reliance, not evidence of causal fault mechanism. The official 4.7135 multiplier represents rolling element characteristic and is ~2 times standard BSF definition.',
     'classes':res}
Path(__file__).with_name('metrics.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out,indent=2))
