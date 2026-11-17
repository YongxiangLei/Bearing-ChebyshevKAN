"""Simulation-only pretraining, plus recording-disjoint real transfer evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path
import torch
import numpy as np
from torch import nn
from data import LABELS, load_csv
from core.model import FAConfig, FaultAnchoredChebyKAN


def prepare(records, source, cache, band):
    """Load a separately computed envelope cache, bound to exact dataset bytes."""
    digest=hashlib.sha256(Path(source).read_bytes()).hexdigest()
    with np.load(cache,allow_pickle=False) as data:
        if str(data['dataset_sha256'])!=digest:
            raise ValueError('Envelope cache does not match current dataset')
        if not np.allclose(data['band_hz'],band):
            raise ValueError('Envelope cache was made with a different band')
        if int(data['count'])!=len(records): raise ValueError('Envelope count mismatch')
        for i,r in enumerate(records):
            env=data[f'env_{i}']
            if env.ndim!=1 or len(env)<2 or not np.isfinite(env).all():
                raise ValueError('Invalid envelope in cache')
            if f'fs_{i}' in data:
                fs=float(data[f'fs_{i}'])
                speed=np.asarray(data[f'speed_{i}'],dtype=np.float32)
                if not np.isfinite(fs) or fs<=0 or speed.ndim>1 or (speed.ndim==1 and speed.shape!=env.shape) or not np.isfinite(speed).all() or (speed<=0).any():
                    raise ValueError('Invalid cached envelope sampling rate or speed')
                expected=int(np.ceil(len(r['x'])*fs/float(r['fs'])))
                if len(env)!=expected:
                    raise ValueError('Cached envelope length does not match output sampling rate')
                r['fs']=fs
                r['shaft_speed']=float(speed) if speed.ndim==0 else speed
            elif env.shape!=r['x'].shape:
                raise ValueError('Legacy native-rate envelope has wrong shape')
            r['envelope']=env.astype(np.float32)
    return records


def batches(records, n, shuffle=False, seed=0):
    """Batch equal-length recordings while keeping native sample rates."""
    rng=np.random.default_rng(seed)
    groups={}
    for r in records:
        groups.setdefault(len(r['envelope']),[]).append(r)
    all_batches=[]
    for rows in groups.values():
        indices=np.arange(len(rows))
        if shuffle: rng.shuffle(indices)
        for k in range(0,len(indices),n):
            all_batches.append([rows[i] for i in indices[k:k+n]])
    if shuffle: rng.shuffle(all_batches)
    return all_batches


def to_tensors(rows):
    x=torch.tensor(np.stack([r['envelope'] for r in rows]),dtype=torch.float32)
    fs=torch.tensor([r['fs'] for r in rows],dtype=torch.float32)
    speed=torch.tensor(np.stack([np.full(len(r['envelope']),float(r['shaft_speed']))
        if np.ndim(r['shaft_speed'])==0 else np.asarray(r['shaft_speed']) for r in rows]),dtype=torch.float32)
    geom=torch.tensor([[r['z'],r['d_mm'],r['D_mm'],r['alpha_deg']] for r in rows],dtype=torch.float32)
    y=torch.tensor([LABELS.index(r['label']) for r in rows],dtype=torch.long)
    return x,fs,speed,geom,y


def infer(model, records, batch_size):
    model.eval(); entries=[]
    with torch.no_grad():
        for rows in batches(records,batch_size):
            x,fs,speed,geom,y=to_tensors(rows)
            out=model(x,fs,speed,geom)
            for i,r in enumerate(rows):
                entries.append({'recording_id':r['recording_id'],'dataset':r['dataset'],
                    'label':int(y[i]),'logits':out['logits'][i].numpy(),
                    'features':out['features'][i].numpy()})
    return entries


def metrics(entries):
    # One independent recording receives one vote, irrespective of segment count.
    groups={}
    for e in entries:
        key=(e['dataset'],e['recording_id'])
        groups.setdefault(key,[]).append(e)
    matrix=np.zeros((len(LABELS),len(LABELS)),dtype=int)
    for key,rows in groups.items():
        ys={r['label'] for r in rows}
        if len(ys)!=1: raise ValueError(f'Recording {key} has inconsistent labels')
        y=ys.pop(); pred=int(np.mean([r['logits'] for r in rows],axis=0).argmax())
        matrix[y,pred]+=1
    support=matrix.sum(1); tp=np.diag(matrix)
    denom=2*tp+(matrix.sum(0)-tp)+(support-tp)
    f1=np.divide(2*tp,denom,out=np.zeros(len(LABELS)),where=denom!=0)
    present=support>0
    return {'recordings':int(support.sum()), 'segments':len(entries),
            'accuracy':float(tp.sum()/support.sum()) if support.sum() else None,
            'macro_f1_present_classes':float(f1[present].mean()) if present.any() else None,
            'class_order':list(LABELS),'class_recording_counts':support.tolist(),
            'confusion_matrix':matrix.tolist()}


def fit(model, train, val, epochs, batch_size, lr, seed, regression):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    best=-1.; best_state=None; hist=[]
    # Each independent recording has equal aggregate training weight even
    # when recording lengths lead to different numbers of segments.
    counts={}
    for r in train:
        key=(r['dataset'],r['recording_id'])
        counts[key]=counts.get(key,0)+1
    if not counts:
        raise ValueError('Training split has no recordings')
    weight_scale=len(train)/len(counts)
    for epoch in range(epochs):
        model.train(); total=0.; count=0
        for rows in batches(train,batch_size,shuffle=True,seed=seed+epoch):
            x,fs,speed,geom,y=to_tensors(rows)
            out=model(x,fs,speed,geom)
            weights=torch.tensor([weight_scale/counts[(r['dataset'],r['recording_id'])] for r in rows],dtype=torch.float32)
            loss=(weights*nn.functional.cross_entropy(out['logits'],y,reduction='none')).mean()
            if regression:
                if any(r['A0'] is None or r['decay'] is None for r in rows):
                    raise ValueError('Simulation pretraining requires A0 and decay')
                a=torch.tensor([r['A0']/2 for r in rows],dtype=torch.float32)
                d=torch.tensor([r['decay']/1000 for r in rows],dtype=torch.float32)
                loss=loss+0.1*(weights*(out['amplitude']-a).square()).mean()+0.1*(weights*(out['decay']-d).square()).mean()
            opt.zero_grad();loss.backward();opt.step()
            total+=float(loss.detach())*len(rows);count+=len(rows)
        v=metrics(infer(model,val,batch_size)); score=v['macro_f1_present_classes']
        hist.append({'epoch':epoch+1,'train_loss':total/count,'validation':v})
        print(f'epoch={epoch+1} loss={total/count:.4f} val_macro_f1={score:.4f}',flush=True)
        if score > best:
            best=score;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return hist


def few_shot(train, labels, k, seed):
    # Select K distinct recordings for each available class. Segments from
    # each chosen recording are permitted, but cannot inflate K.
    rng=np.random.default_rng(seed); chosen=[]
    for label in labels:
        ids=sorted({r['recording_id'] for r in train if r['label']==label})
        if len(ids)<k:
            raise ValueError(f'{label}: K={k} needs {k} independent training recordings, available {len(ids)}')
        chosen.extend(rng.choice(ids,size=k,replace=False).tolist())
    return [r for r in train if r['recording_id'] in chosen], chosen


def coverage(train_entries, development_entries):
    """Descriptive NN coverage against a simulation-only feature reference.

    Standardise using simulated training only. The 95th percentile reference
    distance leaves out all segments of the same recording.
    """
    x=np.stack([r['features'] for r in train_entries]).astype(float)
    z=np.stack([r['features'] for r in development_entries]).astype(float)
    mu=x.mean(0);sd=x.std(0);sd[sd<1e-8]=1.
    x=(x-mu)/sd;z=(z-mu)/sd
    dist=np.sqrt(np.maximum(0.,((x[:,None,:]-x[None,:,:])**2).sum(2)))
    ids=[(r['dataset'],r['recording_id']) for r in train_entries]
    for i in range(len(ids)):
        for j in range(len(ids)):
            if ids[i]==ids[j]: dist[i,j]=np.inf
    nearest=dist.min(1)
    if not np.isfinite(nearest).all():
        raise ValueError('Coverage requires at least two independent simulated training recordings')
    threshold=float(np.percentile(nearest,95))
    real_dist=np.sqrt(np.maximum(0.,((z[:,None,:]-x[None,:,:])**2).sum(2))).min(1)
    return {'reference':'simulated training only, standardized using training mean/std; same-record exclusion',
            'threshold_sim_loo_95':threshold,'development_segments':len(z),
            'development_outside_fraction':float(np.mean(real_dist>threshold)),
            'development_nearest_distance_median':float(np.median(real_dist)),
            'caution':'segment-level descriptive statistic; not a probability or guarantee of transfer'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--simulation',required=True,help='Simulation CSV with fixed train/validation/test')
    p.add_argument('--simulation-envelope',required=True,help='Output of preprocess.py for the simulation CSV')
    p.add_argument('--real',help='Optional real CSV with independent recording splits and complete geometry')
    p.add_argument('--real-envelope',help='Output of preprocess.py for the real CSV')
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--few-shot-epochs',type=int,default=3)
    p.add_argument('--ks',type=int,nargs='*',default=[1,3,5,10])
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--harmonics',type=int,default=3)
    p.add_argument('--sidebands',type=int,default=2)
    p.add_argument('--hidden',type=int,default=16)
    p.add_argument('--band',type=float,nargs=2,default=[1600,4800],metavar=('LOW','HIGH'))
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    if min(a.epochs,a.few_shot_epochs,a.batch_size,a.threads)<1: raise ValueError('positive epochs, batch and threads required')
    if any(k<1 for k in a.ks): raise ValueError('K must be positive')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(a.threads)
    sim=prepare(load_csv(a.simulation),a.simulation,a.simulation_envelope,tuple(a.band))
    if {r['dataset'] for r in sim}!={'synthetic'}: raise ValueError('Pretraining input must be simulation-only')
    tr=[r for r in sim if r['split']=='train'];val=[r for r in sim if r['split']=='validation'];te=[r for r in sim if r['split']=='test']
    if not all((tr,val,te)): raise ValueError('Simulation requires disjoint train, validation and test')
    cfg=FAConfig(harmonics=a.harmonics,sidebands=a.sidebands,hidden=a.hidden)
    model=FaultAnchoredChebyKAN(cfg)
    h=fit(model,tr,val,a.epochs,a.batch_size,a.lr,a.seed,True)
    report={'scope':'Prototype; simulation is illustrative and not validated as a real rig. K counts distinct recordings, not windows.',
            'seed':a.seed,'fixed_band_hz':a.band,'model_config':vars(cfg),
            'simulation_history':h,'simulation_test':metrics(infer(model,te,a.batch_size)),
            'target_scaling':{'A0_divisor':2,'decay_divisor':1000},'real':None}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    pre_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    torch.save({'config':vars(cfg),'state_dict':pre_state,'band_hz':a.band},out/'pretrained.pt')
    if a.real:
        if not a.real_envelope: raise ValueError('--real requires --real-envelope')
        real=prepare(load_csv(a.real),a.real,a.real_envelope,tuple(a.band))
        if any(r['dataset']=='synthetic' for r in real): raise ValueError('Real evaluation must not be synthetic')
        if len({r['dataset'] for r in real}) != 1: raise ValueError('Evaluate one real dataset per run')
        rt=[r for r in real if r['split']=='train'];rv=[r for r in real if r['split']=='validation'];re=[r for r in real if r['split']=='test']
        if not all((rt,rv,re)): raise ValueError('Real train, validation and test recordings are required')
        labels=sorted({r['label'] for r in rt},key=LABELS.index)
        sim_z=infer(model,tr,a.batch_size)
        real_z=infer(model,re,a.batch_size)
        development_z=infer(model,rv,a.batch_size)
        rr={'zero_shot_test':metrics(real_z),'coverage_development':coverage(sim_z,development_z),
            'few_shot_class_set_source':'real training labels only; absent classes are not fine-tuned',
            'few_shot_classes':labels,'few_shot':{}}
        for k in a.ks:
            try: subset,ids=few_shot(rt,labels,k,a.seed+k)
            except ValueError as exc:
                rr['few_shot'][str(k)]={'status':'infeasible','reason':str(exc)};continue
            model.load_state_dict(pre_state)
            hist=fit(model,subset,rv,a.few_shot_epochs,a.batch_size,a.lr,a.seed+k,False)
            transfer_test=metrics(infer(model,re,a.batch_size))
            torch.manual_seed(a.seed+k)
            scratch=FaultAnchoredChebyKAN(cfg)
            scratch_hist=fit(scratch,subset,rv,a.few_shot_epochs,a.batch_size,a.lr,a.seed+k,False)
            rr['few_shot'][str(k)]={'training_recordings':ids,
                'pretrained':{'history':hist,'test':transfer_test},
                'scratch':{'history':scratch_hist,'test':metrics(infer(scratch,re,a.batch_size))}}
        report['real']=rr
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'simulation_test':report['simulation_test'],'real':report['real']},indent=2))


if __name__=='__main__':
    main()
