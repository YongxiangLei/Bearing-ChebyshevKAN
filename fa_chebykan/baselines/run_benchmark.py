"""Shared recording-disjoint pretraining and few-shot benchmark.

The feature/model architectures differ, but data, units, seed, epoch budget,
validation criterion and training implementation are common.  This is a
single-seed exploratory experiment, not a generalization claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent / 'project'
if not (PROJECT / 'experiment.py').is_file():
    # Deployed layout: fa_chebykan/baselines/ next to experiment.py and core/.
    PROJECT = BASE.parent
sys.path.insert(0, str(PROJECT))
from data import LABELS, load_csv
from experiment import prepare, fit, infer, metrics, few_shot
from core.model import FAConfig
from models import make_model

NAMES = ['fa_cheby', 'anchored_mlp', 'spectrum_mlp', 'spectrum_cheby']


def state_copy(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_ckpt(path, model, config, name, seed, band, meta):
    torch.save({'state_dict': state_copy(model), 'config':vars(config), 'model_name':name,
                'seed':seed, 'band_hz':band, **meta}, path)


def split(records):
    parts = tuple([r for r in records if r['split']==name] for name in ('train', 'validation', 'test'))
    if any(not p for p in parts):
        raise ValueError('train, validation and test must all have records')
    return parts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--simulation', required=True)
    p.add_argument('--simulation-envelope', required=True)
    p.add_argument('--real')
    p.add_argument('--real-envelope')
    p.add_argument('--output', required=True)
    p.add_argument('--pretrained-dir', help='Use saved simulated weights for sensitivity scenario; skip pretraining')
    p.add_argument('--models', nargs='+', choices=NAMES, default=NAMES)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--few-shot-epochs', type=int, default=15)
    p.add_argument('--ks', type=int, nargs='+', default=[1,2,3,5,10])
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--harmonics', type=int, default=3)
    p.add_argument('--sidebands', type=int, default=2)
    p.add_argument('--degree', type=int, default=3)
    p.add_argument('--hidden', type=int, default=16)
    p.add_argument('--band', type=float, nargs=2, default=[1600,4800])
    a=p.parse_args()
    if min(a.epochs,a.few_shot_epochs,a.batch_size,a.threads)<1 or any(k<1 for k in a.ks):
        raise ValueError('epochs, batch size, threads and K must all be positive')
    if bool(a.real) != bool(a.real_envelope):
        raise ValueError('--real and --real-envelope must be supplied together')
    torch.set_num_threads(a.threads)
    cfg=FAConfig(harmonics=a.harmonics,sidebands=a.sidebands,degree=a.degree,hidden=a.hidden)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    sim=prepare(load_csv(a.simulation), a.simulation, a.simulation_envelope, a.band)
    if {r['dataset'] for r in sim}!={'synthetic'}:
        raise ValueError('pretraining must use simulation only')
    tr, val, te = split(sim)
    real_parts=None
    if a.real:
        real=prepare(load_csv(a.real), a.real, a.real_envelope, a.band)
        if any(r['dataset']=='synthetic' for r in real) or len({r['dataset'] for r in real})!=1:
            raise ValueError('target dataset must contain one non-synthetic dataset')
        real_parts=split(real)
        real_labels=sorted({r['label'] for r in real_parts[0]},key=LABELS.index)
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    report={'scope':'single-seed recording-disjoint exploratory sim-to-real benchmark; repeated test assessment is exploratory',
            'seed':a.seed,'config':vars(cfg),'epochs':a.epochs,'few_shot_epochs':a.few_shot_epochs,
            'batch_size':a.batch_size,'learning_rate':a.lr,'band_hz':a.band,
            'simulation_csv_sha256':digest(a.simulation),'simulation_envelopes_sha256':digest(a.simulation_envelope),
            'real_csv_sha256':digest(a.real) if a.real else None,
            'real_envelopes_sha256':digest(a.real_envelope) if a.real else None,
            'parameter_counts_include_auxiliary_heads':True,'models':{}}
    # Checkpoints can be reused for a predefined input sensitivity analysis.
    for name in a.models:
        print(f'===== MODEL {name} =====', flush=True)
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
        model=make_model(name,cfg)
        model_stats={'parameter_count':sum(p.numel() for p in model.parameters())}
        if a.pretrained_dir:
            ckpt_path=Path(a.pretrained_dir)/f'{name}_pretrained.pt'
            ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
            if (ckpt['config']!=vars(cfg) or ckpt['model_name']!=name or ckpt['seed']!=a.seed or
                ckpt['simulation_csv_sha256']!=report['simulation_csv_sha256'] or
                ckpt['simulation_envelopes_sha256']!=report['simulation_envelopes_sha256'] or
                ckpt['band_hz']!=a.band):
                raise ValueError(f'Pretrained checkpoint metadata mismatch: {ckpt_path}')
            model.load_state_dict(ckpt['state_dict'])
            model_stats['pretraining_history']=ckpt.get('history', [])
        else:
            history=fit(model,tr,val,a.epochs,a.batch_size,a.lr,a.seed,True)
            model_stats['pretraining_history']=history
            save_ckpt(out/f'{name}_pretrained.pt',model,cfg,name,a.seed,a.band,
                      {'simulation_csv_sha256':report['simulation_csv_sha256'],
                       'simulation_envelopes_sha256':report['simulation_envelopes_sha256'], 'history':history})
        pre_state=state_copy(model)
        model_stats['simulated_test']=metrics(infer(model,te,a.batch_size))
        model_stats['real']=None
        if real_parts:
            rt, rv, re = real_parts
            r={'zero_shot_test':metrics(infer(model,re,a.batch_size)),
               'few_shot_training_classes':real_labels,'few_shot':{}}
            for k in a.ks:
                try:
                    subset,ids=few_shot(rt,real_labels,k,a.seed+k)
                except ValueError as exc:
                    r['few_shot'][str(k)]={'status':'infeasible','reason':str(exc)}
                    continue
                runs={}
                for mode in ['pretrained','scratch']:
                    random.seed(a.seed+k);np.random.seed(a.seed+k);torch.manual_seed(a.seed+k)
                    if mode=='pretrained':
                        model.load_state_dict(pre_state)
                        current=model
                    else:
                        current=make_model(name,cfg)
                    history=fit(current,subset,rv,a.few_shot_epochs,a.batch_size,a.lr,a.seed+k,False)
                    runs[mode]={'history':history,'test':metrics(infer(current,re,a.batch_size))}
                    save_ckpt(out/f'{name}_k{k}_{mode}.pt',current,cfg,name,a.seed+k,a.band,
                              {'selected_recordings':ids,'real_csv_sha256':report['real_csv_sha256'],
                               'real_envelopes_sha256':report['real_envelopes_sha256']})
                r['few_shot'][str(k)]={'status':'completed','training_recordings':ids,**runs}
            model_stats['real']=r
        report['models'][name]=model_stats
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(f'Results saved to {out}',flush=True)


if __name__=='__main__':
    main()
