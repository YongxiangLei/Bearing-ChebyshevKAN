"""Three-class, bearing-disjoint Paderborn benchmark with repeated seeds.

The input CSV uses recording_id as the physical bearing ID.  Consequently all
segments and all acquisitions of an individual bearing stay together in every
split, training loss and evaluation aggregation.  This is an exploratory
comparison under one fixed split, not a confidence interval over bearings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project',type=Path,required=True,help='FA-ChebyKAN project code directory')
parser.add_argument('--simulation',type=Path,required=True)
parser.add_argument('--simulation-envelope',type=Path,required=True)
parser.add_argument('--target',type=Path,required=True,help='Paderborn CSV with bearing ID as recording_id')
parser.add_argument('--target-envelope',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
parser.add_argument('--models',nargs='+',default=['fa_cheby','anchored_mlp','spectrum_mlp','spectrum_cheby'])
parser.add_argument('--pretrain-epochs',type=int,default=20)
parser.add_argument('--epochs',type=int,default=20)
parser.add_argument('--batch-size',type=int,default=8)
parser.add_argument('--threads',type=int,default=2)
parser.add_argument('--lr',type=float,default=0.001)
parser.add_argument('--band',type=float,nargs=2,default=[1600,4800])
parser.add_argument('--hidden',type=int,default=16)
parser.add_argument('--harmonics',type=int,default=3)
parser.add_argument('--sidebands',type=int,default=2)
parser.add_argument('--degree',type=int,default=3)
args=parser.parse_args()

sys.path.insert(0,str(args.project.resolve()))
sys.path.insert(0,str((args.project/'baselines').resolve()))
from data import load_csv
from experiment import prepare,fit,infer
from core.model import FAConfig
from models import make_model

CLASSES=('normal','outer','inner')

def sha(path):
    obj=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(1024*1024): obj.update(b)
    return obj.hexdigest()

def split(rows,domain):
    parts={s:[r for r in rows if r['split']==s and r['label'] in CLASSES]
           for s in ('train','validation','test')}
    seen={}
    for part,rs in parts.items():
        present={r['label'] for r in rs}
        if present != set(CLASSES):
            raise ValueError(f'{domain} {part} lacks a required class: {present}')
        for r in rs:
            ident=(r['dataset'],r['recording_id'])
            if ident in seen and seen[ident]!=part: raise ValueError(f'{ident} crosses split')
            seen[ident]=part
    return parts

def inventory(parts):
    result={}
    for part,rows in parts.items():
        by_class={lab:sorted({r['recording_id'] for r in rows if r['label']==lab}) for lab in CLASSES}
        mapping={}
        for row in rows:
            key=row['recording_id']
            if key in mapping and mapping[key]!=row['label']: raise ValueError(f'{key} has mixed labels')
            mapping[key]=row['label']
        result[part]={'bearings_by_class':by_class,'segment_count':len(rows),
                      'segments_per_bearing':dict(sorted(Counter(r['recording_id'] for r in rows).items()))}
    return result

def assess(model,rows):
    entries=infer(model,rows,args.batch_size)
    groups=defaultdict(list)
    for e in entries: groups[(e['dataset'],e['recording_id'])].append(e)
    matrix=np.zeros((3,3),dtype=int)
    details=[]
    for (_,ident),observations in sorted(groups.items()):
        labels={r['label'] for r in observations}
        if len(labels)!=1: raise ValueError(f'{ident} has inconsistent labels')
        true=labels.pop()
        if true>=3: raise ValueError('target class mapping outside normal/outer/inner')
        logits=np.mean([r['logits'] for r in observations],axis=0)
        pred=int(logits.argmax())
        matrix[true,pred]+=1
        details.append({'bearing':ident,'label':CLASSES[true],'predicted':CLASSES[pred],
                        'segments':len(observations),'mean_logits':[float(x) for x in logits]})
    counts=matrix.sum(axis=1)
    tp=np.diag(matrix)
    recall=np.divide(tp,counts,out=np.zeros(3,dtype=float),where=counts>0)
    pred_counts=matrix.sum(axis=0)
    precision=np.divide(tp,pred_counts,out=np.zeros(3,dtype=float),where=pred_counts>0)
    denom=2*tp+pred_counts-tp+counts-tp
    f1=np.divide(2*tp,denom,out=np.zeros(3,dtype=float),where=denom>0)
    return {'accuracy':float(tp.sum()/counts.sum()),'balanced_accuracy':float(recall.mean()),
            'macro_f1':float(f1.mean()),'class_order':list(CLASSES),'bearings':int(counts.sum()),
            'segments':len(entries),'support':counts.tolist(),'confusion_matrix':matrix.tolist(),
            'precision':precision.tolist(),'recall':recall.tolist(),'f1':f1.tolist(),'per_bearing':details}

def selection(history):
    values=[x['validation']['macro_f1_present_classes'] for x in history]
    return int(np.argmax(values)+1)

def seed_all(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)

def save_report(path,report):
    tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    tmp.replace(path)

if any(x<1 for x in [args.pretrain_epochs,args.epochs,args.batch_size,args.threads]) or len(set(args.seeds))!=len(args.seeds):
    raise ValueError('Budgets must be positive and seeds unique')
torch.set_num_threads(args.threads)
sim=prepare(load_csv(args.simulation),args.simulation,args.simulation_envelope,args.band)
if {r['dataset'] for r in sim}!={'synthetic'}: raise ValueError('Only simulation can be used for pretraining')
target=prepare(load_csv(args.target),args.target,args.target_envelope,args.band)
if {r['dataset'] for r in target}!={'Paderborn'}: raise ValueError('Expected Paderborn target')
source=split(sim,'simulation'); dst=split(target,'Paderborn')
cfg=FAConfig(classes=3,harmonics=args.harmonics,sidebands=args.sidebands,hidden=args.hidden,degree=args.degree)
args.output.mkdir(parents=True,exist_ok=True)
report={'protocol':'Fixed bearing-disjoint partitions; multiple random seeds; target and simulation normal/outer/inner only. Each target recording_id denotes one physical bearing. Test predictions are exploratory and are never used for model selection.',
        'target_inventory':inventory(dst), 'simulation_inventory':inventory(source),
        'seeds':args.seeds,'models':args.models,'config':vars(cfg),'band_hz':args.band,
        'pretrain_epochs_max':args.pretrain_epochs,'target_epochs_max':args.epochs,
        'batch_size':args.batch_size,'threads':args.threads,'learning_rate':args.lr,
        'sha256':{'simulation_csv':sha(args.simulation),'simulation_envelope':sha(args.simulation_envelope),
                  'target_csv':sha(args.target),'target_envelope':sha(args.target_envelope)},'runs':{}}
save_report(args.output/'report.json',report)
for name in args.models:
    report['runs'][name]={}
    for seed in args.seeds:
        print(f'===== {name} seed {seed}: SIMULATION PRETRAIN =====',flush=True)
        seed_all(seed)
        model=make_model(name,cfg)
        source_history=fit(model,source['train'],source['validation'],args.pretrain_epochs,args.batch_size,args.lr,seed,True)
        source_test=assess(model,source['test'])
        zero=assess(model,dst['test'])
        pretrained_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        torch.save({'state_dict':pretrained_state,'model':name,'seed':seed,'config':vars(cfg),
                    'source':'three-class synthetic','selected_epoch':selection(source_history)},args.output/f'{name}_seed{seed}_source.pt')
        run={'parameter_count':sum(p.numel() for p in model.parameters()),'source_history':source_history,
             'source_selected_epoch':selection(source_history),'source_test':source_test,'zero_shot':zero}
        for mode in ('pretrained','scratch'):
            print(f'===== {name} seed {seed}: TARGET {mode} =====',flush=True)
            seed_all(seed)
            current=make_model(name,cfg)
            if mode=='pretrained':current.load_state_dict(pretrained_state)
            history=fit(current,dst['train'],dst['validation'],args.epochs,args.batch_size,args.lr,seed,False)
            run[mode]={'history':history,'selected_epoch':selection(history),'test':assess(current,dst['test'])}
            torch.save({'state_dict':current.state_dict(),'model':name,'seed':seed,'mode':mode,
                        'config':vars(cfg),'selected_epoch':selection(history)},args.output/f'{name}_seed{seed}_{mode}.pt')
        report['runs'][name][str(seed)]=run
        save_report(args.output/'report.json',report)
        print(f'COMPLETED {name} seed {seed}',flush=True)
print(f'All runs saved to {args.output}',flush=True)
