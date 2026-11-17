"""Leave-one-bearing-out XJTU-SY observed-endpoint RUL model comparison.

For each held-out test bearing, one other bearing from each operating condition
is reserved for validation; all remaining bearings are training. Standardization,
target scaling, optimization and validation-driven selection never use the test
bearing. Per-bearing equal weights prevent longer runs dominating model fitting.
"""
import argparse
import copy
import json
import random
import sys
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn

from build_features import FEATURES, CONDITIONS, EXPECTED_COUNTS


def metrics(y,p):
    e=p-y; return {'n':len(y),'mae_min':float(np.mean(np.abs(e))),
                   'rmse_min':float(np.sqrt(np.mean(e**2))),
                   'mean_overestimate_min':float(np.maximum(e,0).mean()),
                   'overestimate_rate':float(np.mean(e>0))}

def macro_mae(y,p,ids):
    return np.mean([np.abs(y[ids==b]-p[ids==b]).mean() for b in np.unique(ids)])

def net_class(root):
    sys.path.insert(0,str(root/'cwru'/'exp'))
    from adaptive_chebyshev_kan import AdaptiveChebyshevKANLayer
    class KAN(nn.Module):
        def __init__(self,dim,width):
            super().__init__(); self.l1=AdaptiveChebyshevKANLayer(dim,width,3); self.l2=AdaptiveChebyshevKANLayer(width,1,3)
        def forward(self,x): return self.l2(self.l1(x)).squeeze(1)
    class MLP(nn.Module):
        def __init__(self,dim,width):
            super().__init__(); self.net=nn.Sequential(nn.Linear(dim,width),nn.ReLU(),nn.Linear(width,1))
        def forward(self,x): return self.net(x).squeeze(1)
    return {'kan':KAN,'mlp':MLP}

def train_neural(kind,width,X,y,w,Xval,yval,vid,seed,epochs,patience,classes):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    net=classes[kind](X.shape[1],width)
    opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=1e-4)
    # Weighted targets use only training bearings.
    ym=np.average(y,weights=w); ys=np.sqrt(np.average((y-ym)**2,weights=w))+1e-8
    xx=torch.tensor(X,dtype=torch.float32); tt=torch.tensor((y-ym)/ys,dtype=torch.float32); ww=torch.tensor(w,dtype=torch.float32)
    xv=torch.tensor(Xval,dtype=torch.float32)
    rng=torch.Generator().manual_seed(seed)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xx,tt,ww),batch_size=512,shuffle=True,generator=rng)
    best=float('inf'); state=None; stale=0; best_epoch=0; hist=[]
    for epoch in range(1,epochs+1):
        net.train()
        for xb,yb,wb in loader:
            opt.zero_grad(); err=net(xb)-yb; loss=(wb*err.square()).mean(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): p=np.maximum(0,net(xv).numpy()*ys+ym)
        score=float(macro_mae(yval,p,vid)); hist.append({'epoch':epoch,'validation_macro_mae_min':score})
        if score<best-1e-6:
            best=score; best_epoch=epoch; state=copy.deepcopy(net.state_dict()); stale=0
        else:
            stale+=1
            if stale>=patience:break
    if state is None: raise ValueError('Model produced no finite validation score')
    net.load_state_dict(state); net.eval()
    return net,{'width':width,'best_epoch':best_epoch,'validation_macro_mae_min':best,'target_train_mean':float(ym),
                'target_train_std':float(ys),'history':hist}

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--features',type=Path,default=Path(__file__).parent/'data'/'features_and_labels.csv')
    a.add_argument('--project',type=Path,default=Path(__file__).resolve().parent.parent)
    a.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    a.add_argument('--epochs',type=int,default=60); a.add_argument('--patience',type=int,default=8)
    a.add_argument('--seed',type=int,default=42); a.add_argument('--threads',type=int,default=2)
    a.add_argument('--seeds',default='42,43,44',help='Comma-separated training seeds. Validation splits stay fixed by --seed.')
    a.add_argument('--only-folds',type=int,default=0,help='Smoke test only; 0 runs every bearing')
    args=a.parse_args(); torch.set_num_threads(args.threads)
    df=pd.read_csv(args.features); args.output.mkdir(parents=True,exist_ok=True)
    required=['bearing','condition','record','elapsed_minutes','rul_minutes','speed_hz','load_kN']+FEATURES
    if set(required)-set(df): raise ValueError('Missing input columns')
    if set(df.bearing.unique())!=set(EXPECTED_COUNTS): raise ValueError('Requires all 15 official bearings')
    for b,g in df.groupby('bearing'):
        if len(g)!=EXPECTED_COUNTS[b] or g.condition.nunique()!=1 or sorted(g.record.tolist())!=list(range(1,len(g)+1)):
            raise ValueError(f'Invalid data for {b}')
        if not np.allclose(g.elapsed_minutes.to_numpy(),g.record.to_numpy()-1) or not np.allclose(g.rul_minutes.to_numpy(),len(g)-g.record.to_numpy()):
            raise ValueError(f'Invalid elapsed or endpoint RUL labels for {b}')
    input_columns=FEATURES+['elapsed_minutes','speed_hz','load_kN']; age_columns=['elapsed_minutes','speed_hz','load_kN']
    vals=df[input_columns].to_numpy(dtype=float); ids=df.bearing.to_numpy(); y=df.rul_minutes.to_numpy(dtype=float)
    if not np.isfinite(vals).all() or not np.isfinite(y).all(): raise ValueError('Nonfinite data')
    condmap={b:df.loc[df.bearing==b,'condition'].iloc[0] for b in sorted(set(ids))}
    if set(condmap.values())!=set(CONDITIONS): raise ValueError('Unexpected condition')
    classes=net_class(args.project); all_predictions=[]; fold_reports=[]
    seeds=[int(s.strip()) for s in args.seeds.split(',')]
    if not seeds or len(set(seeds))!=len(seeds): raise ValueError('seeds must be unique integers')
    fold_pairs=[(i,b,s) for i,b in enumerate(sorted(condmap)) if not args.only_folds or i<args.only_folds for s in seeds]
    for fold,test_b,run_seed in fold_pairs:
        rng=np.random.default_rng(args.seed+fold)
        # Reserve one complete non-test bearing from each condition as validation.
        val_b=[str(rng.choice([b for b in sorted(condmap) if condmap[b]==c and b!=test_b])) for c in CONDITIONS]
        train_b=[b for b in sorted(condmap) if b!=test_b and b not in val_b]
        if (set(train_b)&set(val_b)) or test_b in set(train_b)|set(val_b) or len(train_b)!=11: raise AssertionError('split leakage')
        tr=np.isin(ids,train_b); va=np.isin(ids,val_b); te=(ids==test_b)
        # All scaler statistics are fit solely on training bearings.
        train_ids=ids[tr]; w=np.array([1/np.sum(train_ids==b) for b in train_ids]); w=w/w.mean()
        scaler=StandardScaler().fit(vals[tr], sample_weight=w); Xtr=scaler.transform(vals[tr]); Xv=scaler.transform(vals[va]); Xt=scaler.transform(vals[te])
        age_idx=[input_columns.index(c) for c in age_columns]
        opts={}; pred={}; candidate_logs={}; checkpoint={'test_bearing':test_b,'validation_bearings':val_b,'train_bearings':train_b,
            'input_columns':input_columns,'scaler_mean':scaler.mean_, 'scaler_scale':scaler.scale_,'training_seed':run_seed,'models':{}}
        for name,cols in [('age_ridge',age_idx),('features_ridge',list(range(len(input_columns))))]:
            results=[]
            for alpha in (0.1,10.0):
                m=Ridge(alpha=alpha).fit(Xtr[:,cols],y[tr],sample_weight=w)
                pv=np.maximum(m.predict(Xv[:,cols]),0); score=float(macro_mae(y[va],pv,ids[va])); results.append((score,alpha,m))
            score,alpha,m=min(results,key=lambda x:x[0]); pred[name]=np.maximum(m.predict(Xt[:,cols]),0)
            opts[name]={'alpha':alpha,'validation_macro_mae_min':score,'candidate_validation_scores':{str(al):sc for sc,al,_ in results},'uses_features':[input_columns[i] for i in cols]}
            checkpoint['models'][name]={'coef':m.coef_,'intercept':m.intercept_,'columns':cols,'alpha':alpha}
        for name,widths in [('mlp',[64,128]),('kan',[8,16])]:
            candidates=[]
            for width in widths:
                model,info=train_neural(name,width,Xtr,y[tr],w,Xv,y[va],ids[va],run_seed+fold,args.epochs,args.patience,classes)
                candidates.append((info['validation_macro_mae_min'],width,model,info))
            score,width,model,info=min(candidates,key=lambda x:x[0])
            with torch.no_grad(): pred[name]=np.maximum(model(torch.tensor(Xt,dtype=torch.float32)).numpy()*info['target_train_std']+info['target_train_mean'],0)
            candidate_logs[name]=[q[3] for q in candidates]; opts[name]=info
            checkpoint['models'][name]={'state_dict':model.state_dict(),'width':width,'target_train_mean':info['target_train_mean'],'target_train_std':info['target_train_std']}
        true=y[te]; tdf=df.loc[te,['bearing','condition','record','source_file','elapsed_minutes','rul_minutes']].copy()
        n=len(tdf); late=true<=.2*(n-1) # evaluation only; end-point information is never an input
        fold_metrics={}
        for name,predictions in pred.items():
            tdf[name+'_pred_min']=predictions
            fold_metrics[name]=metrics(true,predictions)
            fold_metrics[name]['late_last_20pct']=metrics(true[late],predictions[late])
        tdf['late_last_20pct_evaluation_only']=late; all_predictions.append(tdf)
        tdf['seed']=run_seed
        report={'fold':fold,'training_seed':run_seed,'test_bearing':test_b,'test_condition':condmap[test_b],
                'validation_bearings':val_b,'train_bearings':train_b,'scaler_fit_bearings':train_b,
                'selected':opts,'neural_candidate_history':candidate_logs,'test_metrics':fold_metrics}
        fold_reports.append(report)
        print('FOLD',fold+1,'seed',run_seed,test_b,{name:round(v['mae_min'],2) for name,v in fold_metrics.items()},flush=True)
        (args.output/'checkpoints').mkdir(exist_ok=True)
        torch.save(checkpoint,args.output/'checkpoints'/f'{test_b}_seed{run_seed}.pt')
        # Save incrementally so partial progress remains inspectable if interrupted.
        pd.concat(all_predictions,ignore_index=True).to_csv(args.output/'predictions.csv',index=False)
        (args.output/'folds.json').write_text(json.dumps(fold_reports,indent=2),encoding='utf8')
    summary={}
    for name in ('age_ridge','features_ridge','mlp','kan'):
        q=[f['test_metrics'][name] for f in fold_reports]
        summary[name]={k:float(np.mean([m[k] for m in q])) for k in ('mae_min','rmse_min','mean_overestimate_min','overestimate_rate')}
        summary[name]['late_last_20pct_macro']={k:float(np.mean([m['late_last_20pct'][k] for m in q])) for k in ('mae_min','rmse_min','mean_overestimate_min','overestimate_rate')}
        summary[name]['mean_mae_by_training_seed']={str(s):float(np.mean([f['test_metrics'][name]['mae_min'] for f in fold_reports if f['training_seed']==s])) for s in seeds}
        summary[name]['mean_mae_by_condition']={c:float(np.mean([f['test_metrics'][name]['mae_min'] for f in fold_reports if f['test_condition']==c])) for c in CONDITIONS if any(f['test_condition']==c for f in fold_reports)}
        summary[name]['mean_mae_by_test_bearing']={b:float(np.mean([f['test_metrics'][name]['mae_min'] for f in fold_reports if f['test_bearing']==b])) for b in sorted({f['test_bearing'] for f in fold_reports})}
    result={'complete_15_folds':len(fold_reports)==15*len(seeds),'folds_completed':len(fold_reports),'split_seed':args.seed,'training_seeds':seeds,
            'protocol':'LOBO outer test. One non-test bearing per condition held out for validation; all remaining bearings train; validation and test bearings excluded from all fitted scaling, model selection and training. Metrics macro-averaged over test bearings.',
            'candidate_budget':'two alpha values per ridge; two widths per neural model; same optimizer and epoch budget per neural candidate; identical fixed validation splits for all training seeds',
            'model_note':'KAN is adaptive gated Chebyshev polynomial layer from paper supplement (degree 3), not sparse B-spline KAN; no explicit sparsity regularization. Widths 8/16 vs MLP widths 64/128 give roughly comparable parameter counts.',
            'target':'minutes to last recorded observation = number of future records at 1-minute interval; last observed timestamp is a proxy for endpoint, not known exact failure time.',
            'limitations':['15 bearings total, only one outer test bearing per fold and one fixed validation split per fold; seeds are not new bearings.',
                           'Present-record features plus elapsed time, speed and load only; no future lifetime, future record or full-test normalization in inputs.',
                           'No independent unseen-condition validation: each fold has training bearings from all three conditions.'],
            'input_columns':input_columns,'age_baseline_columns':age_columns,'epochs_max':args.epochs,'patience':args.patience,
            'versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__},
            'macro_results':summary}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
