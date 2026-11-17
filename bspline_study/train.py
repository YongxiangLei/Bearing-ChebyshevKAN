"""15-bearing exploratory B-spline KAN, using pre-existing CV splits.

Uses exactly the same train/validation/test bearing assignments, inputs,
target transform, validation macro-MAE selection, widths, and stopping budget
as the earlier Chebyshev experiment. Test bearings are never used for tuning.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from bspline_layer import BSplineKAN

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--patience',type=int,default=8)
    p.add_argument('--max-folds',type=int,default=0)
    p.add_argument('--test-bearing',default=None,help='Run exactly one named outer fold; exploratory only')
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args(); torch.set_num_threads(a.threads)
    sys.path.insert(0,str(a.project/'rul_study'))
    from build_features import FEATURES, EXPECTED_COUNTS
    from run_experiments import train_neural, metrics
    study=a.project/'rul_study'; inp=FEATURES+['elapsed_minutes','speed_hz','load_kN']
    df=pd.read_csv(study/'data'/'features_and_labels.csv')
    assert len(df)==sum(EXPECTED_COUNTS.values()) and set(df.bearing.unique())==set(EXPECTED_COUNTS)
    # The prior fold map is fixed by split_seed=42; seed changes only model training.
    existing=json.loads((study/'results_merged'/'folds.json').read_text(encoding='utf8'))
    folds=sorted([x for x in existing if x['training_seed']==42],key=lambda x:x['fold'])
    assert len(folds)==15
    a.output.mkdir(parents=True,exist_ok=True); (a.output/'checkpoints').mkdir(exist_ok=True)
    columns=inp; vals=df[columns].to_numpy(dtype=np.float64); y=df.rul_minutes.to_numpy(dtype=np.float64); ids=df.bearing.to_numpy()
    reports=[]; predictions=[]
    for fold in folds:
        if a.test_bearing and fold['test_bearing']!=a.test_bearing:continue
        if a.max_folds and len(reports)>=a.max_folds: break
        test=fold['test_bearing']; train=fold['train_bearings']; val=fold['validation_bearings']
        assert not(set(train)&set(val)) and test not in set(train)|set(val)
        tr=np.isin(ids,train); va=np.isin(ids,val); te=ids==test
        w=np.array([1/np.sum(ids[tr]==b) for b in ids[tr]]); w=w/w.mean()
        scaler=StandardScaler().fit(vals[tr],sample_weight=w)
        x=scaler.transform(vals)
        candidates=[]
        for width in [8,16]:
            net,info=train_neural('bspline',width,x[tr],y[tr],w,x[va],y[va],ids[va],a.seed+fold['fold'],a.epochs,a.patience,{'bspline':BSplineKAN})
            candidates.append((info['validation_macro_mae_min'],width,net,info))
        _, width, net, info=min(candidates,key=lambda z:z[0]); net.eval()
        with torch.no_grad():
            testp=np.maximum(0,net(torch.tensor(x[te],dtype=torch.float32)).numpy()*info['target_train_std']+info['target_train_mean'])
        late=y[te]<=.2*(np.sum(te)-1)
        test_metrics=metrics(y[te],testp); test_metrics['late_last_20pct']=metrics(y[te][late],testp[late])
        rep={'test_bearing':test,'fold':fold['fold'],'seed':a.seed,'train_bearings':train,'validation_bearings':val,
             'width':width,'validation_macro_mae_min':info['validation_macro_mae_min'],'best_epoch':info['best_epoch'],
             'candidate_validation':[{k:v for k,v in c[3].items() if k!='history'} for c in candidates],
             'test':test_metrics, 'test_fraction_standardized_abs_gt_3_by_feature':np.mean(np.abs(x[te])>3,axis=0).tolist()}
        ck={'state_dict':net.state_dict(),'model':'BSplineKAN','input_columns':columns,'width':width,'intervals':6,'degree':3,
            'scaler_mean':scaler.mean_,'scaler_scale':scaler.scale_,'target_train_mean':info['target_train_mean'],
            'target_train_std':info['target_train_std'],'train_bearings':train,'validation_bearings':val,'test_bearing':test,
            'training_seed':a.seed,'best_epoch':info['best_epoch'],
            'train_feature_quantiles':np.quantile(vals[tr],[.01,.05,.5,.95,.99],axis=0)}
        torch.save(ck,a.output/'checkpoints'/f'{test}_seed{a.seed}.pt')
        q=df.loc[te,['bearing','condition','record','rul_minutes']].copy(); q['predicted_rul_min']=testp;q['seed']=a.seed;predictions.append(q)
        reports.append(rep)
        print(f'fold {fold["fold"]+1} {test} width={width} val_mae={info["validation_macro_mae_min"]:.2f} test_mae={test_metrics["mae_min"]:.2f}',flush=True)
        (a.output/'folds.json').write_text(json.dumps(reports,indent=2),encoding='utf8')
        pd.concat(predictions).to_csv(a.output/'predictions.csv',index=False)
    if not reports: raise ValueError('No folds selected')
    summary={'completed_folds':len(reports),'seed':a.seed,'complete_15_bearings':len(reports)==15,
             'protocol':'Same original CV folds and train-only scaling, degree-3 open-uniform B-spline KAN, width selected on 3 validation bearings',
             'architecture':'phi(z)=sum c_k B_{k,3}(3*tanh(z/3)), z standardized input; each layer sums edge functions; 6 fixed equal intervals, 9 basis functions per edge; AdamW weight decay only, no explicit sparsity penalty',
             'mae_macro_min':float(np.mean([r['test']['mae_min'] for r in reports])),
             'rmse_macro_min':float(np.mean([r['test']['rmse_min'] for r in reports])),
             'late_overestimate_macro_min':float(np.mean([r['test']['late_last_20pct']['mean_overestimate_min'] for r in reports])),
             'limitations':['Targets minutes until final recorded observation, not precise failure time.', 'Only 15 bearings, one validation split per outer fold; seen operating conditions.', 'Bounded tanh mapping compresses extreme values; curves are not raw-input splines or a proven physical law.', 'No explicit sparsity regularization.']}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
