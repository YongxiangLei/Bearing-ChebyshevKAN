"""Independently verify B-spline folds against saved Chebyshev CV protocol."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--spline-dir',type=Path,default=Path(__file__).resolve().parent.parent/'bspline_study')
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('fold_audit.json'))
    a=p.parse_args(); sys.path.insert(0,str(a.spline_dir)); from bspline_layer import BSplineKAN
    torch.set_num_threads(2)
    root=a.project; study=root/'rul_study'; spline=a.spline_dir/'results_seed42'
    features=pd.read_csv(study/'data'/'features_and_labels.csv')
    original=json.loads((study/'results_merged'/'folds.json').read_text(encoding='utf8'))
    original={f['test_bearing']:f for f in original if f['training_seed']==42}
    folds=json.loads((spline/'folds.json').read_text(encoding='utf8'))
    predictions=pd.read_csv(spline/'predictions.csv')
    assert len(folds)==15 and len(original)==15 and len(predictions)==9216
    assert len({f['test_bearing'] for f in folds})==15
    assert not predictions.duplicated(['bearing','record']).any()
    max_scaler=0.; max_pred=0.; max_target=0.; max_metric=0.
    for f in folds:
        b=f['test_bearing']; g=original[b]
        assert f['train_bearings']==g['train_bearings'] and f['validation_bearings']==g['validation_bearings']
        assert b not in set(f['train_bearings'])|set(f['validation_bearings'])
        ck=torch.load(spline/'checkpoints'/f'{b}_seed42.pt',map_location='cpu',weights_only=False)
        assert ck['train_bearings']==f['train_bearings'] and ck['validation_bearings']==f['validation_bearings']
        train=features[features.bearing.isin(f['train_bearings'])]
        counts=train.groupby('bearing').bearing.transform('size').to_numpy(); weights=1/counts;weights/=weights.mean()
        mean=np.average(train.rul_minutes,weights=weights); stdev=np.sqrt(np.average((train.rul_minutes-mean)**2,weights=weights))+1e-8
        max_target=max(max_target,abs(mean-ck['target_train_mean']),abs(stdev-ck['target_train_std']))
        scale=StandardScaler().fit(train[ck['input_columns']],sample_weight=weights)
        max_scaler=max(max_scaler,np.max(np.abs(scale.mean_-ck['scaler_mean'])),np.max(np.abs(scale.scale_-ck['scaler_scale'])))
        d=features[features.bearing==b].sort_values('record'); s=predictions[predictions.bearing==b].sort_values('record')
        assert np.array_equal(d.record,s.record) and np.allclose(d.rul_minutes,s.rul_minutes)
        model=BSplineKAN(len(ck['input_columns']),ck['width'],ck['intervals']);model.load_state_dict(ck['state_dict']);model.eval()
        x=scale.transform(d[ck['input_columns']]);
        with torch.no_grad():out=np.maximum(0,model(torch.tensor(x,dtype=torch.float32)).numpy()*stdev+mean)
        max_pred=max(max_pred,float(np.max(np.abs(out-s.predicted_rul_min))))
        mae=np.abs(out-d.rul_minutes.to_numpy()).mean(); rmse=np.sqrt(((out-d.rul_minutes.to_numpy())**2).mean())
        max_metric=max(max_metric,abs(mae-f['test']['mae_min']),abs(rmse-f['test']['rmse_min']))
    assert max_scaler < 1e-7 and max_target < 1e-7 and max_pred < 0.002 and max_metric < 0.0002
    report={'status':'passed','test_bearings':15,'seed':42,'predictions':9216,'protocol':'same outer and inner bearing splits as prior seed42; train-bearing equal-weight scaler and target transform; loaded model and predictions independently',
            'max_scaler_difference':float(max_scaler),'max_target_transform_difference':float(max_target),
            'max_checkpoint_prediction_difference_min':float(max_pred),'max_metric_difference_min':float(max_metric),
            'caveat':'Reconstruction and split audit do not prove physical truth, generalization to new operating conditions, or prevent exploratory test-use bias.'}
    a.output.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
