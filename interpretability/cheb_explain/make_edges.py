"""Extract example first-layer Chebyshev edges from an existing CV model."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--output',type=Path,default=Path(__file__).parent)
    a=ap.parse_args();root=a.project.resolve();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    sys.path.insert(0,str(root/'rul_study'));from run_experiments import net_class
    KAN=net_class(root)['kan'];study=root/'rul_study';data=pd.read_csv(study/'data'/'features_and_labels.csv')
    ckpt=torch.load(study/'results_merged'/'checkpoints'/'Bearing1_1_seed42.pt',map_location='cpu',weights_only=False)
    train=data[data.bearing.isin(ckpt['train_bearings'])]; model=KAN(len(ckpt['input_columns']),ckpt['models']['kan']['width'])
    model.load_state_dict(ckpt['models']['kan']['state_dict']);model.eval()
    selections=[];series=[]
    for feature in ['horizontal_skewness','vertical_rms']:
        idx=ckpt['input_columns'].index(feature);mean=float(ckpt['scaler_mean'][idx]);sd=float(ckpt['scaler_scale'][idx])
        raw=train[feature].to_numpy();lo,hi=np.quantile(raw,[.05,.95]);z=(raw-mean)/sd
        weights=(model.l1.coeffs*torch.sigmoid(model.l1.order_weights)).detach().numpy()[:,idx,:]
        def bases(z):
            t=np.tanh(z);return np.column_stack([np.ones_like(t),t,2*t*t-1,4*t*t*t-3*t])
        values=bases(z)@weights.T; hidden=int(np.argmax(values.std(axis=0)))
        x=np.linspace(lo,hi,400);zs=(x-mean)/sd;function=bases(zs)@weights[hidden];baseline=float((bases(np.array([0.]))@weights[hidden])[0])
        # A first-layer output difference with all other inputs fixed at zero is
        # exactly this centered edge. Verify against the real torch forward pass.
        isolate=torch.zeros((len(x),len(ckpt['input_columns'])),dtype=torch.float32);isolate[:,idx]=torch.from_numpy(zs.astype(np.float32))
        with torch.no_grad():
            origin=torch.zeros((1,len(ckpt['input_columns'])),dtype=torch.float32)
            actual=(model.l1(isolate)[:,hidden]-model.l1(origin)[0,hidden]).numpy()
        forward_error=float(np.max(np.abs(actual-(function-baseline))))
        if forward_error>1e-5:raise ValueError(f'Edge reconstruction failed: {feature}, max error {forward_error}')
        selections.append({'feature':feature,'hidden_index':hidden,'training_range_5_95':[float(lo),float(hi)],'training_mean':mean,'train_scale':sd,
                           'selection':'maximum edge output standard deviation over training records, not test', 'test_bearing':'Bearing1_1','seed':42,
                           'max_forward_error':forward_error,'train_bearings':ckpt['train_bearings'],'validation_bearings':ckpt['validation_bearings']})
        series.extend({'feature':feature,'x':float(xx),'centered_edge_output':float(yy)} for xx,yy in zip(x,function-baseline))
    (a.output/'edge_selections.json').write_text(json.dumps(selections,indent=2,ensure_ascii=False),encoding='utf8')
    pd.DataFrame(series).to_csv(a.output/'edge_curves.csv',index=False)
    print(json.dumps(selections,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
