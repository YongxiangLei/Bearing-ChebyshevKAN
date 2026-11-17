"""Visualize basis, a training-selected edge, and one-input response.

The test fold and feature are fixed in advance. Hidden edge choice uses training
data only. Dependence sets other inputs to training means and may be unrealistic.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from bspline_layer import BSplineKAN

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--checkpoint',type=Path,default=Path(__file__).parent/'results_seed42'/'checkpoints'/'Bearing1_3_seed42.pt')
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'visuals')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    c=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    net=BSplineKAN(len(c['input_columns']),c['width'],c['intervals']);net.load_state_dict(c['state_dict']);net.eval()
    df=pd.read_csv(a.project/'rul_study'/'data'/'features_and_labels.csv')
    idx=c['input_columns'].index('horizontal_rms')
    train=df[df.bearing.isin(c['train_bearings'])]; held=df[df.bearing==c['test_bearing']]
    vals=train.horizontal_rms.to_numpy();mu=float(c['scaler_mean'][idx]);sd=float(c['scaler_scale'][idx])
    # Selection uses only training inputs. For each hidden node calculate sample
    # standard deviation of the full edge response; do not select by test result.
    standard=(vals-mu)/sd
    with torch.no_grad():
        bases=net.l1.basis(torch.tensor(standard,dtype=torch.float32)[:,None])[:,0,:]
        edge_values=bases@net.l1.coeffs[:,idx,:].T
        variability=edge_values.std(0).numpy()
    node=int(np.argmax(variability)); q01,q05,med,q95,q99=np.quantile(vals,[.01,.05,.5,.95,.99]); grid=np.linspace(q01,q99,400)
    standardized=(grid-mu)/sd
    with torch.no_grad():
        bt=net.l1.basis(torch.tensor(standardized,dtype=torch.float32)[:,None])[:,0,:]
        edge=(bt@net.l1.coeffs[node,idx,:]).numpy()
        examples=torch.zeros((len(grid),len(c['input_columns'])),dtype=torch.float32)
        examples[:,idx]=torch.tensor(standardized,dtype=torch.float32)
        pre=(net(examples).numpy()*float(c['target_train_std'])+float(c['target_train_mean']))
        pred=np.maximum(pre,0)
    z=np.linspace(-2.999999,2.999999,800)
    with torch.no_grad(): b=net.l1.basis_from_mapped(torch.tensor(z,dtype=torch.float64)[:,None]).numpy()[:,0,:]
    partition_err=float(np.max(np.abs(b.sum(1)-1)))
    # Model-wide local explanation on the independently held-out test bearing.
    raw=held[c['input_columns']].to_numpy(dtype=float); x=torch.tensor((raw-c['scaler_mean'])/c['scaler_scale'],dtype=torch.float32)
    grad=torch.zeros_like(x)
    for alpha in (torch.arange(128,dtype=torch.float32)+.5)/128:
        t=(x*alpha).detach().requires_grad_(True)
        (net(t)*float(c['target_train_std'])+float(c['target_train_mean'])).clamp_min(0).sum().backward();grad+=t.grad.detach()
    ig=x*grad/128
    with torch.no_grad():
        preclip=net(x)*float(c['target_train_std'])+float(c['target_train_mean'])
        delta=preclip.clamp_min(0)-(net(torch.zeros_like(x))*float(c['target_train_std'])+float(c['target_train_mean'])).clamp_min(0)
    completeness=float((ig.sum(1)-delta).abs().mean())
    imp=ig.abs().mean(0).numpy(); rank=np.argsort(-imp)
    np.savez_compressed(a.output/'plot_data.npz', z=z, basis=b, knots=net.l1.knots.numpy(), grid=grid, edge=edge, pred=pred,
                        node=node, q05=q05, q95=q95, median=med)
    names=c['input_columns'];out={
      'checkpoint':str(a.checkpoint),'test_bearing':c['test_bearing'],'test_n':len(held),'model':'cubic open-uniform B-spline, fixed knots, z=3*tanh(standardized_input/3)',
      'knots':net.l1.knots.tolist(),'basis_partition_max_error':partition_err,
      'fixed_plotted_feature':'horizontal_rms','hidden_node_index_zero_based':node,'edge_selected_by':'max standard deviation on training records among hidden edges of fixed feature',
      'train_record_quantiles_01_05_50_95_99':[float(q) for q in [q01,q05,med,q95,q99]],'feature_scaler_mean':mu,'feature_scaler_std':sd,
      'heldout_ig_target':'nonnegative-clipped predicted minutes; zero standardized input = train mean','ig_midpoint_steps':128,
      'fraction_test_predictions_clipped':float((preclip<0).float().mean()),
      'mean_absolute_ig_completeness_error_minutes':completeness,'mean_absolute_baseline_difference_minutes':float(delta.abs().mean()),
      'feature_mean_absolute_ig_minutes':{names[i]:float(imp[i]) for i in range(len(names))},'ranked_features':[names[i] for i in rank],
      'caveats':['Edge output is a hidden contribution, not RUL or physical law.', 'One-feature curve fixes all other input values at means; combinations may be unrealistic.', 'IG is on one held-out bearing, with training mean baseline and no causal interpretation.', 'Cubic spline is in z=3*tanh(standardized x/3); not raw x and tails saturate.', 'The training-record range is not an uncertainty band.']}
    (a.output/'interpretability.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
