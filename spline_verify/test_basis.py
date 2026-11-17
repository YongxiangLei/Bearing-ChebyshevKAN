"""Independent numerical cross-check of cubic open-uniform spline basis."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import BSpline
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'bspline_study'))
from bspline_layer import BSplineLayer

layer=BSplineLayer(1,1,intervals=6,degree=3).double()
knots=layer.knots.numpy()
vals=np.r_[np.linspace(-2.999,2.999,1201),-2,-1,0,1,2]
actual=layer.basis_from_mapped(torch.tensor(vals[:,None],dtype=torch.float64))[:,0,:].detach().numpy()
ref=BSpline.design_matrix(vals,knots,3,extrapolate=False).toarray()
partition=np.sum(actual,axis=1)
xs=np.array([-1e10,-20.,-4.,-3.,0.,3.,4.,20.,1e10])
bb=layer.basis(torch.tensor(xs[:,None],dtype=torch.float64))[:,0,:].detach().numpy()
x=torch.tensor([[-2.25],[-1.53],[0.001],[1.37],[2.45]],dtype=torch.float64,requires_grad=True)
passed=torch.autograd.gradcheck(layer.basis,(x,),eps=1e-6,atol=2e-4,rtol=2e-3)
metrics={'knots':knots.tolist(),'samples_checked':len(vals),'max_abs_difference_vs_scipy':float(np.max(np.abs(actual-ref))),
         'max_abs_partition_error':float(np.max(np.abs(partition-1))),
         'min_basis':float(actual.min()),'gradient_check_passed':passed,
         'boundary_input_x':xs.tolist(),'boundary_basis_sum':bb.sum(axis=1).tolist(),
         'boundary_notes':'z=3*tanh(x/3), then clamp z to [-3+1e-6,3-1e-6]. For extreme x the derivative saturates. BSplines are cubic in z, composed with tanh, not piecewise cubic in unbounded input x.'}
assert metrics['max_abs_difference_vs_scipy']<1e-12
assert metrics['max_abs_partition_error']<1e-12
assert passed
(Path(__file__).parent/'basis_verification.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
print(json.dumps(metrics,indent=2))
