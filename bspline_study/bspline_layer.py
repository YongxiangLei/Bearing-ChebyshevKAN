"""Cubic open-uniform B-spline KAN layer; no Chebyshev basis.

Each edge phi(x)=sum_j c_j B_{j,3}(3*tanh(x/3)); inner hidden layer
outputs are summed over the input features. tanh limits extrapolation.
"""
import math
import torch
from torch import nn


class BSplineLayer(nn.Module):
    def __init__(self, in_features, out_features, intervals=6, degree=3):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.intervals, self.degree = intervals, degree
        if degree != 3: raise ValueError('This implementation assumes degree 3')
        interior = torch.linspace(-3., 3., intervals+1)[1:-1]
        knots = torch.cat((torch.full((degree+1,),-3.), interior, torch.full((degree+1,),3.)))
        self.register_buffer('knots', knots)
        self.n_basis = len(knots)-degree-1
        self.coeffs = nn.Parameter(torch.randn(out_features, in_features, self.n_basis) * (0.4 / math.sqrt(in_features)))

    def basis_from_mapped(self, z):
        t = self.knots
        z = z.clamp(min=float(t[0])+1e-6, max=float(t[-1])-1e-6)
        zz = z.unsqueeze(-1)
        b = ((zz >= t[:-1]) & (zz < t[1:])).to(z.dtype)
        for degree in range(1, self.degree+1):
            count = len(t)-degree-1
            d1 = t[degree:degree+count]-t[:count]
            d2 = t[degree+1:degree+1+count]-t[1:count+1]
            r1 = torch.where(d1 > 0, (zz-t[:count])/d1.clamp_min(1e-12), torch.zeros_like(zz+t[:count]))
            r2 = torch.where(d2 > 0, (t[degree+1:degree+1+count]-zz)/d2.clamp_min(1e-12), torch.zeros_like(zz+t[:count]))
            b = r1*b[...,:count] + r2*b[...,1:count+1]
        return b

    def basis(self,x):
        return self.basis_from_mapped(3*torch.tanh(x/3))

    def forward(self,x):
        return torch.einsum('bik,oik->bo',self.basis(x),self.coeffs)

class BSplineKAN(nn.Module):
    def __init__(self,dim,width,intervals=6):
        super().__init__()
        self.l1=BSplineLayer(dim,width,intervals)
        self.l2=BSplineLayer(width,1,intervals)
    def forward(self,x):
        return self.l2(self.l1(x)).squeeze(-1)
