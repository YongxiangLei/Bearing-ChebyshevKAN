"""Fault-frequency-anchored envelope projections and Chebyshev KAN.

Projection amplitudes are descriptive responses, not an orthogonal energy
decomposition. Finite windows, slip, variable speed and sidebands can make
atoms correlated. The m=0 term is included once, not twice.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FAConfig:
    harmonics: int = 3
    sidebands: int = 2
    degree: int = 3
    hidden: int = 32
    classes: int = 4
    max_slip: float = 0.02

    def __post_init__(self):
        if self.harmonics < 1 or self.sidebands < 0 or self.degree < 1:
            raise ValueError('harmonics/degree must be positive and sidebands nonnegative')
        if self.classes < 2 or self.hidden < 1 or not 0 <= self.max_slip < 1:
            raise ValueError('invalid class count, hidden width, or maximum slip')


def fault_orders(geometry):
    """Return [...,4] OR, IR, ball and cage orders from Z,d,D,alpha(deg)."""
    g = torch.as_tensor(geometry)
    if not g.is_floating_point():
        g = g.float()
    if g.shape[-1] != 4:
        raise ValueError('geometry must end in [Z,d,D,alpha_deg]')
    z, d, pitch, alpha = g.unbind(-1)
    if bool(((z < 2) | ((z-z.round()).abs() > 1e-6) | (d <= 0) | (pitch <= d) | (alpha < 0) | (alpha >= 90)).any()):
        raise ValueError('invalid bearing geometry')
    rho = d / pitch * torch.cos(torch.deg2rad(alpha))
    return torch.stack((z/2*(1-rho), z/2*(1+rho), pitch/(2*d)*(1-rho*rho), (1-rho)/2), -1)


def cheby_trig(theta, order):
    """T_n(cos(theta)), sin(theta) U_(n-1)(cos(theta)) by recurrence."""
    x = torch.cos(theta)
    sin = torch.sin(theta)
    c = [torch.ones_like(theta), x]
    u = [torch.ones_like(theta)]
    if order > 1:
        u.append(2*x)
    for _ in range(2, order + 1):
        c.append(2*x*c[-1]-c[-2])
    for _ in range(2, order):
        u.append(2*x*u[-1]-u[-2])
    s = [torch.zeros_like(theta)] + [sin*v for v in u]
    return c, s


class FaultAnchoredProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.eta = nn.Parameter(torch.zeros(3))
        self.feature_names=[]
        for name in ('OR','IR','B'):
            M=0 if name=='OR' else config.sidebands
            for h in range(1,config.harmonics+1):
                for m in range(M+1):
                    for sign in (('',) if m==0 else ('+','-')):
                        self.feature_names.append(f'{name}:h{h}:m{m}{sign}')
        self.feature_count=len(self.feature_names)

    def forward(self, envelope, fs, shaft_speed, geometry):
        if envelope.ndim != 2 or envelope.shape[1] < 2:
            raise ValueError('envelope must have shape [batch, samples]')
        if not envelope.is_floating_point() or not bool(torch.isfinite(envelope).all()):
            raise ValueError('envelope must contain finite floating-point values')
        b, n_samples = envelope.shape
        dev, dtype = envelope.device, envelope.dtype
        fs = torch.as_tensor(fs, device=dev, dtype=dtype).reshape(-1)
        if fs.numel() == 1: fs = fs.expand(b)
        if fs.numel() != b or not bool(torch.isfinite(fs).all()) or bool((fs <= 0).any()):
            raise ValueError('fs must be positive, scalar or [batch]')
        speed = torch.as_tensor(shaft_speed, device=dev, dtype=dtype)
        if speed.ndim == 0: speed = speed.expand(b, n_samples)
        elif speed.ndim == 1 and speed.numel() == b: speed = speed[:, None].expand(-1,n_samples)
        elif speed.shape != (b,n_samples): raise ValueError('shaft_speed must be scalar, [batch] or [batch,samples]')
        if not bool(torch.isfinite(speed).all()) or bool((speed <= 0).any()):
            raise ValueError('shaft speed must be finite and positive (Hz)')
        geometry = torch.as_tensor(geometry, device=dev, dtype=dtype)
        if geometry.shape == (4,): geometry = geometry[None,:].expand(b,-1)
        if geometry.shape != (b,4) or not bool(torch.isfinite(geometry).all()):
            raise ValueError('geometry must have shape [4] or [batch,4]')
        orders = fault_orders(geometry)
        # Keep every possible searched anchor below Nyquist. This is a
        # conservative bound for time-varying speed and the permitted slip.
        upper = (self.config.harmonics * orders[:,:3].amax(dim=1) * (1+self.config.max_slip)
                 + self.config.sidebands * torch.maximum(torch.ones_like(orders[:,3]), orders[:,3]))
        if bool((upper*speed.amax(dim=1) >= fs/2).any()):
            raise ValueError('configured fault/sideband anchors reach or exceed Nyquist')
        phase = 2*math.pi*(torch.cumsum(speed/fs[:,None],dim=1)-speed[:,:1]/fs[:,None])
        slip = self.config.max_slip*torch.tanh(self.eta)
        # The envelope is expected to have its segment mean removed, but also
        # center here to guard callers and keep amplitude normalisation valid.
        centered = envelope - envelope.mean(dim=1,keepdim=True)
        denom = torch.sqrt((2*centered.square().mean(dim=1)).clamp_min(1e-12))
        amplitudes=[]
        by_class=[]
        for c in range(3):
            theta = orders[:,c,None]*(1+slip[c])*phase
            C,S = cheby_trig(theta,self.config.harmonics)
            modphase = phase * (orders[:,3,None] if c==2 else 1.0)
            M = 0 if c==0 else self.config.sidebands
            Cm,Sm = cheby_trig(modphase,max(M,1))
            values=[]
            for h in range(1,self.config.harmonics+1):
                for m in range(M+1):
                    for sign in ((1,) if m==0 else (1,-1)):
                        cosine=C[h]*Cm[m]-sign*S[h]*Sm[m]
                        sine=S[h]*Cm[m]+sign*C[h]*Sm[m]
                        a=2*(centered*cosine).mean(dim=1)
                        q=2*(centered*sine).mean(dim=1)
                        # Smooth near the origin while preserving exact zero.
                        amp=(torch.sqrt(a*a+q*q+1e-20)-1e-10).clamp_min(0)/denom
                        values.append(amp)
            group=torch.stack(values,dim=1)
            amplitudes.append(group)
            by_class.append(group.square().sum(dim=1))
        raw=torch.cat(amplitudes,dim=1)
        return torch.log1p(raw), torch.stack(by_class,dim=1), slip


class ChebyLayer(nn.Module):
    def __init__(self, in_features, out_features, degree):
        super().__init__()
        self.degree=degree
        self.weight=nn.Parameter(torch.empty(in_features,out_features,degree+1))
        nn.init.normal_(self.weight, std=1/math.sqrt(in_features*(degree+1)))
        self.bias=nn.Parameter(torch.zeros(out_features))

    def forward(self,x):
        x=torch.tanh(x)
        terms=[torch.ones_like(x),x]
        for _ in range(2,self.degree+1): terms.append(2*x*terms[-1]-terms[-2])
        return torch.einsum('bik,iok->bo',torch.stack(terms,dim=-1),self.weight)+self.bias


class FaultAnchoredChebyKAN(nn.Module):
    """Output logits and simulated-only targets from one shared model."""
    def __init__(self, config=FAConfig()):
        super().__init__()
        self.config=config
        self.anchor=FaultAnchoredProjection(config)
        self.shared=ChebyLayer(self.anchor.feature_count,config.hidden,config.degree)
        self.classifier=ChebyLayer(config.hidden,config.classes,config.degree)
        self.amplitude_head=nn.Linear(config.hidden,1)
        self.decay_head=nn.Linear(config.hidden,1)

    def forward(self,envelope,fs,shaft_speed,geometry):
        features,response,slip=self.anchor(envelope,fs,shaft_speed,geometry)
        hidden=self.shared(features)
        return {'logits':self.classifier(hidden),
                'amplitude':F.softplus(self.amplitude_head(hidden)).squeeze(-1),
                'decay':F.softplus(self.decay_head(hidden)).squeeze(-1),
                'features':features,'slip':slip,'relative_responses':response}
