"""Baselines for fault-anchored Chebyshev KAN experiments.

All models consume the same envelope and metadata, and expose the same output
contract as ``core.model.FaultAnchoredChebyKAN``.  The spectral baselines do
not inspect bearing geometry; shaft speed sets a fixed order grid.  This is a
constant/mean-speed spectrum, not order tracking for nonstationary rotation.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from core.model import FAConfig, FaultAnchoredProjection, ChebyLayer


class SpectralOrders(nn.Module):
    """Fixed envelope-spectrum samples at orders 1,2,...,feature_count.

    Linear interpolation of FFT magnitude handles non-bin-centred frequencies.
    Values are normalised by segment energy exactly as the FA layer.  The
    procedure is identical in source and target domains; no target statistics
    or defect geometry enter these features. For variable speeds the grid uses
    mean speed, so only constant or near-constant speed should be interpreted.
    """
    def __init__(self, feature_count: int):
        super().__init__()
        if feature_count < 1:
            raise ValueError('feature_count must be positive')
        self.feature_count = feature_count

    def forward(self, envelope, fs, shaft_speed, geometry=None):
        if envelope.ndim != 2 or envelope.shape[1] < 2:
            raise ValueError('envelope must be shaped [batch, samples]')
        if not envelope.is_floating_point() or not torch.isfinite(envelope).all():
            raise ValueError('envelope must be finite floating point')
        b, n = envelope.shape
        fs = torch.as_tensor(fs, device=envelope.device, dtype=envelope.dtype).reshape(-1)
        if fs.numel() == 1:
            fs = fs.expand(b)
        if fs.numel() != b or not torch.isfinite(fs).all() or (fs <= 0).any():
            raise ValueError('fs must be positive scalar or batch vector')
        speed = torch.as_tensor(shaft_speed, device=envelope.device, dtype=envelope.dtype)
        if speed.ndim == 0:
            speed = speed.expand(b)
        elif speed.ndim == 1 and speed.numel() == b:
            pass
        elif speed.shape == (b, n):
            speed = speed.mean(dim=1)
        else:
            raise ValueError('shaft_speed must be scalar, [batch], or [batch,samples]')
        if not torch.isfinite(speed).all() or (speed <= 0).any():
            raise ValueError('shaft_speed must be positive and finite')
        frequencies = speed[:, None] * torch.arange(1, self.feature_count+1, device=envelope.device, dtype=envelope.dtype)
        if (frequencies >= fs[:, None]/2).any():
            raise ValueError('fixed order grid must lie below Nyquist')
        centred = envelope - envelope.mean(dim=1, keepdim=True)
        fft = torch.fft.rfft(centred, dim=1).abs() * (2.0 / n)
        index = frequencies * n / fs[:, None]
        lower = index.floor().long().clamp(min=0, max=fft.shape[1]-2)
        fraction = index - lower
        amplitude = fft.gather(1, lower) * (1-fraction) + fft.gather(1, lower+1) * fraction
        denominator = (2*centred.square().mean(dim=1)).clamp_min(1e-12).sqrt()
        return torch.log1p(amplitude / denominator[:, None])


class BenchmarkClassifier(nn.Module):
    """MLP or Chebyshev KAN with optional fault-anchored feature extractor.

    KAN input is always the geometry-free order grid. The anchored MLP shares
    the FA model's exact differentiable anchored representation and slip range.
    Auxiliary simulation heads and dimensions match the proposed FA model.
    """
    def __init__(self, architecture: str, config: FAConfig = FAConfig()):
        super().__init__()
        if architecture not in {'anchored_mlp', 'spectrum_mlp', 'spectrum_cheby'}:
            raise ValueError(f'Unknown architecture: {architecture}')
        self.architecture = architecture
        self.config = config
        self.feature_count = config.harmonics * (1 + 2*(1+2*config.sidebands))
        self.anchor = FaultAnchoredProjection(config) if architecture == 'anchored_mlp' else None
        self.spectrum = SpectralOrders(self.feature_count) if self.anchor is None else None
        if architecture == 'spectrum_cheby':
            self.shared = ChebyLayer(self.feature_count, config.hidden, config.degree)
            self.classifier = ChebyLayer(config.hidden, config.classes, config.degree)
        else:
            self.shared = nn.Linear(self.feature_count, config.hidden)
            self.classifier = nn.Linear(config.hidden, config.classes)
        self.amplitude_head = nn.Linear(config.hidden, 1)
        self.decay_head = nn.Linear(config.hidden, 1)

    def forward(self, envelope, fs, shaft_speed, geometry):
        if self.anchor is not None:
            features, responses, slip = self.anchor(envelope, fs, shaft_speed, geometry)
        else:
            features = self.spectrum(envelope, fs, shaft_speed, geometry)
            responses = None
            slip = None
        hidden = self.shared(features)
        if self.architecture != 'spectrum_cheby':
            hidden = F.relu(hidden)
        return {'logits':self.classifier(hidden),
                'amplitude':F.softplus(self.amplitude_head(hidden)).squeeze(-1),
                'decay':F.softplus(self.decay_head(hidden)).squeeze(-1),
                'features':features, 'relative_responses':responses, 'slip':slip}


def make_model(name, config):
    from core.model import FaultAnchoredChebyKAN
    if name == 'fa_cheby':
        return FaultAnchoredChebyKAN(config)
    return BenchmarkClassifier(name, config)
