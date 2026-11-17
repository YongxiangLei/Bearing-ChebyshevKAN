"""Signal-only subprocess check; deliberately does not import torch."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from core.signal import envelope_numpy

fs=4096
t=np.arange(fs)/fs
x=(1+.4*np.cos(2*np.pi*64*t))*np.cos(2*np.pi*1000*t)
e,band=envelope_numpy(x,fs,band_hz=(800,1200),return_band=True)
assert abs(e.mean()) < 1e-12
assert np.corrcoef(e[200:-200],np.cos(2*np.pi*64*t[200:-200]))[0,1] > .95
assert band == (800,1200)
auto,band=envelope_numpy(x,fs,return_band=True)
assert np.isfinite(auto).all()
assert 0 < band[0] < band[1] < fs/2
try:
    envelope_numpy(x,fs,(0,300))
    raise AssertionError('invalid passband should fail')
except ValueError:
    pass
