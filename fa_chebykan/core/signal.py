"""Common raw-signal preprocessing for synthetic and real recordings."""
import numpy as np
from scipy import signal
from scipy.stats import kurtosis


def envelope_numpy(x, fs, band_hz=None, return_band=False):
    """Bandpass, Hilbert envelope, remove mean; one 1-D recording at a time.

    If band_hz is None, search eight overlapping bands and select maximum
    envelope kurtosis, using only the current signal. This is a transparent
    heuristic, NOT the Fast Kurtogram. Lock its policy before evaluation and
    never select using labels, simulation parameters or test performance.
    """
    x=np.asarray(x,dtype=float)
    if x.ndim != 1 or len(x)<128 or not np.isfinite(x).all() or not np.isfinite(fs) or fs<=0:
        raise ValueError('x must be a finite 1-D signal of at least 128 samples and fs positive')
    def get_envelope(band):
        lo,hi=map(float,band)
        if not (0 < lo < hi < fs/2):
            raise ValueError('band must satisfy 0 < low < high < Nyquist')
        sos=signal.butter(4,[lo,hi],fs=fs,btype='bandpass',output='sos')
        filtered=signal.sosfiltfilt(sos,x)
        e=np.abs(signal.hilbert(filtered))
        return e-e.mean()
    if band_hz is None:
        # [0.05, 0.95] times Nyquist, with 50% overlap.
        nyq=fs/2
        bands=[((.05+.1*i)*nyq,(.25+.1*i)*nyq) for i in range(8)]
        scored=[]
        for band in bands:
            candidate=get_envelope(band)
            score=float(kurtosis(candidate,fisher=True,bias=False))
            scored.append(score if np.isfinite(score) else -np.inf)
        band_hz=bands[int(np.argmax(scored))]
    envelope=get_envelope(band_hz)
    return (envelope,tuple(band_hz)) if return_band else envelope
