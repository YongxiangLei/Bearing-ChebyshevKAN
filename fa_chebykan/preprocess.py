"""Create a signal-envelope cache in a process that does not import PyTorch."""
import argparse
import hashlib
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from data import load_csv
from core.signal import envelope_numpy

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',required=True)
p.add_argument('--output',required=True)
p.add_argument('--band',type=float,nargs=2,default=[1600,4800],metavar=('LOW','HIGH'))
p.add_argument('--envelope-fs',type=float,default=None,
               help='Optional envelope output sampling rate in Hz; applies anti-alias polyphase filtering')
a=p.parse_args()
src=Path(a.input); records=load_csv(src)
data={'dataset_sha256':np.array(hashlib.sha256(src.read_bytes()).hexdigest()),
      'band_hz':np.asarray(a.band), 'count':np.array(len(records))}
for i,r in enumerate(records):
    env=envelope_numpy(r['x'],r['fs'],band_hz=tuple(a.band))
    fs_out=float(r['fs'])
    speed=np.asarray(r['shaft_speed'],dtype=float)
    if a.envelope_fs is not None:
        if not 0 < a.envelope_fs <= fs_out:
            raise ValueError('envelope-fs must be positive and no greater than input fs')
        ratio=Fraction(float(a.envelope_fs)/fs_out).limit_denominator(10000)
        effective=fs_out*ratio.numerator/ratio.denominator
        if abs(effective-a.envelope_fs)>1e-6:
            raise ValueError('requested output rate cannot be represented accurately')
        env=resample_poly(env,ratio.numerator,ratio.denominator)
        if speed.ndim:
            # Piecewise-linear speed interpolation at the output timestamps;
            # unlike vibration, the slowly varying speed is a phase covariate.
            speed=np.interp(np.arange(len(env))/effective,np.arange(len(speed))/fs_out,speed)
        fs_out=effective
        rho=(r['d_mm']/r['D_mm'])*np.cos(np.deg2rad(r['alpha_deg']))
        cage=.5*(1-rho)
        revolutions=(float(speed)*len(env) if speed.ndim==0 else float(speed.sum()))/fs_out
        if revolutions*cage < 10:
            raise ValueError(f"{r['recording_id']}: resampled segment shorter than ten cage revolutions")
    data[f'env_{i}']=env.astype(np.float32)
    data[f'fs_{i}']=np.array(fs_out)
    data[f'speed_{i}']=speed.astype(np.float32)
dst=Path(a.output);dst.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(dst,**data)
print(f'Saved {len(records)} envelopes to {dst} (band {a.band} Hz, envelope fs {a.envelope_fs or "native"})')
