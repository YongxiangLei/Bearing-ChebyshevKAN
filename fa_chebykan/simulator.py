"""Reproducible, randomized impulse/resonance simulator, not a validated rig model."""
import argparse
import numpy as np
from scipy.signal import lfilter
from data import LABELS, save_csv, import_cwru


def simulate(label, seed, fs=12000, duration=1.2):
    rng = np.random.default_rng(seed)
    speed = rng.uniform(26, 34)
    z = int(rng.integers(8, 11))
    D = rng.uniform(35, 45)
    d = D * rng.uniform(.17, .23)
    alpha = 0.0
    rho = d / D
    ftf = .5 * (1-rho)
    orders = {'outer':z/2*(1-rho), 'inner':z/2*(1+rho),
              'ball':D/(2*d)*(1-rho*rho)}
    n = int(round(fs * duration))
    excit = np.zeros(n, dtype=np.float64)
    amp = 0.0
    decay = rng.uniform(400, 850)
    freq = rng.uniform(2300, 3900)
    if label != 'normal':
        amp = rng.uniform(.6, 1.6)
        period = fs / (orders[label] * speed)
        times = np.arange(rng.uniform(0, period), n, period)
        times += rng.normal(0, .012 * period, len(times))
        ids = np.clip(np.rint(times).astype(int), 0, n-1)
        modulation = 1.0
        if label == 'inner':
            modulation = .65 + .35*np.cos(2*np.pi*speed*ids/fs + rng.uniform(0,2*np.pi))
        elif label == 'ball':
            modulation = .65 + .35*np.cos(2*np.pi*ftf*speed*ids/fs + rng.uniform(0,2*np.pi))
        np.add.at(excit, ids, amp * modulation * rng.uniform(.85, 1.15, len(ids)))
    # Discrete causal damped resonator. Same code for all classes.
    radius = np.exp(-decay/fs)
    signal = lfilter([1.0], [1.0, -2*radius*np.cos(2*np.pi*freq/fs), radius**2], excit)
    t = np.arange(n)/fs
    signal += rng.uniform(.02, .08)*np.sin(2*np.pi*speed*t + rng.uniform(0,2*np.pi))
    snr = rng.uniform(-6, 8)
    noise_scale = max(float(np.std(signal)), .08) / (10**(snr/20))
    signal += rng.normal(0, noise_scale, n)
    return dict(x=signal.astype(np.float32), x_unit='arbitrary', fs=fs, shaft_speed=speed, z=z,
                d_mm=d, D_mm=D, alpha_deg=alpha, label=label, A0=amp, decay=decay,
                snr_db=snr, fault_size_mm='', condition='synthetic random parameters',
                provenance=f'simulator.py seed={seed}; randomized resonance={freq:.3f} Hz')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('simulate')
    s.add_argument('--output',required=True)
    s.add_argument('--train-per-class',type=int,default=12)
    s.add_argument('--val-per-class',type=int,default=4)
    s.add_argument('--test-per-class',type=int,default=4)
    s.add_argument('--seed',type=int,default=42)
    s.add_argument('--fs',type=float,default=12000)
    s.add_argument('--duration',type=float,default=1.2)
    c=sub.add_parser('import-cwru')
    c.add_argument('--manifest',required=True)
    c.add_argument('--output',required=True)
    c.add_argument('--duration',type=float,required=True)
    c.add_argument('--overlap',type=float,default=0.)
    c.add_argument('--common-fs',type=float,default=None,help='Anti-alias resample whole recording before segmentation')
    a=p.parse_args()
    if a.command == 'import-cwru':
        records=import_cwru(a.manifest,a.duration,a.overlap,common_fs=a.common_fs)
    else:
        if min(a.train_per_class,a.val_per_class,a.test_per_class)<1:
            raise ValueError('Each simulated split needs positive counts per class')
        records=[]; i=0
        for split,count in [('train',a.train_per_class),('validation',a.val_per_class),('test',a.test_per_class)]:
            for label in LABELS:
                for _ in range(count):
                    sample=simulate(label,a.seed+i,a.fs,a.duration)
                    sample.update(dataset='synthetic', split=split, recording_id=f'sim-{a.seed+i}')
                    records.append(sample); i+=1
    save_csv(a.output,records)
    print(f'Saved {len(records)} records to {a.output}')


if __name__ == '__main__':
    main()
