"""Convert independently split Paderborn bearing MAT records for FA-ChebyKAN.

The input manifest contains one row per MAT recording and an explicit bearing
split and fault label.  A shared geometry is passed explicitly as a nominal
approximation; it is not silently inferred from a bearing identifier.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly


def _channel(document, name):
    matches = [c for c in document['Y'] if c.get('Name') == name]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one channel {name}; got {len(matches)}')
    item = matches[0]
    values = np.asarray(item['Data'], dtype=float).reshape(-1)
    index = int(item['XIndex']) - 1
    time = np.asarray(document['X'][index]['Data'], dtype=float).reshape(-1)
    if len(time) != len(values) or len(time) < 2 or not np.isfinite(values).all():
        raise ValueError(f'Invalid data or timebase for {name}')
    if not np.isfinite(time).all() or (np.diff(time) <= 0).any():
        raise ValueError(f'Non-monotonic or invalid timebase for {name}')
    fs = (len(time)-1) / (time[-1]-time[0])
    return values, time, fs


def load_manifest(path, data_dir):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    required = {'bearing_id', 'label', 'split', 'condition', 'recording_id',
                'relative_path', 'sha256', 'label_source'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'Manifest must include {sorted(required)}')
    seen_bearing, seen_path = {}, set()
    for row in rows:
        row.update(bearing=row['bearing_id'], recording=row['recording_id'],
                   path=row['relative_path'], provenance=row['label_source'])
        b = row['bearing'].strip()
        label, split = row['label'].strip(), row['split'].strip()
        if not b or label not in {'normal', 'inner', 'outer'} or split not in {'train', 'validation', 'test'}:
            raise ValueError(f'Invalid row: {row}')
        key = (label, split)
        if b in seen_bearing and seen_bearing[b] != key:
            raise ValueError(f'Bearing {b} changes label or crosses splits')
        seen_bearing[b] = key
        path = (data_dir / row['path']).resolve()
        if path in seen_path:
            raise ValueError(f'Duplicate MAT record: {path.name}')
        if not path.is_file() or not row['provenance'].strip():
            raise ValueError(f'Missing MAT or provenance: {path}')
        seen_path.add(path)
        row['_path'] = path
    return rows


def convert(rows, geometry, length_s=1.2, common_fs=12000):
    z, d_mm, D_mm, alpha = geometry
    if (z != int(z) or z < 2 or d_mm <= 0 or D_mm <= d_mm or not 0 <= alpha < 90
            or length_s <= 0 or common_fs != 12000):
        raise ValueError('Invalid nominal geometry, duration, or supported common sample rate')
    samples = []; audit = []
    for row in rows:
        p = row['_path']
        mat = loadmat(p, simplify_cells=True)
        if p.stem not in mat:
            raise KeyError(f'MAT variable for {p.name} missing')
        doc = mat[p.stem]
        vib, vib_time, fs_vib = _channel(doc, 'vibration_1')
        speed, speed_time, fs_speed = _channel(doc, 'speed')
        raw_sha256 = hashlib.sha256(p.read_bytes()).hexdigest()
        if raw_sha256 != row['sha256']:
            raise ValueError(f'{p.name}: source hash does not match manifest')
        vib_dt = np.diff(vib_time)
        # Prespecified metadata-only screening: up to 1% deviation from the
        # nominal 64k record point density is admitted. Larger deviations are
        # excluded, independently of class, before model training/testing.
        # Missing gaps larger than 1/20 kHz would not safely support the
        # intended 0–6 kHz signal band and are rejected as well.
        exclusion = None
        if abs(fs_vib/64000 - 1) > .01:
            exclusion = 'vibration timebase mean sample density differs from nominal 64 kHz by >1%'
        elif vib_dt.max() > 1/20000:
            exclusion = 'vibration timebase has a gap exceeding 50 microseconds'
        elif abs(fs_speed/4000 - 1) > .01:
            exclusion = 'speed timebase mean density differs from nominal 4 kHz by >1%'
        summary = {'bearing':row['bearing'], 'split':row['split'], 'label':row['label'],
            'recording':row['recording'], 'file':p.name, 'sha256':raw_sha256,
            'vibration_samples':len(vib), 'vibration_fs_from_time':fs_vib,
            'vibration_dt_min_sec':float(vib_dt.min()), 'vibration_dt_max_sec':float(vib_dt.max()),
            'speed_samples':len(speed), 'speed_fs_from_time':fs_speed,
            'speed_min_rpm':float(speed.min()), 'speed_max_rpm':float(speed.max()),
            'x_unit':'unknown'}
        if exclusion:
            audit.append({**summary, 'included':False, 'exclusion_reason':exclusion, 'segments':0})
            continue
        # Use actual acquisition times for interpolation to a uniform 64 kHz
        # grid before the same antialias/resampling used for every recording.
        duration = min(4., float(vib_time[-1] - vib_time[0]))
        grid = vib_time[0] + np.arange(int(np.floor(duration*64000)))/64000
        grid = grid[grid <= vib_time[-1]]
        uniform = np.interp(grid, vib_time, vib)
        raw = resample_poly(uniform, 3, 16)
        n = int(round(length_s * common_fs))
        if len(raw) < n:
            raise ValueError(f'{p.name}: duration shorter than segment')
        count = 0
        for start in range(0, len(raw)-n+1, n):
            t_start = vib_time[0] + start/common_fs
            t_end = vib_time[0] + (start+n)/common_fs
            mask = (speed_time >= t_start) & (speed_time < t_end)
            if mask.sum() < int(0.95 * length_s * 4000):
                raise ValueError(f'{p.name}: speed timeline does not cover segment')
            rpm = float(speed[mask].mean())
            if not np.isfinite(rpm) or not 500 <= rpm <= 3000:
                raise ValueError(f'{p.name}: implausible measured speed mean {rpm}')
            samples.append({
                # Existing FA-ChebyKAN engine groups by recording_id.  We use
                # bearing ID here deliberately to weight and score bearings,
                # not to claim that both MAT files are one recording.
                'recording_id': row['bearing'], 'dataset': 'Paderborn',
                'split': row['split'], 'x': raw[start:start+n].astype(np.float32),
                'x_unit':'unknown', 'fs':common_fs, 'shaft_speed':rpm/60,
                'z':z, 'd_mm':d_mm, 'D_mm':D_mm, 'alpha_deg':alpha,
                'label':row['label'], 'condition':row['condition'],
                'fault_size_mm':'', 'A0':'', 'decay':'', 'snr_db':'',
                'provenance':(f'{row["provenance"]}; MAT={p.name}; sha256={raw_sha256}; '
                    f'measured speed mean={rpm:.6f} rpm; time interpolation to 64k then resample_poly 3/16; '
                    f'12k output samples [{start},{start+n}); geometry=nominal 6203')})
            count += 1
        audit.append({**summary, 'included':True, 'exclusion_reason':'', 'segments':count})
    accepted = {a['bearing'] for a in audit if a['included']}
    missing = set(row['bearing'] for row in rows) - accepted
    if missing:
        raise ValueError(f'All recordings excluded for bearings: {sorted(missing)}')
    return samples, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--fa-dir', type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--geometry', nargs=4, type=float, metavar=('Z', 'D_BALL_MM', 'D_PITCH_MM', 'ALPHA_DEG'), required=True)
    p.add_argument('--duration', type=float, default=1.2)
    a = p.parse_args()
    sys.path.insert(0, str(a.fa_dir.resolve()))
    from data import save_csv
    rows = load_manifest(a.manifest, a.data_dir)
    samples, audit = convert(rows, a.geometry, a.duration)
    save_csv(a.output, samples)
    a.audit.parent.mkdir(parents=True, exist_ok=True)
    a.audit.write_text(json.dumps({'geometry_nominal':a.geometry, 'segment_seconds':a.duration,
        'screening':'before training, require monotonic timestamps, 64k mean density within 1%, max vibration gap <=50 us, speed 4k mean density within 1%; exclude failures',
        'resampling':'interpolate actual X timestamps to uniform 64000 Hz, then anti-alias whole signal 64000 -> 12000 Hz via polyphase 3/16 before segmentation',
        'grouping':'recording_id intentionally denotes bearing ID; MAT recording appears in provenance',
        'speed':'each segment uses corresponding 4 kHz MAT speed mean / 60 Hz; time bases aligned',
        'records':audit}, indent=2), encoding='utf8')
    print(f'Saved {len(samples)} segments from {len(audit)} MAT records to {a.output}')


if __name__ == '__main__':
    main()
