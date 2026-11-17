"""Recording-level dataset interface for the FA-ChebyKAN prototype.

Dataset files are CSVs containing one JSON-encoded vibration segment per row.
Real metadata must come from an auditable manifest; nothing is inferred from labels.
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
import numpy as np

LABELS = ('normal', 'outer', 'inner', 'ball')
csv.field_size_limit(min(sys.maxsize, 2**31-1))
FIELDS = ('recording_id', 'dataset', 'split', 'x', 'x_unit', 'fs', 'shaft_speed', 'z',
          'd_mm', 'D_mm', 'alpha_deg', 'label', 'condition', 'fault_size_mm',
          'A0', 'decay', 'snr_db', 'provenance')


def validate_record(r):
    for field in ('recording_id', 'dataset', 'split', 'label'):
        if not str(r[field]).strip():
            raise ValueError(f'Missing {field}')
    if r['label'] not in LABELS:
        raise ValueError(f"Unknown label {r['label']!r}; expected {LABELS}")
    if r['x_unit'] not in ('g', 'm/s^2', 'arbitrary', 'unknown'):
        raise ValueError('x_unit must be g, m/s^2, unknown, or arbitrary (simulation only)')
    if r['x_unit']=='arbitrary' and not r['dataset'].startswith('synthetic'):
        raise ValueError('Real vibration input requires explicit g or m/s^2 units')
    if r['split'] not in ('train', 'validation', 'test'):
        raise ValueError('split must be train, validation or test')
    x = np.asarray(r['x'], dtype=float)
    if x.ndim != 1 or len(x) < 8 or not np.isfinite(x).all():
        raise ValueError('x must be a finite nonempty 1-D signal')
    fs = float(r['fs'])
    speed = np.asarray(r['shaft_speed'], dtype=float)
    if speed.ndim > 1 or (speed.ndim == 1 and speed.shape != x.shape):
        raise ValueError('shaft_speed must be scalar or one value per signal sample')
    if not np.isfinite(speed).all() or (speed <= 0).any():
        raise ValueError('shaft_speed must be finite and positive')
    z, d, D, angle = [float(r[k]) for k in ('z', 'd_mm', 'D_mm', 'alpha_deg')]
    if not np.isfinite([fs,z,d,D,angle]).all() or fs <= 0 or z < 2 or z != int(z) or d <= 0 or D <= d or not 0 <= angle < 90:
        raise ValueError('Invalid sampling rate, speed, or bearing geometry')
    cage_order = .5 * (1 - (d / D) * np.cos(np.deg2rad(angle)))
    revolutions = (float(speed) * len(x) if speed.ndim == 0 else float(speed.sum())) / fs
    if revolutions * cage_order < 10:
        raise ValueError(f"{r['recording_id']}: segment shorter than 10 cage revolutions")
    return r


def validate_splits(records):
    """Disallow a recording (including all its segments) from crossing splits."""
    seen = {}; labels = {}
    for r in records:
        validate_record(r)
        key = (r['dataset'], r['recording_id'])
        if key in seen and seen[key] != r['split']:
            raise ValueError(f'Recording {key} crosses {seen[key]} and {r["split"]}')
        if key in labels and labels[key] != r['label']:
            raise ValueError(f'Recording {key} has inconsistent labels')
        seen[key] = r['split']
        labels[key] = r['label']
    return records


def save_csv(path, records):
    validate_splits(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        for r in records:
            row = dict(r)
            row['x'] = json.dumps(np.asarray(r['x'], dtype=float).tolist(), separators=(',', ':'))
            row['shaft_speed'] = json.dumps(np.asarray(r['shaft_speed'], dtype=float).tolist(), separators=(',', ':'))
            w.writerow(row)


def load_csv(path):
    records = []
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            missing = set(FIELDS) - set(row)
            if missing:
                raise ValueError(f'Missing CSV columns: {sorted(missing)}')
            row['x'] = np.asarray(json.loads(row['x']), dtype=np.float32)
            for k in ('fs', 'z', 'd_mm', 'D_mm', 'alpha_deg'):
                row[k] = float(row[k])
            speed = np.asarray(json.loads(row['shaft_speed']), dtype=float)
            row['shaft_speed'] = float(speed) if speed.ndim == 0 else speed
            for k in ('A0', 'decay', 'snr_db', 'fault_size_mm'):
                row[k] = float(row[k]) if row[k] not in ('', 'None') else None
            records.append(row)
    return validate_splits(records)


def import_cwru(manifest_path, length_seconds, overlap=0.0, common_fs=None):
    """Load explicitly documented MAT records. Split and metadata are manifest inputs.

    Required columns: path, signal_key, rpm_key, x_unit, fs, z, d_mm, D_mm, alpha_deg,
    label, split, condition, provenance. RPM is read from each MAT when available.
    For records without an RPM variable, the optional manifest field nominal_rpm
    must be explicitly specified and its source documented in provenance.
    The bearing geometry and labels are not populated by this function.
    """
    from scipy.io import loadmat
    manifest_path = Path(manifest_path)
    with manifest_path.open(encoding='utf-8-sig', newline='') as source:
        rows = list(csv.DictReader(source))
    required = {'path', 'signal_key', 'rpm_key', 'x_unit', 'fs', 'z', 'd_mm', 'D_mm',
                'alpha_deg', 'label', 'split', 'condition', 'provenance'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'Manifest must contain {sorted(required)}')
    if length_seconds <= 0 or not 0 <= overlap < 1:
        raise ValueError('Invalid segment length or overlap')
    all_records = []
    paths = {}; recording_names = set()
    for row in rows:
        if not row['provenance'].strip():
            raise ValueError('Each row needs a metadata provenance reference')
        path = Path(row['path'])
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        if path in paths:
            raise ValueError(f'Duplicate source recording in manifest: {path}')
        paths[path] = row['split']
        if path.name in recording_names:
            raise ValueError(f'Duplicate CWRU filename: {path.name}')
        recording_names.add(path.name)
        m = loadmat(path)
        if row['signal_key'] not in m:
            raise KeyError(f'{path}: missing signal key {row["signal_key"]}')
        rpm_key = row['rpm_key'].strip()
        if rpm_key and rpm_key in m:
            rpm = np.asarray(m[rpm_key], dtype=float).reshape(-1)
            rpm_source = f'MAT variable {rpm_key}'
        else:
            nominal_rpm = row.get('nominal_rpm', '').strip() or row.get('rpm', '').strip()
            if not nominal_rpm:
                raise ValueError(f'{path}: RPM variable unavailable; supply sourced nominal_rpm')
            rpm = np.asarray([float(nominal_rpm)], dtype=float)
            rpm_source = f'explicit manifest nominal_rpm={nominal_rpm}; {row.get("rpm_source", "")}; {row["provenance"]}'
        if len(rpm) != 1 or not np.isfinite(rpm[0]) or rpm[0] <= 0:
            raise ValueError(f'{path}: RPM must be one finite positive scalar')
        raw = np.asarray(m[row['signal_key']], dtype=float).reshape(-1)
        fs = float(row['fs'])
        original_fs=fs
        if common_fs is not None:
            from fractions import Fraction
            from scipy.signal import resample_poly
            common_fs=float(common_fs)
            if not 0 < common_fs <= fs:
                raise ValueError('common_fs must be positive and no higher than the recording rate')
            ratio=Fraction(common_fs/fs).limit_denominator(10000)
            actual_fs=fs*ratio.numerator/ratio.denominator
            if abs(actual_fs-common_fs)>1e-6:
                raise ValueError('requested common_fs cannot be represented accurately')
            if ratio.numerator!=ratio.denominator:
                raw=resample_poly(raw,ratio.numerator,ratio.denominator)
            fs=actual_fs
        n = int(round(length_seconds * fs))
        if n < 8 or len(raw) < n:
            raise ValueError(f'{path}: insufficient samples')
        step = max(1, int(round(n * (1 - overlap))))
        for start in range(0, len(raw) - n + 1, step):
            r = {'recording_id':path.name, 'dataset':'CWRU', 'split':row['split'],
                 'x':raw[start:start+n], 'x_unit':row['x_unit'], 'fs':fs, 'shaft_speed':float(rpm[0]/60),
                 'z':float(row['z']), 'd_mm':float(row['d_mm']), 'D_mm':float(row['D_mm']),
                 'alpha_deg':float(row['alpha_deg']), 'label':row['label'],
                 'condition':row['condition'], 'fault_size_mm':row.get('fault_size_mm',''),
                 'A0':'', 'decay':'', 'snr_db':'',
                 'provenance':row['provenance'] + f'; speed source: {rpm_source}; original fs={original_fs:g} Hz; output fs={fs:g} Hz; {path.name} resampled samples [{start},{start+n})'}
            validate_record(r)
            all_records.append(r)
    return validate_splits(all_records)
