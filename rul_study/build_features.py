"""Build current-record XJTU-SY engineering features and minute-RUL labels.

Each numeric CSV is one acquired record; label uses its ordinal relative to the
last observed record, assuming the provider's one-minute acquisition cadence.
Endpoints are empirical last observations, not proven failure timestamps.
No input uses future records, a full-life endpoint, or a full-life statistic.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

CONDITIONS = {'35Hz12kN': (35.,12.), '37.5Hz11kN': (37.5,11.), '40Hz10kN': (40.,10.)}
FEATURES = [f'{c}_{s}' for c in ('horizontal','vertical') for s in ('rms','peak','variance','kurtosis','skewness')]
EXPECTED_COUNTS = {'Bearing1_1':123,'Bearing1_2':161,'Bearing1_3':158,'Bearing1_4':122,'Bearing1_5':52,
                   'Bearing2_1':491,'Bearing2_2':161,'Bearing2_3':533,'Bearing2_4':42,'Bearing2_5':339,
                   'Bearing3_1':2538,'Bearing3_2':2496,'Bearing3_3':371,'Bearing3_4':1515,'Bearing3_5':114}

def features(path):
    d = pd.read_csv(path)
    if list(d.columns) != ['Horizontal_vibration_signals', 'Vertical_vibration_signals'] or len(d) != 32768:
        raise ValueError(f'Unexpected columns or sample count: {path}')
    a = d.to_numpy(dtype=np.float64)
    if not np.isfinite(a).all(): raise ValueError(f'Nonfinite sample: {path}')
    m = a.mean(axis=0); c = a-m
    sd = np.sqrt(np.mean(c*c,axis=0)); denom = np.maximum(sd, 1e-12)
    rms=np.sqrt(np.mean(a*a,axis=0)); peak=np.max(np.abs(a),axis=0); var=sd**2
    skew=np.mean((c/denom)**3,axis=0); kurt=np.mean((c/denom)**4,axis=0)-3
    out=np.stack((rms,peak,var,kurt,skew),axis=1).reshape(-1)
    if not np.isfinite(out).all(): raise ValueError(f'Nonfinite feature: {path}')
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True,help='condition/bearing/CSV root')
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'data')
    p.add_argument('--allow-partial',action='store_true',help='Only populate caches for complete verified bearings; do not emit final features or labels')
    p.add_argument('--only-bearings',default='',help='Comma separated bearing IDs to pre-cache; requires --allow-partial')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    only=set(a.only_bearings.split(',')) if a.only_bearings else None
    if only and not a.allow_partial: p.error('--only-bearings requires --allow-partial')
    cache=a.output/'cache'; cache.mkdir(exist_ok=True)
    records=[]; manifest=[]
    for condition,(speed,load) in CONDITIONS.items():
        cond=a.data/condition
        if not cond.is_dir():
            if a.allow_partial: continue
            raise FileNotFoundError(cond)
        bearings=sorted(x for x in cond.glob('Bearing*') if x.is_dir())
        if len(bearings)!=5 and not a.allow_partial: raise ValueError(f'Expected 5 bearings in {cond}; found {len(bearings)}')
        for b in bearings:
            if only and b.name not in only: continue
            if not b.name.startswith('Bearing'+str(list(CONDITIONS).index(condition)+1)+'_'):
                raise ValueError(f'Bearing {b.name} is in the wrong condition directory {condition}')
            fs=list(b.glob('*.csv'))
            fs.sort(key=lambda q:int(q.stem))
            if b.name not in EXPECTED_COUNTS or len(fs) != EXPECTED_COUNTS[b.name]:
                if a.allow_partial: continue
                raise ValueError(f'Incorrect file count for {b}: {len(fs)}')
            if [int(x.stem) for x in fs]!=list(range(1,len(fs)+1)) or len(fs)<2:
                raise ValueError(f'Noncontiguous or insufficient CSV files in {b}')
            cachefile=cache/(b.name+'.npz')
            # Cache records count and paths in addition to computed arrays; caller
            # must delete cache if the underlying source version changes.
            if cachefile.exists():
                stored=np.load(cachefile)
                vals=stored['features']
                if vals.shape!=(len(fs),10): raise ValueError(f'Stale cache: {cachefile}')
            else:
                vals=[]
                for n,f in enumerate(fs,1):
                    vals.append(features(f))
                    if n%100==0: print(f'{b.name}: {n}/{len(fs)}',flush=True)
                vals=np.asarray(vals)
                np.savez_compressed(cachefile,features=vals)
            for i,(f,v) in enumerate(zip(fs,vals)):
                row={'condition':condition,'bearing':b.name,'record':i+1,'source_file':str(f.relative_to(a.data)),
                     'elapsed_minutes':float(i),'speed_hz':speed,'load_kN':load,'rul_minutes':float(len(fs)-1-i)}
                row.update({name:float(value) for name,value in zip(FEATURES,v)})
                records.append(row)
            manifest.append({'bearing':b.name,'condition':condition,'records':len(fs),'first_record':fs[0].name,'last_record':fs[-1].name,
                             'sampling_hz':25600,'samples_per_channel':32768,'cadence_minutes_assumed':1,
                             'observed_life_minutes':len(fs)-1,'endpoint_definition':'timestamp of last observed acquisition, not necessarily exact failure time'})
            print('COMPLETE',b.name,len(fs),flush=True)
    if a.allow_partial:
        print('Partial caching completed;',len(manifest),'complete bearings',flush=True); return
    if len(manifest)!=15 or len({i['bearing'] for i in manifest})!=15: raise ValueError('Expected 15 unique bearings')
    pd.DataFrame(records).to_csv(a.output/'features_and_labels.csv',index=False)
    (a.output/'manifest.json').write_text(json.dumps({'source_root_at_feature_extraction':str(a.data.resolve()),
        'default_project_relative_source_root':'data/XJTU-SY/XJTU-SY_Bearing_Datasets',
        'cache_caution':'Cache shape/count only; delete cache on source changes.',
        'features':FEATURES,'records':len(records),'bearings':manifest},indent=2),encoding='utf8')
    print('Saved',len(records),'records from',len(manifest),'bearings',flush=True)

if __name__=='__main__':main()
