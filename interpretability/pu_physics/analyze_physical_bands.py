"""Exploratory physical-frequency audit for an already-trained PU model.

Requires the companion ig_test.npz produced by PU IG analysis. This is raw
acceleration FFT, NOT envelope analysis. Per-bin masking may be off manifold.
The SNR/NTN bearing is only a documented 6203-family reference, not a verified
match for any actual bearing. Masking is NOT evidence of physical causation.
"""
import argparse
import os
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(os.environ.get('BEARING_PROJECT_ROOT', str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(ROOT / 'pu'))
from eval_pu_bearing_holdout import Classifier

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=ROOT)
    p.add_argument('--ig',type=Path,default=Path(__file__).parent.parent/'pu'/'ig_test.npz')
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'physical_bands.json')
    a=p.parse_args(); torch.set_num_threads(2)
    archive=np.load(a.ig,allow_pickle=False)
    x=archive['standardized_fft']; ig=archive['attribution']; true=archive['true_label']; pred=archive['pred']; rival=archive['rival']; bearing=archive['bearing']
    assert x.shape==(1200,513) and ig.shape==x.shape
    # The IG generation saves consecutive recording 1, recording 2 windows for
    # each bearing, 100 each. Keep KI07 recording 2 out of frequency mapping:
    # its saved HostService time axis is visibly inconsistent with 64 kHz.
    recording=np.zeros(len(bearing),dtype=int)
    for b in sorted(set(bearing.tolist())):
        pos=np.flatnonzero(bearing==b)
        assert len(pos)==200
        recording[pos[:100]]=1; recording[pos[100:]]=2
    anomaly=((bearing=='KI07') & (recording==2))
    original_anomaly_x=x[anomaly].copy(); original_anomaly_pred=pred[anomaly].copy(); original_anomaly_true=true[anomaly].copy()
    keep=~anomaly
    x=x[keep]; ig=ig[keep]; true=true[keep]; pred=pred[keep]; rival=rival[keep]; bearing=bearing[keep]; recording=recording[keep]
    ckpt=torch.load(a.project/'pu'/'outputs'/'model.pt',map_location='cpu',weights_only=False)
    model=Classifier(513); model.load_state_dict(ckpt['model_state_dict']); model.eval()
    with torch.no_grad():
        xx=torch.tensor(x); z=model(xx); verify=z.argmax(1).numpy()
        if not np.array_equal(verify,pred): raise ValueError('IG artifact model predictions mismatch')
        ix=torch.arange(len(x)); orig=(z[ix,torch.tensor(pred)]-z[ix,torch.tensor(rival)]).numpy()
        checks={}
        # Each single bin compared with replacing that input with the training mean.
        for i in list(range(1,9)):
            modified=xx.clone(); modified[:,i]=0
            zz=model(modified); margin=(zz[ix,torch.tensor(pred)]-zz[ix,torch.tensor(rival)]).numpy()
            checks[str(i)]={'frequency_hz_nominal':i*62.5,'changed_prediction_fraction':float((zz.argmax(1).numpy()!=pred).mean()),'mean_margin_drop':float(np.mean(orig-margin))}
    absig=np.abs(ig); denom=absig.sum(axis=1)+1e-20
    bands={'outer_fundamental_example_bin_1':[1],'inner_fundamental_example_bin_2':[2],
           'outer_1st_to_3rd_harmonic_example_bins':[1,2,4],
           'inner_1st_to_3rd_harmonic_example_bins':[2,4,6],
           'both_example_harmonics':[1,2,4,6], 'low_62_5_to_500hz_bins_1_to_8':list(range(1,9))}
    per={}
    for b in sorted(set(bearing.tolist())):
        sel=bearing==b; per[b]={'n':int(sel.sum()),'true_class':int(true[sel][0]),'accuracy':float(np.mean(pred[sel]==true[sel])),
            'mean_abs_ig_share':{k:float((absig[sel][:,inds].sum(axis=1)/denom[sel]).mean()) for k,inds in bands.items()},
            'top_10_bins_by_mean_abs_ig':(np.argsort(absig[sel].mean(axis=0))[::-1][:10]).tolist()}
    files=[]; sensitivity=None
    for b in sorted(set(bearing.tolist())):
        for rec in (1,2):
            stem=f'N15_M07_F10_{b}_{rec}'; path=a.project/'data'/'PU'/'selected_mat'/b/(stem+'.mat')
            m=loadmat(path,simplify_cells=True)[stem]; t=np.asarray(m['X'][1]['Data'],dtype=float); dt=np.diff(t)
            vib=next(c for c in m['Y'] if c['Name']=='vibration_1')['Data']
            if len(t)!=len(vib): raise ValueError(f'Time-vibration length mismatch {path}')
            speed=next(c for c in m['Y'] if c['Name']=='speed')['Data']; speed=np.asarray(speed,dtype=float)
            if b=='KI07' and rec==2:
                # Counterfactual only: assume timestamps in seconds and linearly
                # interpolate onto 64 kHz. This cannot establish acquisition truth.
                # Use same sample-index window starts; clamp final start to fit.
                indices=np.linspace(0,len(vib)//1024-1,100,dtype=int)*1024
                offsets=np.arange(1024)/64000.; starts=np.minimum(t[indices],t[-1]-offsets[-1])
                resampled=np.stack([np.interp(start+offsets,t,vib) for start in starts])
                resampled=(resampled-resampled.mean(1,keepdims=True))/(resampled.std(1,keepdims=True)+1e-8)
                feats=np.abs(np.fft.rfft(resampled,axis=1)); feats=((feats-ckpt['fft_mean'])/ckpt['fft_std']).astype(np.float32)
                with torch.no_grad():
                    rp=model(torch.tensor(feats)).argmax(1).numpy()
                    recovered=model(torch.tensor(original_anomaly_x)).argmax(1).numpy()
                assert np.array_equal(recovered,original_anomaly_pred)
                sensitivity={'n':100,'method':'linear interpolation to 64kHz assuming timestamps in seconds; same original index starts except clipped final start; no retraining',
                    'original_accuracy':float(np.mean(original_anomaly_pred==original_anomaly_true)),
                    'interpolated_accuracy':float(np.mean(rp==original_anomaly_true)),
                    'prediction_change_fraction':float(np.mean(rp!=original_anomaly_pred)),
                    'not_proof_of_correction':'Acquisition timing and timestamps not verified; time interpolation is an assumption and may itself change signal.'}
            files.append({'bearing':b,'record':rec,'time_field':"X[1]['Data']",'raster':str(m['X'][1]['Raster']),'unit':str(m['X'][1]['Unit']),
               'start':float(t[0]),'end':float(t[-1]),'vibration_count':len(t),
               'mean_sample_rate_hz_from_endpoints':float((len(t)-1)/(t[-1]-t[0])),
               'median_interval_us':float(np.median(dt)*1e6),'min_interval_us':float(np.min(dt)*1e6),
               'max_interval_us':float(np.max(dt)*1e6),'intervals_below_10us':int(np.count_nonzero(dt<1e-5)),
               'nonincreasing_intervals':int(np.count_nonzero(dt<=0)),
               'speed_mean_raw':float(speed.mean()),'speed_std_raw':float(speed.std())})
    result={'model_accuracy_on_frequency_audit_subset':float(np.mean(pred==true)), 'test_count':len(x),'n_bearings':len(per),
        'excluded_from_hz_audit':'100 windows from KI07 recording 2: 274149 timestamps spanning about 4 s, 24989 intervals below 10us; mismatch with nominal 64k. Cause and actual sampling unknown; excluded from Hz mapping only.',
        'source_model':'PU 1024-point, raw time-standardized window, rFFT magnitude (513 bins), train-only feature standardization',
        'sample_rate_nominal_hz':64000, 'bin_spacing_hz':62.5,'nominal_shaft_rpm':1500,
        'reference_is_not_verified_bearing_model':'NTN/SNR 6203FT150, https://docs.rs-online.com/a404/A700000014892791.pdf; model identification of actual PU bearings not established',
        'bpfo_reference_hz':3.053*25,'bpfi_reference_hz':4.947*25,
        'nearest_bins': {'bpfo':1,'bpfi':2},'reference_harmonics_bins':bands,
        'interpretation_limits':['Frequency-bin interpolation is not frequency resolution; exact bearing geometry and bearing-specific reference frequencies unverified.',
         'Raw acceleration FFT is not demodulated envelope spectrum; high-frequency resonances may encode the impact rate.',
         'Attribution values are for the original predicted class versus its original runner-up, not necessarily the actual class.',
         'Masking swaps standardized FFT to training feature mean, which can be out of distribution and is not a physical intervention.',
         'Overlapping harmonic bins and only two test bearings per class do not establish frequency-specific physics alignment.'],
        'single_bin_ablation_1100_windows':checks,'per_bearing':per,'measurement_metadata':files,
        'ki07_record_2_interpolation_sensitivity':sensitivity}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
