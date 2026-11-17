"""Compare two timebase assumptions on the first 1.2 s of each PU record.

This is a data-quality sensitivity diagnostic only. It neither uses fault
labels to pick a preprocessing rule nor invokes trained models. Raw PU data
are read in place and never modified.
"""
from pathlib import Path
import argparse
import csv
import math
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, hilbert, resample_poly

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[3],
                    help='Project root containing data/PU/selected_mat')
PROJECT = parser.parse_args().project.resolve()
MANIFEST = HERE/'pu_bearing_record_manifest.csv'
FS = 64000
DURATION = 1.2
N = int(FS*DURATION)
grid = np.arange(N, dtype=np.float64)/FS
sos = butter(4, [1600,4800], btype='bandpass', fs=FS, output='sos')

def get_channel(tree, name):
    matches=[v for v in tree['Y'] if v['Name']==name]
    if len(matches)!=1: raise ValueError(f'Expected one {name}')
    c=matches[0]
    x=np.asarray(c['Data'],dtype=np.float64).reshape(-1)
    t=np.asarray(tree['X'][int(c['XIndex'])-1]['Data'],dtype=np.float64).reshape(-1)
    if len(x)!=len(t) or not np.isfinite(x).all() or not np.isfinite(t).all() or not (np.diff(t)>0).all():
        raise ValueError('Invalid channel data/timestamps')
    return x,t

def features(x, speed_hz):
    filtered=sosfiltfilt(sos,x)
    env=np.abs(hilbert(filtered))
    env-=env.mean()
    env=resample_poly(env,3,64)
    env-=env.mean()
    t=np.arange(len(env),dtype=np.float64)/3000
    phase=2*np.pi*speed_hz*t
    rho=6.75/28.55
    orders=[4*(1-rho),4*(1+rho),28.55/(2*6.75)*(1-rho*rho)]
    cage=(1-rho)/2
    norm=math.sqrt(max(2*np.mean(env**2),1e-12))
    vals=[]
    for c,order in enumerate(orders):
        mmax=0 if c==0 else 2
        mod=phase*(cage if c==2 else 1)
        for h in range(1,4):
            for m in range(-mmax,mmax+1):
                theta=h*order*phase+m*mod
                a=2*np.mean(env*np.cos(theta)); b=2*np.mean(env*np.sin(theta))
                vals.append(np.log1p(np.hypot(a,b)/norm))
    return np.asarray(vals)

with MANIFEST.open(encoding='utf8',newline='') as f: records=list(csv.DictReader(f))
results=[]
for i,r in enumerate(records,1):
    path=PROJECT/Path(r['relative_path'])
    tree=loadmat(path,simplify_cells=True)[r['recording_id']]
    x,t=get_channel(tree,'vibration_1')
    speed,ts=get_channel(tree,'speed')
    if len(x)<N or t[0]>0 or t[-1]<grid[-1] or ts[-1]<grid[-1]:
        raise ValueError(f'Insufficient coverage: {path}')
    mask=(ts>=0)&(ts<DURATION)
    speed_hz=float(speed[mask].mean()/60)
    nominal=x[:N]
    aligned=np.interp(grid,t,x)
    a,b=features(nominal,speed_hz),features(aligned,speed_hz)
    diff=np.abs(a-b)
    denom=max(float(np.abs(b).sum()),1e-12)
    results.append({'recording_id':r['recording_id'],'bearing_id':r['bearing_id'],
       'split':r['split'],'mean_timeaxis_hz':r['time_equivalent_hz'],
       'frequency_deviation_percent':100*(float(r['time_equivalent_hz'])/FS-1),
       'within_1percent':int(abs(float(r['time_equivalent_hz'])/FS-1)<=0.01),
       'speed_hz':speed_hz,'n_features':len(a),
       'mean_abs_difference':float(diff.mean()),'max_abs_difference':float(diff.max()),
       'relative_l1_to_interpolated':float(diff.sum()/denom)})
    print(i,r['recording_id'], 'relative L1', round(results[-1]['relative_l1_to_interpolated'],4),flush=True)
with (HERE/'timebase_feature_sensitivity.csv').open('w',encoding='utf8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=results[0]);w.writeheader();w.writerows(results)
kept=[r for r in results if r['within_1percent']]
rel=np.asarray([r['relative_l1_to_interpolated'] for r in kept])
examples=[r for r in results if r['recording_id'] in {
    'N15_M07_F10_K001_1','N15_M07_F10_K005_1','N15_M07_F10_KI01_1','N15_M07_F10_KI05_1'}]
lines=['# PU 时钟处理的特征敏感性检查','',
 '对 36 条原始录音各自的前 1.2 秒进行诊断，仅比较输入处理，不评估模型或使用故障标签选参数。',
 '本诊断直接在 64 kHz 信号上带通，而主实验先将振动信号降至 12 kHz 后再带通；因此数值仅用于检查时间处理敏感性，不能直接解释为主实验模型预测的变化。',
 '方法 A：取前 76800 点并当作 64 kHz；方法 B：依据 MAT vibration_1 对应的 X 时间轴线性插值至 64 kHz 的前 1.2 秒。两者采用同一段实测速率均值，之后相同 1600–4800 Hz 四阶 Butterworth 零相位带通、Hilbert 包络、抗混叠重采样到 3 kHz，计算同一组 33 个几何锚定相对幅值特征。',
 '相对 L1 差异定义为两组 log1p 特征的绝对差之和，除以插值方案特征绝对值之和。它是特征一致性指标，不是准确率、真实误差或插值正确性的证明。','',
 f'按平均等效采样频率偏离 64 kHz 不超过 1% 选取的 {len(kept)} 条记录：相对 L1 差异中位数 {np.median(rel):.4f}，范围 {rel.min():.4f}–{rel.max():.4f}。','','例子：','']
for r in examples: lines.append(f"- {r['recording_id']}：平均等效 {float(r['mean_timeaxis_hz']):.2f} Hz，特征相对 L1 差异 {r['relative_l1_to_interpolated']:.4f}。")
lines += ['', '阈值只筛选平均频率，不能排除局部时间戳异常。线性插值对不均匀采样不是严格抗混叠滤波，无法证明连续信号可被无失真恢复；固定 64 kHz 与时间轴插值都是待验证假设，需关注处理敏感性。全部逐录音数值见 `timebase_feature_sensitivity.csv`。']
(HERE/'PU_时间轴特征敏感性.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
