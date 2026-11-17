"""Plot numerical arrays in a separate process to avoid OpenMP conflicts."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).parent/'visuals');a=p.parse_args()
d=np.load(a.output/'plot_data.npz');z=d['z'];b=d['basis'];grid=d['grid'];edge=d['edge'];pred=d['pred'];node=int(d['node']);q05=float(d['q05']);q95=float(d['q95']);med=float(d['median'])
reference=BSpline.design_matrix(z,d['knots'],3).toarray();err=float(np.max(np.abs(b-reference)))
meta=json.loads((a.output/'interpretability.json').read_text(encoding='utf8'));meta['basis_scipy_max_error']=err
(a.output/'interpretability.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams['axes.unicode_minus']=False
fig,axs=plt.subplots(1,3,figsize=(17,5.7),constrained_layout=True)
for i in range(b.shape[1]):axs[0].plot(z,b[:,i],lw=1.6,label=f'B{i}')
axs[0].set_title('三次 B 样条基函数（固定）');axs[0].set_xlabel('映射后坐标 z');axs[0].set_ylabel('基函数值');axs[0].legend(ncol=3,fontsize=7,loc='upper center')
axs[1].plot(grid,edge,color='#176b7a',lw=2.2)
axs[1].axvspan(q05,q95,color='#dce8e8',alpha=.75,label='训练记录 5%–95%')
axs[1].axvline(med,color='#687373',linestyle='--',lw=1,label='训练记录中位数')
axs[1].set_title(f'水平 RMS → 隐藏节点 {node} 的已训练边函数');axs[1].set_xlabel('水平 RMS（数据原始单位）');axs[1].set_ylabel('边的隐藏层贡献（非分钟）');axs[1].legend(fontsize=8)
axs[2].plot(grid,pred,color='#aa5721',lw=2.2)
axs[2].axvspan(q05,q95,color='#eee2d7',alpha=.75,label='训练记录 5%–95%')
axs[2].axvline(med,color='#766f6a',linestyle='--',lw=1)
axs[2].set_title('固定其他特征时的整网响应');axs[2].set_xlabel('水平 RMS（数据原始单位）');axs[2].set_ylabel('预测距末次采集的分钟数');axs[2].legend(fontsize=8)
fig.suptitle('真实 B 样条 KAN：基函数、边函数与模型响应\n留出 Bearing1_3；seed 42；边按训练记录变化选定',fontsize=13)
fig.savefig(a.output/'bspline_basis_and_edge.png',dpi=180);plt.close(fig)
features=meta['feature_mean_absolute_ig_minutes'];label_map={
 'horizontal_rms':'水平 RMS','horizontal_peak':'水平峰值','horizontal_variance':'水平振动方差',
 'horizontal_kurtosis':'水平超额峭度','horizontal_skewness':'水平偏度',
 'vertical_rms':'垂直 RMS','vertical_peak':'垂直峰值','vertical_variance':'垂直振动方差',
 'vertical_kurtosis':'垂直超额峭度','vertical_skewness':'垂直偏度',
 'elapsed_minutes':'已运行时间','speed_hz':'转速','load_kN':'载荷'}
rank=sorted(features,key=features.get)  # low to high; top at the upper end
fig,ax=plt.subplots(figsize=(9.5,7.5),constrained_layout=True)
ax.barh([label_map.get(k,k) for k in rank],[features[k] for k in rank],color='#427e8a')
ax.set_xlabel('平均绝对积分梯度（分钟；非因果效应）')
ax.set_title('B 样条 KAN 在 Bearing1_3 的特征归因\n158 条留出记录；基线为训练特征均值；输出裁剪为非负',fontsize=12)
ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
for i,k in enumerate(rank):ax.text(features[k]+max(features.values())*.01,i,f'{features[k]:.1f}',va='center',fontsize=8)
ax.set_xlim(0,max(features.values())*1.18)
fig.savefig(a.output/'bspline_feature_importance.png',dpi=180);plt.close(fig)
print('Plot saved, SciPy max basis difference:',err)
