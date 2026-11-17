"""Visualize cross-fitted feature importance and example learned Chebyshev edges."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

NAMES={'horizontal_rms':'水平 RMS','horizontal_peak':'水平绝对峰值','horizontal_variance':'水平方差',
       'horizontal_kurtosis':'水平超额峭度','horizontal_skewness':'水平偏度','vertical_rms':'垂直 RMS',
       'vertical_peak':'垂直绝对峰值','vertical_variance':'垂直方差','vertical_kurtosis':'垂直超额峭度',
       'vertical_skewness':'垂直偏度','elapsed_minutes':'已运行时间','speed_hz':'转速','load_kN':'径向载荷'}

def setup_font():
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--input-dir',type=Path,default=Path(__file__).parent)
    a=ap.parse_args();out=a.input_dir;setup_font()
    df=pd.read_csv(out/'global_feature_attribution.csv')
    fig,ax=plt.subplots(figsize=(10.5,7.5),layout='constrained')
    vals=df['mean_abs_ig_minutes'].to_numpy()[::-1];labels=[NAMES[f] for f in df.feature.to_numpy()[::-1]]
    bars=ax.barh(labels,vals,color=['#bd754b' if v in ('已运行时间','转速','径向载荷') else '#356c83' for v in labels],height=.72)
    ax.bar_label(bars,labels=[f'{v:.1f}' for v in vals],padding=4,fontsize=9)
    ax.set_xlim(0,float(vals.max())*1.2);ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel('平均绝对归因（预测分钟数相对训练均值基线）')
    ax.set_title('Chebyshev-KAN：15 个留出轴承的输入特征归因\n先按轴承和训练种子等权平均；不表示因果或预测正确',fontsize=12,pad=14)
    fig.savefig(out/'cheb_feature_attribution.png',dpi=190,bbox_inches='tight');plt.close(fig)

    import json
    selections=json.loads((out/'edge_selections.json').read_text(encoding='utf8'))
    curves=pd.read_csv(out/'edge_curves.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for ax,sel in zip(axes,selections):
        feature=sel['feature'];mean=sel['training_mean'];lo,hi=sel['training_range_5_95'];hidden=sel['hidden_index']
        points=curves[curves.feature==feature]
        ax.plot(points.x,points.centered_edge_output,color='#356c83',lw=2)
        if lo<=mean<=hi: ax.axvline(mean,color='#777777',lw=1,ls='--')
        ax.axhline(0,color='#bbbbbb',lw=.8);ax.set_title(f'{NAMES[feature]} → 隐层节点 {hidden}',fontsize=11)
        unit='原始信号单位' if feature=='vertical_rms' else '无量纲'
        ax.set_xlabel(f'{NAMES[feature]}（{unit}；训练样本 5%–95% 范围）');ax.set_ylabel('边函数变化（隐藏层内部单位）')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('示例：留出 Bearing1_1、种子 42 的 Chebyshev 边函数\n以训练均值为零点；单条边不是完整模型的 RUL 曲线',fontsize=12)
    fig.savefig(out/'cheb_example_edges.png',dpi=190,bbox_inches='tight');plt.close(fig)
    print('Saved',out/'cheb_feature_attribution.png',out/'cheb_example_edges.png')

if __name__=='__main__':main()
