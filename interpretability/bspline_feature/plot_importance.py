"""Plot cross-fitted, bearing-equal B-spline KAN IG importances."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=Path(__file__).parent/'global_feature_importance.csv')
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'bspline_global_importance.png')
    a=p.parse_args();df=pd.read_csv(a.input).sort_values('mean_abs_ig_min')
    labels={'speed_hz':'标称转速','load_kN':'径向载荷','elapsed_minutes':'已运行时间',
      'horizontal_rms':'水平 RMS','horizontal_peak':'水平峰值','horizontal_variance':'水平振动方差',
      'horizontal_kurtosis':'水平超额峭度','horizontal_skewness':'水平偏度',
      'vertical_rms':'垂直 RMS','vertical_peak':'垂直峰值','vertical_variance':'垂直振动方差',
      'vertical_kurtosis':'垂直超额峭度','vertical_skewness':'垂直偏度'}
    f=Path('C:/Windows/Fonts/msyh.ttc')
    if f.exists():
        font_manager.fontManager.addfont(str(f));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(f)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(10,7.5),constrained_layout=True)
    colors=['#5A7E85' if n in ['speed_hz','load_kN','elapsed_minutes'] else '#B5C6C8' for n in df.feature]
    bars=ax.barh([labels.get(x,x) for x in df.feature],df.mean_abs_ig_min,color=colors)
    maximum=float(df.mean_abs_ig_min.max())
    ax.set_xlim(0,maximum*1.26)
    for bar,(_,r) in zip(bars,df.iterrows()):
        ax.text(float(r.mean_abs_ig_min)+maximum*.015,bar.get_y()+bar.get_height()/2,
                f"{float(r.mean_abs_ig_min):.1f}    ({int(r.top3_bearing_count)}/15)",va='center',fontsize=9)
    ax.set_xlabel('对模型输出的平均绝对归因（分钟，相对训练均值基线）')
    ax.set_title('B 样条 KAN：15 个留出轴承的特征归因\n每轴承先平均后等权汇总；括号为该特征位列轴承前三的次数（seed 42）',fontsize=11,pad=12)
    ax.grid(axis='x',alpha=.25);ax.set_axisbelow(True)
    fig.savefig(a.output,dpi=190,bbox_inches='tight');plt.close(fig)
    print('Saved',a.output)
if __name__=='__main__':main()
