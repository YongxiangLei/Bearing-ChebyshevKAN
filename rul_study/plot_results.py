"""Plot per-bearing test MAE, averaged across the fixed training seeds."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--summary',type=Path,default=Path(__file__).parent/'results_merged'/'summary.json')
    ap.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'results_merged')
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    d=json.loads(args.summary.read_text(encoding='utf8'))
    if not d['complete_15_folds']:
        raise ValueError('Expected full 15-bearing result')
    keys=['age_ridge','features_ridge','mlp','kan']
    names=['年龄+工况 Ridge','工程特征 Ridge','MLP','Chebyshev-KAN']
    bearing_ids=sorted(d['macro_results'][keys[0]]['mean_mae_by_test_bearing'])
    x=np.array([[d['macro_results'][key]['mean_mae_by_test_bearing'][b] for key in keys] for b in bearing_ids])
    pd.DataFrame(x,index=bearing_ids,columns=names).rename_axis('测试轴承').to_csv(args.output_dir/'per_bearing_mae.csv',encoding='utf-8-sig')
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(10.5,9.2),constrained_layout=True)
    im=ax.imshow(x,cmap='YlOrRd',aspect='auto',vmin=0,vmax=float(np.max(x)))
    ax.set_xticks(range(len(names)),names,fontsize=10)
    ax.set_yticks(range(len(bearing_ids)),bearing_ids,fontsize=9)
    ax.tick_params(length=0)
    ax.set_xlabel('模型（每列独立按验证集选择参数）')
    ax.set_ylabel('留出的测试轴承')
    ax.set_title('XJTU-SY：逐轴承 RUL 平均绝对误差\n15 轴承留一测试；每格为 3 个训练种子的均值；目标为距末次采集的分钟数',fontsize=12,pad=15)
    cutoff=float(np.max(x))*0.58
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            ax.text(col,row,f'{x[row,col]:.0f}',ha='center',va='center',fontsize=9,color='white' if x[row,col]>cutoff else 'black')
    for boundary in (4.5,9.5): ax.axhline(boundary,color='white',lw=1.7)
    cb=fig.colorbar(im,ax=ax,shrink=.78,pad=.02); cb.set_label('平均绝对误差（分钟；越低越好）')
    fig.savefig(args.output_dir/'per_bearing_mae.png',dpi=200,bbox_inches='tight'); plt.close(fig)
    print('Saved',args.output_dir/'per_bearing_mae.csv',args.output_dir/'per_bearing_mae.png')

if __name__=='__main__':main()
