"""Summarize Paderborn results over repeated seeds; no experiment is rerun."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results', type=Path, default=ROOT / 'results',
                    help='Directory containing the multi-seed report.json')
out = parser.parse_args().results.resolve()
data = json.loads((out / 'report.json').read_text(encoding='utf8'))
names = {'fa_cheby':'FA-ChebyKAN', 'anchored_mlp':'Anchored MLP',
         'spectrum_mlp':'Order-spectrum MLP', 'spectrum_cheby':'Order-spectrum ChebyKAN'}
modes = [('zero_shot','Zero-shot'),('pretrained','Pretrained + fine-tuning'),('scratch','From scratch')]
seeds = data['seeds']
for model in names:
    if set(data['runs'].get(model, {})) != {str(i) for i in seeds}:
        raise RuntimeError(f'Missing completed runs for {model}')

def metrics(run, mode):
    return run[mode] if mode == 'zero_shot' else run[mode]['test']

summary = {'seeds':seeds,'std_definition':'sample standard deviation across seeds (ddof=1), not uncertainty across bearings',
           'test_bearings':6,'models':{}}
text = ['# Paderborn 轴承级故障诊断多种子结果', '',
        '这是固定轴承划分的探索性三分类实验；测试集为健康、外圈、内圈各两个轴承，共 6 个轴承。5 个随机种子使用同一划分，因此标准差反映的是当前划分下的随机初始化与训练变化，不是跨轴承或跨划分的置信区间。', '',
        '每类 3 个轴承训练、1 个验证、2 个测试；每个轴承的所有记录和信号段保持在同一划分。使用振动通道及每段实测转速均值，按轴承等权训练并通过平均片段 logits 得到轴承级预测。预训练仅用示例仿真中的健康、外圈和内圈三类，同一仿真数据集，5 个种子独立初始化；其随机信号不等于经过验证的物理数字孪生。目标域比较仿真预训练后微调与从零训练，训练上限及选择规则相同。', '',
        '原始 36 条 MAT 中，按统一的时间轴质量门槛剔除 4 条，32 条保留为 96 个 1.2 s 片段。振动时间轴先插值至 64 kHz，再抗混叠降至 12 kHz；统一用 1600–4800 Hz 带通、Hilbert 包络及 3 kHz 包络采样。1% 密度容差、线性插值、名义几何及 3 个验证轴承等均限制结果；详见 `../evidence/`。', '',
        '| 模型 | 零样本准确率 | 预训练后微调准确率 | 从零训练准确率 | 参数量 |',
        '|---|---:|---:|---:|---:|']
fig, axes = plt.subplots(1,3,figsize=(15,5.5),sharey=True,layout='constrained')
colors=['#24578A','#B95E27','#467B5B','#73548D']
for mi,(mode,title) in enumerate(modes):
    ax=axes[mi]
    for xi,(key,label) in enumerate(names.items()):
        runs=[data['runs'][key][str(seed)] for seed in seeds]
        vals=np.asarray([metrics(run,mode)['accuracy']*100 for run in runs])
        f1=np.asarray([metrics(run,mode)['macro_f1']*100 for run in runs])
        bal=np.asarray([metrics(run,mode)['balanced_accuracy']*100 for run in runs])
        item={'accuracy_percent':vals.tolist(),'accuracy_mean_percent':float(vals.mean()),
              'accuracy_sample_sd_percent':float(vals.std(ddof=1)),
              'macro_f1_percent':f1.tolist(),'macro_f1_mean_percent':float(f1.mean()),
              'macro_f1_sample_sd_percent':float(f1.std(ddof=1)),
              'balanced_accuracy_percent':bal.tolist(), 'balanced_accuracy_mean_percent':float(bal.mean())}
        summary['models'].setdefault(key, {'parameters':runs[0]['parameter_count']})[mode]=item
        ax.bar(xi,vals.mean(),color=colors[xi],alpha=.72,width=.62)
        ax.errorbar(xi,vals.mean(),yerr=vals.std(ddof=1),fmt='none',color='black',capsize=3,lw=1)
        ax.scatter(np.full(len(vals),xi)+np.linspace(-.18,.18,len(vals)),vals,color='black',s=15,zorder=3)
    ax.set_title(title,fontsize=11)
    ax.set_xticks(range(4),['FA\nChebyKAN','Anchored\nMLP','Spectrum\nMLP','Spectrum\nChebyKAN'],fontsize=9)
    ax.set_ylim(0,113);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set_ylabel('Bearing-level accuracy (%)')
fig.suptitle('Paderborn: five seeds, same six test bearings (dots = individual seeds)',fontsize=12)
fig.savefig(out/'paderborn_multiseed_comparison.png',dpi=220,bbox_inches='tight');plt.close(fig)
for key,label in names.items():
    vals=summary['models'][key]
    parts=[f"{vals[mode]['accuracy_mean_percent']:.1f} ± {vals[mode]['accuracy_sample_sd_percent']:.1f}%" for mode,_ in modes]
    text.append('| '+label+' | '+' | '.join(parts)+f" | {vals['parameters']} |")
fa_recall = np.asarray([data['runs']['fa_cheby'][str(seed)]['pretrained']['test']['recall'] for seed in seeds]).mean(axis=0)*100
text += ['', f'FA-ChebyKAN 预训练后微调时，健康、外圈和内圈的跨种子平均召回率分别为 {fa_recall[0]:.0f}%、{fa_recall[1]:.0f}% 和 {fa_recall[2]:.0f}%。每类只有两个测试轴承；健康类平均召回偏低，实际误报风险不能由总体准确率判断。', '',
         '数值为 5 个种子的平均值 ± 样本标准差；各测试分数以 6 个轴承为分母，每个误判改变准确率约 16.7 个百分点。四种模型参数量不同，不能称为严格容量匹配；三个基线覆盖相同锚定特征 MLP 与固定阶次谱的 MLP/普通 ChebyKAN，尚非完整的先进方法对照。各种子的准确率、宏 F1、混淆矩阵、逐轴承预测与验证历史见 `report.json` 和 `summary.json`。', '',
         '本实验仅覆盖 N15_M07_F10 单工况、固定划分和部分人工/自然损伤；该划分与结果均为探索性，已多次检查测试结果。不能据此宣称跨工况、工业应用、统计显著性或真实物理因果。', '',
         '原数据由 Paderborn University KAt DataCenter 提供，许可证为 CC BY-NC 4.0；来源与引用见 `../evidence/Paderborn_元数据与时钟核验.md`。']
(out/'Paderborn_多种子实验报告.md').write_text('\n'.join(text)+'\n',encoding='utf8')
(out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
print('Saved summary and chart to', out)
