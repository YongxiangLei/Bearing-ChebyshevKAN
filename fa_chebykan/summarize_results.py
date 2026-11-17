"""Summarise two saved exploratory benchmark reports without re-running tests."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
docs = [("48 kHz 假设", OUT / 'normal48' / 'report.json'),
        ("12 kHz 假设", OUT / 'normal12' / 'report.json')]
names = {'fa_cheby': 'FA-ChebyKAN', 'anchored_mlp': 'Anchored MLP',
         'spectrum_mlp': 'Order-spectrum MLP', 'spectrum_cheby': 'Order-spectrum ChebyKAN'}
columns = [('Zero-shot', None, None), ('K=1 pretrained', '1', 'pretrained'),
           ('K=1 scratch', '1', 'scratch'), ('K=2 pretrained', '2', 'pretrained'),
           ('K=2 scratch', '2', 'scratch')]

def score(model, k, mode):
    real = model['real']
    item = real['zero_shot_test'] if k is None else real['few_shot'][k][mode]['test']
    return item['accuracy'], item['macro_f1_present_classes']

report = ['# CWRU 仿真预训练与少样本诊断初步结果', '',
'本结果来自示例冲击共振仿真器，不是经验证的工业数字孪生。CWRU 外圈、内圈、滚动体故障的原始文件为 12 kHz；正常记录的采样率尚未逐文件证实。下面的两个情景分别假设正常记录为 48 kHz 与 12 kHz。', '',
'两种情景都固定：0/1 hp 训练、2 hp 验证、3 hp 测试；3 hp 测试共 **4 条录音，每类 1 条**。按录音聚合预测，一条错误即改变准确率 25 个百分点。不同负载录音可能来自同一物理轴承或同一故障，本实验不能证明跨轴承泛化。', '',
'仿真数据由种子 42 生成，每类 32/8/8 条训练/验证/测试记录；各模型用同一仿真数据和划分、20 轮预训练、15 轮微调上限、种子 42、批量 8 和相同的录音加权训练流程。按模拟验证集或真实验证集宏 F1 选轮次；多模型及两个情景均已查看测试结果，因此比较为探索性。48 kHz 情景下正常波形先抗混叠重采样到 12 kHz；所有样本按 1600–4800 Hz 带通、Hilbert 包络和 3 kHz 包络采样处理，单段 1.2 秒、无重叠。', '',
'固定频段为原型选择，尚未证明对 CWRU 各录音最优。K 是每类互不相同的训练录音数，K=1 使用固定种子选择，K=2 使用该类两条训练录音；K=3、5、10 由于每类只有两条训练录音而不可评估。', '']

fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
for idx, (scenario, path) in enumerate(docs):
    data = json.loads(path.read_text(encoding='utf8'))
    report += ['## 正常记录 ' + scenario, '',
               '| 模型 | 零样本 | K=1 预训练 | K=1 从零训练 | K=2 预训练 | K=2 从零训练 | 参数量 |',
               '|---|---:|---:|---:|---:|---:|---:|']
    matrix = []
    for key, name in names.items():
        model = data['models'][key]
        scores = [score(model, k, mode)[0] for _, k, mode in columns]
        matrix.append([int(round(s*4)) for s in scores])
        report.append('| ' + name + ' | ' + ' | '.join(f'{s*100:.0f}% ({round(s*4)}/4)' for s in scores) + f" | {model['parameter_count']} |")
    report += ['']
    ax = axes[idx]
    img = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=4, aspect='auto')
    for (row, col), value in np.ndenumerate(np.asarray(matrix)):
        ax.text(col, row, f'{value}/4', ha='center', va='center', color='white' if value>=3 else 'black', fontsize=10)
    ax.set_title(('Normal 48 kHz assumed' if idx==0 else 'Normal 12 kHz assumed'), fontsize=11)
    ax.set_xticks(range(5), [x[0].replace(' ', '\n') for x in columns], fontsize=8)
    ax.set_yticks(range(4), list(names.values()), fontsize=9)
    ax.tick_params(length=0)
fig.suptitle('CWRU exploratory comparison: four test recordings per scenario', fontsize=12)
fig.savefig(OUT/'cwru_comparison.png', dpi=220, bbox_inches='tight')
plt.close(fig)
report += ['## 解释与局限', '',
'FA-ChebyKAN 在两个情景的零样本均为 1/4；K=1 和 K=2 中，预训练与从零训练均为 4/4，因此没有观察到预训练对 FA 模型的增益。其他模型的个别结果随正常记录采样率假设变化；这不是确定采样率的证据。完整宏 F1、混淆矩阵、训练历史、选中的训练录音和不可行 K 的原因均记录在对应 `report.json`。', '',
'四个模型分别为故障阶次锚定 ChebyKAN、相同锚定特征加 MLP、固定转速阶次谱加 MLP 和固定阶次谱加普通 ChebyKAN；参数量不同，固定阶次谱也不是对论文所有强基线的复现。需要更多独立轴承、多次种子、额外支持集抽样、强基线及不同工况/设备验证，才能评价稳定性与工程价值。', '',
'数据、训练和具体复现步骤见 `FA_CWRU_复现说明.md`；官方来源及尚未证实的元数据见 `CWRU_元数据核查.md`。']
(OUT/'CWRU_实验结果报告.md').write_text('\n'.join(report)+'\n', encoding='utf8')
print('Saved report and comparison figure to', OUT)
