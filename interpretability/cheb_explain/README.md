# 已训练 Chebyshev-KAN 的特征解释

本目录解释项目已有的 15 折 × 3 种子、共 45 个 XJTU-SY 剩余寿命模型检查点。**这些模型的边函数是切比雪夫多项式，不是 B 样条函数。** 输入包括水平、垂直两通道的 RMS、峰值、方差、超额峭度、偏度，以及已运行时间、转速和径向载荷；输出为距最后一次采集的预测分钟数，并保留原先的非负裁剪。

## 全模型特征归因

`cheb_feature_attribution.png` 与 `global_feature_attribution.csv` 为 45 个模型的积分梯度：每折只解释对应的**留出轴承**，基线是该折训练轴承等权的输入均值，使用 128 个中点；对每个轴承内的采集记录求平均，再对 3 种子及 15 轴承等权求平均。数值是相对于此基线的平均绝对归因分钟数，不是直接删除该特征导致的误差增量，也不是失效原因。

本次前三项是径向载荷 72.3、转速 60.5、已运行时间 57.7；振动特征中较高的是水平偏度 38.4、垂直 RMS 37.0。三个工况的转速和载荷完全共线，所以不能把其排名解读为各自独立的物理贡献。直线积分途中及均值基线可能形成未实际观测到的工况和相关特征组合。以归因向量绝对值为依据的同轴承跨种子 top-3 平均 Jaccard 为 0.404；该指标描述这三次训练的排名稳定性，不等于 45 只独立轴承。

128 步积分在各模型及留出轴承等权平均后的完整性误差约 0.311 分钟，对应平均绝对输出差 225.05 分钟的 **0.138%**。这只是数值近似检查，不证明解释有物理真实性。还核对了所有检查点输出与保存预测的最大差为 0.000061 分钟。完整细节见 `metrics.json`、`per_fold_feature_attribution.csv` 和 `within_bearing_seed_stability.csv`。

## 单条边的示例

`cheb_example_edges.png` 显示事先指定的 Bearing1_1 留出、训练种子 42 检查点，两条输入到第一隐藏层的函数。对每个特征，以该折训练记录中的边输出标准差选择一条边：水平偏度→隐藏节点 6，垂直 RMS→隐藏节点 4。曲线只显示训练记录的 5%–95% 分位范围，减去训练加权均值输入时的边输出，因此纵轴是隐藏层内部变化量，并非 RUL 分钟。训练分位数按记录计算，训练均值采用模型训练时的轴承加权缩放统计。完整模型还包含其他输入、其他边和第二层；不能把单条边直接解释为“特征升高多少使寿命改变多少”。

## 复现

在项目根目录的 PowerShell：

```powershell
$env:PYTHONNOUSERSITE = '1'
.\.venv\Scripts\python.exe .\interpretability\cheb_explain\analyze_cheb.py --project $PWD --steps 128
.\.venv\Scripts\python.exe .\interpretability\cheb_explain\make_edges.py --project $PWD
.\.venv\Scripts\python.exe .\interpretability\cheb_explain\plot_cheb.py
```

分析需要已有 `rul_study/data/features_and_labels.csv`、`rul_study/results_merged` 下的检查点和每条记录的预测。重新运行覆盖本目录的 CSV、JSON 和图像。单变量边函数图按已记录选择规则生成；请勿把在该范围外的外推当作已验证结果。
