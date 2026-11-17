# Paderborn 轴承级多随机种子实验

本实验使用 Paderborn University KAt DataCenter 的振动数据，类别为健康、外圈损伤和内圈损伤。来源：[数据中心](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter)及 [Lessmeier 等人的数据论文](https://mb.uni-paderborn.de/fileadmin-mb/kat/PDF/Veroeffentlichungen/20160703_PHME16_CM_bearing.pdf)。数据方标注 CC BY-NC 4.0，非商业学术使用须引用来源；商业使用请联系数据提供方。

- `results/Paderborn_多种子实验报告.md`：4 模型 × 3 阶段 × 5 种子的轴承级结果。
- `results/report.json`、`summary.json`：各轴承预测、混淆矩阵、验证历史、配置与聚合统计；`.pt` 为保存的模型。
- `evidence/`：36 个 MAT 文件的标签、轴承分组、SHA-256、测得转速、时间轴质量及源文献；另含预处理敏感性检查。
- `adapter/`：MAT 导入代码及检查；`runner/`：多种子实验代码。

固定划分对每类健康、外圈、内圈分别为 3 个训练、1 个验证、2 个测试轴承。每个轴承所有录音及片段始终属于同一集合，预测与训练损失按轴承等权。18 个轴承的 36 条 MAT 依时间轴质量规则保留 32 条、96 个 1.2 秒片段；测试集只有 6 个轴承。种子 42–46 独立初始化模型，但共用同一模拟数据集和同一轴承划分；跨种子标准差不能替代跨轴承、跨划分的不确定度。

官方论文以 KB24 FAG 6203 为例给出 Z=8、球径 6.75 mm、节径 28.55 mm、名义压力角 0°；用于其他轴承时是近似。振动 Unit 字段为空，记录为未知。时间戳插值、各轴承厂家几何差异及少量验证/测试轴承均限制结论；具体规则与 4 条排除记录见 `evidence/`。

## 本机复现

从项目根目录运行，依赖原项目 `.venv` 中的 NumPy、SciPy、PyTorch。输入为 `data/PU/selected_mat/<轴承>/` 下的原始 MAT；仓库不含这些文件。若本机仅有官方每轴承压缩包，需先按 manifest 列明的相对路径提取 1 号和 2 号记录。请遵守数据许可。

```powershell
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$py = '.\.venv\Scripts\python.exe'
$fa = '.\fa_chebykan'
$pu = "$fa\paderborn"
$gen = "$fa\generated"
New-Item -ItemType Directory -Force -Path $gen | Out-Null

# 模拟数据与 CWRU 实验相同；多种子脚本会排除模拟 ball 类。
& $py "$fa\simulator.py" simulate --output "$gen\simulation_seed42.csv" --train-per-class 32 --val-per-class 8 --test-per-class 8 --seed 42
& $py "$fa\preprocess.py" --input "$gen\simulation_seed42.csv" --output "$gen\simulation_envelope_3000.npz" --band 1600 4800 --envelope-fs 3000
& $py "$pu\adapter\import_paderborn.py" --manifest "$pu\evidence\pu_bearing_record_manifest.csv" --data-dir . --fa-dir $fa --output "$gen\paderborn_12k.csv" --audit "$gen\paderborn_audit.json" --geometry 8 6.75 28.55 0
& $py "$fa\preprocess.py" --input "$gen\paderborn_12k.csv" --output "$gen\paderborn_env_3000.npz" --band 1600 4800 --envelope-fs 3000
& $py "$pu\runner\run_pu_multiseed.py" --project $fa --simulation "$gen\simulation_seed42.csv" --simulation-envelope "$gen\simulation_envelope_3000.npz" --target "$gen\paderborn_12k.csv" --target-envelope "$gen\paderborn_env_3000.npz" --output "$gen\paderborn_multiseed" --seeds 42 43 44 45 46 --pretrain-epochs 20 --epochs 20 --batch-size 8 --threads 2
& $py "$pu\summarize_paderborn.py" --results "$gen\paderborn_multiseed"
```

预处理缓存与输入文件 SHA-256 绑定，应在更改输入或频带后重新生成。不同 NumPy/SciPy/PyTorch 版本、硬件或浮点运算可能带来结果差异；只加载可信来源的 PyTorch 检查点。实际使用的本地派生 CSV/NPZ 可保留在 `generated/`，不应误认为新的独立轴承数据。
