# FA-ChebyKAN 与 CWRU 实验复现

本目录保存一次**探索性、单种子**实验。CWRU 数据包含在仓库 `data/CWRU/`；源数据归 CWRU Bearing Data Center 所有，来源和 SHA-256 见该目录的 `manifest.json`。本次未在官网查到明确的原始数据再分发许可证，不能把本仓库代码许可证自动套用于这些 MAT 文件；后续使用及再分发需自行向数据提供方核实。

先阅读 `CWRU_元数据核查.md` 与 `results/CWRU_实验结果报告.md`。数据的两套 CSV 清单是相同的文件和分组，只有正常记录的采样率假设不同；两个清单的路径相对本目录。真实信号单位记录为未知，MAT 缺失的两个 RPM 使用官网标称近似值，轴承 Z=9 为推导值、接触角 0° 为假设。0/1 hp 为训练、2 hp 验证、3 hp 测试；不能把不同负载视为不同物理轴承。

## 运行环境

从项目根目录运行，需 Python 3、NumPy、SciPy、Matplotlib 和 CPU 版 PyTorch。原项目的 `.venv` 可使用下列 PowerShell 命令；其他环境替换解释器路径。产生的 `fa_chebykan/generated/` 是重建缓存，不需上传。

```powershell
$env:PYTHONNOUSERSITE = '1'
$py = '.\.venv\Scripts\python.exe'
$root = '.\fa_chebykan'
$gen = "$root\generated"
New-Item -ItemType Directory -Force -Path $gen | Out-Null

# 示例仿真：4 类，各 32 条训练、8 条验证、8 条测试；不是经验证的真实设备仿真。
& $py "$root\simulator.py" simulate --output "$gen\sim.csv" --train-per-class 32 --val-per-class 8 --test-per-class 8 --seed 42
& $py "$root\preprocess.py" --input "$gen\sim.csv" --output "$gen\sim_env.npz" --band 1600 4800 --envelope-fs 3000

# 两个正常记录采样率假设；48k 原始信号先抗混叠重采样到统一的 12k，再分段。
foreach ($scenario in @('assumed48k_normal', 'assumed12k')) {
  & $py "$root\simulator.py" import-cwru --manifest "$root\cwru_manifest_$scenario.csv" --output "$gen\cwru_$scenario.csv" --duration 1.2 --overlap 0 --common-fs 12000
  & $py "$root\preprocess.py" --input "$gen\cwru_$scenario.csv" --output "$gen\cwru_${scenario}_env.npz" --band 1600 4800 --envelope-fs 3000
}

# 四模型，同样录音分组、种子、轮数和训练目标。
& $py "$root\baselines\run_benchmark.py" --simulation "$gen\sim.csv" --simulation-envelope "$gen\sim_env.npz" --real "$gen\cwru_assumed48k_normal.csv" --real-envelope "$gen\cwru_assumed48k_normal_env.npz" --output "$gen\results48" --epochs 20 --few-shot-epochs 15 --ks 1 2 3 5 10 --seed 42 --batch-size 8 --threads 2

# 原样复用前一步的仿真预训练权重，只变更正常数据采样率假设。
& $py "$root\baselines\run_benchmark.py" --simulation "$gen\sim.csv" --simulation-envelope "$gen\sim_env.npz" --real "$gen\cwru_assumed12k.csv" --real-envelope "$gen\cwru_assumed12k_env.npz" --output "$gen\results12" --pretrained-dir "$gen\results48" --epochs 20 --few-shot-epochs 15 --ks 1 2 3 5 10 --seed 42 --batch-size 8 --threads 2
```

预处理缓存与输入 CSV 的 SHA-256 绑定；修改数据、带通或采样率后须重新生成缓存。预处理与 PyTorch 训练使用独立进程，避免某些 Windows 环境中数值库的 OpenMP 运行时冲突。不同依赖或硬件可能导致浮点运算和结果变化。保存的 PyTorch `.pt` 属于模型检查点，仅加载可信来源。

## 结果和限制

`results/normal48`、`results/normal12` 分别保存训练历史、混淆矩阵、模型权重与配置。所有结果仅针对各 4 条测试录音；本次已多次查看测试结果，比较为探索性。K 指每类训练录音数；每类只有两条，因此 K=3、5、10 不可行。各模型的参数量不同，基线并非容量严格匹配或完整的先进方法比较。更多独立轴承、训练种子、支持集抽样和外部设备是后续验证所必需的。
