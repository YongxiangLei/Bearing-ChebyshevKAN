# PU 轴承留出实验

这是参考论文 ChebyshevKAN 层的独立三分类实验，不是论文数值的逐项复现。使用 Paderborn University 官方原始 MATLAB 数据的 `vibration_1` 通道，类别为正常、内圈损伤、外圈损伤；文件与标签见 `outputs/metrics.json`。PU 官方论文列出了这些轴承的损伤位置和类型：[2016 数据集说明](https://mb.uni-paderborn.de/fileadmin-mb/kat/PDF/Veroeffentlichungen/20160703_PHME16_CM_bearing.pdf)。

## 实际运行结果

- 按轴承 ID 划分，训练 / 验证 / 测试分别每类 3 / 1 / 2 个轴承，测试集 1200 个窗口。只采用 `N15_M07_F10` 工况，每个轴承两条记录，每条均匀选取 100 个长度 1024 的不重叠窗口。
- 每窗单独标准化，取 FFT 幅值后仅用训练集统计量作特征标准化；依据验证集 macro-F1 选模，第 2 轮为最佳模型。
- 测试准确率 **38.67%**，macro-F1 **0.3511**；正常、内圈、外圈 F1 分别为 **0.6250、0.4282、0**。混淆矩阵行表示真实类别，列表示预测类别，类别顺序为正常、内圈、外圈：

  ```text
  [[200, 200,   0],
   [  9, 264, 127],
   [ 31, 369,   0]]
  ```

外圈召回率为 0，说明该设置下模型无法可靠识别留出外圈轴承。仅有两个测试轴承/类、一个工况，此次结果不能代表更大轴承群体。训练集拟合程度不等于跨轴承泛化表现，也不能将这些结果直接与论文不同划分和预处理的数值比较。

模型与统计量在 `outputs/model.pt`，数据来源、轴承划分、全部评价和训练过程在 `outputs/metrics.json`。已独立重新加载模型并验证混淆矩阵一致，且检查训练、验证和测试轴承 ID 无交集。

## 运行

在项目根目录的 PowerShell 中执行，需先按项目根目录的环境说明准备 `.venv` 和 `data\PU`。该目录应含各轴承的官方 `.rar` 包；程序使用 Windows 自带的 `tar`，按需仅提取选用文件到 `data\PU\selected_mat`：

```powershell
$env:PYTHONNOUSERSITE = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
.\.venv\Scripts\python.exe .\pu\eval_pu_bearing_holdout.py --data-dir .\data\PU --device cpu --epochs 50 --patience 10 --output .\pu\outputs
```

随机种子默认为 42，完整参数可用 `--help` 查看。若使用支持 CUDA 的 PyTorch，可将 `--device cpu` 改成 `--device cuda`；硬件和库版本不同可能造成数值差异。重复运行会覆盖目标输出目录中的模型和指标文件。
