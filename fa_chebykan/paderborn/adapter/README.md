# Paderborn 原始信号导入

`import_paderborn.py` 读取官方 Paderborn 原始 MAT、由人工核查生成的逐记录清单及显式提供的名义轴承几何。输出的 CSV 可直接供 `fa_chebykan/data.py` 和 `preprocess.py` 使用。因训练代码通过 `recording_id` 分组与评估，**本数据集将该字段设为轴承 ID**，原始 MAT 文件名和每段起止位置写入 provenance。这样训练损失、预测与验证按轴承等权，两个 MAT 不会被误当成两个独立轴承。

参数由调用者显式提供：`--geometry 8 6.75 28.55 0` 对应官方文献的 KB24/FAG 6203 样例名义几何，其他轴承可能有制造商几何差异，并非逐轴承实测。逐段转速采用与该段时间范围对应的 MAT `speed` 信号均值，单位 Hz。振动通道为 `vibration_1`，其 MAT Unit 字段为空，记为 `unknown`。

采用与类别无关、训练前固定的元数据筛查：要求 X 时间戳单调、平均采样密度相对 64 kHz 不超过 1% 偏差、最大间隔不大于 50 μs、速度通道平均采样密度与 4 kHz 偏差不超过 1%。已排除 4 条严重偏离 64 kHz 的记录，所有轴承仍有至少一条记录。保留的非均匀时间轴先线性插值到均匀 64 kHz，再使用 polyphase 低通/重采样到 12 kHz，随后分为 1.2 秒不重叠片段。线性插值及 1% 门限均为工程近似，不能保证非均匀采样伪影完全消失；需在已知精度的额外数据上验证。

```powershell
$env:PYTHONNOUSERSITE = '1'
python .\fa_chebykan\paderborn\adapter\import_paderborn.py --manifest .\fa_chebykan\paderborn\evidence\pu_bearing_record_manifest.csv --data-dir . --fa-dir .\fa_chebykan --output .\fa_chebykan\generated\paderborn_12k.csv --audit .\fa_chebykan\generated\paderborn_audit.json --geometry 8 6.75 28.55 0
python .\fa_chebykan\preprocess.py --input .\fa_chebykan\generated\paderborn_12k.csv --output .\fa_chebykan\generated\paderborn_env_3000.npz --band 1600 4800 --envelope-fs 3000
```

清单路径以项目根目录为基准，应与实际数据位置一致；原始数据的版权与使用条件由数据提供方规定，本导入器本身不附带原始数据。每个 MAT 的 SHA-256、筛选状态、采样密度、最大时间间隔、轴承和分段数保存到审计 JSON。参见 [Paderborn 官方数据论文](https://mb.uni-paderborn.de/fileadmin-mb/kat/PDF/Veroeffentlichungen/20160703_PHME16_CM_bearing.pdf)。
