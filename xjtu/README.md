# XJTU-SY remaining-life experiment

This directory adapts the XJTU-SY experiment printed in the supplementary material of TIMC-26-1736, "Interpretable Bearing Fault Diagnosis via Physics-Aware Chebyshev Kolmogorov-Arnold Networks with Symbolic Equation Discovery". `adaptive_chebyshev_kan.py` is transcribed from that supplement. `run_xjtu.py` retains the experiment's features, model architecture, split, optimizer and 50 epochs; it adds explicit data and output paths, validation, a seed, device selection and saved results. It requires real data and never substitutes simulated observations.

Place the extracted data in `../data/XJTU-SY/XJTU-SY_Bearing_Datasets/`; the script requires `35Hz12kN/Bearing1_1`, `Bearing1_2`, `Bearing1_3`, and `37.5Hz11kN/Bearing2_1` under that root. Each folder must contain numbered CSV files with two vibration columns. The experiment uses every file in each bearing. It trains on Bearing1_1 and Bearing1_2, and evaluates once after the fixed 50 epochs on Bearing1_3 and Bearing2_1. Features are RMS, absolute peak, variance, kurtosis and skewness for each channel, normalized using training statistics. The target is a normalized 1-to-0 progression calculated from the first and last recordings, not remaining time in minutes; it requires observing each run to failure.

From the project root, activate a compatible Python environment with numpy, pandas and PyTorch installed, then run:

```powershell
$env:PYTHONNOUSERSITE = '1'
python xjtu/run_xjtu.py
```

Pass `--data-dir`, `--output-dir`, `--seed`, `--epochs`, or `--device` to override defaults. Saved metrics and model checkpoints go to `xjtu/results/`. Seed 42 is added by this adaptation; the supplementary XJTU script does not specify one. Numerical differences with the paper are expected and the results should not be described as an exact reproduction.
