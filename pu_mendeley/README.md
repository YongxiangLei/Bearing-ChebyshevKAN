# PU experiment using third-party FFT arrays

This experiment adapts `run_adaptive_pu.py` printed in the supplement of TIMC-26-1736. It uses third-party arrays from Yafei Deng (2023), *Paderborn bearing dataset and PHM2009 gearbox dataset*, V1, DOI [10.17632/65d3pzth7v.1](https://doi.org/10.17632/65d3pzth7v.1). The download's SHA-256 was checked against the Mendeley API. These are not verified to be the arrays or preprocessing used by the manuscript authors. The paper describes three PU classes, whereas the supplement code and these arrays have eight classes; the experiment is therefore not an exact reproduction of the paper.

Place `FFT_0.npy`, `FFT_1.npy`, and `Label.npy` under `<project>/data/PU/bearingset/raw/`. They have shapes (8000, 1, 1500), (8000, 1, 1500), and (8000, 8), respectively. The experiment uses FFT_0 for training and FFT_1 for testing and assumes the same one-hot Label rows apply to both, following the supplement. It trains the supplement's two-layer ChebyshevKAN with Adam for 15 epochs and evaluates the target set once at the end, without choosing an epoch using target results. Seed 42 is added for reproducibility; the supplied PU script does not specify one. Source-domain scaling follows the supplement. No synthetic data are used.

In PowerShell, from the project folder, with NumPy and PyTorch available:

```powershell
$env:PYTHONNOUSERSITE = '1'
python pu_mendeley/run_pu.py
```

The script also accepts `--data-dir`, `--output-dir`, `--seed`, `--epochs`, `--threads`, and `--device`. By default it writes `pu_mendeley/results/metrics.json` and `pu_mendeley/results/model.pt`. The recorded CPU run used 2 threads and seed 42. Training took about 29.5 seconds; target accuracy was 42.6375% (3411/8000). This final-epoch test cannot establish the paper's reported 100% accuracy at epoch 2, and the array provenance limits any direct comparison.
