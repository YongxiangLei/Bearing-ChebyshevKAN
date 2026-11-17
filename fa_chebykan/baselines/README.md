# Baseline models and scope

`models.py` exposes `make_model(name, config)` using the same inputs and output dictionary as `FaultAnchoredChebyKAN`, so all methods must use the same recording-weighted `fit`, splits, hyperparameter budget, pretraining and classification-only fine-tuning. Models have the same auxiliary simulation heads and label order.

- `fa_cheby`: fault-frequency-anchored projections and Chebyshev KAN (the proposed model).
- `anchored_mlp`: the identical trainable anchored projection and slip parameter with an MLP classifier; this compares classifiers while retaining physical anchors.
- `spectrum_cheby`: a geometry-free fixed order-grid envelope spectrum, followed by a plain Chebyshev KAN; this is a comparison of feature constructions with the same downstream model family.
- `spectrum_mlp`: the same geometry-free spectrum followed by an MLP.

Spectral inputs are the RMS-normalized envelope FFT magnitudes interpolated at consecutive shaft orders 1 through N, where N equals the number of FA projection features. The mean shaft speed sets the grid. This is not order tracking and is intended only for stationary or near-stationary speed. Real and simulated records are processed identically. Bearing geometry and defect labels do not enter spectral features. FFT magnitude interpolation is an approximation for non-bin-centred spectral lines.

Identical hidden width does not imply identical model capacity. For `harmonics=3, sidebands=2, degree=3, hidden=16`, total parameter counts including auxiliary heads are 2425, 649, 2422, and 646, respectively, in the order listed above. Results are exploratory under this budget and should not be called a capacity-matched benchmark. Normalization and spectral coverage require independent validation; this compact set is not an exhaustive or established state-of-the-art comparison.

Run `test_models.py` with the original project directory and this directory both on `PYTHONPATH` to check outputs, gradients, fixed-order signal selection and input validation.
