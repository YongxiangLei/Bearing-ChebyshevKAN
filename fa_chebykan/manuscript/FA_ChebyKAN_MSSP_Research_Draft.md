# Fault-frequency-anchored Chebyshev networks for bearing fault diagnosis: a simulation-to-real research protocol

**Research and methods draft — not a submission-ready results paper.** Author details, final methods, experimental observations, citations, and declarations require author review. All diagnostic results described below are planned unless explicitly identified as completed and linked to an independently verified experiment record.

## Abstract

Reliable vibration-based bearing diagnosis under changes in machinery, operating conditions and damage morphology remains challenging when labelled industrial recordings are scarce. We propose to investigate a fault-frequency-anchored Chebyshev feature layer followed by a Chebyshev Kolmogorov–Arnold network (KAN) classifier. Bearing geometry and shaft phase determine candidate characteristic orders, their harmonics and selected modulation sidebands. Sine and cosine projections of a vibration envelope onto these atoms provide interpretable order-specific features, while a bounded trainable slip parameter allows small deviations from nominal kinematics. The proposed training protocol pretrains on simulated signals and evaluates frozen zero-shot transfer and recording-disjoint few-shot adaptation on suitable public data, followed by an industrial spindle study if independently labelled recordings become available. Planned controls include a matched ordinary Chebyshev KAN, an MLP, a B-spline KAN, a one-dimensional convolutional network, a spectral-feature baseline and a physics-inspired multiwavelet KAN where a faithful implementation is available. The study will measure classification performance, false alarms, per-class and per-recording variability, and the sensitivity of the characteristic-order features to measurement and parameter errors. **No transfer or diagnostic performance is reported in this protocol.** The proposed architecture must be tested against conventional envelope-spectrum projections, and its physical interpretation depends on geometry, rotational-speed measurement and fault-label validity.

Keywords: rolling-element bearing; fault diagnosis; Chebyshev polynomial; fault-frequency anchoring; simulation-to-real transfer; interpretable machine learning

## 1. Introduction and research questions

Bearing defects can cause approximately periodic excitation of structural resonances. Envelope analysis seeks to reveal this periodicity, but a nominal characteristic frequency can shift or become indistinct because of slip, speed variability, load, transmission paths and other phenomena. A candidate diagnostic model should therefore be judged not only by aggregate classification accuracy, but also by its behaviour on independent physical recordings, changes in operating conditions and deviations from assumed kinematics.

KANs use learnable univariate edge functions [1]. Interpretable KANs have already been applied to fault and severity diagnosis [2], and physics-inspired multiwavelet KANs have been proposed for bearing diagnosis [3]. Their existence precludes claiming that KAN-based or physics-inspired bearing diagnosis is new by itself. The present research question is narrower: can embedding candidate defect orders in the construction of features, with bounded slip and a downstream Chebyshev classifier, improve recording-disjoint simulation-to-real bearing diagnosis and provide mechanically meaningful diagnostics beyond conventional spectral analysis?

Three hypotheses will be evaluated rather than assumed: (H1) defect-order anchoring improves a matched plain Chebyshev classifier on recording-disjoint target data; (H2) simulated pretraining improves zero-shot or few-shot performance relative to an appropriate no-pretraining control; and (H3) order-specific responses agree with independent mechanical evidence sufficiently often to be useful as diagnostic clues. A conventional envelope-spectrum baseline is essential: Chebyshev–harmonic identities make these features closely related to sinusoidal projections, and mathematical equivalence alone does not establish a novel physical mechanism or a performance advantage.

The earlier project manuscript concerns minute-scale remaining-useful-life (RUL) regression for XJTU-SY and remains a separate exploratory study. Its MAE values, RUL labels, spline visualisations and cross-validation outcomes are not results for the proposed fault classifier.

## 2. Data, labels and measurement interface

### 2.1 Measurement interface

For each acquisition, preserve the raw vibration signal `x`, sampling frequency `fs` in Hz, shaft-speed series or documented nominal speed in Hz, bearing identifier and geometry (rolling-element count `Z`, element diameter `d`, pitch diameter `D`, contact angle `alpha`), dataset and operating condition. Retain units and sensor/channel location. Fault type, severity and measurement or teardown provenance should be recorded separately when known. Simulation records additionally contain the parameters used to generate the impact amplitude and resonance decay. A unique physical bearing, rig, run and recording identifier must be maintained wherever documented.

The candidate classification vocabulary is normal, outer-race, inner-race and rolling-element fault. Dataset-specific damaged states must be mapped to this vocabulary only when the mapping is supported by primary documentation. Mixed defects, cage defects, unknown conditions and unknown onset times must not be silently assigned to a single class. A subset with verified labels can be analysed separately, with exclusions and coverage reported. A class absent from a target set is not an evaluated class.

### 2.2 Simulated and real data

The proposed simulation data combine impact-like excitation and a decaying resonance, randomised over physically plausible geometry, speed, load-related modulation, resonance, noise and transmission characteristics. Record the generator version, random seeds, parameters, class priors, geometry units, shaft-speed trajectories and splits. Simulated train, validation and test sets must be independently generated, with grouped latent configurations or parameter regions kept apart. The amplitude and decay regression targets are available only for simulation; estimates cannot be validated against real physical targets that have not been measured.

The proposed real datasets are CWRU [4], Paderborn [5] and XJTU-SY [6]. Dataset access alone does not establish suitable labels, geometry or a feasible independent split. CWRU data include both drive-end and fan-end measurements; the tested bearing, sensor, sampling rate, fault size and operating condition must be stated [4]. Paderborn damage and licence terms require examination before selection or redistribution [5]. XJTU-SY contains full run-to-failure bearings rather than independently labelled fault types for every minute [6]. A teardown label must not automatically be attached to the entire run; early operation is not automatically verified normal, and onset, cage and mixed-fault cases require separate handling. Report how many independent bearings support each retained class. If a target class has too few independent labelled bearings, report that the requested evaluation is not estimable rather than splitting windows of the same bearing to fill the gap.

CNC spindle recordings are an intended external case, contingent on delivery of data, geometry, acquisition metadata, speed and independent defect labels. Without labels they support only descriptive time trends and cannot establish accuracy, false-alarm rate or a confirmed physical fault. No industrial result is claimed here.

### 2.3 Segmentation and independence

Select `N` and overlap before testing, subject to sample rate, duration, operating speed and adequate phase coverage; a target of at least ten cage revolutions is checked per segment where feasible. When a record is shorter, report the limitation and do not pad it with neighbouring held-out data. The XJTU-SY record duration is approximately 1.28 s [6], so feasibility depends on speed and bearing geometry. Any resampling and anti-alias filtering must be documented. Split by physical bearing or independent acquisition before segmentation, choosing the stronger independent unit that the dataset documentation supports; all overlapping segments from a recording remain in one partition. In run-to-failure data, all observations of a test bearing stay out of training and validation. If operating-condition transfer is claimed, hold out that condition, rather than merely holding out segments at known conditions.

## 3. Proposed method

### 3.1 Envelope and shaft phase

Given a band-pass filtered signal `x_bp(t)`, define the centred Hilbert envelope

$$e(t)=\left|x_{bp}(t)+\mathrm{i}\,\mathcal{H}\{x_{bp}(t)\}\right|-\operatorname{mean}\left|x_{bp}+\mathrm{i}\,\mathcal{H}\{x_{bp}\}\right|.$$

The resonance band can be fixed using training data or determined by a fully specified per-record unsupervised rule, such as a kurtogram-style procedure. The rule, filter transients, admissible bands and all hyperparameters must be determined without using held-out labels. The identical preprocessing rule must be applied to simulation and real signals, without access to simulator-only parameters. Define shaft phase as

$$\varphi_r(t)=2\pi\int_0^t f_r(\tau)\,d\tau,$$

where `f_r` is measured or nominal shaft speed. Where only a nominal speed exists, the resulting phase is an approximation, and sensitivity to speed error must be reported. Varying speed can require order tracking or a direct time-varying phase computation; stationarity is not assumed without checking.

### 3.2 Kinematic anchors

Under the usual fixed-outer-ring, ideal rolling assumptions, define `rho=(d/D) cos(alpha)`. The candidate characteristic orders are

$$\kappa_{\rm OR}=\frac Z2(1-\rho),\quad \kappa_{\rm IR}=\frac Z2(1+\rho),\quad \kappa_{\rm B}=\frac D{2d}(1-\rho^2),\quad \kappa_{\rm FTF}=\frac12(1-\rho).$$

For fault family `c`, a bounded global slip correction is `epsilon_c=epsilon_max tanh(eta_c)` and `theta_c=kappa_c(1+epsilon_c) varphi_r`. The factor `epsilon_max`, initialisation, trained versus frozen status, optimisation boundaries, and source of geometry must be specified. These formulas are candidate kinematic orders, not proof that a measured spectral line arises from a particular defect. Ball-spin harmonics such as twice the nominal ball-spin frequency can be relevant and are included through the harmonic index rather than assumed to dominate.

### 3.3 Anchored Chebyshev–harmonic features

For positive harmonic index `n` and sideband index `m`, define `C_n(theta)=T_n(cos theta)=cos(n theta)` and `S_n(theta)=sin(theta) U_{n-1}(cos theta)=sin(n theta)`. Use `S_0=0` and `C_0=1`. These functions can be evaluated by Chebyshev recurrence. For a signed sideband order `m` in the set `{-M_c,...,M_c}`, define

$$\psi_{c,n,m}(t)=n\theta_c(t)+m\phi_{m(c)}(t),$$

where the modulating phase is zero for the proposed outer-race feature set, shaft phase for inner-race features and cage phase `kappa_FTF varphi_r` for rolling-element features. Set `M_OR=0`. Using signed `m` counts the central line `m=0` once; it avoids duplicating identical plus/minus zero-sideband features. With `n=1,...,N_h`, the nominal feature count is `sum_c N_h (2 M_c+1)`, subject to explicit handling of coincident frequencies across different families.

For a centred segment with `N` points, compute

$$a_{c,n,m}=\frac2N\sum_{k=1}^N e_k\cos\psi_{c,n,m,k},\qquad b_{c,n,m}=\frac2N\sum_{k=1}^N e_k\sin\psi_{c,n,m,k},$$

$$A_{c,n,m}=\sqrt{a_{c,n,m}^2+b_{c,n,m}^2},\qquad \widetilde A_{c,n,m}=\frac{A_{c,n,m}}{\max\{\sqrt{(2/N)\sum_k e_k^2},\delta\}}.$$

The small positive `delta` and zero-energy behaviour must be fixed and reported. The two quadratures reduce sensitivity to initial phase under suitable sampling and adequate coverage, but exact phase invariance is not guaranteed in short, nonstationary, nonorthogonal or windowed data. The normalisation makes the features dimensionless; it does not make each a probability or an exact fraction of envelope energy. A proposed family score formed by summing squared normalised amplitudes is only a descriptive spectral indicator. It is not generally bounded by one, because finite-window bases can be nonorthogonal and overlapping or coincident lines can double-count energy. Spectral leakage, coherence and the condition number should be inspected. If a reconstruction or explained-energy fraction is needed, fit a stated joint least-squares projection to a deduplicated design matrix and separately assess its numerical stability and out-of-sample residual; do not attribute shared-line energy uniquely to a defect without additional assumptions.

The projections above are deterministic functions of the data and anchored atoms, not learned outer coefficients. Slip parameters and downstream network weights can be learned. The Chebyshev identities demonstrate a harmonic representation; by themselves they do not prove that the projection is a novel KAN layer or that independent one-dimensional projections solve the multivariate least-squares problem. An equivalent Fourier-order baseline will test whether the architecture contributes beyond conventional spectral features.

### 3.4 Classifier and objectives

Set the classifier input `z` to the vector of `log(1+widetilde A)` values and use layers

$$z^{(\ell+1)}_j=\sum_i\sum_{k=0}^{K} w_{ij,k}^{(\ell)}T_k(\tanh z_i^{(\ell)}).$$

Softmax converts final logits to class scores. These scores are not automatically calibrated fault probabilities. Choose polynomial degree, width, training budget, feature scaling and early stopping using only the appropriate training/validation groups, and report the selected values for each split. For simulation-only pretraining, a proposed auxiliary objective adds weighted mean squared errors for simulated impact-amplitude and decay targets to class cross-entropy. However, away from the numerical floor, the normalised spectral input is invariant to global rescaling, so absolute impact amplitude is not generally identifiable from that input alone. Accordingly, an amplitude head using only `z` is an experimental auxiliary task and must not be claimed to recover absolute impact amplitude. A separately specified variant could provide a calibrated absolute envelope-scale feature to the auxiliary head; sensor gain, transfer path and amplitude confounding still prevent assuming recovery on real machines. Both variants require ablation against classification-only pretraining. Decay is also not guaranteed identifiable from a small set of noisy envelope projections. When fine-tuning on real data, use classification loss only unless independent real physical targets exist. State the simulator target definitions, scaling and loss weights, the trainable layers and checkpoints; no real-domain target is inferred from simulation labels.

## 4. Planned experimental protocol

1. **Simulation validation.** Verify signal generator and measurement interface using deterministic cases with known frequencies and recorded parameters. Quantify coverage and sensitivity to noise, noninteger cycles, speed changes, sidebands, geometry error and small energy. Report simulator-test performance separately from all real outcomes.
2. **Frozen zero-shot evaluation.** Freeze preprocessing configuration, simulated checkpoint, slip and classifier before seeing test labels. Specify how source and target class vocabularies match. Compare against a clearly defined conventional source-trained baseline under identical conditions. If feature-space coverage is inspected, use only a designated unlabeled development partition and fixed training-derived distance scaling/threshold; repeatedly widening simulator ranges after inspecting held-out test data invalidates a strictly untouched final test. Record all feedback and keep a fresh final test group.
3. **Recording-disjoint few-shot evaluation.** Evaluate `K=1,3,5,10` independently labelled support recordings or physical units per class, never `K` correlated segments unless explicitly reported as a weaker separate protocol. Query/test units remain disjoint. If independent support units are insufficient, mark the setting unavailable. Repeat across support selections and model seeds. Compare simulation pretraining with training from scratch using the same support, validation budget and architecture. For `K=1`, avoid using the same record both for tuning and final evaluation; use pre-specified settings or separate validation units.
4. **Ablation and baseline evaluation.** Compare anchored and plain Chebyshev KANs on matched data and training budgets; a conventional envelope/order-spectrum classifier with matched classifier capacity; MLP, B-spline KAN, one-dimensional CNN and a faithful physics-inspired multiwavelet KAN [3] where code and replication permit. Report parameter count, training budget, chosen hyperparameters and failures. Examine no pretraining, fixed versus learned slip, sidebands on/off, different harmonic counts, and auxiliary heads on/off. Physics anchoring and pretraining are separate factors; do not attribute their joint result to either component without ablation.
5. **Interpretability and reliability.** On independently labelled recordings, compare anchored-order responses with independently determined fault orders while acknowledging overlapping lines and load/speed effects. Report class-wise precision, recall, macro-F1, balanced accuracy, confusion matrices, and false positives for verified normal recordings, alongside sample/recording/bearing counts. Summarise variability and uncertainty at the independent-unit level rather than treating overlapping windows as independent observations. Test calibration and out-of-distribution behaviour only if the study is designed to support them.
6. **Industrial spindle case.** Evaluate only after an auditable data, geometry and label inventory. With labels, use held-out operating periods or machines as available and report the number of independent units. Without labels, limit conclusions to exploratory signals, feature distributions and temporal trends; do not present diagnostic accuracy or fault identification as validated.

## 5. Results and discussion placeholders

No FA-ChebyKAN diagnostic experiment is reported here. Replace the following placeholders only after the experiment configuration, code version and independent test records have been checked.

- **Dataset and split inventory:** `[number of bearings/recordings and classes; excluded mixed/unknown cases; geometry and speed provenance]`.
- **Simulation and coverage:** `[generator version; parameter ranges; disjoint validation; source-only distance threshold; development/test separation]`.
- **Zero-shot and few-shot results:** `[per-dataset and per-class metrics, independent units, K support definition, seeds and uncertainty]`.
- **Matched ablations and baselines:** `[common splits and budgets; realised baselines and any failures; comparison with conventional spectral projection]`.
- **Physical interpretability:** `[order-alignment, collision/sensitivity analysis and independent mechanical evidence; no causal claim without support]`.
- **Spindle case:** `[data and ground-truth provenance; results only if available]`.

The key limitations to discuss are simulation mismatch, weak or mixed labels, small numbers of independent bearings, unmeasured shaft-speed variation, uncertain geometry and contact angle, frequency collisions, short records, imperfect phase invariance and leakage caused by treating correlated segments as independent. Selection of resonance bands and simulator coverage feedback can also leak information if performed against final test labels or tuned repeatedly to the final test set. Any model advantage must be interpreted against strong conventional signal-processing baselines and supported by the actual data.

## 6. Data, code and declarations

`[Insert versioned data and code locations, access constraints, licence compliance, and reproducibility instructions after experiments.]` Paderborn's stated licence and citation requirements must be checked for the intended use [5]. `[Complete author contributions, funding, competing interests, and other required declarations.]` OpenAI Codex assisted preparation of this draft; authors must verify the extent of AI use, tool/version and human review, and prepare disclosures according to current publisher policy. The source model specification supplied by the authors should be acknowledged appropriately after authors confirm its provenance and reuse rights.

## References

[1] Z. Liu et al., KAN: Kolmogorov–Arnold networks, International Conference on Learning Representations, 2025. https://arxiv.org/abs/2404.19756.

[2] S. Rigas, M. Papachristou, I. Sotiropoulos, G. Alexandridis, Explainable fault classification and severity diagnosis in rotating machinery using Kolmogorov–Arnold networks, Entropy 27 (2025) 403. https://doi.org/10.3390/e27040403.

[3] J. Li, Y. Feng, J. Chen, S. He, J. Xie, Physics-inspired multiwavelet KAN for interpretable bearing fault diagnosis under harsh operating conditions, Mechanical Systems and Signal Processing 258 (2026) 114643. https://doi.org/10.1016/j.ymssp.2026.114643.

[4] Case Western Reserve University Bearing Data Center, data files and bearing information. https://engineering.case.edu/bearingdatacenter/download-data-file; https://engineering.case.edu/bearingdatacenter/bearing-information (accessed 17 November 2026).

[5] C. Lessmeier et al., KAt DataCenter, Chair of Design and Drive Technology, Paderborn University. https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter (accessed 17 November 2026).

[6] B. Wang, Y. Lei, N. Li, N. Li, A hybrid prognostics approach for estimating remaining useful life of rolling element bearings, IEEE Transactions on Reliability 69 (2020) 401–412. https://doi.org/10.1109/TR.2018.2882682; dataset information: https://biaowang.tech/xjtu-sy-bearing-datasets/.
