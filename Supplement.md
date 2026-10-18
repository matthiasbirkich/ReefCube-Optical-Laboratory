Supplement — Interpreting the AS7343 weighted response sum

Scope: Yoctopuce AS7343 proof of concept versus Torch Bearer Y2 CFL reference, not ReefCube firmware v46 validation.

What the two submitted figures represent

1. Typical channel responsivities and weighted sum: colored curves F1–F8 represent wavelength-dependent sensor response functions. The black sum represents a combination of these responses, normalized for display. These maxima exist primarily because of the sensor’s filter transmission and weighting. Their coincidence with an illumination peak cannot alone demonstrate recovery of the incident CFL spectrum.
2. Y2 CFL spectral irradiance: measures the incident light spectrum, with narrow emissions near 436, 546, and 612 nm plus phosphor bands. Its vertical axis has physical irradiance units (W m^-2 nm^-1), unlike the normalized response-sum graph.

The response sum can diagnose spectral sensitivity and illustrate coverage/gaps, but it is not an estimate of the incident irradiance spectrum unless derived from an explicitly defined inverse model and measurement data.

Physical forward model

For AS7343 channel i, with reference irradiance E(lambda) and spectral sensitivity R_i(lambda):

c_i = d_i + a_i * t * g * integral(E(lambda) R_i(lambda) T_diffuser(lambda) d lambda) + noise_i

where d_i is dark offset, a_i is instrumental responsivity/scaling, t is integration time, g is gain (if linear), and T_diffuser includes the optical chain. The scalar forward predictions should be compared with observed normalized counts before any inversion.

Crucial distinction: summing normalized filter response curves, sum_i w_i R_i(lambda), is an effective instrument spectral weighting function. It is not E(lambda) and has no justification to reproduce sharp Hg emission lines. A model prediction of the observed channel counts is obtained by integrating E(lambda) R_i(lambda) separately for each channel.

Why narrow peaks seem worse with larger counts

Higher signal amplitude does not intrinsically broaden the AS7343 spectral filters. Candidate explanations for the observed output include (a) regularization or spline basis-width choices, (b) calibration factors optimized for broadband channel integrals, (c) normalization or unit-conversion errors, and (d) gain/integration nonlinearity or saturation. Check raw count-vs-intensity and exposure-normalized counts independently before attributing this to physical resolution loss. In particular, do not compare two differently smoothed, rescaled plots by visual peak sharpness alone.

Proposed reproducible extension

• Plot A: normalized eight measured AS7343 counts with uncertainty bars (n=4), alongside Y2-predicted channel integrals using the same response curves. Use separate display normalization, record gain and exposure, and retain signed residuals.
• Plot B: the eight typical response functions and their weighted sum as a diagnostic (explicitly labeled instrument response, not source spectrum).
• Plot C: Y2 measured CFL irradiance versus B-spline reconstruction and, for transparency, the uncorrected baseline. Display consistent physical units and smoothing parameters; preserve Y2’s narrow reference lines rather than implying the AS7343 can resolve them.
• Plot D: compare band-integrated irradiance and PPFD instead of judging success solely by line-shape RMSE; include CV, biases, and uncertainty intervals.
• Sensitivity study: repeat reconstructions across a pre-specified grid of spline knots/regularization strengths, report the sensitivity of PPFD and spectral peak width to these choices, and mark results as in-sample.
• Independent validation: apply CFL-derived channel factors without refitting to at least several temporally paired Y2/AS7343 broadband-sunlight measurements. Keep one or more light conditions completely held out.

Interpretation and reporting constraints

The response-sum pattern is useful for confirming wavelength coverage and exposing overlapping channels; it is not a direct reconstructed spectrum, and matching-looking maxima alone do not prove radiometric accuracy. The four AS7343 files and five non-synchronous Y2 spectra establish short-term repeatability, not transferable absolute calibration. The ReefCube logger uses a different AS7343 implementation and needs its own radiometric and angular characterization before any coefficients can be transferred.