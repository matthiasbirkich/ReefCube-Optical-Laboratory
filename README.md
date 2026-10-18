# ReefCube Optical Laboratory

## Google Colab

The complete demonstration notebook can be opened and executed directly
in Google Colab without a local Python installation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matthiasbirkich/ReefCube-Optical-Laboratory/blob/main/notebooks/ReefCube_Optical_Laboratory_complete_demo.ipynb)

The notebook automatically clones this repository and installs the
ReefCube package in the Colab runtime.

## An Open-Source Python Toolkit for Environmental Spectroscopy and Sensor Calibration

**ReefCube Optical Laboratory** is an open-source Python package for the acquisition, processing, calibration, visualization, and analysis of optical environmental sensor data.

The project is designed for scientific applications in marine and freshwater research, environmental monitoring, laboratory spectroscopy, and sensor development. Although originally developed for the ReefCube environmental logger, the software is intentionally modular and can be used with many different optical sensors.

---

## Key Features

- Spectral reconstruction from multispectral sensors (currently AS7343)
- Environmental sensor data analysis
- Spectral calibration workflows
- PPFD estimation
- Instrument comparison and validation
- Publication-quality visualizations
- CSV and JSONL import/export
- Google Colab support
- Jupyter Notebook demonstrations
- Modular and fully documented Python package

---

## Repository Structure

```text
ReefCube-Optical-Laboratory/

├── reefcube/
├── notebooks/
├── examples/
├── docs/
├── test/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── LICENSE
├── CHANGELOG.md
└── CITATION.cff
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/matthiasbirkich/ReefCube-Optical-Laboratory.git
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

## Google Colab

The repository contains demonstration notebooks that run directly in Google Colab.

The notebook automatically clones the repository and imports the local ReefCube package.

---

## Scientific Applications

Typical applications include

- Coral reef monitoring
- Mangrove ecosystem research
- Water quality assessment
- Environmental spectroscopy
- Optical sensor development
- Instrument calibration
- Citizen Science
- Educational laboratory courses

---

## Development Status

Current status

- Core package under active development
- Demonstration notebooks available
- Publication-quality visualization implemented
- Additional sensor modules under development

---

## Calibration Steps & Status

AS7343 Spectral Reconstruction — Proof of Concept and Transfer to ReefCube

Current development and validation status

The ReefCube Optical Laboratory project distinguishes between an experimental proof-of-concept system and the final autonomous underwater logger.

- Stage 1 — Reference spectroradiometry

A diffuser-calibrated Torch Bearer Y2 spectrometer is used to acquire reference spectral irradiance measurements in W m⁻² nm⁻¹. The instrument covers approximately 340–1050 nm and has a specified spectral resolution of 5–7 nm.

Reference measurements include a compact fluorescent lamp (CFL) with characteristic mercury and phosphor emission features, as well as natural daylight.

- Prelimenary result on 20261018

ReefCube Optical Laboratory — AS7343–Y2 Feasibility Assessment (Stage 1)

Date: 2026-10-18
Status: Exploratory proof of concept; NOT a validated absolute irradiance calibration.
Instrument roles: Torch Bearer Y2 = diffuser-calibrated reference spectrometer; Yoctopuce AS7343 + VirtualHub v2 + Python v4 = intermediate proof-of-concept platform. ReefCube XIAO nRF52840 Sense + AS7343 + BMA400 firmware v46 is a separate, not yet validated hardware system; it does not use Yoctopuce software or calibration constants.

Source data

• 20261018_133515_Y2_CFL_data.zip, spectrum.csv: observation 3, NORMAL, UTC 2026-10-18 11:35:17.077, exposure 115.5 ms; 340–1000 nm at 1 nm export steps, manufacturer-stated spectral resolution 5–7 nm.
• 20261018_CFL_AS7343.zip, *_kanaele.csv and *_metadaten.json: measurement 2026-10-18 13:34:10.694 +02:00 (about 66.4 seconds before Y2); gain exponent 7; integration 500 ms; diffuser made of PTFE tape; 8 F1–F8 channels used for inverse model; dark_raw=0 across exported channels; calibration=null; zero saturation flags.
• yocto_spektrum_v4.py contains approximate typical AS7343 channel-response shapes digitized from ams OSRAM data, credited to photobiologySensors (check GPL-2.0-or-later licensing before copying these embedded response data into ReefCube).

Method — Y2-forward projection against AS7343 raw-response channels

The measured reference spectrum E(lambda) was integrated over each of the 8 nominal AS7343 spectral response curves R_i(lambda) as an independent forward-model check:

p_i = integral[E(lambda) * R_i(lambda) * (lambda/550 nm)] d(lambda)

The optional photon weighting is an assumption requiring verification of the response-curve basis and ADC signal units; no claim of full radiometric traceability is made. Only one single global least-squares scale factor a was allowed: a = sum(s_i p_i)/sum(p_i^2). The measured AS7343 signal s_i is corrected_raw, not a physical irradiance.

Important: This compares channel-integrated responses and is not a comparison of narrow CFL Hg lines to nominal AS7343 central wavelengths. Narrow-line reconstruction with the AS7343 is fundamentally underdetermined.

Preliminary results

• Y2 relative response versus AS7343 across F1–F8: cosine similarity 0.952 with the assumed photon-weighted response model.
• Global-scale relative Euclidean residual ||s-a p||/||s|| = 0.306 (~30.6%). This is not an independently validated calibration error, RMSE in irradiance units, or proof of linearity.
• Range of measured-to-predicted channel ratios: 0.57–4.15. A single scale does not align all channels.
• The AS7343’s own v4 nonnegative B-spline fit to its own 8 normalized channels reported fit_residual ~0.071; it is not evidence that the AS7343 agrees with Y2.
• Y2 integrated CFL irradiance over 400–700 nm from this export is 0.391228 W/m². This is a Y2 reference quantity, not a measurement by the AS7343.

Initial interpretation

The AS7343 clearly exhibits differentiated wavelength-dependent response to CFL illumination (largest F6 in F1–F8, with additional FY and FXL broadband-type channels). The reference-based forward-model prediction captures the broad pattern but channel-specific differences are substantial. Potential causes include the digitized typical response curves, response-unit convention, PTFE diffuser’s spectrally varying transmission, illumination geometry, channel-specific scaling, finite count resolution (F1=4 counts), and timing differences between the two acquisitions.

Current conclusion: The measurement confirms usable nonzero spectrally differentiated AS7343 response and permits a testable forward-model comparison. It does not yet establish that absolute spectral irradiance can be reconstructed accurately from an AS7343. The intermediate Yoctopuce findings must not be presented as validation of firmware v46 or ReefCube hardware.

Required next steps / predefined acceptance testing

1. Record at least 3 paired measurements per light source with the calibrated Y2 diffuser and unchanged AS7343 PTFE optical assembly, fixed parallel normals and comparable fields of view; use CFL, daylight and at least one held-out spectral class.
2. Record and retain original AS7343 raw counts, exposure gain/time, time stamps, sensor temperature when available, saturation and actual dark measurement at matching gain/time; mark channels near digital quantization/noise floor.
3. Obtain/document Y2 radiometric calibration certificate/uncertainty, diffuser angular response, and AS7343 PTFE diffuser transmission/cosine response.
4. Evaluate typical-response forward model with held-out datasets and independently quantified bias, RMSE and band-integrated irradiance/PPFD, including uncertainty. Avoid tuning one correction per channel from a single spectrum and calling that validation.
5. Compare Gaussian signature plotting, v4 nonnegative B-spline inversion, and regularized nonnegative inversion on unseen spectra; never claim 1 nm optical resolution from an AS7343.
6. Independently validate the ReefCube v46 acquisition chain, raw channel ordering and gain/exposure normalization after the Yoctopuce proof of concept.

Reproducibility and licence note

Input CSVs are preserved in the original ZIP files. channel_comparison.csv reports all 8 channels and predictions. The demonstration script reads the Yoctopuce Python v4 script only to decode the published typical sensitivity profiles; because that source declares GPL-2.0-or-later for the embedded data, establish license compatibility/attribution before integration into the ReefCube repository.

- Stage 2 — AS7343 proof of concept using Yoctopuce

A separate AS7343-based measurement system from Yoctopuce, operated through VirtualHub v2 and a dedicated Python analysis script (v4), is used to investigate whether the AS7343 multispectral sensor can support physically meaningful spectral reconstruction and absolute irradiance estimation.

The experimental reconstruction uses non-negative cubic B-splines with regularization and approximate spectral sensitivity functions.

The system currently provides relative spectral reconstructions. Absolute radiometric calibration against the Y2 reference instrument remains under investigation.

This stage is intended to evaluate the feasibility and limitations of the AS7343 sensor principle. It does not constitute a completed calibration of the ReefCube logger.

- Stage 3 — Implementation in ReefCube

The final ReefCube underwater logger uses a Seeed XIAO nRF52840 Sense, an AS7343 spectral sensor, an external BMA400 accelerometer, and custom firmware v46.

The ReefCube logger does not use Yoctopuce hardware, VirtualHub, or the Yoctopuce Python library.

Firmware v46 acquires and stores AS7343 raw-channel measurements together with exposure parameters and quality information. Spectral reconstruction, radiometric calibration, and the calculation of derived optical quantities are performed separately in the ReefCube Optical Laboratory Python software.

The hardware-dependent measurement and calibration performance of this configuration has not yet been experimentally validated.

Calibration transfer and validation

The methods developed using the Yoctopuce proof-of-concept system may be transferred to ReefCube, but the numerical calibration coefficients must not be assumed to be transferable.

The ReefCube AS7343 requires its own characterization of dark response, exposure normalization, channel sensitivity, optical diffuser, housing window, angular response, and instrument-specific radiometric factors.

Only after comparison with independent Y2 reference measurements can ReefCube-derived spectral irradiance values be reported as experimentally calibrated.

Scientific objective

The objective is to determine whether a low-cost AS7343 multispectral sensor can provide useful estimates of absolute spectral irradiance and derived ecological light quantities, including PAR and PPFD, within clearly defined spectral, angular, and uncertainty limits.

Spectral reconstruction must not be interpreted as achieving the wavelength resolution of a conventional spectrometer.

---

## Project Partners
Developed by LZMT (Germany) in collaboration with ICM-CSIC and IAAC (Spain) to advance open-source optical tools for marine research, environmental monitoring, and Citizen Science.

**Leibniz Centre for Tropical Marine Research (LZMT)**  
Bremen, Germany

**Institute of Marine Sciences (ICM-CSIC)**  
Barcelona, Spain

**Institute for Advanced Architecture of Catalonia (IAAC)**  
Barcelona, Spain

---

## License

This project will be released under the MIT License.

---

## Citation

Citation information will be provided after the first public release.

---

## Contact

Matthias Birkicht

Leibniz Centre for Tropical Marine Research (LZMT)

Bremen, Germany
