import numpy as np

from reefcube.ppfd import (
    PPFDCalibration,
    estimate_measurement_ppfd,
    measurement_relative_ppfd_index,
    photon_energy_joule,
    spectral_irradiance_to_ppfd,
)
from reefcube.sensors import SimulationSensor


# ------------------------------------------------------------
# 1. Physical PPFD test using calibrated spectral irradiance
# ------------------------------------------------------------

wavelengths_nm = np.arange(
    380.0,
    901.0,
    1.0,
)

spectral_irradiance = np.zeros_like(
    wavelengths_nm,
)

par_mask = (
    (wavelengths_nm >= 400.0)
    & (wavelengths_nm <= 700.0)
)

# Constant example irradiance:
# 1 W m^-2 nm^-1 throughout the PAR interval.
spectral_irradiance[par_mask] = 1.0

physical_ppfd = spectral_irradiance_to_ppfd(
    wavelengths_nm,
    spectral_irradiance,
)

print("Physical calculation")
print("--------------------")
print(
    f"Photon energy at 550 nm: "
    f"{photon_energy_joule(550.0):.6e} J"
)
print(
    f"PPFD for the example spectrum: "
    f"{physical_ppfd:.2f} µmol m⁻² s⁻¹"
)


# ------------------------------------------------------------
# 2. Relative AS7343 PPFD index
# ------------------------------------------------------------

sensor = SimulationSensor(
    mode="underwater",
    seed=42,
)

measurement = sensor.acquire()

relative_index = measurement_relative_ppfd_index(
    measurement
)

print()
print("Uncalibrated Reef Cube calculation")
print("----------------------------------")
print(
    f"Relative PPFD index: "
    f"{relative_index:.2f} arbitrary units"
)


# ------------------------------------------------------------
# 3. Demonstration calibration
# ------------------------------------------------------------
#
# These coefficients are examples only. They are not valid Reef Cube
# calibration coefficients.

example_calibration = PPFDCalibration(
    slope=1.0e-7,
    intercept=0.0,
    minimum_index=0.0,
    maximum_index=1.0e10,
    reference_name="Example reference sensor",
)

estimated_ppfd = estimate_measurement_ppfd(
    measurement,
    example_calibration,
)

print()
print("Example empirical conversion")
print("----------------------------")
print(
    f"Estimated PPFD: "
    f"{estimated_ppfd:.2f} µmol m⁻² s⁻¹"
)
print(
    "Warning: the calibration coefficients above are "
    "demonstration values only."
)