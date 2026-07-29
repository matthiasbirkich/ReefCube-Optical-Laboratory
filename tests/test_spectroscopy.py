from reefcube.sensors import SimulationSensor
from reefcube.spectroscopy import (
    integrate_band,
    measurement_to_arrays,
    normalize_spectrum,
    peak_wavelength,
    reconstruct_as7343_spectrum,
    spectral_centroid,
)


sensor = SimulationSensor(
    mode="underwater",
    seed=42,
)

measurement = sensor.acquire()

channel_wavelengths, channel_signal = (
    measurement_to_arrays(measurement)
)

normalized_signal = normalize_spectrum(
    channel_signal,
    mode="max",
)

reconstructed_wavelengths, reconstructed_signal = (
    reconstruct_as7343_spectrum(
        measurement,
        start_nm=380.0,
        stop_nm=900.0,
        step_nm=1.0,
    )
)

visible_integral = integrate_band(
    reconstructed_wavelengths,
    reconstructed_signal,
    400.0,
    700.0,
)

centroid_nm = spectral_centroid(
    reconstructed_wavelengths,
    reconstructed_signal,
)

peak_nm = peak_wavelength(
    reconstructed_wavelengths,
    reconstructed_signal,
)

print(measurement.summary())

print()
print("Discrete AS7343 spectrum:")
print("Wavelengths:", channel_wavelengths)
print("Signals    :", channel_signal)
print("Normalized :", normalized_signal)

print()
print(
    "Reconstructed points:",
    len(reconstructed_wavelengths),
)
print(
    f"Visible integral 400-700 nm: "
    f"{visible_integral:.2f}"
)
print(
    f"Spectral centroid: {centroid_nm:.1f} nm"
)
print(
    f"Peak wavelength: {peak_nm:.1f} nm"
)