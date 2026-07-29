from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reefcube.calibration import fit_linear_calibration
from reefcube.sensors import SimulationSensor
from reefcube.visualization import (
    plot_calibration,
    plot_channel_bars,
    plot_comparison,
    plot_as7343_spectrum,
    plot_measurement_spectrum,
    plot_residuals,
    plot_spectral_heatmap,
    plot_spectrum,
    plot_time_series,
    save_figure,
    set_publication_style,
)

out = Path("visualization_test")
out.mkdir(exist_ok=True)
set_publication_style()

sensor = SimulationSensor(mode="underwater", seed=42)
measurements = [sensor.acquire() for _ in range(6)]
start = measurements[0].timestamp
for i, m in enumerate(measurements):
    m.timestamp = start + timedelta(minutes=30 * i)

fig, _ = plot_channel_bars(measurements[0], annotate=True)
save_figure(fig, out / "01_channel_bars.png", close=True)

fig, _ = plot_as7343_spectrum(measurements[0], normalize=True)
save_figure(fig, out / "02_reconstructed_spectrum.png", close=True)

wavelengths = np.arange(400.0, 701.0, 10.0)
signal = np.exp(-0.5 * ((wavelengths - 515.0) / 35.0) ** 2)
fig, _ = plot_spectrum(wavelengths, signal, fill=True, marker="o")
save_figure(fig, out / "03_spectrum.png", close=True)

fig, _ = plot_time_series(
    [m.timestamp for m in measurements],
    [m.lux for m in measurements],
    ylabel="Lux",
    rolling_window=3,
)
save_figure(fig, out / "04_time_series.png", close=True)

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = 2.5 * x + 1.0 + np.array([0.1, -0.2, 0.15, -0.1, 0.05])
cal = fit_linear_calibration(x, y, sensor_name="Index", reference_name="PPFD")

fig, _ = plot_calibration(x, y, cal)
save_figure(fig, out / "05_calibration.png", close=True)

fig, _ = plot_residuals(x, y, cal)
save_figure(fig, out / "06_residuals.png", close=True)

fig, _ = plot_comparison(y, 2.5 * x + 1.0)
save_figure(fig, out / "07_comparison.png", close=True)

fig, _ = plot_spectral_heatmap(measurements, normalize_rows=True)
save_figure(fig, out / "08_heatmap.png", close=True)

files = sorted(out.glob("*.png"))
assert len(files) == 8
assert all(path.stat().st_size > 1000 for path in files)
assert not plt.get_fignums()
print("Visualization test passed.")
for path in files:
    print(path, path.stat().st_size)
