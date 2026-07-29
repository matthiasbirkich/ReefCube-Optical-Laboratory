"""
visualization.py
================

Publication-quality Matplotlib visualizations for the Reef Cube Optical
Laboratory.

The module is deliberately independent of interactive notebook features and
therefore works in ordinary Python, Pyto on iPad, desktop Python, and automated
report-generation workflows.

Author
------
Matthias Birkicht & OpenAI

Version
-------
1.1
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from reefcube.calibration import CalibrationResult
from reefcube.spectroscopy import reconstruct_as7343_spectrum
from reefcube.wavelength import AS7343_CHANNELS, AS7343_CHANNEL_ORDER


FigureFormat = Literal["png", "pdf", "svg", "jpg", "jpeg", "tif", "tiff"]


def set_publication_style(*, font_size: float = 10.0) -> None:
    """Apply a restrained scientific Matplotlib style.

    Parameters
    ----------
    font_size
        Base font size in points.
    """

    if not np.isfinite(font_size) or font_size <= 0.0:
        raise ValueError("font_size must be finite and greater than zero.")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": font_size,
            "axes.titlesize": font_size + 2.0,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1.0,
            "ytick.labelsize": font_size - 1.0,
            "legend.fontsize": font_size - 1.0,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.autolayout": False,
        }
    )


def save_figure(
    figure: Figure,
    path: str | Path,
    *,
    dpi: int = 300,
    transparent: bool = False,
    close: bool = False,
) -> Path:
    """Save a Matplotlib figure and return its resolved path.

    Parameters
    ----------
    figure
        Figure to save.
    path
        Output path. The extension determines the file format.
    dpi
        Raster resolution in dots per inch.
    transparent
        Save with transparent background.
    close
        Close the figure after saving.
    """

    if not isinstance(figure, Figure):
        raise TypeError("figure must be a matplotlib.figure.Figure.")
    if dpi < 1:
        raise ValueError("dpi must be at least 1.")

    output = Path(path).expanduser()
    if not output.suffix:
        output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
        transparent=transparent,
    )
    if close:
        plt.close(figure)
    return output.resolve()


def plot_channel_bars(
    measurement: Any,
    *,
    normalize: bool = False,
    title: str = "AS7343 spectral channels",
    ylabel: str | None = None,
    annotate: bool = False,
    figsize: tuple[float, float] = (9.0, 4.8),
) -> tuple[Figure, Axes]:
    """Plot AS7343 channel signals as a wavelength-ordered bar chart.

    Parameters
    ----------
    measurement
        Measurement object or mapping containing ``spectral_channels``.
    normalize
        Divide values by their maximum before plotting.
    title
        Figure title.
    ylabel
        Y-axis label. A suitable default is selected automatically.
    annotate
        Display values above bars.
    figsize
        Figure size in inches.
    """

    channels = _spectral_channel_mapping(measurement)
    names = [name for name in AS7343_CHANNEL_ORDER if name in channels]
    if not names:
        raise ValueError("No known AS7343 channels are available.")

    values = np.asarray([float(channels[name]) for name in names], dtype=float)
    _require_finite(values, "spectral channel values")

    if normalize:
        maximum = float(np.max(np.abs(values)))
        if maximum == 0.0:
            raise ValueError("Cannot normalize a zero-valued spectrum.")
        values = values / maximum

    wavelengths = [AS7343_CHANNELS[name].peak_wavelength_nm for name in names]
    labels = [f"{name}\n{wavelength:.0f} nm" for name, wavelength in zip(names, wavelengths)]

    figure, axis = plt.subplots(figsize=figsize)
    bars = axis.bar(labels, values)
    axis.set_title(title)
    axis.set_xlabel("AS7343 channel and peak wavelength")
    axis.set_ylabel(ylabel or ("Relative signal" if normalize else "Sensor signal"))
    axis.grid(axis="x", visible=False)

    if annotate:
        for bar, value in zip(bars, values):
            axis.annotate(
                f"{value:.3g}",
                xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=max(7.0, plt.rcParams["font.size"] - 2.0),
            )

    figure.tight_layout()
    return figure, axis


def plot_spectrum(
    wavelengths_nm: Iterable[float],
    signal: Iterable[float],
    *,
    title: str = "Spectrum",
    xlabel: str = "Wavelength (nm)",
    ylabel: str = "Signal",
    label: str | None = None,
    fill: bool = False,
    marker: str | None = None,
    figsize: tuple[float, float] = (8.5, 4.8),
) -> tuple[Figure, Axes]:
    """Plot one spectrum.

    Parameters
    ----------
    wavelengths_nm
        Wavelength values in nanometres.
    signal
        Spectral signal values.
    title, xlabel, ylabel
        Plot labels.
    label
        Optional legend label.
    fill
        Fill the area beneath the spectral curve.
    marker
        Optional Matplotlib marker string.
    figsize
        Figure size in inches.
    """

    wavelengths, values = _paired_arrays(wavelengths_nm, signal)
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    values = values[order]

    figure, axis = plt.subplots(figsize=figsize)
    axis.plot(wavelengths, values, label=label, marker=marker)
    if fill:
        axis.fill_between(wavelengths, 0.0, values, alpha=0.2)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_xlim(float(wavelengths[0]), float(wavelengths[-1]))
    if label:
        axis.legend()
    figure.tight_layout()
    return figure, axis


def plot_measurement_spectrum(
    measurement: Any,
    *,
    wavelength_min_nm: float = 380.0,
    wavelength_max_nm: float = 900.0,
    wavelength_step_nm: float = 1.0,
    normalize: bool = False,
    show_channels: bool = True,
    show_filter_bandwidth: bool = True,
    show_visible_background: bool = True,
    background_alpha: float = 0.12,
    title: str = "Reconstructed AS7343 spectrum",
    figsize: tuple[float, float] = (9.0, 5.0),
) -> tuple[Figure, Axes]:
    """Reconstruct and plot one AS7343 measurement.

    The plot can show a faint wavelength-colour background, the actual channel
    centre wavelengths, and each channel's typical full width at half maximum
    (FWHM). The wavelength and bandwidth values come directly from
    :mod:`reefcube.wavelength`.

    Parameters
    ----------
    measurement
        ReefCubeMeasurement-compatible object or mapping containing
        ``spectral_channels``.
    wavelength_min_nm, wavelength_max_nm
        Reconstruction limits in nanometres.
    wavelength_step_nm
        Wavelength-grid spacing in nanometres.
    normalize
        Normalize the reconstructed curve and channel points to their maxima.
    show_channels
        Overlay original AS7343 channel measurements and labels.
    show_filter_bandwidth
        Draw a translucent FWHM band for every available AS7343 channel.
    show_visible_background
        Draw a faint wavelength-colour background from 380 to 780 nm.
    background_alpha
        Opacity of the wavelength-colour background. Must lie from 0 to 1.
    title
        Plot title.
    figsize
        Figure size in inches.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes.
    """

    if wavelength_max_nm <= wavelength_min_nm:
        raise ValueError("wavelength_max_nm must exceed wavelength_min_nm.")
    if wavelength_step_nm <= 0.0:
        raise ValueError("wavelength_step_nm must be greater than zero.")
    if not 0.0 <= background_alpha <= 1.0:
        raise ValueError("background_alpha must lie between 0 and 1.")

    wavelengths, reconstructed = reconstruct_as7343_spectrum(
        measurement,
        start_nm=wavelength_min_nm,
        stop_nm=wavelength_max_nm,
        step_nm=wavelength_step_nm,
        normalize_basis=True,
    )

    channels = _spectral_channel_mapping(measurement)
    names = [name for name in AS7343_CHANNEL_ORDER if name in channels]
    channel_wavelengths = np.asarray(
        [AS7343_CHANNELS[name].peak_wavelength_nm for name in names], dtype=float
    )
    channel_values = np.asarray([float(channels[name]) for name in names], dtype=float)
    _require_finite(channel_values, "spectral channel values")

    if normalize:
        curve_max = float(np.max(np.abs(reconstructed)))
        point_max = float(np.max(np.abs(channel_values))) if channel_values.size else 0.0
        if curve_max == 0.0 or point_max == 0.0:
            raise ValueError("Cannot normalize a zero-valued spectrum.")
        reconstructed = reconstructed / curve_max
        channel_values = channel_values / point_max

    figure, axis = plt.subplots(figsize=figsize)

    if show_visible_background:
        _add_wavelength_background(
            axis,
            wavelength_min_nm=wavelength_min_nm,
            wavelength_max_nm=wavelength_max_nm,
            alpha=background_alpha,
        )

    if show_filter_bandwidth:
        for name in names:
            channel = AS7343_CHANNELS[name]
            half_width = channel.fwhm_nm / 2.0
            left = max(wavelength_min_nm, channel.peak_wavelength_nm - half_width)
            right = min(wavelength_max_nm, channel.peak_wavelength_nm + half_width)
            if right > left:
                axis.axvspan(
                    left,
                    right,
                    color=_wavelength_to_rgb(channel.peak_wavelength_nm),
                    alpha=0.075,
                    linewidth=0.0,
                    zorder=0.6,
                )

    axis.plot(
        wavelengths,
        reconstructed,
        linewidth=2.0,
        label="Gaussian reconstruction",
        zorder=2.5,
    )
    axis.fill_between(wavelengths, 0.0, reconstructed, alpha=0.13, zorder=1.5)

    if show_channels and channel_values.size:
        point_colours = [_wavelength_to_rgb(value) for value in channel_wavelengths]
        axis.scatter(
            channel_wavelengths,
            channel_values,
            c=point_colours,
            edgecolors="black",
            linewidths=0.5,
            s=42,
            label="AS7343 channels",
            zorder=3.5,
        )
        for name, x_value, y_value in zip(names, channel_wavelengths, channel_values):
            axis.axvline(x_value, color="black", alpha=0.10, linewidth=0.8, zorder=0.8)
            axis.annotate(
                f"{name}\n{x_value:.0f} nm",
                (x_value, y_value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=max(7.0, plt.rcParams["font.size"] - 2.0),
                zorder=4.0,
            )

    axis.set_title(title)
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel("Relative signal" if normalize else "Sensor signal")
    axis.set_xlim(wavelength_min_nm, wavelength_max_nm)
    axis.set_ylim(bottom=0.0)
    axis.legend()
    figure.tight_layout()
    return figure, axis


def plot_as7343_spectrum(
    measurement: Any,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot the signature AS7343 spectrum visualization.

    This convenience wrapper calls :func:`plot_measurement_spectrum` with the
    wavelength background, channel centres, and FWHM bands enabled by default.

    Parameters
    ----------
    measurement
        ReefCubeMeasurement-compatible object or mapping.
    **kwargs
        Keyword arguments forwarded to :func:`plot_measurement_spectrum`.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes.
    """

    defaults: dict[str, Any] = {
        "show_channels": True,
        "show_filter_bandwidth": True,
        "show_visible_background": True,
    }
    defaults.update(kwargs)
    return plot_measurement_spectrum(measurement, **defaults)


def plot_time_series(
    timestamps: Iterable[datetime | str],
    values: Iterable[float],
    *,
    title: str = "Time series",
    ylabel: str = "Value",
    label: str | None = None,
    rolling_window: int | None = None,
    figsize: tuple[float, float] = (10.0, 4.8),
) -> tuple[Figure, Axes]:
    """Plot a timestamped series with an optional rolling mean."""

    time_values = [_parse_datetime(value) for value in timestamps]
    numeric = np.asarray(list(values), dtype=float)
    if len(time_values) != numeric.size:
        raise ValueError("timestamps and values must have the same length.")
    if numeric.size == 0:
        raise ValueError("At least one observation is required.")
    _require_finite(numeric, "values")

    order = np.argsort(np.asarray([value.timestamp() for value in time_values]))
    sorted_times = [time_values[index] for index in order]
    sorted_values = numeric[order]

    figure, axis = plt.subplots(figsize=figsize)
    axis.plot(sorted_times, sorted_values, marker="o", markersize=3.5, label=label or "Observed")

    if rolling_window is not None:
        if rolling_window < 1:
            raise ValueError("rolling_window must be at least 1.")
        rolling = _rolling_mean_full(sorted_values, rolling_window)
        axis.plot(sorted_times, rolling, linewidth=2.0, label=f"{rolling_window}-point mean")

    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axis.set_title(title)
    axis.set_xlabel("Time")
    axis.set_ylabel(ylabel)
    if label or rolling_window is not None:
        axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure, axis


def plot_calibration(
    sensor_values: Iterable[float],
    reference_values: Iterable[float],
    calibration: CalibrationResult,
    *,
    title: str = "Calibration curve",
    sensor_label: str | None = None,
    reference_label: str | None = None,
    show_equation: bool = True,
    show_identity: bool = False,
    figsize: tuple[float, float] = (6.4, 5.2),
) -> tuple[Figure, Axes]:
    """Plot calibration observations and the fitted regression line."""

    x_values, y_values = _paired_arrays(sensor_values, reference_values)
    if not isinstance(calibration, CalibrationResult):
        raise TypeError("calibration must be a CalibrationResult.")

    x_line = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 200)
    y_line = calibration.slope * x_line + calibration.intercept

    figure, axis = plt.subplots(figsize=figsize)
    axis.scatter(x_values, y_values, label="Calibration observations", zorder=3)
    axis.plot(x_line, y_line, label="Linear fit")

    if show_identity:
        lower = min(float(np.min(x_values)), float(np.min(y_values)))
        upper = max(float(np.max(x_values)), float(np.max(y_values)))
        axis.plot([lower, upper], [lower, upper], linestyle="--", label="1:1 line")

    axis.set_title(title)
    axis.set_xlabel(sensor_label or calibration.sensor_name or "Sensor value")
    axis.set_ylabel(reference_label or calibration.reference_name or "Reference value")

    if show_equation:
        text = f"{calibration.equation()}\nR² = {calibration.r_squared:.4f}\nRMSE = {calibration.rmse:.4g}"
        axis.text(
            0.03,
            0.97,
            text,
            transform=axis.transAxes,
            ha="left",
            va="top",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85},
        )

    axis.legend()
    figure.tight_layout()
    return figure, axis


def plot_residuals(
    sensor_values: Iterable[float],
    reference_values: Iterable[float],
    calibration: CalibrationResult,
    *,
    title: str = "Calibration residuals",
    xlabel: str | None = None,
    figsize: tuple[float, float] = (7.2, 4.8),
) -> tuple[Figure, Axes]:
    """Plot calibration residuals against sensor values."""

    x_values, y_values = _paired_arrays(sensor_values, reference_values)
    residuals = calibration.residuals(x_values, y_values)

    figure, axis = plt.subplots(figsize=figsize)
    axis.scatter(x_values, residuals, zorder=3)
    axis.axhline(0.0, linestyle="--", linewidth=1.2)
    axis.set_title(title)
    axis.set_xlabel(xlabel or calibration.sensor_name or "Sensor value")
    axis.set_ylabel("Residual (observed − predicted)")
    figure.tight_layout()
    return figure, axis


def plot_comparison(
    measured_values: Iterable[float],
    reference_values: Iterable[float],
    *,
    title: str = "Measured versus reference",
    measured_label: str = "Measured value",
    reference_label: str = "Reference value",
    show_regression: bool = True,
    show_identity: bool = True,
    figsize: tuple[float, float] = (6.2, 5.2),
) -> tuple[Figure, Axes]:
    """Plot measured values against paired reference observations."""

    measured, reference = _paired_arrays(measured_values, reference_values)
    figure, axis = plt.subplots(figsize=figsize)
    axis.scatter(reference, measured, zorder=3, label="Paired observations")

    lower = min(float(np.min(reference)), float(np.min(measured)))
    upper = max(float(np.max(reference)), float(np.max(measured)))

    if show_identity:
        axis.plot([lower, upper], [lower, upper], linestyle="--", label="1:1 line")

    if show_regression:
        design = np.column_stack([reference, np.ones(reference.size)])
        slope, intercept = np.linalg.lstsq(design, measured, rcond=None)[0]
        x_line = np.linspace(float(np.min(reference)), float(np.max(reference)), 200)
        axis.plot(x_line, slope * x_line + intercept, label="Linear regression")

    axis.set_title(title)
    axis.set_xlabel(reference_label)
    axis.set_ylabel(measured_label)
    axis.set_aspect("equal", adjustable="box")
    axis.legend()
    figure.tight_layout()
    return figure, axis


def plot_spectral_heatmap(
    measurements: Sequence[Any],
    *,
    normalize_rows: bool = False,
    title: str = "AS7343 spectral heatmap",
    figsize: tuple[float, float] = (9.0, 5.6),
) -> tuple[Figure, Axes]:
    """Plot multiple AS7343 measurements as a channel-by-time heatmap.

    Parameters
    ----------
    measurements
        Ordered measurement objects or mappings.
    normalize_rows
        Normalize each measurement to its largest absolute channel value.
    title
        Plot title.
    figsize
        Figure size in inches.
    """

    if not measurements:
        raise ValueError("At least one measurement is required.")

    names = list(AS7343_CHANNEL_ORDER)
    matrix = np.full((len(measurements), len(names)), np.nan, dtype=float)
    timestamps: list[str] = []

    for row, measurement in enumerate(measurements):
        channels = _spectral_channel_mapping(measurement)
        for column, name in enumerate(names):
            if name in channels:
                matrix[row, column] = float(channels[name])
        timestamp = _measurement_field(measurement, "timestamp", row)
        if isinstance(timestamp, datetime):
            timestamps.append(timestamp.strftime("%Y-%m-%d\n%H:%M"))
        else:
            timestamps.append(str(timestamp))

    if not np.any(np.isfinite(matrix)):
        raise ValueError("No finite spectral channel values are available.")

    if normalize_rows:
        for row in range(matrix.shape[0]):
            finite = np.isfinite(matrix[row])
            if np.any(finite):
                maximum = float(np.max(np.abs(matrix[row, finite])))
                if maximum > 0.0:
                    matrix[row, finite] /= maximum

    figure, axis = plt.subplots(figsize=figsize)
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest")
    axis.set_title(title)
    axis.set_xlabel("AS7343 channel")
    axis.set_ylabel("Measurement")
    axis.set_xticks(np.arange(len(names)), names)

    max_labels = 10
    if len(timestamps) <= max_labels:
        ticks = np.arange(len(timestamps))
    else:
        ticks = np.unique(np.linspace(0, len(timestamps) - 1, max_labels, dtype=int))
    axis.set_yticks(ticks, [timestamps[index] for index in ticks])

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Relative signal" if normalize_rows else "Sensor signal")
    figure.tight_layout()
    return figure, axis


def _add_wavelength_background(
    axis: Axes,
    *,
    wavelength_min_nm: float,
    wavelength_max_nm: float,
    alpha: float,
) -> None:
    """Add a faint visible-spectrum background to an axes object."""

    visible_min = max(380.0, wavelength_min_nm)
    visible_max = min(780.0, wavelength_max_nm)
    if visible_max <= visible_min or alpha == 0.0:
        return

    wavelengths = np.linspace(visible_min, visible_max, 801)
    colours = np.asarray([_wavelength_to_rgb(value) for value in wavelengths])
    image = np.repeat(colours[np.newaxis, :, :], 2, axis=0)
    axis.imshow(
        image,
        extent=(visible_min, visible_max, 0.0, 1.0),
        transform=axis.get_xaxis_transform(),
        origin="lower",
        aspect="auto",
        alpha=alpha,
        interpolation="bilinear",
        zorder=0.0,
    )


def _wavelength_to_rgb(wavelength_nm: float) -> tuple[float, float, float]:
    """Approximate a visible wavelength as an sRGB colour.

    Wavelengths outside the visible range are rendered as subdued grey-violet
    (near UV) or grey-red (near infrared), keeping NIR channel markers legible.
    """

    wavelength = float(wavelength_nm)
    if wavelength < 380.0:
        return (0.35, 0.30, 0.45)
    if wavelength > 780.0:
        return (0.45, 0.28, 0.28)

    if wavelength < 440.0:
        red = -(wavelength - 440.0) / 60.0
        green = 0.0
        blue = 1.0
    elif wavelength < 490.0:
        red = 0.0
        green = (wavelength - 440.0) / 50.0
        blue = 1.0
    elif wavelength < 510.0:
        red = 0.0
        green = 1.0
        blue = -(wavelength - 510.0) / 20.0
    elif wavelength < 580.0:
        red = (wavelength - 510.0) / 70.0
        green = 1.0
        blue = 0.0
    elif wavelength < 645.0:
        red = 1.0
        green = -(wavelength - 645.0) / 65.0
        blue = 0.0
    else:
        red = 1.0
        green = 0.0
        blue = 0.0

    if wavelength < 420.0:
        intensity = 0.3 + 0.7 * (wavelength - 380.0) / 40.0
    elif wavelength <= 700.0:
        intensity = 1.0
    else:
        intensity = 0.3 + 0.7 * (780.0 - wavelength) / 80.0

    gamma = 0.8
    rgb = tuple((max(0.0, component) * intensity) ** gamma for component in (red, green, blue))
    return rgb  # type: ignore[return-value]


def _measurement_field(measurement: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(measurement, Mapping):
        return measurement.get(field_name, default)
    return getattr(measurement, field_name, default)


def _spectral_channel_mapping(measurement: Any) -> Mapping[str, Any]:
    channels = _measurement_field(measurement, "spectral_channels")
    if not isinstance(channels, Mapping):
        raise TypeError("measurement must contain a spectral_channels mapping.")
    return channels


def _paired_arrays(
    first: Iterable[float],
    second: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    first_array = np.asarray(list(first), dtype=float)
    second_array = np.asarray(list(second), dtype=float)
    if first_array.ndim != 1 or second_array.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")
    if first_array.size != second_array.size:
        raise ValueError("Inputs must have the same length.")
    if first_array.size == 0:
        raise ValueError("At least one paired observation is required.")
    _require_finite(first_array, "first input")
    _require_finite(second_array, "second input")
    return first_array, second_array


def _require_finite(values: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")


def _parse_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text)
        except ValueError as error:
            raise ValueError(f"Invalid ISO-8601 timestamp: {value!r}.") from error
    raise TypeError("timestamps must contain datetime or ISO-8601 strings.")


def _rolling_mean_full(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    for index in range(window - 1, values.size):
        result[index] = float(np.mean(values[index - window + 1 : index + 1]))
    return result


__all__ = [
    "set_publication_style",
    "save_figure",
    "plot_channel_bars",
    "plot_spectrum",
    "plot_measurement_spectrum",
    "plot_time_series",
    "plot_calibration",
    "plot_residuals",
    "plot_comparison",
    "plot_spectral_heatmap",
]
