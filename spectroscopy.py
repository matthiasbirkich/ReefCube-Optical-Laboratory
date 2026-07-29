"""
spectroscopy.py
===============

General spectral-processing functions for the Reef Cube Optical
Laboratory.

The functions in this module operate on wavelength and signal arrays.
They are intentionally independent of a specific sensor so that they
can be used with:

- Reef Cube AS7343 measurements
- TRIOS RAMSES spectra
- laboratory spectrometer data
- CSV imports
- simulated spectra
- future optical sensors

Author
------
Matthias Birkicht & OpenAI

Version
-------
1.0
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np

from reefcube.measurement import ReefCubeMeasurement
from reefcube.wavelength import (
    AS7343_CHANNELS,
    AS7343_CHANNEL_ORDER,
)


NormalizationMode = Literal[
    "max",
    "area",
    "sum",
    "minmax",
]


def measurement_to_arrays(
    measurement: ReefCubeMeasurement,
    *,
    ignore_unknown_channels: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a ReefCubeMeasurement spectrum to ordered NumPy arrays.

    The spectral channels are sorted by increasing peak wavelength.

    Parameters
    ----------
    measurement
        Reef Cube measurement containing spectral channel values.
    ignore_unknown_channels
        If ``True``, channels without a defined peak wavelength are
        omitted. If ``False``, unknown channels raise ``KeyError``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Wavelengths in nanometres and corresponding spectral signals.

    Raises
    ------
    TypeError
        If measurement is not a ReefCubeMeasurement.
    ValueError
        If no usable spectral channels are present.
    KeyError
        If an unknown channel is encountered and
        ``ignore_unknown_channels`` is ``False``.
    """

    if not isinstance(measurement, ReefCubeMeasurement):
        raise TypeError(
            "measurement must be a ReefCubeMeasurement object."
        )

    wavelengths: list[float] = []
    signals: list[float] = []

    for channel_name in AS7343_CHANNEL_ORDER:
        if channel_name in measurement.spectral_channels:
            wavelengths.append(
                AS7343_CHANNELS[
                    channel_name
                ].peak_wavelength_nm
            )
            signals.append(
                float(
                    measurement.spectral_channels[channel_name]
                )
            )

    if not ignore_unknown_channels:
        for channel_name in measurement.spectral_channels:
            if channel_name not in AS7343_CHANNELS:
                raise KeyError(
                    f"Unknown spectral channel "
                    f"{channel_name!r}."
                )

    if not wavelengths:
        raise ValueError(
            "Measurement contains no usable spectral channels."
        )

    wavelength_array = np.asarray(
        wavelengths,
        dtype=float,
    )

    signal_array = np.asarray(
        signals,
        dtype=float,
    )

    _validate_spectrum(
        wavelength_array,
        signal_array,
        require_sorted=True,
    )

    return wavelength_array, signal_array


def normalize_spectrum(
    signal: Iterable[float] | np.ndarray,
    *,
    wavelengths_nm: Iterable[float] | np.ndarray | None = None,
    mode: NormalizationMode = "max",
) -> np.ndarray:
    """
    Normalize a spectrum.

    Parameters
    ----------
    signal
        Spectral signal values.
    wavelengths_nm
        Wavelength values in nanometres. Required for ``mode="area"``.
    mode
        Normalization method:

        ``"max"``
            Divide by the maximum absolute signal.

        ``"area"``
            Divide by the absolute integrated spectral area.

        ``"sum"``
            Divide by the sum of absolute values.

        ``"minmax"``
            Scale linearly to the interval 0 to 1.

    Returns
    -------
    numpy.ndarray
        Normalized spectrum.

    Raises
    ------
    ValueError
        If the selected normalization cannot be performed.
    """

    signal_array = _as_1d_float_array(
        signal,
        name="signal",
    )

    if mode == "max":
        denominator = float(
            np.max(np.abs(signal_array))
        )

        if denominator == 0.0:
            raise ValueError(
                "Cannot normalize a zero spectrum by maximum."
            )

        return signal_array / denominator

    if mode == "sum":
        denominator = float(
            np.sum(np.abs(signal_array))
        )

        if denominator == 0.0:
            raise ValueError(
                "Cannot normalize a zero spectrum by sum."
            )

        return signal_array / denominator

    if mode == "area":
        if wavelengths_nm is None:
            raise ValueError(
                "wavelengths_nm is required for area "
                "normalization."
            )

        wavelength_array = _as_1d_float_array(
            wavelengths_nm,
            name="wavelengths_nm",
        )

        _validate_spectrum(
            wavelength_array,
            signal_array,
            require_sorted=True,
        )

        denominator = abs(
            _trapezoidal_integral(
                np.abs(signal_array),
                wavelength_array,
            )
        )

        if denominator == 0.0:
            raise ValueError(
                "Cannot normalize a spectrum with zero area."
            )

        return signal_array / denominator

    if mode == "minmax":
        minimum = float(np.min(signal_array))
        maximum = float(np.max(signal_array))
        difference = maximum - minimum

        if difference == 0.0:
            raise ValueError(
                "Cannot apply min-max normalization to a "
                "constant spectrum."
            )

        return (
            signal_array - minimum
        ) / difference

    raise ValueError(
        "Unknown normalization mode. Valid modes are: "
        "'max', 'area', 'sum', and 'minmax'."
    )


def interpolate_spectrum(
    wavelengths_nm: Iterable[float] | np.ndarray,
    signal: Iterable[float] | np.ndarray,
    new_wavelengths_nm: Iterable[float] | np.ndarray,
    *,
    extrapolate: bool = False,
) -> np.ndarray:
    """
    Linearly interpolate a spectrum onto a new wavelength grid.

    Parameters
    ----------
    wavelengths_nm
        Original wavelengths in nanometres.
    signal
        Original spectral signal values.
    new_wavelengths_nm
        Target wavelengths in nanometres.
    extrapolate
        If ``False``, target wavelengths outside the original range
        receive ``NaN``. If ``True``, the nearest edge value is used.

    Returns
    -------
    numpy.ndarray
        Interpolated signal values.
    """

    wavelength_array = _as_1d_float_array(
        wavelengths_nm,
        name="wavelengths_nm",
    )

    signal_array = _as_1d_float_array(
        signal,
        name="signal",
    )

    target_array = _as_1d_float_array(
        new_wavelengths_nm,
        name="new_wavelengths_nm",
    )

    _validate_spectrum(
        wavelength_array,
        signal_array,
        require_sorted=True,
    )

    if extrapolate:
        left_value = float(signal_array[0])
        right_value = float(signal_array[-1])
    else:
        left_value = np.nan
        right_value = np.nan

    return np.interp(
        target_array,
        wavelength_array,
        signal_array,
        left=left_value,
        right=right_value,
    )


def gaussian_reconstruction(
    wavelengths_nm: Iterable[float] | np.ndarray,
    signal: Iterable[float] | np.ndarray,
    output_wavelengths_nm: Iterable[float] | np.ndarray,
    *,
    fwhm_nm: Iterable[float] | np.ndarray | None = None,
    normalize_basis: bool = True,
) -> np.ndarray:
    """
    Reconstruct a continuous spectrum from discrete sensor channels.

    Each sensor channel is represented by a Gaussian response curve
    centred at its channel wavelength. The measured channel value
    scales the corresponding Gaussian.

    Parameters
    ----------
    wavelengths_nm
        Channel peak wavelengths in nanometres.
    signal
        Measured channel values.
    output_wavelengths_nm
        Wavelength grid for the reconstructed spectrum.
    fwhm_nm
        Full width at half maximum for each channel. If omitted,
        each channel receives a width based on half the distance to
        neighbouring channels.
    normalize_basis
        If ``True``, Gaussian contributions are divided by the summed
        Gaussian basis. This reduces artificial signal amplification
        where several channel responses overlap.

    Returns
    -------
    numpy.ndarray
        Reconstructed spectrum on the output wavelength grid.
    """

    channel_wavelengths = _as_1d_float_array(
        wavelengths_nm,
        name="wavelengths_nm",
    )

    channel_signal = _as_1d_float_array(
        signal,
        name="signal",
    )

    output_grid = _as_1d_float_array(
        output_wavelengths_nm,
        name="output_wavelengths_nm",
    )

    _validate_spectrum(
        channel_wavelengths,
        channel_signal,
        require_sorted=True,
    )

    if fwhm_nm is None:
        channel_fwhm = _estimate_channel_fwhm(
            channel_wavelengths
        )
    else:
        channel_fwhm = _as_1d_float_array(
            fwhm_nm,
            name="fwhm_nm",
        )

        if channel_fwhm.size != channel_wavelengths.size:
            raise ValueError(
                "fwhm_nm must contain one value per channel."
            )

        if np.any(channel_fwhm <= 0.0):
            raise ValueError(
                "All FWHM values must be greater than zero."
            )

    reconstructed = np.zeros_like(
        output_grid,
        dtype=float,
    )

    basis_sum = np.zeros_like(
        output_grid,
        dtype=float,
    )

    for centre_nm, amplitude, width_nm in zip(
        channel_wavelengths,
        channel_signal,
        channel_fwhm,
    ):
        sigma_nm = (
            width_nm
            / (
                2.0
                * np.sqrt(
                    2.0 * np.log(2.0)
                )
            )
        )

        gaussian = np.exp(
            -0.5
            * (
                (
                    output_grid - centre_nm
                )
                / sigma_nm
            )
            ** 2
        )

        reconstructed += amplitude * gaussian
        basis_sum += gaussian

    if normalize_basis:
        valid = basis_sum > np.finfo(float).eps

        normalized = np.zeros_like(
            reconstructed,
            dtype=float,
        )

        normalized[valid] = (
            reconstructed[valid]
            / basis_sum[valid]
        )

        return normalized

    return reconstructed


def reconstruct_as7343_spectrum(
    measurement: ReefCubeMeasurement,
    *,
    start_nm: float = 380.0,
    stop_nm: float = 900.0,
    step_nm: float = 1.0,
    normalize_basis: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct a continuous AS7343 spectrum from a measurement.

    The official peak wavelengths and typical FWHM values from
    ``wavelength.py`` are used.

    Parameters
    ----------
    measurement
        Reef Cube measurement containing AS7343 channel values.
    start_nm
        First output wavelength in nanometres.
    stop_nm
        Final output wavelength in nanometres.
    step_nm
        Wavelength spacing in nanometres.
    normalize_basis
        If ``True``, compensate for overlapping Gaussian basis
        functions.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Output wavelengths and reconstructed spectral values.
    """

    if step_nm <= 0.0:
        raise ValueError(
            "step_nm must be greater than zero."
        )

    if start_nm >= stop_nm:
        raise ValueError(
            "start_nm must be smaller than stop_nm."
        )

    channel_wavelengths, channel_signal = (
        measurement_to_arrays(measurement)
    )

    channel_fwhm = np.asarray(
        [
            AS7343_CHANNELS[name].fwhm_nm
            for name in AS7343_CHANNEL_ORDER
            if name in measurement.spectral_channels
        ],
        dtype=float,
    )

    output_wavelengths = np.arange(
        float(start_nm),
        float(stop_nm) + 0.5 * float(step_nm),
        float(step_nm),
        dtype=float,
    )

    reconstructed_signal = gaussian_reconstruction(
        wavelengths_nm=channel_wavelengths,
        signal=channel_signal,
        output_wavelengths_nm=output_wavelengths,
        fwhm_nm=channel_fwhm,
        normalize_basis=normalize_basis,
    )

    return output_wavelengths, reconstructed_signal


def smooth_spectrum(
    signal: Iterable[float] | np.ndarray,
    *,
    window_length: int = 5,
    method: Literal["moving_average", "gaussian"] = (
        "moving_average"
    ),
    gaussian_sigma: float | None = None,
) -> np.ndarray:
    """
    Smooth a one-dimensional spectrum.

    Parameters
    ----------
    signal
        Spectral values.
    window_length
        Number of points in the smoothing window. Must be an odd
        positive integer.
    method
        ``"moving_average"`` or ``"gaussian"``.
    gaussian_sigma
        Standard deviation of the Gaussian kernel in sample units.
        If omitted, a suitable value is derived from window length.

    Returns
    -------
    numpy.ndarray
        Smoothed spectrum with the same length as the input.
    """

    signal_array = _as_1d_float_array(
        signal,
        name="signal",
    )

    if window_length < 1:
        raise ValueError(
            "window_length must be at least 1."
        )

    if window_length % 2 == 0:
        raise ValueError(
            "window_length must be odd."
        )

    if window_length > signal_array.size:
        raise ValueError(
            "window_length must not exceed signal length."
        )

    if window_length == 1:
        return signal_array.copy()

    if method == "moving_average":
        kernel = np.ones(
            window_length,
            dtype=float,
        )
        kernel /= np.sum(kernel)

    elif method == "gaussian":
        if gaussian_sigma is None:
            gaussian_sigma = window_length / 6.0

        if gaussian_sigma <= 0.0:
            raise ValueError(
                "gaussian_sigma must be greater than zero."
            )

        half_window = window_length // 2

        positions = np.arange(
            -half_window,
            half_window + 1,
            dtype=float,
        )

        kernel = np.exp(
            -0.5
            * (
                positions
                / float(gaussian_sigma)
            )
            ** 2
        )

        kernel /= np.sum(kernel)

    else:
        raise ValueError(
            "Unknown smoothing method. Valid methods are "
            "'moving_average' and 'gaussian'."
        )

    half_window = window_length // 2

    padded = np.pad(
        signal_array,
        pad_width=half_window,
        mode="edge",
    )

    return np.convolve(
        padded,
        kernel,
        mode="valid",
    )


def integrate_band(
    wavelengths_nm: Iterable[float] | np.ndarray,
    signal: Iterable[float] | np.ndarray,
    minimum_wavelength_nm: float,
    maximum_wavelength_nm: float,
    *,
    interpolate_limits: bool = True,
) -> float:
    """
    Integrate a spectrum over a wavelength interval.

    Trapezoidal numerical integration is used.

    Parameters
    ----------
    wavelengths_nm
        Spectrum wavelengths in nanometres.
    signal
        Spectral values.
    minimum_wavelength_nm
        Lower integration limit in nanometres.
    maximum_wavelength_nm
        Upper integration limit in nanometres.
    interpolate_limits
        If ``True``, signal values are interpolated exactly at the
        integration limits when necessary.

    Returns
    -------
    float
        Integrated spectral value.

    Raises
    ------
    ValueError
        If the integration interval is invalid or lies outside the
        available wavelength range.
    """

    wavelength_array = _as_1d_float_array(
        wavelengths_nm,
        name="wavelengths_nm",
    )

    signal_array = _as_1d_float_array(
        signal,
        name="signal",
    )

    _validate_spectrum(
        wavelength_array,
        signal_array,
        require_sorted=True,
    )

    minimum = float(minimum_wavelength_nm)
    maximum = float(maximum_wavelength_nm)

    if minimum >= maximum:
        raise ValueError(
            "minimum_wavelength_nm must be smaller than "
            "maximum_wavelength_nm."
        )

    if (
        minimum < wavelength_array[0]
        or maximum > wavelength_array[-1]
    ):
        raise ValueError(
            "Integration limits must lie within the "
            "available wavelength range."
        )

    mask = (
        (wavelength_array >= minimum)
        & (wavelength_array <= maximum)
    )

    selected_wavelengths = wavelength_array[mask]
    selected_signal = signal_array[mask]

    if interpolate_limits:
        if (
            selected_wavelengths.size == 0
            or selected_wavelengths[0] > minimum
        ):
            lower_signal = float(
                np.interp(
                    minimum,
                    wavelength_array,
                    signal_array,
                )
            )

            selected_wavelengths = np.insert(
                selected_wavelengths,
                0,
                minimum,
            )

            selected_signal = np.insert(
                selected_signal,
                0,
                lower_signal,
            )

        if selected_wavelengths[-1] < maximum:
            upper_signal = float(
                np.interp(
                    maximum,
                    wavelength_array,
                    signal_array,
                )
            )

            selected_wavelengths = np.append(
                selected_wavelengths,
                maximum,
            )

            selected_signal = np.append(
                selected_signal,
                upper_signal,
            )

    if selected_wavelengths.size < 2:
        raise ValueError(
            "At least two wavelength points are required "
            "for integration."
        )

    return _trapezoidal_integral(
        selected_signal,
        selected_wavelengths,
    )


def spectral_centroid(
    wavelengths_nm: Iterable[float] | np.ndarray,
    signal: Iterable[float] | np.ndarray,
) -> float:
    """
    Calculate the signal-weighted mean wavelength.

    Parameters
    ----------
    wavelengths_nm
        Spectrum wavelengths in nanometres.
    signal
        Non-negative spectral values.

    Returns
    -------
    float
        Spectral centroid in nanometres.
    """

    wavelength_array = _as_1d_float_array(
        wavelengths_nm,
        name="wavelengths_nm",
    )

    signal_array = _as_1d_float_array(
        signal,
        name="signal",
    )

    _validate_spectrum(
        wavelength_array,
        signal_array,
        require_sorted=True,
    )

    if np.any(signal_array < 0.0):
        raise ValueError(
            "spectral_centroid requires non-negative signals."
        )

    total_signal = float(np.sum(signal_array))

    if total_signal == 0.0:
        raise ValueError(
            "Cannot calculate the centroid of a zero spectrum."
        )

    return float(
        np.sum(
            wavelength_array * signal_array
        )
        / total_signal
    )


def peak_wavelength(
    wavelengths_nm: Iterable[float] | np.ndarray,
    signal: Iterable[float] | np.ndarray,
) -> float:
    """
    Return the wavelength of the maximum spectral value.

    Parameters
    ----------
    wavelengths_nm
        Spectrum wavelengths in nanometres.
    signal
        Spectral values.

    Returns
    -------
    float
        Wavelength of the first maximum in nanometres.
    """

    wavelength_array = _as_1d_float_array(
        wavelengths_nm,
        name="wavelengths_nm",
    )

    signal_array = _as_1d_float_array(
        signal,
        name="signal",
    )

    _validate_spectrum(
        wavelength_array,
        signal_array,
        require_sorted=True,
    )

    peak_index = int(np.argmax(signal_array))

    return float(wavelength_array[peak_index])


def _estimate_channel_fwhm(
    wavelengths_nm: np.ndarray,
) -> np.ndarray:
    """
    Estimate Gaussian FWHM values from channel spacing.
    """

    if wavelengths_nm.size == 1:
        return np.asarray([20.0], dtype=float)

    spacing = np.diff(wavelengths_nm)

    fwhm = np.empty_like(
        wavelengths_nm,
        dtype=float,
    )

    fwhm[0] = spacing[0]
    fwhm[-1] = spacing[-1]

    if wavelengths_nm.size > 2:
        fwhm[1:-1] = (
            spacing[:-1] + spacing[1:]
        ) / 2.0

    return np.maximum(fwhm, 1.0)


def _validate_spectrum(
    wavelengths_nm: np.ndarray,
    signal: np.ndarray,
    *,
    require_sorted: bool,
) -> None:
    """
    Validate paired wavelength and spectral arrays.
    """

    if wavelengths_nm.size != signal.size:
        raise ValueError(
            "Wavelength and signal arrays must have "
            "the same length."
        )

    if wavelengths_nm.size == 0:
        raise ValueError(
            "Spectrum arrays must not be empty."
        )

    if wavelengths_nm.size < 2:
        raise ValueError(
            "A spectrum requires at least two data points."
        )

    if not np.all(np.isfinite(wavelengths_nm)):
        raise ValueError(
            "Wavelengths must contain only finite values."
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            "Signal must contain only finite values."
        )

    if np.any(wavelengths_nm < 0.0):
        raise ValueError(
            "Wavelengths must be non-negative."
        )

    if require_sorted:
        differences = np.diff(wavelengths_nm)

        if np.any(differences <= 0.0):
            raise ValueError(
                "Wavelengths must be strictly increasing "
                "without duplicates."
            )


def _as_1d_float_array(
    values: Iterable[float] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """
    Convert input values to a one-dimensional float array.
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if array.size == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    return array


def _trapezoidal_integral(
    signal: np.ndarray,
    wavelengths_nm: np.ndarray,
) -> float:
    """
    Integrate using the available NumPy trapezoidal function.
    """

    trapezoid_function = getattr(
        np,
        "trapezoid",
        None,
    )

    if trapezoid_function is not None:
        return float(
            trapezoid_function(
                signal,
                wavelengths_nm,
            )
        )

    return float(
        np.trapz(
            signal,
            wavelengths_nm,
        )
    )