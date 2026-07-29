"""
ppfd.py
=======

Photosynthetic photon flux density calculations for the Reef Cube
Optical Laboratory.

This module clearly separates:

1. Physically calibrated PPFD calculations from spectral irradiance.
2. Relative PPFD indices derived from uncalibrated sensor signals.
3. Empirical conversion of relative indices using an external
   reference calibration.

Physical PPFD calculations require spectral irradiance in:

    W m^-2 nm^-1

and return PPFD in:

    micromol m^-2 s^-1

Raw AS7343 channel values are not spectral irradiance. They may only
be converted to physical PPFD after calibration against a traceable
reference instrument.

Author
------
Matthias Birkicht & OpenAI

Version
-------
1.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from reefcube.measurement import ReefCubeMeasurement
from reefcube.spectroscopy import measurement_to_arrays


PLANCK_CONSTANT_J_S: float = 6.626_070_15e-34
"""Planck constant in joule seconds."""

SPEED_OF_LIGHT_M_S: float = 299_792_458.0
"""Speed of light in metres per second."""

AVOGADRO_CONSTANT_MOL: float = 6.022_140_76e23
"""Avogadro constant in reciprocal moles."""

PAR_MINIMUM_NM: float = 400.0
"""Conventional lower wavelength limit of PAR."""

PAR_MAXIMUM_NM: float = 700.0
"""Conventional upper wavelength limit of PAR."""


@dataclass(frozen=True, slots=True)
class PPFDCalibration:
    """
    Linear calibration for converting a relative PPFD index to PPFD.

    The calibration equation is:

    ``PPFD = slope * relative_index + intercept``

    Parameters
    ----------
    slope
        Calibration slope.
    intercept
        Calibration intercept in micromoles per square metre per
        second.
    minimum_index
        Optional minimum relative index covered by the calibration.
    maximum_index
        Optional maximum relative index covered by the calibration.
    reference_name
        Optional name of the reference instrument.
    """

    slope: float
    intercept: float = 0.0
    minimum_index: float | None = None
    maximum_index: float | None = None
    reference_name: str | None = None

    def __post_init__(self) -> None:
        """
        Validate calibration coefficients and optional limits.
        """

        values = {
            "slope": self.slope,
            "intercept": self.intercept,
        }

        for name, value in values.items():
            if not np.isfinite(value):
                raise ValueError(
                    f"{name} must be finite."
                )

        if self.minimum_index is not None:
            if not np.isfinite(self.minimum_index):
                raise ValueError(
                    "minimum_index must be finite."
                )

        if self.maximum_index is not None:
            if not np.isfinite(self.maximum_index):
                raise ValueError(
                    "maximum_index must be finite."
                )

        if (
            self.minimum_index is not None
            and self.maximum_index is not None
            and self.minimum_index >= self.maximum_index
        ):
            raise ValueError(
                "minimum_index must be smaller than "
                "maximum_index."
            )

    def convert(
        self,
        relative_index: float,
        *,
        allow_extrapolation: bool = False,
    ) -> float:
        """
        Convert one relative PPFD index to calibrated PPFD.

        Parameters
        ----------
        relative_index
            Relative PPFD index from the sensor.
        allow_extrapolation
            If ``False``, values outside the calibration interval
            raise ``ValueError``.

        Returns
        -------
        float
            Estimated PPFD in micromoles per square metre per second.
        """

        index = float(relative_index)

        if not np.isfinite(index):
            raise ValueError(
                "relative_index must be finite."
            )

        if not allow_extrapolation:
            if (
                self.minimum_index is not None
                and index < self.minimum_index
            ):
                raise ValueError(
                    "relative_index lies below the calibrated range."
                )

            if (
                self.maximum_index is not None
                and index > self.maximum_index
            ):
                raise ValueError(
                    "relative_index lies above the calibrated range."
                )

        return float(
            self.slope * index + self.intercept
        )


def photon_energy_joule(
    wavelength_nm: float | np.ndarray,
) -> float | np.ndarray:
    """
    Calculate photon energy for one or more wavelengths.

    Parameters
    ----------
    wavelength_nm
        Wavelength in nanometres.

    Returns
    -------
    float or numpy.ndarray
        Photon energy in joules.

    Raises
    ------
    ValueError
        If wavelengths are non-positive or non-finite.
    """

    wavelengths = np.asarray(
        wavelength_nm,
        dtype=float,
    )

    if not np.all(np.isfinite(wavelengths)):
        raise ValueError(
            "Wavelengths must be finite."
        )

    if np.any(wavelengths <= 0.0):
        raise ValueError(
            "Wavelengths must be greater than zero."
        )

    wavelengths_m = wavelengths * 1e-9

    energy = (
        PLANCK_CONSTANT_J_S
        * SPEED_OF_LIGHT_M_S
        / wavelengths_m
    )

    if energy.ndim == 0:
        return float(energy)

    return energy


def spectral_irradiance_to_photon_flux_density(
    wavelengths_nm: Iterable[float] | np.ndarray,
    spectral_irradiance_W_m2_nm: (
        Iterable[float] | np.ndarray
    ),
) -> np.ndarray:
    """
    Convert spectral irradiance to spectral photon flux density.

    Parameters
    ----------
    wavelengths_nm
        Wavelengths in nanometres.
    spectral_irradiance_W_m2_nm
        Spectral irradiance in watts per square metre per nanometre.

    Returns
    -------
    numpy.ndarray
        Spectral photon flux density in micromoles per square metre
        per second per nanometre.

    Notes
    -----
    The conversion uses:

    ``photons = energy * wavelength / (h * c)``

    followed by conversion from photons to micromoles using the
    Avogadro constant.
    """

    wavelengths, irradiance = _validate_spectral_irradiance(
        wavelengths_nm,
        spectral_irradiance_W_m2_nm,
    )

    wavelengths_m = wavelengths * 1e-9

    photon_flux_photons = (
        irradiance
        * wavelengths_m
        / (
            PLANCK_CONSTANT_J_S
            * SPEED_OF_LIGHT_M_S
        )
    )

    photon_flux_micromol = (
        photon_flux_photons
        / AVOGADRO_CONSTANT_MOL
        * 1e6
    )

    return photon_flux_micromol


def spectral_irradiance_to_ppfd(
    wavelengths_nm: Iterable[float] | np.ndarray,
    spectral_irradiance_W_m2_nm: (
        Iterable[float] | np.ndarray
    ),
    *,
    minimum_wavelength_nm: float = PAR_MINIMUM_NM,
    maximum_wavelength_nm: float = PAR_MAXIMUM_NM,
    interpolate_limits: bool = True,
) -> float:
    """
    Calculate PPFD from calibrated spectral irradiance.

    Parameters
    ----------
    wavelengths_nm
        Wavelengths in nanometres.
    spectral_irradiance_W_m2_nm
        Spectral irradiance in watts per square metre per nanometre.
    minimum_wavelength_nm
        Lower integration limit. The conventional PAR limit is
        400 nm.
    maximum_wavelength_nm
        Upper integration limit. The conventional PAR limit is
        700 nm.
    interpolate_limits
        If ``True``, spectral values are interpolated at the exact
        integration limits.

    Returns
    -------
    float
        PPFD in micromoles per square metre per second.

    Raises
    ------
    ValueError
        If the wavelength interval is invalid or unavailable.
    """

    wavelengths, irradiance = _validate_spectral_irradiance(
        wavelengths_nm,
        spectral_irradiance_W_m2_nm,
    )

    minimum = float(minimum_wavelength_nm)
    maximum = float(maximum_wavelength_nm)

    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError(
            "Integration limits must be finite."
        )

    if minimum >= maximum:
        raise ValueError(
            "minimum_wavelength_nm must be smaller than "
            "maximum_wavelength_nm."
        )

    if (
        minimum < wavelengths[0]
        or maximum > wavelengths[-1]
    ):
        raise ValueError(
            "Integration limits must lie within the "
            "available wavelength range."
        )

    selected_wavelengths, selected_irradiance = (
        _select_wavelength_interval(
            wavelengths,
            irradiance,
            minimum,
            maximum,
            interpolate_limits=interpolate_limits,
        )
    )

    spectral_photon_flux = (
        spectral_irradiance_to_photon_flux_density(
            selected_wavelengths,
            selected_irradiance,
        )
    )

    return _trapezoidal_integral(
        spectral_photon_flux,
        selected_wavelengths,
    )


def wavelength_band_ppfd(
    wavelengths_nm: Iterable[float] | np.ndarray,
    spectral_irradiance_W_m2_nm: (
        Iterable[float] | np.ndarray
    ),
    minimum_wavelength_nm: float,
    maximum_wavelength_nm: float,
) -> float:
    """
    Calculate photon flux density within any wavelength band.

    Parameters
    ----------
    wavelengths_nm
        Wavelengths in nanometres.
    spectral_irradiance_W_m2_nm
        Spectral irradiance in watts per square metre per nanometre.
    minimum_wavelength_nm
        Lower integration limit.
    maximum_wavelength_nm
        Upper integration limit.

    Returns
    -------
    float
        Photon flux density in micromoles per square metre per second.
    """

    return spectral_irradiance_to_ppfd(
        wavelengths_nm,
        spectral_irradiance_W_m2_nm,
        minimum_wavelength_nm=minimum_wavelength_nm,
        maximum_wavelength_nm=maximum_wavelength_nm,
    )


def relative_ppfd_index(
    wavelengths_nm: Iterable[float] | np.ndarray,
    signal: Iterable[float] | np.ndarray,
    *,
    minimum_wavelength_nm: float = PAR_MINIMUM_NM,
    maximum_wavelength_nm: float = PAR_MAXIMUM_NM,
    photon_weighted: bool = True,
) -> float:
    """
    Calculate a relative PPFD index from uncalibrated spectral data.

    This function does not produce a physical PPFD value. The result
    remains in arbitrary sensor units.

    Parameters
    ----------
    wavelengths_nm
        Sensor channel wavelengths in nanometres.
    signal
        Raw or corrected sensor signals.
    minimum_wavelength_nm
        Lower wavelength limit.
    maximum_wavelength_nm
        Upper wavelength limit.
    photon_weighted
        If ``True``, signals are weighted proportionally to wavelength
        because longer-wavelength photons contain less energy.

    Returns
    -------
    float
        Relative PPFD index in arbitrary units.

    Notes
    -----
    The index is useful for:

    - comparing repeated measurements,
    - examining temporal trends,
    - calibration against a reference quantum sensor,
    - testing processing algorithms.

    It must not be labelled as micromoles per square metre per second
    unless an empirical calibration has been applied.
    """

    wavelengths, signals = _validate_relative_spectrum(
        wavelengths_nm,
        signal,
    )

    minimum = float(minimum_wavelength_nm)
    maximum = float(maximum_wavelength_nm)

    if minimum >= maximum:
        raise ValueError(
            "minimum_wavelength_nm must be smaller than "
            "maximum_wavelength_nm."
        )

    mask = (
        (wavelengths >= minimum)
        & (wavelengths <= maximum)
    )

    selected_wavelengths = wavelengths[mask]
    selected_signals = signals[mask]

    if selected_wavelengths.size < 2:
        raise ValueError(
            "At least two channels are required within "
            "the selected wavelength range."
        )

    channel_widths = _channel_effective_widths(
        selected_wavelengths
    )

    if photon_weighted:
        weights = (
            selected_wavelengths
            * channel_widths
        )
    else:
        weights = channel_widths

    return float(
        np.sum(
            selected_signals * weights
        )
    )


def measurement_relative_ppfd_index(
    measurement: ReefCubeMeasurement,
    *,
    minimum_wavelength_nm: float = PAR_MINIMUM_NM,
    maximum_wavelength_nm: float = PAR_MAXIMUM_NM,
    photon_weighted: bool = True,
) -> float:
    """
    Calculate a relative PPFD index from a Reef Cube measurement.

    Parameters
    ----------
    measurement
        Reef Cube measurement containing AS7343 spectral channels.
    minimum_wavelength_nm
        Lower wavelength limit.
    maximum_wavelength_nm
        Upper wavelength limit.
    photon_weighted
        Apply wavelength-based photon weighting.

    Returns
    -------
    float
        Relative PPFD index in arbitrary units.
    """

    wavelengths, signals = measurement_to_arrays(
        measurement
    )

    return relative_ppfd_index(
        wavelengths,
        signals,
        minimum_wavelength_nm=minimum_wavelength_nm,
        maximum_wavelength_nm=maximum_wavelength_nm,
        photon_weighted=photon_weighted,
    )


def estimate_measurement_ppfd(
    measurement: ReefCubeMeasurement,
    calibration: PPFDCalibration,
    *,
    allow_extrapolation: bool = False,
    photon_weighted: bool = True,
) -> float:
    """
    Estimate PPFD from a calibrated Reef Cube measurement.

    Parameters
    ----------
    measurement
        Reef Cube measurement containing spectral channels.
    calibration
        Linear calibration relating the relative index to reference
        PPFD.
    allow_extrapolation
        Permit values outside the calibration interval.
    photon_weighted
        Apply wavelength-based photon weighting when calculating the
        relative index.

    Returns
    -------
    float
        Estimated PPFD in micromoles per square metre per second.
    """

    if not isinstance(calibration, PPFDCalibration):
        raise TypeError(
            "calibration must be a PPFDCalibration object."
        )

    index = measurement_relative_ppfd_index(
        measurement,
        photon_weighted=photon_weighted,
    )

    return calibration.convert(
        index,
        allow_extrapolation=allow_extrapolation,
    )


def _validate_spectral_irradiance(
    wavelengths_nm: Iterable[float] | np.ndarray,
    spectral_irradiance_W_m2_nm: (
        Iterable[float] | np.ndarray
    ),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate calibrated spectral irradiance arrays.
    """

    wavelengths = _as_1d_float_array(
        wavelengths_nm,
        name="wavelengths_nm",
    )

    irradiance = _as_1d_float_array(
        spectral_irradiance_W_m2_nm,
        name="spectral_irradiance_W_m2_nm",
    )

    _validate_paired_arrays(
        wavelengths,
        irradiance,
    )

    if np.any(irradiance < 0.0):
        raise ValueError(
            "Spectral irradiance must be non-negative."
        )

    return wavelengths, irradiance


def _validate_relative_spectrum(
    wavelengths_nm: Iterable[float] | np.ndarray,
    signal: Iterable[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate wavelength and relative signal arrays.
    """

    wavelengths = _as_1d_float_array(
        wavelengths_nm,
        name="wavelengths_nm",
    )

    signals = _as_1d_float_array(
        signal,
        name="signal",
    )

    _validate_paired_arrays(
        wavelengths,
        signals,
    )

    if np.any(signals < 0.0):
        raise ValueError(
            "Relative spectral signals must be non-negative."
        )

    return wavelengths, signals


def _validate_paired_arrays(
    wavelengths: np.ndarray,
    values: np.ndarray,
) -> None:
    """
    Validate paired one-dimensional spectral arrays.
    """

    if wavelengths.size != values.size:
        raise ValueError(
            "Wavelength and value arrays must have "
            "the same length."
        )

    if wavelengths.size < 2:
        raise ValueError(
            "At least two spectral points are required."
        )

    if not np.all(np.isfinite(wavelengths)):
        raise ValueError(
            "Wavelengths must contain only finite values."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Spectral values must contain only finite values."
        )

    if np.any(wavelengths <= 0.0):
        raise ValueError(
            "Wavelengths must be greater than zero."
        )

    if np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError(
            "Wavelengths must be strictly increasing "
            "without duplicates."
        )


def _select_wavelength_interval(
    wavelengths: np.ndarray,
    values: np.ndarray,
    minimum: float,
    maximum: float,
    *,
    interpolate_limits: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select a wavelength interval and optionally add exact boundaries.
    """

    mask = (
        (wavelengths >= minimum)
        & (wavelengths <= maximum)
    )

    selected_wavelengths = wavelengths[mask].copy()
    selected_values = values[mask].copy()

    if interpolate_limits:
        if (
            selected_wavelengths.size == 0
            or selected_wavelengths[0] > minimum
        ):
            lower_value = float(
                np.interp(
                    minimum,
                    wavelengths,
                    values,
                )
            )

            selected_wavelengths = np.insert(
                selected_wavelengths,
                0,
                minimum,
            )

            selected_values = np.insert(
                selected_values,
                0,
                lower_value,
            )

        if selected_wavelengths[-1] < maximum:
            upper_value = float(
                np.interp(
                    maximum,
                    wavelengths,
                    values,
                )
            )

            selected_wavelengths = np.append(
                selected_wavelengths,
                maximum,
            )

            selected_values = np.append(
                selected_values,
                upper_value,
            )

    if selected_wavelengths.size < 2:
        raise ValueError(
            "At least two points are required within "
            "the selected wavelength interval."
        )

    return selected_wavelengths, selected_values


def _channel_effective_widths(
    wavelengths_nm: np.ndarray,
) -> np.ndarray:
    """
    Estimate effective integration widths for discrete channels.

    Interior channel widths extend halfway toward each neighbouring
    channel. Edge widths use the distance to the nearest channel.
    """

    if wavelengths_nm.size < 2:
        raise ValueError(
            "At least two channels are required."
        )

    differences = np.diff(wavelengths_nm)

    widths = np.empty_like(
        wavelengths_nm,
        dtype=float,
    )

    widths[0] = differences[0]
    widths[-1] = differences[-1]

    if wavelengths_nm.size > 2:
        widths[1:-1] = (
            differences[:-1]
            + differences[1:]
        ) / 2.0

    return widths


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
    values: np.ndarray,
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
                values,
                wavelengths_nm,
            )
        )

    return float(
        np.trapz(
            values,
            wavelengths_nm,
        )
    )
