"""
sensors.py
==========

Unified measurement-source interfaces for the Reef Cube Optical Laboratory.

This module currently provides:

- SensorBase
- SimulationSensor
- ReefCubeSensor

The SimulationSensor produces complete ReefCubeMeasurement objects with
realistic synthetic AS7343 channel patterns for sunny, cloudy, and
underwater conditions.

Author
------
Matthias Birkicht & OpenAI

Version
-------
1.1
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import math
import random
from typing import Final

from reefcube.measurement import ReefCubeMeasurement
from reefcube.wavelength import AS7343_CHANNEL_ORDER


SIMULATION_MODES: Final[tuple[str, ...]] = (
    "sunny",
    "cloudy",
    "underwater",
)


class SensorBase(ABC):
    """
    Abstract base class for all Reef Cube measurement sources.
    """

    @abstractmethod
    def acquire(self) -> ReefCubeMeasurement:
        """
        Acquire one complete Reef Cube measurement.

        Returns
        -------
        ReefCubeMeasurement
            One synchronized measurement record.
        """

        raise NotImplementedError


class SimulationSensor(SensorBase):
    """
    Generate realistic synthetic Reef Cube measurements.

    The simulated measurements use the official wavelength-selective
    AS7343 channel names:

    F1, F2, FZ, F3, F4, F5, FY, FXL, F6, F7, F8, NIR

    Parameters
    ----------
    mode
        Simulation mode. Supported values are ``"sunny"``,
        ``"cloudy"`` and ``"underwater"``.
    seed
        Optional random seed. Using the same seed produces the same
        sequence of measurements.
    spectral_noise_fraction
        Relative random noise applied independently to every spectral
        channel. For example, ``0.03`` corresponds to approximately
        three percent noise.
    orientation_noise_deg
        Typical simulated orientation variation in degrees.

    Raises
    ------
    ValueError
        If the simulation mode or noise parameters are invalid.
    """

    _SPECTRAL_PROFILES: Final[dict[str, dict[str, float]]] = {
        "sunny": {
            "F1": 0.43,
            "F2": 0.50,
            "FZ": 0.62,
            "F3": 0.73,
            "F4": 0.86,
            "F5": 0.95,
            "FY": 1.00,
            "FXL": 0.97,
            "F6": 0.92,
            "F7": 0.86,
            "F8": 0.78,
            "NIR": 0.74,
        },
        "cloudy": {
            "F1": 0.57,
            "F2": 0.64,
            "FZ": 0.76,
            "F3": 0.86,
            "F4": 0.94,
            "F5": 0.98,
            "FY": 1.00,
            "FXL": 0.96,
            "F6": 0.90,
            "F7": 0.82,
            "F8": 0.72,
            "NIR": 0.66,
        },
        "underwater": {
            "F1": 0.48,
            "F2": 0.65,
            "FZ": 0.88,
            "F3": 1.00,
            "F4": 0.92,
            "F5": 0.72,
            "FY": 0.65,
            "FXL": 0.40,
            "F6": 0.24,
            "F7": 0.12,
            "F8": 0.06,
            "NIR": 0.02,
        },
    }

    _LUX_RANGES: Final[dict[str, tuple[float, float]]] = {
        "sunny": (80_000.0, 110_000.0),
        "cloudy": (5_000.0, 20_000.0),
        "underwater": (500.0, 15_000.0),
    }

    _TEMPERATURE_RANGES_C: Final[dict[str, tuple[float, float]]] = {
        "sunny": (28.0, 32.0),
        "cloudy": (18.0, 24.0),
        "underwater": (22.0, 29.0),
    }

    _SIGNAL_SCALE_RANGES: Final[dict[str, tuple[float, float]]] = {
        "sunny": (35_000.0, 55_000.0),
        "cloudy": (8_000.0, 20_000.0),
        "underwater": (2_000.0, 18_000.0),
    }

    def __init__(
        self,
        mode: str = "underwater",
        seed: int | None = None,
        spectral_noise_fraction: float = 0.03,
        orientation_noise_deg: float = 2.0,
    ) -> None:
        normalized_mode = self._normalize_mode(mode)

        if spectral_noise_fraction < 0.0:
            raise ValueError(
                "spectral_noise_fraction must be non-negative."
            )

        if orientation_noise_deg < 0.0:
            raise ValueError(
                "orientation_noise_deg must be non-negative."
            )

        self.mode = normalized_mode
        self.spectral_noise_fraction = float(
            spectral_noise_fraction
        )
        self.orientation_noise_deg = float(
            orientation_noise_deg
        )
        self._random = random.Random(seed)

    def acquire(self) -> ReefCubeMeasurement:
        """
        Generate one simulated measurement.

        Returns
        -------
        ReefCubeMeasurement
            Synthetic synchronized Reef Cube measurement.
        """

        lux = self._random_value(
            self._LUX_RANGES[self.mode]
        )

        temperature_C = self._random_value(
            self._TEMPERATURE_RANGES_C[self.mode]
        )

        signal_scale = self._random_value(
            self._SIGNAL_SCALE_RANGES[self.mode]
        )

        spectral_channels = self._generate_spectral_channels(
            signal_scale=signal_scale
        )

        pitch_deg = self._random.gauss(
            0.0,
            self.orientation_noise_deg,
        )

        roll_deg = self._random.gauss(
            0.0,
            self.orientation_noise_deg,
        )

        tilt_deg = math.sqrt(
            pitch_deg**2 + roll_deg**2
        )

        ax_g, ay_g, az_g = self._orientation_to_acceleration(
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
        )

        return ReefCubeMeasurement(
            timestamp=datetime.now(timezone.utc),
            temperature_C=temperature_C,
            lux=lux,
            spectral_channels=spectral_channels,
            ax_g=ax_g,
            ay_g=ay_g,
            az_g=az_g,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            tilt_deg=tilt_deg,
            valid=True,
            comment=f"Simulated {self.mode} measurement",
        )

    def set_mode(self, mode: str) -> None:
        """
        Change the active simulation mode.

        Parameters
        ----------
        mode
            New simulation mode.
        """

        self.mode = self._normalize_mode(mode)

    def _generate_spectral_channels(
        self,
        signal_scale: float,
    ) -> dict[str, float]:
        """
        Generate one complete simulated AS7343 channel set.
        """

        profile = self._SPECTRAL_PROFILES[self.mode]
        spectral_channels: dict[str, float] = {}

        for channel_name in AS7343_CHANNEL_ORDER:
            base_signal = (
                signal_scale * profile[channel_name]
            )

            noise_standard_deviation = (
                base_signal * self.spectral_noise_fraction
            )

            noisy_signal = self._random.gauss(
                base_signal,
                noise_standard_deviation,
            )

            spectral_channels[channel_name] = max(
                0.0,
                float(noisy_signal),
            )

        return spectral_channels

    def _orientation_to_acceleration(
        self,
        pitch_deg: float,
        roll_deg: float,
    ) -> tuple[float, float, float]:
        """
        Convert pitch and roll to an idealized gravity vector.
        """

        pitch_rad = math.radians(pitch_deg)
        roll_rad = math.radians(roll_deg)

        ax_g = -math.sin(pitch_rad)

        ay_g = (
            math.sin(roll_rad)
            * math.cos(pitch_rad)
        )

        az_g = (
            math.cos(roll_rad)
            * math.cos(pitch_rad)
        )

        acceleration_noise = 0.003

        ax_g += self._random.gauss(
            0.0,
            acceleration_noise,
        )

        ay_g += self._random.gauss(
            0.0,
            acceleration_noise,
        )

        az_g += self._random.gauss(
            0.0,
            acceleration_noise,
        )

        return (
            float(ax_g),
            float(ay_g),
            float(az_g),
        )

    def _random_value(
        self,
        value_range: tuple[float, float],
    ) -> float:
        """
        Return a random floating-point value within a range.
        """

        minimum, maximum = value_range

        return float(
            self._random.uniform(minimum, maximum)
        )

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        """
        Normalize and validate a simulation mode.
        """

        if not isinstance(mode, str):
            raise TypeError("mode must be a string.")

        normalized_mode = mode.strip().lower()

        if normalized_mode not in SIMULATION_MODES:
            valid_modes = ", ".join(SIMULATION_MODES)

            raise ValueError(
                f"Unknown simulation mode {mode!r}. "
                f"Valid modes are: {valid_modes}."
            )

        return normalized_mode


class ReefCubeSensor(SensorBase):
    """
    Public interface for measurements from the physical Reef Cube.

    This class reserves the stable API that future BLE, USB, serial
    or stored-data implementations will use.

    The scientific analysis code should depend only on ``acquire()``
    and not on the underlying transport mechanism.
    """

    def acquire(self) -> ReefCubeMeasurement:
        """
        Acquire one physical Reef Cube measurement.

        Raises
        ------
        NotImplementedError
            The hardware transport interface is not yet implemented.
        """

        raise NotImplementedError(
            "The physical Reef Cube interface is not implemented yet."
        )