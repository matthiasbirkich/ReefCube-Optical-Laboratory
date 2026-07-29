"""
measurement.py
==============

Measurement objects used throughout the Reef Cube Optical Laboratory.

Author
------
Matthias Birkicht & OpenAI

Version
-------
0.1
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

import numpy as np


@dataclass(slots=True)
class ReefCubeMeasurement:
    """
    Represents one complete Reef Cube measurement.

    One measurement corresponds to one logging interval
    (typically every 30 minutes).

    All values stored here are RAW measurements.

    Derived quantities such as PPFD, CCT, XYZ,
    dominant wavelength, etc. are calculated later by
    the Optical Laboratory and are NOT stored here.
    """

    # ------------------------------------------------------------
    # Time
    # ------------------------------------------------------------

    timestamp: datetime

    # ------------------------------------------------------------
    # Environmental sensors
    # ------------------------------------------------------------

    temperature_C: float

    lux: float

    # ------------------------------------------------------------
    # Spectral sensor
    # ------------------------------------------------------------

    spectral_channels: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------
    # Orientation
    # ------------------------------------------------------------

    ax_g: float = 0.0

    ay_g: float = 0.0

    az_g: float = 1.0

    pitch_deg: float = 0.0

    roll_deg: float = 0.0

    tilt_deg: float = 0.0

    # ------------------------------------------------------------
    # Quality flags
    # ------------------------------------------------------------

    valid: bool = True

    comment: str = ""

    # ------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------

    def channel_names(self):
        """Return the spectral channel names."""

        return list(self.spectral_channels.keys())

    def channel_values(self):
        """Return the spectral values."""

        return np.array(
            list(self.spectral_channels.values()),
            dtype=float
        )

    def spectrum(self):
        """
        Return wavelength-independent spectrum representation.

        Returns
        -------
        tuple(list, ndarray)
            (channel names, values)
        """

        return (
            self.channel_names(),
            self.channel_values()
        )

    def mean_signal(self):
        """
        Mean spectral signal.
        """

        values = self.channel_values()

        if len(values) == 0:
            return np.nan

        return float(np.mean(values))

    def max_signal(self):
        """
        Maximum spectral signal.
        """

        values = self.channel_values()

        if len(values) == 0:
            return np.nan

        return float(np.max(values))

    def min_signal(self):
        """
        Minimum spectral signal.
        """

        values = self.channel_values()

        if len(values) == 0:
            return np.nan

        return float(np.min(values))

    def number_of_channels(self):
        """
        Number of spectral channels.
        """

        return len(self.spectral_channels)

    def summary(self):
        """
        Human-readable summary.
        """

        return (
            f"Time       : {self.timestamp}\n"
            f"Temperature: {self.temperature_C:.2f} °C\n"
            f"Lux        : {self.lux:.1f}\n"
            f"Channels   : {self.number_of_channels()}\n"
            f"Pitch      : {self.pitch_deg:.2f}°\n"
            f"Roll       : {self.roll_deg:.2f}°\n"
            f"Tilt       : {self.tilt_deg:.2f}°"
        )
