"""
Reef Cube Optical Laboratory
----------------------------

Central configuration file.

Author : Matthias Birkicht & OpenAI
Version: 4.0
"""

from dataclasses import dataclass


@dataclass
class ReefCubeConfig:
    """
    Global project configuration.
    """

    # ------------------------------------------------------------------
    # Project
    # ------------------------------------------------------------------

    PROJECT_NAME: str = "Reef Cube"

    VERSION: str = "4.0"

    LOCATION: str = "Barcelona"

    SIMULATION_MODE: bool = True

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    SAMPLE_INTERVAL_SECONDS: int = 1800

    DEPLOYMENT_DEPTH_M: float = 5.0

    TEMPERATURE_C: float = 25.0

    SALINITY_PSU: float = 35.0

    # ------------------------------------------------------------------
    # Optical system
    # ------------------------------------------------------------------

    PTFE_THICKNESS_MM: float = 0.30

    PTFE_LAYERS: int = 3

    REFERENCE_SENSOR: str = "TRIOS RAMSES"

    # ------------------------------------------------------------------
    # AS7343
    # ------------------------------------------------------------------

    AS7343_GAIN: int = 128

    AS7343_INTEGRATION_MS: float = 100.0

    AUTO_GAIN: bool = True

    # ------------------------------------------------------------------
    # BH1750
    # ------------------------------------------------------------------

    BH1750_MTREG: int = 69

    # ------------------------------------------------------------------
    # BMA400
    # ------------------------------------------------------------------

    ENABLE_TILT_CORRECTION: bool = True

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    SAVE_FIGURES: bool = True

    EXPORT_CSV: bool = True

    EXPORT_PDF: bool = False
