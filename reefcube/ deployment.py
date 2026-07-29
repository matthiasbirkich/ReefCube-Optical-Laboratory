"""
deployment.py
=============

Deployment metadata for the Reef Cube Optical Laboratory.

Author
------
Matthias Birkicht & OpenAI

Version
-------
0.1
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class DeploymentConfig:
    """
    Metadata describing a Reef Cube deployment.

    These values normally remain constant during one deployment and
    are transferred from the smartphone to the logger before logging
    starts.
    """

    deployment_name: str

    station: str

    latitude: float

    longitude: float

    start_datetime: datetime

    salinity_psu: float

    operator: str = ""

    notes: str = ""

    timezone: str = "UTC"

    country: str = ""

    location: str = ""

    def summary(self) -> str:
        """
        Return a short human-readable deployment summary.
        """

        return (
            f"{self.deployment_name}\n"
            f"Station : {self.station}\n"
            f"Location: {self.location}\n"
            f"Lat/Lon : {self.latitude:.6f}, "
            f"{self.longitude:.6f}\n"
            f"Start   : {self.start_datetime}\n"
            f"Salinity: {self.salinity_psu:.1f} PSU"
        )
