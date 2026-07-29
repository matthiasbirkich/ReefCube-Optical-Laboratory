"""
wavelength.py
=============

Spectral channel definitions for the Reef Cube Optical Laboratory.

The wavelength values are the typical peak wavelengths specified for
the ams OSRAM AS7343 multispectral sensor. The bandwidth values are
typical full widths at half maximum (FWHM).

Clear/VIS and flicker channels are not assigned a single wavelength
because they have broad spectral responses.

Author
------
Matthias Birkicht & OpenAI

Version
-------
1.0
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Iterable


@dataclass(frozen=True, slots=True)
class SpectralChannel:
    """
    Description of one wavelength-selective sensor channel.

    Parameters
    ----------
    name
        Official AS7343 channel name.
    peak_wavelength_nm
        Typical peak wavelength in nanometres.
    fwhm_nm
        Typical full width at half maximum in nanometres.
    description
        Short description of the spectral region.
    """

    name: str
    peak_wavelength_nm: float
    fwhm_nm: float
    description: str


# ------------------------------------------------------------------
# Official AS7343 wavelength-selective channels
# ------------------------------------------------------------------

AS7343_CHANNELS: Final = MappingProxyType(
    {
        "F1": SpectralChannel(
            name="F1",
            peak_wavelength_nm=405.0,
            fwhm_nm=30.0,
            description="violet",
        ),
        "F2": SpectralChannel(
            name="F2",
            peak_wavelength_nm=425.0,
            fwhm_nm=22.0,
            description="violet-blue",
        ),
        "FZ": SpectralChannel(
            name="FZ",
            peak_wavelength_nm=450.0,
            fwhm_nm=55.0,
            description="blue / CIE Z-related",
        ),
        "F3": SpectralChannel(
            name="F3",
            peak_wavelength_nm=475.0,
            fwhm_nm=30.0,
            description="blue-cyan",
        ),
        "F4": SpectralChannel(
            name="F4",
            peak_wavelength_nm=515.0,
            fwhm_nm=40.0,
            description="green",
        ),
        "F5": SpectralChannel(
            name="F5",
            peak_wavelength_nm=550.0,
            fwhm_nm=35.0,
            description="green-yellow",
        ),
        "FY": SpectralChannel(
            name="FY",
            peak_wavelength_nm=555.0,
            fwhm_nm=100.0,
            description="photopic green / CIE Y-related",
        ),
        "FXL": SpectralChannel(
            name="FXL",
            peak_wavelength_nm=600.0,
            fwhm_nm=80.0,
            description="orange / long-wave CIE X-related",
        ),
        "F6": SpectralChannel(
            name="F6",
            peak_wavelength_nm=640.0,
            fwhm_nm=50.0,
            description="red",
        ),
        "F7": SpectralChannel(
            name="F7",
            peak_wavelength_nm=690.0,
            fwhm_nm=55.0,
            description="deep red",
        ),
        "F8": SpectralChannel(
            name="F8",
            peak_wavelength_nm=745.0,
            fwhm_nm=60.0,
            description="far red",
        ),
        "NIR": SpectralChannel(
            name="NIR",
            peak_wavelength_nm=855.0,
            fwhm_nm=54.0,
            description="near infrared",
        ),
    }
)


# Canonical order from shortest to longest peak wavelength.
AS7343_CHANNEL_ORDER: Final[tuple[str, ...]] = tuple(
    AS7343_CHANNELS.keys()
)

AS7343_PEAK_WAVELENGTHS_NM: Final = MappingProxyType(
    {
        name: channel.peak_wavelength_nm
        for name, channel in AS7343_CHANNELS.items()
    }
)

AS7343_FWHM_NM: Final = MappingProxyType(
    {
        name: channel.fwhm_nm
        for name, channel in AS7343_CHANNELS.items()
    }
)


# Broad channels without one representative peak wavelength.
AS7343_BROAD_CHANNELS: Final[tuple[str, ...]] = (
    "VIS",
    "CLEAR",
    "FLICKER",
)


def get_channel(channel_name: str) -> SpectralChannel:
    """
    Return the definition of an AS7343 spectral channel.

    Parameters
    ----------
    channel_name
        Channel name such as ``"F1"``, ``"FZ"`` or ``"NIR"``.
        Matching is case-insensitive and surrounding whitespace is
        ignored.

    Returns
    -------
    SpectralChannel
        Immutable channel description.

    Raises
    ------
    KeyError
        If the channel is unknown or has no defined peak wavelength.
    """

    normalized_name = _normalize_channel_name(channel_name)

    try:
        return AS7343_CHANNELS[normalized_name]
    except KeyError as exc:
        if normalized_name in AS7343_BROAD_CHANNELS:
            raise KeyError(
                f"Channel {normalized_name!r} has a broad response "
                "and no single defined peak wavelength."
            ) from exc

        valid_names = ", ".join(AS7343_CHANNEL_ORDER)
        raise KeyError(
            f"Unknown AS7343 channel {channel_name!r}. "
            f"Valid spectral channels are: {valid_names}."
        ) from exc


def get_peak_wavelength_nm(channel_name: str) -> float:
    """
    Return the typical peak wavelength of a channel.

    Parameters
    ----------
    channel_name
        AS7343 spectral channel name.

    Returns
    -------
    float
        Typical peak wavelength in nanometres.
    """

    return get_channel(channel_name).peak_wavelength_nm


def get_fwhm_nm(channel_name: str) -> float:
    """
    Return the typical channel bandwidth.

    Parameters
    ----------
    channel_name
        AS7343 spectral channel name.

    Returns
    -------
    float
        Typical full width at half maximum in nanometres.
    """

    return get_channel(channel_name).fwhm_nm


def ordered_channel_names(
    channel_names: Iterable[str],
    *,
    ignore_unknown: bool = False,
) -> list[str]:
    """
    Sort channel names by increasing peak wavelength.

    Parameters
    ----------
    channel_names
        Iterable containing AS7343 channel names.
    ignore_unknown
        If ``True``, unknown and broad-response channels are omitted.
        If ``False``, an invalid channel raises ``KeyError``.

    Returns
    -------
    list[str]
        Normalized channel names ordered by peak wavelength.
    """

    valid_names: list[str] = []

    for name in channel_names:
        normalized_name = _normalize_channel_name(name)

        if normalized_name in AS7343_CHANNELS:
            valid_names.append(normalized_name)
        elif not ignore_unknown:
            get_channel(normalized_name)

    return sorted(
        valid_names,
        key=lambda name: AS7343_CHANNELS[name].peak_wavelength_nm,
    )


def wavelengths_for_channels(
    channel_names: Iterable[str],
    *,
    ignore_unknown: bool = False,
) -> list[float]:
    """
    Return peak wavelengths corresponding to supplied channels.

    The returned values preserve the input order.

    Parameters
    ----------
    channel_names
        Iterable containing AS7343 channel names.
    ignore_unknown
        If ``True``, unknown and broad-response channels are omitted.
        If ``False``, an invalid channel raises ``KeyError``.

    Returns
    -------
    list[float]
        Peak wavelengths in nanometres.
    """

    wavelengths: list[float] = []

    for name in channel_names:
        try:
            wavelengths.append(get_peak_wavelength_nm(name))
        except KeyError:
            if not ignore_unknown:
                raise

    return wavelengths


def channel_names_in_range(
    minimum_wavelength_nm: float,
    maximum_wavelength_nm: float,
) -> list[str]:
    """
    Return channels whose peak wavelengths lie within a range.

    Both limits are inclusive.

    Parameters
    ----------
    minimum_wavelength_nm
        Lower wavelength limit in nanometres.
    maximum_wavelength_nm
        Upper wavelength limit in nanometres.

    Returns
    -------
    list[str]
        Channel names ordered by increasing wavelength.

    Raises
    ------
    ValueError
        If either limit is invalid or the minimum exceeds the maximum.
    """

    minimum = float(minimum_wavelength_nm)
    maximum = float(maximum_wavelength_nm)

    if minimum < 0.0 or maximum < 0.0:
        raise ValueError("Wavelength limits must be non-negative.")

    if minimum > maximum:
        raise ValueError(
            "minimum_wavelength_nm must not exceed "
            "maximum_wavelength_nm."
        )

    return [
        name
        for name in AS7343_CHANNEL_ORDER
        if minimum
        <= AS7343_CHANNELS[name].peak_wavelength_nm
        <= maximum
    ]


def _normalize_channel_name(channel_name: str) -> str:
    """
    Normalize and validate a channel-name argument.
    """

    if not isinstance(channel_name, str):
        raise TypeError("channel_name must be a string.")

    normalized_name = channel_name.strip().upper()

    if not normalized_name:
        raise ValueError("channel_name must not be empty.")

    return normalized_name
