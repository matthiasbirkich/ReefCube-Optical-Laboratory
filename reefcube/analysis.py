"""
analysis.py
===========

Statistical and time-series analysis tools for the Reef Cube Optical
Laboratory.

The module provides:

- extraction of measurement fields
- validity and time-range filtering
- descriptive statistics
- spectral-channel summaries
- rolling averages
- elapsed-time calculations
- sampling-interval diagnostics
- comparison with reference measurements
- linear interpolation of reference data onto Reef Cube timestamps
- RMSE, MAE, bias, correlation, and regression statistics

The functions accept both ReefCubeMeasurement objects and dictionaries
loaded through ``reefcube.storage``.

Author
------
Matthias Birkicht & OpenAI

Version
-------
1.1
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DescriptiveStatistics:
    """
    Descriptive statistics for one numeric variable.

    Attributes
    ----------
    count
        Number of finite observations.
    mean
        Arithmetic mean.
    standard_deviation
        Sample standard deviation using ``ddof=1``.
    minimum
        Minimum observation.
    maximum
        Maximum observation.
    median
        Median observation.
    q25
        First quartile.
    q75
        Third quartile.
    """

    count: int
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    median: float
    q25: float
    q75: float

    def summary(
        self,
        *,
        precision: int = 6,
    ) -> str:
        """
        Return a readable statistical summary.

        Parameters
        ----------
        precision
            Number of decimal places.

        Returns
        -------
        str
            Multi-line summary.
        """

        if precision < 0:
            raise ValueError(
                "precision must be non-negative."
            )

        return "\n".join(
            [
                f"Count     : {self.count}",
                f"Mean      : {self.mean:.{precision}f}",
                (
                    "Std. dev. : "
                    f"{self.standard_deviation:.{precision}f}"
                ),
                f"Minimum   : {self.minimum:.{precision}f}",
                f"Q25       : {self.q25:.{precision}f}",
                f"Median    : {self.median:.{precision}f}",
                f"Q75       : {self.q75:.{precision}f}",
                f"Maximum   : {self.maximum:.{precision}f}",
            ]
        )


@dataclass(frozen=True)
class SamplingIntervalStatistics:
    """
    Statistics describing measurement sampling intervals.

    Attributes
    ----------
    count
        Number of intervals.
    mean_seconds
        Mean interval in seconds.
    median_seconds
        Median interval in seconds.
    minimum_seconds
        Minimum interval.
    maximum_seconds
        Maximum interval.
    standard_deviation_seconds
        Sample standard deviation of intervals.
    expected_seconds
        Optional expected interval.
    missing_intervals
        Estimated number of missing measurement intervals.
    """

    count: int
    mean_seconds: float
    median_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    standard_deviation_seconds: float
    expected_seconds: float | None
    missing_intervals: int

    def summary(
        self,
        *,
        precision: int = 3,
    ) -> str:
        """
        Return a readable sampling-interval summary.
        """

        expected_text = (
            "not specified"
            if self.expected_seconds is None
            else f"{self.expected_seconds:.{precision}f} s"
        )

        return "\n".join(
            [
                f"Intervals      : {self.count}",
                (
                    "Mean interval  : "
                    f"{self.mean_seconds:.{precision}f} s"
                ),
                (
                    "Median interval: "
                    f"{self.median_seconds:.{precision}f} s"
                ),
                (
                    "Minimum        : "
                    f"{self.minimum_seconds:.{precision}f} s"
                ),
                (
                    "Maximum        : "
                    f"{self.maximum_seconds:.{precision}f} s"
                ),
                (
                    "Std. deviation : "
                    f"{self.standard_deviation_seconds:.{precision}f} s"
                ),
                f"Expected       : {expected_text}",
                f"Missing periods: {self.missing_intervals}",
            ]
        )


@dataclass(frozen=True)
class ComparisonStatistics:
    """
    Statistics comparing measured and reference values.

    Attributes
    ----------
    count
        Number of paired finite observations.
    bias
        Mean of ``measured - reference``.
    mean_absolute_error
        Mean absolute error.
    root_mean_square_error
        Root mean square error.
    percent_bias
        Bias as percentage of the mean reference value.
    mean_absolute_percentage_error
        Mean absolute percentage error, excluding zero references.
    pearson_r
        Pearson correlation coefficient.
    slope
        Ordinary least-squares slope predicting measured values from
        reference values.
    intercept
        Regression intercept.
    r_squared
        Coefficient of determination for the fitted regression.
    """

    count: int
    bias: float
    mean_absolute_error: float
    root_mean_square_error: float
    percent_bias: float
    mean_absolute_percentage_error: float
    pearson_r: float
    slope: float
    intercept: float
    r_squared: float

    def summary(
        self,
        *,
        precision: int = 6,
    ) -> str:
        """
        Return a readable comparison summary.
        """

        return "\n".join(
            [
                f"Pairs      : {self.count}",
                f"Bias       : {self.bias:.{precision}f}",
                (
                    "MAE        : "
                    f"{self.mean_absolute_error:.{precision}f}"
                ),
                (
                    "RMSE       : "
                    f"{self.root_mean_square_error:.{precision}f}"
                ),
                (
                    "Percent bias: "
                    f"{self.percent_bias:.{precision}f} %"
                ),
                (
                    "MAPE       : "
                    f"{self.mean_absolute_percentage_error:.{precision}f} %"
                ),
                f"Pearson r  : {self.pearson_r:.{precision}f}",
                f"Slope      : {self.slope:.{precision}f}",
                f"Intercept  : {self.intercept:.{precision}f}",
                f"R²         : {self.r_squared:.{precision}f}",
            ]
        )


def measurement_field(
    measurement: Any,
    field_name: str,
    default: Any = None,
) -> Any:
    """
    Read a field from a measurement object or mapping.

    Parameters
    ----------
    measurement
        Dataclass-like object or mapping.
    field_name
        Name of the requested field.
    default
        Value returned when the field is absent.

    Returns
    -------
    Any
        Stored field value or ``default``.
    """

    if isinstance(measurement, Mapping):
        return measurement.get(
            field_name,
            default,
        )

    return getattr(
        measurement,
        field_name,
        default,
    )


def extract_numeric_field(
    measurements: Iterable[Any],
    field_name: str,
    *,
    finite_only: bool = True,
) -> np.ndarray:
    """
    Extract one numeric field from a measurement collection.

    Parameters
    ----------
    measurements
        Measurement objects or dictionaries.
    field_name
        Numeric field to extract.
    finite_only
        Exclude NaN and infinite values.

    Returns
    -------
    numpy.ndarray
        One-dimensional float array.
    """

    values: list[float] = []

    for measurement in measurements:
        value = measurement_field(
            measurement,
            field_name,
        )

        if value is None:
            continue

        try:
            numeric_value = float(value)

        except (
            TypeError,
            ValueError,
        ):
            continue

        if finite_only and not np.isfinite(
            numeric_value
        ):
            continue

        values.append(numeric_value)

    return np.asarray(
        values,
        dtype=float,
    )


def extract_timestamps(
    measurements: Iterable[Any],
    *,
    field_name: str = "timestamp",
) -> list[datetime]:
    """
    Extract and normalize measurement timestamps.

    Parameters
    ----------
    measurements
        Measurement objects or dictionaries.
    field_name
        Timestamp field name.

    Returns
    -------
    list[datetime]
        Timezone-aware UTC timestamps.

    Raises
    ------
    ValueError
        If a timestamp is missing or invalid.
    """

    timestamps: list[datetime] = []

    for index, measurement in enumerate(
        measurements
    ):
        value = measurement_field(
            measurement,
            field_name,
        )

        if value is None:
            raise ValueError(
                f"Measurement {index} has no "
                f"{field_name!r} value."
            )

        timestamps.append(
            _parse_datetime(value)
        )

    return timestamps


def filter_valid_measurements(
    measurements: Iterable[Any],
    *,
    valid_field: str = "valid",
    default_valid: bool = True,
) -> list[Any]:
    """
    Retain measurements marked as valid.

    Parameters
    ----------
    measurements
        Measurement collection.
    valid_field
        Field containing the validity flag.
    default_valid
        Validity assumed when the field is absent.

    Returns
    -------
    list
        Valid measurements.
    """

    result: list[Any] = []

    for measurement in measurements:
        valid = measurement_field(
            measurement,
            valid_field,
            default_valid,
        )

        if bool(valid):
            result.append(measurement)

    return result


def filter_time_range(
    measurements: Iterable[Any],
    *,
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    timestamp_field: str = "timestamp",
) -> list[Any]:
    """
    Filter measurements by an inclusive UTC time range.

    Parameters
    ----------
    measurements
        Measurement collection.
    start
        Optional inclusive start time.
    end
        Optional inclusive end time.
    timestamp_field
        Timestamp field name.

    Returns
    -------
    list
        Measurements inside the requested interval.
    """

    start_time = (
        None
        if start is None
        else _parse_datetime(start)
    )

    end_time = (
        None
        if end is None
        else _parse_datetime(end)
    )

    if (
        start_time is not None
        and end_time is not None
        and start_time > end_time
    ):
        raise ValueError(
            "start must not be later than end."
        )

    selected: list[Any] = []

    for measurement in measurements:
        value = measurement_field(
            measurement,
            timestamp_field,
        )

        if value is None:
            continue

        timestamp = _parse_datetime(value)

        if (
            start_time is not None
            and timestamp < start_time
        ):
            continue

        if (
            end_time is not None
            and timestamp > end_time
        ):
            continue

        selected.append(measurement)

    return selected


def sort_measurements_by_time(
    measurements: Iterable[Any],
    *,
    timestamp_field: str = "timestamp",
) -> list[Any]:
    """
    Return measurements ordered by timestamp.
    """

    return sorted(
        measurements,
        key=lambda measurement: _parse_datetime(
            measurement_field(
                measurement,
                timestamp_field,
            )
        ),
    )


def descriptive_statistics(
    values: Iterable[float],
) -> DescriptiveStatistics:
    """
    Calculate descriptive statistics for finite values.

    Parameters
    ----------
    values
        Numeric observations.

    Returns
    -------
    DescriptiveStatistics
        Statistical summary.

    Raises
    ------
    ValueError
        If no finite values are available.
    """

    array = _finite_float_array(
        values,
        name="values",
    )

    count = int(array.size)

    if count == 0:
        raise ValueError(
            "At least one finite value is required."
        )

    standard_deviation = (
        float(np.std(array, ddof=1))
        if count > 1
        else 0.0
    )

    return DescriptiveStatistics(
        count=count,
        mean=float(np.mean(array)),
        standard_deviation=standard_deviation,
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        median=float(np.median(array)),
        q25=float(np.percentile(array, 25.0)),
        q75=float(np.percentile(array, 75.0)),
    )


def measurement_statistics(
    measurements: Iterable[Any],
    field_name: str,
) -> DescriptiveStatistics:
    """
    Calculate descriptive statistics for a measurement field.
    """

    values = extract_numeric_field(
        measurements,
        field_name,
    )

    return descriptive_statistics(values)


def elapsed_seconds(
    measurements: Iterable[Any],
    *,
    timestamp_field: str = "timestamp",
) -> np.ndarray:
    """
    Calculate elapsed seconds from the first measurement.

    Measurements are sorted by timestamp before calculation.

    Returns
    -------
    numpy.ndarray
        Elapsed seconds beginning at zero.
    """

    ordered = sort_measurements_by_time(
        measurements,
        timestamp_field=timestamp_field,
    )

    timestamps = extract_timestamps(
        ordered,
        field_name=timestamp_field,
    )

    if not timestamps:
        return np.asarray(
            [],
            dtype=float,
        )

    first = timestamps[0]

    return np.asarray(
        [
            (timestamp - first).total_seconds()
            for timestamp in timestamps
        ],
        dtype=float,
    )


def sampling_interval_statistics(
    measurements: Iterable[Any],
    *,
    timestamp_field: str = "timestamp",
    expected_seconds: float | None = None,
    missing_tolerance: float = 0.25,
) -> SamplingIntervalStatistics:
    """
    Analyse intervals between successive measurements.

    Parameters
    ----------
    measurements
        Measurement collection.
    timestamp_field
        Timestamp field name.
    expected_seconds
        Intended sampling interval. When omitted, the median observed
        interval is used only for estimating missing periods.
    missing_tolerance
        Relative tolerance before an interval is treated as larger
        than one expected period.

    Returns
    -------
    SamplingIntervalStatistics
        Interval statistics.

    Raises
    ------
    ValueError
        If fewer than two measurements are supplied.
    """

    if expected_seconds is not None:
        if (
            not np.isfinite(expected_seconds)
            or expected_seconds <= 0.0
        ):
            raise ValueError(
                "expected_seconds must be finite "
                "and greater than zero."
            )

    if (
        not np.isfinite(missing_tolerance)
        or missing_tolerance < 0.0
    ):
        raise ValueError(
            "missing_tolerance must be finite "
            "and non-negative."
        )

    timestamps = sorted(
        extract_timestamps(
            measurements,
            field_name=timestamp_field,
        )
    )

    if len(timestamps) < 2:
        raise ValueError(
            "At least two timestamps are required."
        )

    intervals = np.asarray(
        [
            (
                timestamps[index]
                - timestamps[index - 1]
            ).total_seconds()
            for index in range(
                1,
                len(timestamps),
            )
        ],
        dtype=float,
    )

    if np.any(intervals <= 0.0):
        raise ValueError(
            "Timestamps must be unique and increasing."
        )

    interval_for_detection = (
        float(np.median(intervals))
        if expected_seconds is None
        else float(expected_seconds)
    )

    missing_intervals = 0

    threshold = (
        interval_for_detection
        * (1.0 + missing_tolerance)
    )

    for interval in intervals:
        if interval > threshold:
            periods = int(
                np.rint(
                    interval
                    / interval_for_detection
                )
            )

            missing_intervals += max(
                0,
                periods - 1,
            )

    standard_deviation = (
        float(np.std(intervals, ddof=1))
        if intervals.size > 1
        else 0.0
    )

    return SamplingIntervalStatistics(
        count=int(intervals.size),
        mean_seconds=float(np.mean(intervals)),
        median_seconds=float(np.median(intervals)),
        minimum_seconds=float(np.min(intervals)),
        maximum_seconds=float(np.max(intervals)),
        standard_deviation_seconds=standard_deviation,
        expected_seconds=expected_seconds,
        missing_intervals=missing_intervals,
    )


def rolling_mean(
    values: Iterable[float],
    window: int,
    *,
    centered: bool = False,
    minimum_count: int | None = None,
) -> np.ndarray:
    """
    Calculate a NaN-aware rolling arithmetic mean.

    Parameters
    ----------
    values
        Numeric sequence.
    window
        Number of observations per window.
    centered
        Center the window around each observation. Otherwise, the
        result is trailing.
    minimum_count
        Minimum finite observations required for an output value.
        The default is the full window size.

    Returns
    -------
    numpy.ndarray
        Rolling means with the same length as the input.
    """

    array = np.asarray(
        list(values),
        dtype=float,
    )

    if array.ndim != 1:
        raise ValueError(
            "values must be one-dimensional."
        )

    if window < 1:
        raise ValueError(
            "window must be at least 1."
        )

    if minimum_count is None:
        minimum_count = window

    if (
        minimum_count < 1
        or minimum_count > window
    ):
        raise ValueError(
            "minimum_count must be between 1 and window."
        )

    result = np.full(
        array.shape,
        np.nan,
        dtype=float,
    )

    count = array.size

    for index in range(count):
        if centered:
            left = (window - 1) // 2
            right = window // 2

            start = max(
                0,
                index - left,
            )

            stop = min(
                count,
                index + right + 1,
            )

        else:
            start = max(
                0,
                index - window + 1,
            )

            stop = index + 1

        current = array[start:stop]
        finite = current[
            np.isfinite(current)
        ]

        if finite.size >= minimum_count:
            result[index] = float(
                np.mean(finite)
            )

    return result


def spectral_channel_matrix(
    measurements: Iterable[Any],
    *,
    channel_field: str = "spectral_channels",
    channel_order: Sequence[str] | None = None,
    missing_value: float = np.nan,
) -> tuple[list[str], np.ndarray]:
    """
    Convert spectral channel mappings into a two-dimensional matrix.

    Parameters
    ----------
    measurements
        Measurement objects or dictionaries.
    channel_field
        Field containing a channel-name-to-signal mapping.
    channel_order
        Optional explicit channel order. Without it, first-seen order
        is retained.
    missing_value
        Value used when a channel is missing.

    Returns
    -------
    tuple[list[str], numpy.ndarray]
        Channel names and matrix shaped
        ``(measurement_count, channel_count)``.
    """

    measurement_list = list(
        measurements
    )

    channel_mappings: list[Mapping[str, Any]] = []

    for index, measurement in enumerate(
        measurement_list
    ):
        channels = measurement_field(
            measurement,
            channel_field,
        )

        if not isinstance(channels, Mapping):
            raise ValueError(
                f"Measurement {index} does not contain "
                "a spectral-channel mapping."
            )

        channel_mappings.append(channels)

    if channel_order is None:
        names: list[str] = []
        seen: set[str] = set()

        for channels in channel_mappings:
            for name in channels:
                text_name = str(name)

                if text_name not in seen:
                    seen.add(text_name)
                    names.append(text_name)

    else:
        names = [
            str(name)
            for name in channel_order
        ]

        if len(set(names)) != len(names):
            raise ValueError(
                "channel_order contains duplicate names."
            )

    matrix = np.full(
        (
            len(channel_mappings),
            len(names),
        ),
        float(missing_value),
        dtype=float,
    )

    for row_index, channels in enumerate(
        channel_mappings
    ):
        for column_index, name in enumerate(
            names
        ):
            value = channels.get(name)

            if value is None:
                continue

            try:
                matrix[
                    row_index,
                    column_index,
                ] = float(value)

            except (
                TypeError,
                ValueError,
            ):
                continue

    return names, matrix


def spectral_channel_statistics(
    measurements: Iterable[Any],
    *,
    channel_field: str = "spectral_channels",
    channel_order: Sequence[str] | None = None,
) -> dict[str, DescriptiveStatistics]:
    """
    Calculate descriptive statistics for every spectral channel.

    Returns
    -------
    dict[str, DescriptiveStatistics]
        Statistics indexed by channel name.
    """

    names, matrix = spectral_channel_matrix(
        measurements,
        channel_field=channel_field,
        channel_order=channel_order,
    )

    result: dict[
        str,
        DescriptiveStatistics,
    ] = {}

    for column_index, name in enumerate(
        names
    ):
        finite_values = matrix[
            :,
            column_index,
        ]

        finite_values = finite_values[
            np.isfinite(finite_values)
        ]

        if finite_values.size == 0:
            continue

        result[name] = descriptive_statistics(
            finite_values
        )

    return result


def interpolate_reference_values(
    measurement_times: Iterable[datetime | str],
    reference_times: Iterable[datetime | str],
    reference_values: Iterable[float],
    *,
    allow_extrapolation: bool = False,
) -> np.ndarray:
    """
    Interpolate reference values onto measurement timestamps.

    Parameters
    ----------
    measurement_times
        Target timestamps.
    reference_times
        Reference-observation timestamps.
    reference_values
        Reference values corresponding to ``reference_times``.
    allow_extrapolation
        Permit constant endpoint extrapolation. When ``False``,
        measurements outside the reference time range receive NaN.

    Returns
    -------
    numpy.ndarray
        Interpolated reference values.
    """

    target_datetimes = [
        _parse_datetime(value)
        for value in measurement_times
    ]

    source_datetimes = [
        _parse_datetime(value)
        for value in reference_times
    ]

    source_values = np.asarray(
        list(reference_values),
        dtype=float,
    )

    if len(source_datetimes) != source_values.size:
        raise ValueError(
            "reference_times and reference_values "
            "must have the same length."
        )

    if len(source_datetimes) < 2:
        raise ValueError(
            "At least two reference observations are required."
        )

    if not np.all(
        np.isfinite(source_values)
    ):
        raise ValueError(
            "reference_values must be finite."
        )

    order = np.argsort(
        np.asarray(
            [
                timestamp.timestamp()
                for timestamp in source_datetimes
            ],
            dtype=float,
        )
    )

    source_seconds = np.asarray(
        [
            source_datetimes[index].timestamp()
            for index in order
        ],
        dtype=float,
    )

    sorted_values = source_values[order]

    if np.any(
        np.diff(source_seconds) <= 0.0
    ):
        raise ValueError(
            "reference_times must be unique."
        )

    target_seconds = np.asarray(
        [
            timestamp.timestamp()
            for timestamp in target_datetimes
        ],
        dtype=float,
    )

    interpolated = np.interp(
        target_seconds,
        source_seconds,
        sorted_values,
    )

    if not allow_extrapolation:
        outside = (
            (target_seconds < source_seconds[0])
            | (target_seconds > source_seconds[-1])
        )

        interpolated[outside] = np.nan

    return interpolated


def compare_series(
    measured_values: Iterable[float],
    reference_values: Iterable[float],
) -> ComparisonStatistics:
    """
    Compare paired measured and reference observations.

    Parameters
    ----------
    measured_values
        Reef Cube or other measured values.
    reference_values
        Corresponding reference values.

    Returns
    -------
    ComparisonStatistics
        Agreement and regression statistics.
    """

    measured = np.asarray(
        list(measured_values),
        dtype=float,
    )

    reference = np.asarray(
        list(reference_values),
        dtype=float,
    )

    if measured.ndim != 1 or reference.ndim != 1:
        raise ValueError(
            "Both inputs must be one-dimensional."
        )

    if measured.size != reference.size:
        raise ValueError(
            "measured_values and reference_values "
            "must have the same length."
        )

    finite_mask = (
        np.isfinite(measured)
        & np.isfinite(reference)
    )

    measured = measured[
        finite_mask
    ]

    reference = reference[
        finite_mask
    ]

    if measured.size < 2:
        raise ValueError(
            "At least two finite pairs are required."
        )

    errors = measured - reference

    bias = float(
        np.mean(errors)
    )

    mae = float(
        np.mean(
            np.abs(errors)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                errors**2
            )
        )
    )

    mean_reference = float(
        np.mean(reference)
    )

    percent_bias = (
        float("nan")
        if mean_reference == 0.0
        else 100.0 * bias / mean_reference
    )

    nonzero_reference = (
        reference != 0.0
    )

    if np.any(nonzero_reference):
        mape = float(
            100.0
            * np.mean(
                np.abs(
                    errors[nonzero_reference]
                    / reference[nonzero_reference]
                )
            )
        )

    else:
        mape = float("nan")

    reference_centered = (
        reference
        - np.mean(reference)
    )

    measured_centered = (
        measured
        - np.mean(measured)
    )

    denominator = float(
        np.sqrt(
            np.sum(reference_centered**2)
            * np.sum(measured_centered**2)
        )
    )

    pearson_r = (
        float("nan")
        if denominator == 0.0
        else float(
            np.sum(
                reference_centered
                * measured_centered
            )
            / denominator
        )
    )

    design_matrix = np.column_stack(
        [
            reference,
            np.ones(
                reference.size,
                dtype=float,
            ),
        ]
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        design_matrix,
        measured,
        rcond=None,
    )

    slope = float(
        coefficients[0]
    )

    intercept = float(
        coefficients[1]
    )

    predicted = (
        slope * reference
        + intercept
    )

    residual_sum_squares = float(
        np.sum(
            (measured - predicted) ** 2
        )
    )

    total_sum_squares = float(
        np.sum(
            (
                measured
                - np.mean(measured)
            )
            ** 2
        )
    )

    r_squared = (
        1.0
        if total_sum_squares == 0.0
        and residual_sum_squares == 0.0
        else (
            float("nan")
            if total_sum_squares == 0.0
            else (
                1.0
                - residual_sum_squares
                / total_sum_squares
            )
        )
    )

    return ComparisonStatistics(
        count=int(measured.size),
        bias=bias,
        mean_absolute_error=mae,
        root_mean_square_error=rmse,
        percent_bias=percent_bias,
        mean_absolute_percentage_error=mape,
        pearson_r=pearson_r,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
    )


def compare_measurement_field(
    measurements: Iterable[Any],
    reference_values: Iterable[float],
    field_name: str,
) -> ComparisonStatistics:
    """
    Compare one measurement field with paired reference values.

    Missing or non-numeric measurement values are represented as NaN
    and excluded pairwise by :func:`compare_series`.
    """

    measurement_list = list(
        measurements
    )

    measured_values: list[float] = []

    for measurement in measurement_list:
        value = measurement_field(
            measurement,
            field_name,
        )

        try:
            measured_values.append(
                float(value)
            )

        except (
            TypeError,
            ValueError,
        ):
            measured_values.append(
                float("nan")
            )

    return compare_series(
        measured_values,
        reference_values,
    )


def _finite_float_array(
    values: Iterable[float],
    *,
    name: str,
) -> np.ndarray:
    """
    Convert input into a one-dimensional finite float array.
    """

    array = np.asarray(
        list(values),
        dtype=float,
    )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    return array[
        np.isfinite(array)
    ]


def _parse_datetime(
    value: datetime | str,
) -> datetime:
    """
    Parse and normalize a datetime to UTC.
    """

    if isinstance(value, datetime):
        parsed = value

    elif isinstance(value, str):
        text = value.strip()

        if not text:
            raise ValueError(
                "Timestamp text must not be empty."
            )

        if text.endswith("Z"):
            text = (
                text[:-1]
                + "+00:00"
            )

        try:
            parsed = datetime.fromisoformat(
                text
            )

        except ValueError as error:
            raise ValueError(
                f"Invalid ISO-8601 timestamp: {value!r}."
            ) from error

    else:
        raise TypeError(
            "Timestamp values must be datetime or str."
        )

    if parsed.tzinfo is None:
        return parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed.astimezone(
        timezone.utc
    )
