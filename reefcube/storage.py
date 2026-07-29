"""
storage.py
==========

Persistent storage utilities for the Reef Cube Optical Laboratory.

The module provides:

- CSV storage for Reef Cube measurements
- JSON Lines storage for measurements
- CSV storage for calibration pairs
- safe conversion of dataclasses, NumPy values, timestamps, mappings,
  sequences, and optional values
- atomic file replacement to reduce the risk of corrupted files
- optional reconstruction through a user-supplied factory function

The storage format deliberately avoids depending on one exact version
of ``ReefCubeMeasurement``. This allows older measurement files to
remain readable after the measurement model is extended.

Complex values such as spectral-channel mappings are JSON-encoded
inside CSV cells.

Author
------
Matthias Birkicht & OpenAI

Version
-------
1.0
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, fields, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import numpy as np


PathLike = str | os.PathLike[str]
Record = dict[str, Any]
T = TypeVar("T")


SCHEMA_VERSION: int = 1
"""Current Reef Cube storage-schema version."""

CSV_SCHEMA_FIELD: str = "_schema_version"
"""CSV column containing the storage-schema version."""

CSV_TYPE_FIELD: str = "_record_type"
"""CSV column identifying the stored record type."""

MEASUREMENT_RECORD_TYPE: str = "reefcube_measurement"
"""Record-type identifier for measurement rows."""

CALIBRATION_RECORD_TYPE: str = "reefcube_calibration_pair"
"""Record-type identifier for calibration rows."""


class StorageError(RuntimeError):
    """
    Base exception for Reef Cube storage errors.
    """


class StorageFormatError(StorageError):
    """
    Raised when a storage file has invalid or unsupported content.
    """


def measurement_to_record(
    measurement: Any,
) -> Record:
    """
    Convert one measurement-like object to a serializable record.

    Dataclass instances are converted using ``dataclasses.asdict``.
    Mapping objects are copied directly. Other objects are converted
    from their public instance attributes.

    Parameters
    ----------
    measurement
        Measurement dataclass, mapping, or object with public
        attributes.

    Returns
    -------
    dict[str, Any]
        JSON-compatible measurement record.

    Raises
    ------
    TypeError
        If the object cannot be converted to a record.
    """

    if is_dataclass(measurement):
        raw_record = asdict(measurement)

    elif isinstance(measurement, Mapping):
        raw_record = dict(measurement)

    elif hasattr(measurement, "__dict__"):
        raw_record = {
            key: value
            for key, value in vars(measurement).items()
            if not key.startswith("_")
        }

    else:
        raise TypeError(
            "measurement must be a dataclass, mapping, or object "
            "with public instance attributes."
        )

    record = {
        str(key): _to_json_compatible(value)
        for key, value in raw_record.items()
    }

    return record


def record_to_object(
    record: Mapping[str, Any],
    factory: Callable[..., T],
    *,
    strict: bool = False,
) -> T:
    """
    Reconstruct an object from a stored record.

    Parameters
    ----------
    record
        Stored measurement record.
    factory
        Callable accepting keyword arguments, for example
        ``ReefCubeMeasurement``.
    strict
        If ``True``, all record fields are passed to the factory. If
        ``False`` and the factory is a dataclass type, unknown fields
        are discarded.

    Returns
    -------
    T
        Reconstructed object.

    Notes
    -----
    Timestamp strings are converted to ``datetime`` objects when their
    field names contain ``time``, ``date``, or ``timestamp`` and the
    stored text is a valid ISO-8601 value.
    """

    if not callable(factory):
        raise TypeError(
            "factory must be callable."
        )

    converted = {
        str(key): _restore_common_value(
            str(key),
            value,
        )
        for key, value in record.items()
        if not str(key).startswith("_")
    }

    if not strict and is_dataclass(factory):
        allowed_fields = {
            field.name
            for field in fields(factory)
        }

        converted = {
            key: value
            for key, value in converted.items()
            if key in allowed_fields
        }

    try:
        return factory(**converted)

    except TypeError as error:
        raise StorageFormatError(
            "The stored record could not be passed to the supplied "
            f"factory: {error}"
        ) from error


def save_measurements_csv(
    path: PathLike,
    measurements: Iterable[Any],
    *,
    overwrite: bool = True,
    atomic: bool = True,
) -> Path:
    """
    Save Reef Cube measurements to a CSV file.

    Complex values such as mappings and sequences are JSON-encoded
    within individual CSV cells.

    Parameters
    ----------
    path
        Destination CSV path.
    measurements
        Iterable of measurement-like objects.
    overwrite
        Permit replacement of an existing file.
    atomic
        Write to a temporary file and replace the destination only
        after writing succeeds.

    Returns
    -------
    pathlib.Path
        Final file path.

    Raises
    ------
    FileExistsError
        If the destination exists and ``overwrite`` is ``False``.
    ValueError
        If no measurements are supplied.
    """

    destination = _prepare_destination(
        path,
        overwrite=overwrite,
    )

    records = [
        measurement_to_record(measurement)
        for measurement in measurements
    ]

    if not records:
        raise ValueError(
            "At least one measurement is required."
        )

    field_names = _collect_field_names(records)

    complete_field_names = [
        CSV_SCHEMA_FIELD,
        CSV_TYPE_FIELD,
        *field_names,
    ]

    rows = []

    for record in records:
        row: dict[str, str] = {
            CSV_SCHEMA_FIELD: str(SCHEMA_VERSION),
            CSV_TYPE_FIELD: MEASUREMENT_RECORD_TYPE,
        }

        for field_name in field_names:
            row[field_name] = _encode_csv_value(
                record.get(field_name)
            )

        rows.append(row)

    _write_csv_rows(
        destination,
        complete_field_names,
        rows,
        atomic=atomic,
    )

    return destination


def load_measurements_csv(
    path: PathLike,
    *,
    factory: Callable[..., T] | None = None,
    strict_factory: bool = False,
) -> list[Record] | list[T]:
    """
    Load Reef Cube measurements from a CSV file.

    Parameters
    ----------
    path
        Source CSV path.
    factory
        Optional callable for reconstructing measurement objects.
        Without a factory, dictionaries are returned.
    strict_factory
        Pass all stored fields to the factory. This is usually best
        left as ``False`` for dataclass factories.

    Returns
    -------
    list[dict[str, Any]] or list[T]
        Loaded records or reconstructed objects.
    """

    source = _validate_source_file(path)

    records: list[Record] = []

    try:
        with source.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)

            if reader.fieldnames is None:
                raise StorageFormatError(
                    "CSV file does not contain a header."
                )

            for row_number, row in enumerate(
                reader,
                start=2,
            ):
                schema_version = row.get(
                    CSV_SCHEMA_FIELD,
                    "",
                )

                _validate_schema_version(
                    schema_version,
                    row_number=row_number,
                )

                record_type = row.get(
                    CSV_TYPE_FIELD,
                    "",
                )

                if (
                    record_type
                    and record_type != MEASUREMENT_RECORD_TYPE
                ):
                    raise StorageFormatError(
                        f"Row {row_number} contains record type "
                        f"{record_type!r}, not a measurement."
                    )

                record = {
                    key: _decode_csv_value(value)
                    for key, value in row.items()
                    if key not in {
                        CSV_SCHEMA_FIELD,
                        CSV_TYPE_FIELD,
                    }
                    and key is not None
                }

                records.append(record)

    except UnicodeDecodeError as error:
        raise StorageFormatError(
            "The CSV file is not valid UTF-8 text."
        ) from error

    except csv.Error as error:
        raise StorageFormatError(
            f"Invalid CSV data: {error}"
        ) from error

    if factory is None:
        return records

    return [
        record_to_object(
            record,
            factory,
            strict=strict_factory,
        )
        for record in records
    ]


def append_measurement_csv(
    path: PathLike,
    measurement: Any,
) -> Path:
    """
    Append one measurement to an existing CSV file.

    If the file does not exist, it is created. If the new measurement
    introduces fields absent from the existing header, the complete
    file is rewritten with the expanded header.

    Parameters
    ----------
    path
        Destination CSV path.
    measurement
        Measurement-like object.

    Returns
    -------
    pathlib.Path
        Final file path.
    """

    destination = Path(path).expanduser()

    if not destination.exists():
        return save_measurements_csv(
            destination,
            [measurement],
        )

    existing_records = load_measurements_csv(
        destination
    )

    existing_records.append(
        measurement_to_record(measurement)
    )

    return save_measurements_csv(
        destination,
        existing_records,
        overwrite=True,
        atomic=True,
    )


def save_measurements_jsonl(
    path: PathLike,
    measurements: Iterable[Any],
    *,
    overwrite: bool = True,
    atomic: bool = True,
) -> Path:
    """
    Save measurements in JSON Lines format.

    Each line contains one independent JSON object. JSON Lines is
    suitable for long-running logger exports because individual
    records remain readable even if later writing is interrupted.

    Parameters
    ----------
    path
        Destination path, conventionally ending in ``.jsonl``.
    measurements
        Iterable of measurement-like objects.
    overwrite
        Permit replacement of an existing file.
    atomic
        Use atomic replacement.

    Returns
    -------
    pathlib.Path
        Final file path.
    """

    destination = _prepare_destination(
        path,
        overwrite=overwrite,
    )

    records = [
        measurement_to_record(measurement)
        for measurement in measurements
    ]

    if not records:
        raise ValueError(
            "At least one measurement is required."
        )

    lines = []

    for record in records:
        wrapped_record = {
            CSV_SCHEMA_FIELD: SCHEMA_VERSION,
            CSV_TYPE_FIELD: MEASUREMENT_RECORD_TYPE,
            **record,
        }

        lines.append(
            json.dumps(
                wrapped_record,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        )

    text = "\n".join(lines) + "\n"

    _write_text(
        destination,
        text,
        atomic=atomic,
    )

    return destination


def load_measurements_jsonl(
    path: PathLike,
    *,
    factory: Callable[..., T] | None = None,
    strict_factory: bool = False,
) -> list[Record] | list[T]:
    """
    Load measurements from JSON Lines storage.

    Parameters
    ----------
    path
        Source JSON Lines path.
    factory
        Optional object factory.
    strict_factory
        Pass all fields to the factory.

    Returns
    -------
    list[dict[str, Any]] or list[T]
        Loaded records or reconstructed objects.
    """

    source = _validate_source_file(path)
    records: list[Record] = []

    try:
        with source.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line_number, line in enumerate(
                handle,
                start=1,
            ):
                stripped = line.strip()

                if not stripped:
                    continue

                try:
                    raw_record = json.loads(stripped)

                except json.JSONDecodeError as error:
                    raise StorageFormatError(
                        f"Invalid JSON on line {line_number}: "
                        f"{error.msg}"
                    ) from error

                if not isinstance(raw_record, dict):
                    raise StorageFormatError(
                        f"Line {line_number} does not contain "
                        "a JSON object."
                    )

                _validate_schema_version(
                    raw_record.get(
                        CSV_SCHEMA_FIELD,
                        SCHEMA_VERSION,
                    ),
                    row_number=line_number,
                )

                record_type = raw_record.get(
                    CSV_TYPE_FIELD,
                    MEASUREMENT_RECORD_TYPE,
                )

                if record_type != MEASUREMENT_RECORD_TYPE:
                    raise StorageFormatError(
                        f"Line {line_number} contains unsupported "
                        f"record type {record_type!r}."
                    )

                record = {
                    str(key): value
                    for key, value in raw_record.items()
                    if key not in {
                        CSV_SCHEMA_FIELD,
                        CSV_TYPE_FIELD,
                    }
                }

                records.append(record)

    except UnicodeDecodeError as error:
        raise StorageFormatError(
            "The JSON Lines file is not valid UTF-8 text."
        ) from error

    if factory is None:
        return records

    return [
        record_to_object(
            record,
            factory,
            strict=strict_factory,
        )
        for record in records
    ]


def append_measurement_jsonl(
    path: PathLike,
    measurement: Any,
) -> Path:
    """
    Append one measurement to a JSON Lines file.

    Parameters
    ----------
    path
        Destination JSON Lines path.
    measurement
        Measurement-like object.

    Returns
    -------
    pathlib.Path
        Final file path.
    """

    destination = Path(path).expanduser()
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = {
        CSV_SCHEMA_FIELD: SCHEMA_VERSION,
        CSV_TYPE_FIELD: MEASUREMENT_RECORD_TYPE,
        **measurement_to_record(measurement),
    }

    serialized = json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )

    try:
        with destination.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(serialized)
            handle.write("\n")

    except OSError as error:
        raise StorageError(
            f"Could not append to {destination}: {error}"
        ) from error

    return destination


def save_calibration_pairs_csv(
    path: PathLike,
    sensor_values: Iterable[float],
    reference_values: Iterable[float],
    *,
    weights: Iterable[float] | None = None,
    timestamps: Iterable[datetime | str | None] | None = None,
    sensor_name: str = "sensor_value",
    reference_name: str = "reference_value",
    overwrite: bool = True,
    atomic: bool = True,
) -> Path:
    """
    Save paired sensor and reference calibration measurements.

    Parameters
    ----------
    path
        Destination CSV path.
    sensor_values
        Reef Cube sensor values or relative PPFD indices.
    reference_values
        Corresponding reference values.
    weights
        Optional regression weights.
    timestamps
        Optional timestamps corresponding to each pair.
    sensor_name
        CSV column name for sensor values.
    reference_name
        CSV column name for reference values.
    overwrite
        Permit replacement of an existing file.
    atomic
        Use atomic replacement.

    Returns
    -------
    pathlib.Path
        Final file path.
    """

    destination = _prepare_destination(
        path,
        overwrite=overwrite,
    )

    sensor_array = _as_1d_finite_float_array(
        sensor_values,
        name="sensor_values",
    )

    reference_array = _as_1d_finite_float_array(
        reference_values,
        name="reference_values",
    )

    if sensor_array.size != reference_array.size:
        raise ValueError(
            "sensor_values and reference_values must have "
            "the same length."
        )

    if sensor_array.size == 0:
        raise ValueError(
            "At least one calibration pair is required."
        )

    weight_array: np.ndarray | None = None

    if weights is not None:
        weight_array = _as_1d_finite_float_array(
            weights,
            name="weights",
        )

        if weight_array.size != sensor_array.size:
            raise ValueError(
                "weights must contain one value per calibration pair."
            )

        if np.any(weight_array < 0.0):
            raise ValueError(
                "weights must be non-negative."
            )

    timestamp_values: list[str] | None = None

    if timestamps is not None:
        timestamp_values = [
            _format_optional_timestamp(value)
            for value in timestamps
        ]

        if len(timestamp_values) != sensor_array.size:
            raise ValueError(
                "timestamps must contain one value per "
                "calibration pair."
            )

    sensor_column = _validate_column_name(
        sensor_name,
        name="sensor_name",
    )

    reference_column = _validate_column_name(
        reference_name,
        name="reference_name",
    )

    if sensor_column == reference_column:
        raise ValueError(
            "sensor_name and reference_name must differ."
        )

    field_names = [
        CSV_SCHEMA_FIELD,
        CSV_TYPE_FIELD,
        sensor_column,
        reference_column,
    ]

    if weight_array is not None:
        field_names.append("weight")

    if timestamp_values is not None:
        field_names.append("timestamp")

    rows: list[dict[str, str]] = []

    for index in range(sensor_array.size):
        row = {
            CSV_SCHEMA_FIELD: str(SCHEMA_VERSION),
            CSV_TYPE_FIELD: CALIBRATION_RECORD_TYPE,
            sensor_column: repr(
                float(sensor_array[index])
            ),
            reference_column: repr(
                float(reference_array[index])
            ),
        }

        if weight_array is not None:
            row["weight"] = repr(
                float(weight_array[index])
            )

        if timestamp_values is not None:
            row["timestamp"] = timestamp_values[index]

        rows.append(row)

    _write_csv_rows(
        destination,
        field_names,
        rows,
        atomic=atomic,
    )

    return destination


def load_calibration_pairs_csv(
    path: PathLike,
    *,
    sensor_column: str = "sensor_value",
    reference_column: str = "reference_value",
) -> dict[str, np.ndarray | list[datetime | None]]:
    """
    Load paired calibration values from CSV.

    Parameters
    ----------
    path
        Source CSV path.
    sensor_column
        Name of the sensor-value column.
    reference_column
        Name of the reference-value column.

    Returns
    -------
    dict
        Dictionary containing ``sensor_values``,
        ``reference_values``, and optionally ``weights`` and
        ``timestamps``.
    """

    source = _validate_source_file(path)

    sensor_values: list[float] = []
    reference_values: list[float] = []
    weights: list[float] = []
    timestamps: list[datetime | None] = []

    has_weight_column = False
    has_timestamp_column = False

    try:
        with source.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)

            if reader.fieldnames is None:
                raise StorageFormatError(
                    "Calibration CSV does not contain a header."
                )

            missing_columns = {
                sensor_column,
                reference_column,
            } - set(reader.fieldnames)

            if missing_columns:
                missing_text = ", ".join(
                    sorted(missing_columns)
                )

                raise StorageFormatError(
                    "Calibration CSV is missing required columns: "
                    f"{missing_text}."
                )

            has_weight_column = (
                "weight" in reader.fieldnames
            )

            has_timestamp_column = (
                "timestamp" in reader.fieldnames
            )

            for row_number, row in enumerate(
                reader,
                start=2,
            ):
                _validate_schema_version(
                    row.get(
                        CSV_SCHEMA_FIELD,
                        SCHEMA_VERSION,
                    ),
                    row_number=row_number,
                )

                record_type = row.get(
                    CSV_TYPE_FIELD,
                    CALIBRATION_RECORD_TYPE,
                )

                if record_type != CALIBRATION_RECORD_TYPE:
                    raise StorageFormatError(
                        f"Row {row_number} does not contain "
                        "a calibration pair."
                    )

                sensor_values.append(
                    _parse_required_float(
                        row.get(sensor_column),
                        column=sensor_column,
                        row_number=row_number,
                    )
                )

                reference_values.append(
                    _parse_required_float(
                        row.get(reference_column),
                        column=reference_column,
                        row_number=row_number,
                    )
                )

                if has_weight_column:
                    weights.append(
                        _parse_required_float(
                            row.get("weight"),
                            column="weight",
                            row_number=row_number,
                        )
                    )

                if has_timestamp_column:
                    timestamps.append(
                        _parse_optional_datetime(
                            row.get("timestamp")
                        )
                    )

    except UnicodeDecodeError as error:
        raise StorageFormatError(
            "The calibration CSV is not valid UTF-8 text."
        ) from error

    except csv.Error as error:
        raise StorageFormatError(
            f"Invalid calibration CSV data: {error}"
        ) from error

    result: dict[
        str,
        np.ndarray | list[datetime | None],
    ] = {
        "sensor_values": np.asarray(
            sensor_values,
            dtype=float,
        ),
        "reference_values": np.asarray(
            reference_values,
            dtype=float,
        ),
    }

    if has_weight_column:
        result["weights"] = np.asarray(
            weights,
            dtype=float,
        )

    if has_timestamp_column:
        result["timestamps"] = timestamps

    return result


def _collect_field_names(
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """
    Collect field names while retaining first-seen order.
    """

    field_names: list[str] = []
    seen: set[str] = set()

    for record in records:
        for key in record:
            field_name = str(key)

            if field_name in {
                CSV_SCHEMA_FIELD,
                CSV_TYPE_FIELD,
            }:
                continue

            if field_name not in seen:
                seen.add(field_name)
                field_names.append(field_name)

    return field_names


def _to_json_compatible(
    value: Any,
) -> Any:
    """
    Convert common Python and NumPy values to JSON-compatible values.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            bool,
            int,
        ),
    ):
        return value

    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(
                "NaN and infinite values cannot be stored."
            )

        return value

    if isinstance(value, np.generic):
        return _to_json_compatible(
            value.item()
        )

    if isinstance(value, np.ndarray):
        return [
            _to_json_compatible(item)
            for item in value.tolist()
        ]

    if isinstance(value, datetime):
        return _normalise_datetime(value).isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if is_dataclass(value):
        return {
            str(key): _to_json_compatible(item)
            for key, item in asdict(value).items()
        }

    if isinstance(value, Mapping):
        return {
            str(key): _to_json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        Sequence,
    ) and not isinstance(
        value,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        return [
            _to_json_compatible(item)
            for item in value
        ]

    raise TypeError(
        f"Unsupported storage value type: "
        f"{type(value).__name__}."
    )


def _encode_csv_value(
    value: Any,
) -> str:
    """
    Encode one value for storage in a CSV cell.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(
                "NaN and infinite values cannot be stored."
            )

        return repr(value)

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _decode_csv_value(
    value: str | None,
) -> Any:
    """
    Decode one CSV cell to a common Python value.
    """

    if value is None or value == "":
        return None

    stripped = value.strip()

    if not stripped:
        return None

    if stripped == "true":
        return True

    if stripped == "false":
        return False

    if stripped[0] in "[{":
        try:
            return json.loads(stripped)

        except json.JSONDecodeError:
            return value

    try:
        if any(
            character in stripped
            for character in ".eE"
        ):
            numeric_value = float(stripped)

            if np.isfinite(numeric_value):
                return numeric_value

        else:
            return int(stripped)

    except ValueError:
        pass

    return value


def _restore_common_value(
    field_name: str,
    value: Any,
) -> Any:
    """
    Restore common typed values using conservative field-name hints.
    """

    if not isinstance(value, str):
        return value

    normalised_name = field_name.lower()

    datetime_hint = any(
        token in normalised_name
        for token in (
            "timestamp",
            "datetime",
            "date_time",
            "created_at",
            "measured_at",
            "time_utc",
        )
    )

    if datetime_hint:
        parsed = _parse_optional_datetime(value)

        if parsed is not None:
            return parsed

    return value


def _normalise_datetime(
    value: datetime,
) -> datetime:
    """
    Return a timezone-aware UTC datetime.
    """

    if value.tzinfo is None:
        return value.replace(
            tzinfo=timezone.utc
        )

    return value.astimezone(
        timezone.utc
    )


def _format_optional_timestamp(
    value: datetime | str | None,
) -> str:
    """
    Convert an optional timestamp to ISO-8601 text.
    """

    if value is None:
        return ""

    if isinstance(value, datetime):
        return _normalise_datetime(value).isoformat()

    if isinstance(value, str):
        parsed = _parse_optional_datetime(value)

        if parsed is None:
            raise ValueError(
                f"Invalid timestamp value: {value!r}."
            )

        return parsed.isoformat()

    raise TypeError(
        "Timestamp values must be datetime, str, or None."
    )


def _parse_optional_datetime(
    value: str | None,
) -> datetime | None:
    """
    Parse an optional ISO-8601 datetime.
    """

    if value is None:
        return None

    stripped = value.strip()

    if not stripped:
        return None

    iso_value = stripped

    if iso_value.endswith("Z"):
        iso_value = (
            iso_value[:-1]
            + "+00:00"
        )

    try:
        parsed = datetime.fromisoformat(
            iso_value
        )

    except ValueError as error:
        raise StorageFormatError(
            f"Invalid ISO-8601 timestamp: {value!r}."
        ) from error

    return _normalise_datetime(parsed)


def _validate_schema_version(
    value: Any,
    *,
    row_number: int,
) -> None:
    """
    Validate a stored schema version.
    """

    try:
        version = int(value)

    except (
        TypeError,
        ValueError,
    ) as error:
        raise StorageFormatError(
            f"Invalid schema version in row {row_number}."
        ) from error

    if version > SCHEMA_VERSION:
        raise StorageFormatError(
            f"Row {row_number} uses schema version {version}, "
            f"but this software supports up to {SCHEMA_VERSION}."
        )

    if version < 1:
        raise StorageFormatError(
            f"Unsupported schema version {version} "
            f"in row {row_number}."
        )


def _prepare_destination(
    path: PathLike,
    *,
    overwrite: bool,
) -> Path:
    """
    Validate and prepare a destination path.
    """

    destination = Path(path).expanduser()

    if destination.exists():
        if destination.is_dir():
            raise IsADirectoryError(
                f"Destination is a directory: {destination}"
            )

        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {destination}"
            )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return destination


def _validate_source_file(
    path: PathLike,
) -> Path:
    """
    Validate a source file path.
    """

    source = Path(path).expanduser()

    if not source.exists():
        raise FileNotFoundError(
            f"Storage file does not exist: {source}"
        )

    if not source.is_file():
        raise IsADirectoryError(
            f"Storage path is not a file: {source}"
        )

    return source


def _write_csv_rows(
    destination: Path,
    field_names: Sequence[str],
    rows: Iterable[Mapping[str, str]],
    *,
    atomic: bool,
) -> None:
    """
    Write CSV rows safely.
    """

    def writer_function(handle: Any) -> None:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(field_names),
            extrasaction="raise",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    _write_using_handle(
        destination,
        writer_function,
        atomic=atomic,
        newline="",
    )


def _write_text(
    destination: Path,
    text: str,
    *,
    atomic: bool,
) -> None:
    """
    Write UTF-8 text safely.
    """

    def writer_function(handle: Any) -> None:
        handle.write(text)

    _write_using_handle(
        destination,
        writer_function,
        atomic=atomic,
        newline=None,
    )


def _write_using_handle(
    destination: Path,
    writer_function: Callable[[Any], None],
    *,
    atomic: bool,
    newline: str | None,
) -> None:
    """
    Write to a file directly or through atomic replacement.
    """

    if not atomic:
        try:
            with destination.open(
                "w",
                encoding="utf-8",
                newline=newline,
            ) as handle:
                writer_function(handle)

        except OSError as error:
            raise StorageError(
                f"Could not write {destination}: {error}"
            ) from error

        return

    temporary_path: Path | None = None

    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )

        temporary_path = Path(
            temporary_name
        )

        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
            newline=newline,
        ) as handle:
            writer_function(handle)
            handle.flush()
            os.fsync(
                handle.fileno()
            )

        os.replace(
            temporary_path,
            destination,
        )

    except OSError as error:
        raise StorageError(
            f"Could not write {destination}: {error}"
        ) from error

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            try:
                temporary_path.unlink()

            except OSError:
                pass


def _as_1d_finite_float_array(
    values: Iterable[float],
    *,
    name: str,
) -> np.ndarray:
    """
    Convert values to a finite one-dimensional float array.
    """

    array = np.asarray(
        list(values),
        dtype=float,
    )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return array


def _validate_column_name(
    value: str,
    *,
    name: str,
) -> str:
    """
    Validate a CSV column name.
    """

    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be a string."
        )

    stripped = value.strip()

    if not stripped:
        raise ValueError(
            f"{name} must not be empty."
        )

    if stripped in {
        CSV_SCHEMA_FIELD,
        CSV_TYPE_FIELD,
    }:
        raise ValueError(
            f"{name} uses a reserved column name."
        )

    return stripped


def _parse_required_float(
    value: str | None,
    *,
    column: str,
    row_number: int,
) -> float:
    """
    Parse one required finite floating-point value.
    """

    if value is None or not value.strip():
        raise StorageFormatError(
            f"Missing {column!r} value in row {row_number}."
        )

    try:
        result = float(value)

    except ValueError as error:
        raise StorageFormatError(
            f"Invalid numeric value in column {column!r}, "
            f"row {row_number}: {value!r}."
        ) from error

    if not np.isfinite(result):
        raise StorageFormatError(
            f"Non-finite numeric value in column {column!r}, "
            f"row {row_number}."
        )

    return result
