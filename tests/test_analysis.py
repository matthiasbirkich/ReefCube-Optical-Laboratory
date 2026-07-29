from datetime import timedelta
from pathlib import Path

import numpy as np

from reefcube.analysis import (
    compare_series,
    descriptive_statistics,
    elapsed_seconds,
    filter_valid_measurements,
    measurement_statistics,
    rolling_mean,
    sampling_interval_statistics,
    spectral_channel_statistics,
)
from reefcube.sensors import SimulationSensor
from reefcube.storage import (
    load_measurements_csv,
)


# ------------------------------------------------------------
# Load the measurement file created by storage.py
# ------------------------------------------------------------

measurement_path = (
    Path("storage_test")
    / "measurements.csv"
)

measurements = load_measurements_csv(
    measurement_path
)

valid_measurements = (
    filter_valid_measurements(
        measurements
    )
)


# ------------------------------------------------------------
# General measurement statistics
# ------------------------------------------------------------

print("Measurement collection")
print("----------------------")
print(
    f"Loaded measurements: {len(measurements)}"
)
print(
    f"Valid measurements : {len(valid_measurements)}"
)


temperature_statistics = (
    measurement_statistics(
        valid_measurements,
        "temperature_C",
    )
)

print()
print("Temperature statistics")
print("----------------------")
print(
    temperature_statistics.summary(
        precision=3
    )
)


lux_statistics = measurement_statistics(
    valid_measurements,
    "lux",
)

print()
print("Lux statistics")
print("--------------")
print(
    lux_statistics.summary(
        precision=2
    )
)


# ------------------------------------------------------------
# Spectral-channel statistics
# ------------------------------------------------------------

channel_statistics = (
    spectral_channel_statistics(
        valid_measurements
    )
)

print()
print("Spectral-channel means")
print("----------------------")

for channel_name, statistics in (
    channel_statistics.items()
):
    print(
        f"{channel_name:>7s}: "
        f"{statistics.mean:10.2f}"
    )


# ------------------------------------------------------------
# Rolling average
# ------------------------------------------------------------

lux_values = np.asarray(
    [
        measurement["lux"]
        for measurement in valid_measurements
    ],
    dtype=float,
)

rolling_lux = rolling_mean(
    lux_values,
    window=2,
    minimum_count=1,
)

print()
print("Rolling lux mean")
print("----------------")
print(rolling_lux)


# ------------------------------------------------------------
# Elapsed times
# ------------------------------------------------------------

elapsed = elapsed_seconds(
    valid_measurements
)

print()
print("Elapsed seconds")
print("---------------")
print(elapsed)


# ------------------------------------------------------------
# Build a longer sequence for sampling diagnostics
# ------------------------------------------------------------

sensor = SimulationSensor(
    mode="underwater",
    seed=123,
)

sampling_measurements = [
    sensor.acquire()
    for _ in range(6)
]

first_timestamp = (
    sampling_measurements[0].timestamp
)

for index, measurement in enumerate(
    sampling_measurements
):
    object.__setattr__(
        measurement,
        "timestamp",
        first_timestamp
        + timedelta(
            minutes=30 * index
        ),
    )

# Simulate one missing 30-minute record.
object.__setattr__(
    sampling_measurements[4],
    "timestamp",
    first_timestamp
    + timedelta(
        minutes=150
    ),
)

object.__setattr__(
    sampling_measurements[5],
    "timestamp",
    first_timestamp
    + timedelta(
        minutes=180
    ),
)

sampling_statistics = (
    sampling_interval_statistics(
        sampling_measurements,
        expected_seconds=1800.0,
    )
)

print()
print("Sampling diagnostics")
print("--------------------")
print(
    sampling_statistics.summary()
)


# ------------------------------------------------------------
# Reference comparison
# ------------------------------------------------------------

reference_lux = (
    lux_values * 0.95
    + np.asarray(
        [50.0, -40.0, 25.0],
        dtype=float,
    )
)

comparison = compare_series(
    lux_values,
    reference_lux,
)

print()
print("Lux comparison")
print("--------------")
print(
    comparison.summary(
        precision=4
    )
)


# ------------------------------------------------------------
# Standalone descriptive-statistics test
# ------------------------------------------------------------

example_statistics = (
    descriptive_statistics(
        [1.0, 2.0, 3.0, 4.0, 5.0]
    )
)

print()
print("Standalone statistics")
print("---------------------")
print(
    example_statistics.summary(
        precision=2
    )
)