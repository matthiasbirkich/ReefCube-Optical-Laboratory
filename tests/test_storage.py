from pathlib import Path

import numpy as np

from reefcube.calibration import fit_linear_calibration
from reefcube.ppfd import measurement_relative_ppfd_index
from reefcube.sensors import SimulationSensor
from reefcube.storage import (
    append_measurement_jsonl,
    load_calibration_pairs_csv,
    load_measurements_csv,
    load_measurements_jsonl,
    save_calibration_pairs_csv,
    save_measurements_csv,
    save_measurements_jsonl,
)


output_directory = Path("storage_test")
output_directory.mkdir(
    exist_ok=True
)


# ------------------------------------------------------------
# Generate three simulated Reef Cube measurements
# ------------------------------------------------------------

sensor = SimulationSensor(
    mode="underwater",
    seed=42,
)

measurements = [
    sensor.acquire()
    for _ in range(3)
]


# ------------------------------------------------------------
# CSV measurement round trip
# ------------------------------------------------------------

measurement_csv_path = (
    output_directory
    / "measurements.csv"
)

save_measurements_csv(
    measurement_csv_path,
    measurements,
)

loaded_csv_records = load_measurements_csv(
    measurement_csv_path
)

print("Measurement CSV")
print("---------------")
print(
    f"Saved records : {len(measurements)}"
)
print(
    f"Loaded records: {len(loaded_csv_records)}"
)
print(
    "First fields  :",
    list(
        loaded_csv_records[0].keys()
    ),
)


# ------------------------------------------------------------
# JSON Lines measurement round trip
# ------------------------------------------------------------

measurement_jsonl_path = (
    output_directory
    / "measurements.jsonl"
)

save_measurements_jsonl(
    measurement_jsonl_path,
    measurements[:2],
)

append_measurement_jsonl(
    measurement_jsonl_path,
    measurements[2],
)

loaded_jsonl_records = load_measurements_jsonl(
    measurement_jsonl_path
)

print()
print("Measurement JSON Lines")
print("----------------------")
print(
    f"Loaded records: {len(loaded_jsonl_records)}"
)


# ------------------------------------------------------------
# Calibration-pair CSV
# ------------------------------------------------------------

relative_indices = np.asarray(
    [
        measurement_relative_ppfd_index(
            measurement
        )
        for measurement in measurements
    ],
    dtype=float,
)

# Demonstration reference measurements only.
reference_ppfd = (
    relative_indices
    * 1.0e-7
    + np.asarray(
        [2.0, 2.2, 1.8],
        dtype=float,
    )
)

calibration_path = (
    output_directory
    / "calibration_pairs.csv"
)

save_calibration_pairs_csv(
    calibration_path,
    relative_indices,
    reference_ppfd,
    sensor_name="relative_index",
    reference_name="reference_ppfd",
)

loaded_calibration = (
    load_calibration_pairs_csv(
        calibration_path,
        sensor_column="relative_index",
        reference_column="reference_ppfd",
    )
)

print()
print("Calibration CSV")
print("---------------")
print(
    "Sensor values   :",
    loaded_calibration[
        "sensor_values"
    ],
)
print(
    "Reference values:",
    loaded_calibration[
        "reference_values"
    ],
)


# ------------------------------------------------------------
# Use loaded values directly for calibration
# ------------------------------------------------------------

calibration_result = fit_linear_calibration(
    loaded_calibration[
        "sensor_values"
    ],
    loaded_calibration[
        "reference_values"
    ],
    sensor_name="Reef Cube relative PPFD index",
    reference_name="Demonstration reference PPFD",
)

print()
print("Calibration from stored data")
print("----------------------------")
print(
    calibration_result.summary()
)

print()
print(
    "Files written to:",
    output_directory.resolve(),
)