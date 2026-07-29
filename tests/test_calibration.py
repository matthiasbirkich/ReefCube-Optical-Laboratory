import numpy as np

from reefcube.calibration import (
    calibration_table,
    fit_linear_calibration,
    inverse_variance_weights,
)


# Simulated paired calibration measurements:
#
# x = Reef Cube relative PPFD index
# y = reference PPFD in µmol m⁻² s⁻¹

sensor_index = np.asarray(
    [
        0.0,
        1.0e8,
        2.0e8,
        3.0e8,
        4.0e8,
        5.0e8,
        6.0e8,
    ],
    dtype=float,
)

reference_ppfd = np.asarray(
    [
        2.0,
        12.5,
        22.0,
        32.8,
        42.1,
        52.9,
        62.0,
    ],
    dtype=float,
)

reference_standard_deviation = np.asarray(
    [
        1.0,
        1.2,
        1.5,
        1.8,
        2.0,
        2.3,
        2.5,
    ],
    dtype=float,
)

weights = inverse_variance_weights(
    reference_standard_deviation
)


ordinary_result = fit_linear_calibration(
    sensor_index,
    reference_ppfd,
    sensor_name="Reef Cube relative PPFD index",
    reference_name="Example reference PPFD",
)

weighted_result = fit_linear_calibration(
    sensor_index,
    reference_ppfd,
    weights=weights,
    sensor_name="Reef Cube relative PPFD index",
    reference_name="Example reference PPFD",
)

forced_zero_result = fit_linear_calibration(
    sensor_index,
    reference_ppfd,
    weights=weights,
    force_zero=True,
    sensor_name="Reef Cube relative PPFD index",
    reference_name="Example reference PPFD",
)


print("Ordinary linear regression")
print("--------------------------")
print(ordinary_result.summary())

print()
print("Weighted linear regression")
print("--------------------------")
print(weighted_result.summary())

print()
print("Weighted forced-zero regression")
print("-------------------------------")
print(forced_zero_result.summary())


table = calibration_table(
    weighted_result,
    sensor_index,
    reference_ppfd,
)

print()
print("Weighted calibration table")
print("--------------------------")

for index in range(sensor_index.size):
    print(
        f"x={table['sensor_value'][index]:12.1f}  "
        f"observed={table['reference_value'][index]:6.2f}  "
        f"predicted={table['predicted_value'][index]:6.2f}  "
        f"residual={table['residual'][index]:+6.2f}"
    )


ppfd_calibration = (
    weighted_result.to_ppfd_calibration()
)

example_index = 5.5e8

estimated_ppfd = ppfd_calibration.convert(
    example_index
)

print()
print(
    f"Estimated PPFD at index {example_index:.2e}: "
    f"{estimated_ppfd:.2f} µmol m⁻² s⁻¹"
)