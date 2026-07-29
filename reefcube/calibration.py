"""
calibration.py
==============

Regression and calibration tools for the Reef Cube Optical Laboratory.

This module provides NumPy-based calibration functions for relating
Reef Cube sensor indices to reference measurements such as:

- PPFD from a quantum sensor
- PPFD derived from a TRIOS RAMSES spectrum
- calibrated irradiance
- lux
- laboratory reference values

Supported calibration methods include:

- ordinary linear regression
- weighted linear regression
- forced-zero linear regression
- weighted forced-zero regression

The resulting calibration can be converted directly into a
PPFDCalibration object from ``reefcube.ppfd``.

Author
------
Matthias Birkicht & OpenAI

Version
-------
1.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from reefcube.ppfd import PPFDCalibration


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """
    Results of a linear calibration fit.

    The fitted equation is:

    ``reference_value = slope * sensor_value + intercept``

    Parameters
    ----------
    slope
        Fitted calibration slope.
    intercept
        Fitted calibration intercept.
    r_squared
        Coefficient of determination.
    rmse
        Root mean square error.
    mae
        Mean absolute error.
    residual_standard_error
        Residual standard error.
    sample_count
        Number of calibration observations.
    degrees_of_freedom
        Residual degrees of freedom.
    forced_zero
        Whether the regression intercept was fixed at zero.
    weighted
        Whether weighted regression was used.
    minimum_sensor_value
        Lowest sensor value included in the calibration.
    maximum_sensor_value
        Highest sensor value included in the calibration.
    sensor_name
        Optional name of the sensor variable.
    reference_name
        Optional name of the reference instrument or variable.
    """

    slope: float
    intercept: float
    r_squared: float
    rmse: float
    mae: float
    residual_standard_error: float
    sample_count: int
    degrees_of_freedom: int
    forced_zero: bool
    weighted: bool
    minimum_sensor_value: float
    maximum_sensor_value: float
    sensor_name: str | None = None
    reference_name: str | None = None

    def predict(
        self,
        sensor_value: float | Iterable[float] | np.ndarray,
        *,
        allow_extrapolation: bool = False,
    ) -> float | np.ndarray:
        """
        Predict reference values from sensor values.

        Parameters
        ----------
        sensor_value
            One sensor value or an array of sensor values.
        allow_extrapolation
            If ``False``, values outside the calibration range raise
            ``ValueError``.

        Returns
        -------
        float or numpy.ndarray
            Predicted reference value or values.
        """

        values = np.asarray(
            sensor_value,
            dtype=float,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "sensor_value must contain only finite values."
            )

        if not allow_extrapolation:
            if np.any(values < self.minimum_sensor_value):
                raise ValueError(
                    "sensor_value lies below the calibration range."
                )

            if np.any(values > self.maximum_sensor_value):
                raise ValueError(
                    "sensor_value lies above the calibration range."
                )

        predictions = (
            self.slope * values + self.intercept
        )

        if predictions.ndim == 0:
            return float(predictions)

        return predictions

    def residuals(
        self,
        sensor_values: Iterable[float] | np.ndarray,
        reference_values: Iterable[float] | np.ndarray,
    ) -> np.ndarray:
        """
        Calculate residuals for supplied calibration data.

        Residuals are defined as:

        ``observed reference - predicted reference``

        Parameters
        ----------
        sensor_values
            Sensor measurements.
        reference_values
            Corresponding reference measurements.

        Returns
        -------
        numpy.ndarray
            Residual values.
        """

        x, y = _validate_xy(
            sensor_values,
            reference_values,
        )

        predicted = (
            self.slope * x + self.intercept
        )

        return y - predicted

    def to_ppfd_calibration(self) -> PPFDCalibration:
        """
        Convert the fitted result into a PPFDCalibration object.

        Returns
        -------
        PPFDCalibration
            Calibration usable by ``estimate_measurement_ppfd``.
        """

        return PPFDCalibration(
            slope=self.slope,
            intercept=self.intercept,
            minimum_index=self.minimum_sensor_value,
            maximum_index=self.maximum_sensor_value,
            reference_name=self.reference_name,
        )

    def equation(
        self,
        precision: int = 6,
        *,
        scientific_threshold: float = 1.0e-4,
    ) -> str:
        """
        Return the fitted equation as text.

        Small and very large coefficients are displayed in scientific
        notation so that meaningful non-zero slopes are not rounded to
        zero.

        Parameters
        ----------
        precision
            Number of digits after the decimal point.
        scientific_threshold
            Positive absolute values below this threshold are displayed
            in scientific notation. Values greater than or equal to
            one million are also displayed in scientific notation.

        Returns
        -------
        str
            Human-readable calibration equation.
        """

        if precision < 0:
            raise ValueError(
                "precision must be non-negative."
            )

        if (
            not np.isfinite(scientific_threshold)
            or scientific_threshold <= 0.0
        ):
            raise ValueError(
                "scientific_threshold must be finite "
                "and greater than zero."
            )

        slope_text = _format_coefficient(
            self.slope,
            precision=precision,
            scientific_threshold=scientific_threshold,
        )

        if self.forced_zero:
            return f"y = {slope_text} × x"

        intercept_text = _format_coefficient(
            abs(self.intercept),
            precision=precision,
            scientific_threshold=scientific_threshold,
        )
        sign = "+" if self.intercept >= 0.0 else "-"

        return (
            f"y = {slope_text} × x "
            f"{sign} {intercept_text}"
        )

    def summary(self, precision: int = 6) -> str:
        """
        Return a multi-line calibration summary.

        Parameters
        ----------
        precision
            Number of decimal places.

        Returns
        -------
        str
            Formatted summary.
        """

        lines = [
            f"Equation   : {self.equation(precision)}",
            f"R²         : {self.r_squared:.{precision}f}",
            f"RMSE       : {self.rmse:.{precision}f}",
            f"MAE        : {self.mae:.{precision}f}",
            (
                "Residual SE: "
                f"{self.residual_standard_error:.{precision}f}"
            ),
            f"Samples    : {self.sample_count}",
            f"DoF        : {self.degrees_of_freedom}",
            f"Forced zero: {self.forced_zero}",
            f"Weighted   : {self.weighted}",
            (
                "Range      : "
                f"{self.minimum_sensor_value:.{precision}f} "
                f"to {self.maximum_sensor_value:.{precision}f}"
            ),
        ]

        if self.sensor_name is not None:
            lines.append(
                f"Sensor     : {self.sensor_name}"
            )

        if self.reference_name is not None:
            lines.append(
                f"Reference  : {self.reference_name}"
            )

        return "\n".join(lines)


def fit_linear_calibration(
    sensor_values: Iterable[float] | np.ndarray,
    reference_values: Iterable[float] | np.ndarray,
    *,
    weights: Iterable[float] | np.ndarray | None = None,
    force_zero: bool = False,
    sensor_name: str | None = None,
    reference_name: str | None = None,
) -> CalibrationResult:
    """
    Fit a linear calibration model.

    The model is:

    ``reference_value = slope * sensor_value + intercept``

    Parameters
    ----------
    sensor_values
        Sensor measurements or relative sensor indices.
    reference_values
        Corresponding reference measurements.
    weights
        Optional non-negative regression weights. Larger weights give
        greater influence to an observation.
    force_zero
        If ``True``, fix the intercept at zero.
    sensor_name
        Optional name of the sensor variable.
    reference_name
        Optional name of the reference variable or instrument.

    Returns
    -------
    CalibrationResult
        Fitted coefficients and diagnostics.

    Raises
    ------
    ValueError
        If the data are invalid or the regression cannot be fitted.
    """

    x, y = _validate_xy(
        sensor_values,
        reference_values,
    )

    weight_array = _validate_weights(
        weights,
        expected_size=x.size,
    )

    weighted = weight_array is not None

    if weight_array is None:
        weight_array = np.ones_like(
            x,
            dtype=float,
        )

    if force_zero:
        slope, intercept = _fit_forced_zero(
            x,
            y,
            weight_array,
        )

        parameter_count = 1

    else:
        slope, intercept = _fit_with_intercept(
            x,
            y,
            weight_array,
        )

        parameter_count = 2

    predicted = slope * x + intercept
    residuals = y - predicted

    degrees_of_freedom = (
        int(x.size) - parameter_count
    )

    if degrees_of_freedom < 1:
        raise ValueError(
            "Not enough observations for the selected model."
        )

    r_squared = _weighted_r_squared(
        observed=y,
        predicted=predicted,
        weights=weight_array,
        force_zero=force_zero,
    )

    weighted_squared_error = np.sum(
        weight_array * residuals**2
    )

    total_weight = float(
        np.sum(weight_array)
    )

    rmse = float(
        np.sqrt(
            weighted_squared_error
            / total_weight
        )
    )

    mae = float(
        np.sum(
            weight_array * np.abs(residuals)
        )
        / total_weight
    )

    residual_standard_error = float(
        np.sqrt(
            weighted_squared_error
            / degrees_of_freedom
        )
    )

    return CalibrationResult(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        rmse=rmse,
        mae=mae,
        residual_standard_error=residual_standard_error,
        sample_count=int(x.size),
        degrees_of_freedom=degrees_of_freedom,
        forced_zero=bool(force_zero),
        weighted=weighted,
        minimum_sensor_value=float(np.min(x)),
        maximum_sensor_value=float(np.max(x)),
        sensor_name=sensor_name,
        reference_name=reference_name,
    )


def inverse_variance_weights(
    standard_deviations: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """
    Calculate inverse-variance regression weights.

    The weights are:

    ``weight = 1 / standard_deviation²``

    Parameters
    ----------
    standard_deviations
        Positive standard deviations for individual observations.

    Returns
    -------
    numpy.ndarray
        Inverse-variance weights.

    Raises
    ------
    ValueError
        If a standard deviation is zero, negative, or non-finite.
    """

    standard_deviation_array = _as_1d_float_array(
        standard_deviations,
        name="standard_deviations",
    )

    if not np.all(
        np.isfinite(standard_deviation_array)
    ):
        raise ValueError(
            "standard_deviations must be finite."
        )

    if np.any(standard_deviation_array <= 0.0):
        raise ValueError(
            "standard_deviations must be greater than zero."
        )

    return 1.0 / standard_deviation_array**2


def relative_error_weights(
    reference_values: Iterable[float] | np.ndarray,
    *,
    minimum_reference_value: float | None = None,
) -> np.ndarray:
    """
    Create weights that reduce domination by high reference values.

    The weights are approximately proportional to:

    ``1 / reference_value²``

    This can be useful when a calibration spans several orders of
    magnitude and relative rather than absolute errors are important.

    Parameters
    ----------
    reference_values
        Positive reference measurements.
    minimum_reference_value
        Optional lower limit used when constructing the weights.
        Values below the limit are replaced by the limit.

    Returns
    -------
    numpy.ndarray
        Relative-error weights.

    Notes
    -----
    This weighting method should only be used when scientifically
    justified. It is not a substitute for measured uncertainties.
    """

    values = _as_1d_float_array(
        reference_values,
        name="reference_values",
    )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "reference_values must be finite."
        )

    if np.any(values < 0.0):
        raise ValueError(
            "reference_values must be non-negative."
        )

    if minimum_reference_value is None:
        positive_values = values[values > 0.0]

        if positive_values.size == 0:
            raise ValueError(
                "At least one positive reference value is required."
            )

        floor_value = float(
            np.min(positive_values)
        )

    else:
        floor_value = float(
            minimum_reference_value
        )

        if (
            not np.isfinite(floor_value)
            or floor_value <= 0.0
        ):
            raise ValueError(
                "minimum_reference_value must be finite "
                "and greater than zero."
            )

    adjusted_values = np.maximum(
        values,
        floor_value,
    )

    weights = 1.0 / adjusted_values**2

    return weights / np.mean(weights)


def calibration_table(
    result: CalibrationResult,
    sensor_values: Iterable[float] | np.ndarray,
    reference_values: Iterable[float] | np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Create arrays for a calibration results table.

    Parameters
    ----------
    result
        Fitted calibration result.
    sensor_values
        Sensor measurements.
    reference_values
        Corresponding reference measurements.

    Returns
    -------
    dict[str, numpy.ndarray]
        Dictionary containing sensor values, reference values,
        predictions, residuals, and percentage errors.
    """

    if not isinstance(result, CalibrationResult):
        raise TypeError(
            "result must be a CalibrationResult object."
        )

    x, y = _validate_xy(
        sensor_values,
        reference_values,
    )

    predicted = (
        result.slope * x + result.intercept
    )

    residuals = y - predicted

    percentage_error = np.full_like(
        y,
        np.nan,
        dtype=float,
    )

    nonzero_mask = y != 0.0

    percentage_error[nonzero_mask] = (
        residuals[nonzero_mask]
        / y[nonzero_mask]
        * 100.0
    )

    return {
        "sensor_value": x.copy(),
        "reference_value": y.copy(),
        "predicted_value": predicted,
        "residual": residuals,
        "percentage_error": percentage_error,
    }


def _fit_with_intercept(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """
    Fit a weighted straight line with an intercept.
    """

    design_matrix = np.column_stack(
        (
            x,
            np.ones_like(x),
        )
    )

    square_root_weights = np.sqrt(
        weights
    )

    weighted_design = (
        design_matrix
        * square_root_weights[:, np.newaxis]
    )

    weighted_response = (
        y * square_root_weights
    )

    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted_design,
        weighted_response,
        rcond=None,
    )

    if rank < 2:
        raise ValueError(
            "Calibration fit failed because the sensor "
            "values do not contain sufficient variation."
        )

    slope = float(coefficients[0])
    intercept = float(coefficients[1])

    return slope, intercept


def _fit_forced_zero(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """
    Fit a weighted straight line through the origin.
    """

    denominator = float(
        np.sum(
            weights * x**2
        )
    )

    if denominator <= np.finfo(float).eps:
        raise ValueError(
            "Forced-zero calibration cannot be fitted "
            "because all sensor values are zero."
        )

    numerator = float(
        np.sum(
            weights * x * y
        )
    )

    slope = numerator / denominator

    return float(slope), 0.0


def _weighted_r_squared(
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
    *,
    force_zero: bool,
) -> float:
    """
    Calculate the weighted coefficient of determination.
    """

    residual_sum_squares = float(
        np.sum(
            weights
            * (
                observed - predicted
            )
            ** 2
        )
    )

    if force_zero:
        total_sum_squares = float(
            np.sum(
                weights * observed**2
            )
        )

    else:
        weighted_mean = float(
            np.sum(
                weights * observed
            )
            / np.sum(weights)
        )

        total_sum_squares = float(
            np.sum(
                weights
                * (
                    observed - weighted_mean
                )
                ** 2
            )
        )

    if total_sum_squares <= np.finfo(float).eps:
        if residual_sum_squares <= np.finfo(float).eps:
            return 1.0

        return 0.0

    return float(
        1.0
        - residual_sum_squares
        / total_sum_squares
    )


def _validate_xy(
    sensor_values: Iterable[float] | np.ndarray,
    reference_values: Iterable[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate paired calibration arrays.
    """

    x = _as_1d_float_array(
        sensor_values,
        name="sensor_values",
    )

    y = _as_1d_float_array(
        reference_values,
        name="reference_values",
    )

    if x.size != y.size:
        raise ValueError(
            "sensor_values and reference_values must "
            "have the same length."
        )

    if x.size < 2:
        raise ValueError(
            "At least two calibration observations are required."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "sensor_values must contain only finite values."
        )

    if not np.all(np.isfinite(y)):
        raise ValueError(
            "reference_values must contain only finite values."
        )

    if np.ptp(x) <= np.finfo(float).eps:
        raise ValueError(
            "sensor_values must contain variation."
        )

    return x, y


def _validate_weights(
    weights: Iterable[float] | np.ndarray | None,
    *,
    expected_size: int,
) -> np.ndarray | None:
    """
    Validate optional regression weights.
    """

    if weights is None:
        return None

    weight_array = _as_1d_float_array(
        weights,
        name="weights",
    )

    if weight_array.size != expected_size:
        raise ValueError(
            "weights must contain one value per observation."
        )

    if not np.all(np.isfinite(weight_array)):
        raise ValueError(
            "weights must contain only finite values."
        )

    if np.any(weight_array < 0.0):
        raise ValueError(
            "weights must be non-negative."
        )

    if not np.any(weight_array > 0.0):
        raise ValueError(
            "At least one weight must be greater than zero."
        )

    return weight_array


def _format_coefficient(
    value: float,
    *,
    precision: int,
    scientific_threshold: float,
) -> str:
    """
    Format a regression coefficient clearly.

    Scientific notation is used for small non-zero values and values
    greater than or equal to one million.
    """

    numeric_value = float(value)

    if not np.isfinite(numeric_value):
        raise ValueError(
            "Regression coefficients must be finite."
        )

    absolute_value = abs(numeric_value)

    if absolute_value == 0.0:
        return f"{0.0:.{precision}f}"

    if (
        absolute_value < scientific_threshold
        or absolute_value >= 1.0e6
    ):
        return f"{numeric_value:.{precision}e}"

    return f"{numeric_value:.{precision}f}"


def _as_1d_float_array(
    values: Iterable[float] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """
    Convert input data to a one-dimensional float array.
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if array.size == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    return array
