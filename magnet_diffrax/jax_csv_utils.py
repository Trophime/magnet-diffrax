import jax
import jax.numpy as jnp
import pandas as pd
from typing import Tuple, Callable, List


def create_jax_function_from_csv(
    csv_file_path: str, x_column: str, y_column: str, method: str = "linear"
) -> Tuple[Callable, jnp.ndarray, jnp.ndarray]:
    """
    Create a JAX function from tabulated CSV data.

    Args:
        csv_file_path: Path to the CSV file
        x_column: Name of the x-axis column (independent variable)
        y_column: Name of the y-axis column (dependent variable)
        method: Interpolation method ('linear', 'nearest', or 'lookup')

    Returns:
        Tuple of (jax_function, x_data, y_data)
    """

    # Load CSV data
    df = pd.read_csv(csv_file_path)

    # Extract and sort data by x values
    df_sorted = df.sort_values(x_column)
    x_data = jnp.array(df_sorted[x_column].values, dtype=jnp.float32)
    y_data = jnp.array(df_sorted[y_column].values, dtype=jnp.float32)

    if method == "linear":

        @jax.jit
        def jax_function(x):
            """Linear interpolation function"""
            return jnp.interp(x, x_data, y_data)

    elif method == "nearest":

        @jax.jit
        def jax_function(x):
            """Nearest neighbor interpolation"""
            indices = jnp.searchsorted(x_data, x)
            # Handle edge cases
            indices = jnp.clip(indices, 0, len(x_data) - 1)

            # Find nearest point
            left_dist = jnp.where(
                indices > 0, jnp.abs(x - x_data[indices - 1]), jnp.inf
            )
            right_dist = jnp.abs(x - x_data[indices])

            nearest_idx = jnp.where(left_dist < right_dist, indices - 1, indices)
            return y_data[nearest_idx]

    else:
        raise ValueError(f"Unknown method: {method}. Use 'linear' or 'nearest'")

    return jax_function, x_data, y_data


def create_2d_jax_function_from_csv(
    csv_file_path: str,
    x_column: str,
    y_column: str,
    z_column: str,
    method: str = "linear",
) -> Tuple[Callable, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Create a 2D JAX function from tabulated CSV data for z = f(x, y).

    Args:
        csv_file_path: Path to the CSV file
        x_column: Name of the first independent variable column
        y_column: Name of the second independent variable column
        z_column: Name of the dependent variable column
        method: Interpolation method ('linear' or 'nearest')

    Returns:
        Tuple of (jax_function, x_data, y_data, z_data)
    """

    # Load CSV data
    df = pd.read_csv(csv_file_path)

    # Extract data
    x_data = jnp.array(df[x_column].values, dtype=jnp.float32)
    y_data = jnp.array(df[y_column].values, dtype=jnp.float32)
    z_data = jnp.array(df[z_column].values, dtype=jnp.float32)

    if method == "linear":
        # For 2D linear interpolation, we'll use a simplified approach
        # Stack the coordinates for easier processing
        points = jnp.column_stack([x_data, y_data])

        @jax.jit
        def jax_function_2d(x, y):
            """2D linear interpolation function using nearest neighbors"""
            query_point = jnp.array([x, y])

            # Calculate distances to all points
            distances = jnp.linalg.norm(points - query_point, axis=1)

            # Find the 4 nearest points for bilinear interpolation
            # For simplicity, we'll use inverse distance weighting
            weights = 1.0 / (
                distances + 1e-10
            )  # Add small epsilon to avoid division by zero
            weights = weights / jnp.sum(weights)

            return jnp.sum(weights * z_data)

    elif method == "nearest":

        @jax.jit
        def jax_function_2d(x, y):
            """2D nearest neighbor interpolation"""
            query_point = jnp.array([x, y])
            distances = jnp.linalg.norm(points - query_point, axis=1)
            nearest_idx = jnp.argmin(distances)
            return z_data[nearest_idx]

    else:
        raise ValueError(f"Unknown method: {method}. Use 'linear' or 'nearest'")

    return jax_function_2d, x_data, y_data, z_data


def create_multi_column_jax_function_from_csv(
    csv_file_path: str, x_column: str, y_columns: List[str], method: str = "linear"
) -> Tuple[Callable, jnp.ndarray, jnp.ndarray]:
    """
    Create a JAX function that returns multiple outputs from CSV data.

    Args:
        csv_file_path: Path to the CSV file
        x_column: Name of the x-axis column (independent variable)
        y_columns: List of column names for dependent variables
        method: Interpolation method ('linear' or 'nearest')

    Returns:
        Tuple of (jax_function, x_data, y_data_array)
        where jax_function returns an array of interpolated values
    """

    # Load CSV data
    df = pd.read_csv(csv_file_path)

    # Extract and sort data by x values
    df_sorted = df.sort_values(x_column)
    x_data = jnp.array(df_sorted[x_column].values, dtype=jnp.float32)

    # Stack all y columns into a 2D array
    y_data_list = []
    for col in y_columns:
        y_data_list.append(df_sorted[col].values)
    y_data = jnp.array(y_data_list, dtype=jnp.float32).T  # Shape: (n_points, n_outputs)

    if method == "linear":

        @jax.jit
        def jax_function_multi(x):
            """Multi-output linear interpolation function"""
            # Vectorize interpolation across all columns
            return jnp.array(
                [jnp.interp(x, x_data, y_data[:, i]) for i in range(len(y_columns))]
            )

    elif method == "nearest":

        @jax.jit
        def jax_function_multi(x):
            """Multi-output nearest neighbor interpolation"""
            indices = jnp.searchsorted(x_data, x)
            indices = jnp.clip(indices, 0, len(x_data) - 1)

            # Find nearest point
            left_dist = jnp.where(
                indices > 0, jnp.abs(x - x_data[indices - 1]), jnp.inf
            )
            right_dist = jnp.abs(x - x_data[indices])
            nearest_idx = jnp.where(left_dist < right_dist, indices - 1, indices)

            return y_data[nearest_idx, :]

    else:
        raise ValueError(f"Unknown method: {method}. Use 'linear' or 'nearest'")

    return jax_function_multi, x_data, y_data


def create_parametric_jax_function_from_csv(
    csv_file_path: str,
    param_column: str,
    x_column: str,
    y_column: str,
    method: str = "linear",
) -> Tuple[Callable, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Create a parametric JAX function from CSV data: y = f(x, param).

    Args:
        csv_file_path: Path to the CSV file
        param_column: Name of the parameter column
        x_column: Name of the x-axis column
        y_column: Name of the y-axis column (dependent variable)
        method: Interpolation method ('linear' or 'nearest')

    Returns:
        Tuple of (jax_function, param_data, x_data, y_data)
        where jax_function takes (x, param) and returns interpolated y
    """

    # Load CSV data
    df = pd.read_csv(csv_file_path)

    # Extract data
    param_data = jnp.array(df[param_column].values, dtype=jnp.float32)
    x_data = jnp.array(df[x_column].values, dtype=jnp.float32)
    y_data = jnp.array(df[y_column].values, dtype=jnp.float32)

    if method == "linear":

        @jax.jit
        def jax_function_param(x, param):
            """Parametric interpolation function"""
            # Stack coordinates for 2D interpolation
            points = jnp.column_stack([x_data, param_data])
            query_point = jnp.array([x, param])

            # Use inverse distance weighting
            distances = jnp.linalg.norm(points - query_point, axis=1)
            weights = 1.0 / (distances + 1e-10)
            weights = weights / jnp.sum(weights)

            return jnp.sum(weights * y_data)

    elif method == "nearest":

        @jax.jit
        def jax_function_param(x, param):
            """Parametric nearest neighbor interpolation"""
            points = jnp.column_stack([x_data, param_data])
            query_point = jnp.array([x, param])
            distances = jnp.linalg.norm(points - query_point, axis=1)
            nearest_idx = jnp.argmin(distances)
            return y_data[nearest_idx]

    else:
        raise ValueError(f"Unknown method: {method}. Use 'linear' or 'nearest'")

    return jax_function_param, param_data, x_data, y_data
