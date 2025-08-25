import numpy as np
import pandas as pd
import jax.numpy as jnp

from magnet_diffrax.jax_csv_utils import (
    create_2d_jax_function_from_csv,
)


def create_test_gridded_csv(filename: str = "test_grid.csv"):
    """Create a test CSV file with gridded data for testing"""

    # Define grid
    x_vals = np.linspace(0, 5, 11)  # 11 points from 0 to 5
    y_vals = np.linspace(0, 3, 7)  # 7 points from 0 to 3

    # Create meshgrid
    X, Y = np.meshgrid(x_vals, y_vals)

    # Define test function: f(x,y) = x^2 + y^2 + sin(x*y)
    Z = X**2 + Y**2 + np.sin(X * Y)

    # Flatten to create CSV format (this creates duplicate X,Y values)
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()

    # Create DataFrame and save
    df = pd.DataFrame({"X": x_flat, "Y": y_flat, "f": z_flat})

    df.to_csv(filename, index=False)
    print(f"Created test gridded CSV: {filename}")
    print(f"Grid dimensions: {len(x_vals)} x {len(y_vals)} = {len(df)} points")
    print(f"X range: {x_vals.min():.2f} to {x_vals.max():.2f}")
    print(f"Y range: {y_vals.min():.2f} to {y_vals.max():.2f}")
    print("Sample data:")
    print(df.head(10))

    return x_vals, y_vals, Z


def test_gridded_interpolation():
    """Test the corrected 2D interpolation function"""

    # Create test data
    x_true, y_true, z_true = create_test_gridded_csv("test_grid.csv")

    # Load using corrected function
    jax_func, x_unique, y_unique, z_grid = create_2d_jax_function_from_csv(
        "test_grid.csv", "X", "Y", "f", method="linear"
    )

    print("\nLoaded grid:")
    print(f"X unique values: {len(x_unique)} points")
    print(f"Y unique values: {len(y_unique)} points")
    print(f"Z grid shape: {z_grid.shape}")

    # Test interpolation at grid points (should be exact)
    print("\nTesting interpolation at grid points:")
    test_points = [
        (x_true[0], y_true[0]),
        (x_true[5], y_true[3]),
        (x_true[-1], y_true[-1]),
    ]

    for x_test, y_test in test_points:
        # Get expected value
        x_idx = np.searchsorted(x_true, x_test)
        y_idx = np.searchsorted(y_true, y_test)
        expected = z_true[y_idx, x_idx]

        # Get interpolated value
        result = jax_func(x_test, y_test)

        error = abs(float(result) - expected)
        print(
            f"  Point ({x_test:.2f}, {y_test:.2f}): expected={expected:.6f}, got={float(result):.6f}, error={error:.2e}"
        )

    # Test interpolation at intermediate points
    print("\nTesting interpolation at intermediate points:")
    test_interp_points = [(1.5, 1.5), (2.7, 0.8), (4.2, 2.1)]

    for x_test, y_test in test_interp_points:
        result = jax_func(x_test, y_test)

        # Calculate expected value using true function
        expected = x_test**2 + y_test**2 + np.sin(x_test * y_test)

        error = abs(float(result) - expected)
        print(
            f"  Point ({x_test:.2f}, {y_test:.2f}): expected={expected:.6f}, got={float(result):.6f}, error={error:.2e}"
        )

    # Test boundary handling
    print("\nTesting boundary handling:")
    boundary_points = [(-0.5, 1.5), (5.5, 1.5), (2.5, -0.5), (2.5, 3.5)]

    for x_test, y_test in boundary_points:
        result = jax_func(x_test, y_test)
        print(f"  Point ({x_test:.2f}, {y_test:.2f}): result={float(result):.6f}")

    # Test JAX compatibility
    print("\nTesting JAX JIT compilation:")
    import jax

    @jax.jit
    def jit_interpolate(x, y):
        return jax_func(x, y)

    result_jit = jit_interpolate(2.0, 1.5)
    result_regular = jax_func(2.0, 1.5)

    print(f"  JIT result: {float(result_jit):.6f}")
    print(f"  Regular result: {float(result_regular):.6f}")
    print(f"  Match: {jnp.allclose(result_jit, result_regular)}")

    print("\n✓ All tests completed!")


if __name__ == "__main__":
    test_gridded_interpolation()
