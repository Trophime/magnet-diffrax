"""
Tests for JAX CSV utilities functionality.
"""

import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp
import tempfile
from pathlib import Path

from magnet_diffrax.jax_csv_utils import (
    create_jax_function_from_csv,
    create_2d_jax_function_from_csv,
    create_multi_column_jax_function_from_csv,
    create_parametric_jax_function_from_csv,
)


class TestCreateJaxFunctionFromCSV:
    """Test 1D JAX function creation from CSV."""
    
    @pytest.fixture
    def simple_csv_data(self, test_data_dir):
        """Create simple CSV data for testing."""
        x = np.linspace(0, 10, 21)
        y = 2 * x + 3  # Linear function
        
        df = pd.DataFrame({"x_values": x, "y_values": y})
        csv_path = test_data_dir / "simple_data.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)
    
    @pytest.fixture  
    def nonlinear_csv_data(self, test_data_dir):
        """Create nonlinear CSV data for testing."""
        x = np.linspace(0, 2*np.pi, 50)
        y = np.sin(x) + 0.5 * np.cos(2*x)  # Nonlinear function
        
        df = pd.DataFrame({"time": x, "signal": y})
        csv_path = test_data_dir / "nonlinear_data.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)
    
    def test_linear_interpolation(self, simple_csv_data):
        """Test linear interpolation functionality."""
        jax_func, x_data, y_data = create_jax_function_from_csv(
            simple_csv_data, "x_values", "y_values", method="linear"
        )
        
        # Test at data points
        for i in range(len(x_data)):
            result = jax_func(float(x_data[i]))
            expected = float(y_data[i])
            assert abs(result - expected) < 1e-6
        
        # Test interpolation between points
        x_mid = (float(x_data[5]) + float(x_data[6])) / 2
        y_mid = jax_func(x_mid)
        expected_mid = (float(y_data[5]) + float(y_data[6])) / 2
        assert abs(y_mid - expected_mid) < 1e-6
    
    def test_nearest_interpolation(self, simple_csv_data):
        """Test nearest neighbor interpolation."""
        jax_func, x_data, y_data = create_jax_function_from_csv(
            simple_csv_data, "x_values", "y_values", method="nearest"
        )
        
        # Test at data points
        for i in range(len(x_data)):
            result = jax_func(float(x_data[i]))
            expected = float(y_data[i])
            assert abs(result - expected) < 1e-6
        
        # Test between points (should snap to nearest)
        x_mid = (float(x_data[5]) + float(x_data[6])) / 2
        y_mid = jax_func(x_mid)
        
        # Should be one of the neighboring values
        assert abs(y_mid - float(y_data[5])) < 1e-6 or abs(y_mid - float(y_data[6])) < 1e-6
    
    def test_extrapolation_behavior(self, simple_csv_data):
        """Test behavior outside data range."""
        jax_func, x_data, y_data = create_jax_function_from_csv(
            simple_csv_data, "x_values", "y_values", method="linear"
        )
        
        # Test below range
        x_min = float(jnp.min(x_data))
        y_below = jax_func(x_min - 1.0)
        assert jnp.isfinite(y_below)
        
        # Test above range
        x_max = float(jnp.max(x_data))
        y_above = jax_func(x_max + 1.0)
        assert jnp.isfinite(y_above)
    
    def test_jax_compatibility(self, nonlinear_csv_data):
        """Test JAX JIT compilation compatibility."""
        import jax
        
        jax_func, _, _ = create_jax_function_from_csv(
            nonlinear_csv_data, "time", "signal", method="linear"
        )
        
        @jax.jit
        def jit_wrapper(x):
            return jax_func(x)
        
        # Should compile and run
        result = jit_wrapper(3.14)
        assert jnp.isfinite(result)
        assert isinstance(result, jnp.ndarray) or isinstance(result, float)
    
    def test_vectorized_input(self, simple_csv_data):
        """Test function with vectorized input."""
        jax_func, x_data, _ = create_jax_function_from_csv(
            simple_csv_data, "x_values", "y_values", method="linear"
        )
        
        # Test with array input
        x_test = jnp.array([1.0, 3.0, 5.0, 7.0])
        y_test = jnp.array([jax_func(x) for x in x_test])
        
        assert len(y_test) == len(x_test)
        assert jnp.all(jnp.isfinite(y_test))
    
    def test_invalid_method(self, simple_csv_data):
        """Test error handling for invalid interpolation method."""
        with pytest.raises(ValueError, match="Unknown method"):
            create_jax_function_from_csv(
                simple_csv_data, "x_values", "y_values", method="invalid"
            )
    
    def test_missing_columns(self, test_data_dir):
        """Test error handling for missing columns."""
        # Create CSV without required columns
        df = pd.DataFrame({"wrong_x": [1, 2, 3], "wrong_y": [4, 5, 6]})
        csv_path = test_data_dir / "missing_columns.csv"
        df.to_csv(csv_path, index=False)
        
        with pytest.raises(KeyError):
            create_jax_function_from_csv(
                str(csv_path), "x_values", "y_values", method="linear"
            )


class TestCreate2DJaxFunctionFromCSV:
    """Test 2D JAX function creation from CSV."""
    
    @pytest.fixture
    def grid_csv_data(self, test_data_dir):
        """Create 2D grid CSV data for testing."""
        x_vals = np.linspace(0, 5, 10)
        y_vals = np.linspace(0, 3, 8)
        
        data = []
        for x in x_vals:
            for y in y_vals:
                z = x**2 + y**2  # Simple 2D function
                data.append({"x_coord": x, "y_coord": y, "z_value": z})
        
        df = pd.DataFrame(data)
        csv_path = test_data_dir / "grid_data.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)
    
    def test_2d_linear_interpolation(self, grid_csv_data):
        """Test 2D linear interpolation."""
        jax_func_2d, x_data, y_data, z_data = create_2d_jax_function_from_csv(
            grid_csv_data, "x_coord", "y_coord", "z_value", method="linear"
        )
        
        # Test at a few points
        test_points = [(1.0, 1.0), (2.5, 1.5), (4.0, 2.0)]
        
        for x_test, y_test in test_points:
            result = jax_func_2d(x_test, y_test)
            assert jnp.isfinite(result)
            assert isinstance(result, (jnp.ndarray, float))
    
    def test_2d_nearest_interpolation(self, grid_csv_data):
        """Test 2D nearest neighbor interpolation."""
        jax_func_2d, _, _, _ = create_2d_jax_function_from_csv(
            grid_csv_data, "x_coord", "y_coord", "z_value", method="nearest"
        )
        
        result = jax_func_2d(2.0, 1.5)
        assert jnp.isfinite(result)
    
    def test_2d_jax_compatibility(self, grid_csv_data):
        """Test 2D function JAX compatibility."""
        import jax
        
        jax_func_2d, _, _, _ = create_2d_jax_function_from_csv(
            grid_csv_data, "x_coord", "y_coord", "z_value", method="linear"
        )
        
        @jax.jit
        def jit_2d_wrapper(x, y):
            return jax_func_2d(x, y)
        
        result = jit_2d_wrapper(2.0, 1.0)
        assert jnp.isfinite(result)
    
    def test_2d_gradients(self, grid_csv_data):
        """Test that 2D function can compute gradients."""
        import jax
        
        jax_func_2d, _, _, _ = create_2d_jax_function_from_csv(
            grid_csv_data, "x_coord", "y_coord", "z_value", method="linear"
        )
        
        # Compute gradients
        grad_func = jax.grad(lambda x, y: jax_func_2d(x, y), argnums=(0, 1))
        
        # Should be able to compute gradients
        grad_x, grad_y = grad_func(2.0, 1.0)
        assert jnp.isfinite(grad_x)
        assert jnp.isfinite(grad_y)
    
    def test_2d_invalid_method(self, grid_csv_data):
        """Test error handling for invalid 2D method."""
        with pytest.raises(ValueError, match="Unknown method"):
            create_2d_jax_function_from_csv(
                grid_csv_data, "x_coord", "y_coord", "z_value", method="invalid"
            )


class TestCreateMultiColumnJaxFunction:
    """Test multi-column JAX function creation."""
    
    @pytest.fixture
    def multi_output_csv_data(self, test_data_dir):
        """Create multi-output CSV data."""
        x = np.linspace(0, 10, 20)
        y1 = np.sin(x)
        y2 = np.cos(x)
        y3 = x**2
        
        df = pd.DataFrame({
            "input": x,
            "output1": y1,
            "output2": y2, 
            "output3": y3
        })
        csv_path = test_data_dir / "multi_output.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)
    
    def test_multi_column_linear(self, multi_output_csv_data):
        """Test multi-column linear interpolation."""
        jax_func_multi, x_data, y_data = create_multi_column_jax_function_from_csv(
            multi_output_csv_data, 
            "input", 
            ["output1", "output2", "output3"],
            method="linear"
        )
        
        # Test at a point
        result = jax_func_multi(5.0)
        
        assert isinstance(result, jnp.ndarray)
        assert len(result) == 3  # Should have 3 outputs
        assert jnp.all(jnp.isfinite(result))
    
    def test_multi_column_nearest(self, multi_output_csv_data):
        """Test multi-column nearest interpolation."""
        jax_func_multi, _, _ = create_multi_column_jax_function_from_csv(
            multi_output_csv_data,
            "input",
            ["output1", "output2"],
            method="nearest"
        )
        
        result = jax_func_multi(3.0)
        assert len(result) == 2
        assert jnp.all(jnp.isfinite(result))
    
    def test_multi_column_jax_compatibility(self, multi_output_csv_data):
        """Test multi-column JAX compatibility."""
        import jax
        
        jax_func_multi, _, _ = create_multi_column_jax_function_from_csv(
            multi_output_csv_data,
            "input",
            ["output1", "output2", "output3"],
            method="linear"
        )
        
        @jax.jit
        def jit_multi_wrapper(x):
            return jax_func_multi(x)
        
        result = jit_multi_wrapper(4.0)
        assert len(result) == 3
        assert jnp.all(jnp.isfinite(result))
    
    def test_multi_column_vectorized(self, multi_output_csv_data):
        """Test multi-column function with vectorized inputs."""
        jax_func_multi, _, _ = create_multi_column_jax_function_from_csv(
            multi_output_csv_data,
            "input",
            ["output1", "output2"],
            method="linear"
        )
        
        # Test multiple inputs
        x_test = jnp.array([1.0, 3.0, 5.0])
        
        results = []
        for x in x_test:
            results.append(jax_func_multi(x))
        
        results = jnp.array(results)
        assert results.shape == (3, 2)  # 3 inputs, 2 outputs each
        assert jnp.all(jnp.isfinite(results))


class TestCreateParametricJaxFunction:
    """Test parametric JAX function creation."""
    
    @pytest.fixture
    def parametric_csv_data(self, test_data_dir):
        """Create parametric CSV data."""
        data = []
        
        # Different parameter values
        params = [1.0, 2.0, 3.0]
        x_vals = np.linspace(0, 5, 15)
        
        for param in params:
            for x in x_vals:
                y = param * x + 0.5 * param**2  # Parametric function
                data.append({
                    "parameter": param,
                    "x_input": x,
                    "y_output": y
                })
        
        df = pd.DataFrame(data)
        csv_path = test_data_dir / "parametric_data.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)
    
    def test_parametric_linear_interpolation(self, parametric_csv_data):
        """Test parametric linear interpolation."""
        jax_func_param, param_data, x_data, y_data = create_parametric_jax_function_from_csv(
            parametric_csv_data,
            "parameter",
            "x_input", 
            "y_output",
            method="linear"
        )
        
        # Test at specific parameter and x values
        result = jax_func_param(2.0, 1.5)  # x=2.0, param=1.5
        assert jnp.isfinite(result)
        assert isinstance(result, (jnp.ndarray, float))
    
    def test_parametric_nearest_interpolation(self, parametric_csv_data):
        """Test parametric nearest interpolation."""
        jax_func_param, _, _, _ = create_parametric_jax_function_from_csv(
            parametric_csv_data,
            "parameter",
            "x_input",
            "y_output", 
            method="nearest"
        )
        
        result = jax_func_param(1.5, 2.5)  # x=1.5, param=2.5
        assert jnp.isfinite(result)
    
    def test_parametric_jax_compatibility(self, parametric_csv_data):
        """Test parametric function JAX compatibility."""
        import jax
        
        jax_func_param, _, _, _ = create_parametric_jax_function_from_csv(
            parametric_csv_data,
            "parameter",
            "x_input",
            "y_output",
            method="linear"
        )
        
        @jax.jit
        def jit_param_wrapper(x, param):
            return jax_func_param(x, param)
        
        result = jit_param_wrapper(3.0, 2.0)
        assert jnp.isfinite(result)
    
    def test_parametric_gradients(self, parametric_csv_data):
        """Test parametric function gradient computation."""
        import jax
        
        jax_func_param, _, _, _ = create_parametric_jax_function_from_csv(
            parametric_csv_data,
            "parameter", 
            "x_input",
            "y_output",
            method="linear"
        )
        
        # Gradients with respect to both x and param
        grad_func = jax.grad(lambda x, param: jax_func_param(x, param), argnums=(0, 1))
        
        grad_x, grad_param = grad_func(2.0, 1.5)
        assert jnp.isfinite(grad_x)
        assert jnp.isfinite(grad_param)
    
    def test_parametric_invalid_method(self, parametric_csv_data):
        """Test error handling for invalid parametric method."""
        with pytest.raises(ValueError, match="Unknown method"):
            create_parametric_jax_function_from_csv(
                parametric_csv_data,
                "parameter",
                "x_input",
                "y_output",
                method="invalid"
            )


@pytest.mark.unit
class TestCSVUtilsEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_csv_file(self, test_data_dir):
        """Test handling of empty CSV files."""
        # Create empty CSV
        df = pd.DataFrame()
        csv_path = test_data_dir / "empty.csv"
        df.to_csv(csv_path, index=False)
        
        with pytest.raises((ValueError, KeyError, IndexError)):
            create_jax_function_from_csv(
                str(csv_path), "x", "y", method="linear"
            )
    
    def test_single_point_csv(self, test_data_dir):
        """Test handling of CSV with single data point."""
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        csv_path = test_data_dir / "single_point.csv"
        df.to_csv(csv_path, index=False)
        
        jax_func, _, _ = create_jax_function_from_csv(
            str(csv_path), "x", "y", method="nearest"
        )
        
        # Should return the single value for any input
        result = jax_func(5.0)
        assert abs(result - 2.0) < 1e-6
    
    def test_unsorted_data(self, test_data_dir):
        """Test handling of unsorted CSV data."""
        # Create unsorted data
        x = np.array([5, 1, 3, 2, 4])
        y = x * 2  # Simple relationship
        
        df = pd.DataFrame({"x_vals": x, "y_vals": y})
        csv_path = test_data_dir / "unsorted.csv"
        df.to_csv(csv_path, index=False)
        
        jax_func, x_data, y_data = create_jax_function_from_csv(
            str(csv_path), "x_vals", "y_vals", method="linear"
        )
        
        # Data should be sorted internally
        assert jnp.all(x_data[:-1] <= x_data[1:])  # Should be sorted
        
        # Function should still work correctly
        result = jax_func(2.5)
        assert jnp.isfinite(result)
    
    def test_duplicate_x_values(self, test_data_dir):
        """Test handling of duplicate x values."""
        # Create data with duplicate x values
        x = np.array([1, 2, 2, 3, 4])  # Duplicate x=2
        y = np.array([2, 4, 5, 6, 8])  # Different y values for x=2
        
        df = pd.DataFrame({"x_vals": x, "y_vals": y})
        csv_path = test_data_dir / "duplicates.csv"
        df.to_csv(csv_path, index=False)
        
        # Should handle duplicates (possibly by taking last value or averaging)
        jax_func, _, _ = create_jax_function_from_csv(
            str(csv_path), "x_vals", "y_vals", method="linear"
        )
        
        result = jax_func(2.0)
        assert jnp.isfinite(result)
    
    def test_nan_values(self, test_data_dir):
        """Test handling of NaN values in CSV."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, np.nan, 8, 10])  # Contains NaN
        
        df = pd.DataFrame({"x_vals": x, "y_vals": y})
        csv_path = test_data_dir / "with_nan.csv"
        df.to_csv(csv_path, index=False)
        
        # Should either handle NaN or raise appropriate error
        try:
            jax_func, _, _ = create_jax_function_from_csv(
                str(csv_path), "x_vals", "y_vals", method="linear"
            )
            
            # If it succeeds, function should still work at non-NaN points
            result = jax_func(1.0)
            assert jnp.isfinite(result)
            
        except (ValueError, RuntimeError):
            # It's acceptable to fail with NaN values
            pass
    
    def test_nonexistent_file(self):
        """Test handling of nonexistent CSV file."""
        with pytest.raises(FileNotFoundError):
            create_jax_function_from_csv(
                "nonexistent_file.csv", "x", "y", method="linear"
            )
    
    def test_malformed_csv(self, test_data_dir):
        """Test handling of malformed CSV file."""
        # Create malformed CSV
        malformed_content = "x,y\n1,2\n3,4,extra_column\n5,6"
        csv_path = test_data_dir / "malformed.csv"
        
        with open(csv_path, 'w') as f:
            f.write(malformed_content)
        
        # Pandas should handle this gracefully or raise appropriate error
        try:
            jax_func, _, _ = create_jax_function_from_csv(
                str(csv_path), "x", "y", method="linear"
            )
            # If it succeeds, test that it works
            result = jax_func(2.0)
            assert jnp.isfinite(result)
            
        except (pd.errors.Error, ValueError):
            # It's acceptable to fail with malformed CSV
            pass


@pytest.mark.performance
class TestCSVUtilsPerformance:
    """Performance tests for CSV utilities."""
    
    def test_interpolation_speed(self, test_data_dir, performance_timer):
        """Test interpolation performance with medium dataset."""
        # Create medium-sized dataset
        n_points = 500
        x = np.linspace(0, 50, n_points)
        y = np.sin(x) + 0.1 * x  # Simple function
        
        df = pd.DataFrame({"x": x, "y": y})
        csv_path = test_data_dir / "perf_dataset.csv"
        df.to_csv(csv_path, index=False)
        
        # Load function
        jax_func, _, _ = create_jax_function_from_csv(
            str(csv_path), "x", "y", method="linear"
        )
        
        # Time multiple evaluations
        test_points = np.linspace(5, 45, 100)
        
        with performance_timer() as timer:
            results = [jax_func(x_val) for x_val in test_points]
        
        # Should be reasonably fast (relaxed for CI)
        assert timer.elapsed < 2.0  # Less than 2 seconds
        assert all(jnp.isfinite(r) for r in results)
    
    def test_jit_compilation_overhead(self, simple_csv_data):
        """Test JIT compilation overhead vs benefit."""
        import jax
        import time
        
        jax_func, _, _ = create_jax_function_from_csv(
            simple_csv_data, "x_values", "y_values", method="linear"
        )
        
        @jax.jit
        def jit_wrapper(x):
            return jax_func(x)
        
        # First call (includes compilation time)
        start_time = time.perf_counter()
        result1 = jit_wrapper(5.0)
        compile_time = time.perf_counter() - start_time
        
        # Subsequent calls (compiled)
        start_time = time.perf_counter() 
        for _ in range(10):
            result2 = jit_wrapper(5.0)
        execution_time = (time.perf_counter() - start_time) / 10
        
        # Results should be consistent
        assert jnp.allclose(result1, result2)
        
        # Compilation should be reasonable (relaxed for different systems)
        assert compile_time < 10.0  # Less than 10 seconds
        assert execution_time < 0.01  # Fast after compilation
    
    @pytest.mark.skipif(
        not pytest.importorskip("psutil", reason="psutil not available"),
        reason="Requires psutil for memory testing"
    )
    def test_memory_usage_large_dataset(self, test_data_dir):
        """Test memory efficiency with larger datasets.""" 
        try:
            import gc
            import psutil
            import os
        except ImportError:
            pytest.skip("psutil not available for memory testing")
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        n_points = 5000
        x = np.linspace(0, 100, n_points)
        y = np.sin(x) + 0.01 * np.random.randn(n_points)
        
        df = pd.DataFrame({"x": x, "y": y})
        csv_path = test_data_dir / "large_memory_test.csv"
        df.to_csv(csv_path, index=False)
        
        # Load and use function
        jax_func, _, _ = create_jax_function_from_csv(
            str(csv_path), "x", "y", method="linear"
        )
        
        # Test multiple evaluations
        for i in range(100):
            result = jax_func(float(i))
            assert jnp.isfinite(result)
        
        # Check memory usage hasn't grown excessively
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not use excessive memory (relaxed for different systems)
        assert memory_increase < 200


@pytest.mark.integration
class TestCSVUtilsIntegration:
    """Integration tests with real simulation use cases."""
    
    def test_resistance_function_integration(self, sample_resistance_csv):
        """Test integration with resistance function use case."""
        # This mimics how resistance CSV is used in RLCircuitPID
        jax_func_2d, current_data, temp_data, resistance_data = create_2d_jax_function_from_csv(
            sample_resistance_csv,
            "current",
            "temperature", 
            "resistance",
            method="linear"
        )
        
        # Test that it behaves as expected for circuit simulation
        test_currents = [0.0, 50.0, 100.0, 150.0]
        test_temperature = 30.0
        
        resistances = []
        for current in test_currents:
            R = jax_func_2d(current, test_temperature)
            resistances.append(float(R))
            assert R > 0  # Resistance should be positive
            assert jnp.isfinite(R)
        
        # Resistance should generally increase with current (for typical model)
        # This depends on the specific resistance model used in the test data
    
    def test_reference_function_integration(self, sample_reference_csv):
        """Test integration with reference current use case."""
        # This mimics how reference CSV is used in RLCircuitPID
        jax_func, time_data, current_data = create_jax_function_from_csv(
            sample_reference_csv,
            "time",
            "current", 
            method="linear"
        )
        
        # Test time series evaluation
        test_times = np.linspace(float(time_data.min()), float(time_data.max()), 50)
        
        for t in test_times:
            i_ref = jax_func(t)
            assert jnp.isfinite(i_ref)
            assert float(i_ref) >= 0  # Current should be non-negative
    
    @pytest.mark.slow
    def test_performance_with_large_dataset(self, test_data_dir):
        """Test performance with larger datasets.""" 
        # Create larger dataset
        n_points = 1000
        x = np.linspace(0, 100, n_points)
        y = np.sin(x) + 0.1 * np.random.randn(n_points)
        
        df = pd.DataFrame({"x": x, "y": y})
        csv_path = test_data_dir / "large_dataset.csv"
        df.to_csv(csv_path, index=False)
        
        # Should handle large dataset efficiently
        jax_func, _, _ = create_jax_function_from_csv(
            str(csv_path), "x", "y", method="linear"
        )
        
        # Test multiple evaluations (simulating time series)
        test_x = np.linspace(10, 90, 100)
        
        import time
        start_time = time.time()
        
        for x_val in test_x:
            result = jax_func(x_val)
            assert jnp.isfinite(result)
        
        end_time = time.time()
        
        # Should complete in reasonable time (less than 1 second for this size)
        assert (end_time - start_time) < 1.0
    
    def test_jit_compilation_benefits(self, sample_reference_csv):
        """Test that JIT compilation provides performance benefits."""
        import jax
        import time
        
        jax_func, _, _ = create_jax_function_from_csv(
            sample_reference_csv, "time", "current", method="linear"
        )
        
        @jax.jit
        def jit_wrapper(x):
            return jax_func(x)
        
        # Warm up JIT
        _ = jit_wrapper(1.0)
        
        # Time regular function
        start_time = time.time()
        for _ in range(100):
            _ = jax_func(2.0)
        regular_time = time.time() - start_time
        
        # Time JIT function
        start_time = time.time()
        for _ in range(100):
            _ = jit_wrapper(2.0)
        jit_time = time.time() - start_time
        
        # JIT should be at least as fast (usually much faster)
        assert jit_time <= regular_time * 2  # Allow some tolerance
    
    def test_gradient_computation_integration(self, nonlinear_csv_data):
        """Test gradient computation for optimization use cases."""
        import jax
        
        jax_func, _, _ = create_jax_function_from_csv(
            nonlinear_csv_data, "time", "signal", method="linear"
        )
        
        # Create a loss function using the CSV function
        def loss_function(x):
            y_pred = jax_func(x)
            y_target = 0.5  # Target value
            return (y_pred - y_target)**2
        
        # Compute gradients
        grad_loss = jax.grad(loss_function)
        
        # Test gradient at several points
        test_points = [1.0, 2.0, 3.0, 4.0]
        
        for x in test_points:
            gradient = grad_loss(x)
            assert jnp.isfinite(gradient)
            assert isinstance(gradient, (jnp.ndarray, float))
        
        # Could be used for optimization to find x where jax_func(x) ≈ 0.5


class TestCSVUtilsErrorRecovery:
    """Test error recovery and robustness."""
    
    def test_corrupted_csv_graceful_handling(self, test_data_dir):
        """Test graceful handling of slightly corrupted CSV."""
        # Create CSV with some problematic rows
        content = """x,y
1.0,2.0
2.0,4.0
3.0,corrupted_value
4.0,8.0
5.0,10.0"""
        
        csv_path = test_data_dir / "corrupted.csv"
        with open(csv_path, 'w') as f:
            f.write(content)
        
        # Should either handle gracefully or raise clear error
        try:
            jax_func, _, _ = create_jax_function_from_csv(
                str(csv_path), "x", "y", method="linear"
            )
            
            # If successful, test that it works for valid data
            result = jax_func(1.5)
            assert jnp.isfinite(result)
            
        except (ValueError, pd.errors.Error) as e:
            # Acceptable to fail with clear error message
            assert "corrupted" in str(e).lower() or "convert" in str(e).lower()
    
    def test_extreme_values_handling(self, test_data_dir):
        """Test handling of extreme numerical values."""
        # Create data with extreme values
        x = np.array([1e-10, 1e-5, 1.0, 1e5, 1e10])
        y = np.array([1e-15, 1e-8, 1.0, 1e8, 1e15])
        
        df = pd.DataFrame({"x": x, "y": y})
        csv_path = test_data_dir / "extreme_values.csv"
        df.to_csv(csv_path, index=False)
        
        jax_func, _, _ = create_jax_function_from_csv(
            str(csv_path), "x", "y", method="linear"
        )
        
        # Test interpolation with extreme values
        test_points = [1e-8, 1.0, 1e8]
        
        for x_test in test_points:
            result = jax_func(x_test)
            # Result should be finite (not NaN or inf)
            assert jnp.isfinite(result)
    
    def test_unicode_handling(self, test_data_dir):
        """Test handling of CSV files with unicode characters."""
        # Create CSV with unicode column names
        x = np.linspace(0, 5, 10)
        y = x**2
        
        df = pd.DataFrame({"time_μs": x, "signal_Ω": y})  # Unicode characters
        csv_path = test_data_dir / "unicode.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Should handle unicode column names
        jax_func, _, _ = create_jax_function_from_csv(
            str(csv_path), "time_μs", "signal_Ω", method="linear"
        )
        
        result = jax_func(2.5)
        assert jnp.isfinite(result)
    
    def test_mixed_data_types(self, test_data_dir):
        """Test handling of CSV with mixed data types."""
        # Create CSV with mixed types
        data = {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0],
            "category": ["A", "B", "A", "C", "B"],  # String column
            "flag": [True, False, True, False, True]  # Boolean column
        }
        
        df = pd.DataFrame(data)
        csv_path = test_data_dir / "mixed_types.csv"
        df.to_csv(csv_path, index=False)
        
        # Should work with numeric columns
        jax_func, _, _ = create_jax_function_from_csv(
            str(csv_path), "x", "y", method="linear"
        )
        
        result = jax_func(2.5)
        assert jnp.isfinite(result)
        assert abs(result - 5.0) < 1e-6  # Should interpolate correctly


class TestCSVUtilsComplexScenarios:
    """Test complex real-world scenarios."""
    
    def test_time_series_with_gaps(self, test_data_dir):
        """Test time series data with gaps."""
        # Create time series with gaps
        t1 = np.linspace(0, 5, 50)
        t2 = np.linspace(8, 12, 40)  # Gap from 5 to 8
        t3 = np.linspace(15, 20, 30)  # Gap from 12 to 15
        
        time = np.concatenate([t1, t2, t3])
        signal = np.sin(time) + 0.1 * time
        
        df = pd.DataFrame({"time": time, "signal": signal})
        csv_path = test_data_dir / "time_series_gaps.csv"
        df.to_csv(csv_path, index=False)
        
        jax_func, _, _ = create_jax_function_from_csv(
            str(csv_path), "time", "signal", method="linear"
        )
        
        # Test interpolation in gaps (extrapolation)
        result_in_gap = jax_func(6.5)  # In gap between 5 and 8
        assert jnp.isfinite(result_in_gap)
        
        # Test normal interpolation
        result_normal = jax_func(2.5)  # In dense region
        assert jnp.isfinite(result_normal)
    
    def test_noisy_experimental_data(self, test_data_dir):
        """Test with noisy experimental-like data."""
        # Simulate noisy experimental data
        n_points = 200
        x = np.sort(np.random.uniform(0, 10, n_points))  # Irregular spacing
        
        # Underlying function with noise
        y_true = 2 * x + 5 * np.sin(x)
        noise = 0.5 * np.random.randn(n_points)
        y = y_true + noise
        
        df = pd.DataFrame({"input": x, "output": y})
        csv_path = test_data_dir / "noisy_experimental.csv"
        df.to_csv(csv_path, index=False)
        
        jax_func, _, _ = create_jax_function_from_csv(
            str(csv_path), "input", "output", method="linear"
        )
        
        # Test that interpolation works despite noise
        test_points = np.linspace(1, 9, 20)
        results = [jax_func(x_val) for x_val in test_points]
        
        assert all(jnp.isfinite(r) for r in results)
        
        # Results should be roughly in the expected range
        assert min(results) > min(y) - 5  # Reasonable bounds
        assert max(results) < max(y) + 5
    
    def test_multi_physics_simulation_data(self, test_data_dir):
        """Test with multi-physics simulation-like data."""
        # Simulate resistance data as function of current and temperature
        currents = np.linspace(0, 100, 25)
        temperatures = np.linspace(20, 80, 20)
        
        data = []
        for T in temperatures:
            for I in currents:
                # Realistic resistance model
                R0 = 1.2
                alpha = 0.004  # Temperature coefficient
                beta = 0.0001  # Current coefficient  
                R = R0 * (1 + alpha * (T - 25) + beta * I)
                
                data.append({
                    "current_A": I,
                    "temperature_C": T, 
                    "resistance_ohm": R
                })
        
        df = pd.DataFrame(data)
        csv_path = test_data_dir / "resistance_model.csv"
        df.to_csv(csv_path, index=False)
        
        # Test 2D function creation
        jax_func_2d, _, _, _ = create_2d_jax_function_from_csv(
            str(csv_path), 
            "current_A", 
            "temperature_C",
            "resistance_ohm",
            method="linear"
        )
        
        # Test realistic operating points
        test_points = [
            (25.0, 30.0),  # Low current, low temp
            (50.0, 45.0),  # Medium current, medium temp
            (75.0, 60.0),  # High current, high temp
        ]
        
        for I_test, T_test in test_points:
            R_test = jax_func_2d(I_test, T_test)
            
            # Should be physically reasonable
            assert 1.0 < float(R_test) < 2.0  # Reasonable resistance range
            assert jnp.isfinite(R_test)
        
        # Test that resistance increases with temperature and current
        R_low = jax_func_2d(10.0, 25.0)
        R_high_temp = jax_func_2d(10.0, 50.0)  # Same current, higher temp
        R_high_current = jax_func_2d(50.0, 25.0)  # Same temp, higher current
        
        assert float(R_high_temp) > float(R_low)  # Higher temp = higher resistance
        assert float(R_high_current) > float(R_low)  # Higher current = higher resistance
    
    def test_control_system_reference_signals(self, test_data_dir):
        """Test with complex control system reference signals.""" 
        # Create complex reference signal
        t = np.linspace(0, 10, 500)
        
        # Multi-step reference with smooth transitions
        ref = np.zeros_like(t)
        ref[t >= 1.0] = 10.0  # Step to 10
        ref[t >= 3.0] = 25.0  # Step to 25  
        ref[t >= 5.0] = 50.0 + 20.0 * np.sin(2*np.pi*0.5*(t[t >= 5.0] - 5.0))  # Sinusoidal
        ref[t >= 8.0] = 15.0  # Step back down
        
        # Add some simple smoothing (without scipy dependency)
        # Simple moving average smoothing
        window_size = 5
        ref_smooth = np.copy(ref)
        for i in range(window_size, len(ref) - window_size):
            ref_smooth[i] = np.mean(ref[i-window_size:i+window_size])
        
        df = pd.DataFrame({"time_s": t, "reference_A": ref_smooth})
        csv_path = test_data_dir / "control_reference.csv"
        df.to_csv(csv_path, index=False)
        
        jax_func, _, _ = create_jax_function_from_csv(
            str(csv_path), "time_s", "reference_A", method="linear"  
        )
        
        # Test that we can evaluate at arbitrary times
        test_times = [0.5, 2.0, 4.0, 6.5, 9.0]
        
        references = []
        for t_test in test_times:
            ref_val = jax_func(t_test)
            references.append(float(ref_val))
            assert jnp.isfinite(ref_val)
            assert ref_val >= 0  # Current should be non-negative
        
        # Should show the step changes
        assert references[0] < 5.0   # Before first step
        assert 5.0 < references[1] < 15.0  # After first step
        assert 20.0 < references[2] < 30.0  # After second step
        assert references[3] > 30.0  # In sinusoidal region
        assert 10.0 < references[4] < 20.0  # After final step
    
    @pytest.mark.slow
    def test_high_frequency_sampling(self, test_data_dir):
        """Test with high-frequency sampled data."""
        # High frequency data (simulating fast ADC sampling)
        fs = 1000  # 1 kHz sampling (reduced for faster tests)
        duration = 0.1  # 100 ms
        t = np.linspace(0, duration, int(fs * duration))
        
        # Signal with multiple frequency components
        signal = (2.0 * np.sin(2*np.pi*50*t) +    # 50 Hz component
                 0.5 * np.sin(2*np.pi*200*t) +    # 200 Hz component  
                 0.02 * np.random.randn(len(t)))   # Noise (reduced amplitude)
        
        df = pd.DataFrame({"time": t, "voltage": signal})
        csv_path = test_data_dir / "high_freq_data.csv"
        df.to_csv(csv_path, index=False)
        
        jax_func, _, _ = create_jax_function_from_csv(
            str(csv_path), "time", "voltage", method="linear"
        )
        
        # Test interpolation at intermediate points
        t_interp = np.linspace(0.01, 0.09, 50)  # Reduced number of points
        
        voltages = []
        for t_val in t_interp:
            v = jax_func(t_val) 
            voltages.append(float(v))
            assert jnp.isfinite(v)
        
        # Should preserve signal characteristics
        voltages = np.array(voltages)
        assert np.std(voltages) > 0.1  # Should have reasonable variation
        assert abs(np.mean(voltages)) < 1.0  # Should be roughly zero-mean


class TestCSVUtilsUtilities:
    """Test utility functions and helper methods."""
    
    def test_function_serialization(self, simple_csv_data):
        """Test that created functions can be used in different contexts.""" 
        jax_func, _, _ = create_jax_function_from_csv(
            simple_csv_data, "x_values", "y_values", method="linear"
        )
        
        # Test that function can be called multiple times
        results = []
        for i in range(10):
            result = jax_func(float(i))
            results.append(result)
        
        assert len(results) == 10
        assert all(jnp.isfinite(r) for r in results)
    
    def test_function_closure_behavior(self, simple_csv_data):
        """Test that functions properly capture data in closure."""
        jax_func1, _, _ = create_jax_function_from_csv(
            simple_csv_data, "x_values", "y_values", method="linear"
        )
        
        # Create another function from different data
        x2 = np.array([1, 2, 3, 4])
        y2 = np.array([10, 20, 30, 40])  # Different slope
        df2 = pd.DataFrame({"x": x2, "y": y2})
        
        csv_path2 = Path(simple_csv_data).parent / "different_data.csv"
        df2.to_csv(csv_path2, index=False)
        
        jax_func2, _, _ = create_jax_function_from_csv(
            str(csv_path2), "x", "y", method="linear"
        )
        
        # Functions should give different results
        test_x = 2.5
        result1 = jax_func1(test_x)
        result2 = jax_func2(test_x)
        
        assert result1 != result2  # Should be different
        assert jnp.isfinite(result1)
        assert jnp.isfinite(result2)
    
    def test_data_range_properties(self, nonlinear_csv_data):
        """Test that data range information is preserved."""
        jax_func, x_data, y_data = create_jax_function_from_csv(
            nonlinear_csv_data, "time", "signal", method="linear"
        )
        
        # Check data properties
        assert len(x_data) == len(y_data)
        assert len(x_data) > 0
        
        # Data should be sorted
        assert jnp.all(x_data[:-1] <= x_data[1:])
        
        # All data should be finite
        assert jnp.all(jnp.isfinite(x_data))
        assert jnp.all(jnp.isfinite(y_data))
    
    def test_interpolation_consistency(self, simple_csv_data):
        """Test that linear and nearest give consistent results at data points."""
        jax_func_linear, x_data, y_data = create_jax_function_from_csv(
            simple_csv_data, "x_values", "y_values", method="linear"
        )
        
        jax_func_nearest, _, _ = create_jax_function_from_csv(
            simple_csv_data, "x_values", "y_values", method="nearest"
        )
        
        # At data points, both methods should give same result
        for i in range(0, len(x_data), 3):  # Test every 3rd point
            x_test = float(x_data[i])
            
            result_linear = jax_func_linear(x_test)
            result_nearest = jax_func_nearest(x_test)
            
            assert jnp.allclose(result_linear, result_nearest, rtol=1e-5)


class TestCSVUtilsRegressionTests:
    """Regression tests for specific issues that might arise."""
    
    def test_single_value_interpolation(self, test_data_dir):
        """Regression test: ensure single-point data doesn't crash."""
        df = pd.DataFrame({"x": [5.0], "y": [10.0]})
        csv_path = test_data_dir / "single_point_regression.csv"
        df.to_csv(csv_path, index=False)
        
        jax_func, _, _ = create_jax_function_from_csv(
            str(csv_path), "x", "y", method="nearest"
        )
        
        # Should return the single value for any input
        test_inputs = [-10.0, 0.0, 5.0, 100.0]
        for x_test in test_inputs:
            result = jax_func(x_test)
            assert jnp.allclose(result, 10.0)
    
    def test_identical_x_values_handling(self, test_data_dir):
        """Regression test: handle duplicate x values gracefully."""
        # Data with some duplicate x values
        x = np.array([1, 2, 2, 2, 3, 4])  # Multiple x=2 values
        y = np.array([1, 2, 3, 4, 5, 6])  # Different y values
        
        df = pd.DataFrame({"x": x, "y": y})
        csv_path = test_data_dir / "duplicate_x_regression.csv"
        df.to_csv(csv_path, index=False)
        
        # Should handle duplicates (typically by taking the last value)
        jax_func, x_data, y_data = create_jax_function_from_csv(
            str(csv_path), "x", "y", method="linear"
        )
        
        # Should still create a valid function
        result = jax_func(2.0)
        assert jnp.isfinite(result)
        
        # Data arrays should be well-formed
        assert jnp.all(jnp.isfinite(x_data))
        assert jnp.all(jnp.isfinite(y_data))
    
    def test_very_small_dataset(self, test_data_dir):
        """Regression test: handle very small datasets."""
        # Minimal viable dataset
        df = pd.DataFrame({
            "input": [0.0, 1.0],
            "output": [0.0, 1.0]
        })
        csv_path = test_data_dir / "minimal_dataset.csv"
        df.to_csv(csv_path, index=False)
        
        jax_func, _, _ = create_jax_function_from_csv(
            str(csv_path), "input", "output", method="linear"
        )
        
        # Should work for interpolation
        result_middle = jax_func(0.5)
        assert jnp.allclose(result_middle, 0.5)  # Linear interpolation
        
        # Should work for extrapolation
        result_extrap = jax_func(2.0)
        assert jnp.isfinite(result_extrap)
    
    def test_large_value_ranges(self, test_data_dir):
        """Regression test: handle large value ranges without overflow."""
        # Data spanning many orders of magnitude
        x = np.logspace(-5, 5, 20)  # 10^-5 to 10^5
        y = x**0.5  # Square root relationship
        
        df = pd.DataFrame({"x": x, "y": y})
        csv_path = test_data_dir / "large_range.csv"
        df.to_csv(csv_path, index=False)
        
        jax_func, _, _ = create_jax_function_from_csv(
            str(csv_path), "x", "y", method="linear"
        )
        
        # Test at various scales
        test_points = [1e-3, 1.0, 1e3]
        
        for x_test in test_points:
            result = jax_func(x_test)
            assert jnp.isfinite(result)
            assert result > 0  # Should be positive for sqrt function
    
    def test_zero_and_negative_values(self, test_data_dir):
        """Regression test: handle zero and negative values properly."""
        x = np.array([-5, -2, 0, 2, 5])
        y = np.array([-10, -4, 0, 4, 10])  # Linear relationship
        
        df = pd.DataFrame({"x": x, "y": y})
        csv_path = test_data_dir / "negative_values.csv"
        df.to_csv(csv_path, index=False)
        
        jax_func, _, _ = create_jax_function_from_csv(
            str(csv_path), "x", "y", method="linear"
        )
        
        # Test around zero
        test_points = [-1.0, 0.0, 1.0]
        expected_results = [-2.0, 0.0, 2.0]
        
        for x_test, expected in zip(test_points, expected_results):
            result = jax_func(x_test)
            assert jnp.allclose(result, expected, rtol=1e-5)


# Mark all tests with csv marker for easy filtering
pytestmark = pytest.mark.csv