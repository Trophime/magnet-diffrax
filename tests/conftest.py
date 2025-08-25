"""
Test configuration and fixtures for magnet_diffrax package tests.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import jax.numpy as jnp

# Import package modules for testing
from magnet_diffrax import (
    RLCircuitPID,
    CoupledRLCircuitsPID,
    create_adaptive_pid_controller,
    create_default_pid_controller,
)


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_reference_csv(test_data_dir):
    """Create a sample reference current CSV file."""
    time = np.linspace(0, 5, 100)
    current = np.where(
        time < 1.0, 10.0, np.where(time < 2.0, 50.0, np.where(time < 3.0, 20.0, 100.0))
    )

    df = pd.DataFrame({"time": time, "current": current})
    csv_path = test_data_dir / "sample_reference.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_resistance_csv(test_data_dir):
    """Create a sample resistance CSV file."""
    data = []
    currents = np.linspace(0, 200, 20)
    temperatures = np.linspace(20, 60, 10)

    for temp in temperatures:
        for curr in currents:
            # Simple resistance model: R = R0 * (1 + alpha*dT + beta*I)
            R0 = 1.2
            alpha = 0.004  # Temperature coefficient
            beta = 0.0001  # Current coefficient
            resistance = R0 * (1 + alpha * (temp - 25) + beta * curr)
            data.append(
                {"current": curr, "temperature": temp, "resistance": resistance}
            )

    df = pd.DataFrame(data)
    csv_path = test_data_dir / "sample_resistance.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_voltage_csv(test_data_dir):
    """Create a sample input voltage CSV file."""
    time = np.linspace(0, 5, 200)
    voltage = 5.0 + 2.0 * np.sin(2 * np.pi * time * 0.5) * (time > 0.5)

    df = pd.DataFrame({"time": time, "voltage": voltage})
    csv_path = test_data_dir / "sample_voltage.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def basic_pid_controller():
    """Create a basic PID controller for testing."""
    return create_default_pid_controller()


@pytest.fixture
def adaptive_pid_controller():
    """Create an adaptive PID controller with custom parameters."""
    return create_adaptive_pid_controller(
        Kp_low=15.0,
        Ki_low=10.0,
        Kd_low=0.08,
        Kp_medium=18.0,
        Ki_medium=12.0,
        Kd_medium=0.06,
        Kp_high=22.0,
        Ki_high=15.0,
        Kd_high=0.04,
        low_threshold=60.0,
        high_threshold=200.0,
    )


@pytest.fixture
def basic_rl_circuit(basic_pid_controller):
    """Create a basic RL circuit for testing."""
    return RLCircuitPID(
        R=1.5,
        L=0.1,
        pid_controller=basic_pid_controller,
        temperature=25.0,
        circuit_id="test_circuit",
    )


@pytest.fixture
def rl_circuit_with_csv(
    adaptive_pid_controller, sample_reference_csv, sample_resistance_csv
):
    """Create an RL circuit with CSV data."""
    return RLCircuitPID(
        R=1.0,
        L=0.1,
        pid_controller=adaptive_pid_controller,
        reference_csv=sample_reference_csv,
        resistance_csv=sample_resistance_csv,
        temperature=30.0,
        circuit_id="csv_test_circuit",
    )


@pytest.fixture
def multiple_circuits(basic_pid_controller, adaptive_pid_controller):
    """Create multiple circuits for coupled system testing."""
    circuits = []

    # Circuit 1: Basic PID
    circuits.append(
        RLCircuitPID(
            R=1.0,
            L=0.08,
            pid_controller=basic_pid_controller,
            temperature=25.0,
            circuit_id="circuit_1",
        )
    )

    # Circuit 2: Adaptive PID
    circuits.append(
        RLCircuitPID(
            R=1.2,
            L=0.10,
            pid_controller=adaptive_pid_controller,
            temperature=30.0,
            circuit_id="circuit_2",
        )
    )

    # Circuit 3: Different parameters
    circuits.append(
        RLCircuitPID(
            R=1.5,
            L=0.12,
            pid_controller=create_adaptive_pid_controller(
                Kp_low=20.0, Ki_low=8.0, Kd_low=0.05
            ),
            temperature=35.0,
            circuit_id="circuit_3",
        )
    )

    return circuits


@pytest.fixture
def coupled_rl_system(multiple_circuits):
    """Create a coupled RL circuit system."""
    # Create symmetric coupling matrix
    n_circuits = len(multiple_circuits)
    coupling_strength = 0.03
    M = np.full((n_circuits, n_circuits), coupling_strength)
    np.fill_diagonal(M, 0.0)

    return CoupledRLCircuitsPID(multiple_circuits, mutual_inductances=M)


@pytest.fixture
def simulation_time_params():
    """Standard simulation time parameters."""
    return {"t0": 0.0, "t1": 2.0, "dt": 0.01}  # Shorter for faster tests


@pytest.fixture
def jax_random_key():
    """JAX random key for reproducible tests."""
    import jax.random as random

    return random.PRNGKey(42)


# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "csv: marks tests that use CSV functionality")
    config.addinivalue_line(
        "markers", "plotting: marks tests that require plotting libraries"
    )


# Utility functions for tests
class TestUtils:
    """Utility functions for testing."""

    @staticmethod
    def create_test_mutual_inductance_matrix(
        n_circuits: int, coupling_strength: float = 0.05
    ):
        """Create a test mutual inductance matrix."""
        M = np.full((n_circuits, n_circuits), coupling_strength)
        np.fill_diagonal(M, 0.0)
        # Make it symmetric
        M = (M + M.T) / 2
        return M

    @staticmethod
    def assert_jax_array_close(a, b, rtol=1e-5, atol=1e-8):
        """Assert JAX arrays are close."""
        assert jnp.allclose(a, b, rtol=rtol, atol=atol), f"Arrays not close: {a} vs {b}"

    @staticmethod
    def create_step_reference(times, steps):
        """Create step reference signal for testing."""
        reference = np.zeros_like(times)
        for i, (t_step, value) in enumerate(steps):
            reference[times >= t_step] = value
        return reference

    @staticmethod
    def validate_solution_format(sol):
        """Validate that solution has expected format."""
        assert hasattr(sol, "ts"), "Solution missing time array"
        assert hasattr(sol, "ys"), "Solution missing state array"
        assert len(sol.ts.shape) == 1, "Time array should be 1D"
        assert len(sol.ys.shape) >= 2, "State array should be at least 2D"
        assert (
            sol.ts.shape[0] == sol.ys.shape[0]
        ), "Time and state arrays should have same length"


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils


# Skip tests if dependencies are missing
def check_dependencies():
    """Check if all required dependencies are available."""
    required = ["jax", "jaxlib", "diffrax", "numpy", "pandas"]
    missing = []

    for dep in required:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)

    return missing


missing_deps = check_dependencies()
if missing_deps:
    pytest.skip(
        f"Missing required dependencies: {missing_deps}", allow_module_level=True
    )


# Session-scoped JAX configuration
@pytest.fixture(scope="session", autouse=True)
def configure_jax_for_testing():
    """Configure JAX settings for testing."""
    try:
        import jax
        from jax import config

        # Enable 64-bit precision for testing accuracy
        config.update("jax_enable_x64", True)

        # Use CPU for consistent testing
        config.update("jax_platform_name", "cpu")

        print("JAX configured for testing: 64-bit precision, CPU platform")

    except ImportError:
        pytest.skip("JAX not available", allow_module_level=True)


# Performance testing utilities
@pytest.fixture
def performance_timer():
    """Timer for performance tests."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end_time = time.perf_counter()

        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

    return Timer


# Error tolerance settings for numerical tests
@pytest.fixture
def numerical_tolerances():
    """Standard numerical tolerances for testing."""
    return {"rtol": 1e-5, "atol": 1e-8, "rtol_loose": 1e-3, "atol_loose": 1e-6}
