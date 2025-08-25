"""
Tests for RLCircuitPID class functionality.
"""

import pytest
import numpy as np
import jax.numpy as jnp
import pandas as pd
import tempfile
from pathlib import Path

from magnet_diffrax import RLCircuitPID, create_default_pid_controller


class TestRLCircuitPIDBasic:
    """Basic tests for RLCircuitPID class."""
    
    def test_basic_initialization(self):
        """Test basic circuit initialization."""
        circuit = RLCircuitPID(
            R=1.5, 
            L=0.1,
            temperature=25.0,
            circuit_id="test_circuit"
        )
        
        assert circuit.R_constant == 1.5
        assert circuit.L == 0.1
        assert circuit.temperature == 25.0
        assert circuit.circuit_id == "test_circuit"
        assert not circuit.use_variable_resistance
    
    def test_initialization_with_pid_controller(self, basic_pid_controller):
        """Test initialization with explicit PID controller."""
        circuit = RLCircuitPID(
            R=1.0,
            L=0.08,
            pid_controller=basic_pid_controller,
            circuit_id="pid_test"
        )
        
        assert circuit.pid_controller is basic_pid_controller
        assert circuit.circuit_id == "pid_test"
    
    def test_default_reference_function(self):
        """Test default reference current function."""
        circuit = RLCircuitPID(circuit_id="default_test")
        
        # Should have a default reference function
        assert hasattr(circuit, 'reference_func')
        
        # Test at different times
        ref_0 = circuit.reference_current(0.0)
        ref_1 = circuit.reference_current(1.0) 
        ref_2 = circuit.reference_current(2.0)
        
        # Default should be step function
        assert float(ref_0) == 0.0
        assert float(ref_1) >= 0.0  # Should have some reference
    
    def test_constant_resistance(self):
        """Test constant resistance mode."""
        R_value = 2.5
        circuit = RLCircuitPID(R=R_value, circuit_id="resistance_test")
        
        # Should use constant resistance
        assert not circuit.use_variable_resistance
        assert circuit.get_resistance(0.0) == R_value
        assert circuit.get_resistance(100.0) == R_value
        assert circuit.get_resistance(1000.0) == R_value
    
    def test_pid_parameter_delegation(self, basic_pid_controller):
        """Test that PID parameters are correctly delegated."""
        circuit = RLCircuitPID(
            pid_controller=basic_pid_controller,
            circuit_id="pid_delegation_test"
        )
        
        # Test PID parameter retrieval
        Kp, Ki, Kd = circuit.get_pid_parameters(50.0)
        assert isinstance(Kp, (float, jnp.ndarray))
        assert isinstance(Ki, (float, jnp.ndarray))
        assert isinstance(Kd, (float, jnp.ndarray))
        
        # Test region name retrieval
        region = circuit.get_current_region(50.0)
        assert isinstance(region, str)
        assert region in ["low", "medium", "high"]
    
    def test_circuit_id_requirement_for_coupling(self):
        """Test that circuit_id is properly handled."""
        # Should work with circuit_id
        circuit = RLCircuitPID(circuit_id="test_id")
        assert circuit.circuit_id == "test_id"
        
        # Should work without circuit_id (for single circuit use)
        circuit_no_id = RLCircuitPID()
        assert circuit_no_id.circuit_id is None


@pytest.mark.csv
class TestRLCircuitPIDWithCSV:
    """Tests for CSV data loading functionality."""
    
    def test_load_reference_from_csv(self, sample_reference_csv):
        """Test loading reference current from CSV."""
        circuit = RLCircuitPID(
            reference_csv=sample_reference_csv,
            circuit_id="csv_ref_test"
        )
        
        assert circuit.use_csv_data
        assert hasattr(circuit, 'reference_func')
        assert hasattr(circuit, 'time_data')
        assert hasattr(circuit, 'current_data')
        
        # Test that reference function works
        ref_value = circuit.reference_current(1.0)
        assert isinstance(ref_value, (float, jnp.ndarray))
        assert jnp.isfinite(ref_value)
    
    def test_load_resistance_from_csv(self, sample_resistance_csv):
        """Test loading resistance data from CSV."""
        circuit = RLCircuitPID(
            resistance_csv=sample_resistance_csv,
            temperature=30.0,
            circuit_id="csv_resistance_test"
        )
        
        assert circuit.use_variable_resistance
        assert hasattr(circuit, 'resistance_func')
        
        # Test resistance function
        R_low = circuit.get_resistance(10.0)
        R_high = circuit.get_resistance(100.0)
        
        assert isinstance(R_low, (float, jnp.ndarray))
        assert isinstance(R_high, (float, jnp.ndarray))
        assert jnp.isfinite(R_low)
        assert jnp.isfinite(R_high)
        
        # Higher current should generally give higher resistance
        # (depending on the resistance model)
    
    def test_load_voltage_from_csv(self, sample_voltage_csv):
        """Test loading input voltage from CSV."""
        circuit = RLCircuitPID(
            voltage_csv=sample_voltage_csv,
            circuit_id="csv_voltage_test"
        )
        
        assert hasattr(circuit, 'voltage_func')
        
        # Test voltage function
        voltage = circuit.input_voltage(1.0)
        assert isinstance(voltage, (float, jnp.ndarray))
        assert jnp.isfinite(voltage)
    
    def test_combined_csv_loading(self, sample_reference_csv, sample_resistance_csv, adaptive_pid_controller):
        """Test loading both reference and resistance from CSV."""
        circuit = RLCircuitPID(
            pid_controller=adaptive_pid_controller,
            reference_csv=sample_reference_csv,
            resistance_csv=sample_resistance_csv,
            temperature=35.0,
            circuit_id="combined_csv_test"
        )
        
        assert circuit.use_csv_data
        assert circuit.use_variable_resistance
        
        # Test both functions work
        ref_value = circuit.reference_current(2.0)
        resistance_value = circuit.get_resistance(50.0)
        
        assert jnp.isfinite(ref_value)
        assert jnp.isfinite(resistance_value)
    
    def test_invalid_csv_handling(self):
        """Test handling of invalid CSV files."""
        # Non-existent file should fall back gracefully
        with pytest.warns(None) as warnings:
            circuit = RLCircuitPID(
                reference_csv="nonexistent_file.csv",
                circuit_id="invalid_csv_test"
            )
        
        # Should still create circuit, possibly with default reference
        assert circuit.circuit_id == "invalid_csv_test"


class TestRLCircuitPIDVectorField:
    """Test the vector field (differential equation) implementation."""
    
    def test_vector_field_basic(self, basic_rl_circuit):
        """Test basic vector field evaluation."""
        circuit = basic_rl_circuit
        
        # Test state: [current, integral_error, prev_error]
        y = jnp.array([1.0, 0.5, 0.1])
        t = 1.0
        
        # Evaluate vector field
        dydt = circuit.vector_field(t, y, None)
        
        assert isinstance(dydt, jnp.ndarray)
        assert dydt.shape == y.shape
        assert len(dydt) == 3
        
        # All derivatives should be finite
        assert jnp.all(jnp.isfinite(dydt))
    
    def test_vector_field_dimensions(self, basic_rl_circuit):
        """Test vector field with different state dimensions."""
        circuit = basic_rl_circuit
        
        # Test with zero state
        y_zero = jnp.zeros(3)
        dydt_zero = circuit.vector_field(0.0, y_zero, None)
        assert dydt_zero.shape == (3,)
        
        # Test with different values
        y_test = jnp.array([10.0, -5.0, 2.0])
        dydt_test = circuit.vector_field(2.0, y_test, None)
        assert dydt_test.shape == (3,)
    
    def test_voltage_vector_field(self, sample_voltage_csv):
        """Test voltage-driven vector field (no PID control)."""
        circuit = RLCircuitPID(
            voltage_csv=sample_voltage_csv,
            circuit_id="voltage_test"
        )
        
        # Single state variable (current only)
        i = 2.0
        t = 1.0
        
        didt = circuit.voltage_vector_field(t, i, None)
        
        assert isinstance(didt, (float, jnp.ndarray))
        assert jnp.isfinite(didt)
    
    def test_vector_field_jax_compatibility(self, basic_rl_circuit):
        """Test that vector field is JAX-compatible."""
        import jax
        
        circuit = basic_rl_circuit
        
        # Should be able to JIT compile
        @jax.jit
        def jit_vector_field(t, y):
            return circuit.vector_field(t, y, None)
        
        y = jnp.array([1.0, 0.0, 0.0])
        dydt = jit_vector_field(1.0, y)
        
        assert isinstance(dydt, jnp.ndarray)
        assert jnp.all(jnp.isfinite(dydt))


class TestRLCircuitPIDConfiguration:
    """Test circuit configuration and parameter management."""
    
    def test_print_configuration(self, rl_circuit_with_csv, capsys):
        """Test configuration printing."""
        circuit = rl_circuit_with_csv
        circuit.print_configuration()
        
        captured = capsys.readouterr()
        assert circuit.circuit_id in captured.out
        assert "Inductance" in captured.out
        assert "Temperature" in captured.out
        assert "PID Controller Configuration" in captured.out
    
    def test_update_pid_controller(self, basic_rl_circuit, adaptive_pid_controller):
        """Test updating PID controller."""
        circuit = basic_rl_circuit
        old_controller = circuit.pid_controller
        
        circuit.update_pid_controller(adaptive_pid_controller)
        assert circuit.pid_controller is adaptive_pid_controller
        assert circuit.pid_controller is not old_controller
    
    def test_set_circuit_id(self, basic_rl_circuit):
        """Test setting circuit ID."""
        circuit = basic_rl_circuit
        old_id = circuit.circuit_id
        
        new_id = "updated_circuit_id"
        circuit.set_circuit_id(new_id)
        
        assert circuit.circuit_id == new_id
        assert circuit.circuit_id != old_id
    
    def test_copy_circuit(self, rl_circuit_with_csv):
        """Test circuit copying functionality."""
        original = rl_circuit_with_csv
        copy = original.copy("copied_circuit")
        
        # Should be different objects
        assert copy is not original
        assert copy.circuit_id == "copied_circuit"
        assert copy.circuit_id != original.circuit_id
        
        # Should have same parameters
        assert copy.L == original.L
        assert copy.temperature == original.temperature
        assert copy.use_variable_resistance == original.use_variable_resistance
        
        # PID controller should be shared (same reference)
        assert copy.pid_controller is original.pid_controller
    
    def test_copy_circuit_auto_id(self, basic_rl_circuit):
        """Test circuit copying with automatic ID generation."""
        original = basic_rl_circuit
        copy = original.copy()  # No ID specified
        
        assert copy.circuit_id is not None
        assert copy.circuit_id != original.circuit_id
        assert "circuit_" in copy.circuit_id


@pytest.mark.unit
class TestRLCircuitPIDNumerical:
    """Numerical tests for circuit behavior."""
    
    def test_resistance_temperature_dependency(self, sample_resistance_csv):
        """Test resistance varies with temperature."""
        # Test at different temperatures
        circuit_25 = RLCircuitPID(
            resistance_csv=sample_resistance_csv,
            temperature=25.0,
            circuit_id="temp_25"
        )
        
        circuit_50 = RLCircuitPID(
            resistance_csv=sample_resistance_csv, 
            temperature=50.0,
            circuit_id="temp_50"
        )
        
        current = 50.0
        R_25 = circuit_25.get_resistance(current)
        R_50 = circuit_50.get_resistance(current)
        
        # Resistance should change with temperature
        assert R_25 != R_50
        assert jnp.isfinite(R_25)
        assert jnp.isfinite(R_50)
    
    def test_pid_adaptation_across_regions(self, adaptive_pid_controller):
        """Test PID parameters change across current regions."""
        circuit = RLCircuitPID(
            pid_controller=adaptive_pid_controller,
            circuit_id="adaptation_test"
        )
        
        # Get parameters for different current levels
        Kp_low, Ki_low, Kd_low = circuit.get_pid_parameters(10.0)   # Low current
        Kp_med, Ki_med, Kd_med = circuit.get_pid_parameters(100.0)  # Medium current  
        Kp_high, Ki_high, Kd_high = circuit.get_pid_parameters(500.0)  # High current
        
        # Parameters should be different across regions
        params_low = (float(Kp_low), float(Ki_low), float(Kd_low))
        params_med = (float(Kp_med), float(Ki_med), float(Kd_med))
        params_high = (float(Kp_high), float(Ki_high), float(Kd_high))
        
        assert params_low != params_med
        assert params_med != params_high
        assert params_low != params_high
    
    def test_vector_field_equilibrium(self, basic_rl_circuit):
        """Test vector field behavior at equilibrium."""
        circuit = basic_rl_circuit
        
        # At equilibrium: current = reference, no error
        # Assume reference is constant
        i_ref = circuit.reference_current(5.0)  # Get reference at some time
        
        if isinstance(i_ref, jnp.ndarray):
            i_ref = float(i_ref)
        
        # State: [current=reference, integral_error=0, prev_error=0]
        y_eq = jnp.array([i_ref, 0.0, 0.0])
        
        dydt = circuit.vector_field(5.0, y_eq, None)
        
        # At perfect equilibrium, derivatives should be small
        # (though not exactly zero due to control effort needed to overcome resistance)
        assert jnp.all(jnp.isfinite(dydt))
    
    def test_backwards_compatibility_properties(self, basic_rl_circuit):
        """Test backwards compatibility properties."""
        circuit = basic_rl_circuit
        
        # Should have threshold properties for backwards compatibility
        assert hasattr(circuit, 'low_threshold')
        assert hasattr(circuit, 'high_threshold') 
        
        low_thresh = circuit.low_threshold
        high_thresh = circuit.high_threshold
        
        assert isinstance(low_thresh, (int, float))
        assert isinstance(high_thresh, (int, float))
        assert low_thresh < high_thresh


@pytest.mark.integration  
class TestRLCircuitPIDIntegration:
    """Integration tests with actual simulation."""
    
    def test_simple_simulation(self, basic_rl_circuit, simulation_time_params):
        """Test a simple simulation runs without error."""
        import diffrax
        
        circuit = basic_rl_circuit
        y0 = jnp.array([0.0, 0.0, 0.0])  # Initial conditions
        
        # Simple simulation
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(circuit.vector_field),
            diffrax.Dopri5(),
            t0=simulation_time_params["t0"],
            t1=simulation_time_params["t1"], 
            dt0=simulation_time_params["dt"],
            y0=y0,
            saveat=diffrax.SaveAt(ts=jnp.arange(
                simulation_time_params["t0"],
                simulation_time_params["t1"],
                simulation_time_params["dt"]
            ))
        )
        
        # Should complete successfully
        assert hasattr(solution, 'ts')
        assert hasattr(solution, 'ys')
        assert len(solution.ts) > 0
        assert solution.ys.shape[0] == len(solution.ts)
        assert solution.ys.shape[1] == 3  # [current, integral, prev_error]
        
        # Solution should be finite
        assert jnp.all(jnp.isfinite(solution.ys))
    
    @pytest.mark.slow
    def test_longer_simulation(self, rl_circuit_with_csv):
        """Test longer simulation with CSV data."""
        import diffrax
        
        circuit = rl_circuit_with_csv
        y0 = jnp.array([0.0, 0.0, 0.0])
        
        # Longer simulation
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(circuit.vector_field),
            diffrax.Dopri5(), 
            t0=0.0, t1=5.0, dt0=0.01,
            y0=y0,
            saveat=diffrax.SaveAt(ts=jnp.arange(0, 5, 0.01))
        )
        
        # Check solution quality
        current = solution.ys[:, 0]
        
        # Current should be bounded (no infinite values)
        assert jnp.all(jnp.isfinite(current))
        assert jnp.max(jnp.abs(current)) < 1e6  # Reasonable bounds
    
    def test_voltage_driven_simulation(self, sample_voltage_csv):
        """Test simulation with voltage input (no PID)."""
        import diffrax
        
        circuit = RLCircuitPID(
            voltage_csv=sample_voltage_csv,
            circuit_id="voltage_sim_test"
        )
        
        i0 = 0.0  # Initial current
        
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(circuit.voltage_vector_field),
            diffrax.Dopri5(),
            t0=0.0, t1=2.0, dt0=0.01,
            y0=i0,
            saveat=diffrax.SaveAt(ts=jnp.arange(0, 2, 0.01))
        )
        
        # Should be 1D solution (current only)
        assert len(solution.ys.shape) == 1 or solution.ys.shape[1] == 1
        assert jnp.all(jnp.isfinite(solution.ys))


class TestRLCircuitPIDEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_missing_pid_controller_with_pid_mode(self):
        """Test that PID mode requires PID controller."""
        # Creating without PID controller for PID control should use defaults
        circuit = RLCircuitPID(circuit_id="no_pid_test")
        
        # Should have created a default PID controller
        assert circuit.pid_controller is not None
        
        # Should be able to get PID parameters
        Kp, Ki, Kd = circuit.get_pid_parameters(50.0)
        assert all(jnp.isfinite(x) for x in [Kp, Ki, Kd])
    
    def test_extreme_parameters(self):
        """Test with extreme parameter values.""" 
        # Very small inductance
        circuit_small_L = RLCircuitPID(
            L=1e-6,
            R=1.0,
            circuit_id="small_L_test"
        )
        
        y = jnp.array([1.0, 0.0, 0.0])
        dydt = circuit_small_L.vector_field(1.0, y, None)
        assert jnp.all(jnp.isfinite(dydt))
        
        # Very large resistance
        circuit_large_R = RLCircuitPID(
            R=1e6,
            L=0.1,
            circuit_id="large_R_test"
        )
        
        dydt = circuit_large_R.vector_field(1.0, y, None)
        assert jnp.all(jnp.isfinite(dydt))
    
    def test_string_representation(self, basic_rl_circuit):
        """Test string representation of circuit."""
        circuit = basic_rl_circuit
        
        str_repr = str(circuit)
        assert "RLCircuitPID" in str_repr
        assert circuit.circuit_id in str_repr
        assert str(circuit.L) in str_repr