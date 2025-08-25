"""
Tests for PID controller functionality.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from magnet_diffrax.pid_controller import (
    PIDParams, 
    RegionConfig,
    PIDController,
    create_default_pid_controller,
    create_adaptive_pid_controller,
    create_custom_pid_controller,
)


class TestPIDParams:
    """Test PIDParams dataclass."""
    
    def test_valid_parameters(self):
        """Test creation with valid parameters."""
        params = PIDParams(Kp=10.0, Ki=5.0, Kd=0.1)
        assert params.Kp == 10.0
        assert params.Ki == 5.0  
        assert params.Kd == 0.1
    
    def test_zero_parameters(self):
        """Test creation with zero parameters (valid)."""
        params = PIDParams(Kp=0.0, Ki=0.0, Kd=0.0)
        assert params.Kp == 0.0
        assert params.Ki == 0.0
        assert params.Kd == 0.0
    
    def test_negative_parameters(self):
        """Test that negative parameters raise ValueError."""
        with pytest.raises(ValueError, match="PID parameters must be non-negative"):
            PIDParams(Kp=-1.0, Ki=5.0, Kd=0.1)
        
        with pytest.raises(ValueError, match="PID parameters must be non-negative"):
            PIDParams(Kp=10.0, Ki=-1.0, Kd=0.1)
            
        with pytest.raises(ValueError, match="PID parameters must be non-negative"):
            PIDParams(Kp=10.0, Ki=5.0, Kd=-0.1)


class TestRegionConfig:
    """Test RegionConfig dataclass."""
    
    def test_valid_config(self):
        """Test creation with valid configuration."""
        params = PIDParams(15.0, 8.0, 0.05)
        config = RegionConfig(params, threshold=100.0)
        
        assert config.params.Kp == 15.0
        assert config.threshold == 100.0
    
    def test_no_threshold(self):
        """Test creation without threshold (highest region)."""
        params = PIDParams(25.0, 12.0, 0.02)
        config = RegionConfig(params, threshold=None)
        
        assert config.params.Kp == 25.0
        assert config.threshold is None
    
    def test_negative_threshold(self):
        """Test that negative threshold raises ValueError."""
        params = PIDParams(10.0, 5.0, 0.1)
        with pytest.raises(ValueError, match="Threshold must be non-negative"):
            RegionConfig(params, threshold=-10.0)


class TestPIDController:
    """Test PIDController class."""
    
    def test_default_initialization(self):
        """Test initialization with default regions."""
        controller = PIDController()
        
        assert len(controller.regions) == 3
        assert "low" in controller.regions
        assert "medium" in controller.regions  
        assert "high" in controller.regions
        
        # Check that high region has no threshold
        assert controller.regions["high"].threshold is None
    
    def test_custom_initialization(self):
        """Test initialization with custom regions."""
        regions = {
            "startup": RegionConfig(PIDParams(50.0, 30.0, 0.2), threshold=10.0),
            "normal": RegionConfig(PIDParams(20.0, 15.0, 0.1), threshold=100.0),
            "overload": RegionConfig(PIDParams(5.0, 2.0, 0.01), threshold=None)
        }
        
        controller = PIDController(regions)
        assert len(controller.regions) == 3
        assert "startup" in controller.regions
        assert controller.regions["startup"].params.Kp == 50.0
    
    def test_get_pid_parameters_low_current(self):
        """Test PID parameter retrieval for low current."""
        controller = create_default_pid_controller()
        
        # Test low current (below first threshold)
        Kp, Ki, Kd = controller.get_pid_parameters(30.0)
        
        # Should return low region parameters
        expected = controller.regions["low"].params
        assert float(Kp) == expected.Kp
        assert float(Ki) == expected.Ki
        assert float(Kd) == expected.Kd
    
    def test_get_pid_parameters_medium_current(self):
        """Test PID parameter retrieval for medium current."""  
        controller = create_default_pid_controller()
        
        # Test medium current (between thresholds)
        Kp, Ki, Kd = controller.get_pid_parameters(500.0)
        
        # Should return medium region parameters
        expected = controller.regions["medium"].params
        assert float(Kp) == expected.Kp
        assert float(Ki) == expected.Ki
        assert float(Kd) == expected.Kd
    
    def test_get_pid_parameters_high_current(self):
        """Test PID parameter retrieval for high current."""
        controller = create_default_pid_controller()
        
        # Test high current (above all thresholds)
        Kp, Ki, Kd = controller.get_pid_parameters(1000.0)
        
        # Should return high region parameters
        expected = controller.regions["high"].params
        assert float(Kp) == expected.Kp
        assert float(Ki) == expected.Ki
        assert float(Kd) == expected.Kd
    
    def test_get_pid_parameters_negative_current(self):
        """Test PID parameters with negative current (should use absolute value)."""
        controller = create_default_pid_controller()
        
        # Test negative current
        Kp_pos, Ki_pos, Kd_pos = controller.get_pid_parameters(500.0)
        Kp_neg, Ki_neg, Kd_neg = controller.get_pid_parameters(-500.0)
        
        # Should be identical (uses absolute value)
        assert float(Kp_pos) == float(Kp_neg)
        assert float(Ki_pos) == float(Ki_neg)
        assert float(Kd_pos) == float(Kd_neg)
    
    def test_get_current_region_name(self):
        """Test getting current region name."""
        controller = create_default_pid_controller()
        
        # Test different current levels
        assert controller.get_current_region_name(30.0) == "low"
        assert controller.get_current_region_name(500.0) == "medium" 
        assert controller.get_current_region_name(1000.0) == "high"
        
        # Test negative current
        assert controller.get_current_region_name(-500.0) == "medium"
    
    def test_jax_compatibility(self):
        """Test that PID parameter retrieval is JAX-compatible."""
        controller = create_default_pid_controller()
        
        # Test with JAX array input
        i_ref_array = jnp.array([30.0, 500.0, 1000.0])
        
        # Should work with vectorized inputs
        for i_ref in i_ref_array:
            Kp, Ki, Kd = controller.get_pid_parameters(i_ref)
            assert isinstance(Kp, jnp.ndarray) or isinstance(Kp, float)
    
    def test_add_region(self):
        """Test adding a new region."""
        controller = PIDController()
        
        # Add new region
        new_config = RegionConfig(PIDParams(30.0, 20.0, 0.15), threshold=50.0)
        
        # This should fail because it breaks the ordering
        with pytest.raises(ValueError):
            controller.add_region("very_low", new_config)
    
    def test_update_region(self):
        """Test updating an existing region."""
        controller = PIDController()
        
        # Update low region
        new_config = RegionConfig(PIDParams(12.0, 7.0, 0.12), threshold=60.0)
        controller.update_region("low", new_config) 
        
        # Check that parameters were updated
        Kp, Ki, Kd = controller.get_pid_parameters(30.0)
        assert float(Kp) == 12.0
        assert float(Ki) == 7.0
        assert float(Kd) == 0.12
    
    def test_remove_region(self):
        """Test removing a region."""
        controller = PIDController()
        
        # Cannot remove when only 3 regions
        with pytest.raises(ValueError, match="Cannot remove the last region"):
            controller.remove_region("medium")
    
    def test_invalid_region_configuration(self):
        """Test invalid region configurations."""
        # No regions
        with pytest.raises(ValueError, match="At least one region must be defined"):
            PIDController({})
        
        # Multiple regions without threshold
        regions = {
            "low": RegionConfig(PIDParams(10.0, 5.0, 0.1), threshold=None),
            "high": RegionConfig(PIDParams(20.0, 10.0, 0.05), threshold=None)
        }
        with pytest.raises(ValueError, match="Exactly one region must have no threshold"):
            PIDController(regions)
        
        # No region without threshold
        regions = {
            "low": RegionConfig(PIDParams(10.0, 5.0, 0.1), threshold=60.0),
            "high": RegionConfig(PIDParams(20.0, 10.0, 0.05), threshold=800.0)
        }
        with pytest.raises(ValueError, match="Exactly one region must have no threshold"):
            PIDController(regions)
    
    def test_get_thresholds(self):
        """Test getting all thresholds."""
        controller = create_default_pid_controller()
        thresholds = controller.get_thresholds()
        
        assert "low" in thresholds
        assert "medium" in thresholds
        assert "high" in thresholds
        assert thresholds["high"] is None  # Highest region
        assert isinstance(thresholds["low"], float)
        assert isinstance(thresholds["medium"], float)
    
    def test_print_summary(self, capsys):
        """Test printing controller summary."""
        controller = create_default_pid_controller()
        controller.print_summary()
        
        captured = capsys.readouterr()
        assert "PID Controller Configuration" in captured.out
        assert "low" in captured.out
        assert "medium" in captured.out
        assert "high" in captured.out


class TestFactoryFunctions:
    """Test factory functions for creating PID controllers."""
    
    def test_create_default_pid_controller(self):
        """Test default PID controller creation."""
        controller = create_default_pid_controller()
        
        assert isinstance(controller, PIDController)
        assert len(controller.regions) == 3
        assert "low" in controller.regions
        assert "medium" in controller.regions
        assert "high" in controller.regions
    
    def test_create_adaptive_pid_controller(self):
        """Test adaptive PID controller creation with custom parameters."""
        controller = create_adaptive_pid_controller(
            Kp_low=15.0, Ki_low=10.0, Kd_low=0.08,
            Kp_medium=18.0, Ki_medium=12.0, Kd_medium=0.06,
            Kp_high=22.0, Ki_high=15.0, Kd_high=0.04,
            low_threshold=60.0, high_threshold=200.0
        )
        
        assert isinstance(controller, PIDController)
        
        # Test parameters at different current levels
        Kp_low, Ki_low, Kd_low = controller.get_pid_parameters(30.0)
        assert float(Kp_low) == 15.0
        assert float(Ki_low) == 10.0
        assert float(Kd_low) == 0.08
        
        Kp_high, Ki_high, Kd_high = controller.get_pid_parameters(500.0)
        assert float(Kp_high) == 22.0  # Should be in high region
    
    def test_create_custom_pid_controller(self):
        """Test custom PID controller creation."""
        configs = {
            "startup": ((50.0, 30.0, 0.2), 10.0),
            "normal": ((20.0, 15.0, 0.1), 100.0),
            "overload": ((5.0, 2.0, 0.01), None)
        }
        
        controller = create_custom_pid_controller(configs)
        
        assert isinstance(controller, PIDController)
        assert len(controller.regions) == 3
        assert "startup" in controller.regions
        assert "normal" in controller.regions
        assert "overload" in controller.regions
        
        # Test startup region parameters
        Kp, Ki, Kd = controller.get_pid_parameters(5.0)
        assert float(Kp) == 50.0
        assert float(Ki) == 30.0
        assert float(Kd) == 0.2


@pytest.mark.unit
class TestPIDControllerNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_zero_current(self):
        """Test behavior with zero current."""
        controller = create_default_pid_controller()
        Kp, Ki, Kd = controller.get_pid_parameters(0.0)
        
        # Should return low region parameters
        assert float(Kp) > 0
        assert float(Ki) >= 0  
        assert float(Kd) >= 0
    
    def test_very_large_current(self):
        """Test behavior with very large current."""
        controller = create_default_pid_controller()
        Kp, Ki, Kd = controller.get_pid_parameters(1e6)
        
        # Should return high region parameters
        expected = controller.regions["high"].params
        assert float(Kp) == expected.Kp
    
    def test_boundary_currents(self):
        """Test behavior exactly at threshold boundaries."""
        controller = create_adaptive_pid_controller(
            low_threshold=100.0, high_threshold=200.0
        )
        
        # Test exactly at thresholds
        region_99 = controller.get_current_region_name(99.9)
        region_100 = controller.get_current_region_name(100.0)
        region_101 = controller.get_current_region_name(100.1)
        
        assert region_99 == "low"
        assert region_100 == "medium"  # At threshold, should be next region
        assert region_101 == "medium"
    
    def test_jax_jit_compilation(self):
        """Test that PID parameter retrieval can be JIT compiled."""
        import jax
        
        controller = create_default_pid_controller()
        
        @jax.jit
        def get_params_jit(i_ref):
            return controller.get_pid_parameters(i_ref)
        
        # Should compile and execute without error
        Kp, Ki, Kd = get_params_jit(100.0)
        assert isinstance(Kp, jnp.ndarray) or isinstance(Kp, float)
        
    def test_vectorized_operation(self):
        """Test vectorized PID parameter retrieval."""
        controller = create_default_pid_controller()
        
        currents = jnp.array([10.0, 100.0, 1000.0])
        
        # Should work with array inputs
        for curr in currents:
            Kp, Ki, Kd = controller.get_pid_parameters(curr)
            assert jnp.isfinite(Kp)
            assert jnp.isfinite(Ki)
            assert jnp.isfinite(Kd)
