"""
Magnet Diffrax - Magnetic coupling simulation for RL circuits with adaptive PID control

This package provides tools for simulating RL circuits with:
- Adaptive PID control based on current magnitude
- Variable resistance dependent on current and temperature
- Magnetic coupling between multiple circuits
- JAX-based high-performance computation
- CSV data integration for experimental validation

Main modules:
- rlcircuitpid: Single RL circuit with adaptive PID control
- coupled_circuits: Multiple magnetically coupled RL circuits  
- pid_controller: Flexible adaptive PID controller implementation
- jax_csv_utils: JAX-compatible CSV data loading utilities
- plotting: Visualization tools for simulation results

Example usage:
    >>> from magnet_diffrax import RLCircuitPID, create_default_pid_controller
    >>> from magnet_diffrax.coupled_circuits import CoupledRLCircuitsPID
    
    # Single circuit simulation
    >>> circuit = RLCircuitPID(R=1.5, L=0.1, temperature=25.0)
    >>> circuit.print_configuration()
    
    # Coupled circuits simulation  
    >>> circuits = [RLCircuitPID(circuit_id=f"motor_{i}") for i in range(3)]
    >>> coupled = CoupledRLCircuitsPID(circuits, coupling_strength=0.05)
"""

__version__ = "0.1.0"
__author__ = "Magnet Diffrax Team"
__email__ = "team@magnetdiffrax.com"

# Core classes and functions
from .rlcircuitpid import RLCircuitPID
from .coupled_circuits import CoupledRLCircuitsPID, create_example_coupled_circuits
from .pid_controller import (
    PIDController,
    PIDParams, 
    RegionConfig,
    create_default_pid_controller,
    create_adaptive_pid_controller,
    create_custom_pid_controller,
)
from .jax_csv_utils import (
    create_jax_function_from_csv,
    create_2d_jax_function_from_csv,
    create_multi_column_jax_function_from_csv,
    create_parametric_jax_function_from_csv,
)

# Plotting utilities (optional import to handle missing matplotlib gracefully)
try:
    from .plotting import prepare_post, plot_results, plot_vresults, analytics
    from .coupled_plotting import (
        prepare_coupled_post,
        plot_coupled_results, 
        plot_region_analysis,
        analyze_coupling_effects,
        save_coupled_results,
        load_coupled_results,
    )
    _PLOTTING_AVAILABLE = True
except ImportError as e:
    _PLOTTING_AVAILABLE = False
    import warnings
    warnings.warn(
        f"Plotting functionality not available: {e}. "
        "Install matplotlib and seaborn to enable plotting.",
        ImportWarning
    )

# Expose main entry points (for CLI compatibility)
try:
    from .main import main as run_single_simulation
    from .coupled_main import main as run_coupled_simulation
    _CLI_AVAILABLE = True
except ImportError:
    _CLI_AVAILABLE = False

# Define what gets imported with "from magnet_diffrax import *"
__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__email__",
    
    # Core classes
    "RLCircuitPID",
    "CoupledRLCircuitsPID",
    "PIDController",
    "PIDParams",
    "RegionConfig",
    
    # Factory functions
    "create_default_pid_controller",
    "create_adaptive_pid_controller", 
    "create_custom_pid_controller",
    "create_example_coupled_circuits",
    
    # CSV utilities
    "create_jax_function_from_csv",
    "create_2d_jax_function_from_csv", 
    "create_multi_column_jax_function_from_csv",
    "create_parametric_jax_function_from_csv",
]

# Add plotting functions if available
if _PLOTTING_AVAILABLE:
    __all__.extend([
        "prepare_post",
        "plot_results",
        "plot_vresults", 
        "analytics",
        "prepare_coupled_post",
        "plot_coupled_results",
        "plot_region_analysis", 
        "analyze_coupling_effects",
        "save_coupled_results",
        "load_coupled_results",
    ])

# Add CLI functions if available  
if _CLI_AVAILABLE:
    __all__.extend([
        "run_single_simulation",
        "run_coupled_simulation",
    ])

def get_version() -> str:
    """Get the package version string."""
    return __version__

def check_dependencies() -> dict:
    """
    Check if all required dependencies are available.
    
    Returns:
        Dictionary with dependency status information
    """
    deps = {
        "jax": False,
        "diffrax": False, 
        "numpy": False,
        "pandas": False,
        "matplotlib": False,
        "seaborn": False,
    }
    
    for dep in deps:
        try:
            __import__(dep)
            deps[dep] = True
        except ImportError:
            deps[dep] = False
    
    return {
        "dependencies": deps,
        "plotting_available": _PLOTTING_AVAILABLE,
        "cli_available": _CLI_AVAILABLE,
        "all_required": all(deps[d] for d in ["jax", "diffrax", "numpy", "pandas"]),
    }

def print_info():
    """Print package information and dependency status."""
    print(f"Magnet Diffrax v{__version__}")
    print(f"Magnetic coupling simulation for RL circuits with adaptive PID control\n")
    
    status = check_dependencies()
    
    print("Dependency Status:")
    for dep, available in status["dependencies"].items():
        status_symbol = "✓" if available else "✗"
        print(f"  {status_symbol} {dep}")
    
    print(f"\nFeature Availability:")
    print(f"  ✓ Core simulation" if status["all_required"] else "  ✗ Core simulation (missing dependencies)")
    print(f"  ✓ Plotting" if status["plotting_available"] else "  ✗ Plotting")
    print(f"  ✓ CLI tools" if status["cli_available"] else "  ✗ CLI tools")

# Package-level configuration
def configure_jax(enable_x64=True, platform=None):
    """
    Configure JAX settings for the package.
    
    Args:
        enable_x64: Whether to enable 64-bit precision
        platform: JAX platform to use ('cpu', 'gpu', 'tpu')
    """
    try:
        import jax
        
        if enable_x64:
            from jax import config
            config.update("jax_enable_x64", True)
            
        if platform:
            jax.config.update('jax_platform_name', platform)
            
        print(f"JAX configured: precision={'64-bit' if enable_x64 else '32-bit'}")
        if platform:
            print(f"JAX platform: {platform}")
            
    except ImportError:
        warnings.warn("JAX not available - cannot configure JAX settings", ImportWarning)
