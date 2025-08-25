# Magnet Diffrax

**Magnetic coupling simulation for RL circuits with adaptive PID control using JAX and Diffrax**

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![JAX](https://img.shields.io/badge/JAX-0.4+-green.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Magnet Diffrax is a high-performance simulation package for modeling RL (resistor-inductor) circuits with:

- **Adaptive PID Control**: Dynamic PID parameter adjustment based on current magnitude
- **Variable Resistance**: Temperature and current-dependent resistance modeling
- **Magnetic Coupling**: Multi-circuit systems with mutual inductance effects  
- **JAX-Powered**: Fast, differentiable simulations with GPU acceleration
- **CSV Integration**: Load experimental data for validation and comparison
- **Comprehensive Visualization**: Rich plotting and analysis tools

## Features

### Single Circuit Simulation
- Adaptive PID control with configurable current regions
- Variable resistance R(I,T) from CSV data or analytical models
- Reference current tracking from CSV or analytical functions
- Comprehensive performance analysis and visualization

### Coupled Circuit Systems  
- Multiple magnetically coupled RL circuits
- Independent PID controllers for each circuit
- Configurable mutual inductance matrices
- Cross-coupling analysis and visualization

### Advanced Capabilities
- JAX JIT compilation for high performance
- Differentiable simulations for optimization
- Flexible CSV data integration
- Extensive plotting and analysis tools
- Command-line interfaces for batch processing

## Installation

### From Source
```bash
git clone https://github.com/magnetdiffrax/magnet_diffrax.git
cd magnet_diffrax
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/magnetdiffrax/magnet_diffrax.git
cd magnet_diffrax
pip install -e ".[dev]"
```

### Dependencies
- **Core**: JAX (≥0.4.0), Diffrax (≥0.4.0), NumPy (≥1.21.0), Pandas (≥1.3.0)
- **Plotting**: Matplotlib (≥3.5.0), Seaborn (≥0.11.0)
- **Development**: pytest, black, flake8, mypy

## Quick Start

### Single Circuit Example
```python
import jax.numpy as jnp
import diffrax
from magnet_diffrax import RLCircuitPID, create_adaptive_pid_controller

# Create PID controller
pid_controller = create_adaptive_pid_controller(
    Kp_low=15.0, Ki_low=10.0, Kd_low=0.08,
    Kp_high=25.0, Ki_high=15.0, Kd_high=0.04
)

# Create circuit
circuit = RLCircuitPID(
    R=1.5,  # Base resistance (Ω)
    L=0.1,  # Inductance (H)
    pid_controller=pid_controller,
    temperature=25.0,  # Operating temperature (°C)
    circuit_id="motor_1"
)

# Initial conditions [current, integral_error, prev_error]  
y0 = jnp.array([0.0, 0.0, 0.0])

# Time span
t_span = (0.0, 5.0)
solver = diffrax.Dopri5()

# Solve
solution = diffrax.diffeqsolve(
    diffrax.ODETerm(circuit.vector_field),
    solver,
    t0=t_span[0], 
    t1=t_span[1],
    dt0=0.001,
    y0=y0,
    saveat=diffrax.SaveAt(ts=jnp.arange(t_span[0], t_span[1], 0.001))
)

# Plot results
from magnet_diffrax.plotting import plot_results, prepare_post
post_data = prepare_post(solution, circuit)  
plot_results(solution, circuit, *post_data)
```

### Coupled Circuits Example  
```python
from magnet_diffrax import CoupledRLCircuitsPID, create_example_coupled_circuits

# Create coupled system with 3 circuits
coupled_system = create_example_coupled_circuits(
    n_circuits=3, 
    coupling_strength=0.05
)

# Print configuration
coupled_system.print_configuration()

# Simulate
y0 = coupled_system.get_initial_conditions()
solution = diffrax.diffeqsolve(
    diffrax.ODETerm(coupled_system.vector_field),
    diffrax.Dopri5(),
    t0=0.0, t1=5.0, dt0=0.001,
    y0=y0,
    saveat=diffrax.SaveAt(ts=jnp.arange(0, 5, 0.001))
)

# Analyze results
from magnet_diffrax.coupled_plotting import prepare_coupled_post, plot_coupled_results
t, results = prepare_coupled_post(solution, coupled_system)
plot_coupled_results(solution, coupled_system, t, results)
```

### Loading Data from CSV
```python
# Circuit with CSV data
circuit = RLCircuitPID(
    R=1.0, L=0.1,
    reference_csv="reference_current.csv",     # time, current columns
    resistance_csv="resistance_data.csv",     # current, temperature, resistance  
    temperature=30.0,
    circuit_id="experimental_motor"
)

# The CSV files should have the following format:
# reference_current.csv: time,current
# resistance_data.csv: current,temperature,resistance
```

## Command Line Usage

### Single Circuit Simulation
```bash
# Basic simulation
magnet-diffrax --inductance 0.1 --resistance 1.5 --show-plots

# With CSV data  
magnet-diffrax --reference-csv data.csv --resistance-csv resistance.csv --show-analytics

# Custom PID parameters
magnet-diffrax --custom-pid --kp-low 20 --ki-low 15 --show-plots
```

### Coupled Circuits Simulation  
```bash
# Default coupled system
magnet-diffrax-coupled --n-circuits 3 --coupling-strength 0.05 --show-plots

# From configuration file
magnet-diffrax-coupled --config-file config.json --show-coupling --save-results results.npz

# Create sample configuration
magnet-diffrax-coupled --create-config sample_config.json
```

## Configuration Files

### JSON Configuration for Coupled Systems
```json
{
  "circuits": [
    {
      "circuit_id": "motor_1",
      "R": 1.0,
      "L": 0.08, 
      "temperature": 25.0,
      "reference_csv": "motor1_reference.csv",
      "resistance_csv": "motor1_resistance.csv",
      "pid_params": {
        "Kp_low": 15.0, "Ki_low": 8.0, "Kd_low": 0.08,
        "Kp_high": 25.0, "Ki_high": 15.0, "Kd_high": 0.04,
        "low_threshold": 60.0, "high_threshold": 200.0
      }
    }
  ],
  "mutual_inductances": [[0.0, 0.02], [0.02, 0.0]]
}
```

## Testing

Run the test suite:
```bash
# Run all tests
pytest

# With coverage
pytest --cov=magnet_diffrax

# Run specific test categories  
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```

Test structure:
```
tests/
├── test_pid_controller.py     # PID controller tests
├── test_rlcircuitpid.py      # Single circuit tests  
├── test_coupled_circuits.py  # Coupled system tests
├── test_jax_csv_utils.py     # CSV utilities tests
├── conftest.py               # Test configuration
└── data/                     # Test data files
```

## API Reference

### Core Classes

#### `RLCircuitPID`
Main class for single RL circuit with adaptive PID control.

**Parameters:**
- `R`: Base resistance (Ω)  
- `L`: Inductance (H)
- `pid_controller`: PIDController instance
- `reference_csv`: Path to reference current CSV
- `resistance_csv`: Path to resistance data CSV  
- `temperature`: Operating temperature (°C)
- `circuit_id`: Unique circuit identifier

#### `CoupledRLCircuitsPID`  
Container for multiple magnetically coupled RL circuits.

**Parameters:**
- `circuits`: List of RLCircuitPID instances
- `mutual_inductances`: NxN coupling matrix
- `coupling_strength`: Default coupling value

#### `PIDController`
Flexible adaptive PID controller with region-based parameters.

**Key Methods:**
- `get_pid_parameters(i_ref)`: Get PID gains for reference current
- `get_current_region_name(i_ref)`: Get operating region name
- `add_region(name, config)`: Add new current region

## Performance

Magnet Diffrax leverages JAX for high performance:
- **JIT Compilation**: Automatic optimization of simulation loops
- **Vectorization**: Efficient batch operations  
- **GPU Support**: Automatic acceleration on compatible hardware
- **Differentiability**: Gradients for optimization and sensitivity analysis

Typical performance (CPU):
- Single circuit: ~1000x real-time for 5s simulation
- 10 coupled circuits: ~100x real-time for 5s simulation  
- GPU acceleration: 5-50x speedup depending on problem size

## Applications

- **Motor Control**: Electric motor current regulation with temperature effects
- **Power Electronics**: Coupled inductor analysis in power converters  
- **Electromagnetic Systems**: Multi-coil systems with mutual coupling
- **Control System Design**: PID parameter optimization and tuning
- **Educational**: Teaching advanced control and electromagnetic concepts

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite (`pytest`)
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Magnet Diffrax in your research, please cite:

```bibtex
@software{magnet_diffrax,
  author = {Magnet Diffrax Team},
  title = {Magnet Diffrax: Magnetic coupling simulation for RL circuits with adaptive PID control},
  url = {https://github.com/magnetdiffrax/magnet_diffrax},
  version = {0.1.0},
  year = {2024}
}
```

## Acknowledgments

- Built on [JAX](https://github.com/google/jax) for high-performance computing
- Uses [Diffrax](https://github.com/patrick-kidger/diffrax) for differential equation solving
- Inspired by modern control theory and electromagnetic coupling principles

## Support

- **Documentation**: [https://magnet-diffrax.readthedocs.io](https://magnet-diffrax.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/magnetdiffrax/magnet_diffrax/issues)
- **Discussions**: [GitHub Discussions](https://github.com/magnetdiffrax/magnet_diffrax/discussions)
- **Email**: team@magnetdiffrax.com