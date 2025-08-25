import jax
import jax.numpy as jnp
from typing import Tuple

# Import the new PID controller and CSV utilities
from .pid_controller import PIDController, create_adaptive_pid_controller
from .jax_csv_utils import create_jax_function_from_csv, create_2d_jax_function_from_csv


class RLCircuitPID:
    """RL Circuit with Adaptive PID Controller and variable resistance from CSV"""

    def __init__(
        self,
        R: float = 1.0,
        L: float = 0.1,
        pid_controller: PIDController = None,
        reference_csv: str = None,
        voltage_csv: str = None,
        resistance_csv: str = None,
        temperature: float = 25.0,
        circuit_id: str = None,  # NEW: Circuit identifier
        # Backward compatibility: individual PID parameters (deprecated)
        **pid_kwargs,
    ):
        """
        Initialize circuit parameters with adaptive PID controller

        Args:
            R: Constant resistance (Ohms) - used if resistance_csv is None
            L: Inductance (Henry)
            pid_controller: PIDController instance for managing adaptive PID gains
            reference_csv: Path to CSV file with reference current data
            resistance_csv: Path to CSV file with resistance data R(I, Tin)
            temperature: Temperature (°C) for resistance calculation
            circuit_id: Unique identifier for this circuit (required for coupled systems)
            **pid_kwargs: Backward compatibility parameters for creating PID controller
        """
        self.L = L
        self.temperature = temperature

        # Set circuit ID
        self.circuit_id = circuit_id

        # Initialize PID controller
        self.pid_controller = None
        if pid_controller is not None:
            self.pid_controller = pid_controller
        else:
            # Create PID controller from kwargs (backward compatibility)
            if voltage_csv is None:
                print(f"create PID from kwargs: {voltage_csv}")
                self.pid_controller = self._create_pid_from_kwargs(**pid_kwargs)

        # Handle resistance
        if resistance_csv:
            self.load_resistance_from_csv(resistance_csv)
        else:
            self.R_constant = R
            self.use_variable_resistance = False
            print(f"Using constant resistance: {R} Ω")

        # Load reference current from CSV if provided
        if reference_csv:
            self.load_reference_from_csv(reference_csv)

        # Load input voltage from CSV if provided
        print(f"voltage_csv: {voltage_csv}")
        if voltage_csv:
            self.load_voltage_from_csv(voltage_csv)

    def _create_pid_from_kwargs(self, **kwargs) -> PIDController:
        """
        Create PID controller from individual parameters for backward compatibility
        """
        # Extract PID parameters with defaults
        pid_params = {
            "Kp_low": kwargs.get("Kp_low", 10.0),
            "Ki_low": kwargs.get("Ki_low", 5.0),
            "Kd_low": kwargs.get("Kd_low", 0.1),
            "Kp_medium": kwargs.get("Kp_medium", 15.0),
            "Ki_medium": kwargs.get("Ki_medium", 8.0),
            "Kd_medium": kwargs.get("Kd_medium", 0.05),
            "Kp_high": kwargs.get("Kp_high", 25.0),
            "Ki_high": kwargs.get("Ki_high", 12.0),
            "Kd_high": kwargs.get("Kd_high", 0.02),
            "low_threshold": kwargs.get("low_current_threshold", 60.0),
            "high_threshold": kwargs.get("high_current_threshold", 800.0),
        }

        return create_adaptive_pid_controller(**pid_params)

    def load_resistance_from_csv(self, csv_file: str):
        """Load variable resistance from CSV file"""
        try:
            self.resistance_func, self.current_range, self.temp_range, self.R_grid = (
                create_2d_jax_function_from_csv(
                    csv_file, "current", "temperature", "resistance", method="linear"
                )
            )
            self.use_variable_resistance = True
            print(f"Loaded variable resistance from {csv_file}")
            print(
                f"Current range: {float(self.current_range.min()):.3f} to {float(self.current_range.max()):.3f} A"
            )
            print(
                f"Temperature range: {float(self.temp_range.min()):.1f} to {float(self.temp_range.max()):.1f} °C"
            )
        except Exception as e:
            print(f"Error loading resistance CSV file: {e}")
            print("Using default constant resistance instead")
            self.R_constant = 1.0
            self.use_variable_resistance = False

    def load_reference_from_csv(self, csv_file: str):
        """Load reference current from CSV file using JAX"""
        print(f"loading reference from csv {csv_file}")
        try:
            # Use JAX-based CSV loading
            self.reference_func, self.time_data, self.current_data = (
                create_jax_function_from_csv(
                    csv_file, "time", "current", method="linear"
                )
            )

            self.use_csv_data = True
            print(f"Loaded reference current from {csv_file} using JAX interpolation")
            print(
                f"Time range: {float(self.time_data[0]):.3f} to {float(self.time_data[-1]):.3f} seconds"
            )

        except Exception as e:
            print(f"Error loading CSV file: {e}")
            # print("Using default reference current instead")
            # self.use_default_reference()

    def load_voltage_from_csv(self, csv_file: str):
        """Load input voltage from CSV file using JAX"""
        print(f"loading voltage from csv {csv_file}")
        try:
            # Use JAX-based CSV loading
            self.voltage_func, self.time_data, self.voltage_data = (
                create_jax_function_from_csv(
                    csv_file, "time", "voltage", method="linear"
                )
            )

            self.use_csv_data = True
            print(f"Loaded input voltage from {csv_file} using JAX interpolation")
            print(
                f"Time range: {float(self.time_data[0]):.3f} to {float(self.time_data[-1]):.3f} seconds"
            )

        except Exception as e:
            print(f"Error loading CSV file: {e}")
            # print("Using default reference current instead")
            # self.use_default_reference()

    def use_default_reference(self):
        """Use default step reference current"""
        self.use_csv_data = False

        @jax.jit
        def default_reference(t):
            return jnp.where(
                t < 0.5, 0.0, jnp.where(t < 1.5, 2.0, jnp.where(t < 2.5, 1.0, 3.0))
            )

        self.reference_func = default_reference

    def use_default_voltage(self):
        """Use default step input voltage"""
        self.use_csv_data = False

        @jax.jit
        def default_voltage(t):
            return jnp.where(
                t < 0.5, 0.0, jnp.where(t < 1.5, 2.0, jnp.where(t < 2.5, 1.0, 3.0))
            )

        self.voltage_func = default_voltage

    def get_pid_parameters(self, i_ref: float) -> Tuple[float, float, float]:
        """
        Get PID parameters based on reference current magnitude
        Delegates to the PID controller

        Args:
            i_ref: Reference current value

        Returns:
            Tuple of (Kp, Ki, Kd) for the current operating region
        """
        return self.pid_controller.get_pid_parameters(i_ref)

    def get_current_region(self, i_ref: float) -> str:
        """
        Get the current operating region name for logging/plotting
        Delegates to the PID controller
        """
        return self.pid_controller.get_current_region_name(i_ref)

    def get_resistance(self, current: float) -> float:
        """Get resistance value based on current and temperature"""
        if self.use_variable_resistance:
            return self.resistance_func(current, self.temperature)
        else:
            return self.R_constant

    def reference_current(self, t: float) -> float:
        """Get reference current at time t"""
        return self.reference_func(t)

    def input_voltage(self, t: float) -> float:
        """Get input voltage at time t"""
        return self.voltage_func(t)

    def vector_field(self, t: float, y: jnp.ndarray, args) -> jnp.ndarray:
        """
        Define the system dynamics as a vector field with variable resistance and adaptive PID

        State vector y = [i, integral_error, prev_error]
        where:
        - i: current
        - integral_error: integral of error for PID
        - prev_error: previous error for derivative calculation
        """
        i, integral_error, prev_error = y

        # Get current-dependent resistance
        R_current = self.get_resistance(i)

        # Reference current
        i_ref = self.reference_current(t)

        # Get adaptive PID parameters based on reference current
        Kp, Ki, Kd = self.get_pid_parameters(i_ref)

        # Current error
        error = i_ref - i

        # PID control signal (voltage) with adaptive parameters
        derivative_error = error - prev_error

        u = Kp * error + Ki * integral_error + Kd * derivative_error

        # Circuit dynamics: L * di/dt = -R(i,T) * i + u
        di_dt = (-R_current * i + u) / self.L

        # Integral error dynamics
        dintegral_dt = error

        # Store current error for next derivative calculation
        dprev_error_dt = error - prev_error

        return jnp.array([di_dt, dintegral_dt, dprev_error_dt])

    def voltage_vector_field(self, t: float, y, args):
        """
        RL circuit ODE
        """

        i = y  # Current is a scalar now, not an array

        # Get voltage from CSV data
        u = self.input_voltage(t)

        # Get current-dependent resistance
        R_current = self.get_resistance(i)

        # Circuit dynamics: L * di/dt = -R(i,T) * i + u
        di_dt = (-R_current * i + u) / self.L

        return di_dt

    # Convenience properties to access PID controller attributes
    @property
    def low_threshold(self):
        """Get low current threshold (backward compatibility)"""
        thresholds = self.pid_controller.get_thresholds()
        # Return first threshold found (assumes low is first region)
        for threshold in thresholds.values():
            if threshold is not None:
                return threshold
        return 0.0

    @property
    def high_threshold(self):
        """Get high current threshold (backward compatibility)"""
        thresholds = self.pid_controller.get_thresholds()
        # Return last threshold found (assumes high is last region with threshold)
        valid_thresholds = [t for t in thresholds.values() if t is not None]
        return valid_thresholds[-1] if valid_thresholds else float("inf")

    def print_configuration(self):
        """Print circuit and PID controller configuration"""
        print(f"=== {self.circuit_id} Configuration ===")
        print(f"Circuit ID: {self.circuit_id}")
        print(f"Inductance (L): {self.L} H")
        print(f"Temperature: {self.temperature}°C")

        if self.use_variable_resistance:
            print("Using variable resistance from CSV")
            # compute Resistance range for current range at given temperature
            R_min = self.get_resistance(float(self.current_range.min()))
            R_max = self.get_resistance(float(self.current_range.max()))
            print(f"Resistance range: {R_min:.3f} to {R_max:.3f} Ω over current range")
        else:
            print(f"Constant resistance: {self.R_constant} Ω")

        # Print PID controller configuration
        if self.pid_controller:
            self.pid_controller.print_summary()

    def update_pid_controller(self, pid_controller: PIDController):
        """Update the PID controller"""
        self.pid_controller = pid_controller

    def set_circuit_id(self, circuit_id: str):
        """Update the circuit ID"""
        old_id = self.circuit_id
        self.circuit_id = circuit_id
        print(f"Circuit ID changed from '{old_id}' to '{circuit_id}'")

    def copy(self, new_circuit_id: str = None):
        """Create a copy of this circuit with optionally different ID"""
        if new_circuit_id is None:
            import uuid

            new_circuit_id = f"circuit_{str(uuid.uuid4())[:8]}"

        # Create new circuit with same parameters
        new_circuit = RLCircuitPID(
            R=self.R_constant if not self.use_variable_resistance else 1.0,
            L=self.L,
            pid_controller=self.pid_controller,  # Share the same PID controller
            temperature=self.temperature,
            circuit_id=new_circuit_id,
        )

        # Copy resistance and reference functions if they exist
        if self.use_variable_resistance:
            new_circuit.resistance_func = self.resistance_func
            new_circuit.current_range = self.current_range
            new_circuit.temp_range = self.temp_range
            new_circuit.R_grid = self.R_grid
            new_circuit.use_variable_resistance = True

        new_circuit.reference_func = self.reference_func
        new_circuit.voltage_func = self.voltage_func
        new_circuit.use_csv_data = self.use_csv_data

        if hasattr(self, "time_data"):
            new_circuit.time_data = self.time_data
            new_circuit.current_data = self.current_data

        return new_circuit

    def __repr__(self):
        """String representation of the circuit"""
        return f"RLCircuitPID(id='{self.circuit_id}', L={self.L}, R={'variable' if self.use_variable_resistance else self.R_constant}, T={self.temperature}°C)"
