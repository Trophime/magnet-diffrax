import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Dict

from .pid_controller import create_adaptive_pid_controller
from .rlcircuitpid import RLCircuitPID


class CoupledRLCircuitsPID:
    """
    Multiple RL Circuits with Magnetic Coupling and Independent PID Controllers

    Each circuit is electrically independent but magnetically coupled through
    mutual inductances. Uses RLCircuitPID instances directly for maximum flexibility.
    """

    def __init__(
        self,
        circuits: List[RLCircuitPID],  # List of RLCircuitPID instances
        mutual_inductances: np.ndarray = None,
        coupling_strength: float = 0.1,
    ):
        """
        Initialize coupled RL circuits

        Args:
            circuits: List of RLCircuitPID instances
            mutual_inductances: NxN matrix of mutual inductances M[i,j] between circuits i and j
                               If None, creates symmetric coupling matrix
            coupling_strength: Default coupling strength if mutual_inductances not provided
        """
        self.circuits = circuits
        self.n_circuits = len(circuits)

        if self.n_circuits < 2:
            raise ValueError("Need at least 2 circuits for coupling")

        # Validate that all circuits have circuit_id
        self.circuit_ids = []
        for i, circuit in enumerate(self.circuits):
            if not hasattr(circuit, "circuit_id") or circuit.circuit_id is None:
                raise ValueError(f"Circuit {i} must have a circuit_id")
            self.circuit_ids.append(circuit.circuit_id)

        # Check for duplicate circuit IDs
        if len(set(self.circuit_ids)) != len(self.circuit_ids):
            raise ValueError("All circuits must have unique circuit_id values")

        # Initialize mutual inductance matrix
        if mutual_inductances is not None:
            if mutual_inductances.shape != (self.n_circuits, self.n_circuits):
                raise ValueError(
                    f"Mutual inductance matrix must be {self.n_circuits}x{self.n_circuits}"
                )
            self.M = jnp.array(mutual_inductances)
        else:
            # Create symmetric coupling matrix with zero diagonal
            self.M = jnp.full((self.n_circuits, self.n_circuits), coupling_strength)
            self.M = self.M.at[jnp.diag_indices(self.n_circuits)].set(0.0)

        # Validate mutual inductance matrix
        self._validate_coupling_matrix()

        print(f"Initialized {self.n_circuits} coupled RL circuits")
        print(f"Circuit IDs: {self.circuit_ids}")

    def _validate_coupling_matrix(self):
        """Validate the mutual inductance matrix"""
        # Check symmetry
        if not jnp.allclose(self.M, self.M.T):
            print("Warning: Mutual inductance matrix is not symmetric")

        # Check diagonal is zero
        if not jnp.allclose(jnp.diag(self.M), 0.0):
            print("Warning: Mutual inductance matrix has non-zero diagonal elements")

    def get_resistance(self, circuit_idx: int, current: float) -> float:
        """Get resistance for a specific circuit"""
        circuit = self.circuits[circuit_idx]
        return circuit.get_resistance(current)

    def get_reference_current(self, circuit_idx: int, t: float) -> float:
        """Get reference current for a specific circuit"""
        circuit = self.circuits[circuit_idx]
        return circuit.reference_current(t)

    def get_pid_parameters(
        self, circuit_idx: int, i_ref: float
    ) -> Tuple[float, float, float]:
        """Get PID parameters for a specific circuit"""
        circuit = self.circuits[circuit_idx]
        return circuit.get_pid_parameters(i_ref)

    def get_current_region(self, circuit_idx: int, i_ref: float) -> str:
        """Get current region name for a specific circuit"""
        circuit = self.circuits[circuit_idx]
        return circuit.get_current_region(i_ref)

    def vector_field(self, t: float, y: jnp.ndarray, args) -> jnp.ndarray:
        """
        Define the coupled system dynamics

        State vector y has shape (n_circuits * 3,) with structure:
        [i1, integral_error1, prev_error1, i2, integral_error2, prev_error2, ...]

        The coupling appears in the voltage equation for each circuit:
        L_i * di_i/dt = -R_i(i_i,T_i) * i_i + u_i - sum_j(M_ij * dj_j/dt)

        where u_i is the PID control voltage for circuit i
        """
        # Reshape state vector: (n_circuits, 3)
        y_reshaped = y.reshape(self.n_circuits, 3)
        currents = y_reshaped[:, 0]  # [i1, i2, ..., in]
        integral_errors = y_reshaped[:, 1]
        prev_errors = y_reshaped[:, 2]

        # Initialize derivatives
        dy_dt = jnp.zeros_like(y_reshaped)

        # Calculate current derivatives (will be updated with coupling)
        di_dt = jnp.zeros(self.n_circuits)

        # Calculate PID control signals and individual circuit dynamics
        for i in range(self.n_circuits):
            circuit = self.circuits[i]

            # Current and reference for this circuit
            current_i = currents[i]
            i_ref = self.get_reference_current(i, t)

            # Get resistance for this circuit
            R_i = self.get_resistance(i, current_i)

            # Get adaptive PID parameters
            Kp, Ki, Kd = self.get_pid_parameters(i, i_ref)

            # Error calculations
            error = i_ref - current_i
            derivative_error = error - prev_errors[i]

            # PID control voltage
            u_i = Kp * error + Ki * integral_errors[i] + Kd * derivative_error

            # Individual circuit equation (without coupling yet)
            # L_i * di_i/dt = -R_i * i_i + u_i - coupling_term
            L_i = circuit.L
            di_dt = di_dt.at[i].set((-R_i * current_i + u_i) / L_i)

            # Update integral and previous error
            dy_dt = dy_dt.at[i, 1].set(error)  # dintegral_dt
            dy_dt = dy_dt.at[i, 2].set(error - prev_errors[i])  # dprev_error_dt

        # Add magnetic coupling terms
        # For each circuit i: subtract sum_j(M_ij * dj_j/dt) / L_i
        for i in range(self.n_circuits):
            L_i = self.circuits[i].L
            coupling_term = 0.0

            for j in range(self.n_circuits):
                if i != j:  # No self-coupling
                    coupling_term += self.M[i, j] * di_dt[j]

            # Update current derivative with coupling
            dy_dt = dy_dt.at[i, 0].set(di_dt[i] - coupling_term / L_i)

        # Flatten back to original shape
        return dy_dt.flatten()

    def get_initial_conditions(self) -> jnp.ndarray:
        """Get initial conditions for all circuits"""
        # Each circuit: [current, integral_error, prev_error]
        y0 = jnp.zeros(self.n_circuits * 3)
        return y0

    def print_configuration(self):
        """Print configuration of all circuits and coupling"""
        print("\n=== Coupled RL Circuits Configuration ===")
        print(f"Number of circuits: {self.n_circuits}")
        print(f"Circuit IDs: {self.circuit_ids}")

        print("\nMutual Inductance Matrix (H):")
        for i in range(self.n_circuits):
            row_str = "  "
            for j in range(self.n_circuits):
                row_str += f"{float(self.M[i,j]):8.4f} "
            print(row_str)

        print("\nIndividual Circuit Configurations:")
        for i, circuit in enumerate(self.circuits):
            print(f"\n  === {circuit.circuit_id} ===")
            circuit.print_configuration()

    def update_mutual_inductance(self, i: int, j: int, M_ij: float):
        """Update a specific mutual inductance value"""
        if i >= self.n_circuits or j >= self.n_circuits:
            raise ValueError("Circuit indices out of range")

        # Update symmetrically
        self.M = self.M.at[i, j].set(M_ij)
        self.M = self.M.at[j, i].set(M_ij)

    def add_circuit(self, circuit) -> None:
        """
        Add a new circuit to the system (requires rebuilding coupling matrix)

        Args:
            circuit: RLCircuitPID instance to add
        """
        if not hasattr(circuit, "circuit_id") or circuit.circuit_id is None:
            raise ValueError("Circuit must have a circuit_id")

        if circuit.circuit_id in self.circuit_ids:
            raise ValueError(f"Circuit ID '{circuit.circuit_id}' already exists")

        self.circuits.append(circuit)
        self.circuit_ids.append(circuit.circuit_id)
        old_n_circuits = self.n_circuits
        self.n_circuits += 1

        # Rebuild mutual inductance matrix with default coupling to new circuit
        old_M = self.M
        coupling_strength = 0.05  # Default coupling for new circuit

        # Create new matrix
        new_M = jnp.zeros((self.n_circuits, self.n_circuits))

        # Copy old matrix
        new_M = new_M.at[:old_n_circuits, :old_n_circuits].set(old_M)

        # Add coupling to new circuit
        new_M = new_M.at[self.n_circuits - 1, :old_n_circuits].set(coupling_strength)
        new_M = new_M.at[:old_n_circuits, self.n_circuits - 1].set(coupling_strength)

        self.M = new_M
        print(
            f"Added circuit '{circuit.circuit_id}' with default coupling strength {coupling_strength}"
        )

    def remove_circuit(self, circuit_id: str) -> None:
        """
        Remove a circuit from the system

        Args:
            circuit_id: ID of the circuit to remove
        """
        if circuit_id not in self.circuit_ids:
            raise ValueError(f"Circuit ID '{circuit_id}' not found")

        if self.n_circuits <= 2:
            raise ValueError(
                "Cannot remove circuit - need at least 2 circuits for coupling"
            )

        # Find index of circuit to remove
        remove_idx = self.circuit_ids.index(circuit_id)

        # Remove circuit from list
        self.circuits.pop(remove_idx)
        self.circuit_ids.remove(circuit_id)
        self.n_circuits -= 1

        # Rebuild mutual inductance matrix
        indices_to_keep = [i for i in range(self.n_circuits + 1) if i != remove_idx]
        new_M = self.M[jnp.ix_(indices_to_keep, indices_to_keep)]
        self.M = new_M

        print(f"Removed circuit '{circuit_id}'")

    def get_circuit_names(self) -> List[str]:
        """Get list of circuit IDs"""
        return self.circuit_ids.copy()

    def get_coupling_strength(self, i: int, j: int) -> float:
        """Get mutual inductance between circuits i and j"""
        if i >= self.n_circuits or j >= self.n_circuits:
            raise ValueError("Circuit indices out of range")
        return float(self.M[i, j])

    def get_circuit_by_id(self, circuit_id: str):
        """Get a circuit by its ID"""
        for circuit in self.circuits:
            if circuit.circuit_id == circuit_id:
                return circuit
        raise ValueError(f"Circuit ID '{circuit_id}' not found")

    def get_circuit_by_index(self, index: int):
        """Get a circuit by its index"""
        if 0 <= index < self.n_circuits:
            return self.circuits[index]
        raise ValueError(f"Circuit index {index} out of range [0, {self.n_circuits-1}]")

    def get_circuit_index(self, circuit_id: str) -> int:
        """Get the index of a circuit by its ID"""
        try:
            return self.circuit_ids.index(circuit_id)
        except ValueError:
            raise ValueError(f"Circuit ID '{circuit_id}' not found")

    def set_coupling_matrix(self, mutual_inductances: np.ndarray):
        """Set the entire mutual inductance matrix"""
        if mutual_inductances.shape != (self.n_circuits, self.n_circuits):
            raise ValueError(
                f"Mutual inductance matrix must be {self.n_circuits}x{self.n_circuits}"
            )

        self.M = jnp.array(mutual_inductances)
        self._validate_coupling_matrix()
        print("Updated mutual inductance matrix")

    def get_coupling_matrix(self) -> np.ndarray:
        """Get the current mutual inductance matrix"""
        return np.array(self.M)

    def __len__(self) -> int:
        """Return the number of circuits"""
        return self.n_circuits

    def __getitem__(self, key):
        """Allow indexing circuits by index or ID"""
        if isinstance(key, int):
            return self.get_circuit_by_index(key)
        elif isinstance(key, str):
            return self.get_circuit_by_id(key)
        else:
            raise TypeError("Key must be int (index) or str (circuit_id)")

    def __repr__(self) -> str:
        """String representation of the coupled system"""
        return f"CoupledRLCircuitsPID(n_circuits={self.n_circuits}, ids={self.circuit_ids})"


def create_example_coupled_circuits(
    n_circuits: int = 3, coupling_strength: float = 0.05
):
    """
    Create an example system of coupled RL circuits for testing

    Args:
        n_circuits: Number of circuits to create
        coupling_strength: Mutual inductance between circuits

    Returns:
        Configured CoupledRLCircuitsPID instance
    """
    # Import here to avoid circular imports
    from rlcircuitpid import RLCircuitPID

    circuits = []

    for i in range(n_circuits):
        # Create different PID controllers for each circuit
        if i % 2 == 0:
            # More aggressive PID for even circuits
            pid_controller = create_adaptive_pid_controller(
                Kp_low=15.0,
                Ki_low=10.0,
                Kd_low=0.08,
                Kp_medium=20.0,
                Ki_medium=12.0,
                Kd_medium=0.06,
                Kp_high=25.0,
                Ki_high=15.0,
                Kd_high=0.04,
            )
        else:
            # More conservative PID for odd circuits
            pid_controller = create_adaptive_pid_controller(
                Kp_low=10.0,
                Ki_low=6.0,
                Kd_low=0.12,
                Kp_medium=12.0,
                Ki_medium=8.0,
                Kd_medium=0.08,
                Kp_high=15.0,
                Ki_high=10.0,
                Kd_high=0.06,
            )

        # Create RLCircuitPID instance
        circuit = RLCircuitPID(
            R=1.0 + 0.2 * i,  # Different resistances
            L=0.1 + 0.02 * i,  # Different inductances
            pid_controller=pid_controller,
            temperature=25.0 + 5.0 * i,  # Different temperatures
            circuit_id=f"circuit_{i+1}",
        )

        circuits.append(circuit)

    # Create symmetric coupling matrix
    M = np.full((n_circuits, n_circuits), coupling_strength)
    np.fill_diagonal(M, 0.0)

    return CoupledRLCircuitsPID(circuits, M)


def create_custom_coupled_system(
    circuit_params: List[Dict],
    mutual_inductances: np.ndarray = None,
    coupling_strength: float = 0.05,
):
    """
    Create a custom coupled system from parameter dictionaries

    Args:
        circuit_params: List of dictionaries with circuit parameters
        mutual_inductances: Optional coupling matrix
        coupling_strength: Default coupling if no matrix provided

    Returns:
        CoupledRLCircuitsPID instance

    Example:
        params = [
            {'R': 1.0, 'L': 0.1, 'circuit_id': 'motor1', 'temperature': 25},
            {'R': 1.2, 'L': 0.12, 'circuit_id': 'motor2', 'temperature': 30}
        ]
        system = create_custom_coupled_system(params)
    """
    from rlcircuitpid import RLCircuitPID

    circuits = []

    for i, params in enumerate(circuit_params):
        # Set defaults
        circuit_params_with_defaults = {
            "R": 1.0,
            "L": 0.1,
            "temperature": 25.0,
            "circuit_id": f"circuit_{i+1}",
            **params,  # Override with user parameters
        }

        circuit = RLCircuitPID(**circuit_params_with_defaults)
        circuits.append(circuit)

    return CoupledRLCircuitsPID(circuits, mutual_inductances, coupling_strength)


# Example usage and testing
if __name__ == "__main__":
    # Create example coupled system
    coupled_system = create_example_coupled_circuits(
        n_circuits=3, coupling_strength=0.08
    )
    coupled_system.print_configuration()

    # Get initial conditions
    y0 = coupled_system.get_initial_conditions()
    print(f"\nInitial conditions shape: {y0.shape}")
    print(f"Initial conditions: {y0}")

    # Test circuit access methods
    print("\nCircuit access test:")
    print(
        f"Circuit by ID 'circuit_1': {coupled_system.get_circuit_by_id('circuit_1').circuit_id}"
    )
    print(f"Circuit by index 0: {coupled_system.get_circuit_by_index(0).circuit_id}")
    print(f"Circuit by indexing [0]: {coupled_system[0].circuit_id}")
    print(
        f"Circuit by indexing ['circuit_1']: {coupled_system['circuit_1'].circuit_id}"
    )
    print(
        f"Coupling between circuits 0 and 1: {coupled_system.get_coupling_strength(0, 1):.4f} H"
    )
    print(f"Number of circuits: {len(coupled_system)}")

    # Test dynamic operations
    print("\nDynamic operations test:")
    from rlcircuitpid import RLCircuitPID

    new_circuit = RLCircuitPID(R=2.0, L=0.2, circuit_id="test_circuit")
    coupled_system.add_circuit(new_circuit)
    print(f"After adding circuit: {len(coupled_system)} circuits")
    print(f"Circuit IDs: {coupled_system.get_circuit_names()}")

    coupled_system.remove_circuit("test_circuit")
    print(f"After removing circuit: {len(coupled_system)} circuits")
    print(f"Circuit IDs: {coupled_system.get_circuit_names()}")
