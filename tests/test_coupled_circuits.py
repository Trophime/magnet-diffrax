"""
Tests for CoupledRLCircuitsPID class functionality.
"""

import pytest
import numpy as np
import jax.numpy as jnp

from magnet_diffrax.rlcircuitpid import RLCircuitPID
from magnet_diffrax.coupled_circuits import (
    CoupledRLCircuitsPID,
    create_example_coupled_circuits,
    create_custom_coupled_system,
)


class TestCoupledRLCircuitsPIDBasic:
    """Basic tests for coupled circuits system."""

    def test_initialization_with_circuits_list(self, multiple_circuits):
        """Test initialization with list of circuits."""
        coupling_strength = 0.05
        coupled = CoupledRLCircuitsPID(
            multiple_circuits, coupling_strength=coupling_strength
        )

        assert coupled.n_circuits == len(multiple_circuits)
        assert len(coupled.circuit_ids) == len(multiple_circuits)
        assert coupled.circuits == multiple_circuits

        # Check coupling matrix
        assert coupled.M.shape == (len(multiple_circuits), len(multiple_circuits))

        # Diagonal should be zero
        assert jnp.allclose(jnp.diag(coupled.M), 0.0)

        # Off-diagonal should be coupling_strength
        expected_coupling = jnp.full_like(coupled.M, coupling_strength)
        expected_coupling = expected_coupling.at[
            jnp.diag_indices(len(multiple_circuits))
        ].set(0.0)
        assert jnp.allclose(coupled.M, expected_coupling)

    def test_initialization_with_mutual_inductances(self, multiple_circuits):
        """Test initialization with explicit mutual inductance matrix."""
        n_circuits = len(multiple_circuits)
        M_custom = np.random.rand(n_circuits, n_circuits) * 0.1
        M_custom = (M_custom + M_custom.T) / 2  # Make symmetric
        np.fill_diagonal(M_custom, 0.0)  # Zero diagonal

        coupled = CoupledRLCircuitsPID(multiple_circuits, mutual_inductances=M_custom)

        assert jnp.allclose(coupled.M, M_custom)

    def test_minimum_circuits_requirement(self):
        """Test that at least 2 circuits are required."""
        single_circuit = [RLCircuitPID(circuit_id="single")]

        with pytest.raises(ValueError, match="Need at least 2 circuits"):
            CoupledRLCircuitsPID(single_circuit)

    def test_circuit_id_validation(self, basic_pid_controller):
        """Test circuit ID validation."""
        # Circuits without IDs should raise error
        circuit_no_id = RLCircuitPID(pid_controller=basic_pid_controller)
        circuit_with_id = RLCircuitPID(
            circuit_id="test", pid_controller=basic_pid_controller
        )

        with pytest.raises(ValueError, match="must have a circuit_id"):
            CoupledRLCircuitsPID([circuit_no_id, circuit_with_id])

        # Duplicate IDs should raise error
        circuit1 = RLCircuitPID(
            circuit_id="duplicate", pid_controller=basic_pid_controller
        )
        circuit2 = RLCircuitPID(
            circuit_id="duplicate", pid_controller=basic_pid_controller
        )

        with pytest.raises(ValueError, match="unique circuit_id"):
            CoupledRLCircuitsPID([circuit1, circuit2])

    def test_mutual_inductance_matrix_validation(self, multiple_circuits):
        """Test mutual inductance matrix validation."""
        n_circuits = len(multiple_circuits)

        # Wrong size matrix
        wrong_size_M = np.ones((n_circuits + 1, n_circuits))
        with pytest.raises(ValueError, match="must be"):
            CoupledRLCircuitsPID(multiple_circuits, mutual_inductances=wrong_size_M)

        # Non-symmetric matrix (should warn, not error)
        non_symmetric_M = np.random.rand(n_circuits, n_circuits) * 0.1
        np.fill_diagonal(non_symmetric_M, 0.0)

        # Should create but warn about non-symmetry
        coupled = CoupledRLCircuitsPID(
            multiple_circuits, mutual_inductances=non_symmetric_M
        )
        assert coupled.n_circuits == n_circuits


class TestCoupledRLCircuitsPIDAccess:
    """Test circuit access methods."""

    def test_circuit_access_by_index(self, coupled_rl_system):
        """Test accessing circuits by index."""
        coupled = coupled_rl_system

        # Test valid indices
        for i in range(coupled.n_circuits):
            circuit = coupled.get_circuit_by_index(i)
            assert circuit is coupled.circuits[i]

        # Test invalid indices
        with pytest.raises(ValueError, match="out of range"):
            coupled.get_circuit_by_index(-1)

        with pytest.raises(ValueError, match="out of range"):
            coupled.get_circuit_by_index(coupled.n_circuits)

    def test_circuit_access_by_id(self, coupled_rl_system):
        """Test accessing circuits by ID."""
        coupled = coupled_rl_system

        # Test valid IDs
        for circuit_id in coupled.circuit_ids:
            circuit = coupled.get_circuit_by_id(circuit_id)
            assert circuit.circuit_id == circuit_id

        # Test invalid ID
        with pytest.raises(ValueError, match="not found"):
            coupled.get_circuit_by_id("nonexistent_id")

    def test_circuit_indexing_operators(self, coupled_rl_system):
        """Test indexing operators (__getitem__)."""
        coupled = coupled_rl_system

        # Test integer indexing
        circuit_0 = coupled[0]
        assert circuit_0 is coupled.circuits[0]

        # Test string indexing
        first_id = coupled.circuit_ids[0]
        circuit_by_id = coupled[first_id]
        assert circuit_by_id.circuit_id == first_id

        # Test invalid key type
        with pytest.raises(TypeError, match="Key must be int or str"):
            coupled[3.14]

    def test_get_circuit_index(self, coupled_rl_system):
        """Test getting circuit index by ID."""
        coupled = coupled_rl_system

        for i, circuit_id in enumerate(coupled.circuit_ids):
            index = coupled.get_circuit_index(circuit_id)
            assert index == i

        # Test invalid ID
        with pytest.raises(ValueError, match="not found"):
            coupled.get_circuit_index("invalid_id")

    def test_get_circuit_names(self, coupled_rl_system):
        """Test getting list of circuit names."""
        coupled = coupled_rl_system
        names = coupled.get_circuit_names()

        assert isinstance(names, list)
        assert len(names) == coupled.n_circuits
        assert names == coupled.circuit_ids

        # Should be a copy (not the same object)
        names.append("modified")
        assert len(coupled.get_circuit_names()) == coupled.n_circuits


class TestCoupledRLCircuitsPIDParameters:
    """Test parameter retrieval methods."""

    def test_get_resistance(self, coupled_rl_system):
        """Test resistance retrieval for individual circuits."""
        coupled = coupled_rl_system

        for i in range(coupled.n_circuits):
            resistance = coupled.get_resistance(i, 50.0)

            # Should match individual circuit resistance
            expected_resistance = coupled.circuits[i].get_resistance(50.0)
            assert resistance == expected_resistance
            assert jnp.isfinite(resistance)

    def test_get_reference_current(self, coupled_rl_system):
        """Test reference current retrieval."""
        coupled = coupled_rl_system
        t = 1.0

        for i in range(coupled.n_circuits):
            ref_current = coupled.get_reference_current(i, t)

            # Should match individual circuit reference
            expected_ref = coupled.circuits[i].reference_current(t)
            assert ref_current == expected_ref
            assert jnp.isfinite(ref_current)

    def test_get_pid_parameters(self, coupled_rl_system):
        """Test PID parameter retrieval."""
        coupled = coupled_rl_system
        i_ref = 100.0

        for i in range(coupled.n_circuits):
            Kp, Ki, Kd = coupled.get_pid_parameters(i, i_ref)

            # Should match individual circuit PID parameters
            expected_params = coupled.circuits[i].get_pid_parameters(i_ref)
            assert (Kp, Ki, Kd) == expected_params

            assert all(jnp.isfinite(x) for x in [Kp, Ki, Kd])

    def test_get_current_region(self, coupled_rl_system):
        """Test current region retrieval."""
        coupled = coupled_rl_system
        i_ref = 150.0

        for i in range(coupled.n_circuits):
            region = coupled.get_current_region(i, i_ref)

            # Should match individual circuit region
            expected_region = coupled.circuits[i].get_current_region(i_ref)
            assert region == expected_region
            assert isinstance(region, str)


class TestCoupledRLCircuitsPIDCoupling:
    """Test coupling matrix operations and mutual inductance."""

    def test_get_coupling_strength(self, coupled_rl_system):
        """Test getting coupling strength between circuits."""
        coupled = coupled_rl_system

        for i in range(coupled.n_circuits):
            for j in range(coupled.n_circuits):
                coupling = coupled.get_coupling_strength(i, j)

                if i == j:
                    assert coupling == 0.0  # Self-coupling should be zero
                else:
                    assert coupling > 0.0  # Mutual coupling should be positive
                    assert coupling == float(coupled.M[i, j])

        # Test invalid indices
        with pytest.raises(ValueError, match="out of range"):
            coupled.get_coupling_strength(-1, 0)
        with pytest.raises(ValueError, match="out of range"):
            coupled.get_coupling_strength(0, coupled.n_circuits)

    def test_update_mutual_inductance(self, coupled_rl_system):
        """Test updating individual mutual inductance values."""
        coupled = coupled_rl_system

        # Update coupling between circuits 0 and 1
        new_coupling = 0.08
        coupled.update_mutual_inductance(0, 1, new_coupling)

        # Should be updated symmetrically
        assert coupled.get_coupling_strength(0, 1) == new_coupling
        assert coupled.get_coupling_strength(1, 0) == new_coupling
        assert coupled.M[0, 1] == new_coupling
        assert coupled.M[1, 0] == new_coupling

        # Other couplings should be unchanged
        if coupled.n_circuits > 2:
            original_coupling = coupled.get_coupling_strength(0, 2)
            assert original_coupling != new_coupling

    def test_get_coupling_matrix(self, coupled_rl_system):
        """Test getting the full coupling matrix."""
        coupled = coupled_rl_system

        M = coupled.get_coupling_matrix()

        assert isinstance(M, np.ndarray)
        assert M.shape == (coupled.n_circuits, coupled.n_circuits)
        assert np.allclose(M, coupled.M)

        # Should be symmetric
        assert np.allclose(M, M.T)

        # Diagonal should be zero
        assert np.allclose(np.diag(M), 0.0)

    def test_set_coupling_matrix(self, coupled_rl_system):
        """Test setting the entire coupling matrix."""
        coupled = coupled_rl_system
        n = coupled.n_circuits

        # Create new coupling matrix
        new_M = np.random.rand(n, n) * 0.1
        new_M = (new_M + new_M.T) / 2  # Make symmetric
        np.fill_diagonal(new_M, 0.0)  # Zero diagonal

        coupled.set_coupling_matrix(new_M)

        assert np.allclose(coupled.get_coupling_matrix(), new_M)

        # Test invalid size
        wrong_size_M = np.ones((n + 1, n))
        with pytest.raises(ValueError, match="must be"):
            coupled.set_coupling_matrix(wrong_size_M)


class TestCoupledRLCircuitsPIDVectorField:
    """Test the coupled vector field implementation."""

    def test_vector_field_dimensions(self, coupled_rl_system):
        """Test vector field with correct dimensions."""
        coupled = coupled_rl_system
        n_circuits = coupled.n_circuits

        # State vector: n_circuits * 3 elements
        y = jnp.zeros(n_circuits * 3)
        t = 1.0

        dydt = coupled.vector_field(t, y, None)

        assert isinstance(dydt, jnp.ndarray)
        assert dydt.shape == y.shape
        assert len(dydt) == n_circuits * 3
        assert jnp.all(jnp.isfinite(dydt))

    def test_vector_field_structure(self, coupled_rl_system):
        """Test vector field returns properly structured derivatives."""
        coupled = coupled_rl_system
        n_circuits = coupled.n_circuits

        # Test with non-zero state
        y = jnp.ones(n_circuits * 3)  # [i1, int1, prev1, i2, int2, prev2, ...]

        dydt = coupled.vector_field(1.0, y, None)

        # Reshape to check structure
        dydt_reshaped = dydt.reshape(n_circuits, 3)

        # Each circuit should have 3 derivatives
        for i in range(n_circuits):
            di_dt = dydt_reshaped[i, 0]  # Current derivative
            dint_dt = dydt_reshaped[i, 1]  # Integral error derivative
            dprev_dt = dydt_reshaped[i, 2]  # Previous error derivative

            assert jnp.isfinite(di_dt)
            assert jnp.isfinite(dint_dt)
            assert jnp.isfinite(dprev_dt)

    def test_get_initial_conditions(self, coupled_rl_system):
        """Test initial conditions generation."""
        coupled = coupled_rl_system

        y0 = coupled.get_initial_conditions()

        assert isinstance(y0, jnp.ndarray)
        assert len(y0) == coupled.n_circuits * 3
        assert jnp.all(y0 == 0.0)  # Should be zeros

    def test_vector_field_jax_compatibility(self, coupled_rl_system):
        """Test JAX compatibility of vector field."""
        import jax

        coupled = coupled_rl_system

        @jax.jit
        def jit_vector_field(t, y):
            return coupled.vector_field(t, y, None)

        y = jnp.ones(coupled.n_circuits * 3)
        dydt = jit_vector_field(1.0, y)

        assert isinstance(dydt, jnp.ndarray)
        assert jnp.all(jnp.isfinite(dydt))

    def test_coupling_effects_in_vector_field(self, coupled_rl_system):
        """Test that coupling affects the vector field."""
        coupled = coupled_rl_system

        # Create state with different currents in each circuit
        n_circuits = coupled.n_circuits
        y = jnp.zeros(n_circuits * 3)

        # Set different currents
        for i in range(n_circuits):
            y = y.at[i * 3].set(10.0 * (i + 1))  # Different currents

        dydt_coupled = coupled.vector_field(1.0, y, None)

        # Compare with uncoupled system (zero coupling)
        M_original = coupled.M
        coupled.M = jnp.zeros_like(M_original)  # Remove coupling

        dydt_uncoupled = coupled.vector_field(1.0, y, None)

        # Restore original coupling
        coupled.M = M_original

        # Current derivatives should be different due to coupling
        dydt_coupled_reshaped = dydt_coupled.reshape(n_circuits, 3)
        dydt_uncoupled_reshaped = dydt_uncoupled.reshape(n_circuits, 3)

        current_derivs_coupled = dydt_coupled_reshaped[:, 0]
        current_derivs_uncoupled = dydt_uncoupled_reshaped[:, 0]

        # Should be different (unless coupling is very small)
        if jnp.max(M_original) > 1e-6:
            assert not jnp.allclose(
                current_derivs_coupled, current_derivs_uncoupled, rtol=1e-3
            )


class TestCoupledRLCircuitsPIDDynamicOperations:
    """Test dynamic addition/removal of circuits."""

    def test_add_circuit(self, coupled_rl_system):
        """Test adding a new circuit to the system."""
        coupled = coupled_rl_system
        original_n = coupled.n_circuits

        # Create new circuit
        new_circuit = RLCircuitPID(
            R=2.0, L=0.15, temperature=40.0, circuit_id="new_circuit"
        )

        coupled.add_circuit(new_circuit)

        # Check system updated correctly
        assert coupled.n_circuits == original_n + 1
        assert "new_circuit" in coupled.circuit_ids
        assert coupled.circuits[-1] is new_circuit

        # Check coupling matrix expanded
        assert coupled.M.shape == (original_n + 1, original_n + 1)

        # New circuit should have default coupling
        for i in range(original_n):
            coupling = coupled.get_coupling_strength(i, original_n)
            assert coupling > 0.0  # Should have some coupling

    def test_add_circuit_duplicate_id(self, coupled_rl_system):
        """Test that adding circuit with duplicate ID fails."""
        coupled = coupled_rl_system
        existing_id = coupled.circuit_ids[0]

        duplicate_circuit = RLCircuitPID(circuit_id=existing_id)

        with pytest.raises(ValueError, match="already exists"):
            coupled.add_circuit(duplicate_circuit)

    def test_add_circuit_no_id(self, coupled_rl_system):
        """Test that adding circuit without ID fails."""
        coupled = coupled_rl_system
        no_id_circuit = RLCircuitPID()

        with pytest.raises(ValueError, match="must have a circuit_id"):
            coupled.add_circuit(no_id_circuit)

    def test_remove_circuit(self, coupled_rl_system):
        """Test removing a circuit from the system."""
        coupled = coupled_rl_system
        original_n = coupled.n_circuits

        # Only test if we have more than 2 circuits
        if original_n <= 2:
            pytest.skip("Need more than 2 circuits to test removal")

        remove_id = coupled.circuit_ids[1]  # Remove second circuit
        coupled.remove_circuit(remove_id)

        # Check system updated correctly
        assert coupled.n_circuits == original_n - 1
        assert remove_id not in coupled.circuit_ids

        # Check coupling matrix size reduced
        assert coupled.M.shape == (original_n - 1, original_n - 1)

    def test_remove_circuit_invalid_id(self, coupled_rl_system):
        """Test removing non-existent circuit."""
        coupled = coupled_rl_system

        with pytest.raises(ValueError, match="not found"):
            coupled.remove_circuit("nonexistent_id")

    def test_remove_circuit_minimum_requirement(self):
        """Test that removing circuit fails when only 2 remain."""
        # Create system with exactly 2 circuits
        circuits = [
            RLCircuitPID(circuit_id="circuit_1"),
            RLCircuitPID(circuit_id="circuit_2"),
        ]
        coupled = CoupledRLCircuitsPID(circuits)

        with pytest.raises(ValueError, match="need at least 2 circuits"):
            coupled.remove_circuit("circuit_1")


class TestCoupledRLCircuitsPIDUtilities:
    """Test utility methods and properties."""

    def test_len_operator(self, coupled_rl_system):
        """Test __len__ operator."""
        coupled = coupled_rl_system
        assert len(coupled) == coupled.n_circuits

    def test_string_representation(self, coupled_rl_system):
        """Test __repr__ method."""
        coupled = coupled_rl_system
        repr_str = repr(coupled)

        assert "CoupledRLCircuitsPID" in repr_str
        assert str(coupled.n_circuits) in repr_str
        assert all(circuit_id in repr_str for circuit_id in coupled.circuit_ids)

    def test_print_configuration(self, coupled_rl_system, capsys):
        """Test configuration printing."""
        coupled = coupled_rl_system
        coupled.print_configuration()

        captured = capsys.readouterr()
        assert "Coupled RL Circuits Configuration" in captured.out
        assert str(coupled.n_circuits) in captured.out
        assert "Mutual Inductance Matrix" in captured.out

        # Should print individual circuit configurations
        for circuit_id in coupled.circuit_ids:
            assert circuit_id in captured.out


class TestFactoryFunctions:
    """Test factory functions for creating coupled systems."""

    def test_create_example_coupled_circuits(self):
        """Test example coupled circuits creation."""
        n_circuits = 4
        coupling_strength = 0.06

        coupled = create_example_coupled_circuits(
            n_circuits=n_circuits, coupling_strength=coupling_strength
        )

        assert isinstance(coupled, CoupledRLCircuitsPID)
        assert coupled.n_circuits == n_circuits
        assert len(coupled.circuits) == n_circuits

        # Check coupling matrix
        expected_coupling = jnp.full((n_circuits, n_circuits), coupling_strength)
        expected_coupling = expected_coupling.at[jnp.diag_indices(n_circuits)].set(0.0)
        assert jnp.allclose(coupled.M, expected_coupling)

        # Check that circuits have different parameters
        for i, circuit in enumerate(coupled.circuits):
            assert circuit.circuit_id == f"circuit_{i+1}"
            assert circuit.circuit_id is not None

    def test_create_custom_coupled_system(self):
        """Test custom coupled system creation."""
        circuit_params = [
            {"R": 1.0, "L": 0.08, "circuit_id": "motor1", "temperature": 25.0},
            {"R": 1.2, "L": 0.10, "circuit_id": "motor2", "temperature": 30.0},
            {"R": 1.5, "L": 0.12, "circuit_id": "motor3", "temperature": 35.0},
        ]

        # Test with custom mutual inductances
        M_custom = np.array([[0.0, 0.02, 0.01], [0.02, 0.0, 0.03], [0.01, 0.03, 0.0]])

        coupled = create_custom_coupled_system(
            circuit_params, mutual_inductances=M_custom
        )

        assert isinstance(coupled, CoupledRLCircuitsPID)
        assert coupled.n_circuits == 3
        assert jnp.allclose(coupled.M, M_custom)

        # Check circuit parameters
        for i, params in enumerate(circuit_params):
            circuit = coupled.circuits[i]
            assert circuit.circuit_id == params["circuit_id"]
            assert circuit.R_constant == params["R"]
            assert circuit.L == params["L"]
            assert circuit.temperature == params["temperature"]

    def test_create_custom_coupled_system_default_coupling(self):
        """Test custom system with default coupling."""
        circuit_params = [{"circuit_id": "test1"}, {"circuit_id": "test2"}]
        coupling_strength = 0.04

        coupled = create_custom_coupled_system(
            circuit_params, coupling_strength=coupling_strength
        )

        assert coupled.n_circuits == 2

        # Should have default coupling matrix
        expected_M = np.array([[0.0, coupling_strength], [coupling_strength, 0.0]])
        assert np.allclose(coupled.M, expected_M)


@pytest.mark.integration
class TestCoupledRLCircuitsPIDIntegration:
    """Integration tests with actual simulation."""

    def test_simple_coupled_simulation(self, coupled_rl_system, simulation_time_params):
        """Test that a simple coupled simulation runs."""
        import diffrax

        coupled = coupled_rl_system
        y0 = coupled.get_initial_conditions()

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(coupled.vector_field),
            diffrax.Dopri5(),
            t0=simulation_time_params["t0"],
            t1=simulation_time_params["t1"],
            dt0=simulation_time_params["dt"],
            y0=y0,
            saveat=diffrax.SaveAt(
                ts=jnp.arange(
                    simulation_time_params["t0"],
                    simulation_time_params["t1"],
                    simulation_time_params["dt"],
                )
            ),
        )

        # Check solution format
        assert hasattr(solution, "ts")
        assert hasattr(solution, "ys")
        assert solution.ys.shape[0] == len(solution.ts)
        assert solution.ys.shape[1] == coupled.n_circuits * 3

        # Solution should be finite
        assert jnp.all(jnp.isfinite(solution.ys))

        # Currents should be bounded
        y_reshaped = solution.ys.reshape(-1, coupled.n_circuits, 3)
        currents = y_reshaped[:, :, 0]
        assert jnp.max(jnp.abs(currents)) < 1e6

    @pytest.mark.slow
    def test_longer_coupled_simulation(self):
        """Test longer simulation with more circuits."""
        import diffrax

        # Create system with 5 circuits
        coupled = create_example_coupled_circuits(n_circuits=5, coupling_strength=0.04)
        y0 = coupled.get_initial_conditions()

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(coupled.vector_field),
            diffrax.Dopri5(),
            t0=0.0,
            t1=3.0,
            dt0=0.02,
            y0=y0,
            saveat=diffrax.SaveAt(ts=jnp.arange(0, 3, 0.02)),
        )

        # Check that coupling effects are visible
        y_reshaped = solution.ys.reshape(-1, coupled.n_circuits, 3)
        currents = y_reshaped[:, :, 0]  # Extract all currents

        # Different circuits should have different current profiles
        # (unless reference currents are identical)
        current_variations = jnp.var(currents, axis=0)  # Variance for each circuit
        assert jnp.any(current_variations > 1e-6)  # At least some variation

    def test_simulation_stability(self):
        """Test simulation stability with different coupling strengths."""
        import diffrax

        coupling_strengths = [0.01, 0.05, 0.10]

        for coupling in coupling_strengths:
            coupled = create_example_coupled_circuits(
                n_circuits=3, coupling_strength=coupling
            )
            y0 = coupled.get_initial_conditions()

            solution = diffrax.diffeqsolve(
                diffrax.ODETerm(coupled.vector_field),
                diffrax.Dopri5(),
                t0=0.0,
                t1=1.0,
                dt0=0.01,
                y0=y0,
                saveat=diffrax.SaveAt(ts=jnp.arange(0, 1, 0.01)),
            )

            # Solution should remain stable (no exponential growth)
            assert jnp.all(jnp.isfinite(solution.ys))
            assert jnp.max(jnp.abs(solution.ys)) < 1e3  # Reasonable bounds


@pytest.mark.unit
class TestCoupledRLCircuitsPIDNumerical:
    """Numerical accuracy and edge case tests."""

    def test_symmetric_coupling_preservation(self, multiple_circuits):
        """Test that coupling matrix symmetry is preserved."""
        n_circuits = len(multiple_circuits)

        # Create asymmetric matrix
        M_asym = np.random.rand(n_circuits, n_circuits) * 0.1
        np.fill_diagonal(M_asym, 0.0)

        coupled = CoupledRLCircuitsPID(multiple_circuits, mutual_inductances=M_asym)

        # System should handle asymmetric input (possibly with warning)
        M_result = coupled.get_coupling_matrix()

        # Diagonal should still be zero
        assert np.allclose(np.diag(M_result), 0.0)

    def test_zero_coupling_equivalence(self, multiple_circuits):
        """Test that zero coupling behaves like independent circuits."""
        # Create system with zero coupling
        n_circuits = len(multiple_circuits)
        M_zero = np.zeros((n_circuits, n_circuits))

        coupled = CoupledRLCircuitsPID(multiple_circuits, mutual_inductances=M_zero)

        # Vector field should behave like independent circuits
        y = jnp.ones(n_circuits * 3)
        dydt = coupled.vector_field(1.0, y, None)

        # Each circuit's current derivative should only depend on its own state
        dydt_reshaped = dydt.reshape(n_circuits, 3)

        for i in range(n_circuits):
            circuit = multiple_circuits[i]
            y_individual = y[i * 3 : (i + 1) * 3]  # This circuit's state

            # Individual circuit vector field
            dydt_individual = circuit.vector_field(1.0, y_individual, None)

            # Should match coupled system result for this circuit
            assert jnp.allclose(dydt_reshaped[i], dydt_individual, rtol=1e-6)

    def test_large_coupling_stability(self, multiple_circuits):
        """Test behavior with large coupling values."""
        n_circuits = len(multiple_circuits)

        # Very large coupling
        M_large = np.full((n_circuits, n_circuits), 0.5)
        np.fill_diagonal(M_large, 0.0)

        coupled = CoupledRLCircuitsPID(multiple_circuits, mutual_inductances=M_large)

        # Vector field should still be finite
        y = jnp.ones(n_circuits * 3) * 0.1  # Small currents
        dydt = coupled.vector_field(1.0, y, None)

        assert jnp.all(jnp.isfinite(dydt))

    def test_numerical_precision(self, coupled_rl_system):
        """Test numerical precision with small perturbations."""
        coupled = coupled_rl_system

        y_base = jnp.ones(coupled.n_circuits * 3)
        dydt_base = coupled.vector_field(1.0, y_base, None)

        # Small perturbation
        epsilon = 1e-8
        y_pert = y_base + epsilon
        dydt_pert = coupled.vector_field(1.0, y_pert, None)

        # Change should be small and proportional
        diff = dydt_pert - dydt_base
        relative_change = jnp.max(jnp.abs(diff)) / jnp.max(jnp.abs(dydt_base))

        assert relative_change < 1e-6  # Should be very small change
