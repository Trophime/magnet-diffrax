import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import seaborn as sns

from .coupled_circuits import CoupledRLCircuitsPID


def prepare_coupled_post(sol, coupled_system: CoupledRLCircuitsPID):
    """
    Post-process results for coupled RL circuits

    Returns data structures for plotting and analysis
    """
    t = sol.ts
    n_circuits = coupled_system.n_circuits
    circuit_ids = coupled_system.circuit_ids

    # Reshape solution: (time_steps, n_circuits, 3)
    y_reshaped = sol.ys.reshape(len(t), n_circuits, 3)

    results = {}

    for i, circuit_id in enumerate(circuit_ids):
        circuit = coupled_system.circuits[i]  # Access circuit directly from list

        # Extract state variables for this circuit
        current = y_reshaped[:, i, 0]
        integral_error = y_reshaped[:, i, 1]

        # Calculate reference current
        i_ref = jnp.array(
            [coupled_system.get_reference_current(i, t_val) for t_val in t]
        )

        # Calculate adaptive PID gains over time
        Kp_over_time = []
        Ki_over_time = []
        Kd_over_time = []
        current_regions = []

        for j, i_ref_val in enumerate(i_ref):
            i_ref_float = float(i_ref_val)
            Kp, Ki, Kd = coupled_system.get_pid_parameters(i, i_ref_float)
            Kp_over_time.append(float(Kp))
            Ki_over_time.append(float(Ki))
            Kd_over_time.append(float(Kd))

            region_name = coupled_system.get_current_region(i, i_ref_float)
            current_regions.append(region_name)

        # Calculate variable resistance over time
        resistance_over_time = jnp.array(
            [coupled_system.get_resistance(i, float(curr)) for curr in current]
        )

        # Calculate control signals and errors
        error = i_ref - current
        derivative_error = jnp.gradient(error, t[1] - t[0])

        Kp_array = jnp.array(Kp_over_time)
        Ki_array = jnp.array(Ki_over_time)
        Kd_array = jnp.array(Kd_over_time)

        voltage = (
            Kp_array * error + Ki_array * integral_error + Kd_array * derivative_error
        )

        # Calculate power dissipation
        power = resistance_over_time * current**2

        # Store results for this circuit
        results[circuit_id] = {
            "current": current,
            "reference": i_ref,
            "error": error,
            "voltage": voltage,
            "power": power,
            "resistance": resistance_over_time,
            "Kp": Kp_array,
            "Ki": Ki_array,
            "Kd": Kd_array,
            "regions": current_regions,
            "integral_error": integral_error,
        }

    return t, results


def plot_coupled_results(
    sol,
    coupled_system: CoupledRLCircuitsPID,
    t: np.ndarray,
    results: Dict,
    show_coupling_analysis: bool = True,
    show_individual_details: bool = True,
):
    """
    Plot comprehensive results for coupled RL circuits
    """
    n_circuits = coupled_system.n_circuits
    circuit_ids = coupled_system.circuit_ids

    # Set up color palette
    colors = plt.cm.Set1(np.linspace(0, 1, n_circuits))

    # Create figure with subplots
    if show_individual_details:
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

    # 1. Current tracking for all circuits
    ax = axes[0]
    for i, circuit_id in enumerate(circuit_ids):
        data = results[circuit_id]
        ax.plot(
            t,
            data["current"],
            color=colors[i],
            linewidth=2,
            label=f"{circuit_id} - Actual",
            linestyle="-",
        )
        ax.plot(
            t,
            data["reference"],
            color=colors[i],
            linewidth=2,
            label=f"{circuit_id} - Reference",
            linestyle="--",
            alpha=0.7,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (A)")
    ax.set_title("Current Tracking - All Circuits")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # 2. Tracking errors
    ax = axes[1]
    for i, circuit_id in enumerate(circuit_ids):
        data = results[circuit_id]
        ax.plot(t, data["error"], color=colors[i], linewidth=2, label=circuit_id)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (A)")
    ax.set_title("Tracking Errors")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. Control voltages
    ax = axes[2]
    for i, circuit_id in enumerate(circuit_ids):
        data = results[circuit_id]
        ax.plot(t, data["voltage"], color=colors[i], linewidth=2, label=circuit_id)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Control Voltages")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Power dissipation
    ax = axes[3]
    for i, circuit_id in enumerate(circuit_ids):
        data = results[circuit_id]
        ax.plot(t, data["power"], color=colors[i], linewidth=2, label=circuit_id)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.set_title("Power Dissipation")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if show_individual_details and len(axes) > 4:
        # 5. Adaptive PID gains (Kp)
        ax = axes[4]
        for i, circuit_id in enumerate(circuit_ids):
            data = results[circuit_id]
            ax.plot(
                t, data["Kp"], color=colors[i], linewidth=2, label=f"{circuit_id} - Kp"
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Kp")
        ax.set_title("Proportional Gains")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 6. Variable resistance
        ax = axes[5]
        for i, circuit_id in enumerate(circuit_ids):
            data = results[circuit_id]
            circuit = coupled_system.circuits[circuit_id]
            label = f'{circuit_id} (T={circuit["temperature"]}°C)'
            ax.plot(t, data["resistance"], color=colors[i], linewidth=2, label=label)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Resistance (Ω)")
        ax.set_title("Circuit Resistances")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 7. Current derivatives (coupling effect)
        if len(axes) > 6:
            ax = axes[6]
            y_reshaped = sol.ys.reshape(len(t), n_circuits, 3)

            for i, circuit_id in enumerate(circuit_ids):
                current = y_reshaped[:, i, 0]
                di_dt = jnp.gradient(current, t[1] - t[0])
                ax.plot(t, di_dt, color=colors[i], linewidth=2, label=circuit_id)

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("dI/dt (A/s)")
            ax.set_title("Current Derivatives (shows coupling effects)")
            ax.grid(True, alpha=0.3)
            ax.legend()

    plt.tight_layout()
    plt.show()

    # Show coupling analysis if requested
    if show_coupling_analysis:
        plot_coupling_analysis(coupled_system, t, results)


def plot_coupling_analysis(
    coupled_system: CoupledRLCircuitsPID, t: np.ndarray, results: Dict
):
    """
    Create additional plots to analyze magnetic coupling effects
    """
    n_circuits = coupled_system.n_circuits
    circuit_ids = coupled_system.circuit_ids
    colors = plt.cm.Set1(np.linspace(0, 1, n_circuits))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Current correlation matrix
    ax = axes[0, 0]
    current_matrix = np.array([results[cid]["current"] for cid in circuit_ids])
    correlation_matrix = np.corrcoef(current_matrix)

    im = ax.imshow(correlation_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n_circuits))
    ax.set_yticks(range(n_circuits))
    ax.set_xticklabels(circuit_ids, rotation=45)
    ax.set_yticklabels(circuit_ids)
    ax.set_title("Current Correlation Matrix")

    # Add correlation values as text
    for i in range(n_circuits):
        for j in range(n_circuits):
            text = ax.text(
                j,
                i,
                f"{correlation_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if abs(correlation_matrix[i, j]) < 0.5 else "white",
            )

    plt.colorbar(im, ax=ax)

    # 2. Phase plot of first two circuits (if available)
    if n_circuits >= 2:
        ax = axes[0, 1]
        circuit1_current = results[circuit_ids[0]]["current"]
        circuit2_current = results[circuit_ids[1]]["current"]

        ax.plot(circuit1_current, circuit2_current, color="blue", alpha=0.7)
        ax.set_xlabel(f"{circuit_ids[0]} Current (A)")
        ax.set_ylabel(f"{circuit_ids[1]} Current (A)")
        ax.set_title("Phase Plot: Circuit 1 vs Circuit 2")
        ax.grid(True, alpha=0.3)

    # 3. Mutual inductance matrix visualization
    ax = axes[1, 0]
    M = coupled_system.M
    im = ax.imshow(M, cmap="viridis")
    ax.set_xticks(range(n_circuits))
    ax.set_yticks(range(n_circuits))
    ax.set_xticklabels(circuit_ids, rotation=45)
    ax.set_yticklabels(circuit_ids)
    ax.set_title("Mutual Inductance Matrix (H)")

    # Add values as text
    for i in range(n_circuits):
        for j in range(n_circuits):
            text = ax.text(
                j,
                i,
                f"{float(M[i, j]):.3f}",
                ha="center",
                va="center",
                color="black" if M[i, j] < 0.5 * M.max() else "white",
            )

    plt.colorbar(im, ax=ax)

    # 4. Energy analysis
    ax = axes[1, 1]
    total_magnetic_energy = np.zeros(len(t))
    total_resistive_power = np.zeros(len(t))

    for i, circuit_id in enumerate(circuit_ids):
        data = results[circuit_id]
        circuit = coupled_system.circuits[circuit_id]

        # Magnetic energy in self-inductance
        magnetic_energy_self = 0.5 * circuit["L"] * data["current"] ** 2
        total_magnetic_energy += magnetic_energy_self

        # Resistive power
        total_resistive_power += data["power"]

    # Add mutual inductance energy
    for i in range(n_circuits):
        for j in range(
            i + 1, n_circuits
        ):  # Only upper triangle to avoid double counting
            circuit_i_current = results[circuit_ids[i]]["current"]
            circuit_j_current = results[circuit_ids[j]]["current"]
            mutual_energy = (
                coupled_system.M[i, j] * circuit_i_current * circuit_j_current
            )
            total_magnetic_energy += mutual_energy

    ax.plot(
        t, total_magnetic_energy, "blue", linewidth=2, label="Total Magnetic Energy"
    )
    ax.plot(
        t,
        np.cumsum(total_resistive_power) * (t[1] - t[0]),
        "red",
        linewidth=2,
        label="Cumulative Resistive Energy",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (J)")
    ax.set_title("System Energy Analysis")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_region_analysis(
    coupled_system: CoupledRLCircuitsPID, t: np.ndarray, results: Dict
):
    """
    Analyze and plot PID region usage for each circuit
    """
    n_circuits = coupled_system.n_circuits
    circuit_ids = coupled_system.circuit_ids

    fig, axes = plt.subplots(2, n_circuits, figsize=(5 * n_circuits, 10))
    if n_circuits == 1:
        axes = axes.reshape(2, 1)

    colors_regions = {
        "low": "lightgreen",
        "medium": "lightyellow",
        "high": "lightcoral",
    }

    for i, circuit_id in enumerate(circuit_ids):
        data = results[circuit_id]

        # Top row: Current with region coloring
        ax = axes[0, i]
        ax.plot(t, data["current"], "b-", linewidth=2, label="Actual")
        ax.plot(t, data["reference"], "r--", linewidth=2, label="Reference")

        # Color background by region
        prev_region = None
        for j, region in enumerate(data["regions"][::50]):  # Sample for performance
            if region != prev_region:
                region_start = t[j * 50] if j * 50 < len(t) else t[-1]
                # Find next region change
                region_end = t[-1]
                for k in range(j + 1, len(data["regions"][::50])):
                    if data["regions"][k * 50] != region and k * 50 < len(t):
                        region_end = t[k * 50]
                        break

                color = colors_regions.get(region.lower(), "lightgray")
                ax.axvspan(
                    region_start,
                    region_end,
                    alpha=0.3,
                    color=color,
                    label=f"{region}" if prev_region != region else "",
                )
                prev_region = region

        ax.set_title(f"{circuit_id} - Current & Regions")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Current (A)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Bottom row: PID gains
        ax = axes[1, i]
        ax.plot(t, data["Kp"], "g-", linewidth=2, label="Kp")
        ax.plot(t, data["Ki"], "b-", linewidth=2, label="Ki")
        ax.plot(t, data["Kd"] * 100, "r-", linewidth=2, label="Kd × 100")

        ax.set_title(f"{circuit_id} - PID Gains")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Gain Values")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_coupling_effects(
    coupled_system: CoupledRLCircuitsPID, t: np.ndarray, results: Dict
):
    """
    Provide detailed numerical analysis of coupling effects
    """
    n_circuits = coupled_system.n_circuits
    circuit_ids = coupled_system.circuit_ids

    print("\n=== Coupling Effects Analysis ===")

    # Current statistics
    print("\nCurrent Statistics:")
    for circuit_id in circuit_ids:
        data = results[circuit_id]
        current = data["current"]
        print(f"  {circuit_id}:")
        print(f"    Max current: {float(jnp.max(jnp.abs(current))):.3f} A")
        print(f"    RMS current: {float(jnp.sqrt(jnp.mean(current**2))):.3f} A")
        print(f"    Current variation (std): {float(jnp.std(current)):.3f} A")

    # Cross-correlation analysis
    print("\nCross-Correlation Analysis:")
    current_matrix = np.array([results[cid]["current"] for cid in circuit_ids])
    correlation_matrix = np.corrcoef(current_matrix)

    for i, circuit_i in enumerate(circuit_ids):
        for j, circuit_j in enumerate(circuit_ids):
            if i < j:  # Only upper triangle
                corr = correlation_matrix[i, j]
                coupling_strength = coupled_system.M[i, j]
                print(f"  {circuit_i} ↔ {circuit_j}:")
                print(f"    Mutual inductance: {coupling_strength:.4f} H")
                print(f"    Current correlation: {corr:.3f}")

    # Error analysis
    print("\nTracking Performance:")
    for circuit_id in circuit_ids:
        data = results[circuit_id]
        error = data["error"]
        rms_error = float(jnp.sqrt(jnp.mean(error**2)))
        max_error = float(jnp.max(jnp.abs(error)))
        print(f"  {circuit_id}:")
        print(f"    RMS error: {rms_error:.4f} A")
        print(f"    Max error: {max_error:.4f} A")

    # PID region usage
    print("\nPID Region Usage:")
    for circuit_id in circuit_ids:
        data = results[circuit_id]
        regions = data["regions"]
        unique_regions, counts = np.unique(regions, return_counts=True)
        total_time = len(regions) * (t[1] - t[0])

        print(f"  {circuit_id}:")
        for region, count in zip(unique_regions, counts):
            time_percent = (count / len(regions)) * 100
            print(f"    {region} region: {time_percent:.1f}% of time")

    # Energy analysis
    print("\nEnergy Analysis:")
    total_energy_dissipated = 0.0
    max_instantaneous_power = 0.0

    for circuit_id in circuit_ids:
        data = results[circuit_id]
        power = data["power"]
        energy_dissipated = float(jnp.sum(power) * (t[1] - t[0]))
        max_power = float(jnp.max(power))

        total_energy_dissipated += energy_dissipated
        max_instantaneous_power = max(max_instantaneous_power, max_power)

        print(f"  {circuit_id}:")
        print(f"    Energy dissipated: {energy_dissipated:.3f} J")
        print(f"    Max power: {max_power:.3f} W")

    print(f"  Total system energy dissipated: {total_energy_dissipated:.3f} J")
    print(f"  Max instantaneous power (any circuit): {max_instantaneous_power:.3f} W")


def create_coupling_comparison_plot(
    uncoupled_results: Dict,
    coupled_results: Dict,
    t: np.ndarray,
    circuit_ids: List[str],
):
    """
    Compare results with and without magnetic coupling
    """
    n_circuits = len(circuit_ids)
    colors = plt.cm.Set1(np.linspace(0, 1, n_circuits))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Current comparison
    ax = axes[0, 0]
    for i, circuit_id in enumerate(circuit_ids):
        uncoupled = uncoupled_results[circuit_id]["current"]
        coupled = coupled_results[circuit_id]["current"]

        ax.plot(
            t,
            uncoupled,
            color=colors[i],
            linestyle="--",
            alpha=0.7,
            label=f"{circuit_id} (uncoupled)",
        )
        ax.plot(
            t,
            coupled,
            color=colors[i],
            linestyle="-",
            linewidth=2,
            label=f"{circuit_id} (coupled)",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (A)")
    ax.set_title("Current: Coupled vs Uncoupled")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error comparison
    ax = axes[0, 1]
    for i, circuit_id in enumerate(circuit_ids):
        uncoupled_error = uncoupled_results[circuit_id]["error"]
        coupled_error = coupled_results[circuit_id]["error"]

        ax.plot(
            t,
            jnp.abs(uncoupled_error),
            color=colors[i],
            linestyle="--",
            alpha=0.7,
            label=f"{circuit_id} (uncoupled)",
        )
        ax.plot(
            t,
            jnp.abs(coupled_error),
            color=colors[i],
            linestyle="-",
            linewidth=2,
            label=f"{circuit_id} (coupled)",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|Error| (A)")
    ax.set_title("Absolute Error: Coupled vs Uncoupled")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Voltage comparison
    ax = axes[1, 0]
    for i, circuit_id in enumerate(circuit_ids):
        uncoupled_voltage = uncoupled_results[circuit_id]["voltage"]
        coupled_voltage = coupled_results[circuit_id]["voltage"]

        ax.plot(
            t,
            uncoupled_voltage,
            color=colors[i],
            linestyle="--",
            alpha=0.7,
            label=f"{circuit_id} (uncoupled)",
        )
        ax.plot(
            t,
            coupled_voltage,
            color=colors[i],
            linestyle="-",
            linewidth=2,
            label=f"{circuit_id} (coupled)",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Control Voltage: Coupled vs Uncoupled")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Power comparison
    ax = axes[1, 1]
    for i, circuit_id in enumerate(circuit_ids):
        uncoupled_power = uncoupled_results[circuit_id]["power"]
        coupled_power = coupled_results[circuit_id]["power"]

        ax.plot(
            t,
            uncoupled_power,
            color=colors[i],
            linestyle="--",
            alpha=0.7,
            label=f"{circuit_id} (uncoupled)",
        )
        ax.plot(
            t,
            coupled_power,
            color=colors[i],
            linestyle="-",
            linewidth=2,
            label=f"{circuit_id} (coupled)",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.set_title("Power: Coupled vs Uncoupled")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def save_coupled_results(
    coupled_system: CoupledRLCircuitsPID,
    t: np.ndarray,
    results: Dict,
    filename: str = "coupled_simulation_results.npz",
):
    """
    Save simulation results to a file for later analysis
    """
    # Prepare data for saving
    save_data = {
        "time": t,
        "n_circuits": coupled_system.n_circuits,
        "circuit_ids": coupled_system.circuit_ids,
        "mutual_inductances": coupled_system.M,
    }

    # Add results for each circuit
    for circuit_id in coupled_system.circuit_ids:
        data = results[circuit_id]
        save_data[f"{circuit_id}_current"] = data["current"]
        save_data[f"{circuit_id}_reference"] = data["reference"]
        save_data[f"{circuit_id}_error"] = data["error"]
        save_data[f"{circuit_id}_voltage"] = data["voltage"]
        save_data[f"{circuit_id}_power"] = data["power"]
        save_data[f"{circuit_id}_resistance"] = data["resistance"]
        save_data[f"{circuit_id}_Kp"] = data["Kp"]
        save_data[f"{circuit_id}_Ki"] = data["Ki"]
        save_data[f"{circuit_id}_Kd"] = data["Kd"]

    # Save to file
    np.savez_compressed(filename, **save_data)
    print(f"Results saved to {filename}")


def load_coupled_results(
    filename: str = "coupled_simulation_results.npz",
) -> Tuple[np.ndarray, Dict]:
    """
    Load previously saved simulation results
    """
    data = np.load(filename)

    t = data["time"]
    n_circuits = int(data["n_circuits"])
    circuit_ids = list(data["circuit_ids"])

    results = {}
    for circuit_id in circuit_ids:
        results[circuit_id] = {
            "current": data[f"{circuit_id}_current"],
            "reference": data[f"{circuit_id}_reference"],
            "error": data[f"{circuit_id}_error"],
            "voltage": data[f"{circuit_id}_voltage"],
            "power": data[f"{circuit_id}_power"],
            "resistance": data[f"{circuit_id}_resistance"],
            "Kp": data[f"{circuit_id}_Kp"],
            "Ki": data[f"{circuit_id}_Ki"],
            "Kd": data[f"{circuit_id}_Kd"],
        }

    print(f"Results loaded from {filename}")
    return t, results
