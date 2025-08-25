import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from .rlcircuitpid import RLCircuitPID


def prepare_post(sol, circuit):
    t = sol.ts
    current = sol.ys[:, 0]
    integral_error = sol.ys[:, 1]

    # Calculate reference current for plotting
    if circuit is None:
        circuit = RLCircuitPID()

    # Vectorized evaluation using JAX
    i_ref = circuit.reference_func(t)

    # Calculate adaptive PID gains over time
    Kp_over_time = []
    Ki_over_time = []
    Kd_over_time = []
    current_regions = []

    # Use non-JAX version for post-processing
    for i, i_ref_val in enumerate(i_ref):
        # Convert to Python float for region name lookup
        i_ref_float = float(i_ref_val)
        Kp, Ki, Kd = circuit.get_pid_parameters(i_ref_float)
        Kp_over_time.append(float(Kp))
        Ki_over_time.append(float(Ki))
        Kd_over_time.append(float(Kd))

        region_name = circuit.get_current_region(i_ref_float)
        current_regions.append(region_name)

    Kp_over_time = jnp.array(Kp_over_time)
    Ki_over_time = jnp.array(Ki_over_time)
    Kd_over_time = jnp.array(Kd_over_time)

    # Calculate variable resistance over time
    if circuit.use_variable_resistance:
        resistance_over_time = jnp.array(
            [circuit.get_resistance(float(i)) for i in current]
        )
    else:
        resistance_over_time = jnp.full_like(current, circuit.R_constant)

    # Calculate control voltage with adaptive gains
    error = i_ref - current
    derivative_error = jnp.gradient(error, t[1] - t[0])
    voltage = (
        Kp_over_time * error
        + Ki_over_time * integral_error
        + Kd_over_time * derivative_error
    )

    # Calculate power dissipation
    power = resistance_over_time * current**2

    return (
        current_regions,
        Kp_over_time,
        Ki_over_time,
        Kd_over_time,
        voltage,
        error,
        resistance_over_time,
        power,
    )


def plot_results(
    sol,
    circuit,
    current_regions,
    Kp_over_time,
    Ki_over_time,
    Kd_over_time,
    voltage,
    error,
    resistance_over_time,
    power,
    experimental_data=None,  # New optional parameter
):
    """Plot the simulation results including adaptive PID behavior and optional experimental data"""
    t = sol.ts
    current = sol.ys[:, 0]

    # Vectorized evaluation using JAX
    i_ref = circuit.reference_func(t)

    # Create comprehensive plots
    fig, axes = plt.subplots(6, 1, figsize=(14, 20))

    # Current tracking with regions highlighted
    axes[0].plot(t, current, "b-", label="Actual Current", linewidth=2)
    axes[0].plot(t, i_ref, "r--", label="Reference Current", linewidth=2)

    # Color background by current region
    prev_region = None
    region_colors = {
        "Low": "lightgreen",
        "Medium": "lightyellow",
        "High": "lightcoral",
        "low": "lightgreen",
        "medium": "lightyellow",
        "high": "lightcoral",
    }

    for i, region in enumerate(
        current_regions[::100]
    ):  # Sample every 100 points for performance
        if region != prev_region:
            region_start = t[i * 100] if i * 100 < len(t) else t[-1]
            # Find next region change
            region_end = t[-1]
            for j in range(i + 1, len(current_regions[::100])):
                if current_regions[j * 100] != region and j * 100 < len(t):
                    region_end = t[j * 100]
                    break

            color_key = region.lower() if region.lower() in region_colors else region
            axes[0].axvspan(
                region_start,
                region_end,
                alpha=0.2,
                color=region_colors.get(color_key, "lightgray"),
                label=f"{region} Current" if prev_region != region else "",
            )
            prev_region = region

    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Current (A)")
    axes[0].set_title("Current Tracking Performance with Adaptive PID Regions")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Adaptive PID gains
    axes[1].plot(t, Kp_over_time, "g-", label="Kp", linewidth=2)
    axes[1].plot(t, Ki_over_time, "b-", label="Ki", linewidth=2)
    axes[1].plot(
        t, Kd_over_time * 100, "r-", label="Kd × 100", linewidth=2
    )  # Scale Kd for visibility
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("PID Gains")
    axes[1].set_title("Adaptive PID Parameters")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Control voltage with optional experimental comparison
    axes[2].plot(t, voltage, "purple", label="Computed Control Voltage", linewidth=2)

    # Add experimental data if available
    if experimental_data is not None:
        exp_time = experimental_data["time"].to_numpy()
        exp_voltage = experimental_data["voltage"].to_numpy()
        axes[2].plot(
            exp_time,
            exp_voltage,
            "orange",
            marker="o",
            markersize=3,
            alpha=0.7,
            label="Experimental Voltage",
            linewidth=1,
        )

        # Calculate and display comparison metrics if time ranges overlap
        t_min = max(float(t.min()), float(exp_time.min()))
        t_max = min(float(t.max()), float(exp_time.max()))

        if t_max > t_min:
            # Interpolate both datasets to common time grid for comparison
            import numpy as np

            common_time = np.linspace(t_min, t_max, 200)
            computed_interp = np.interp(common_time, t, voltage)
            exp_interp = np.interp(common_time, exp_time, exp_voltage)

            # Calculate RMS difference and MAE
            rms_diff = np.sqrt(np.mean((computed_interp - exp_interp) ** 2))
            mae_diff = np.mean(np.abs(computed_interp - exp_interp))

            # Add comparison metrics to the plot
            axes[2].text(
                0.02,
                0.98,
                f"RMS Diff: {rms_diff:.2f} V\nMAE Diff: {mae_diff:.2f} V",
                transform=axes[2].transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
            )

        axes[2].set_title("Control Voltage: Computed vs Experimental")
    else:
        axes[2].set_title("Adaptive PID Control Signal")

    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Voltage (V)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # Variable resistance
    axes[3].plot(
        t,
        resistance_over_time,
        "m-",
        label=f"Resistance (T={circuit.temperature}°C)",
        linewidth=2,
    )
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Resistance (Ω)")
    if circuit.use_variable_resistance:
        axes[3].set_title("Variable Resistance R(I, T)")
    else:
        axes[3].set_title("Constant Resistance")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    # Power dissipation
    axes[4].plot(t, power, "orange", label="Power Dissipation", linewidth=2)
    axes[4].set_xlabel("Time (s)")
    axes[4].set_ylabel("Power (W)")
    axes[4].set_title("Power Dissipation P = R(I,T) × I²")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()

    # Tracking error
    axes[5].plot(t, error, "r-", label="Tracking Error", linewidth=2)
    axes[5].set_xlabel("Time (s)")
    axes[5].set_ylabel("Error (A)")
    axes[5].set_title("Current Tracking Error")
    axes[5].grid(True, alpha=0.3)
    axes[5].legend()

    plt.tight_layout()
    plt.show()


def plot_vresults(
    sol,
    circuit,
    experimental_data=None,  # New optional parameter
):
    """Plot the simulation results including adaptive PID behavior and optional experimental data"""
    t = sol.ts
    current = sol.ys
    print("get current", type(current))

    voltage = circuit.voltage_func(t)
    print("get voltage", flush=True)

    # Calculate variable resistance over time
    # if circuit.use_variable_resistance:
    if circuit.use_variable_resistance:
        resistance_over_time = jnp.array(
            [circuit.get_resistance(float(i)) for i in current]
        )
    else:
        resistance_over_time = jnp.full_like(current, circuit.R_constant)
    print("get resistance", type(resistance_over_time), flush=True)

    # Calculate power dissipation
    power = resistance_over_time * current**2
    print("get power", flush=True)

    # Create comprehensive plots
    fig, axes = plt.subplots(4, 1, figsize=(14, 20), sharex=True)
    print("create subplots", flush=True)

    # Current tracking with regions highlighted
    axes[0].plot(t, current, "b-", label="Actual Current", linewidth=2)
    print("plot current...", flush=True)

    # Add experimental data if available
    print("experimental_data", experimental_data)
    if experimental_data is not None:
        exp_time = experimental_data["time"].to_numpy()
        print("exp_time", exp_time, flush=True)
        exp_current = experimental_data["current"].to_numpy()
        print("exp_current", exp_current, flush=True)
        axes[0].plot(
            exp_time,
            exp_current,
            "orange",
            marker="o",
            markersize=3,
            alpha=0.7,
            label="Experimental Current",
            linewidth=1,
        )

        axes[0].set_title("Current: Computed vs Experimental")
        print("plot experimental current", flush=True)
    else:
        axes[0].set_title("Current")
        print("setting title Current")

    # axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Current (A)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    print("plot current ", flush=True)

    # Control voltage with optional experimental comparison
    axes[1].plot(t, voltage, "purple", label="Input Voltage", linewidth=2)

    # axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    print("plot voltage", flush=True)

    # Variable resistance
    axes[2].plot(
        t,
        resistance_over_time,
        "m-",
        label=f"Resistance (T={circuit.temperature}°C)",
        linewidth=2,
    )
    # axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Resistance (Ω)")
    if circuit.use_variable_resistance:
        axes[2].set_title("Variable Resistance R(I, T)")
    else:
        axes[2].set_title("Constant Resistance")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    print("plot resistance", flush=True)

    # Power dissipation
    axes[3].plot(t, power, "orange", label="Power Dissipation", linewidth=2)
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Power (W)")
    axes[3].set_title("Power Dissipation P = R(I,T) × I²")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    print("plot power", flush=True)

    plt.tight_layout()
    plt.show()


def analytics(
    sol,
    circuit,
    current_regions,
    Kp_over_time,
    Ki_over_time,
    Kd_over_time,
    voltage,
    error,
    resistance_over_time,
    power,
):
    t = sol.ts
    current = sol.ys[:, 0]

    # Print comprehensive summary statistics
    print("\n=== Adaptive PID Simulation Summary ===")
    print(f"Temperature: {circuit.temperature}°C")

    # Get thresholds safely
    print("\nCurrent threshold:")
    print(circuit.pid_controller.get_thresholds())

    # Analyze time spent in each region
    print("\nTime Distribution by Current Region:")
    unique_regions, region_counts = np.unique(current_regions, return_counts=True)
    total_time = len(current_regions) * (t[1] - t[0])

    print("\nTime Distribution by Current Region:")
    for region, count in zip(unique_regions, region_counts):
        time_percent = (count / len(current_regions)) * 100
        print(f"  {region} Current: {time_percent:.1f}% of simulation time")

    # PID parameter ranges
    print("\nPID Parameter Ranges:")
    print(
        f"  Kp: {float(jnp.min(Kp_over_time)):.2f} - {float(jnp.max(Kp_over_time)):.2f}"
    )
    print(
        f"  Ki: {float(jnp.min(Ki_over_time)):.2f} - {float(jnp.max(Ki_over_time)):.2f}"
    )
    print(
        f"  Kd: {float(jnp.min(Kd_over_time)):.4f} - {float(jnp.max(Kd_over_time)):.4f}"
    )

    if circuit.use_variable_resistance:
        print(
            f"\nResistance range: {float(jnp.min(resistance_over_time)):.4f} - {float(jnp.max(resistance_over_time)):.4f} Ω"
        )
    else:
        print(f"\nConstant resistance: {circuit.R_constant:.4f} Ω")

    print(f"Max current: {float(jnp.max(jnp.abs(current))):.1f} A")
    print(f"Max power: {float(jnp.max(power)):.1f} W")
    print(f"RMS error: {float(jnp.sqrt(jnp.mean(error**2))):.4f} A")

    # Region-specific performance analysis
    print("\nRegion-Specific RMS Errors:")
    for region in unique_regions:
        region_mask = np.array(current_regions) == region
        if np.any(region_mask):
            region_error = error[region_mask]
            region_rms = float(jnp.sqrt(jnp.mean(region_error**2)))
            print(f"  {region} Current Region: {region_rms:.4f} A")
