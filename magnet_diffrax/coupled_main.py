#!/usr/bin/env python3
"""
Coupled RL Circuits PID Control System

Command-line interface for running coupled RL circuit simulations with
independent adaptive PID controllers and magnetic coupling.
"""

import argparse
import sys
import json
import jax.numpy as jnp
import diffrax
import pandas as pd
import numpy as np
from typing import List

from .coupled_circuits import CoupledRLCircuitsPID
from .rlcircuitpid import RLCircuitPID
from .coupled_plotting import (
    prepare_coupled_post,
    plot_coupled_results,
    plot_region_analysis,
    analyze_coupling_effects,
    save_coupled_results,
)
from pid_controller import create_adaptive_pid_controller


def create_sample_coupled_data(n_circuits: int = 3):
    """Create sample CSV files for testing coupled circuits"""

    time = np.linspace(0, 5, 1000)

    # Create different reference currents for each circuit
    references = []

    # Circuit 1: Step changes
    current1 = np.zeros_like(time)
    current1[time >= 0.5] = 15.0
    current1[time >= 1.5] = 150.0
    current1[time >= 2.5] = 75.0
    current1[time >= 3.5] = 400.0
    current1[time >= 4.5] = 100.0
    references.append(current1)

    # Circuit 2: Sinusoidal with offset
    current2 = 50.0 + 40.0 * np.sin(2 * np.pi * time * 0.8) * (time > 0.5)
    current2 = np.maximum(current2, 0.0)
    references.append(current2)

    # Circuit 3: Ramp with steps
    current3 = np.zeros_like(time)
    current3[time >= 1.0] = 20.0 * (time[time >= 1.0] - 1.0)
    current3[time >= 3.0] = 200.0
    current3 = np.minimum(current3, 300.0)
    references.append(current3)

    # Add more circuits if needed
    for i in range(3, n_circuits):
        # Create varied patterns
        if i % 2 == 0:
            current = 30.0 + 25.0 * np.sin(2 * np.pi * time * (0.5 + 0.2 * i))
        else:
            current = np.zeros_like(time)
            current[time >= 0.5 + 0.3 * i] = 50.0 + 20.0 * i
            current[time >= 2.0 + 0.2 * i] = 100.0 + 30.0 * i

        current = np.maximum(current, 0.0)
        references.append(current)

    # Save reference files
    for i in range(min(n_circuits, len(references))):
        df = pd.DataFrame({"time": time, "current": references[i]})
        filename = f"sample_reference_circuit_{i+1}.csv"
        df.to_csv(filename, index=False)
        print(f"Created {filename}")

    # Create sample resistance data for different circuits
    current_vals = np.linspace(0, 500, 40)
    temp_vals = np.linspace(20, 60, 25)

    for circuit_idx in range(n_circuits):
        data = []
        for temp in temp_vals:
            for curr in current_vals:
                # Different resistance models for each circuit
                R0 = 1.2 + 0.3 * circuit_idx  # Different base resistance
                alpha = 0.003 + 0.001 * circuit_idx  # Different temperature coefficient
                beta = 0.00008 + 0.00002 * circuit_idx  # Different current coefficient

                resistance = R0 * (1 + alpha * (temp - 25) + beta * curr)
                data.append(
                    {"current": curr, "temperature": temp, "resistance": resistance}
                )

        df = pd.DataFrame(data)
        filename = f"sample_resistance_circuit_{circuit_idx+1}.csv"
        df.to_csv(filename, index=False)
        print(f"Created {filename}")


def load_circuit_configuration(config_file: str) -> List[RLCircuitPID]:
    """Load circuit configuration from JSON file"""
    try:
        with open(config_file, "r") as f:
            config_data = json.load(f)

        circuits = []
        for circuit_data in config_data["circuits"]:
            # Create PID controller if parameters provided
            pid_controller = None
            if "pid_params" in circuit_data:
                pid_params = circuit_data["pid_params"]
                pid_controller = create_adaptive_pid_controller(**pid_params)

            # Create RLCircuitPID instance
            circuit = RLCircuitPID(
                R=circuit_data.get("R", 1.0),
                L=circuit_data.get("L", 0.1),
                pid_controller=pid_controller,
                reference_csv=circuit_data.get("reference_csv"),
                resistance_csv=circuit_data.get("resistance_csv"),
                temperature=circuit_data.get("temperature", 25.0),
                circuit_id=circuit_data.get("circuit_id", f"circuit_{len(circuits)+1}"),
            )
            circuits.append(circuit)

        # Load mutual inductance matrix if provided
        mutual_inductances = None
        if "mutual_inductances" in config_data:
            mutual_inductances = np.array(config_data["mutual_inductances"])

        return circuits, mutual_inductances

    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)


def create_sample_config(n_circuits: int = 3, filename: str = "sample_config.json"):
    """Create a sample configuration file"""
    config = {"circuits": [], "mutual_inductances": None}

    # Create circuit configurations
    for i in range(n_circuits):
        circuit_config = {
            "circuit_id": f"circuit_{i+1}",
            "R": 1.0 + 0.2 * i,
            "L": 0.08 + 0.02 * i,
            "temperature": 25.0 + 5.0 * i,
            "reference_csv": f"sample_reference_circuit_{i+1}.csv",
            "resistance_csv": f"sample_resistance_circuit_{i+1}.csv",
            "pid_params": {
                "Kp_low": 15.0 + 2.0 * i,
                "Ki_low": 8.0 + 1.0 * i,
                "Kd_low": 0.08 + 0.01 * i,
                "Kp_medium": 18.0 + 2.0 * i,
                "Ki_medium": 10.0 + 1.0 * i,
                "Kd_medium": 0.06 + 0.01 * i,
                "Kp_high": 22.0 + 2.0 * i,
                "Ki_high": 12.0 + 1.0 * i,
                "Kd_high": 0.04 + 0.01 * i,
                "low_threshold": 60.0,
                "high_threshold": 200.0 + 100.0 * i,
            },
        }
        config["circuits"].append(circuit_config)

    # Create mutual inductance matrix
    coupling_strength = 0.02
    mutual_inductances = np.full((n_circuits, n_circuits), coupling_strength)
    np.fill_diagonal(mutual_inductances, 0.0)
    config["mutual_inductances"] = mutual_inductances.tolist()

    # Save configuration
    with open(filename, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Created sample configuration: {filename}")


def run_coupled_simulation(args):
    """Run the coupled circuits simulation"""

    print("\n=== Coupled RL Circuits PID Simulation ===")

    # Load or create circuit configuration
    if args.config_file:
        print(f"Loading configuration from: {args.config_file}")
        circuits, mutual_inductances = load_circuit_configuration(args.config_file)
    else:
        print("Creating default configuration...")
        circuits = []

        for i in range(args.n_circuits):
            circuit = RLCircuitPID(
                R=args.resistance + 0.1 * i,
                L=args.inductance + 0.01 * i,
                temperature=args.temperature + 2.0 * i,
                circuit_id=f"circuit_{i+1}",
            )
            circuits.append(circuit)

        mutual_inductances = None

    # Create coupled system
    if mutual_inductances is not None:
        coupled_system = CoupledRLCircuitsPID(
            circuits, mutual_inductances=mutual_inductances
        )
    else:
        coupled_system = CoupledRLCircuitsPID(
            circuits, coupling_strength=args.coupling_strength
        )

    # Print configuration
    coupled_system.print_configuration()

    # Initial conditions
    y0 = coupled_system.get_initial_conditions()

    # Time parameters
    t0, t1 = args.time_start, args.time_end
    dt = args.time_step

    print(  "\nRunning simulation...")
    print(f"Time span: {t0} to {t1} seconds")
    print(f"Time step: {dt} seconds")
    print(f"State vector size: {len(y0)}")

    # Create solver
    solver = diffrax.Dopri5()

    # Solve the differential equation
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(coupled_system.vector_field),
        solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=y0,
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-4),
        saveat=diffrax.SaveAt(ts=jnp.arange(t0, t1, dt)),
    )

    print("✓ Simulation completed")

    # Post-process results
    print("Processing results...")
    t, results = prepare_coupled_post(sol, coupled_system)

    # Show analytics
    if args.show_analytics:
        analyze_coupling_effects(coupled_system, t, results)

    # Show plots
    if args.show_plots:
        print("Generating plots...")
        plot_coupled_results(
            sol,
            coupled_system,
            t,
            results,
            show_coupling_analysis=args.show_coupling,
            show_individual_details=args.show_details,
        )

        if args.show_regions:
            plot_region_analysis(coupled_system, t, results)

    # Save results if requested
    if args.save_results:
        save_coupled_results(coupled_system, t, results, args.save_results)

    return sol, coupled_system, t, results


def main():
    """Main function with argument parsing"""

    parser = argparse.ArgumentParser(
        description="Coupled RL Circuits PID Control Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration options
    parser.add_argument(
        "--config-file",
        "-c",
        type=str,
        help="Path to JSON configuration file with circuit definitions",
    )

    parser.add_argument(
        "--n-circuits",
        "-n",
        type=int,
        default=3,
        help="Number of circuits (used if no config file)",
    )

    # Circuit parameters (used if no config file)
    parser.add_argument(
        "--inductance",
        "-L",
        type=float,
        default=0.1,
        help="Base inductance in Henry (varied for each circuit)",
    )

    parser.add_argument(
        "--resistance",
        "-R",
        type=float,
        default=1.0,
        help="Base resistance in Ohms (varied for each circuit)",
    )

    parser.add_argument(
        "--temperature",
        "-T",
        type=float,
        default=25.0,
        help="Base temperature in Celsius (varied for each circuit)",
    )

    parser.add_argument(
        "--coupling-strength",
        type=float,
        default=0.05,
        help="Mutual inductance coupling strength (if no config file)",
    )

    # Simulation parameters
    parser.add_argument(
        "--time-start", type=float, default=0.0, help="Simulation start time in seconds"
    )

    parser.add_argument(
        "--time-end", type=float, default=5.0, help="Simulation end time in seconds"
    )

    parser.add_argument(
        "--time-step", type=float, default=0.001, help="Simulation time step in seconds"
    )

    # Output options
    parser.add_argument(
        "--show-plots",
        "-p",
        action="store_true",
        help="Show plots of simulation results",
    )

    parser.add_argument(
        "--show-analytics",
        "-a",
        action="store_true",
        help="Show detailed analytics of simulation results",
    )

    parser.add_argument(
        "--show-coupling",
        action="store_true",
        help="Show detailed coupling analysis plots",
    )

    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Show detailed individual circuit plots",
    )

    parser.add_argument(
        "--show-regions", action="store_true", help="Show PID region analysis plots"
    )

    parser.add_argument(
        "--save-results",
        type=str,
        help="Save results to specified file (e.g., results.npz)",
    )

    # Sample generation
    parser.add_argument(
        "--create-samples",
        action="store_true",
        help="Create sample CSV files and configuration",
    )

    parser.add_argument(
        "--create-config",
        type=str,
        help="Create sample configuration file with specified name",
    )

    # Parse arguments
    args = parser.parse_args()

    # Create sample files if requested
    if args.create_samples:
        create_sample_coupled_data(args.n_circuits)
        return

    if args.create_config:
        create_sample_config(args.n_circuits, args.create_config)
        return

    # Validate configuration file if provided
    if args.config_file:
        try:
            with open(args.config_file, "r") as f:
                config = json.load(f)
            print(f"✓ Configuration file loaded: {args.config_file}")
        except FileNotFoundError:
            print(f"✗ Configuration file not found: {args.config_file}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error reading configuration file: {e}")
            sys.exit(1)

    # Run simulation
    try:
        sol, coupled_system, t, results = run_coupled_simulation(args)
        print("\n✓ Coupled simulation completed successfully!")
        print(f"  Circuits simulated: {coupled_system.n_circuits}")
        print(f"  Time points: {len(t)}")
        print(f"  Total simulation time: {float(t[-1] - t[0]):.3f} seconds")

    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
