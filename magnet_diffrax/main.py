#!/usr/bin/env python3
"""
Basic RL Circuit PID Control System

Simple command-line interface for running RL circuit simulations with
adaptive PID control. Supports loading resistance and reference data from CSV files.
"""

import argparse
import sys
import jax.numpy as jnp
import diffrax
import pandas as pd
import numpy as np

from .pid_controller import (
    create_default_pid_controller,
    create_adaptive_pid_controller,
)
from .rlcircuitpid import RLCircuitPID
from .plotting import prepare_post, plot_results, analytics, plot_vresults


def run_simulation(args):
    """Run the main simulation with given arguments"""

    print("\n=== RL Circuit PID Simulation ===")
    print(f"Inductance: {args.inductance} H")
    print(f"Resistance: {args.resistance} Ω")
    print(f"Temperature: {args.temperature} °C")
    print(f"Time span: {args.time_start} to {args.time_end} seconds")

    # Load experimental data if provided
    experimental_data = None
    if args.experimental_csv:
        try:
            experimental_data = pd.read_csv(args.experimental_csv)

            key = "voltage"
            if args.voltage_csv:
                key = "current"

            # Validate required columns
            if (
                "time" not in experimental_data.columns
                or key not in experimental_data.columns
            ):
                raise ValueError(
                    f"Experimental CSV must contain 'time' and {key} columns"
                )
            print(
                f"✓ Experimental data loaded: {len(experimental_data['time'])} points"
            )
        except Exception as e:
            print(f"❌ Error loading experimental data: {e}")
            experimental_data = None

    # Create PID controller
    pid_controller = None
    if not args.voltage_csv:
        if args.custom_pid:
            print("Using custom PID parameters")
            pid_controller = create_adaptive_pid_controller(
                Kp_low=args.kp_low,
                Ki_low=args.ki_low,
                Kd_low=args.kd_low,
                Kp_medium=args.kp_medium,
                Ki_medium=args.ki_medium,
                Kd_medium=args.kd_medium,
                Kp_high=args.kp_high,
                Ki_high=args.ki_high,
                Kd_high=args.kd_high,
                low_threshold=args.low_threshold,
                high_threshold=args.high_threshold,
            )
        else:
            print("Using default PID parameters")
            pid_controller = create_default_pid_controller()

    # Create circuit
    print("args.voltage_csv:", args.voltage_csv)
    circuit = RLCircuitPID(
        circuit_id=args.circuit_id,
        R=args.resistance,
        L=args.inductance,
        pid_controller=pid_controller,
        reference_csv=args.reference_csv,
        voltage_csv=args.voltage_csv,
        resistance_csv=args.resistance_csv,
        temperature=args.temperature,
    )

    # Print configuration
    circuit.print_configuration()

    # Time parameters
    t0, t1 = args.time_start, args.time_end
    dt = args.time_step

    # Regular ODE
    post_data = None
    if args.voltage_csv:
        print(f"Using input voltage CSV: {args.voltage_csv}")
        i0 = args.value_start
        print(f"Initial current: {i0} A at t={t0} s")
        print(f"Initial voltage: {circuit.input_voltage(t0)} V at t={t0} s")
        y0 = i0

        # Create solver
        solver = diffrax.Dopri5()

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(circuit.voltage_vector_field),
            solver,
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=y0,
            max_steps=10000,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-4),
            saveat=diffrax.SaveAt(ts=jnp.arange(t0, t1, dt)),
        )

        print("✓ Simulation completed")
        plot_vresults(sol, circuit, experimental_data=experimental_data)

    else:
        # Initial conditions [current, integral_error, prev_error]
        i0 = args.value_start
        print(f"Initial current: {i0} A at t={t0} s")
        i0_ref = circuit.reference_current(t0)
        print(f"init ref: {i0_ref:.3f} A at t={t0:.3f} s")
        if experimental_data is not None:
            v0 = np.interp(t0, experimental_data["time"], experimental_data["voltage"])
            print(f"init exp: {v0:.3f} V at t={t0:.3f} s")

        y0 = jnp.array([i0, 0.0, (i0 - i0_ref)])

        print("\nRunning simulation...")
        print(f"Time step: {dt} seconds")

        # Create solver
        solver = diffrax.Dopri5()

        # Solve the differential equation
        # (max_steps: int | None = 4096)
        # stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-4)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(circuit.vector_field),
            solver,
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=y0,
            max_steps=10000,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-4),
            saveat=diffrax.SaveAt(ts=jnp.arange(t0, t1, dt)),
        )

        print("✓ Simulation completed")

        # Post-process results
        print("Processing results...")
        post_data = prepare_post(sol, circuit)
        (
            current_regions,
            Kp_over_time,
            Ki_over_time,
            Kd_over_time,
            voltage,
            error,
            resistance_over_time,
            power,
        ) = post_data

        # Show analytics
        if args.show_analytics:
            analytics(
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
            )

        # Show plots
        if args.show_plots:
            print("Generating plots...")
            plot_results(
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
                experimental_data=experimental_data,
            )

    return sol, circuit, post_data


def main():
    """Main function with argument parsing"""

    parser = argparse.ArgumentParser(
        description="RL Circuit PID Control Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input files
    parser.add_argument(
        "--circuit_id",
        type=str,
        help="Circuit id (ex: M9Bitters)",
        default="M9Bitters",
    )

    # Input files
    parser.add_argument(
        "--reference_csv",
        type=str,
        help="Path to CSV file with reference current data (columns: time, current)",
    )

    parser.add_argument(
        "--voltage_csv",
        type=str,
        help="Path to CSV file with input voltage data (columns: time, voltage)",
    )

    parser.add_argument(
        "--resistance_csv",
        type=str,
        help="Path to CSV file with resistance data (columns: current, temperature, resistance)",
    )

    parser.add_argument(
        "--experimental_csv",
        "-e",
        type=str,
        help="Path to CSV file with experimental voltage data for comparison (columns: time, voltage)",
    )

    # Circuit parameters
    parser.add_argument(
        "--inductance",
        type=float,
        default=0.1,
        help="Circuit inductance in Henry",
    )

    parser.add_argument(
        "--resistance",
        type=float,
        default=1.5,
        help="Constant resistance in Ohms (used if --resistance-csv not provided)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=25.0,
        help="Operating temperature in Celsius",
    )

    # Simulation parameters
    parser.add_argument(
        "--value_start",
        type=float,
        default=0.0,
        help="Current value at start time in Ampere",
    )

    parser.add_argument(
        "--time_start", type=float, default=0.0, help="Simulation start time in seconds"
    )

    parser.add_argument(
        "--time_end", type=float, default=5.0, help="Simulation end time in seconds"
    )

    parser.add_argument(
        "--time_step", type=float, default=0.001, help="Simulation time step in seconds"
    )

    # PID parameters (optional custom settings)
    parser.add_argument(
        "--custom_pid",
        action="store_true",
        help="Use custom PID parameters instead of defaults",
    )

    parser.add_argument(
        "--kp_low", type=float, default=20.0, help="Kp for low current region"
    )
    parser.add_argument(
        "--ki_low", type=float, default=15.0, help="Ki for low current region"
    )
    parser.add_argument(
        "--kd_low", type=float, default=0.1, help="Kd for low current region"
    )

    parser.add_argument(
        "--kp_medium", type=float, default=12.0, help="Kp for medium current region"
    )
    parser.add_argument(
        "--ki_medium", type=float, default=8.0, help="Ki for medium current region"
    )
    parser.add_argument(
        "--kd_medium", type=float, default=0.05, help="Kd for medium current region"
    )

    parser.add_argument(
        "--kp_high", type=float, default=8.0, help="Kp for high current region"
    )
    parser.add_argument(
        "--ki_high", type=float, default=5.0, help="Ki for high current region"
    )
    parser.add_argument(
        "--kd_high", type=float, default=0.02, help="Kd for high current region"
    )

    parser.add_argument(
        "--low_threshold",
        type=float,
        default=60.0,
        help="Low to medium current threshold",
    )
    parser.add_argument(
        "--high_threshold",
        type=float,
        default=800.0,
        help="Medium to high current threshold",
    )

    # Output options
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show plots of simulation results",
    )

    parser.add_argument(
        "--show-analytics",
        action="store_true",
        help="Show detailed analytics of simulation results",
    )

    # Parse arguments
    args = parser.parse_args()
    print(f"args: {args}")

    # Check if files exist when specified
    if args.reference_csv:
        try:
            pd.read_csv(args.reference_csv)
            print(f"✓ Reference CSV loaded: {args.reference_csv}")
        except FileNotFoundError:
            print(f"❌ Reference CSV file not found: {args.reference_csv}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading reference CSV: {e}")
            sys.exit(1)

    # Check if files exist when specified
    if args.voltage_csv:
        try:
            pd.read_csv(args.voltage_csv)
            print(f"✓ Input Voltage CSV loaded: {args.voltage_csv}")
        except FileNotFoundError:
            print(f"❌ Input voltage CSV file not found: {args.voltage_csv}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading reference CSV: {e}")
            sys.exit(1)

    if args.resistance_csv:
        try:
            pd.read_csv(args.resistance_csv)
            print(f"✓ Resistance CSV loaded: {args.resistance_csv}")
        except FileNotFoundError:
            print(f"❌ Resistance CSV file not found: {args.resistance_csv}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading resistance CSV: {e}")
            sys.exit(1)

    if args.experimental_csv:
        try:
            pd.read_csv(args.experimental_csv)
            print(f"✓ Experimental CSV loaded: {args.experimental_csv}")
        except FileNotFoundError:
            print(f"❌ Experimental CSV file not found: {args.experimental_csv}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading experimental CSV: {e}")
            sys.exit(1)

    # Run simulation
    try:
        sol, circuit, post_data = run_simulation(args)
        print("\n✓ Simulation completed successfully!")

    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
