"""Evaluate a trained DQN signal controller against a fixed-time baseline.

Runs the same simulation twice — once with the classic round-robin
fixed-time controller, once with the trained agent acting greedily —
and reports total vehicle waiting time for each.

Example:
    python evaluate.py --config configuration.sumocfg --model city1 --steps 500
"""

import argparse
import os
import sys

import sumo_env
from dqn_agent import Agent


def run_fixed_time(args):
    """Round-robin baseline: cycle approaches with fixed green times."""
    sumo_env.start_sumo(args.config, gui=args.gui, seed=args.seed)
    junctions = sumo_env.traci.trafficlight.getIDList()
    lanes = {j: sumo_env.get_junction_lanes(j) for j in junctions}
    phases = {j: sumo_env.build_phases(j) for j in junctions}

    current = {j: 0 for j in junctions}
    timers = {j: 0 for j in junctions}
    pending_green = {j: None for j in junctions}

    total_waiting = 0.0
    for _ in range(args.steps):
        sumo_env.traci.simulationStep()
        for junction in junctions:
            total_waiting += sumo_env.get_waiting_time(lanes[junction])
            if pending_green[junction] is not None:
                steps_left, state, duration = pending_green[junction]
                if steps_left <= 1:
                    sumo_env.set_signal_state(junction, state, duration)
                    pending_green[junction] = None
                else:
                    pending_green[junction] = (steps_left - 1, state, duration)
            if timers[junction] > 0:
                timers[junction] -= 1
                continue
            current[junction] = (current[junction] + 1) % len(phases[junction])
            yellow, green = phases[junction][current[junction]]
            sumo_env.set_signal_state(junction, yellow, args.yellow_time)
            pending_green[junction] = (args.yellow_time, green, args.green_time)
            timers[junction] = args.yellow_time + args.green_time
    sumo_env.close_sumo()
    return total_waiting


def run_agent(args):
    """Trained DQN agent acting greedily."""
    sumo_env.start_sumo(args.config, gui=args.gui, seed=args.seed)
    junctions = sumo_env.traci.trafficlight.getIDList()
    lanes = {j: sumo_env.get_junction_lanes(j) for j in junctions}
    phases = {j: sumo_env.build_phases(j) for j in junctions}
    input_dims = len(lanes[junctions[0]])
    n_actions = len(phases[junctions[0]])

    agent = Agent(input_dims=input_dims, n_actions=n_actions, junctions=junctions,
                  epsilon=0.0, seed=args.seed)
    model_path = os.path.join("models", f"{args.model}.pt")
    if not os.path.exists(model_path):
        sumo_env.close_sumo()
        sys.exit(f"model not found: {model_path} — train one with train.py first")
    agent.load(model_path)

    timers = {j: 0 for j in junctions}
    pending_green = {j: None for j in junctions}

    total_waiting = 0.0
    for _ in range(args.steps):
        sumo_env.traci.simulationStep()
        for junction in junctions:
            total_waiting += sumo_env.get_waiting_time(lanes[junction])
            if pending_green[junction] is not None:
                steps_left, state, duration = pending_green[junction]
                if steps_left <= 1:
                    sumo_env.set_signal_state(junction, state, duration)
                    pending_green[junction] = None
                else:
                    pending_green[junction] = (steps_left - 1, state, duration)
            if timers[junction] > 0:
                timers[junction] -= 1
                continue
            state = sumo_env.get_queue_lengths(lanes[junction])
            action = agent.choose_action(state, greedy=True)
            yellow, green = phases[junction][action]
            sumo_env.set_signal_state(junction, yellow, args.yellow_time)
            pending_green[junction] = (args.yellow_time, green, args.green_time)
            timers[junction] = args.yellow_time + args.green_time
    sumo_env.close_sumo()
    return total_waiting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configuration.sumocfg")
    parser.add_argument("--model", default="model")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--green-time", type=int, default=15)
    parser.add_argument("--yellow-time", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true", help="show SUMO-GUI")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    agent_time = run_agent(args)
    print(f"DQN agent      | total waiting time: {agent_time:>12.0f} s")
    if not args.skip_baseline:
        baseline_time = run_fixed_time(args)
        print(f"fixed-time     | total waiting time: {baseline_time:>12.0f} s")
        if baseline_time > 0:
            change = 100.0 * (baseline_time - agent_time) / baseline_time
            print(f"improvement over fixed-time baseline: {change:+.1f}%")


if __name__ == "__main__":
    main()
