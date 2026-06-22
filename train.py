"""Train a DQN agent to control traffic signals in a SUMO network.

Example:
    python train.py --config configuration.sumocfg --model city1 --epochs 50 --steps 500
"""

import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sumo_env  # noqa: E402
from dqn_agent import Agent  # noqa: E402


def run_episode(agent, args, train=True, gui=False):
    """Run one simulation episode; returns total accumulated waiting time."""
    sumo_env.start_sumo(args.config, gui=gui, seed=args.seed)
    junctions = sumo_env.traci.trafficlight.getIDList()
    lanes = {j: sumo_env.get_junction_lanes(j) for j in junctions}
    phases = {j: sumo_env.build_phases(j) for j in junctions}

    prev_state = {j: [0] * len(lanes[j]) for j in junctions}
    prev_action = {j: 0 for j in junctions}
    timers = {j: 0 for j in junctions}          # steps until next decision
    pending_green = {j: None for j in junctions}  # (steps_left, state, duration)

    total_waiting = 0.0
    for step in range(args.steps):
        sumo_env.traci.simulationStep()
        for junction in junctions:
            total_waiting += sumo_env.get_waiting_time(lanes[junction])

            # Switch from yellow to the scheduled green phase.
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

            # Decision point: observe, store the completed transition, act.
            state = sumo_env.get_queue_lengths(lanes[junction])
            reward = -sumo_env.get_waiting_time(lanes[junction])
            done = step == args.steps - 1
            if train:
                agent.store_transition(
                    junction, prev_state[junction], prev_action[junction],
                    reward, state, done,
                )
                agent.learn(junction)

            action = agent.choose_action(state, greedy=not train)
            yellow, green = phases[junction][action]
            sumo_env.set_signal_state(junction, yellow, args.yellow_time)
            pending_green[junction] = (args.yellow_time, green, args.green_time)

            prev_state[junction] = state
            prev_action[junction] = action
            timers[junction] = args.yellow_time + args.green_time

    sumo_env.close_sumo()
    return total_waiting


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Probe the network once to size the agent's state/action spaces.
    sumo_env.start_sumo(args.config, gui=False, seed=args.seed)
    junctions = sumo_env.traci.trafficlight.getIDList()
    if not junctions:
        sys.exit("no traffic lights found in this network")
    lane_counts = {len(sumo_env.get_junction_lanes(j)) for j in junctions}
    phase_counts = {len(sumo_env.build_phases(j)) for j in junctions}
    if len(lane_counts) != 1 or len(phase_counts) != 1:
        sys.exit(f"junctions are not homogeneous (lane counts {lane_counts}, "
                 f"phase counts {phase_counts}); a single shared network "
                 "requires every junction to have the same geometry")
    input_dims = lane_counts.pop()
    n_actions = phase_counts.pop()
    sumo_env.close_sumo()

    agent = Agent(
        input_dims=input_dims,
        n_actions=n_actions,
        junctions=junctions,
        lr=args.lr,
        gamma=args.gamma,
        epsilon=1.0,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"device: {agent.device} | junctions: {len(junctions)} | "
          f"state dims: {input_dims} | actions: {n_actions}")

    model_path = os.path.join("models", f"{args.model}.pt")
    history = []
    best_time = float("inf")
    for epoch in range(args.epochs):
        total_waiting = run_episode(agent, args, train=True)
        history.append(total_waiting)
        marker = ""
        if total_waiting < best_time:
            best_time = total_waiting
            agent.save(model_path)
            marker = "  (new best, saved)"
        print(f"epoch {epoch + 1}/{args.epochs} | total waiting time: "
              f"{total_waiting:.0f} | epsilon: {agent.epsilon:.3f}{marker}")

    os.makedirs("plots", exist_ok=True)
    csv_path = os.path.join("plots", f"history_{args.model}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "total_waiting_time"])
        writer.writerows(enumerate(history, start=1))

    plt.plot(range(1, len(history) + 1), history)
    plt.xlabel("epoch")
    plt.ylabel("total waiting time (s)")
    plt.title(f"DQN training: {args.model}")
    plot_path = os.path.join("plots", f"time_vs_epoch_{args.model}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"best epoch: {best_time:.0f} | model: {model_path} | "
          f"plot: {plot_path} | history: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configuration.sumocfg",
                        help="SUMO configuration file")
    parser.add_argument("--model", default="model", help="model name to save")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps", type=int, default=500,
                        help="simulation steps per epoch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--green-time", type=int, default=15)
    parser.add_argument("--yellow-time", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
