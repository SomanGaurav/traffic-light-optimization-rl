"""Thin wrapper around SUMO/TraCI for junction-level signal control."""

import os
import sys

# Prefer the tools shipped with a local SUMO install; fall back to the
# pip-installed traci/sumolib packages.
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci  # noqa: E402
from sumolib import checkBinary  # noqa: E402


def start_sumo(config, gui=False, tripinfo=None, seed=None):
    binary = checkBinary("sumo-gui" if gui else "sumo")
    cmd = [binary, "-c", config, "--no-warnings", "--no-step-log"]
    if tripinfo:
        cmd += ["--tripinfo-output", tripinfo]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    traci.start(cmd)


def close_sumo():
    traci.close()


def get_junction_lanes(junction):
    """Unique incoming lanes of a junction, in first-appearance order.

    getControlledLanes() repeats a lane once per connection; the de-duplicated
    list (one entry per approach lane) is what we use as the state vector.
    """
    seen = []
    for lane in traci.trafficlight.getControlledLanes(junction):
        if lane not in seen:
            seen.append(lane)
    return seen


def get_queue_lengths(lanes, min_position=10.0):
    """Vehicles per lane, ignoring vehicles right at the lane entrance."""
    counts = []
    for lane in lanes:
        count = 0
        for vehicle in traci.lane.getLastStepVehicleIDs(lane):
            if traci.vehicle.getLanePosition(vehicle) > min_position:
                count += 1
        counts.append(count)
    return counts


def get_waiting_time(lanes):
    return sum(traci.lane.getWaitingTime(lane) for lane in lanes)


def build_phases(junction):
    """Build (yellow, green) signal-state pairs, one per approach.

    Link indices are grouped by their incoming edge rather than by slicing
    the RYG string into equal chunks, so this works for any geometry:
    multi-lane approaches, dedicated turn lanes, or an uneven number of
    connections per approach. Exactly one approach is green at a time, so
    no conflicting movements are ever released together.
    """
    links = traci.trafficlight.getControlledLinks(junction)
    n_links = len(links)
    groups = {}
    for index, link in enumerate(links):
        if not link:
            continue  # unused link index
        incoming_edge = link[0][0].rsplit("_", 1)[0]
        groups.setdefault(incoming_edge, []).append(index)

    phases = []
    for indices in groups.values():
        yellow = ["r"] * n_links
        green = ["r"] * n_links
        for index in indices:
            yellow[index] = "y"
            green[index] = "G"
        phases.append(("".join(yellow), "".join(green)))
    return phases


def set_signal_state(junction, state, duration):
    traci.trafficlight.setRedYellowGreenState(junction, state)
    traci.trafficlight.setPhaseDuration(junction, duration)
