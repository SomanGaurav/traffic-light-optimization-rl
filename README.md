# Traffic Signal Control with Deep Q-Learning

Adaptive traffic signal control for multi-junction road networks, built on
[SUMO](https://sumo.dlr.de/) and PyTorch. A Deep Q-Network observes live queue
lengths at each intersection and picks which approach gets the green light,
minimizing total vehicle waiting time — evaluated head-to-head against a
classic fixed-time controller.

## How it works

Every junction is a decision point for a single shared DQN:

| Component | Definition |
|---|---|
| **State** | Queue length (vehicle count) on each incoming lane of the junction |
| **Action** | Which of the 4 approaches receives the green phase |
| **Reward** | Negative total waiting time on the junction's lanes |
| **Transition timing** | A 4 s yellow phase precedes every green; a minimum green time prevents rapid flickering |

The agent is a standard DQN with the pieces that make it actually work:

- **Experience replay** — each junction has its own ring buffer; learning
  samples random minibatches rather than replaying history in order, breaking
  the correlation between consecutive transitions.
- **Target network** — bootstrap targets come from a periodically synced copy
  of the policy network, stabilizing training.
- **ε-greedy exploration** — ε decays from 1.0 to 0.05 over training;
  evaluation runs fully greedy.
- **Huber loss + gradient clipping** — robust to the occasional
  large-magnitude reward when a queue spikes.
- One shared network controls all junctions, so what the agent learns at one
  intersection transfers to every other — and to unseen maps with the same
  junction geometry.

State size and action count are both read from the network at startup, and
green phases are built by grouping a junction's controlled links by their
incoming edge. That keeps the same code working across single-lane grids and
multi-lane corridors with dedicated turn lanes, instead of assuming a fixed
four-value state or a particular link ordering.

## Project structure

```
├── train.py               # Training loop (CLI: epochs, steps, hyperparameters, seed)
├── evaluate.py            # Trained agent vs. fixed-time baseline comparison
├── dqn_agent.py           # Q-network, replay buffer, DQN agent
├── sumo_env.py            # TraCI helpers: queues, waiting time, signal phases
├── configuration.sumocfg  # Default scenario (city1)
└── maps/                  # Road networks + route files
    ├── city1..city3       # Single-lane grid networks
    ├── corridor.nod/.edg  # Two-junction multi-lane corridor (source)
    └── corridor.sumocfg   # ...built with: netconvert -n *.nod.xml -e *.edg.xml
```

### The corridor scenario

`maps/corridor.*` models a pair of signalized 4-way intersections ~200 m apart
on a shared arterial, three lanes per direction — the layout of a typical urban
corridor. Unlike the single-lane `city*` grids, each junction has 20 controlled
connections (5 per approach), so approach grouping has to be derived from the
network rather than assumed. It is defined as plain node/edge XML and compiled
with `netconvert`, so the geometry is reproducible and easy to edit:

```bash
netconvert -n maps/corridor.nod.xml -e maps/corridor.edg.xml \
           -o maps/corridor.net.xml --no-turnarounds --tls.default-type static
python $SUMO_HOME/tools/randomTrips.py -n maps/corridor.net.xml \
       -r maps/corridor.rou.xml -e 1000 -p 2.0 --seed 42 --validate
```

## Setup

1. Install [SUMO](https://sumo.dlr.de/docs/Downloads.php) and set `SUMO_HOME`
   (on Ubuntu: `sudo apt install sumo sumo-tools` and
   `export SUMO_HOME=/usr/share/sumo`). Alternatively `pip install eclipse-sumo`
   provides the binaries.
2. Install the Python dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train on the map referenced by `configuration.sumocfg`:

```bash
python train.py --model city1 --epochs 50 --steps 500
```

The best checkpoint is saved to `models/city1.pt`; the training curve and a
per-epoch CSV land in `plots/`.

Evaluate against the fixed-time baseline (add `--gui` to watch it live):

```bash
python evaluate.py --model city1 --steps 500
```

which reports total waiting time for both controllers and the relative
improvement. To switch maps, point `configuration.sumocfg` at another
`maps/*.net.xml` / `maps/*.rou.xml` pair. New route files for a network can be
generated with SUMO's bundled tool:

```bash
python $SUMO_HOME/tools/randomTrips.py -n maps/city1.net.xml -r maps/city1.rou.xml -e 500
```

All hyperparameters (learning rate, γ, ε schedule, batch size, green/yellow
durations, seed) are exposed as CLI flags — see `python train.py --help`.

## Results

Trained with `python train.py --model city1 --epochs 50 --steps 500` (CPU,
seed 42, all defaults otherwise); evaluated with
`python evaluate.py --model city1 --steps 500`. Training reduced total
waiting time per episode from ~1.84M s (random policy) to ~550K s, and the
greedy agent beats the fixed-time round-robin controller on the same seeded
traffic:

| Map | Junctions | Lanes/direction | Fixed-time waiting (s) | DQN waiting (s) | Improvement |
|---|---|---|---|---|---|
| city1 | 5 | 1 | 688,596 | 576,228 | **+16.3%** |
| corridor | 2 | 3 | 236,144 | 153,321 | **+35.1%** |
| city2 | 4 | 1 | _train to reproduce_ | | |
| city3 | 4 | 1 | _train to reproduce_ | | |

Runs are seeded (`--seed`, default 42) so numbers are directly comparable.
Train and evaluate the corridor scenario with:

```bash
python train.py --config maps/corridor.sumocfg --model corridor --epochs 50 --steps 500
python evaluate.py --config maps/corridor.sumocfg --model corridor --steps 500
```

### Note on demand calibration

The corridor scenario initially showed *no* learning — a flat training curve and
a result far worse than the baseline. The cause was traffic demand, not the
agent: at one vehicle every 0.7 s the two junctions could not discharge what
was being inserted, so vehicles accumulated indefinitely (168 queued, zero
completed trips) and total waiting time was dominated by gridlock rather than
by signal policy. With demand rescaled to one vehicle every 2 s — congested
enough to matter (40 vehicles queued at peak) but still clearing 210 trips per
episode — the agent learns normally, cutting episode waiting time from ~630K to
~155K. Signal control can only help a network that is under capacity; past
saturation, no policy recovers it.

## Acknowledgments

This project is a ground-up rebuild of
[RekhaChittaloori/traffic-optimization-rl](https://github.com/RekhaChittaloori/traffic-optimization-rl),
which provided the original concept and the SUMO city networks in `maps/`.
The RL implementation here is rewritten: the original trained without
exploration (ε started at 0), replayed the full buffer in order instead of
sampling minibatches, never used its target network, and skipped the yellow
phase; this version fixes all of that and adds seeding, a fixed-time baseline,
and a reproducible evaluation harness.

## Author

**Soman Gaurav** — [GitHub](https://github.com/SomanGaurav)
