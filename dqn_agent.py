"""Deep Q-Network agent for adaptive traffic signal control.

One shared Q-network controls every junction; each junction keeps its own
replay buffer so transitions from different intersections don't interleave
within a single minibatch.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QNetwork(nn.Module):
    """MLP mapping a per-junction state (queue lengths) to Q-values per phase."""

    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.out = nn.Linear(fc2_dims, n_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.out(x)


class ReplayBuffer:
    """Fixed-size ring buffer of (s, a, r, s', done) transitions."""

    def __init__(self, capacity, input_dims):
        self.capacity = capacity
        self.mem_cntr = 0
        self.states = np.zeros((capacity, input_dims), dtype=np.float32)
        self.next_states = np.zeros((capacity, input_dims), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

    def __len__(self):
        return min(self.mem_cntr, self.capacity)

    def store(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.capacity
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done
        self.mem_cntr += 1

    def sample(self, batch_size, rng):
        indices = rng.choice(len(self), size=batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )


class Agent:
    def __init__(
        self,
        input_dims,
        n_actions,
        junctions,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=5e-4,
        buffer_size=100_000,
        batch_size=128,
        fc1_dims=256,
        fc2_dims=256,
        target_update=100,
        seed=None,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.target_update = target_update
        self.learn_step = 0
        self.rng = np.random.default_rng(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_policy = QNetwork(input_dims, fc1_dims, fc2_dims, n_actions).to(self.device)
        self.q_target = QNetwork(input_dims, fc1_dims, fc2_dims, n_actions).to(self.device)
        self.q_target.load_state_dict(self.q_policy.state_dict())
        self.q_target.eval()
        self.optimizer = optim.Adam(self.q_policy.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.buffers = {j: ReplayBuffer(buffer_size, input_dims) for j in junctions}

    def choose_action(self, state, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
            return int(torch.argmax(self.q_policy(state_t)).item())

    def store_transition(self, junction, state, action, reward, next_state, done):
        self.buffers[junction].store(state, action, reward, next_state, done)

    def learn(self, junction):
        buffer = self.buffers[junction]
        if len(buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = buffer.sample(self.batch_size, self.rng)
        states = torch.tensor(states, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        dones = torch.tensor(dones, device=self.device)

        q_pred = self.q_policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.q_target(next_states).max(dim=1).values
            q_next[dones] = 0.0
            q_target = rewards + self.gamma * q_next

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_policy.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.q_target.load_state_dict(self.q_policy.state_dict())
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        return float(loss.item())

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.q_policy.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.q_policy.load_state_dict(state_dict)
        self.q_target.load_state_dict(state_dict)
