"""
Agent classes for Experiment 2: The Generator Hierarchy.

Level 0 (FixedAgent): Hardcoded behavior — a single idea.
Level 1/2 (NeuralAgent): Parameterized neural network with an Architecture
    that defines its representational space.
"""

import numpy as np
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class Architecture:
    """
    Defines a neural agent's representational space.

    This is the *generator structure* — the thing that Level 1 cannot change
    but Level 2 can evolve. It determines what the agent can perceive,
    how it processes information, and what actions it can take.
    """
    input_channels: list[str] = field(default_factory=lambda: ["food", "hazard"])
    hidden_sizes: list[int] = field(default_factory=lambda: [16, 8])
    activation: str = "tanh"
    n_actions: int = 5  # 5 = cardinal + stay, 9 = + diagonals

    def obs_size(self) -> int:
        """Total observation size based on input channels."""
        sizes = {"food": 3, "hazard": 3, "density": 2, "memory": 2, "gradient": 2}
        return sum(sizes[ch] for ch in self.input_channels)

    def parameter_count(self) -> int:
        """Total number of trainable parameters."""
        sizes = [self.obs_size()] + self.hidden_sizes + [self.n_actions]
        total = 0
        for i in range(len(sizes) - 1):
            total += sizes[i] * sizes[i + 1]  # weights
            total += sizes[i + 1]              # biases
        return total

    def to_dict(self) -> dict:
        return {
            "input_channels": self.input_channels,
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
            "n_actions": self.n_actions,
            "obs_size": self.obs_size(),
            "parameter_count": self.parameter_count(),
        }

    def label(self) -> str:
        ch = "+".join(self.input_channels)
        layers = "x".join(str(s) for s in self.hidden_sizes)
        return f"{ch}|{layers}|act{self.n_actions}"


def _activate(x: np.ndarray, name: str) -> np.ndarray:
    """Apply activation function."""
    if name == "tanh":
        return np.tanh(x)
    elif name == "relu":
        return np.maximum(0, x)
    elif name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    else:
        return np.tanh(x)


class NeuralAgent:
    """
    A neural network agent with a given Architecture.

    The architecture defines the representational space (what the agent
    can perceive and do). The weights define a specific solution within
    that space. Level 1 evolves weights; Level 2 evolves the architecture.
    """

    def __init__(self, architecture: Architecture, weights: np.ndarray | None = None):
        self.architecture = architecture
        self._build_layers()

        if weights is not None:
            self.set_weights(weights)

    def _build_layers(self):
        """Initialize layer weight/bias arrays."""
        sizes = [self.architecture.obs_size()] + self.architecture.hidden_sizes + [self.architecture.n_actions]
        self.layers = []
        for i in range(len(sizes) - 1):
            w = np.zeros((sizes[i], sizes[i + 1]))
            b = np.zeros(sizes[i + 1])
            self.layers.append((w, b))

    def get_weights(self) -> np.ndarray:
        """Flatten all weights into a single vector."""
        parts = []
        for w, b in self.layers:
            parts.append(w.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts)

    def set_weights(self, weights: np.ndarray):
        """Set weights from a flat vector."""
        idx = 0
        for i, (w, b) in enumerate(self.layers):
            w_size = w.size
            b_size = b.size
            new_w = weights[idx:idx + w_size].reshape(w.shape)
            idx += w_size
            new_b = weights[idx:idx + b_size]
            idx += b_size
            self.layers[i] = (new_w, new_b)

    def act(self, observation: np.ndarray) -> int:
        """Forward pass: observation → action index."""
        x = observation
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = _activate(x, self.architecture.activation)

        # Softmax for action selection
        x = x - x.max()
        exp_x = np.exp(x)
        probs = exp_x / (exp_x.sum() + 1e-12)

        return int(np.argmax(probs))


class FixedAgent:
    """
    Level 0 — A fixed idea: hardcoded behavior with no learning.

    Always moves toward the nearest visible food. This is a single
    solution to a specific problem, not a generator of solutions.
    """

    def __init__(self):
        self.architecture = Architecture(
            input_channels=["food"],
            hidden_sizes=[],
            n_actions=5,
        )

    def act(self, observation: np.ndarray) -> int:
        """Move toward nearest food based on dx, dy from observation."""
        # observation = [dx, dy, distance] from food channel
        dx, dy = observation[0], observation[1]

        if abs(dx) < 0.01 and abs(dy) < 0.01:
            return 4  # stay

        # Move in dominant direction
        if abs(dy) > abs(dx):
            return 0 if dy < 0 else 2  # up or down
        else:
            return 1 if dx > 0 else 3  # right or left
