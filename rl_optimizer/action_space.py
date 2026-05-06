"""Discrete resource configuration action space used by CALO."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class Configuration:
    """One serverless resource configuration."""

    memory_mb: int
    architecture: str
    timeout_sec: int

    def to_dict(self) -> Dict[str, object]:
        """Returns the configuration as a serializable dictionary."""
        return {
            "memory_mb": self.memory_mb,
            "architecture": self.architecture,
            "timeout_sec": self.timeout_sec,
        }

    def __str__(self) -> str:
        """Returns a human-readable configuration label."""
        return (
            f"mem={self.memory_mb}MB, "
            f"arch={self.architecture}, "
            f"timeout={self.timeout_sec}s"
        )


class ActionSpace:
    """Discrete configuration space for the simulator and baselines."""

    DEFAULT_MEMORY_OPTIONS = [128, 256, 512, 1024, 2048, 3008]
    DEFAULT_ARCHITECTURE_OPTIONS = ["x64", "arm64"]
    DEFAULT_TIMEOUT_OPTIONS = [60, 120, 300, 900]
    PRESETS = {
        "full_48": {
            "memory_options": DEFAULT_MEMORY_OPTIONS,
            "architecture_options": DEFAULT_ARCHITECTURE_OPTIONS,
            "timeout_options": DEFAULT_TIMEOUT_OPTIONS,
        },
        "full_x64_24": {
            "memory_options": DEFAULT_MEMORY_OPTIONS,
            "architecture_options": ["x64"],
            "timeout_options": DEFAULT_TIMEOUT_OPTIONS,
        },
        "calibrated_x64_8": {
            "memory_options": [512, 1024, 2048, 3008],
            "architecture_options": ["x64"],
            "timeout_options": [120, 300],
        },
    }

    def __init__(
        self,
        memory_options: Optional[List[int]] = None,
        architecture_options: Optional[List[str]] = None,
        timeout_options: Optional[List[int]] = None,
        preset: Optional[str] = None,
    ):
        """Builds the action space from explicit lists or a preset."""
        preset_config = self._resolve_preset_config(preset)
        self.MEMORY_OPTIONS = self._normalize_sorted_ints(
            memory_options or preset_config["memory_options"]
        )
        self.ARCHITECTURE_OPTIONS = self._normalize_strings(
            architecture_options or preset_config["architecture_options"]
        )
        self.TIMEOUT_OPTIONS = self._normalize_sorted_ints(
            timeout_options or preset_config["timeout_options"]
        )
        self._validate_configuration_options()

        self.configurations: List[Configuration] = []
        self.action_to_config: Dict[int, Configuration] = {}

        action_id = 0
        for memory in self.MEMORY_OPTIONS:
            for architecture in self.ARCHITECTURE_OPTIONS:
                for timeout in self.TIMEOUT_OPTIONS:
                    config = Configuration(memory, architecture, timeout)
                    self.configurations.append(config)
                    self.action_to_config[action_id] = config
                    action_id += 1

        self.config_to_action = {
            (config.memory_mb, config.architecture, config.timeout_sec): index
            for index, config in enumerate(self.configurations)
        }

    @property
    def n_actions(self) -> int:
        """Returns the number of valid actions."""
        return len(self.configurations)

    def sample(self) -> int:
        """Samples a random valid action."""
        return int(np.random.randint(0, self.n_actions))

    def get_configuration(self, action: int) -> Configuration:
        """Returns the resource configuration for one action id."""
        if action < 0 or action >= self.n_actions:
            raise ValueError(
                f"Action id must be in [0, {self.n_actions}); received {action}"
            )
        return self.action_to_config[action]

    def get_action_id(
        self,
        memory_mb: int,
        architecture: str,
        timeout_sec: int,
    ) -> int:
        """Returns the action id for a configuration or its nearest neighbor."""
        key = (int(memory_mb), str(architecture), int(timeout_sec))
        if key in self.config_to_action:
            return self.config_to_action[key]
        return self._find_closest_action(
            memory_mb=int(memory_mb),
            architecture=str(architecture),
            timeout_sec=int(timeout_sec),
        )

    def get_all_configurations(self) -> List[Configuration]:
        """Returns a shallow copy of all configurations."""
        return self.configurations.copy()

    def encode_action(self, action: int) -> np.ndarray:
        """Encodes one action as concatenated one-hot vectors."""
        config = self.get_configuration(action)
        vector = np.zeros(
            len(self.MEMORY_OPTIONS)
            + len(self.ARCHITECTURE_OPTIONS)
            + len(self.TIMEOUT_OPTIONS),
            dtype=np.float32,
        )

        memory_idx = self.MEMORY_OPTIONS.index(config.memory_mb)
        vector[memory_idx] = 1.0

        arch_idx = self.ARCHITECTURE_OPTIONS.index(config.architecture)
        arch_offset = len(self.MEMORY_OPTIONS)
        vector[arch_offset + arch_idx] = 1.0

        timeout_idx = self.TIMEOUT_OPTIONS.index(config.timeout_sec)
        timeout_offset = arch_offset + len(self.ARCHITECTURE_OPTIONS)
        vector[timeout_offset + timeout_idx] = 1.0
        return vector

    def decode_action(self, vector: np.ndarray) -> int:
        """Decodes a concatenated one-hot vector back to an action id."""
        vector = np.asarray(vector, dtype=np.float32)
        memory_end = len(self.MEMORY_OPTIONS)
        arch_end = memory_end + len(self.ARCHITECTURE_OPTIONS)

        memory_mb = self.MEMORY_OPTIONS[int(np.argmax(vector[:memory_end]))]
        architecture = self.ARCHITECTURE_OPTIONS[
            int(np.argmax(vector[memory_end:arch_end]))
        ]
        timeout_sec = self.TIMEOUT_OPTIONS[int(np.argmax(vector[arch_end:]))]
        return self.get_action_id(memory_mb, architecture, timeout_sec)

    def get_action_mask(self, constraints: Optional[Dict[str, object]] = None) -> np.ndarray:
        """Returns a boolean mask for valid actions under optional constraints."""
        mask = np.ones(self.n_actions, dtype=bool)
        if constraints is None:
            return mask

        for action_id, config in enumerate(self.configurations):
            if "max_memory" in constraints:
                if config.memory_mb > int(constraints["max_memory"]):
                    mask[action_id] = False
                    continue

            if "min_memory" in constraints:
                if config.memory_mb < int(constraints["min_memory"]):
                    mask[action_id] = False
                    continue

            if "allowed_arch" in constraints:
                allowed_arch = {str(value) for value in constraints["allowed_arch"]}
                if config.architecture not in allowed_arch:
                    mask[action_id] = False
                    continue

            if "max_timeout" in constraints:
                if config.timeout_sec > int(constraints["max_timeout"]):
                    mask[action_id] = False
                    continue

            if "min_timeout" in constraints:
                if config.timeout_sec < int(constraints["min_timeout"]):
                    mask[action_id] = False
                    continue

        return mask

    def get_nearby_actions(self, action: int, radius: int = 1) -> List[int]:
        """Returns neighboring actions for local search and exploration."""
        config = self.get_configuration(action)

        memory_idx = self.MEMORY_OPTIONS.index(config.memory_mb)
        timeout_idx = self.TIMEOUT_OPTIONS.index(config.timeout_sec)

        memory_neighbors = self.MEMORY_OPTIONS[
            max(0, memory_idx - radius): min(
                len(self.MEMORY_OPTIONS), memory_idx + radius + 1
            )
        ]
        timeout_neighbors = self.TIMEOUT_OPTIONS[
            max(0, timeout_idx - radius): min(
                len(self.TIMEOUT_OPTIONS), timeout_idx + radius + 1
            )
        ]

        nearby_actions = set()
        for memory in memory_neighbors:
            for architecture in self.ARCHITECTURE_OPTIONS:
                for timeout in timeout_neighbors:
                    action_id = self.get_action_id(memory, architecture, timeout)
                    if action_id != action:
                        nearby_actions.add(action_id)
        return sorted(nearby_actions)

    def print_action_space(self) -> None:
        """Prints a compact summary of the current action space."""
        print("=" * 60)
        print("Action Space")
        print("=" * 60)
        print(f"Total actions: {self.n_actions}")
        print(f"Memory options: {self.MEMORY_OPTIONS}")
        print(f"Architecture options: {self.ARCHITECTURE_OPTIONS}")
        print(f"Timeout options: {self.TIMEOUT_OPTIONS}")
        print("\nFirst configurations:")
        for index in range(min(10, self.n_actions)):
            print(f"  Action {index}: {self.get_configuration(index)}")
        print("  ...")
        print("=" * 60)

    def get_default_action(self) -> int:
        """Returns the default action or the nearest calibrated fallback."""
        return self.get_action_id(128, "x64", 60)

    def get_stats(self) -> Dict[str, object]:
        """Returns basic action-space statistics."""
        return {
            "n_actions": self.n_actions,
            "n_memory_options": len(self.MEMORY_OPTIONS),
            "n_architecture_options": len(self.ARCHITECTURE_OPTIONS),
            "n_timeout_options": len(self.TIMEOUT_OPTIONS),
            "memory_range": (min(self.MEMORY_OPTIONS), max(self.MEMORY_OPTIONS)),
            "timeout_range": (min(self.TIMEOUT_OPTIONS), max(self.TIMEOUT_OPTIONS)),
        }

    def _find_closest_action(
        self,
        memory_mb: int,
        architecture: str,
        timeout_sec: int,
    ) -> int:
        """Returns the nearest available configuration in the current space."""
        closest_memory = min(self.MEMORY_OPTIONS, key=lambda value: abs(value - memory_mb))
        if architecture not in self.ARCHITECTURE_OPTIONS:
            architecture = self.ARCHITECTURE_OPTIONS[0]
        closest_timeout = min(
            self.TIMEOUT_OPTIONS,
            key=lambda value: abs(value - timeout_sec),
        )
        key = (closest_memory, architecture, closest_timeout)
        return self.config_to_action[key]

    def _resolve_preset_config(self, preset: Optional[str]) -> Dict[str, List[object]]:
        """Returns the configuration lists behind a named preset."""
        preset_name = preset or "full_48"
        if preset_name not in self.PRESETS:
            raise ValueError(
                f"Unknown action-space preset: {preset_name}. "
                f"Available presets: {sorted(self.PRESETS.keys())}"
            )
        preset_config = self.PRESETS[preset_name]
        return {
            "memory_options": list(preset_config["memory_options"]),
            "architecture_options": list(preset_config["architecture_options"]),
            "timeout_options": list(preset_config["timeout_options"]),
        }

    def _normalize_sorted_ints(self, values: List[int]) -> List[int]:
        """Returns sorted unique integer options."""
        return sorted({int(value) for value in values})

    def _normalize_strings(self, values: List[str]) -> List[str]:
        """Returns unique string options while preserving order."""
        normalized: List[str] = []
        seen = set()
        for value in values:
            current = str(value)
            if current in seen:
                continue
            normalized.append(current)
            seen.add(current)
        return normalized

    def _validate_configuration_options(self) -> None:
        """Validates that every configuration dimension is non-empty."""
        if not self.MEMORY_OPTIONS:
            raise ValueError("Action space requires at least one memory option")
        if not self.ARCHITECTURE_OPTIONS:
            raise ValueError("Action space requires at least one architecture option")
        if not self.TIMEOUT_OPTIONS:
            raise ValueError("Action space requires at least one timeout option")


def test_action_space() -> None:
    """Runs a simple interactive self-test."""
    print("\nTesting action space\n")

    action_space = ActionSpace()
    action_space.print_action_space()

    print("\nSampling five random actions:")
    for _ in range(5):
        action = action_space.sample()
        print(f"  Action {action}: {action_space.get_configuration(action)}")

    print("\nTesting encode/decode:")
    action = min(10, action_space.n_actions - 1)
    print(f"  Original action: {action} -> {action_space.get_configuration(action)}")
    vector = action_space.encode_action(action)
    print(f"  Encoded vector: {vector}")
    decoded_action = action_space.decode_action(vector)
    print(
        "  Decoded action: "
        f"{decoded_action} -> {action_space.get_configuration(decoded_action)}"
    )

    print("\nTesting action mask with max_memory=1024:")
    mask = action_space.get_action_mask({"max_memory": 1024})
    print(f"  Available actions: {int(mask.sum())} / {action_space.n_actions}")

    print("\nTesting neighboring actions:")
    action = 0
    print(f"  Current action: {action} -> {action_space.get_configuration(action)}")
    nearby = action_space.get_nearby_actions(action, radius=1)
    print(f"  Neighbor count: {len(nearby)}")
    for nearby_action in nearby[:5]:
        print(f"    Action {nearby_action}: {action_space.get_configuration(nearby_action)}")


if __name__ == "__main__":
    test_action_space()
