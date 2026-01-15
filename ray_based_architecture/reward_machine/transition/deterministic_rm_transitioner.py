from typing import Optional

import numpy as np

from ray_based_architecture.reward_machine.reward_machine import RewardMachine
from ray_based_architecture.reward_machine.transition.rm_transitioner import RMTransitioner


class DeterministicRMTransitioner(RMTransitioner):
    # None can be provided if we are learning the RM
    def __init__(self, rm: Optional[RewardMachine], num_envs: int = 1):
        super().__init__(rm)
        self._num_envs = num_envs

    def get_initial_state(self):
        assert isinstance(self.rm.u0, (str, int))
        initial_state = self._to_one_hot(self.rm.u0)
        return np.repeat(np.expand_dims(initial_state, axis=0), self._num_envs, axis=0)

    def get_next_state(self, curr_state, label_vectors):
        """
        Get next state(s) given current state(s) and binary label vector(s).
        
        Args:
            curr_state: Current states as one-hot vectors of shape (num_envs, num_states)
            label_vectors: Binary label vectors of shape (num_envs, num_labels)
        
        Returns:
            Next states as one-hot vectors of shape (num_envs, num_states)
        """
        # Check that transition matrix is built
        if self.rm.transition_requirements is None or self.rm.label_order is None:
            raise RuntimeError(
                "Transition matrix not built. Call rm.build_transition_matrix(label_order) first."
            )
        
        # Validate shapes
        num_envs = curr_state.shape[0]
        if label_vectors.shape[0] != num_envs:
            raise ValueError(
                f"Batch size mismatch: curr_state has {num_envs} envs, "
                f"label_vectors has {label_vectors.shape[0]} envs"
            )
        if label_vectors.shape[1] != len(self.rm.label_order):
            raise ValueError(
                f"Label vector length {label_vectors.shape[1]} doesn't match "
                f"label_order length {len(self.rm.label_order)}"
            )
        
        # Select transition requirements for current states
        # curr_state @ transition_requirements: (batch, num_states) @ (num_states, num_labels, num_states)
        # Result: (batch, num_labels, num_states)
        selected_reqs = np.einsum('bn,nlm->blm', curr_state, self.rm.transition_requirements)
        
        # Check which transitions match the labels
        # For each label: requirement is satisfied if (req == -1) OR (label == req)
        labels_int = label_vectors.astype(np.int8)  # (batch, num_labels)
        matches = (selected_reqs == -1) | (labels_int[:, :, None] == selected_reqs)  # (batch, num_labels, num_states)
        
        # Valid if all label requirements match AND at least one requirement exists
        valid_transitions = np.all(matches, axis=1) & np.any(selected_reqs != -1, axis=1)  # (batch, num_states)
        
        # Select first valid transition, or stay in current state
        # If no valid transitions, current state will be selected by argmax (but we prefer to select a valid transition)
        transitions_with_fallback = valid_transitions.astype(np.float32) * 2 + curr_state  # (batch, num_states)
        next_states = np.zeros_like(curr_state)  # (batch, num_states)
        next_states[np.arange(num_envs), np.argmax(transitions_with_fallback, axis=1)] = 1.0  # argmax: (batch,)
        
        return next_states  # (batch, num_states)

    def _to_one_hot(self, u):
        idx = self.rm.to_idx(u)
        u_one_hot = np.zeros(len(self.rm.states), dtype=np.float32)
        u_one_hot[idx] = 1
        return u_one_hot