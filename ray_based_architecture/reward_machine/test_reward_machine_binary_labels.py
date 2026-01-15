"""
Tests for RewardMachine with binary vector labels and transition matrix.
"""

import numpy as np
import pytest

# Import RewardMachine and DeterministicRMTransitioner
from ray_based_architecture.reward_machine.reward_machine import RewardMachine
from ray_based_architecture.reward_machine.transition.deterministic_rm_transitioner import DeterministicRMTransitioner


class TestRewardMachineBinaryLabels:
    """Test suite for binary vector label transitions using transition matrix."""

    def _get_next_state_name(self, transitioner, state_name, label_vector):
        """Helper to get next state name from transitioner."""
        # Convert to batch format (add batch dimension)
        curr_state_one_hot = transitioner._to_one_hot(state_name)[np.newaxis, :]
        label_vector_batch = label_vector[np.newaxis, :]
        
        # Get next state (returns batch)
        next_state_one_hot = transitioner.get_next_state(curr_state_one_hot, label_vector_batch)
        
        # Extract single result
        next_state_idx = np.argmax(next_state_one_hot[0])
        return transitioner.rm.states[next_state_idx]

    def test_simple_two_state_machine(self):
        """Test a simple machine with two states and one label."""
        rm = RewardMachine()
        rm.add_states(["u0", "u1"])
        rm.set_u0("u0")
        
        # Label 0 corresponds to "P"
        # Transition: u0 --(P)--> u1
        rm.add_transition("u0", "u1", "P")
        
        # Build transition matrix with label order ["P"]
        rm.build_transition_matrix(["P"])
        
        # Create transitioner
        transitioner = DeterministicRMTransitioner(rm, num_envs=1)
        
        # Test: label vector [1] (P is true) should transition u0 -> u1
        next_state = self._get_next_state_name(transitioner, "u0", np.array([1]))
        assert next_state == "u1", f"Expected u1, got {next_state}"
        
        # Test: label vector [0] (P is false) should stay in u0
        next_state = self._get_next_state_name(transitioner, "u0", np.array([0]))
        assert next_state == "u0", f"Expected u0, got {next_state}"

    def test_three_labels_machine(self):
        """Test machine with multiple labels."""
        rm = RewardMachine()
        rm.add_states(["u0", "u1", "u2"])
        rm.set_u0("u0")
        
        # Label order: ["P", "Q", "R"]
        # Transitions:
        # u0 --(P)--> u1
        # u1 --(Q)--> u2
        rm.add_transition("u0", "u1", "P")
        rm.add_transition("u1", "u2", "Q")
        
        rm.build_transition_matrix(["P", "Q", "R"])
        transitioner = DeterministicRMTransitioner(rm, num_envs=1)
        
        # Test: [1, 0, 0] (P=true) should transition u0 -> u1
        next_state = self._get_next_state_name(transitioner, "u0", np.array([1, 0, 0]))
        assert next_state == "u1"
        
        # Test: [0, 1, 0] (Q=true) should transition u1 -> u2
        next_state = self._get_next_state_name(transitioner, "u1", np.array([0, 1, 0]))
        assert next_state == "u2"
        
        # Test: [1, 1, 0] (P and Q true) from u0 should go to u1 (only P matters)
        next_state = self._get_next_state_name(transitioner, "u0", np.array([1, 1, 0]))
        assert next_state == "u1"

    def test_negated_labels(self):
        """Test machine with negated labels (~P means P is false)."""
        rm = RewardMachine()
        rm.add_states(["u0", "u1", "u2"])
        rm.set_u0("u0")
        
        # Label order: ["P", "Q"]
        # Transitions:
        # u0 --(P)--> u1
        # u0 --(~P)--> u2
        rm.add_transition("u0", "u1", "P")
        rm.add_transition("u0", "u2", "~P")
        
        rm.build_transition_matrix(["P", "Q"])
        transitioner = DeterministicRMTransitioner(rm, num_envs=1)
        
        # Test: [1, 0] (P=true) should transition u0 -> u1
        next_state = self._get_next_state_name(transitioner, "u0", np.array([1, 0]))
        assert next_state == "u1"
        
        # Test: [0, 0] (P=false, so ~P is true) should transition u0 -> u2
        next_state = self._get_next_state_name(transitioner, "u0", np.array([0, 0]))
        assert next_state == "u2"
        
        # Test: [0, 1] (P=false, Q=true, so ~P is true) should transition u0 -> u2
        next_state = self._get_next_state_name(transitioner, "u0", np.array([0, 1]))
        assert next_state == "u2"
        
        # Test: [1, 1] (P=true, Q=true) should transition u0 -> u1 (P takes precedence)
        next_state = self._get_next_state_name(transitioner, "u0", np.array([1, 1]))
        assert next_state == "u1"

    def test_multiple_negated_labels(self):
        """Test machine with multiple negated labels."""
        rm = RewardMachine()
        rm.add_states(["u0", "u1", "u2", "u3"])
        rm.set_u0("u0")
        
        # Label order: ["P", "Q", "R"]
        # Transitions:
        # u0 --(~P)--> u1
        # u0 --(~Q)--> u2
        # u0 --(~R)--> u3
        rm.add_transition("u0", "u1", "~P")
        rm.add_transition("u0", "u2", "~Q")
        rm.add_transition("u0", "u3", "~R")
        
        rm.build_transition_matrix(["P", "Q", "R"])
        transitioner = DeterministicRMTransitioner(rm, num_envs=1)
        
        # Test: [0, 1, 1] (P=false, Q=true, R=true) should transition u0 -> u1 (~P matches first)
        next_state = self._get_next_state_name(transitioner, "u0", np.array([0, 1, 1]))
        assert next_state == "u1"
        
        # Test: [1, 0, 1] (P=true, Q=false, R=true) should transition u0 -> u2 (~Q matches)
        next_state = self._get_next_state_name(transitioner, "u0", np.array([1, 0, 1]))
        assert next_state == "u2"
        
        # Test: [1, 1, 0] (P=true, Q=true, R=false) should transition u0 -> u3 (~R matches)
        next_state = self._get_next_state_name(transitioner, "u0", np.array([1, 1, 0]))
        assert next_state == "u3"
        
        # Test: [0, 0, 0] (all false) should match ~P first
        next_state = self._get_next_state_name(transitioner, "u0", np.array([0, 0, 0]))
        assert next_state == "u1"

    def test_mixed_positive_and_negated_labels(self):
        """Test transitions with both positive and negated labels."""
        rm = RewardMachine()
        rm.add_states(["u0", "u1", "u2", "u3"])
        rm.set_u0("u0")
        
        # Label order: ["P", "Q"]
        # Transitions:
        # u0 --(P)--> u1
        # u0 --(~P)--> u2
        # u0 --(Q)--> u3
        # u0 --(~Q)--> u1
        rm.add_transition("u0", "u1", "P")
        rm.add_transition("u0", "u2", "~P")
        rm.add_transition("u0", "u3", "Q")
        rm.add_transition("u0", "u1", "~Q")
        
        rm.build_transition_matrix(["P", "Q"])
        transitioner = DeterministicRMTransitioner(rm, num_envs=1)
        
        # Test: [1, 0] (P=true, Q=false) should transition u0 -> u1 (P matches first)
        next_state = self._get_next_state_name(transitioner, "u0", np.array([1, 0]))
        assert next_state == "u1"
        
        # Test: [0, 0] (P=false, Q=false) should transition u0 -> u2 (~P matches first)
        next_state = self._get_next_state_name(transitioner, "u0", np.array([0, 0]))
        assert next_state == "u2"
        
        # Test: [0, 1] (P=false, Q=true) should transition u0 -> u2 (~P matches first, before Q)
        # Note: Both ~P and Q match, but transitions are checked in order, so ~P wins
        next_state = self._get_next_state_name(transitioner, "u0", np.array([0, 1]))
        assert next_state == "u2"

    def test_negated_labels_with_tuple_conditions(self):
        """Test negated labels in tuple conditions (AND of conditions)."""
        rm = RewardMachine()
        rm.add_states(["u0", "u1", "u2"])
        rm.set_u0("u0")
        
        # Label order: ["P", "Q"]
        # Transitions:
        # u0 --(P, ~Q)--> u1  (P AND not Q)
        # u0 --(~P, Q)--> u2  (not P AND Q)
        rm.add_transition("u0", "u1", ("P", "~Q"))
        rm.add_transition("u0", "u2", ("~P", "Q"))
        
        rm.build_transition_matrix(["P", "Q"])
        transitioner = DeterministicRMTransitioner(rm, num_envs=1)
        
        # Test: [1, 0] (P=true, Q=false) should transition u0 -> u1 (P AND ~Q)
        next_state = self._get_next_state_name(transitioner, "u0", np.array([1, 0]))
        assert next_state == "u1"
        
        # Test: [0, 1] (P=false, Q=true) should transition u0 -> u2 (~P AND Q)
        next_state = self._get_next_state_name(transitioner, "u0", np.array([0, 1]))
        assert next_state == "u2"
        
        # Test: [1, 1] (P=true, Q=true) should stay in u0 (neither condition matches)
        next_state = self._get_next_state_name(transitioner, "u0", np.array([1, 1]))
        assert next_state == "u0"
        
        # Test: [0, 0] (P=false, Q=false) should stay in u0 (neither condition matches)
        next_state = self._get_next_state_name(transitioner, "u0", np.array([0, 0]))
        assert next_state == "u0"

    def test_accepting_state_transition(self):
        """Test transitions to accepting state."""
        rm = RewardMachine()
        rm.add_states(["u0", "uacc"])
        rm.set_u0("u0")
        rm.set_uacc("uacc")
        
        # Transition: u0 --(P)--> uacc
        rm.add_transition("u0", "uacc", "P")
        
        rm.build_transition_matrix(["P"])
        transitioner = DeterministicRMTransitioner(rm, num_envs=1)
        
        # Label vector [1] (P=true) should transition to accepting state
        next_state = self._get_next_state_name(transitioner, "u0", np.array([1]))
        assert next_state == "uacc"
        
        # Label vector [0] (P=false) should stay in u0
        next_state = self._get_next_state_name(transitioner, "u0", np.array([0]))
        assert next_state == "u0"

    def test_no_transition_available(self):
        """Test when no transition matches the label vector."""
        rm = RewardMachine()
        rm.add_states(["u0", "u1"])
        rm.set_u0("u0")
        
        # Only transition: u0 --(P)--> u1
        rm.add_transition("u0", "u1", "P")
        
        rm.build_transition_matrix(["P", "Q"])
        transitioner = DeterministicRMTransitioner(rm, num_envs=1)
        
        # Test: [0, 1] (P=false, Q=true) should stay in u0
        next_state = self._get_next_state_name(transitioner, "u0", np.array([0, 1]))
        assert next_state == "u0"

    def test_multiple_possible_transitions(self):
        """Test that transitions are deterministic (same condition can't lead to different states)."""
        rm = RewardMachine()
        rm.add_states(["u0", "u1", "u2"])
        rm.set_u0("u0")
        
        # Multiple transitions from u0 with different conditions
        rm.add_transition("u0", "u1", "P")
        rm.add_transition("u0", "u2", "Q")
        
        rm.build_transition_matrix(["P", "Q"])
        transitioner = DeterministicRMTransitioner(rm, num_envs=1)
        
        # P=true should go to u1
        next_state = self._get_next_state_name(transitioner, "u0", np.array([1, 0]))
        assert next_state == "u1"
        
        # Q=true should go to u2
        next_state = self._get_next_state_name(transitioner, "u0", np.array([0, 1]))
        assert next_state == "u2"
        
        # Both true - P should be checked first (order in transition_matrix)
        next_state = self._get_next_state_name(transitioner, "u0", np.array([1, 1]))
        assert next_state == "u1"

    def test_transition_matrix_shape(self):
        """Test that transition data structures have correct shape."""
        rm = RewardMachine()
        rm.add_states(["u0", "u1", "u2"])
        rm.set_u0("u0")
        rm.add_transition("u0", "u1", "P")
        rm.add_transition("u1", "u2", "Q")
        
        label_order = ["P", "Q", "R"]
        rm.build_transition_matrix(label_order)
        
        # Check that transition data structures exist and have correct shapes
        num_states = 3
        num_labels = 3
        
        assert hasattr(rm, 'transition_requirements')
        assert rm.transition_requirements is not None
        assert rm.transition_requirements.shape == (num_states, num_labels, num_states)
        assert rm.transition_requirements.dtype == np.int8

    def test_label_order_consistency(self):
        """Test that label order is stored and used consistently."""
        rm = RewardMachine()
        rm.add_states(["u0", "u1"])
        rm.set_u0("u0")
        rm.add_transition("u0", "u1", "P")
        
        label_order = ["P", "Q"]
        rm.build_transition_matrix(label_order)
        
        assert hasattr(rm, 'label_order')
        assert rm.label_order == label_order
        
        # Label vector length should match label_order length
        transitioner = DeterministicRMTransitioner(rm, num_envs=1)
        with pytest.raises(ValueError):
            curr_state_one_hot = transitioner._to_one_hot("u0")
            transitioner.get_next_state(curr_state_one_hot, np.array([1]))  # Wrong length

    def test_batch_transitions(self):
        """Test batch processing of label vectors."""
        rm = RewardMachine()
        rm.add_states(["u0", "u1"])
        rm.set_u0("u0")
        rm.add_transition("u0", "u1", "P")
        
        rm.build_transition_matrix(["P", "Q"])
        transitioner = DeterministicRMTransitioner(rm, num_envs=3)
        
        # Batch of label vectors
        label_vectors = np.array([
            [1, 0],  # P=true -> u1
            [0, 0],  # P=false -> u0
            [1, 1],  # P=true -> u1
        ])
        
        # Batch of current states (all u0)
        curr_states_one_hot = np.repeat(
            np.expand_dims(transitioner._to_one_hot("u0"), axis=0), 
            3, 
            axis=0
        )
        
        next_states_one_hot = transitioner.get_next_state(curr_states_one_hot, label_vectors)
        next_state_indices = np.argmax(next_states_one_hot, axis=1)
        next_states = [rm.states[idx] for idx in next_state_indices]
        
        assert len(next_states) == 3
        assert next_states[0] == "u1"
        assert next_states[1] == "u0"
        assert next_states[2] == "u1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
