"""
Actor in system memory (RAM) that stores SAC replay buffer data using NumPy arrays.
"""

import ray
import numpy as np

@ray.remote
class SACReplayBuffer:
    def __init__(self, capacity: int, seed: int): 
        self.capacity = capacity
        self._buffer = {}
        self._ptr = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def add_batch(self, **batches):
        """
        Adds a batch of transitions to the buffer.
        
        Args:
            **batches: Key-value pairs where each value is a NumPy array 
                       of shape (batch_size, ...).
        """
        if not batches:
            return

        assert "observations" in batches, "Observations must be present in the batch"
        assert "next_observations" in batches, "Next observations must be present in the batch"
        assert "actions" in batches, "Actions must be present in the batch"
        assert "rewards" in batches, "Rewards must be present in the batch"
        assert "terminateds" in batches, "Terminateds must be present in the batch"
        assert "truncateds" in batches, "Truncateds must be present in the batch"

        first_key = next(iter(batches))
        batch_size = len(batches[first_key])

        if not self._buffer:
            # Initialize buffers based on the first batch received
            for key, value in batches.items():
                val_arr = np.array(value)
                self._buffer[key] = np.zeros((self.capacity, *val_arr.shape[1:]), dtype=val_arr.dtype)
        
        # Calculate how many items to add before wrap-around
        end_ptr = self._ptr + batch_size
        if end_ptr <= self.capacity:
            # No wrap-around needed
            for key, value in batches.items():
                self._buffer[key][self._ptr:end_ptr] = value
        else:
            # Wrap-around: fill to the end, then start from the beginning
            overflow = end_ptr - self.capacity
            num_to_end = self.capacity - self._ptr
            for key, value in batches.items():
                self._buffer[key][self._ptr:] = value[:num_to_end]
                self._buffer[key][:overflow] = value[num_to_end:]
            
        self._ptr = (self._ptr + batch_size) % self.capacity
        self._size = min(self._size + batch_size, self.capacity)

    def sample(self, batch_size: int):
        """
        Samples a batch of transitions from the buffer.
        
        Returns:
            A dictionary where each key is a transition field and 
            each value is a NumPy array of shape (batch_size, ...).
        """
        if self._size == 0:
            return {}
            
        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        return {key: arr[indices] for key, arr in self._buffer.items()}

    def __len__(self):
        return self._size
