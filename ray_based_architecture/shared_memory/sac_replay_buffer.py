"""
Actor in system memory (RAM) that stores SAC replay buffer data using NumPy arrays.
"""

import ray
import numpy as np
import threading

@ray.remote
class SACReplayBuffer:
    def __init__(self, capacity: int, seed: int): 
        self.capacity = capacity
        self._buffer = {}
        self._ptr = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)
        # Protects ring-buffer pointer/size updates and sampling consistency.
        self._lock = threading.Lock()

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

        # Normalize inputs once (keeps dtype/shape consistent and avoids repeated conversions).
        arrs = {k: np.asarray(v) for k, v in batches.items()}
        first_key = next(iter(arrs))
        batch_size = int(arrs[first_key].shape[0])

        # Reserve write regions (disjoint) under lock so multiple calls can safely overlap.
        with self._lock:
            if not self._buffer:
                # Initialize buffers based on the first batch received
                for key, value in arrs.items():
                    self._buffer[key] = np.zeros((self.capacity, *value.shape[1:]), dtype=value.dtype)

            start = int(self._ptr)
            end_ptr = start + batch_size
            self._ptr = (start + batch_size) % self.capacity
            self._size = min(self._size + batch_size, self.capacity)

        if end_ptr <= self.capacity:
            # No wrap-around
            sl = slice(start, end_ptr)
            for key, value in arrs.items():
                np.copyto(self._buffer[key][sl], value, casting="no")
        else:
            # Wrap-around: fill to the end, then start from the beginning
            num_to_end = self.capacity - start
            overflow = end_ptr - self.capacity
            sl1 = slice(start, self.capacity)
            sl2 = slice(0, overflow)
            for key, value in arrs.items():
                np.copyto(self._buffer[key][sl1], value[:num_to_end], casting="no")
                np.copyto(self._buffer[key][sl2], value[num_to_end:], casting="no")

    def sample(self, batch_size: int):
        """
        Samples a batch of transitions from the buffer.
        
        Returns:
            A dictionary where each key is a transition field and 
            each value is a NumPy array of shape (batch_size, ...).
        """
        with self._lock:
            if self._size == 0:
                return {}
            # Guard against requesting more than we have.
            bs = int(min(batch_size, self._size))
            indices = self._rng.choice(self._size, size=bs, replace=False)
            # Copy so training sees a consistent snapshot even if writers run concurrently.
            return {key: arr[indices].copy() for key, arr in self._buffer.items()}

    def __len__(self):
        with self._lock:
            return self._size
