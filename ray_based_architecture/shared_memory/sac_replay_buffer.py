"""
Actor in system memory (RAM) that stores SAC replay buffer data using NumPy arrays.

Optimized for HIGH-PRIORITY sample() (training-critical) and LOW-PRIORITY add_batch() (async writes).
"""
# TODO: Review replay buffer yourself. I am not happy with how it is implemented.

import ray
import numpy as np
import threading
import time

@ray.remote
class SACReplayBuffer:
    def __init__(self, capacity: int, seed: int): 
        self.capacity = capacity
        self._buffer = {}
        self._ptr = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)
        
        # Priority-aware locking: sample() can "cut in line" ahead of add_batch()
        self._lock = threading.Lock()
        self._sample_waiting = threading.Condition(self._lock)
        self._num_waiting_samples = 0  # Counter for high-priority sample() calls
        self._num_active_writes = 0    # Counter for active add_batch() operations

    def add_batch(self, **batches):
        """
        LOW PRIORITY: Fire-and-forget writes from VLM service.
        Yields to sample() if it's waiting.
        
        Args:
            **batches: Key-value pairs where each value is a NumPy array 
                       of shape (batch_size, ...) or a Dict of arrays.
        """
        if not batches:
            return

        # Handle Dict observations by flattening them
        processed_batches = {}
        for k, v in batches.items():
            if isinstance(v, dict):
                # Flatten Dict into separate keys
                for sub_key, sub_val in v.items():
                    processed_batches[f"{k}_{sub_key}"] = np.asarray(sub_val)
            else:
                processed_batches[k] = np.asarray(v)
        
        if not processed_batches:
            return
        
        # Get batch size from first array
        first_arr = next(iter(processed_batches.values()))
        batch_size = 1 if first_arr.ndim == 0 else int(first_arr.shape[0])

        # Priority mechanism: wait if sample() is waiting (give it priority)
        with self._sample_waiting:
            # Yield to any waiting sample() calls
            while self._num_waiting_samples > 0:
                self._sample_waiting.wait()
            
            self._num_active_writes += 1
            
            # Initialize buffer if needed
            if not self._buffer:
                for key, value in processed_batches.items():
                    shape = (self.capacity,) if value.ndim == 0 else (self.capacity, *value.shape[1:])
                    self._buffer[key] = np.empty(shape, dtype=value.dtype)

            # Reserve write slots
            start = int(self._ptr)
            end_ptr = start + batch_size
            self._ptr = (start + batch_size) % self.capacity
            self._size = min(self._size + batch_size, self.capacity)
        
        # Perform numpy writes outside lock (concurrent with other add_batch calls)
        for key, value in processed_batches.items():
            if end_ptr <= self.capacity:
                self._buffer[key][start:end_ptr] = value
            else:
                # Handle wrap-around
                num_to_end = self.capacity - start
                self._buffer[key][start:] = value[:num_to_end]
                self._buffer[key][:end_ptr - self.capacity] = value[num_to_end:]
        
        # Signal completion
        with self._sample_waiting:
            self._num_active_writes -= 1
            if self._num_active_writes == 0 and self._num_waiting_samples > 0:
                self._sample_waiting.notify_all()

    def sample(self, batch_size: int):
        """
        HIGH PRIORITY: Training loop blocks on this.
        Gets priority over add_batch() calls.
        
        Returns:
            A dictionary where each key is a transition field and 
            each value is a NumPy array of shape (batch_size, ...).
        """
        start_time = time.time()
        
        # Priority mechanism: signal that we're waiting, make add_batch() yield
        with self._sample_waiting:
            self._num_waiting_samples += 1
            
            # Wait for any active add_batch() to finish their writes
            wait_start = time.time()
            while self._num_active_writes > 0:
                self._sample_waiting.wait()
            wait_time_ms = (time.time() - wait_start) * 1000
            
            # Log if we waited a long time to acquire the lock
            if wait_time_ms > 10:
                print(f"[SACReplayBuffer.sample] Waited {wait_time_ms:.1f}ms to acquire lock "
                      f"({self._num_active_writes} active writes)", flush=True)
            
            # Now we have exclusive access
            current_size = self._size
            
            if current_size == 0:
                self._num_waiting_samples -= 1
                self._sample_waiting.notify_all()
                return {}
            
            # Sample indices and copy data atomically
            actual_batch_size = min(batch_size, current_size)
            indices = self._rng.choice(current_size, size=actual_batch_size, replace=True)
            
            # Group keys by prefix to reconstruct Dict observations
            dict_keys = {}  # prefix -> {sub_key: buffer_key}
            simple_keys = []
            
            for key in self._buffer.keys():
                if key.startswith('next_observations_'):
                    prefix = 'next_observations'
                    sub_key = key[len('next_observations_'):]
                    dict_keys.setdefault(prefix, {})[sub_key] = key
                elif key.startswith('observations_'):
                    prefix = 'observations'
                    sub_key = key[len('observations_'):]
                    dict_keys.setdefault(prefix, {})[sub_key] = key
                else:
                    simple_keys.append(key)
            
            # Copy simple fields
            result = {key: np.take(self._buffer[key], indices, axis=0) for key in simple_keys}
            
            # Reconstruct Dict observations
            for prefix, sub_keys in dict_keys.items():
                result[prefix] = {
                    sub_key: np.take(self._buffer[buffer_key], indices, axis=0)
                    for sub_key, buffer_key in sub_keys.items()
                }
            
            # Release priority
            self._num_waiting_samples -= 1
            self._sample_waiting.notify_all()
            
            # Log slow operations
            total_time_ms = (time.time() - start_time) * 1000
            if total_time_ms > 50:
                print(f"[SACReplayBuffer.sample] Total: {total_time_ms:.1f}ms "
                      f"(wait: {wait_time_ms:.1f}ms)", flush=True)
            
            return result

    def __len__(self):
        with self._sample_waiting:
            return self._size
