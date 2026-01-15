"""
Actor in system memory (RAM) that stores SAC replay buffer data using NumPy arrays.

Optimized for HIGH-PRIORITY sample() (training-critical) and LOW-PRIORITY add_batch() (async writes).
"""

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
                       of shape (batch_size, ...).
        """
        if not batches:
            return

        # Normalize inputs once (keeps dtype/shape consistent)
        arrs = {k: np.asarray(v) for k, v in batches.items()}
        first_key = next(iter(arrs))
        batch_size = int(arrs[first_key].shape[0])

        # Priority mechanism: wait if sample() is waiting (give it priority)
        with self._sample_waiting:
            # Yield to any waiting sample() calls
            while self._num_waiting_samples > 0:
                self._sample_waiting.wait()
            
            self._num_active_writes += 1
            
            # Initialize buffer if needed
            if not self._buffer:
                for key, value in arrs.items():
                    self._buffer[key] = np.empty((self.capacity, *value.shape[1:]), dtype=value.dtype)

            # Reserve write slots
            start = int(self._ptr)
            end_ptr = start + batch_size
            self._ptr = (start + batch_size) % self.capacity
            self._size = min(self._size + batch_size, self.capacity)
        
        # Perform numpy writes outside lock (concurrent with other add_batch calls)
        if end_ptr <= self.capacity:
            for key, value in arrs.items():
                self._buffer[key][start:end_ptr] = value
        else:
            num_to_end = self.capacity - start
            overflow = end_ptr - self.capacity
            for key, value in arrs.items():
                self._buffer[key][start:self.capacity] = value[:num_to_end]
                self._buffer[key][:overflow] = value[num_to_end:]
        
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
        lock_acquired_time = None
        
        # Priority mechanism: signal that we're waiting, make add_batch() yield
        with self._sample_waiting:
            self._num_waiting_samples += 1
            
            # Wait for any active add_batch() to finish their writes
            wait_start = time.time()
            while self._num_active_writes > 0:
                self._sample_waiting.wait()
            lock_acquired_time = time.time()
            wait_time_ms = (lock_acquired_time - wait_start) * 1000
            
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
            
            # Profile per-field copy times to identify bottlenecks
            copy_start = time.time()
            result = {}
            field_times = {}
            for key, arr in self._buffer.items():
                field_start = time.time()
                # Use np.take for better performance with random indices
                result[key] = np.take(arr, indices, axis=0)
                field_times[key] = (time.time() - field_start) * 1000
            
            # Release priority
            self._num_waiting_samples -= 1
            self._sample_waiting.notify_all()
            
            total_time_ms = (time.time() - start_time) * 1000
            copy_time_ms = (time.time() - copy_start) * 1000
            
            # Log slow operations with per-field breakdown
            if total_time_ms > 50:
                slowest_field = max(field_times.items(), key=lambda x: x[1])
                print(f"[SACReplayBuffer.sample] Total: {total_time_ms:.1f}ms "
                      f"(wait: {wait_time_ms:.1f}ms, copy: {copy_time_ms:.1f}ms) | "
                      f"Slowest field: {slowest_field[0]}={slowest_field[1]:.1f}ms "
                      f"dtype={self._buffer[slowest_field[0]].dtype} "
                      f"shape={self._buffer[slowest_field[0]].shape}", flush=True)
            
            return result

    def __len__(self):
        with self._sample_waiting:
            return self._size
