import ray
from typing import List, Optional

@ray.remote
class UnlabelledTransitionQueue:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.buffer = []

    def add(self, transition_ref: ray.ObjectRef):
        """
        Agents call this.
        CRITICAL: transition_ref should be a Ray ObjectRef (ray.put(data)), 
        not the data itself. This keeps the heavy image data in the Object Store,
        passing only the tiny pointer ID through this actor.
        """

        self.buffer.append(transition_ref)

    def pop_batch(self, max_items: Optional[int] = None) -> List[ray.ObjectRef]:
        """
        Pops up to `max_items` transition refs (FIFO).
        Returns an empty list if nothing is available.
        """
        if not self.buffer:
            return []

        n = len(self.buffer) if max_items is None else min(len(self.buffer), int(max_items))
        batch = self.buffer[:n]
        del self.buffer[:n]
        return batch

    def __len__(self) -> int:
        return len(self.buffer)

    