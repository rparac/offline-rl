import ray

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
        if len(self.buffer) >= self.batch_size:
            pass

    