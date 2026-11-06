"""Display cadence without adding processing-time jitter to every deadline."""

import math


class DisplayPacer:
    def __init__(self, hz, start):
        if not math.isfinite(hz) or hz <= 0:
            raise ValueError("display frequency must be positive and finite")
        self.hz, self.start, self.last_slot = hz, start, -1

    def ready(self, now):
        slot = int(math.floor(max(0.0, now - self.start) * self.hz))
        if slot <= self.last_slot:
            return False
        self.last_slot = slot
        return True
