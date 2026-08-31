import time
from collections import deque

class CircuitBreaker:
    def __init__(self, window_seconds=300, max_block_rate=0.08):
        self.window = window_seconds
        self.max_block_rate = max_block_rate
        self.history = deque()  # Stores (timestamp, action)

    def record_and_check(self, action: str) -> bool:
        now = time.time()
        self.history.append((now, action))
        
        # Evict old entries outside window
        while self.history and self.history[0][0] < now - self.window:
            self.history.popleft()
            
        if len(self.history) < 20:  # Minimum sample size
            return False
            
        blocks = sum(1 for _, act in self.history if act == "ACTION_BLOCK")
        block_rate = blocks / len(self.history)
        
        # Tripped if block rate exceeds safety threshold (e.g. 8%)
        return block_rate > self.max_block_rate
