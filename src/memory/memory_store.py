from collections import deque

class MemoryStore:
    def __init__(self, k=5):
        self.buffer = deque(maxlen=k)

    def add(self, q, a):
        self.buffer.append({"q": q, "a": a})

    def get(self):
        return "\n".join([f"Q:{m['q']} A:{m['a']}" for m in self.buffer])