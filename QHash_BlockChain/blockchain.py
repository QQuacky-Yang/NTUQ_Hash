# blockchain.py
from time import time
from hash_function import custom_hash


class Block:
    def __init__(self, index, data, previous_hash):
        self.index = index
        self.data = data
        self.previous_hash = previous_hash
        self.timestamp = time()
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        data_str = f"{self.index}{self.data}{self.previous_hash}{self.timestamp}"
        return custom_hash(data_str)

    def __repr__(self):
        return f"Block({self.index}, {self.data}, {self.hash[:8]}...)"


class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        return Block(0, "Genesis Block", "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, data):
        new_block = Block(
            index=self.get_latest_block().index + 1,
            data=data,
            previous_hash=self.get_latest_block().hash
        )
        self.chain.append(new_block)

    def is_valid(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            prev = self.chain[i - 1]
            if current.hash != current.calculate_hash() or prev.hash != current.previous_hash:
                return False
        return True