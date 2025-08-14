# blockchain.py

from time import time
from hash_function import custom_hash


class Block:
    def __init__(self, index: int, data: str, previous_hash: str):
        self.index = index
        self.data = data
        self.previous_hash = previous_hash
        self.timestamp = time()
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        data_str = f"{self.index}{self.data}{self.previous_hash}{self.timestamp}"
        return custom_hash(data_str)

    def __repr__(self):
        return (f"Block({self.index}, "
                f"data='{self.data[:32]}...', "
                f"hash={self.hash[:8]}..., "
                f"prev={self.previous_hash[:8] if self.previous_hash else 'None'}...)")


class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self) -> Block:
        return Block(0, "Genesis Block", "0")

    def get_latest_block(self) -> Block:
        return self.chain[-1]

    def add_block(self, data: str):
        latest_block = self.get_latest_block()
        new_block = Block(
            index=latest_block.index + 1,
            data=data,
            previous_hash=latest_block.hash
        )
        self.chain.append(new_block)

    def is_valid(self) -> bool:
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            if current.hash != current.calculate_hash():
                print(f"Block {i} has invalid hash!")
                return False

            if current.previous_hash != previous.hash:
                print(f"Block {i} links to wrong previous hash!")
                return False

        return True

    def __repr__(self):
        return '\n'.join([str(block) for block in self.chain])