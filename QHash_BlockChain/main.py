# main.py
from blockchain import Blockchain

# Create blockchain
bc = Blockchain()

# Add blocks
bc.add_block("Alice sends 1 BTC to Bob")
bc.add_block("Charlie sends 2 BTC to Alice")
bc.add_block("Miner reward: 6.25 BTC")

# Print chain
for block in bc.chain:
    print(block)

# Validate
print("Is blockchain valid?", bc.is_valid())