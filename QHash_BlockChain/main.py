# main.py

import time
import os
from blockchain import Blockchain

LOG_FILE = "logs/honest.log"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.ctime()}] {msg}\n")
    print(msg)

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    bc = Blockchain()  # Auto-loads from file

    # Add new blocks
    transactions = [
        "Alice sends 5 QBC to Bob",
        "Bob sends 2 QBC to Charlie",
        "Mining reward: 6.25 QBC",
        "Dex trade: 1 QBC → 100 Tokens"
    ]

    for tx in transactions:
        log(f"Adding block: {tx}")
        bc.add_block(tx)
        time.sleep(1)  # Simulate time passing

    # Final check
    valid = bc.is_valid()
    log(f"Blockchain valid: {valid}")