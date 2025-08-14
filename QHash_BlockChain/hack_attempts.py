# hack_attempts.py

import time
import os
from blockchain import Blockchain

LOG_FILE = "logs/eve.log"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.ctime()}] {msg}\n")
    print(f"EVE: {msg}")

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    log("Starting attack sequence...")

    # Load the current blockchain (Eve gets a copy)
    bc = Blockchain()

    if not bc.is_valid():
        log("Blockchain already corrupted. Nothing to do.")
        exit()

    # === ATTACK: Try to tamper with Block 1 ===
    target_block = 1
    if len(bc.chain) <= target_block:
        log("Not enough blocks to attack.")
        exit()

    old_data = bc.chain[target_block].data
    log(f"Targeting Block {target_block}: '{old_data}'")

    # Change data
    bc.chain[target_block].data = "EVE HACKED: Alice sends 99999 QBC to Eve"
    log("Modified block data!")

    # Recompute hash of modified block
    old_hash = bc.chain[target_block].hash
    bc.chain[target_block].hash = bc.chain[target_block].calculate_hash()
    log(f"🔧 Recomputed hash: {old_hash[:8]}... → {bc.chain[target_block].hash[:8]}...")

    # Now fix all next blocks (relink chain)
    log("⛓️  Relinking the chain...")
    for i in range(target_block + 1, len(bc.chain)):
        bc.chain[i].previous_hash = bc.chain[i - 1].hash
        old_hash = bc.chain[i].hash
        bc.chain[i].hash = bc.chain[i].calculate_hash()
        log(f"Block {i} relinked, hash updated: {old_hash[:8]}... → {bc.chain[i].hash[:8]}...")

    # Save the fake chain
    bc.save()
    log("Fake chain saved! Network will detect on next validation.")

    # Final validation (local)
    if bc.is_valid():
        log("SUCCESS! Fake chain is valid!")
    else:
        log("FAILED: Chain invalid — attack detected!")