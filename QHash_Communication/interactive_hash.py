#!/usr/bin/env python3
"""
Interactive CLI for CTQW hash using ctqw_hash from final.py.

Usage:
  python interactive_hash.py
Then type messages. Type 'q' or 'quit' to exit.
"""
import sys
import os

# Ensure we can import final.py from the same directory as this script
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    from final import ctqw_hash
except Exception as e:
    print("Error: could not import ctqw_hash from final.py:", e)
    sys.exit(1)

def str_to_bits(s: str) -> str:
    """UTF-8 encode string to bitstring like '010101'."""
    b = s.encode('utf-8')
    return ''.join(f'{byte:08b}' for byte in b)

def main():
    print("CTQW Hash — Interactive Mode")
    print('Type "q" or "quit" to exit.\n')

    # Default parameters are those in final.ctqw_hash signature.
    # You can tweak them below if desired.
    while True:
        try:
            msg = input("Enter message: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if msg.lower() in ("q", "quit"):
            print("Goodbye!")
            break

        # Convert to bits and hash
        bits = str_to_bits(msg)
        try:
            digest = ctqw_hash(bits)  # uses defaults in final.py (n=15, k=12, scale=20000, keep_bits_max=256)
        except Exception as e:
            print("Error while hashing:", e)
            continue

        print("Hash:", digest, "\n")

if __name__ == "__main__":
    main()
