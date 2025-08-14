# tabs = 2 spaces
"""
CTQW hash in pure math using the paper's quantization:
  vals_i = floor(p_i * 20000) % 2^k  (k=12 by default)

Includes:
- ctqw_hash(bits: str, n=15, t1=pi, t2=pi, k=12, scale=20000) -> uppercase hex digest
- collision_test(...): strict full-digest equality under 1-bit flip (should be ~0%)
- run_omega_test_flip_one_bit(...): ω histogram (per-vertex chunk equality)
- run_statistical_analysis(...): β̄, ᾱ, p, Δβ, Δp (paper Sec. 3.2)
"""

from __future__ import annotations
import os
import csv
import random
import numpy as np

try:
  from scipy.linalg import expm
  _HAS_SCIPY = True
except Exception:
  _HAS_SCIPY = False


# ---------- graph + math -----------------------------------------------------

def _path_adjacency(n: int) -> np.ndarray:
  """Adjacency matrix A for the path graph P_n."""
  A = np.zeros((n, n), dtype=float)
  for i in range(n - 1):
    A[i, i + 1] = 1.0
    A[i + 1, i] = 1.0
  return A

def _laplacian_from_A(A: np.ndarray) -> np.ndarray:
  """Graph Laplacian L = D - A with D_ii = degree(i)."""
  deg = A.sum(axis=1)
  return np.diag(deg) - A

def _unitary_from_hermitian(H: np.ndarray, t: float) -> np.ndarray:
  """Compute U = exp(-i t H). SciPy expm if available; else spectral decomposition."""
  if _HAS_SCIPY:
    return expm(-1j * t * H)
  w, V = np.linalg.eigh(H)                 # H = V diag(w) V†
  phase = np.exp(-1j * t * w)              # e^{-i t w_j}
  return (V * phase) @ V.conj().T          # V diag(phase) V†


# ---------- hash (paper quantization) ----------------------------------------

def ctqw_hash(bits: str, n: int = 15, t1: float = np.pi, t2: float = np.pi,
              k: int = 12, scale: int = 20000) -> str:
  """
  Continuous-Time Quantum Walk hash (matrix-only) returning an UPPERCASE hex digest.
  Probability quantization follows the paper: floor(p * scale) % 2^k, default scale=20000.

  Args:
    bits:  bitstring message, e.g. "101001..."
    n:     number of vertices in P_n
    t1:    evolution time for A (usually π)
    t2:    evolution time for L (usually π)
    k:     bits per vertex (total digest length = n * k)
    scale: scaling factor before modulo (paper uses 20000)

  Returns:
    Uppercase hex string of length ceil(n*k/4).
  """
  # Build A and L
  A = _path_adjacency(n)
  L = _laplacian_from_A(A)

  # Unitaries
  U0 = _unitary_from_hermitian(A, t1)
  U1 = _unitary_from_hermitian(L, t2)

  # Initial state |0>
  state = np.zeros(n, dtype=complex)
  state[0] = 1.0

  # Apply per message bit
  for b in bits:
    state = (U0 @ state) if b == '0' else (U1 @ state)

  # Probabilities (normalized)
  probs = np.abs(state) ** 2
  s = probs.sum()
  probs = probs / s if s != 0 else probs

  # Paper quantization: floor(p * scale) % 2^k
  K = 1 << k
  vals = (np.floor(probs * scale).astype(int)) % K  # integers in [0, 2^k-1]

  # Concatenate k-bit chunks and convert to uppercase hex without "0x"
  binary_out = ''.join(format(v, f'0{k}b') for v in vals)
  hex_out = hex(int(binary_out, 2))[2:].upper()
  return hex_out


# ---------- helpers (rng, conversions) --------------------------------------

def _rng(seed: int | None) -> np.random.Generator:
  return np.random.default_rng(seed)

def random_bits(L: int, rng: np.random.Generator) -> str:
  return ''.join('1' if x else '0' for x in rng.integers(0, 2, size=L))

def flip_one_bit(bitstr: str, pos: int) -> str:
  b = list(bitstr)
  b[pos] = '1' if b[pos] == '0' else '0'
  return ''.join(b)

def _hex_to_bits(hex_str: str, total_bits: int) -> str:
  """Hex → binary string, left-padded to total_bits."""
  bs = bin(int(hex_str, 16))[2:]
  return bs.zfill(total_bits)

def _chunk_ints_from_hex(hex_str: str, n: int, k: int) -> list[int]:
  """Recover per-vertex integers from the hex digest (inverse of the concat)."""
  bits = _hex_to_bits(hex_str, total_bits=n * k)
  return [int(bits[i * k:(i + 1) * k], 2) for i in range(n)]

def _hamming_bits_from_hex(h1: str, h2: str, total_bits: int) -> int:
  b1 = _hex_to_bits(h1, total_bits)
  b2 = _hex_to_bits(h2, total_bits)
  return sum(x != y for x, y in zip(b1, b2))


# ---------- experiment: strict 1-bit-flip full collision --------------------

def collision_test(trials: int = 10000, msg_len: int = 12,
                   n: int = 15, t1: float = np.pi, t2: float = np.pi,
                   k: int = 12, scale: int = 20000) -> None:
  """
  Strict 1-bit-flip collision test:
  Draw random bitstrings of length msg_len, flip one random bit, and check if
  full hex digests are identical (h(m) == h(m')). Prints collision rate.
  """
  collisions = 0
  for _ in range(trials):
    bits = ''.join(random.choice('01') for _ in range(msg_len))
    h1 = ctqw_hash(bits, n=n, t1=t1, t2=t2, k=k, scale=scale)
    pos = random.randrange(len(bits))
    flipped_bits = bits[:pos] + ('1' if bits[pos] == '0' else '0') + bits[pos+1:]
    h2 = ctqw_hash(flipped_bits, n=n, t1=t1, t2=t2, k=k, scale=scale)
    if h1 == h2:
      collisions += 1
  rate = collisions / trials * 100.0
  print(f"Total tests: {trials}, Collisions: {collisions}, Collision rate: {rate:.4f}%")


# ---------- experiment: ω test (paper Sec. 3.5) -----------------------------

def run_omega_test_flip_one_bit(
  iterations: int = 1000,
  n: int = 15, t1: float = np.pi, t2: float = np.pi, k: int = 12,
  base_msg_len: int = 128,
  out_csv: str | None = "./run/omega_test.csv",
  seed: int | None = 123,
  scale: int = 20000,
) -> dict:
  """
  Flip one random input bit, hash both, and count ω = number of equal per-vertex k-bit chunks
  at the same positions. Report ω=0,1,2 counts and the rate of ω≥1.
  """
  rng = _rng(seed)
  omega_counts: dict[int, int] = {}
  if out_csv is not None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    f = open(out_csv, "w", newline="")
    w = csv.writer(f)
    w.writerow(["trial", "msg_len", "flip_pos", "omega", "digest1_hex", "digest2_hex"])
  else:
    f = None
    w = None

  for t in range(iterations):
    m1 = random_bits(base_msg_len, rng)
    pos = int(rng.integers(0, base_msg_len))
    m2 = flip_one_bit(m1, pos)

    h1 = ctqw_hash(m1, n=n, t1=t1, t2=t2, k=k, scale=scale)
    h2 = ctqw_hash(m2, n=n, t1=t1, t2=t2, k=k, scale=scale)

    v1 = _chunk_ints_from_hex(h1, n=n, k=k)
    v2 = _chunk_ints_from_hex(h2, n=n, k=k)
    omega = sum(1 for a, b in zip(v1, v2) if a == b)

    omega_counts[omega] = omega_counts.get(omega, 0) + 1
    if w:
      w.writerow([t, base_msg_len, pos, omega, h1, h2])

  if f:
    f.close()

  omega0 = omega_counts.get(0, 0)
  omega1 = omega_counts.get(1, 0)
  omega2 = omega_counts.get(2, 0)
  trials_with_match = sum(c for wcnt, c in omega_counts.items() if wcnt >= 1)
  collision_rate = (trials_with_match / iterations) * 100.0

  return {
    "iterations": iterations,
    "omega_hist": dict(sorted(omega_counts.items())),
    "omega0": omega0,
    "omega1": omega1,
    "omega2": omega2,
    "collision_rate_percent": collision_rate,
    "csv": out_csv,
  }


# ---------- experiment: statistical analysis (paper Sec. 3.2) ---------------

def run_statistical_analysis(
  iterations: int = 25000,          # η
  n: int = 15, t1: float = np.pi, t2: float = np.pi, k: int = 12,
  base_msg_len: int = 128,
  out_csv: str | None = "./run/stats.csv",
  seed: int | None = 321,
  scale: int = 20000,
) -> dict:
  """
  Compute:
    β̄  : mean changed bit number (Hamming distance between full n*k-bit digests)
    ᾱ  : mean added zeros to reach fixed k bits per vertex
    p   : mean changed probability (%) = (β̄ / (n*k)) * 100
    Δβ  : std of changed bit number (sample)
    Δp  : std of changed probability (%) (sample)
  Each trial flips one random input bit; we compare the two digests.
  """
  rng = _rng(seed)
  total_bits = n * k

  betas: list[int] = []
  alphas: list[int] = []

  if out_csv is not None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    f = open(out_csv, "w", newline="")
    w = csv.writer(f)
    w.writerow(["trial", "msg_len", "flip_pos", "beta", "alpha", "p_percent"])
  else:
    f = None
    w = None

  for t in range(iterations):
    m1 = random_bits(base_msg_len, rng)
    pos = int(rng.integers(0, base_msg_len))
    m2 = flip_one_bit(m1, pos)

    h1 = ctqw_hash(m1, n=n, t1=t1, t2=t2, k=k, scale=scale)
    h2 = ctqw_hash(m2, n=n, t1=t1, t2=t2, k=k, scale=scale)

    # β_j: Hamming distance between full digests
    beta_j = _hamming_bits_from_hex(h1, h2, total_bits=total_bits)
    betas.append(beta_j)

    # α_j: “added zeros” if each vertex value were stored minimally then padded to k bits.
    vals2 = _chunk_ints_from_hex(h2, n=n, k=k)   # use modified hash side
    alpha_j = sum(max(0, k - max(1, v.bit_length())) for v in vals2)
    alphas.append(alpha_j)

    if w:
      p_j = (beta_j / total_bits) * 100.0
      w.writerow([t, base_msg_len, pos, beta_j, alpha_j, p_j])

  if f:
    f.close()

  # Means
  beta_mean = float(np.mean(betas)) if betas else 0.0
  alpha_mean = float(np.mean(alphas)) if alphas else 0.0
  p_mean_percent = (beta_mean / total_bits) * 100.0

  # Sample std (η-1 in denominator)
  if len(betas) >= 2:
    delta_beta = float(np.std(betas, ddof=1))
    p_trials = [(b / total_bits) * 100.0 for b in betas]
    delta_p_percent = float(np.std(p_trials, ddof=1))
  else:
    delta_beta = 0.0
    delta_p_percent = 0.0

  return {
    "iterations": iterations,
    "beta_mean": beta_mean,                 # β̄
    "alpha_mean": alpha_mean,               # ᾱ
    "p_mean_percent": p_mean_percent,       # p
    "delta_beta": delta_beta,               # Δβ
    "delta_p_percent": delta_p_percent,     # Δp
    "csv": out_csv,
  }


# ---------- demo / CLI -------------------------------------------------------

if __name__ == "__main__":
  # Baseline params (paper)
  n = 15
  k = 12
  scale = 20000
  msg_bits = "101001001011"

  # Single hash
  h = ctqw_hash(msg_bits, n=n, t1=np.pi, t2=np.pi, k=k, scale=scale)
  print(f"Total bits = {n*k}=180 when n=15,k=12")
  print("HEX (45 hex chars when n=15,k=12):")
  print(h)

  # Strict 1-bit-flip full-collision test (should be ~0%)
  collision_test(trials=10000, msg_len=len(msg_bits), n=n, t1=np.pi, t2=np.pi, k=k, scale=scale)

  # ω test (paper Sec. 3.5)
  omega = run_omega_test_flip_one_bit(
    iterations=1000, n=n, t1=np.pi, t2=np.pi, k=k,
    base_msg_len=128, out_csv="./run/omega_test.csv", seed=123, scale=scale
  )
  print("\n[Paper-style ω test]")
  print("Iterations:", omega["iterations"])
  print("ω histogram:", omega["omega_hist"])
  print("Counts → ω=0:", omega["omega0"], "ω=1:", omega["omega1"], "ω=2:", omega["omega2"])
  print("“Collision rate” (ω ≥ 1) %:", round(omega["collision_rate_percent"], 3))
  print("CSV:", omega["csv"])

  # Statistical analysis (Sec. 3.2)
  stats = run_statistical_analysis(
    iterations=1000, n=n, t1=np.pi, t2=np.pi, k=k,
    base_msg_len=128, out_csv="./run/stats.csv", seed=321, scale=scale
  )
  print("\n[Statistical analysis]")
  print("η:", stats["iterations"])
  print("β̄ (mean changed bits):", round(stats["beta_mean"], 3))
  print("ᾱ (mean added zeros):", round(stats["alpha_mean"], 3))
  print("p (mean changed probability %):", round(stats["p_mean_percent"], 3))
  print("Δβ (std of changed bits):", round(stats["delta_beta"], 3))
  print("Δp (std of changed probability %):", round(stats["delta_p_percent"], 3))
  print("CSV:", stats["csv"])
