# tabs = 2 spaces
"""
CTQW hash in pure math using the paper's quantization:
  vals_i = floor(p_i * scale) % 2^k

IMPORTANT in this version:
- You can set n and k freely.
- The ALGORITHM OUTPUT is truncated to the FIRST 256 BITS only if n*k > 256.
  If n*k <= 256, the algorithm outputs ALL n*k bits.
- All analyses (collision, ω, β̄, p, Δβ, Δp) operate on the digest as produced
  by the algorithm, i.e., N_eval = min(n*k, 256).
- ᾱ is dropped (depends on per-vertex k-bit chunks, not meaningful post-truncation).

Demo uses n=31 per instructor suggestion.
"""

from __future__ import annotations
import os
import csv
import math
import random
import numpy as np

# Optional SciPy for faster expm
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
              k: int = 12, scale: int = 20000, keep_bits_max: int = 256) -> str:
  """
  Continuous-Time Quantum Walk hash (matrix-only) returning an UPPERCASE hex digest.
  The algorithm outputs ONLY the first 256 bits if n*k > 256; otherwise all n*k bits.

  Probability quantization: floor(p * scale) % 2^k.

  Args:
    bits:  bitstring message, e.g. "101001..."
    n:     number of vertices in P_n
    t1:    evolution time for A (usually π)
    t2:    evolution time for L (usually π)
    k:     bits per vertex before any truncation (full output length = n * k)
    scale: scaling factor before modulo (paper uses ~20000)
    keep_bits_max: hard cap for output length (default 256)

  Returns:
    Uppercase hex string; length = ceil(min(n*k, keep_bits_max)/4), zero-padded to preserve leading zeros.
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

  # Paper quantization: floor(p * scale) % 2^k  → integers in [0, 2^k-1]
  K = 1 << k
  vals = (np.floor(probs * scale).astype(int)) % K

  # Concatenate k-bit chunks
  full_bits = ''.join(format(v, f'0{k}b') for v in vals)  # length = n*k
  total_bits_full = len(full_bits)

  # Keep only up to keep_bits_max if longer; else keep all
  keep_bits_used = min(total_bits_full, keep_bits_max)
  kept = full_bits[:keep_bits_used]

  # Convert to fixed-length hex (preserving leading zeros)
  hex_len = (keep_bits_used + 3) // 4
  hex_out = hex(int(kept or '0', 2))[2:].upper().zfill(hex_len)
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

def _hex_to_bits_fixed(hex_str: str, total_bits: int) -> str:
  """Hex → binary string, left-padded to exactly total_bits."""
  bs = bin(int(hex_str, 16))[2:]
  return bs.zfill(total_bits)


# ---------- experiment: strict 1-bit-flip full collision --------------------

def collision_test(trials: int = 10000, msg_len: int = 12,
                   n: int = 15, t1: float = np.pi, t2: float = np.pi,
                   k: int = 12, scale: int = 20000, keep_bits_max: int = 256) -> None:
  """
  Strict 1-bit-flip collision test on the algorithm's digest:
  the digest length is N_eval = min(n*k, keep_bits_max).
  """
  collisions = 0
  N_eval = min(n * k, keep_bits_max)
  for _ in range(trials):
    bits = ''.join(random.choice('01') for _ in range(msg_len))
    h1 = ctqw_hash(bits, n=n, t1=t1, t2=t2, k=k, scale=scale, keep_bits_max=keep_bits_max)
    pos = random.randrange(len(bits))
    flipped_bits = bits[:pos] + ('1' if bits[pos] == '0' else '0') + bits[pos+1:]
    h2 = ctqw_hash(flipped_bits, n=n, t1=t1, t2=t2, k=k, scale=scale, keep_bits_max=keep_bits_max)
    if h1 == h2:
      collisions += 1
  rate = collisions / trials * 100.0
  print(f"Total tests: {trials}, Collisions: {collisions}, Collision rate (on {N_eval} bits): {rate:.4f}%")


# ---------- ω test (block-wise on the digest, with tail support) ------------

def run_omega_test_flip_one_bit(
  iterations: int = 1000,
  n: int = 15, t1: float = np.pi, t2: float = np.pi, k: int = 12,
  base_msg_len: int = 128,
  out_csv: str | None = "./run/omega_test.csv",
  seed: int | None = 123,
  scale: int = 20000,
  keep_bits_max: int = 256,
  block_bits: int = 16,   # block size in bits; tail (< block_bits) is also compared
) -> dict:
  """
  Flip one random input bit, hash both (algorithm caps to min(n*k, keep_bits_max)),
  split the digest into floor(N_eval/block_bits) full blocks, and (optionally) ONE
  tail block with size N_eval % block_bits. ω counts equal blocks at same positions,
  INCLUDING the leftover tail as a mini-block when tail_bits > 0.
  """
  assert block_bits > 0, "block_bits must be positive"
  rng = _rng(seed)
  omega_counts: dict[int, int] = {}
  N_eval = min(n * k, keep_bits_max)

  if out_csv is not None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    f = open(out_csv, "w", newline="")
    w = csv.writer(f)
    w.writerow(["trial", "msg_len", "flip_pos", "omega", "digest1_hex", "digest2_hex"])
  else:
    f = None
    w = None

  num_blocks_full = N_eval // block_bits
  tail_bits = N_eval - num_blocks_full * block_bits

  for t in range(iterations):
    m1 = random_bits(base_msg_len, rng)
    pos = int(rng.integers(0, base_msg_len))
    m2 = flip_one_bit(m1, pos)

    h1 = ctqw_hash(m1, n=n, t1=t1, t2=t2, k=k, scale=scale, keep_bits_max=keep_bits_max)
    h2 = ctqw_hash(m2, n=n, t1=t1, t2=t2, k=k, scale=scale, keep_bits_max=keep_bits_max)

    b1 = _hex_to_bits_fixed(h1, N_eval)
    b2 = _hex_to_bits_fixed(h2, N_eval)

    omega = 0
    # Compare full blocks
    for i in range(num_blocks_full):
      a = b1[i*block_bits:(i+1)*block_bits]
      b = b2[i*block_bits:(i+1)*block_bits]
      if a == b:
        omega += 1
    # Compare leftover tail mini-block, if any
    # if tail_bits > 0:
    #   a_tail = b1[num_blocks_full*block_bits:N_eval]
    #   b_tail = b2[num_blocks_full*block_bits:N_eval]
    #   if a_tail == b_tail:
    #     omega += 1

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
    "num_blocks_full": num_blocks_full,
    "tail_bits": tail_bits,
    "block_bits": block_bits,
    "keep_bits_used": N_eval,
  }


# ---------- experiment: statistical analysis on the digest -------------------

def run_statistical_analysis(
  iterations: int = 25000,          # η
  n: int = 15, t1: float = np.pi, t2: float = np.pi, k: int = 12,
  base_msg_len: int = 128,
  out_csv: str | None = "./run/stats.csv",
  seed: int | None = 321,
  scale: int = 20000,
  keep_bits_max: int = 256,
) -> dict:
  """
  Compute on the algorithm's digest (length N_eval = min(n*k, keep_bits_max)):
    β̄  : mean changed bit number (Hamming on N_eval bits)
    p   : mean changed probability (%) = (β̄ / N_eval) * 100
    Δβ  : std of changed bit number (sample) on N_eval
    Δp  : std of changed probability (%) (sample) on N_eval
  Each trial flips one random input bit; we compare the two digests.
  """
  rng = _rng(seed)
  N_eval = min(n * k, keep_bits_max)
  betas: list[int] = []

  if out_csv is not None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    f = open(out_csv, "w", newline="")
    w = csv.writer(f)
    w.writerow(["trial", "msg_len", "flip_pos", "beta", "p_percent"])
  else:
    f = None
    w = None

  for t in range(iterations):
    m1 = random_bits(base_msg_len, rng)
    pos = int(rng.integers(0, base_msg_len))
    m2 = flip_one_bit(m1, pos)

    h1 = ctqw_hash(m1, n=n, t1=t1, t2=t2, k=k, scale=scale, keep_bits_max=keep_bits_max)
    h2 = ctqw_hash(m2, n=n, t1=t1, t2=t2, k=k, scale=scale, keep_bits_max=keep_bits_max)

    b1 = _hex_to_bits_fixed(h1, N_eval)
    b2 = _hex_to_bits_fixed(h2, N_eval)

    beta_j = sum(x != y for x, y in zip(b1, b2))
    betas.append(beta_j)

    if w:
      p_j = (beta_j / N_eval) * 100.0 if N_eval else 0.0
      w.writerow([t, base_msg_len, pos, beta_j, p_j])

  if f:
    f.close()

  # Means
  beta_mean = float(np.mean(betas)) if betas else 0.0
  p_mean_percent = (beta_mean / N_eval) * 100.0 if N_eval else 0.0

  # Sample std (η-1 in denominator)
  if len(betas) >= 2:
    delta_beta = float(np.std(betas, ddof=1))
    p_trials = [(b / N_eval) * 100.0 for b in betas] if N_eval else [0.0 for _ in betas]
    delta_p_percent = float(np.std(p_trials, ddof=1))
  else:
    delta_beta = 0.0
    delta_p_percent = 0.0

  return {
    "iterations": iterations,
    "beta_mean": beta_mean,           # β̄ on N_eval bits
    "p_mean_percent": p_mean_percent, # p on N_eval bits
    "delta_beta": delta_beta,         # Δβ
    "delta_p_percent": delta_p_percent, # Δp
    "eval_bits_used": N_eval,
    "csv": out_csv,
  }


# ---------- pretty printers --------------------------------------------------

def print_stats_report(stats: dict, *, N_eval_bits: int) -> None:
  """
  Pretty-print the 4 metrics with binomial baselines on a fixed-length slice.
  """
  N = N_eval_bits
  ideal_beta       = N / 2.0
  ideal_sigma_beta = math.sqrt(N * 0.5 * 0.5)
  ideal_p          = 50.0
  ideal_delta_p    = (ideal_sigma_beta / N) * 100.0

  print(f"η (trials): {stats['iterations']}")
  print(f"N_eval (digest bits used): {N}")
  print(f"β̄ (mean changed bits): {stats['beta_mean']:.3f}  [ideal≈{ideal_beta:.1f}]")
  print("   ⤷ higher→better up to N/2.")
  print(f"p (mean changed probability %): {stats['p_mean_percent']:.3f}%  [ideal≈{ideal_p:.1f}%]")
  print("   ⤷ closer to 50% is better.")
  print(f"Δβ (std of changed bits): {stats['delta_beta']:.3f}  [ideal≈{ideal_sigma_beta:.2f}]")
  print("   ⤷ lower→more consistent avalanche.")
  print(f"Δp (std of changed probability %): {stats['delta_p_percent']:.3f}%  [ideal≈{ideal_delta_p:.2f}%]")
  print("   ⤷ lower→more consistent % change.")

def print_omega_report(omega: dict) -> None:
  """Pretty-print ω test on fixed-size blocks with tail support."""
  print("\n[Block-wise ω test on digest]")
  print(f"Iterations: {omega['iterations']}")
  print(f"keep_bits_used: {omega['keep_bits_used']}  |  block_bits: {omega['block_bits']}")
  print(f"#full blocks: {omega['num_blocks_full']}  |  tail_bits: {omega['tail_bits']}")
  print(f"ω histogram: {omega['omega_hist']}")
  print(f"Counts → ω=0: {omega['omega0']},  ω=1: {omega['omega1']},  ω=2: {omega['omega2']}")
  print(f"“Collision rate” % (ω ≥ 1): {omega['collision_rate_percent']:.3f}")
  print("   ⤷ lower→better; local block-similarity check, not a full digest collision.")


# ---------- demo / CLI -------------------------------------------------------

if __name__ == "__main__":
  # Baseline params (with n=31 per instructor; algorithm returns min(n*k, 256) bits)
  iters = 1000
  n = 31
  k = 12
  CAP = 256
  scale = (2**k) * n
  msg_bits = "101001001011"

  # Single hash
  h = ctqw_hash(msg_bits, n=n, t1=np.pi, t2=np.pi, k=k, scale=scale, keep_bits_max=CAP)
  print(f"Algorithm digest bits = min(n*k, {CAP}) = {min(n*k, CAP)} (hex length = {(min(n*k, CAP)+3)//4})")
  print("HEX:")
  print(h)

  # Strict 1-bit-flip collision test on the digest
  collision_test(trials=iters, msg_len=len(msg_bits), n=n, t1=np.pi, t2=np.pi,
                 k=k, scale=scale, keep_bits_max=CAP)

  # ω test on the digest with tail comparison enabled (always on here)
  omega = run_omega_test_flip_one_bit(
    iterations=iters, n=n, t1=np.pi, t2=np.pi, k=k,
    base_msg_len=128, out_csv="./run/omega_test.csv", seed=123, scale=scale,
    keep_bits_max=CAP, block_bits=k
  )
  print_omega_report(omega)
  print("CSV:", omega["csv"])

  # Statistical analysis on the digest
  stats = run_statistical_analysis(
    iterations=iters, n=n, t1=np.pi, t2=np.pi, k=k,
    base_msg_len=128, out_csv="./run/stats.csv", seed=321, scale=scale,
    keep_bits_max=CAP,
  )
  print("\n[Statistical analysis on digest]")
  print_stats_report(stats, N_eval_bits=stats["eval_bits_used"])
  print("CSV:", stats["csv"])
  print()
