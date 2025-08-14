# tabs = 2 spaces
from __future__ import annotations
import os
import csv
import numpy as np
from math import ceil, log2
from typing import Iterable, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

try:
  from scipy.linalg import expm
  _HAS_SCIPY = True
except Exception:
  _HAS_SCIPY = False


# ---------- graph + math -----------------------------------------------------

def path_adjacency(n: int) -> np.ndarray:
  """
  Build adjacency matrix A for the path graph P_n.
  Maps to Algorithm 1: graph structure used for H=A or H=L.
  """
  A = np.zeros((n, n), dtype=float)
  for i in range(n - 1):
    A[i, i + 1] = 1.0
    A[i + 1, i] = 1.0
  return A

def degree_from_adjacency(A: np.ndarray) -> np.ndarray:
  """Degree matrix D = diag(deg(v_i)) for Laplacian L = D - A."""
  return np.diag(A.sum(axis=1))

def pad_to_power_of_two(M: np.ndarray) -> Tuple[np.ndarray, int]:
  """
  Pad n×n matrix to top-left of a 2^q×2^q matrix, so it can be a UnitaryGate.
  """
  n = M.shape[0]
  q = max(1, ceil(log2(n)))
  dim = 1 << q
  P = np.zeros((dim, dim), dtype=complex)
  P[:n, :n] = M.astype(complex)
  return P, q

def unitary_from_hermitian(H: np.ndarray, t: float) -> np.ndarray:
  """Compute U = exp(-i t H)."""
  if _HAS_SCIPY:
    return expm(-1j * t * H)
  w, U = np.linalg.eigh(H)
  phases = np.exp(-1j * t * w)
  return (U * phases) @ U.conj().T


# cache to avoid recomputing e^{-i t A} and e^{-i t L}
_UNITARY_CACHE: dict[tuple[int, float, float], tuple[UnitaryGate, UnitaryGate, int]] = {}


def _get_u0_u1_gates(n: int, t1: float, t2: float) -> tuple[UnitaryGate, UnitaryGate, int]:
  """
  Return (U0_gate, U1_gate, q) for given (n, t1, t2), using a small cache.
  U0 = e^{-i t1 A}, U1 = e^{-i t2 L}; both padded to 2^q.
  """
  key = (n, float(t1), float(t2))
  if key in _UNITARY_CACHE:
    return _UNITARY_CACHE[key]

  A = path_adjacency(n)
  D = degree_from_adjacency(A)
  L = D - A

  A_pad, q = pad_to_power_of_two(A)
  L_pad, _ = pad_to_power_of_two(L)

  U0_gate = UnitaryGate(unitary_from_hermitian(A_pad, t1), label="U0=e^{-i t A}")
  U1_gate = UnitaryGate(unitary_from_hermitian(L_pad, t2), label="U1=e^{-i t L}")

  _UNITARY_CACHE[key] = (U0_gate, U1_gate, q)
  return _UNITARY_CACHE[key]


# ---------- Algorithm 1: CTQW hash ------------------------------------------

def ctqw_hash(
  bits: Iterable[int] | str,
  n: int = 15,                 # paper uses n=15
  t1: float = np.pi,           # paper uses t = π
  t2: float = np.pi,           # paper uses t = π
  k_bits: int = 12,            # 12 bits per vertex → 180-bit digest
) -> dict:
  """
  Continuous-Time Quantum Walk Hash (Algorithm 1 implementation).
  Returns:
    - 'binary': n*k_bits bitstring (180 when n=15,k_bits=12)
    - 'hex': hex digest (pretty print)
    - 'groups_hex_3': per-vertex 12-bit chunks (3 hex digits)
    - 'groups_hex_4': same chunks shown as 4 hex digits (cosmetic)
    - 'probs': vertex probabilities (sum≈1)
  """
  # normalize message bits into '0'/'1'
  if isinstance(bits, str):
    bitstr = ''.join('1' if b in ('1', 'True', 'true') else '0' for b in bits)
  else:
    bitstr = ''.join('1' if int(b) else '0' for b in bits)

  # precompute U0/U1 once per (n,t1,t2)
  U0_gate, U1_gate, q = _get_u0_u1_gates(n=n, t1=t1, t2=t2)

  # build final state: |ψ_f> = U_{m_p} ... U_{m_1} |0...0>
  qc = QuantumCircuit(q)
  for b in bitstr:
    qc.append(U0_gate if b == '0' else U1_gate, qc.qubits)
  qc.save_statevector()

  # simulate; robustly convert to ndarray
  try:
    sim = AerSimulator(method="statevector")
    tqc = transpile(qc, sim)
    sv_obj = sim.run(tqc).result().get_statevector(tqc)
    sv = np.asarray(getattr(sv_obj, "data", sv_obj), dtype=complex)
  except Exception:
    sv = Statevector.from_instruction(qc).data

  # vertex subspace = first n basis states
  amps = sv[:n]
  probs = np.abs(amps) ** 2
  s = probs.sum()
  probs = probs / s if s != 0 else probs

  # map probs → k_bits per vertex
  K = 1 << k_bits
  vals = (np.floor(probs * K).astype(int)) % K

  chunks_bin = [format(v, f'0{k_bits}b') for v in vals]
  bit_concat = ''.join(chunks_bin)

  hex_len = (len(bit_concat) + 3) // 4
  hex_digest = format(int(bit_concat, 2), f'0{hex_len}x')
  groups_hex_3 = [format(v, '03x') for v in vals]
  groups_hex_4 = [format(v, '04x') for v in vals]  # cosmetic

  return {
    'binary': bit_concat,
    'hex': hex_digest,
    'groups_hex_3': groups_hex_3,
    'groups_hex_4': groups_hex_4,
    'probs': probs,
  }


# ---------- circuit drawing (optional) ---------------------------------------

def build_ctqw_circuit(bits: str | list[int], n: int = 15, t1: float = np.pi, t2: float = np.pi) -> QuantumCircuit:
  """Build the CTQW circuit (one block per bit)."""
  if isinstance(bits, str):
    bitstr = ''.join('1' if b in ('1', 'True', 'true') else '0' for b in bits)
  else:
    bitstr = ''.join('1' if int(b) else '0' for b in bits)

  U0, U1, q = _get_u0_u1_gates(n=n, t1=t1, t2=t2)
  qc = QuantumCircuit(q, name="CTQW Hash")
  for b in bitstr:
    qc.append(U0 if b == '0' else U1, qc.qubits)
  return qc

def save_hash_circuit_png(bits: str | list[int], out_path: str = "./fig/hash_circuit.png",
                          n: int = 15, t1: float = np.pi, t2: float = np.pi,
                          fold_cols: int = 120, dpi: int = 300) -> str:
  """Render the circuit diagram to a PNG file."""
  import matplotlib.pyplot as plt
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  qc = build_ctqw_circuit(bits, n=n, t1=t1, t2=t2)
  fig = qc.draw(output="mpl", fold=fold_cols)
  fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
  plt.close(fig)
  return out_path


# ---------- helpers for experiments -----------------------------------------

def _rng(seed: int | None) -> np.random.Generator:
  return np.random.default_rng(seed)

def random_bits(L: int, rng: np.random.Generator) -> str:
  """Draw a random bitstring of length L."""
  return ''.join('1' if x else '0' for x in rng.integers(0, 2, size=L))

def flip_one_bit(bitstr: str, pos: int) -> str:
  """Return bitstr with bit at pos flipped."""
  b = list(bitstr)
  b[pos] = '1' if b[pos] == '0' else '0'
  return ''.join(b)


# ---------- Experiment A: Full digest-collision search ----------------------

def run_full_collision_search(
  iterations: int = 1000,
  n: int = 15, t1: float = np.pi, t2: float = np.pi, k_bits: int = 12,
  msg_len_min: int = 64, msg_len_max: int = 128,
  out_csv: str = "./run/hash_values.csv",
  seed: int | None = 42,
) -> dict:
  """
  Classical collision test: count times we see SAME 180-bit digest for DIFFERENT messages.
  Writes every trial to CSV; returns summary dict with 'collisions' and 'unique_digests'.
  """
  os.makedirs(os.path.dirname(out_csv), exist_ok=True)
  rng = _rng(seed)
  seen: dict[str, str] = {}          # digest_hex -> first message seen
  collisions: list[tuple[int, str, str]] = []  # (trial_idx, digest_hex, previous_message)

  with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["trial", "msg_len", "message_bits", "digest_hex"])
    for t in range(iterations):
      L = int(rng.integers(msg_len_min, msg_len_max + 1))
      m = random_bits(L, rng)
      h = ctqw_hash(m, n=n, t1=t1, t2=t2, k_bits=k_bits)["hex"]
      w.writerow([t, L, m, h])

      if h in seen and seen[h] != m:
        collisions.append((t, h, seen[h]))
      else:
        seen[h] = m

  summary = {
    "iterations": iterations,
    "unique_digests": len(seen),
    "collisions": len(collisions),
    "collision_examples": collisions[:10],  # first few if any (likely zero)
    "csv": out_csv,
  }
  return summary


# ---------- Experiment B: Paper-style ω test (Sec. 3.5) ---------------------

def run_omega_test_flip_one_bit(
  iterations: int = 1000,
  n: int = 15, t1: float = np.pi, t2: float = np.pi, k_bits: int = 12,
  base_msg_len: int = 128,
  out_csv: str = "./run/omega_test.csv",
  seed: int | None = 123,
) -> dict:
  """
  Paper-like test: For each trial, flip one random bit in the message, hash both,
  and compute ω = number of per-vertex chunks (4-hex) that are identical at the same positions.
  Returns a summary that includes counts for ω=0,1,2 and the collision rate (ω >= 1).
  """
  os.makedirs(os.path.dirname(out_csv), exist_ok=True)
  rng = _rng(seed)
  omega_counts: dict[int, int] = {}

  with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["trial", "msg_len", "flip_pos", "omega", "digest1_hex", "digest2_hex"])
    for t in range(iterations):
      m1 = random_bits(base_msg_len, rng)
      pos = int(rng.integers(0, base_msg_len))
      m2 = flip_one_bit(m1, pos)

      out1 = ctqw_hash(m1, n=n, t1=t1, t2=t2, k_bits=k_bits)
      out2 = ctqw_hash(m2, n=n, t1=t1, t2=t2, k_bits=k_bits)

      # ω: count equal per-vertex chunks at the same indices
      g1 = out1["groups_hex_4"]
      g2 = out2["groups_hex_4"]
      omega = sum(1 for a, b in zip(g1, g2) if a == b)

      omega_counts[omega] = omega_counts.get(omega, 0) + 1
      w.writerow([t, base_msg_len, pos, omega, out1["hex"], out2["hex"]])

  # Pull out ω=0,1,2 counts and compute collision rate (ω >= 1)
  omega0 = omega_counts.get(0, 0)
  omega1 = omega_counts.get(1, 0)
  omega2 = omega_counts.get(2, 0)
  trials_with_match = sum(c for w, c in omega_counts.items() if w >= 1)
  collision_rate = (trials_with_match / iterations) * 100.0

  summary = {
    "iterations": iterations,
    "omega_hist": dict(sorted(omega_counts.items())),
    "omega0": omega0,
    "omega1": omega1,
    "omega2": omega2,
    "collision_rate_percent": collision_rate,  # ω ≥ 1 fraction
    "csv": out_csv,
  }
  return summary


# ---------- Statistical analysis (Sec. 3.2) ----------------------------------

def run_statistical_analysis(
  iterations: int = 25000,          # η
  n: int = 15, t1: float = np.pi, t2: float = np.pi, k_bits: int = 12,
  base_msg_len: int = 128,          # message length used for flip-one-bit
  out_csv: str = "./run/stats.csv",
  seed: int | None = 321,
) -> dict:
  """
  Compute:
    β̄  : mean changed bit number
    ᾱ  : mean added zeros (to reach fixed k_bits per vertex)
    p   : mean changed probability (%) = (β̄ / (n*k_bits)) * 100
    Δβ  : std of changed bit number (sample)
    Δp  : std of changed probability (%) (sample)
  Each trial flips one random input bit; we compare the two 180-bit digests.
  """
  os.makedirs(os.path.dirname(out_csv), exist_ok=True)
  rng = _rng(seed)
  total_bits = n * k_bits

  betas: list[int] = []
  alphas: list[int] = []

  with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["trial", "msg_len", "flip_pos", "beta", "alpha", "p_percent"])
    for t in range(iterations):
      m1 = random_bits(base_msg_len, rng)
      pos = int(rng.integers(0, base_msg_len))
      m2 = flip_one_bit(m1, pos)

      out1 = ctqw_hash(m1, n=n, t1=t1, t2=t2, k_bits=k_bits)
      out2 = ctqw_hash(m2, n=n, t1=t1, t2=t2, k_bits=k_bits)

      # β_j: Hamming distance between 180-bit digests
      b1 = out1["binary"]
      b2 = out2["binary"]
      beta_j = sum(c1 != c2 for c1, c2 in zip(b1, b2))
      betas.append(beta_j)

      # α_j: total leading zeros added if storing each vertex value in minimal bits then padding to k_bits
      vals2 = [int(h, 16) for h in out2["groups_hex_3"]]
      alpha_j = sum(max(0, k_bits - max(1, v.bit_length())) for v in vals2)
      alphas.append(alpha_j)

      p_j = (beta_j / total_bits) * 100.0
      w.writerow([t, base_msg_len, pos, beta_j, alpha_j, p_j])

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
  # Example single hash (unchanged)
  message_bits = "01101010011010"
  out = ctqw_hash(bits=message_bits, n=15, t1=np.pi, t2=np.pi, k_bits=12)
  print("Total bits:", len(out['binary']))     # 180
  print("Hex digest:", out['hex'])
  print("Groups (12 bits each):", ' '.join(out['groups_hex_3']))

  # Save circuit figure
  saved = save_hash_circuit_png(message_bits, "./fig/hash_circuit.png", n=15, t1=np.pi, t2=np.pi)
  print("Saved circuit to:", saved)

  # ---- Collision experiments ----
  iters = 1000

  # A) Full digest-collision search (classical)
  summary_full = run_full_collision_search(
    iterations=iters,
    n=15, t1=np.pi, t2=np.pi, k_bits=12,
    msg_len_min=64, msg_len_max=128,
    out_csv="./run/hash_values.csv",
    seed=42,
  )
  print("\n[Full collision search]")
  print("Iterations:", summary_full["iterations"])
  print("Unique digests:", summary_full["unique_digests"])
  print("Collisions (different inputs → same digest):", summary_full["collisions"])
  if summary_full["collisions"]:
    print("Examples:", summary_full["collision_examples"])
  print("CSV:", summary_full["csv"])

  # B) Paper-style ω test (flip one bit)
  summary_omega = run_omega_test_flip_one_bit(
    iterations=iters,
    n=15, t1=np.pi, t2=np.pi, k_bits=12,
    base_msg_len=128,
    out_csv="./run/omega_test.csv",
    seed=123,
  )
  print("\n[Paper-style ω test]")
  print("Iterations:", summary_omega["iterations"])
  print("ω histogram:", summary_omega["omega_hist"])
  print("Counts → ω=0:", summary_omega["omega0"],
        "ω=1:", summary_omega["omega1"],
        "ω=2:", summary_omega["omega2"])
  print("“Collision rate” (ω ≥ 1) %:", round(summary_omega["collision_rate_percent"], 3))
  print("CSV:", summary_omega["csv"])

  # C) Statistical analysis (Sec. 3.2)
  stats = run_statistical_analysis(
    iterations=iters,                 # η
    n=15, t1=np.pi, t2=np.pi, k_bits=12,
    base_msg_len=128,
    out_csv="./run/stats.csv",
    seed=321,
  )
  print("\n[Statistical analysis]")
  print("η:", stats["iterations"])
  print("β̄ (mean changed bits):", round(stats["beta_mean"], 3))
  print("ᾱ (mean added zeros):", round(stats["alpha_mean"], 3))
  print("p (mean changed probability %):", round(stats["p_mean_percent"], 3))
  print("Δβ (std of changed bits):", round(stats["delta_beta"], 3))
  print("Δp (std of changed probability %):", round(stats["delta_p_percent"], 3))
  print("CSV:", stats["csv"])
