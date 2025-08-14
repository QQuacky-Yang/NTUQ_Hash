# hash_function.py

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
    Continuous-Time Quantum Walk hash returning an UPPERCASE hex digest.
    """
    A = _path_adjacency(n)
    L = _laplacian_from_A(A)

    U0 = _unitary_from_hermitian(A, t1)
    U1 = _unitary_from_hermitian(L, t2)

    state = np.zeros(n, dtype=complex)
    state[0] = 1.0

    for b in bits:
        state = (U0 @ state) if b == '0' else (U1 @ state)

    probs = np.abs(state) ** 2
    s = probs.sum()
    probs = probs / s if s != 0 else probs

    K = 1 << k
    vals = (np.floor(probs * scale).astype(int)) % K

    binary_out = ''.join(format(v, f'0{k}b') for v in vals)
    hex_out = hex(int(binary_out, 2))[2:].upper()
    return hex_out


# ---------- utility: string to bitstring -------------------------------------

def str_to_bitstring(data: str) -> str:
    """Convert any string input (e.g. 'Hello') to a bitstring."""
    return ''.join(format(b, '08b') for b in data.encode('utf-8'))


# ---------- final hash interface ---------------------------------------------

def custom_hash(data: str) -> str:
    """
    Unified hash interface: takes string, returns uppercase hex digest.
    Uses your CTQW hash.
    """
    bitstr = str_to_bitstring(data)
    return ctqw_hash(bitstr)