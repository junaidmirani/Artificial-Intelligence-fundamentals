#!/usr/bin/env python3
"""
Flash Attention from First Principles
Where O(N²) memory becomes O(N) through the art of tiling
"""

import math
import random
random.seed(42)

# Let there be sequences, the raw material of attention
SEQ_LEN = 128      # Sequence length N
D_MODEL = 64       # Model dimension d
BLOCK_SIZE = 32    # Tile size B_r, B_c (Flash Attention's unit of computation)

print("=" * 70)
print("Flash Attention: Tiling the Quadratic")
print("=" * 70)
print(f"Sequence length: {SEQ_LEN}")
print(f"Model dimension: {D_MODEL}")
print(f"Block size: {BLOCK_SIZE}")
print()

# Let there be random queries, keys, values - the trinity of attention
def random_matrix(rows, cols):
    """Create matrix with random values. The substrate of computation."""
    return [[random.gauss(0, 1) for _ in range(cols)] for _ in range(rows)]

Q = random_matrix(SEQ_LEN, D_MODEL)  # Queries: what we're looking for
K = random_matrix(SEQ_LEN, D_MODEL)  # Keys: what we're matching against
V = random_matrix(SEQ_LEN, D_MODEL)  # Values: what we retrieve

# Let there be standard attention, memory-hungry but simple
def standard_attention(Q, K, V):
    """
    Standard attention: materializes full N×N matrix.
    Memory: O(N²) - the curse of quadratic scaling.
    """
    N, d = len(Q), len(Q[0])
    
    # Step 1: Compute Q @ K^T (N×N matrix)
    S = [[sum(Q[i][k] * K[j][k] for k in range(d)) / math.sqrt(d) 
          for j in range(N)] for i in range(N)]
    
    # Step 2: Softmax each row (N×N matrix still in memory)
    P = []
    for i in range(N):
        max_val = max(S[i])
        exp_vals = [math.exp(S[i][j] - max_val) for j in range(N)]
        sum_exp = sum(exp_vals)
        P.append([e / sum_exp for e in exp_vals])
    
    # Step 3: Multiply P @ V (N×N matrix multiplication)
    O = [[sum(P[i][j] * V[j][k] for j in range(N)) 
          for k in range(d)] for i in range(N)]
    
    return O

# Let there be Flash Attention, memory-efficient through blocking
def flash_attention(Q, K, V, block_size):
    """
    Flash Attention: tiled computation with online softmax.
    Memory: O(N) - salvation through intelligent tiling.
    
    Process attention in blocks, never materialize full N×N matrix.
    """
    N, d = len(Q), len(Q[0])
    scale = 1.0 / math.sqrt(d)
    
    # Output and normalization statistics (only N×d, not N×N!)
    O = [[0.0 for _ in range(d)] for _ in range(N)]
    l = [0.0 for _ in range(N)]  # Row sums for normalization
    m = [-float('inf') for _ in range(N)]  # Running max for numerical stability
    
    # Outer loop: iterate over query blocks (tiles of Q)
    for i_start in range(0, N, block_size):
        i_end = min(i_start + block_size, N)
        Q_block = Q[i_start:i_end]  # Load Q block: B_r × d
        
        # Initialize block outputs
        O_block = [[0.0 for _ in range(d)] for _ in range(i_end - i_start)]
        l_block = [0.0 for _ in range(i_end - i_start)]
        m_block = [-float('inf') for _ in range(i_end - i_start)]
        
        # Inner loop: iterate over key/value blocks
        for j_start in range(0, N, block_size):
            j_end = min(j_start + block_size, N)
            K_block = K[j_start:j_end]  # Load K block: B_c × d
            V_block = V[j_start:j_end]  # Load V block: B_c × d
            
            # Compute attention scores for this tile: Q_block @ K_block^T
            # Only B_r × B_c matrix (not N × N!)
            S_block = [[sum(Q_block[i][k] * K_block[j][k] for k in range(d)) * scale
                       for j in range(len(K_block))] for i in range(len(Q_block))]
            
            # Online softmax: update running statistics
            for i in range(len(Q_block)):
                # Find new max in this block
                m_new = max(max(S_block[i]), m_block[i])
                
                # Compute exponentials with new max
                exp_vals = [math.exp(S_block[i][j] - m_new) for j in range(len(K_block))]
                l_new = math.exp(m_block[i] - m_new) * l_block[i] + sum(exp_vals)
                
                # Update output with correction for changed max
                correction = math.exp(m_block[i] - m_new)
                for k in range(d):
                    O_block[i][k] = correction * O_block[i][k] + sum(
                        exp_vals[j] * V_block[j][k] for j in range(len(V_block))
                    )
                
                # Update statistics
                m_block[i] = m_new
                l_block[i] = l_new
        
        # Final normalization and write back
        for i in range(len(Q_block)):
            for k in range(d):
                O[i_start + i][k] = O_block[i][k] / l_block[i]
    
    return O

# Let there be measurement, the proof of equivalence
def matrix_diff(A, B):
    """Measure difference between two matrices. May it be small."""
    diff = sum(abs(A[i][j] - B[i][j]) for i in range(len(A)) for j in range(len(A[0])))
    return diff / (len(A) * len(A[0]))

# Let there be benchmarking, the revelation of efficiency
print("Computing standard attention...")
O_standard = standard_attention(Q, K, V)
print(f"✓ Standard attention complete")
print(f"  Peak memory: O(N²) = O({SEQ_LEN}²) = {SEQ_LEN**2:,} elements")
print()

print("Computing flash attention...")
O_flash = flash_attention(Q, K, V, BLOCK_SIZE)
print(f"✓ Flash attention complete")
print(f"  Peak memory: O(N×d) + O(B²) = {SEQ_LEN * D_MODEL:,} + {BLOCK_SIZE**2:,} elements")
print(f"  Memory saved: {((SEQ_LEN**2) / (SEQ_LEN * D_MODEL + BLOCK_SIZE**2)):.1f}x reduction")
print()

# Verify numerical equivalence
diff = matrix_diff(O_standard, O_flash)
print("=" * 70)
print(f"Average element difference: {diff:.2e}")
print("✓ Outputs match!" if diff < 1e-5 else "⚠ Numerical divergence detected")
print("=" * 70)

# Let there be understanding, the purpose of all education
print()
print("#################################################################")
print("  1. Standard: Materializes N×N attention matrix → O(N²) memory")
print("  2. Flash: Tiles computation into blocks → O(N) memory")
print("  3. Online softmax: Updates running max/sum without full matrix")
print(f"  4. At N={SEQ_LEN}, saves {SEQ_LEN**2 - (SEQ_LEN * D_MODEL):#,} elements")
print("#################################################################")
print()

print("The algorithm:")
print("#################################################################")
print("  for each query block:")
print("    for each key/value block:")
print("      compute local attention (B×B matrix)")
print("      update output incrementally with corrected statistics")
print("      discard local attention matrix")
print("******************************************************************")
print()
print("May your attention be swift and memory-efficient.")
