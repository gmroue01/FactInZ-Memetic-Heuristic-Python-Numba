import numpy as np
import time
import os
import sys
import argparse
import warnings
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, cpu_count
from typing import Tuple, List, Dict, Any

# --- Numba Configuration ---
try:
    from numba import njit
    print(f"[INFO] Numba detected. CPU Cores available: {cpu_count()}")
except ImportError:
    print("[WARNING] Numba not found. Performance will be severely degraded.")
    # Fallback dummy decorator
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator

warnings.filterwarnings("ignore")

# ==========================================
# PART 1: Core Numba Functions (High Performance)
# ==========================================

@njit(fastmath=True, nogil=True)
def calculate_frobenius_error(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    """Calculates squared Frobenius norm error between X and W@H."""
    m, n = X.shape
    r = W.shape[1]
    err = 0.0
    for i in range(m):
        for j in range(n):
            approx = 0.0
            for k in range(r):
                approx += float(W[i, k]) * float(H[k, j])
            diff = X[i, j] - approx
            err += diff * diff
    return err

@njit(fastmath=True, nogil=True)
def calculate_manhattan_distance(W1, H1, W2, H2) -> float:
    """Calculates L1 distance between two solutions (Phenotypic Diversity)."""
    m, r = W1.shape
    n = H1.shape[1]
    dist = 0.0
    for i in range(m):
        for k in range(r): dist += abs(W1[i, k] - W2[i, k])
    for k in range(r):
        for j in range(n): dist += abs(H1[k, j] - H2[k, j])
    return dist

@njit(fastmath=True, nogil=True)
def fast_local_search(X, W, H, current_score, LW, UW, LH, UH, steps) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Coordinate Descent Local Search. 
    Optimizes W and H iteratively to minimize error.
    """
    m, r = W.shape
    n = H.shape[1]
    
    # Calculate Residual Matrix R = X - WH
    R = np.empty((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for k in range(r): acc += float(W[i, k]) * float(H[k, j])
            R[i, j] = X[i, j] - acc

    # Precompute norms
    H_norms = np.zeros(r, dtype=np.float32)
    W_norms = np.zeros(r, dtype=np.float32)

    for k in range(r):
        s = 1e-10
        for j in range(n): s += float(H[k, j])**2
        H_norms[k] = s
        s = 1e-10
        for i in range(m): s += float(W[i, k])**2
        W_norms[k] = s

    for _ in range(steps):
        total_gain = 0.0

        # Update W
        for k in range(r):
            norm_val = H_norms[k]
            if norm_val < 1e-9: continue
            inv_norm = 1.0 / norm_val

            for i in range(m):
                dot = 0.0
                for j in range(n): dot += R[i, j] * float(H[k, j])

                delta = int(round(dot * inv_norm))
                if delta == 0: continue

                curr = W[i, k]
                targ = min(max(curr + delta, LW), UW)
                real_delta = targ - curr

                if real_delta != 0:
                    gain = -2 * real_delta * dot + (real_delta**2) * norm_val
                    if gain < -1e-3:
                        W[i, k] = targ
                        for j in range(n): R[i, j] -= real_delta * float(H[k, j])
                        total_gain += 1

        # Update H
        for k in range(r):
            norm_val = 1e-10
            for i in range(m): norm_val += float(W[i, k])**2
            W_norms[k] = norm_val

            if norm_val < 1e-9: continue
            inv_norm = 1.0 / norm_val

            for j in range(n):
                dot = 0.0
                for i in range(m): dot += R[i, j] * float(W[i, k])

                delta = int(round(dot * inv_norm))
                if delta == 0: continue

                curr = H[k, j]
                targ = min(max(curr + delta, LH), UH)
                real_delta = targ - curr

                if real_delta != 0:
                    gain = -2 * real_delta * dot + (real_delta**2) * norm_val
                    if gain < -1e-3:
                        H[k, j] = targ
                        for i in range(m): R[i, j] -= real_delta * float(W[i, k])
                        
                        delta_norm = 2*curr*real_delta + real_delta**2
                        H_norms[k] += delta_norm
                        if H_norms[k] < 1e-10: H_norms[k] = 1e-10
                        total_gain += 1

        if total_gain == 0: break
    
    final_score = calculate_frobenius_error(X, W, H)
    return W, H, final_score

# ==========================================
# PART 2: Mutation Operators
# ==========================================

@njit(fastmath=True, nogil=True)
def mutate_zero_blockout(W, H, m, n, r):
    """Destructive mutation: Zeroes out a random block."""
    W_mut = W.copy(); H_mut = H.copy()
    if np.random.random() < 0.5:
        h_size = np.random.randint(1, max(2, m // 5))
        w_size = np.random.randint(1, r + 1)
        row = np.random.randint(0, m - h_size)
        col = np.random.randint(0, r - w_size + 1)
        for i in range(row, row + h_size):
            for k in range(col, col + w_size):
                W_mut[i, k] = 0
    else:
        h_size = np.random.randint(1, r + 1)
        w_size = np.random.randint(1, max(2, n // 5))
        row = np.random.randint(0, r - h_size + 1)
        col = np.random.randint(0, n - w_size)
        for k in range(row, row + h_size):
            for j in range(col, col + w_size):
                H_mut[k, j] = 0
    return W_mut, H_mut

@njit(fastmath=True, nogil=True)
def mutate_optimal_vector_replacement(W, H, m, n, r, X, LW, UW, LH, UH):
    """
    Greedy Mutation: Replaces a random column of W or row of H 
    with its mathematically optimal value (projection).
    """
    W_mut = W.copy(); H_mut = H.copy()
    k = np.random.randint(0, r)
    
    # Calculate Norm of H[k]
    norm_Hk = 0.0
    for j in range(n): norm_Hk += float(H[k, j])**2
    norm_Hk += 1e-10

    if np.random.random() < 0.5:
        # Optimize column k of W
        for i in range(m):
            row_res = 0.0
            for j in range(n):
                approx_excl = 0.0
                for l in range(r):
                    if l != k: approx_excl += float(W[i, l]) * float(H[l, j])
                row_res += (X[i, j] - approx_excl) * float(H[k, j])
            val = row_res / norm_Hk
            W_mut[i, k] = min(max(int(round(val)), LW), UW)
    else:
        # Optimize row k of H
        norm_Wk = 0.0
        for i in range(m): norm_Wk += float(W[i, k])**2
        norm_Wk += 1e-10
        
        for j in range(n):
            col_res = 0.0
            for i in range(m):
                approx_excl = 0.0
                for l in range(r):
                    if l != k: approx_excl += float(W[i, l]) * float(H[l, j])
                col_res += (X[i, j] - approx_excl) * float(W[i, k])
            val = col_res / norm_Wk
            H_mut[k, j] = min(max(int(round(val)), LH), UH)
            
    return W_mut, H_mut

@njit(fastmath=True, nogil=True)
def mutate_residual_guided_jolt(W, H, m, n, r, X, LW, UW, LH, UH):
    """'Sniper' mutation: Targets matrix cells with highest error."""
    W_mut = W.copy(); H_mut = H.copy()
    corrections = 0
    attempts = 0
    while corrections < 25 and attempts < 100:
        attempts += 1
        i = np.random.randint(0, m); j = np.random.randint(0, n)
        approx = 0.0
        for k in range(r): approx += float(W_mut[i, k]) * float(H_mut[k, j])
        diff = X[i, j] - approx
        if abs(diff) < 0.5: continue

        corrections += 1
        base_dir = 1 if diff > 0 else -1
        k = np.random.randint(0, r)

        if np.random.random() < 0.5:
            h_sign = 1 if H_mut[k, j] >= 0 else -1
            final_dir = base_dir * h_sign
            force = 1 if abs(diff) < 5 else 2
            W_mut[i, k] = min(max(W_mut[i, k] + (final_dir * force), LW), UW)
        else:
            w_sign = 1 if W_mut[i, k] >= 0 else -1
            final_dir = base_dir * w_sign
            force = 1 if abs(diff) < 5 else 2
            H_mut[k, j] = min(max(H_mut[k, j] + (final_dir * force), LH), UH)

    # Random jitter fallback
    for _ in range(25):
        i = np.random.randint(0, m); j = np.random.randint(0, n)
        direction = 1 if np.random.random() < 0.5 else -1
        k = np.random.randint(0, r)
        if np.random.random() < 0.5:
            W_mut[i, k] = min(max(W_mut[i, k] + direction, LW), UW)
        else:
            H_mut[k, j] = min(max(H_mut[k, j] + direction, LH), UH)
    return W_mut, H_mut

@njit(fastmath=True, nogil=True)
def apply_earthquake(W, H, m, n, r, LW, UW, LH, UH, intensity=0.15):
    """Major random perturbation to escape deep local optima."""
    W_copy = W.copy(); H_copy = H.copy()
    nb_cells_W = int(m * r * intensity)
    for _ in range(nb_cells_W):
        i = np.random.randint(0, m); k = np.random.randint(0, r)
        W_copy[i, k] = np.random.randint(LW, UW + 1)
    
    nb_cells_H = int(n * r * intensity)
    for _ in range(nb_cells_H):
        k = np.random.randint(0, r); j = np.random.randint(0, n)
        H_copy[k, j] = np.random.randint(LH, UH + 1)
        
    if np.random.random() < 0.5:
        k_nuke = np.random.randint(0, r)
        for i in range(m): W_copy[i, k_nuke] = np.random.randint(LW, UW + 1)
        for j in range(n): H_copy[k_nuke, j] = np.random.randint(LH, UH + 1)
    return W_copy, H_copy

@njit(fastmath=True, nogil=True)
def apply_factor_reset(W, H, m, n, r, LW, UW, LH, UH):
    """Completely resets 2 random factors."""
    W_mut = W.copy(); H_mut = H.copy()
    nb_reset = 2
    indices = np.random.choice(r, nb_reset, replace=False)
    for k in indices:
        for i in range(m): W_mut[i, k] = np.random.randint(LW, UW + 1)
        for j in range(n): H_mut[k, j] = np.random.randint(LH, UH + 1)
    return W_mut, H_mut

# ==========================================
# PART 3: Solver Logic & Workers
# ==========================================

def load_matrix_file(filepath):
    if not os.path.exists(filepath): raise FileNotFoundError(f"File {filepath} not found")
    with open(filepath, 'r') as f:
        header = f.readline().strip().split()
        params = [int(x) for x in header]
        m, n, r, LW, UW, LH, UH = params
        X = np.loadtxt(f, dtype=np.float32)
    return X, params

def worker_initialization(seed, strategy, X, bounds, ls_steps):
    np.random.seed(seed)
    m, n = X.shape
    r = bounds['r']
    LW, UW, LH, UH = bounds['LW'], bounds['UW'], bounds['LH'], bounds['UH']
    
    W = np.zeros((m, r), dtype=np.int32)
    H = np.zeros((r, n), dtype=np.int32)

    if strategy == 'exact_svd':
        u, s, vt = np.linalg.svd(X, full_matrices=False)
        u, s, vt = u[:, :r], s[:r], vt[:r, :]
        s_sqrt = np.diag(np.sqrt(s))
        W_f = u @ s_sqrt; H_f = s_sqrt @ vt
        scale = np.random.uniform(0.9, 1.1)
        W = np.clip(np.round((W_f * scale)), LW, UW).astype(np.int32)
        H = np.clip(np.round((H_f / scale)), LH, UH).astype(np.int32)
    else:
        W = np.random.randint(LW, UW+1, (m, r)).astype(np.int32)
        H = np.random.randint(LH, UH+1, (r, n)).astype(np.int32)

    if not W.flags['C_CONTIGUOUS']: W = np.ascontiguousarray(W)
    if not H.flags['C_CONTIGUOUS']: H = np.ascontiguousarray(H)

    score = calculate_frobenius_error(X, W, H)
    W, H, score = fast_local_search(X, W, H, score, LW, UW, LH, UH, ls_steps)
    return {'W': W, 'H': H, 'score': score}

def worker_breeding_batch(seeds, parents_batch, args_communs):
    """Processes a batch of breeding tasks for parallelism."""
    results = []
    X, bounds, mut_rate, phase, aos_probs, ls_steps = args_communs
    m, n = X.shape
    r = bounds['r']
    LW, UW, LH, UH = bounds['LW'], bounds['UW'], bounds['LH'], bounds['UH']

    for i in range(len(seeds)):
        p1, p2 = parents_batch[i]
        np.random.seed(seeds[i] + os.getpid())

        # Crossover
        cW = np.empty((m, r), dtype=np.int32)
        cH = np.empty((r, n), dtype=np.int32)
        for k in range(r):
            if np.random.random() < 0.5: cW[:, k] = p1['W'][:, k]; cH[k, :] = p1['H'][k, :]
            else: cW[:, k] = p2['W'][:, k]; cH[k, :] = p2['H'][k, :]

        # Mutation (AOS Driven)
        op_idx = -1
        applied = False
        if np.random.random() < mut_rate:
            applied = True
            op_idx = np.searchsorted(np.cumsum(aos_probs), np.random.random())
            if op_idx > 2: op_idx = 2

            if op_idx == 0:
                cW, cH = mutate_optimal_vector_replacement(cW, cH, m, n, r, X, LW, UW, LH, UH)
            elif op_idx == 1:
                cW, cH = mutate_zero_blockout(cW, cH, m, n, r)
            else:
                cW, cH = mutate_residual_guided_jolt(cW, cH, m, n, r, X, LW, UW, LH, UH)

        # Local Search
        c_score = calculate_frobenius_error(X, cW, cH)
        cW, cH, c_score = fast_local_search(X, cW, cH, c_score, LW, UW, LH, UH, ls_steps)

        parent_best = min(p1['score'], p2['score'])
        gain = max(0.0, float(parent_best - c_score))
        
        results.append({'W': cW, 'H': cH, 'score': c_score, 'op_idx': op_idx, 'gain': gain, 'applied': applied})
    
    return results

def worker_cataclysm(seed, best_W, best_H, X, bounds):
    np.random.seed(seed + os.getpid())
    m, n = X.shape
    r = bounds['r']
    LW, UW, LH, UH = bounds['LW'], bounds['UW'], bounds['LH'], bounds['UH']
    
    cW, cH = apply_factor_reset(best_W, best_H, m, n, r, LW, UW, LH, UH)
    score = calculate_frobenius_error(X, cW, cH)
    cW, cH, score = fast_local_search(X, cW, cH, score, LW, UW, LH, UH, 200)
    return {'W': cW, 'H': cH, 'score': score}

def run_solver(args):
    print(f"--- Loading data from {args.input} ---")
    X, params = load_matrix_file(args.input)
    if not X.flags['C_CONTIGUOUS']: X = np.ascontiguousarray(X)
    m, n, r, LW, UW, LH, UH = params
    bounds = {'r': r, 'LW': LW, 'UW': UW, 'LH': LH, 'UH': UH}
    
    print(f"Matrix: {m}x{n}, Rank: {r}, Range: [{LW}:{UW}]")

    # JIT Warmup
    print("Compiling Numba functions...")
    dummy_X = np.zeros((10, 10), dtype=np.float32)
    dummy_W = np.zeros((10, 2), dtype=np.int32)
    dummy_H = np.zeros((2, 10), dtype=np.int32)
    fast_local_search(dummy_X, dummy_W, dummy_H, 0, LW, UW, LH, UH, 1)
    mutate_optimal_vector_replacement(dummy_W, dummy_H, 10, 10, 2, dummy_X, LW, UW, LH, UH)

    start_time = time.time()
    pop_size = args.pop_size
    mutation_names = ["OptVec", "Block", "ResJolt"]
    nb_ops = len(mutation_names)
    aos_credits = np.zeros(nb_ops); aos_counts = np.zeros(nb_ops)
    aos_probs = np.ones(nb_ops) / nb_ops
    p_min = 0.1

    print("Initializing Population...")
    seeds = np.random.randint(0, 1000000, pop_size)
    strategies = ['exact_svd'] * (pop_size - 5) + ['random'] * 5
    population = Parallel(n_jobs=-1)(
        delayed(worker_initialization)(seeds[i], strategies[i], X, bounds, ls_steps=200) for i in range(pop_size)
    )
    population.sort(key=lambda x: x['score'])
    best_global = population[0].copy()
    print(f"Initial Best Score: {int(best_global['score'])}")

    score_history = []
    gen = 0; gens_since_record = 0; gens_in_p1 = 0; consecutive_earthquakes = 0
    PHASE = 2
    
    n_jobs = cpu_count()
    chunk_size = max(1, pop_size // n_jobs)

    while (time.time() - start_time) < args.time:
        gen += 1
        score_history.append(int(best_global['score']))

        # --- Earthquake / Restart Logic ---
        if consecutive_earthquakes >= args.earthquake_limit:
            print(f"\n >>> ☠️ CATACLYSM: RESET TOTAL <<<")
            best_W_mut, best_H_mut = apply_earthquake(best_global['W'], best_global['H'], m, n, r, LW, UW, LH, UH, 0.30)
            cataclysm_seeds = np.random.randint(0, 1000000, pop_size - 1)
            new_population = Parallel(n_jobs=-1)(
                delayed(worker_cataclysm)(cataclysm_seeds[i], best_W_mut, best_H_mut, X, bounds) 
                for i in range(pop_size - 1)
            )
            population = new_population + [best_global.copy()]
            PHASE = 1; gens_since_record = 0; gens_in_p1 = 0; consecutive_earthquakes = 0
            aos_credits.fill(0); aos_counts.fill(0); aos_probs.fill(1.0/nb_ops)
            continue
        
        elif PHASE == 2 and gens_since_record >= args.stag_limit:
            consecutive_earthquakes += 1
            print(f"\n >>> 🌋 EARTHQUAKE #{consecutive_earthquakes} (Stag {gens_since_record}) -> P1")
            intensity = 0.15 + (consecutive_earthquakes * 0.05)
            eq_W, eq_H = apply_earthquake(best_global['W'], best_global['H'], m, n, r, LW, UW, LH, UH, intensity)
            eq_score = calculate_frobenius_error(X, eq_W, eq_H)
            eq_W, eq_H, eq_score = fast_local_search(X, eq_W, eq_H, eq_score, LW, UW, LH, UH, 500)
            population.sort(key=lambda x: x['score'])
            population[-1] = {'W': eq_W, 'H': eq_H, 'score': eq_score}
            PHASE = 1; gens_since_record = 0; gens_in_p1 = 0

        elif PHASE == 1:
            gens_in_p1 += 1
            if gens_in_p1 > 15:
                print(f" --- End P1 (Exploration) -> Switch P2 (Polish) ---")
                PHASE = 2; gens_in_p1 = 0; gens_since_record = 0

        # --- Breeding (Gender Selection) ---
        population.sort(key=lambda x: x['score'])
        n_dist = pop_size // 2
        
        # Diversity Parents (Females) vs Fitness Parents (Males)
        males = population[:n_dist]
        females_cand = population[n_dist:]
        females = sorted(females_cand, key=lambda c: calculate_manhattan_distance(c['W'], c['H'], best_global['W'], best_global['H']), reverse=True)
        
        pairs = []
        for _ in range(n_dist):
            pairs.append((np.random.choice(males), np.random.choice(females))) # Gender Pairs
        
        # Tournament Pairs
        for _ in range(pop_size - n_dist):
            c1, c2 = np.random.choice(population, 2)
            p1 = c1 if c1['score'] < c2['score'] else c2
            c3, c4 = np.random.choice(population, 2)
            p2 = c3 if c3['score'] < c4['score'] else c4
            pairs.append((p1, p2))
            
        np.random.shuffle(pairs)

        # --- Parallel Execution ---
        child_seeds = np.random.randint(0, 1000000, pop_size)
        tasks = []
        for i in range(0, pop_size, chunk_size):
            end = min(i + chunk_size, pop_size)
            tasks.append((
                child_seeds[i:end], pairs[i:end],
                (X, bounds, args.mut_rate, PHASE, aos_probs, args.ls_steps)
            ))

        results_batched = Parallel(n_jobs=-1, prefer="threads")(delayed(worker_breeding_batch)(*t) for t in tasks)
        children = [item for sublist in results_batched for item in sublist]

        # --- Update & AOS ---
        for child in children:
            if child['applied'] and child['gain'] > 0:
                idx = child['op_idx']
                if idx >= 0:
                    aos_credits[idx] += child['gain']
                    aos_counts[idx] += 1
        
        # Update Probs
        if gen % 5 == 0 and np.sum(aos_credits) > 0:
            avg_rewards = np.divide(aos_credits, aos_counts, out=np.zeros(nb_ops), where=aos_counts!=0)
            sum_avg = np.sum(avg_rewards)
            if sum_avg > 0:
                aos_probs = p_min + (1.0 - nb_ops * p_min) * (avg_rewards / sum_avg)
            else:
                aos_probs.fill(1.0/nb_ops)
            aos_credits *= 0.8
            aos_counts *= 0.8

        # RTS Replacement
        curr_rts = 20 if PHASE == 1 else 10
        for child in children:
            window = np.random.choice(population, min(len(population), curr_rts), replace=False)
            nearest = min(window, key=lambda p: calculate_manhattan_distance(child['W'], child['H'], p['W'], p['H']))
            if child['score'] <= nearest['score']:
                # Find index in main population to replace
                # Note: Dictionary comparison is tricky in lists, relying on reference update here might require finding index
                # We simply swap content to keep references if possible, or search linear.
                # Simplified for this script:
                for i, p in enumerate(population):
                    if p is nearest:
                        population[i] = child
                        break

        current_best = min(population, key=lambda x: x['score'])
        if current_best['score'] < best_global['score']:
            old = best_global['score']
            best_global = current_best.copy()
            consecutive_earthquakes = 0; gens_since_record = 0
            print(f" Gen {gen} [P{PHASE}] | 🌟 Record : {int(old)} -> {int(best_global['score'])}")
        else:
            gens_since_record += 1
            if gen % 10 == 0:
                best_op = mutation_names[np.argmax(aos_probs)]
                print(f" Gen {gen} [P{PHASE}] | Best: {int(best_global['score'])} | AOS: {best_op} ({int(np.max(aos_probs)*100)}%)")

        if int(best_global['score']) == 0: break

    # --- Final Polish ---
    print("\n" + "="*40)
    print(" >>> GOLDEN POLISH FINAL (5000 steps) <<<")
    print("="*40)
    W_f, H_f, s_f = fast_local_search(X, best_global['W'], best_global['H'], best_global['score'], LW, UW, LH, UH, 5000)
    if s_f < best_global['score']:
        print(f" FINAL GAIN: {int(best_global['score'])} -> {int(s_f)}")
        best_global = {'W': W_f, 'H': H_f, 'score': s_f}

    # Save Results
    with open("solution.txt", "w") as f:
        f.write(f"{int(best_global['score'])}\n")
        np.savetxt(f, best_global['W'], fmt='%d')
        np.savetxt(f, best_global['H'], fmt='%d')
    print(f"Final Score: {int(best_global['score'])} saved to solution.txt")

    if args.plot:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(score_history)
            plt.title(f"Convergence (Final: {int(best_global['score'])})")
            plt.xlabel("Generations"); plt.ylabel("Score")
            plt.savefig("convergence.png")
            print("[INFO] Plot saved: convergence.png")
        except Exception as e:
            print(f"[WARN] Plot failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHC Gender Solver")
    parser.add_argument("--input", type=str, required=True, help="Path to input file")
    parser.add_argument("--time", type=int, default=300, help="Max execution time (s)")
    parser.add_argument("--pop_size", type=int, default=40, help="Population size")
    parser.add_argument("--mut_rate", type=float, default=0.3, help="Mutation rate")
    parser.add_argument("--stag_limit", type=int, default=15, help="Stagnation before Earthquake")
    parser.add_argument("--earthquake_limit", type=int, default=3, help="Max Earthquakes before Cataclysm")
    parser.add_argument("--ls_steps", type=int, default=5, help="LS steps per child")
    parser.add_argument("--plot", action='store_true', help="Enable plotting")
    
    args = parser.parse_args()
    run_solver(args)