# FactInZ : Memetic Heuristic 


> **🎓 Context:** This project was developed as part of the **Graph Theory & Combinatorial Optimization** course. The algorithm was designed to compete in a class-wide tournament, benchmarking various metaheuristic approaches against each other to solve the NP-hard **FactInZ** problem under strict time constraints.


## 🛠️ Installation

Follow these steps to set up the project environment on your local machine.

### 1. Clone the repository
First, clone the source code into a new directory on your machine.
Open your terminal (Command Prompt, PowerShell, or Terminal) and run:

```bash
git clone [https://github.com/gmroue01/FactInZ-Memetic-Heuristic-Python-Numba.git](https://github.com/gmroue01/FactInZ-Memetic-Heuristic-Python-Numba.git)
cd FactInZ-Memetic-Heuristic-Python-Numba
```

### 2. Create a virtual environment
It is highly recommended to use a **virtual environment**. This creates an isolated space for the project's dependecies, ensuring they don't conflict with other Python projects on your system.

- **For Windows**
  ```
  # Create the environment named 'venv'
  python -m venv venv

  # Activate the environment
  .\venv\Scripts\activate
  ```
  *Note : if the command fails (error : "PSSecurityException : Unauthorized Acces") and you use PowerShell Terminal, you might enable the script execution. Temporary , you fix it with this command ```Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process``` in your current PowerShell Terminal.*
  
- **For MacOs/Linus**
  ```
  # Create the environment named 'venv'
  python3 -m venv venv

  # Activate the environment
  source venv/bin/activate
  ```
  *Note : if the command fails, you might need to install the venv package first (eg. ```sudo apt install python3-venv``` on Ubuntu/Debian)*

### Install Dependencies 
Once the virtual environment is active, install the required libraries using the provided requirements.txt file :
```
pip install -r requirements.txt
```
The libraries installed are :
  - Numpy (>=1.20.0) : Features mathematical and optimized tools for scientific computing. (see https://numpy.org/)
  - numba (>=0.55.0) : Translates Python functions into optimized machine code at runtime using LLVM.(see https://numba.pydata.org/)
  - joblib(>=1.1.0) : Provides a set of tools that enable parallel computing. (see https://joblib.readthedocs.io/en/stable/)
  - matplotlib(>=3.5.0) : Provides tools to visualize data using graphs.(see https://matplotlib.org/)
### Verification
To verify that everything is installed correclty, you cand display the help menu of the solver

```
python Memetic.py --help
```
It should display these lines in your terminal:
```
  usage: Memetic.py [-h] --input INPUT [--time TIME] [--pop_size POP_SIZE] [--mut_rate MUT_RATE]
                    [--stag_limit STAG_LIMIT] [--earthquake_limit EARTHQUAKE_LIMIT]
                    [--ls_steps LS_STEPS] [--plot]
  
  CHC Gender Solver
  
  options:
    -h, --help            show this help message and exit
    --input INPUT         Path to input file
    --time TIME           Max execution time (s)
    --pop_size POP_SIZE   Population size
    --mut_rate MUT_RATE   Mutation rate
    --stag_limit STAG_LIMIT
                          Stagnation before Earthquake
    --earthquake_limit EARTHQUAKE_LIMIT
                          Max Earthquakes before Cataclysm
    --ls_steps LS_STEPS   LS steps per child
    --plot                Enable plotting
```
## 🧩 Problem Description: FactInZ

This project tackles the **Integer Matrix Factorization (FactInZ)** problem (see https://github.com/pashtari/qmf). Unlike standard Non-negative Matrix Factorization (NMF), this variant imposes strict discrete integer constraints on the factors, making the optimization landscape highly non-convex and combinatorial.

### Formal Definition
Given a target matrix $X \in \mathbb{Z}^{m \times n}$, a target rank $r$, and integer bounds defined by $[L_W, U_W]$ and $[L_H, U_H]$, the goal is to determine two factor matrices:
* $W \in \mathbb{Z}^{m \times r}$
* $H \in \mathbb{Z}^{r \times n}$

Such that they minimize the **Squared Frobenius Norm** of the reconstruction error:

$$\min_{W, H} \|X - WH\|_F^2 = \sum_{i=1}^{m} \sum_{j=1}^{n} (X_{ij} - (WH)_{ij})^2$$

**Subject to:**


$$\forall (i,k), \quad L_W \leq W_{ik} \leq U_W$$

$$\forall (k,j), \quad L_H \leq H_{kj} \leq U_H$$

### Complexity
Solving the **FactInZ** problem is proven to be **NP-hard**. Classical gradient-based methods are ineffective due to the discrete nature of the search space. Therefore, this solver employs a high-performance **Metaheuristic** approach (CHC Evolutionary Algorithm) to find near-optimal solutions within a reasonable time.


## 🧠 Algorithm Overview

This solver implements a **Hybrid Memetic Algorithm** designed for high-dimensional discrete optimization. It combines global evolutionary exploration with aggressive local search (exploitation).

The solver follows all these 5 steps: 
 -  Initialisation of the population (SVD & Random)
 -  Parents Selection (Tournament & Gender-Based Selection)
 -  Cross-over & Mutations (Uniform Cross-over & Adaptive Operator Selection)
 -  Local Search (Memetic Phase : Coordinate Descent)
 -  Selective breeding (Restricted Tournament Selection)

Also, two mecanishms of restart are implemented :
 - Earthquake : Destruction of random columns/line
 - Cataclysm : Keeps the best individuals and resets the rest of the population


## 🏆 Results & Performance

Below are the scores achieved by the solver on the official instances provided during the class competition. The objective is to minimize the Squared Frobenius Error, so a **lower score indicates a better solution**.

| Instance Name | Dimensions ($m \times n$) | Rank ($r$) | Final Score ($\|E\|_F^2$) | Time (min) |
| :--- | :---: | :---: | :---: | :---: |
| **Challenge 1** | $101 \times 17$ | 2 | **76** | 15
| **Challenge 2** | $101 \times 17$ | 5 | **130** | 15
| **Challenge 3** | $101 \times 17$ | 5 | **130** | 15
| **Challenge 4** | $120 \times 120$ | 10 | **17 603** | 15
| **Challenge 5** | $1000 \times 1000$ | 10 | **159 070** | 40


## ⚙️ Parameters & Configuration

The solver is highly configurable via command-line arguments. You can tune the balance between speed (performance) and solution quality (accuracy).

### CLI Arguments

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--input` | `str` | **Required** | Path to the input data file (formatted with header). |
| `--time` | `int` | `300` | Maximum execution time in seconds. The solver stops cleanly after this limit. |
| `--pop_size` | `int` | `40` | Number of individuals in the population. Larger populations explore better but are slower. |
| `--mut_rate` | `float` | `0.3` | Probability (0.0 to 1.0) that a child undergoes mutation after crossover. |
| `--ls_steps` | `int` | `5` | Number of Local Search (Coordinate Descent) steps applied to every new child. |
| `--stag_limit` | `int` | `15` | Number of generations without improvement before triggering an **Earthquake**. |
| `--earthquake_limit`| `int` | `3` | Number of Earthquakes allowed before triggering a **Cataclysm** (Total Restart). |
| `--plot` | `flag` | `False` | If present, saves a convergence graph to `convergence.png` at the end. |



## 🔄 Algorithmic Workflow
### 🌱 Population Initialization

The initialization phase is critical for the solver's performance. Instead of starting from a purely random state (which converges slowly), we employ a **hybrid initialization strategy** that provides a "hot start" while ensuring sufficient genetic diversity.

#### 1. Hybrid Composition
The population (size $N$) is composed of two types of individuals:

* **SVD-Based Individuals ($N - 5$):** The vast majority of the population is initialized using mathematical approximation. This places the solver immediately in a "good" region of the search space.
* **Random Individuals (5):** A small fixed number (5) of individuals are generated purely randomly. This ensures that the genetic pool retains "wild" genetic material to avoid getting trapped in the SVD's bias.

#### 2. Technique: Randomized SVD Projection
Standard Singular Value Decomposition (SVD) is deterministic (it always gives the same result). To generate distinct individuals from a single SVD, we apply a **random scalar perturbation**:

1.  **Decomposition:** We compute $X \approx U \Sigma V^T$ using standard floating-point SVD.
2.  **Balancing:** We distribute the singular values $\Sigma$ evenly: $W_{float} = U\sqrt{\Sigma}$ and $H_{float} = \sqrt{\Sigma}V^T$.
3.  **Random Scaling (The "Shake"):** To create diversity, we introduce a random scalar $\alpha \sim U[0.9, 1.1]$:
    $W_{init} = W_{float} \times \alpha$
    $H_{init} = H_{float} \times \frac{1}{\alpha}$
    
    *This operation modifies the internal values of* $W$ *and* $H$ *while preserving the product* $WH \approx X$.
5.  **Integer Projection:** The floating-point values are rounded to the nearest integer and clipped to the bounds $[L, U]$.

#### 3. Immediate Refinement
Raw SVD approximations are often infeasible or suboptimal on the integer grid. Therefore, **every** new individual undergoes an immediate **Local Search (200 steps)** (`fast_local_search`) right after generation. This "polishes" the rough mathematical approximation into a valid, high-quality integer solution before the evolutionary loop begins.

## 🧬 Parent Selection Strategy

To prevent premature convergence (a common issue in discrete matrix factorization), the solver employs a dual selection strategy that enforces a strict balance between **Exploitation** (Fitness) and **Exploration** (Diversity). The population is split into two pools to form breeding pairs.

### A. Gender-Based Selection (50% of pairs)
This unique mechanism treats Fitness and Diversity as two separate "genders" that must mate:

* **"Males" (Fitness Carriers):** The top 50% of the population, sorted strictly by score (lowest error). They carry the genetic material that works well.
* **"Females" (Diversity Carriers):** The bottom 50% of the population, but re-sorted based on their **Manhattan Distance** from the current Global Best solution. The most distant individuals are ranked highest.
* **The Pairing Logic:** A breeding pair is formed by selecting one random "Male" and one random "Female". This forces the algorithm to combine high-performance genes with genetically distinct features, maintaining a healthy gene pool.

### B. Tournament Selection (50% of pairs)
To ensure standard evolutionary pressure, the remaining pairs are formed using **Binary Tournament Selection**. Two individuals are picked at random from the entire population, and the one with the better score becomes a parent. This is a classic method to favor "good enough" solutions without being too elitist.

---

## ⚙️ Breeding Pipeline & Parallelism

Once parents are selected, the creation of new offspring is a multi-step process fully optimized for multi-core CPUs.

### A. Factor-Wise Uniform Crossover
Offspring are created using a **Uniform Crossover** that operates on the rank dimension $r$ (the factors).
For every factor index $k \in [0, r]$, the child inherits the **entire** $k$-th component (Column $k$ of $W$ and Row $k$ of $H$) from either Parent 1 or Parent 2 with a 50/50 probability.

$$
\begin{cases}
W_{child}[:, k] = W_{P1}[:, k] \\
H_{child}[k, :] = H_{P1}[k, :]
\end{cases}
\quad \text{OR} \quad
\begin{cases}
W_{child}[:, k] = W_{P2}[:, k] \\
H_{child}[k, :] = H_{P2}[k, :]
\end{cases}
$$

### B. Parallel Execution (Joblib)
The breeding phase is the most computationally intensive part of the algorithm. To maximize efficiency, it is parallelized using `joblib`. Each CPU core acts as an independent worker processing a batch of parents.

**The Worker Workflow:**
1.  **Crossover:** Creation of the raw child from P1 and P2.
2.  **Adaptive Mutation:** Application of a specific mutation operator (`OptVec`, `BlockZero`, or `ResJolt`) determined by the AOS probabilities.
3.  **Immediate Improvement:** Execution of `fast_local_search` to "polish" the child. This is a Memetic Algorithm feature: we only add **local optima** to the population, not random candidates.


### C. Adaptive Mutation Operators

Unlike standard genetic algorithms that rely on random bit-flipping, this solver utilizes three specialized, domain-specific mutation operators. These are controlled by the **Adaptive Operator Selection (AOS)** mechanism, which favors the operators that have historically yielded the best fitness gains during the run.

#### 1. OptVec: Optimal Vector Replacement (Greedy)
* **Code:** `mutate_optimal_vector_replacement`
* **Concept:** This is a "Greedy" operator. It selects a random factor (a column $k$ of $W$ or a row $k$ of $H$) and replaces it entirely with its **mathematically optimal projection**.
* **The Math:** It solves the 1-variable least squares problem for the selected vector, assuming the rest of the matrix is fixed.
  $$W_{opt}[:,k] = \frac{(X - W_{\neg k}H_{\neg k}) H[k,:]^T}{\|H[k,:]\|^2}$$
* **Effect:** drastically reduces error by mathematically forcing a vector to its ideal position in a single step.

#### 2. BlockZero: Destructive Blockout (Exploration)
* **Code:** `mutate_zero_blockout`
* **Concept:** A "Destructive" operator designed to escape local optima.
* **Mechanism:** It selects a random rectangular block within $W$ or $H$ and forces all values inside it to **zero**.
* **Effect:** This effectively "erases" a part of the solution. When the Local Search runs immediately after, it is forced to "rebuild" this missing part from scratch, often finding a different (and better) configuration than what was there before.

#### 3. ResJolt: Residual-Guided Jolt (Targeted)
* **Code:** `mutate_residual_guided_jolt`
* **Concept:** A "Sniper" operator that targets high-error regions.
* **Mechanism:**
    1.  It scans the Residual Matrix ($R = X - WH$) to find coordinates $(i, j)$ where the error is high.
    2.  It identifies which factors in $W$ and $H$ contribute to this specific cell.
    3.  It "jolts" (nudges) these specific factors by $+1$ or $-1$ in the direction that reduces the error.
* **Effect:** Unlike random noise, this operator only touches the parts of the matrix that are actually causing problems.

---

## Local Search : Memetic feature

The heart of the solver's performance is the `fast_local_search` function. Unlike standard gradient descent which is slow for discrete problems, this implementation uses a highly optimized **Coordinate Descent** algorithm tailored for integer factorization.

#### How it Works
Instead of updating the entire matrix at once, the algorithm iterates through every single cell of $W$ and $H$ sequentially and solves the 1D optimization problem: *"What is the best integer value for this specific cell, assuming all others are fixed?"*

#### 🚀 Key Optimizations

##### 1. The Residual Matrix ($R$) Trick
Recomputing the error $\|X - WH\|^2$ after every single cell change would be computationally prohibitive ($O(mnr)$).
* **Solution:** We maintain a **Residual Matrix** $R = X - WH$.
* **Incremental Update:** When a cell $W_{ik}$ changes by a value $\delta$, we update $R$ incrementally:
    $$R_{ij} \leftarrow R_{ij} - (\delta \times H_{kj})$$
* **Benefit:** This reduces the complexity of an update to linear time relative to the dimensions, avoiding full matrix multiplication.

##### 2. Greedy Integer Projection
For a given cell (e.g., $W_{ik}$), the optimal change $\delta$ is calculated analytically:
$\delta = \text{round}\left( \frac{\text{Projection of } R \text{ on } H_k}{\|H_k\|^2} \right)$

This value is then clipped to the user-defined bounds $[L_W, U_W]$. A change is applied only if the theoretical **Gain** (reduction in error) is positive.

##### 3. Numba Acceleration (`@njit`)
The function is decorated with `@njit(fastmath=True, nogil=True)`.
* **Machine Code:** It is compiled to optimized machine code via LLVM, running as fast as C++.
* **No GIL:** It releases the Python Global Interpreter Lock, allowing multiple CPU cores to run this search in parallel on different individuals without blocking each other.*


## 🛡️ Survival Strategy: Restricted Tournament Selection (RTS)

To update the population, the solver does not simply replace the worst individuals (which would lead to rapid loss of diversity). Instead, it employs **Restricted Tournament Selection (RTS)**, a crowding technique designed to maintain multiple distinct niches in the population.

### The Mechanism
For every new child generated:

1.  **Window Selection:** We select a random subset (window) of the current population.
    **Window Size (w):** Set to **20** during the Exploration Phase (P1) to be very strict about diversity, and reduced to **10** during the Polish Phase (P2).
2.  **Nearest Neighbor Search:** Within this window, we identify the individual that is **most similar** to the child (phenotypically nearest) using the Manhattan Distance metric:
    $$d(Child, P) = \sum |W_{child} - W_P| + \sum |H_{child} - H_P|$$
3.  **Competition:** The child competes *only* against this nearest neighbor.
    * If `Score(Child) <= Score(Nearest)`, the child **replaces** the neighbor.
    * Otherwise, the child is discarded.

### Why RTS?


By forcing children to compete against their "family" (solutions that look like them), RTS prevents a single super-solution from taking over the entire population. It allows different sub-optimal but distinct solutions to survive, preserving genetic material that might be crucial for escaping future local optima.

## 🛑 Escaping Local Optima: Earthquake & Cataclysm

The non-convex nature of integer matrix factorization makes it prone to deep local optima. The solver employs a two-tier escalation strategy to detect stagnation and force the search into new areas.

### 1. 🌋 Earthquake (Local Perturbation)
**Trigger:** Triggered when the global best score hasn't improved for `stag_limit` generations (default: 15).

**Mechanism:**
Instead of restarting, we take the current best solution and "shake" it aggressively:
1.  **Cellular Noise:** We randomize ~15% of the cells in $W$ and $H$ (intensity increases by +5% for each consecutive earthquake).
2.  **Factor Nuke:** With a 50% probability, one entire factor (column $k$ of $W$ / row $k$ of $H$) is completely randomized.
3.  **Stabilization:** The shaken solution undergoes a deep Local Search (500 steps) to settle into a new local basin.
4.  **Injection:** This new solution replaces the **worst** individual in the population, and the search phase resets to **Exploration (P1)**.



### 2. ☠️ Cataclysm (Global Restart)
**Trigger:** Triggered if `earthquake_limit` consecutive earthquakes (default: 3) fail to improve the best score.

**Mechanism:**
The solver assumes the entire population is trapped in a sub-optimal valley. It initiates a "Mass Extinction":
1.  **Preservation:** The single Best Global solution is saved.
2.  **Population Wipe:** All other individuals are deleted.
3.  **Repopulation:** The population is refilled with mutated clones of the Best Global solution. Each clone undergoes a **Factor Reset** (2 random factors are completely re-randomized) followed by Local Search.
4.  **Memory Wipe:** The Adaptive Operator Selection (AOS) history is cleared to allow unbiased learning for the new phase.

**Outcome:** This effectively respawns the search around the best known solution but with massive structural variations, forcing the algorithm to explore radically different directions.


