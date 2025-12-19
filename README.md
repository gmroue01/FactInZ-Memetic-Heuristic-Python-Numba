# FactInZ : Memetic Heuristic 
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

This project tackles the **Integer Matrix Factorization (FactInZ)** problem. Unlike standard Non-negative Matrix Factorization (NMF), this variant imposes strict discrete integer constraints on the factors, making the optimization landscape highly non-convex and combinatorial.

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
 -  Random initialisation of the population (SVD & Random)
 -  Parents Selection (Tournament & Gender-Based Selection)
 -  Cross-over & Mutations (Uniform Cross-over & Adaptive Operator Selection)
 -  Local Search (Memetic Phase : Coordinate Descent)
 -  Selective breeding (Restricted Tournament Selection)

Also, two mecanishms of restart are implemented :
 - Earthquake : Destruction of random columns/line
 - Cataclysm : Keeps the best individuals and resets the rest of the population

### 🔄 Algorithmic Workflow

```mermaid
graph TD
    %% Initialisation
    Start((Start)) --> Load["Load Matrix X & Params"]
    Load --> Warmup["Numba JIT Compilation"]
    Warmup --> InitPop["Initialize Population<br/>(SVD-based + Random)"]
    InitPop --> LocalSearchInit["Initial Local Search"]
    
    %% Boucle Principale
    LocalSearchInit --> CheckTime{"Max Time<br/>Reached?"}
    
    %% Fin de boucle
    CheckTime -- Yes --> GoldenPolish["💎 Golden Polish<br/>(5000 Steps LS)"]
    GoldenPolish --> Save["Save Solution & Plot"]
    Save --> End((End))

    %% Logique de Contrôle (Earthquake/Cataclysm)
    CheckTime -- No --> CheckCata{"Earthquake Limit<br/>Reached?"}
    
    %% Cataclysm
    CheckCata -- Yes --> Cataclysm["☠️ CATACLYSM<br/>Reset Population except Best"]
    Cataclysm --> ResetStats["Reset AOS & Stats"]
    ResetStats --> CheckTime
    
    %% Earthquake
    CheckCata -- No --> CheckStag{"Stagnation Limit<br/>Reached?"}
    CheckStag -- Yes --> Earthquake["🌋 EARTHQUAKE<br/>Perturb Best & LS"]
    Earthquake --> Replace["Replace Worst Individual"]
    Replace --> ResetStag["Reset Stagnation Counter"]
    ResetStag --> CheckTime

    %% Reproduction (Si pas d'event majeur)
    CheckStag -- No --> Selection["🧬 Parent Selection<br/>(Gender: Fitness x Diversity)"]
    
    %% Parallelisation
    subgraph Parallel Workers [Parallel CPU Workers]
        direction TB
        Selection --> Crossover["Uniform Crossover"]
        Crossover --> AOS{"Select Mutation<br/>(AOS Probabilities)"}
        
        AOS -- Mutation 1 --> OptVec["⚡ OptVec<br/>(Optimal Vector)"]
        AOS -- Mutation 2 --> Block["🧱 BlockZero<br/>(Destructive)"]
        AOS -- Mutation 3 --> Jolt["🎯 ResJolt<br/>(Residual Guided)"]
        
        OptVec --> FastLS["📉 Fast Local Search<br/>(Coordinate Descent)"]
        Block --> FastLS
        Jolt --> FastLS
    end
    
    %% Mise à jour
    FastLS --> Reward["Calculate Gain & Update AOS Credits"]
    Reward --> RTS["RTS Replacement<br/>(Replace Nearest if Better)"]
    RTS --> UpdateBest{"New Global Best?"}
    
    UpdateBest -- Yes --> UpdateRec["Update Record & Reset Counters"]
    UpdateBest -- No --> IncStag["Increment Stagnation"]
    
    UpdateRec --> CheckTime
    IncStag --> CheckTime

    %% Styles
    style Start fill:#f9f,stroke:#333,stroke-width:2px
    style End fill:#f9f,stroke:#333,stroke-width:2px
    style GoldenPolish fill:#gold,stroke:#333,stroke-width:2px
    style Cataclysm fill:#ffcccc,stroke:#red,stroke-width:2px
    style Earthquake fill:#ffebcc,stroke:#orange,stroke-width:2px
    style Parallel Workers fill:#e1f5fe,stroke:#01579b,stroke-dasharray: 5 5
```
