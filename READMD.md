# CHC-IMF: Evolutionary Integer Matrix Factorization

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Numba](https://img.shields.io/badge/Numba-Accelerated-green)

A high-performance heuristic solver for the **Integer Matrix Factorization (IMF)** problem. It approximates a target matrix $X$ as the product of two integer factor matrices $W$ and $H$:

$$X \approx W \times H$$

This implementation uses a hybrid **CHC Evolutionary Algorithm** combined with **Adaptive Operator Selection (AOS)** and aggressive **Coordinate Descent Local Search**, accelerated using `Numba` (JIT) and `Joblib` (Parallelism).



## Features

* **Hybrid Metaheuristic:** Combines Memetic Algorithms with CHC selection (maintaining diversity).
* **Performance:** Critical path functions are compiled to machine code using **Numba** (`@njit`).
* **Adaptive Mutation:** Uses probability matching to favor the most efficient mutation operators (Smart Noise, Column Swap, Residual Jolt) dynamically during the run.
* **Parallelism:** Multi-core evaluation of the population using `joblib`.

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/chc-imf-solver.git](https://github.com/your-username/chc-imf-solver.git)
   cd chc-imf-solver