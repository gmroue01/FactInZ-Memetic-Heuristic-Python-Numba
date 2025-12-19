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
It should display these lines :
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
