**NSGA-II Algorithm for the Algorithm Selection Problem using Decision Trees**

# Description

This repository contains multiple versions of the NSGA-II algorithm, developed to address the algorithm selection problem using decision trees.

Each version is organized in its own subdirectory and includes a dedicated `README.md` file describing the updates and improvements introduced in that version.

Results of the experiments are stored in the `experiment_logs_valids` folder, within each version's directory.

**How to run a single experiment:**  
`python3.11 main.py {nog} {ps} {cr} {mr} {mtd} {r} {ilnp}`

**Script to run multiple experiments:**  
`python3.11 run_experiments.py`

## 📊 Dataset Used in the Experiments

- **File:** `/bases/input_data_paper_2019.txt`

### 📝 Dataset Description

- **Instances:** 1004  
- **Candidate Algorithms:** 532  
- **Features:** 37
